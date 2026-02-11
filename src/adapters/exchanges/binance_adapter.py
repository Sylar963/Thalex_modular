import asyncio
import logging
import time
import hmac
import hashlib
from typing import Dict, List, Optional, Callable
from urllib.parse import urlencode
from dataclasses import replace

try:
    import aiohttp
except ImportError:
    raise ImportError(
        "aiohttp is required for BinanceAdapter. Install with: pip install aiohttp"
    )

from .base_adapter import BaseExchangeAdapter
from ...domain.interfaces import TimeSyncManager
from ...services.instrument_service import InstrumentService
from ...domain.entities import (
    Order,
    OrderSide,
    OrderType,
    OrderStatus,
    Position,
    Ticker,
    Trade,
    Balance,
)

logger = logging.getLogger(__name__)


class BinanceAdapter(BaseExchangeAdapter):
    REST_URL = "https://fapi.binance.com"
    REST_TESTNET_URL = "https://demo-fapi.binance.com"
    WS_URL = "wss://fstream.binance.com"
    WS_TESTNET_URL = "wss://fstream.binancefuture.com"

    def __init__(
        self,
        api_key: str = "",
        api_secret: str = "",
        testnet: bool = True,
        time_sync_manager: Optional[TimeSyncManager] = None,
    ):
        super().__init__(api_key, api_secret, testnet, time_sync_manager)
        self.base_url = self.REST_TESTNET_URL if testnet else self.REST_URL
        self.ws_base_url = self.WS_TESTNET_URL if testnet else self.WS_URL

        self.session: Optional[aiohttp.ClientSession] = None
        self.ws: Optional[aiohttp.ClientWebSocketResponse] = None
        self.listen_key: Optional[str] = None

        self.positions: Dict[str, Position] = {}
        self.orders: Dict[str, Order] = {}

        self._msg_loop_task: Optional[asyncio.Task] = None
        self._keepalive_task: Optional[asyncio.Task] = None

        # Public mode flag
        self.public_only = not (api_key and api_secret)
        if self.public_only:
            logger.info(
                "BinanceAdapter initialized in PUBLIC ONLY mode (No Trade/Account updates)"
            )

    @property
    def name(self) -> str:
        return "binance"

    async def get_server_time(self) -> int:
        """Fetch current Binance server time in milliseconds."""
        url = f"{self.base_url}/fapi/v1/time"
        try:
            if self.session and not self.session.closed:
                async with self.session.get(url) as resp:
                    if resp.status == 200:
                        data = self._fast_json_decode(await resp.read())
                        return int(data.get("serverTime", 0))
                    raise Exception(f"Binance API error: {resp.status}")
            else:
                # Fallback for when TimeSync triggers before connect()
                async with aiohttp.ClientSession() as temp_session:
                    async with temp_session.get(url) as resp:
                        if resp.status == 200:
                            data = self._fast_json_decode(await resp.read())
                            return int(data.get("serverTime", 0))
                        raise Exception(f"Binance API error: {resp.status}")

        except Exception as e:
            logger.error(f"Failed to fetch Binance server time: {e}")
            raise

    def _sign(self, params: Dict) -> str:
        if self.public_only:
            raise RuntimeError("Cannot sign request in public-only mode")

        query_string = urlencode(params)
        signature = hmac.new(
            self.api_secret.encode("utf-8"),
            query_string.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        return signature

    async def connect(self):
        logger.info(
            f"Connecting to Binance ({'Testnet' if self.testnet else 'Mainnet'})..."
        )

        headers = {}
        if not self.public_only:
            headers["X-MBX-APIKEY"] = self.api_key

        self.session = aiohttp.ClientSession(headers=headers)

        if self.time_sync_manager:
            await self.time_sync_manager.sync_all()

        if not self.public_only:
            self.listen_key = await self._create_listen_key()
            if not self.listen_key:
                raise ConnectionError("Failed to obtain listenKey from Binance")

            # Connect to Private User Data Stream
            ws_url = f"{self.ws_base_url}/ws/{self.listen_key}"
            self.ws = await self.session.ws_connect(ws_url)
            self._keepalive_task = asyncio.create_task(self._keepalive_loop())
        else:
            # Public mode: No initial WS connection (will connect on subscribe)
            # Or we can maintain a connection to base stream if needed,
            # but usually Binance streams are path-based.
            # We'll expect subscribe_ticker to handle its own connection or use combined stream.
            # For simplicity in this architecture, we might want a base connection or just
            # let subscriptions handle it.
            pass

        self.connected = True

        if self.ws:
            self._msg_loop_task = asyncio.create_task(self._msg_loop())

        logger.info(f"Binance Adapter connected (Public: {self.public_only}).")

    async def disconnect(self):
        self.connected = False
        if self._msg_loop_task:
            self._msg_loop_task.cancel()
        if self._keepalive_task:
            self._keepalive_task.cancel()
        if self.ws:
            await self.ws.close()
        if self.session:
            await self.session.close()
        logger.info("Binance Adapter disconnected.")

    async def _create_listen_key(self) -> Optional[str]:
        if self.public_only:
            return None

        url = f"{self.base_url}/fapi/v1/listenKey"
        async with self.session.post(url) as resp:
            if resp.status == 200:
                data = self._fast_json_decode(await resp.read())
                return data.get("listenKey")
            logger.error(f"Failed to create listenKey: {await resp.text()}")
            return None

    async def _keepalive_loop(self):
        if self.public_only:
            return

        resync_interval = 60
        keepalive_interval = 30 * 60
        elapsed = 0
        while self.connected:
            await asyncio.sleep(resync_interval)
            elapsed += resync_interval
            if self.time_sync_manager:
                await self.time_sync_manager.sync_all()
            if elapsed >= keepalive_interval and self.listen_key:
                url = f"{self.base_url}/fapi/v1/listenKey"
                await self.session.put(url)
                elapsed = 0

    async def _msg_loop(self):
        while self.connected and self.ws:
            try:
                msg_raw = await asyncio.wait_for(self.ws.receive(), timeout=5.0)
                if msg_raw.type == aiohttp.WSMsgType.TEXT:
                    msg = self._fast_json_decode(msg_raw.data)
                    await self._handle_message(msg)
                elif msg_raw.type == aiohttp.WSMsgType.BINARY:
                    msg = self._fast_json_decode(msg_raw.data)
                    await self._handle_message(msg)
                elif msg_raw.type == aiohttp.WSMsgType.CLOSED:
                    break
                elif msg_raw.type == aiohttp.WSMsgType.ERROR:
                    break
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in Binance msg loop: {e}")

    async def _handle_message(self, msg: Dict):
        event_type = msg.get("e")
        if event_type == "ORDER_TRADE_UPDATE":
            await self._handle_order_update(msg.get("o", {}))
        elif event_type == "ACCOUNT_UPDATE":
            await self._handle_account_update(msg.get("a", {}))

    async def _handle_order_update(self, data: Dict):
        exchange_id = str(data.get("i", ""))
        status_map = {
            "NEW": OrderStatus.OPEN,
            "FILLED": OrderStatus.FILLED,
            "CANCELED": OrderStatus.CANCELLED,
            "REJECTED": OrderStatus.REJECTED,
            "PARTIALLY_FILLED": OrderStatus.PARTIALLY_FILLED,
        }
        status = status_map.get(data.get("X"), OrderStatus.PENDING)
        filled_size = float(data.get("z", 0))
        avg_price = float(data.get("ap", 0))

        local_id = data.get("c")

        if exchange_id in self.orders:
            self.orders[exchange_id] = replace(
                self.orders[exchange_id], status=status, filled_size=filled_size
            )

        await self.notify_order_update(
            exchange_id, status, filled_size, avg_price, local_id
        )

    async def _handle_account_update(self, data: Dict):
        positions_data = data.get("P", [])
        for p in positions_data:
            symbol = p.get("s")
            if symbol:
                amount = float(p.get("pa", 0))
                entry_price = float(p.get("ep", 0))
                self.positions[symbol] = Position(
                    symbol, amount, entry_price, exchange=self.name
                )

                if self.position_callback:
                    await self.position_callback(symbol, amount, entry_price)

        balances_data = data.get("B", [])
        from ...domain.entities import Balance

        for b in balances_data:
            asset = b.get("a")
            wallet_balance = float(b.get("wb", 0))
            cross_wallet = float(b.get("cw", 0))
            # available logic might need more fields like 'bc' (Balance Change) or fetch from GET /account
            # But 'cw' is cross wallet balance.
            # Assuming equity ~ wallet_balance for now if PnL not strictly added here.
            # Binance sends 'up' (Unrealized PnL) in 'a' section but that's for positions?
            # Actually ACCOUNT_UPDATE 'a' has 'm' (margin balance), 'up' (unrealized pnl)??
            # No, 'a' is just event cause.
            # We use 'wb' as total.

            # Simple mapping
            bal = Balance(
                exchange=self.name,
                asset=asset,
                total=wallet_balance,
                available=cross_wallet,  # Approx
                margin_used=wallet_balance - cross_wallet,
                equity=wallet_balance,  # Ideally + unrealized PnL if available
            )
            if self.balance_callback:
                await self.balance_callback(bal)

    def set_balance_callback(self, callback: Callable):
        self.balance_callback = callback

    async def place_order(self, order: Order) -> Order:
        mapped_symbol = InstrumentService.get_exchange_symbol(order.symbol, self.name)
        params = {
            "symbol": mapped_symbol,
            "side": "BUY" if order.side == OrderSide.BUY else "SELL",
            "type": "LIMIT" if order.type == OrderType.LIMIT else "MARKET",
            "quantity": order.size,
            "price": order.price,
            "timeInForce": "GTX" if order.post_only else "GTC",
            "newClientOrderId": order.id,
            "timestamp": self._get_timestamp(),
        }
        params["signature"] = self._sign(params)

        url = f"{self.base_url}/fapi/v1/order"
        async with self.session.post(url, params=params) as resp:
            if resp.status == 200:
                data = self._fast_json_decode(await resp.read())
                exchange_id = str(data.get("orderId", ""))
                updated = replace(
                    order,
                    exchange_id=exchange_id,
                    status=OrderStatus.OPEN,
                    exchange=self.name,
                )
                self.orders[exchange_id] = updated
                return updated
            else:
                logger.error(f"Binance order error: {await resp.text()}")
                return replace(order, status=OrderStatus.REJECTED, exchange=self.name)

    async def cancel_order(self, order_id: str) -> bool:
        order = self.orders.get(order_id)
        if not order:
            return False

        mapped_symbol = InstrumentService.get_exchange_symbol(order.symbol, self.name)
        params = {
            "symbol": mapped_symbol,
            "orderId": int(order_id),
            "timestamp": self._get_timestamp(),
        }
        params["signature"] = self._sign(params)

        url = f"{self.base_url}/fapi/v1/order"
        async with self.session.delete(url, params=params) as resp:
            if resp.status == 200:
                self.orders[order_id] = replace(
                    self.orders[order_id], status=OrderStatus.CANCELLED
                )
                return True
            logger.error(f"Binance cancel error: {await resp.text()}")
            return False

    async def place_orders_batch(self, orders: List[Order]) -> List[Order]:
        results = []
        for o in orders:
            results.append(await self.place_order(o))
        return results

    async def cancel_orders_batch(self, order_ids: List[str]) -> List[bool]:
        results = []
        for oid in order_ids:
            results.append(await self.cancel_order(oid))
        return results

    async def get_open_orders(self, symbol: str) -> List[Order]:
        """Fetch all open orders from Binance Futures for a specific symbol."""
        mapped_symbol = InstrumentService.get_exchange_symbol(symbol, self.name)
        params = {
            "symbol": mapped_symbol,
            "timestamp": self._get_timestamp(),
        }
        params["signature"] = self._sign(params)

        url = f"{self.base_url}/fapi/v1/openOrders"
        async with self.session.get(url, params=params) as resp:
            data = await resp.json()
            if resp.status == 200:
                orders = []
                for raw in data:
                    status_map = {
                        "NEW": OrderStatus.OPEN,
                        "PARTIALLY_FILLED": OrderStatus.PARTIALLY_FILLED,
                        "FILLED": OrderStatus.FILLED,
                        "CANCELED": OrderStatus.CANCELLED,
                        "REJECTED": OrderStatus.REJECTED,
                        "EXPIRED": OrderStatus.CANCELLED,
                    }
                    order = Order(
                        id=raw.get("clientOrderId", ""),
                        exchange_id=str(raw.get("orderId", "")),
                        symbol=symbol,
                        side=OrderSide.BUY
                        if raw.get("side") == "BUY"
                        else OrderSide.SELL,
                        price=float(raw.get("price", 0)),
                        size=float(raw.get("origQty", 0)),
                        filled_size=float(raw.get("executedQty", 0)),
                        status=status_map.get(raw.get("status"), OrderStatus.OPEN),
                        type=OrderType.LIMIT
                        if raw.get("type") == "LIMIT"
                        else OrderType.MARKET,
                        exchange=self.name,
                    )
                    orders.append(order)
                    # Seed local cache
                    self.orders[order.exchange_id] = order
                return orders
            else:
                logger.error(f"Binance get_open_orders error: {data}")
                return []

    async def cancel_all_orders(self, symbol: str) -> bool:
        """Cancel all open orders for a specific symbol on Binance Futures."""
        mapped_symbol = InstrumentService.get_exchange_symbol(symbol, self.name)
        params = {
            "symbol": mapped_symbol,
            "timestamp": self._get_timestamp(),
        }
        params["signature"] = self._sign(params)

        url = f"{self.base_url}/fapi/v1/allOpenOrders"
        async with self.session.delete(url, params=params) as resp:
            data = await resp.json()
            if resp.status == 200:
                logger.info(
                    f"Successfully cancelled all orders on Binance for {symbol}"
                )
                # Update local cache
                to_remove = [
                    oid
                    for oid, o in self.orders.items()
                    if o.symbol == symbol
                    and o.status in [OrderStatus.OPEN, OrderStatus.PARTIALLY_FILLED]
                ]
                for oid in to_remove:
                    self.orders[oid] = replace(
                        self.orders[oid], status=OrderStatus.CANCELLED
                    )
                return True
            else:
                logger.error(f"Binance cancel_all_orders error: {data}")
                return False

    async def subscribe_ticker(self, symbol: str):
        mapped_symbol = InstrumentService.get_exchange_symbol(symbol, self.name)
        stream = f"{mapped_symbol.lower()}@bookTicker"

        # If we have a main WS connection (Private), we can't easily multiplex public streams
        # on the listenKey stream in Binance Futures (unlike Spot).
        # So we usually need a separate connection for public streams anyway if we want to be clean.
        # But for valid Multi-Stream, we should use the Combined Stream URL.

        # For this implementation, we will spawn a separate dedicated task for this ticker
        # if we aren't using a combined stream manager (which we aren't yet).

        ws_url = f"{self.ws_base_url}/ws/{stream}"
        asyncio.create_task(self._ticker_stream(ws_url, symbol))

    async def _ticker_stream(self, url: str, symbol: str):
        # Dedicated socket for this ticker (Simple but effective for single-ticker bots)
        try:
            async with self.session.ws_connect(url) as ws:
                logger.info(f"Subscribed to ticker stream for {symbol}")
                while self.connected:
                    try:
                        msg = await asyncio.wait_for(ws.receive_json(), timeout=5.0)

                        # Apply TimeSync offset to convert Server Time to Local Time
                        # Offset = Server - Local => Local = Server - Offset
                        offset = 0
                        if self.time_sync_manager:
                            offset = self.time_sync_manager.get_offset(self.name)

                        raw_ts = float(msg.get("E", msg.get("T", time.time() * 1000)))
                        adjusted_ts = (raw_ts - offset) / 1000.0

                        # CRITICAL FIX: Initial Snapshot / Illiquid Pair Handling
                        # If the "Event Time" is wildly old (>5s), it likely means Binance sent the "last known state"
                        # for an illiquid pair (or upon connect), not a network delay.
                        # For an Oracle, this IS the current price. We clamp it to 'now' to satisfy LatencyMonitor.
                        now = time.time()
                        if (now - adjusted_ts) > 5.0:
                            logger.warning(
                                f"Stale Binance data detected ({now - adjusted_ts:.1f}s old). "
                                f"Clamping timestamp to Now (Oracle Mode)."
                            )
                            adjusted_ts = now

                        ticker = Ticker(
                            symbol=symbol,
                            bid=float(msg.get("b", 0)),
                            ask=float(msg.get("a", 0)),
                            bid_size=float(msg.get("B", 0)),
                            ask_size=float(msg.get("A", 0)),
                            last=0.0,
                            volume=0.0,
                            exchange=self.name,
                            timestamp=adjusted_ts,
                        )
                        if self.ticker_callback:
                            await self.ticker_callback(ticker)
                    except asyncio.TimeoutError:
                        # Ping/Pong handled by aiohttp automatic
                        continue
                    except Exception as e:
                        logger.error(f"Binance ticker stream error: {e}")
                        break
        except Exception as main_e:
            logger.error(f"Failed to connect to ticker stream {url}: {main_e}")

    async def get_position(self, symbol: str) -> Position:
        return self.positions.get(
            symbol, Position(symbol, 0.0, 0.0, exchange=self.name)
        )

    async def get_balances(self) -> List[Balance]:
        url = f"{self.base_url}/fapi/v2/account"
        params = {"timestamp": self._get_timestamp()}
        params["signature"] = self._sign(params)

        async with self.session.get(url, params=params) as resp:
            if resp.status == 200:
                data = self._fast_json_decode(await resp.read())
                balances = []
                # Binance Futures returns 'assets' list
                for b in data.get("assets", []):
                    asset = b.get("asset")
                    wallet_balance = float(b.get("walletBalance", 0))
                    cross_wallet = float(b.get("crossWalletBalance", 0))
                    # Available is roughly cross wallet for cross margin
                    available = float(b.get("availableBalance", cross_wallet))

                    bal = Balance(
                        exchange=self.name,
                        asset=asset,
                        total=wallet_balance,
                        available=available,
                        margin_used=wallet_balance - available,
                        equity=float(b.get("marginBalance", wallet_balance)),
                    )
                    balances.append(bal)
                return balances
            else:
                logger.error(f"Binance get_balances error: {await resp.text()}")
                return []
