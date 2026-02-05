import asyncio
import logging
import time
import hmac
import hashlib
from typing import Dict, List, Optional
from urllib.parse import urlencode
from dataclasses import replace

try:
    import aiohttp
except ImportError:
    raise ImportError(
        "aiohttp is required for BinanceAdapter. Install with: pip install aiohttp"
    )

from .base_adapter import BaseExchangeAdapter
from ...services.instrument_service import InstrumentService
from ...domain.entities import (
    Order,
    OrderSide,
    OrderType,
    OrderStatus,
    Position,
    Ticker,
    Trade,
)

logger = logging.getLogger(__name__)


class BinanceAdapter(BaseExchangeAdapter):
    REST_URL = "https://fapi.binance.com"
    REST_TESTNET_URL = "https://demo-fapi.binance.com"
    WS_URL = "wss://fstream.binance.com"
    WS_TESTNET_URL = "wss://fstream.binancefuture.com"

    def __init__(self, api_key: str, api_secret: str, testnet: bool = True):
        super().__init__(api_key, api_secret, testnet)
        self.base_url = self.REST_TESTNET_URL if testnet else self.REST_URL
        self.ws_base_url = self.WS_TESTNET_URL if testnet else self.WS_URL

        self.session: Optional[aiohttp.ClientSession] = None
        self.ws: Optional[aiohttp.ClientWebSocketResponse] = None
        self.listen_key: Optional[str] = None

        self.positions: Dict[str, Position] = {}
        self.positions: Dict[str, Position] = {}
        self.orders: Dict[str, Order] = {}
        self.balance_callback: Optional[Callable] = None

        self._msg_loop_task: Optional[asyncio.Task] = None
        self._keepalive_task: Optional[asyncio.Task] = None
        self._time_offset_ms: int = 0

    @property
    def name(self) -> str:
        return "binance"

    async def _sync_server_time(self):
        url = f"{self.base_url}/fapi/v1/time"
        try:
            async with self.session.get(url) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    server_time = data.get("serverTime", 0)
                    local_time = int(time.time() * 1000)
                    self._time_offset_ms = server_time - local_time
                    logger.info(
                        f"Binance time offset: {self._time_offset_ms}ms (server ahead)"
                        if self._time_offset_ms > 0
                        else f"Binance time offset: {self._time_offset_ms}ms (local ahead)"
                    )
        except Exception as e:
            logger.warning(f"Failed to sync Binance server time: {e}")

    def _get_timestamp(self) -> int:
        return int(time.time() * 1000) + self._time_offset_ms

    def _sign(self, params: Dict) -> str:
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
        self.session = aiohttp.ClientSession(headers={"X-MBX-APIKEY": self.api_key})

        await self._sync_server_time()

        self.listen_key = await self._create_listen_key()
        if not self.listen_key:
            raise ConnectionError("Failed to obtain listenKey from Binance")

        ws_url = f"{self.ws_base_url}/ws/{self.listen_key}"
        self.ws = await self.session.ws_connect(ws_url)
        self.connected = True

        self._msg_loop_task = asyncio.create_task(self._msg_loop())
        self._keepalive_task = asyncio.create_task(self._keepalive_loop())

        logger.info("Binance Adapter connected.")

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
        url = f"{self.base_url}/fapi/v1/listenKey"
        async with self.session.post(url) as resp:
            if resp.status == 200:
                data = await resp.json()
                return data.get("listenKey")
            logger.error(f"Failed to create listenKey: {await resp.text()}")
            return None

    async def _keepalive_loop(self):
        resync_interval = 60
        keepalive_interval = 30 * 60
        elapsed = 0
        while self.connected:
            await asyncio.sleep(resync_interval)
            elapsed += resync_interval
            await self._sync_server_time()
            if elapsed >= keepalive_interval and self.listen_key:
                url = f"{self.base_url}/fapi/v1/listenKey"
                await self.session.put(url)
                elapsed = 0

    async def _msg_loop(self):
        while self.connected and self.ws:
            try:
                msg = await asyncio.wait_for(self.ws.receive_json(), timeout=5.0)
                if not msg:
                    continue
                await self._handle_message(msg)
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

        if exchange_id in self.orders:
            self.orders[exchange_id] = replace(
                self.orders[exchange_id], status=status, filled_size=filled_size
            )

        if self.order_callback:
            await self.order_callback(exchange_id, status, filled_size, avg_price)

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
                data = await resp.json()
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

    async def subscribe_ticker(self, symbol: str):
        mapped_symbol = InstrumentService.get_exchange_symbol(symbol, self.name)
        stream = f"{mapped_symbol.lower()}@bookTicker"
        ws_url = f"{self.ws_base_url}/ws/{stream}"
        asyncio.create_task(self._ticker_stream(ws_url, symbol))

    async def _ticker_stream(self, url: str, symbol: str):
        async with self.session.ws_connect(url) as ws:
            while self.connected:
                try:
                    msg = await asyncio.wait_for(ws.receive_json(), timeout=5.0)
                    ticker = Ticker(
                        symbol=symbol,
                        bid=float(msg.get("b", 0)),
                        ask=float(msg.get("a", 0)),
                        bid_size=float(msg.get("B", 0)),
                        ask_size=float(msg.get("A", 0)),
                        last=0.0,
                        volume=0.0,
                        exchange=self.name,
                        timestamp=float(msg.get("T", time.time() * 1000)) / 1000.0,
                    )
                    if self.ticker_callback:
                        await self.ticker_callback(ticker)
                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    logger.error(f"Binance ticker stream error: {e}")
                    break

    async def get_position(self, symbol: str) -> Position:
        return self.positions.get(
            symbol, Position(symbol, 0.0, 0.0, exchange=self.name)
        )
