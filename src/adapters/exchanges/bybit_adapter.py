import asyncio
import logging
import time
import hmac
import hashlib
from typing import Dict, List, Optional, Callable
from dataclasses import replace

try:
    import aiohttp
except ImportError:
    raise ImportError(
        "aiohttp is required for BybitAdapter. Install with: pip install aiohttp"
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
    Balance,
    Trade,
)

logger = logging.getLogger(__name__)


class BybitAdapter(BaseExchangeAdapter):
    REST_URL = "https://api.bybit.com"
    REST_TESTNET_URL = "https://api-testnet.bybit.com"
    WS_PUBLIC_URL = "wss://stream.bybit.com/v5/public/linear"
    WS_PRIVATE_URL = "wss://stream.bybit.com/v5/private"
    WS_TESTNET_PUBLIC_URL = "wss://stream-testnet.bybit.com/v5/public/linear"
    WS_TESTNET_PRIVATE_URL = "wss://stream-testnet.bybit.com/v5/private"

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        testnet: bool = True,
        time_sync_manager: Optional[TimeSyncManager] = None,
    ):
        super().__init__(api_key, api_secret, testnet, time_sync_manager)
        self.base_url = self.REST_TESTNET_URL if testnet else self.REST_URL
        self.ws_public_url = (
            self.WS_TESTNET_PUBLIC_URL if testnet else self.WS_PUBLIC_URL
        )
        self.ws_private_url = (
            self.WS_TESTNET_PRIVATE_URL if testnet else self.WS_PRIVATE_URL
        )

        self.session: Optional[aiohttp.ClientSession] = None
        self.ws_private: Optional[aiohttp.ClientWebSocketResponse] = None

        self.positions: Dict[str, Position] = {}
        self.orders: Dict[str, Order] = {}
        self.tick_size: float = 0.01
        self.lot_size: float = 0.001

        self._msg_loop_task: Optional[asyncio.Task] = None
        self._ping_task: Optional[asyncio.Task] = None
        self._public_loop_task: Optional[asyncio.Task] = None
        self.ws_public: Optional[aiohttp.ClientWebSocketResponse] = None
        self.balance_callback: Optional[Callable] = None

    @property
    def name(self) -> str:
        return "bybit"

    async def get_server_time(self) -> int:
        """Fetch current Bybit server time in milliseconds."""
        url = f"{self.base_url}/v5/market/time"
        try:
            async with self.session.get(url) as resp:
                data = await resp.json()
                if data.get("retCode") == 0:
                    res = data.get("result", {})
                    # Prefer timeSecond (s) * 1000, or use timeNano (ns) // 1M
                    s_time = int(res.get("timeSecond", 0)) * 1000
                    if s_time == 0:
                        s_time = int(res.get("timeNano", 0)) // 1_000_000
                    return s_time
                raise Exception(f"Bybit API error: {data.get('retMsg')}")
        except Exception as e:
            logger.error(f"Failed to fetch Bybit server time: {e}")
            raise

    def _get_timestamp(self) -> int:
        return super()._get_timestamp()

    def _sign(self, timestamp: int, payload: str) -> str:
        recv_window = str(self.RECV_WINDOW)
        param_str = f"{timestamp}{self.api_key}{recv_window}{payload}"
        return hmac.new(
            self.api_secret.encode("utf-8"),
            param_str.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

    def _get_headers(self, payload: str = "") -> Dict[str, str]:
        timestamp = self._get_timestamp()
        return {
            "X-BAPI-API-KEY": self.api_key,
            "X-BAPI-TIMESTAMP": str(timestamp),
            "X-BAPI-RECV-WINDOW": str(self.RECV_WINDOW),
            "X-BAPI-SIGN": self._sign(timestamp, payload),
            "Content-Type": "application/json",
        }

    async def connect(self):
        logger.info(
            f"Connecting to Bybit ({'Testnet' if self.testnet else 'Mainnet'})..."
        )
        self.session = aiohttp.ClientSession()

        if self.time_sync_manager:
            await self.time_sync_manager.sync_all()

        timestamp = self._get_timestamp()
        expires = timestamp + 10000
        signature = hmac.new(
            self.api_secret.encode("utf-8"),
            f"GET/realtime{expires}".encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

        self.ws_private = await self.session.ws_connect(self.ws_private_url)
        auth_msg = {"op": "auth", "args": [self.api_key, expires, signature]}
        await self.ws_private.send_json(auth_msg)

        self.connected = True
        self._msg_loop_task = asyncio.create_task(self._msg_loop())
        self._ping_task = asyncio.create_task(self._ping_loop())

        await self._subscribe_private()
        logger.info("Bybit Adapter connected.")

    async def fetch_instrument_info(
        self, symbol: str, category: str = "linear"
    ) -> Dict:
        mapped_symbol = InstrumentService.get_exchange_symbol(symbol, self.name)
        url = f"{self.base_url}/v5/market/instruments-info"
        params = {"category": category, "symbol": mapped_symbol}
        async with self.session.get(url, params=params) as resp:
            data = await resp.json()
            items = data.get("result", {}).get("list", [])

            if data.get("retCode") == 0 and items:
                pass
            elif category == "linear":
                # Fallback to option
                params["category"] = "option"
                async with self.session.get(url, params=params) as resp2:
                    data2 = await resp2.json()
                    if data2.get("retCode") == 0 and data2.get("result", {}).get(
                        "list", []
                    ):
                        data = data2
                        category = "option"  # Update for future use if needed
                        items = data.get("result", {}).get("list", [])

            if items:
                info = items[0]
                self.tick_size = self._safe_float(
                    info.get("priceFilter", {}).get("tickSize"), 0.01
                )
                self.lot_size = self._safe_float(
                    info.get("lotSizeFilter", {}).get("minOrderQty"), 0.001
                )
                logger.info(
                    f"Fetched {symbol} tick_size={self.tick_size}, lot_size={self.lot_size}"
                )
                return info
        return {}

    async def get_balances(self) -> List[Balance]:
        url = f"{self.base_url}/v5/account/wallet-balance"
        params = {"accountType": "UNIFIED"}
        try:
            timestamp = self._get_timestamp()
            query_str = "accountType=UNIFIED"
            signature = self._sign(timestamp, query_str)
            headers = {
                "X-BAPI-API-KEY": self.api_key,
                "X-BAPI-TIMESTAMP": str(timestamp),
                "X-BAPI-RECV-WINDOW": "10000",
                "X-BAPI-SIGN": signature,
                # "Content-Type" not strictly needed for GET but ok
            }

            async with self.session.get(url, params=params, headers=headers) as resp:
                data = await resp.json()
                if data.get("retCode") == 0:
                    result = data.get("result", {})
                    list_data = result.get("list", [])
                    balances = []
                    if list_data:
                        account_data = list_data[0]
                        coins = account_data.get("coin", [])
                        for c in coins:
                            coin_name = c.get("coin")
                            wallet_bal = self._safe_float(c.get("walletBalance"))
                            available = self._safe_float(c.get("availableToWithdraw"))
                            equity = self._safe_float(c.get("equity"))

                            bal = Balance(
                                exchange=self.name,
                                asset=coin_name,
                                total=wallet_bal,
                                available=available,
                                margin_used=wallet_bal - available,
                                equity=equity,
                            )
                            balances.append(bal)
                            if self.balance_callback:
                                await self.balance_callback(bal)
                    return balances
        except Exception as e:
            logger.error(f"Failed to fetch balances: {e}")
        return []

    async def disconnect(self):
        self.connected = False
        if self._msg_loop_task:
            self._msg_loop_task.cancel()
        if self._ping_task:
            self._ping_task.cancel()
        if self.ws_private:
            await self.ws_private.close()
        if self.session:
            await self.session.close()
        logger.info("Bybit Adapter disconnected.")

    async def _subscribe_private(self):
        sub_msg = {
            "op": "subscribe",
            "args": ["order", "execution", "position", "wallet"],
        }
        await self.ws_private.send_json(sub_msg)

    async def _ping_loop(self):
        resync_counter = 0
        while self.connected:
            await asyncio.sleep(20)
            if self.ws_private:
                await self.ws_private.send_json({"op": "ping"})
            resync_counter += 1
            if resync_counter >= 2:
                if self.time_sync_manager:
                    await self.time_sync_manager.sync_all()
                resync_counter = 0

    async def _msg_loop(self):
        while self.connected and self.ws_private:
            try:
                msg = await asyncio.wait_for(
                    self.ws_private.receive_json(), timeout=5.0
                )
                if not msg:
                    continue
                await self._handle_message(msg)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                import traceback

                logger.error(f"Error in Bybit msg loop: {e}\n{traceback.format_exc()}")

    async def _handle_message(self, msg: Dict):
        topic = msg.get("topic")
        if topic == "order":
            for item in msg.get("data", []):
                await self._handle_order_update(item)
        elif topic == "position":
            for item in msg.get("data", []):
                await self._handle_position_update(item)
        elif topic == "wallet":
            for item in msg.get("data", []):
                await self._handle_wallet_update(item)

    async def _handle_order_update(self, data: Dict):
        exchange_id = data.get("orderId", "")
        status_map = {
            "New": OrderStatus.OPEN,
            "Filled": OrderStatus.FILLED,
            "Cancelled": OrderStatus.CANCELLED,
            "Rejected": OrderStatus.REJECTED,
            "PartiallyFilled": OrderStatus.PARTIALLY_FILLED,
        }
        status = status_map.get(data.get("orderStatus"), OrderStatus.PENDING)
        filled_size = self._safe_float(data.get("cumExecQty"))
        avg_price = self._safe_float(data.get("avgPrice"))

        local_id = data.get("orderLinkId")

        if exchange_id in self.orders:
            self.orders[exchange_id] = replace(
                self.orders[exchange_id], status=status, filled_size=filled_size
            )

        await self.notify_order_update(
            exchange_id, status, filled_size, avg_price, local_id
        )

    async def _handle_position_update(self, data: Dict):
        symbol = data.get("symbol")
        if symbol:
            amount = self._safe_float(data.get("size"))
            entry_price = self._safe_float(data.get("entryPrice"))
            self.positions[symbol] = Position(
                symbol, amount, entry_price, exchange=self.name
            )

            if self.position_callback:
                await self.position_callback(symbol, amount, entry_price)

    async def _handle_wallet_update(self, data: Dict):
        logger.info(f"raw wallet data: {data}")
        coins = data.get("coin", [])
        for coin_data in coins:
            coin = coin_data.get("coin", "USD")
            equity = self._safe_float(coin_data.get("equity"))
            wallet_balance = self._safe_float(coin_data.get("walletBalance"))
            available = self._safe_float(coin_data.get("availableToWithdraw"))
            # Bybit Unified Account: totalEquity, etc. might be in different fields
            # For standard account or UTA, 'equity' usually present.
            # margin used = equity - available (approx) or totalMargin

            bal = Balance(
                exchange=self.name,
                asset=coin,
                total=wallet_balance,
                available=available,
                margin_used=wallet_balance - available,  # Approx
                equity=equity,
            )
            if self.balance_callback:
                await self.balance_callback(bal)

    async def place_order(self, order: Order) -> Order:
        mapped_symbol = InstrumentService.get_exchange_symbol(order.symbol, self.name)
        payload = {
            "category": "linear",
            "symbol": mapped_symbol,
            "side": "Buy" if order.side == OrderSide.BUY else "Sell",
            "orderType": "Limit" if order.type == OrderType.LIMIT else "Market",
            "qty": str(order.size),
            "price": str(order.price),
            "timeInForce": "PostOnly" if order.post_only else "GTC",
            "orderLinkId": order.id,
        }
        payload_str = self._fast_json_encode(payload)

        url = f"{self.base_url}/v5/order/create"
        async with self.session.post(
            url, data=payload_str, headers=self._get_headers(payload_str)
        ) as resp:
            data = await resp.json()
            if data.get("retCode") == 0:
                result = data.get("result", {})
                exchange_id = result.get("orderId", "")
                updated = replace(
                    order,
                    exchange_id=exchange_id,
                    status=OrderStatus.OPEN,
                    exchange=self.name,
                )
                self.orders[exchange_id] = updated
                return updated
            else:
                logger.error(f"Bybit order error: {data}")
                return replace(order, status=OrderStatus.REJECTED, exchange=self.name)

    async def cancel_order(self, order_id: str) -> bool:
        order = self.orders.get(order_id)
        if not order:
            return False

        mapped_symbol = InstrumentService.get_exchange_symbol(order.symbol, self.name)
        payload = {"category": "linear", "symbol": mapped_symbol, "orderId": order_id}
        payload_str = self._fast_json_encode(payload)

        url = f"{self.base_url}/v5/order/cancel"
        async with self.session.post(
            url, data=payload_str, headers=self._get_headers(payload_str)
        ) as resp:
            data = await resp.json()
            ret_code = data.get("retCode")
            if ret_code == 0:
                self.orders[order_id] = replace(
                    self.orders[order_id], status=OrderStatus.CANCELLED
                )
                return True

            if ret_code == 110001:
                logger.warning(
                    f"Bybit order {order_id} not found or too late to cancel. Treating as cancelled."
                )
                self.orders[order_id] = replace(
                    self.orders[order_id], status=OrderStatus.CANCELLED
                )
                await self.notify_order_update(
                    order_id, OrderStatus.CANCELLED, order.filled_size, 0.0, order.id
                )
                return True

            logger.error(f"Bybit cancel error: {data}")
            return False

    async def place_orders_batch(self, orders: List[Order]) -> List[Order]:
        return [await self.place_order(o) for o in orders]

    async def cancel_orders_batch(self, order_ids: List[str]) -> List[bool]:
        return [await self.cancel_order(oid) for oid in order_ids]

    async def get_open_orders(self, symbol: str) -> List[Order]:
        """Fetch all open orders from Bybit for a specific symbol."""
        mapped_symbol = InstrumentService.get_exchange_symbol(symbol, self.name)
        params = f"category=linear&symbol={mapped_symbol}"
        url = f"{self.base_url}/v5/order/realtime?{params}"

        async with self.session.get(url, headers=self._get_headers(params)) as resp:
            data = await resp.json()
            if data.get("retCode") == 0:
                result = data.get("result", {})
                raw_orders = result.get("list", [])
                orders = []
                for raw in raw_orders:
                    status_map = {
                        "New": OrderStatus.OPEN,
                        "PartiallyFilled": OrderStatus.PARTIALLY_FILLED,
                        "Filled": OrderStatus.FILLED,
                        "Cancelled": OrderStatus.CANCELLED,
                        "Rejected": OrderStatus.REJECTED,
                    }
                    order = Order(
                        id=raw.get("orderLinkId", ""),
                        exchange_id=raw.get("orderId", ""),
                        symbol=symbol,
                        side=OrderSide.BUY
                        if raw.get("side") == "Buy"
                        else OrderSide.SELL,
                        price=float(raw.get("price", 0)),
                        size=float(raw.get("qty", 0)),
                        filled_size=float(raw.get("cumExecQty", 0)),
                        status=status_map.get(raw.get("orderStatus"), OrderStatus.OPEN),
                        type=OrderType.LIMIT
                        if raw.get("orderType") == "Limit"
                        else OrderType.MARKET,
                        exchange=self.name,
                    )
                    orders.append(order)
                    # Seed local cache
                    self.orders[order.exchange_id] = order
                return orders
            else:
                logger.error(f"Bybit get_open_orders error: {data}")
                return []

    async def cancel_all_orders(self, symbol: str) -> bool:
        """Cancel all open orders for a specific symbol on Bybit."""
        mapped_symbol = InstrumentService.get_exchange_symbol(symbol, self.name)
        payload = {"category": "linear", "symbol": mapped_symbol}
        payload_str = self._fast_json_encode(payload)

        url = f"{self.base_url}/v5/order/cancel-all"
        async with self.session.post(
            url, data=payload_str, headers=self._get_headers(payload_str)
        ) as resp:
            data = await resp.json()
            if data.get("retCode") == 0:
                logger.info(f"Successfully cancelled all orders on Bybit for {symbol}")
                # Clear matching local orders
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
                logger.error(f"Bybit cancel_all_orders error: {data}")
                return False

    async def subscribe_ticker(self, symbol: str):
        """Subscribe to orderbook stream."""
        await self._ensure_public_conn()
        mapped_symbol = InstrumentService.get_exchange_symbol(symbol, self.name)
        topic = f"orderbook.1.{mapped_symbol}"
        if self.ws_public and not self.ws_public.closed:
            await self.ws_public.send_json({"op": "subscribe", "args": [topic]})
            logger.info(f"Subscribed to {topic}")

    async def subscribe_trades(self, symbol: str):
        """Subscribe to public trade stream."""
        await self._ensure_public_conn()
        mapped_symbol = InstrumentService.get_exchange_symbol(symbol, self.name)
        topic = f"publicTrade.{mapped_symbol}"
        if self.ws_public and not self.ws_public.closed:
            await self.ws_public.send_json({"op": "subscribe", "args": [topic]})
            logger.info(f"Subscribed to {topic}")

    async def _ensure_public_conn(self):
        if self.ws_public is None or self.ws_public.closed:
            if not self._public_loop_task or self._public_loop_task.done():
                self._public_loop_task = asyncio.create_task(self._public_msg_loop())
                # Wait for connection
                for _ in range(10):
                    if self.ws_public and not self.ws_public.closed:
                        break
                    await asyncio.sleep(0.5)

    async def _public_msg_loop(self):
        """Unified loop for public topics (Ticker, Trades)."""
        while self.connected:
            try:
                # Reconnect logic
                async with self.session.ws_connect(self.ws_public_url) as ws:
                    self.ws_public = ws
                    logger.info("Connected to Bybit Public WS")

                    while not ws.closed and self.connected:
                        try:
                            msg = await asyncio.wait_for(ws.receive_json(), timeout=5.0)
                            if not msg:
                                continue
                            topic = msg.get("topic", "")
                            if topic.startswith("orderbook"):
                                await self._handle_orderbook_msg(msg)
                            elif topic.startswith("publicTrade"):
                                await self._handle_public_trade_msg(msg)
                        except asyncio.TimeoutError:
                            # Send Ping
                            await ws.send_json({"op": "ping"})
                            continue

            except Exception as e:
                logger.error(f"Bybit Public WS Error: {e}")
                await asyncio.sleep(2.0)
            finally:
                self.ws_public = None

    async def _handle_orderbook_msg(self, msg: Dict):
        symbol = msg.get("topic", "").split(".")[-1]  # orderbook.1.BTCUSDT
        # We need to map back to our symbol if needed, or just pipe it thru
        # Ideally we map back. But for now let's use the symbol from topic if plausible.

        data = msg.get("data", {})
        bids = data.get("b", [])
        asks = data.get("a", [])

        # safely handle empty lists
        best_bid = bids[0] if bids else [0, 0]
        best_ask = asks[0] if asks else [0, 0]

        ticker = Ticker(
            symbol=symbol,
            bid=self._safe_float(best_bid[0]),
            ask=self._safe_float(best_ask[0]),
            bid_size=self._safe_float(best_bid[1]),
            ask_size=self._safe_float(best_ask[1]),
            last=0.0,
            volume=0.0,
            exchange=self.name,
            timestamp=time.time(),
        )
        if self.ticker_callback:
            await self.ticker_callback(ticker)

    async def _handle_public_trade_msg(self, msg: Dict):
        # topic: publicTrade.BTCUSDT
        data = msg.get("data", [])
        # publicTrade data is a list of trades
        for item in data:
            # "T": 1672304486866, "s": "BTCUSDT", "S": "Buy", "v": "0.001", "p": "16578.50", ...
            ts_ms = item.get("T")
            timestamp = ts_ms / 1000.0 if ts_ms else time.time()

            trade = Trade(
                id=item.get("i", ""),
                order_id="",  # Public trade has no order_id for us
                symbol=item.get("s"),
                side=OrderSide.BUY if item.get("S") == "Buy" else OrderSide.SELL,
                price=self._safe_float(item.get("p")),
                size=self._safe_float(item.get("v")),
                exchange=self.name,
                timestamp=timestamp,
            )
            if self.trade_callback:
                asyncio.create_task(self.trade_callback(trade))

    async def get_position(self, symbol: str) -> Position:
        return self.positions.get(
            symbol, Position(symbol, 0.0, 0.0, exchange=self.name)
        )
