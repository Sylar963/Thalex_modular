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
from ...services.instrument_service import InstrumentService
from ...domain.entities import (
    Order,
    OrderSide,
    OrderType,
    OrderStatus,
    Position,
    Ticker,
    Balance,
)

logger = logging.getLogger(__name__)


class BybitAdapter(BaseExchangeAdapter):
    REST_URL = "https://api.bybit.com"
    REST_TESTNET_URL = "https://api-testnet.bybit.com"
    WS_PUBLIC_URL = "wss://stream.bybit.com/v5/public/linear"
    WS_PRIVATE_URL = "wss://stream.bybit.com/v5/private"
    WS_TESTNET_PUBLIC_URL = "wss://stream-testnet.bybit.com/v5/public/linear"
    WS_TESTNET_PRIVATE_URL = "wss://stream-testnet.bybit.com/v5/private"

    def __init__(self, api_key: str, api_secret: str, testnet: bool = True):
        super().__init__(api_key, api_secret, testnet)
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
        self._time_offset_ms: int = 0
        self.balance_callback: Optional[Callable] = None

    def _safe_float(self, value, default: float = 0.0) -> float:
        if value is None or value == "":
            return default
        try:
            return float(value)
        except (ValueError, TypeError):
            return default

    @property
    def name(self) -> str:
        return "bybit"

    async def _sync_server_time(self):
        url = f"{self.base_url}/v5/market/time"
        try:
            async with self.session.get(url) as resp:
                data = await resp.json()
                if data.get("retCode") == 0:
                    server_time = (
                        int(data.get("result", {}).get("timeSecond", 0)) * 1000
                    )
                    if server_time == 0:
                        server_time = (
                            int(data.get("result", {}).get("timeNano", 0)) // 1_000_000
                        )
                    local_time = int(time.time() * 1000)
                    self._time_offset_ms = server_time - local_time
                    logger.info(
                        f"Bybit time offset: {self._time_offset_ms}ms (server ahead)"
                        if self._time_offset_ms > 0
                        else f"Bybit time offset: {self._time_offset_ms}ms (local ahead)"
                    )
        except Exception as e:
            logger.warning(f"Failed to sync Bybit server time: {e}")
        self._last_sync_time = time.time()

    def _get_timestamp(self) -> int:
        if time.time() - getattr(self, "_last_sync_time", 0) > 10:
            asyncio.create_task(self._sync_server_time())
        return int(time.time() * 1000) + self._time_offset_ms

    def _sign(self, timestamp: int, payload: str) -> str:
        recv_window = "10000"
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
            "X-BAPI-RECV-WINDOW": "10000",
            "X-BAPI-SIGN": self._sign(timestamp, payload),
            "Content-Type": "application/json",
        }

    async def connect(self):
        logger.info(
            f"Connecting to Bybit ({'Testnet' if self.testnet else 'Mainnet'})..."
        )
        self.session = aiohttp.ClientSession()

        await self._sync_server_time()

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
                await self._sync_server_time()
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
                logger.error(f"Error in Bybit msg loop: {e}")

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

        if exchange_id in self.orders:
            self.orders[exchange_id] = replace(
                self.orders[exchange_id], status=status, filled_size=filled_size
            )

        if self.order_callback:
            await self.order_callback(exchange_id, status, filled_size, avg_price)

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

    def set_balance_callback(self, callback: Callable):
        self.balance_callback = callback

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
        import json

        payload_str = json.dumps(payload)

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

        import json

        mapped_symbol = InstrumentService.get_exchange_symbol(order.symbol, self.name)
        payload = {"category": "linear", "symbol": mapped_symbol, "orderId": order_id}
        payload_str = json.dumps(payload)

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
                if self.order_callback:
                    await self.order_callback(
                        order_id, OrderStatus.CANCELLED, order.filled_size, 0.0
                    )
                return True

            logger.error(f"Bybit cancel error: {data}")
            return False

    async def place_orders_batch(self, orders: List[Order]) -> List[Order]:
        return [await self.place_order(o) for o in orders]

    async def cancel_orders_batch(self, order_ids: List[str]) -> List[bool]:
        return [await self.cancel_order(oid) for oid in order_ids]

    async def subscribe_ticker(self, symbol: str):
        asyncio.create_task(self._ticker_stream(symbol))

    async def _ticker_stream(self, symbol: str):
        mapped_symbol = InstrumentService.get_exchange_symbol(symbol, self.name)
        async with self.session.ws_connect(self.ws_public_url) as ws:
            sub_msg = {"op": "subscribe", "args": [f"orderbook.1.{mapped_symbol}"]}
            await ws.send_json(sub_msg)

            while self.connected:
                try:
                    msg = await asyncio.wait_for(ws.receive_json(), timeout=5.0)
                    if msg.get("topic", "").startswith("orderbook"):
                        data = msg.get("data", {})
                        bids = data.get("b", [[0, 0]])
                        asks = data.get("a", [[0, 0]])
                        ticker = Ticker(
                            symbol=symbol,
                            bid=self._safe_float(bids[0][0]) if bids else 0.0,
                            ask=self._safe_float(asks[0][0]) if asks else 0.0,
                            bid_size=self._safe_float(bids[0][1]) if bids else 0.0,
                            ask_size=self._safe_float(asks[0][1]) if asks else 0.0,
                            last=0.0,
                            volume=0.0,
                            exchange=self.name,
                            timestamp=time.time(),
                        )
                        if self.ticker_callback:
                            await self.ticker_callback(ticker)
                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    logger.error(f"Bybit ticker stream error: {e}")
                    break

    async def get_position(self, symbol: str) -> Position:
        return self.positions.get(
            symbol, Position(symbol, 0.0, 0.0, exchange=self.name)
        )
