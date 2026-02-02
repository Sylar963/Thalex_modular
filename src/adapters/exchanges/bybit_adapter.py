import asyncio
import logging
import time
import hmac
import hashlib
from typing import Dict, List, Optional
from dataclasses import replace

try:
    import aiohttp
except ImportError:
    raise ImportError(
        "aiohttp is required for BybitAdapter. Install with: pip install aiohttp"
    )

from .base_adapter import BaseExchangeAdapter
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

        self._msg_loop_task: Optional[asyncio.Task] = None
        self._ping_task: Optional[asyncio.Task] = None

    @property
    def name(self) -> str:
        return "bybit"

    def _sign(self, timestamp: int, payload: str) -> str:
        param_str = f"{timestamp}{self.api_key}5000{payload}"
        return hmac.new(
            self.api_secret.encode("utf-8"),
            param_str.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

    def _get_headers(self, payload: str = "") -> Dict[str, str]:
        timestamp = int(time.time() * 1000)
        return {
            "X-BAPI-API-KEY": self.api_key,
            "X-BAPI-TIMESTAMP": str(timestamp),
            "X-BAPI-RECV-WINDOW": "5000",
            "X-BAPI-SIGN": self._sign(timestamp, payload),
            "Content-Type": "application/json",
        }

    async def connect(self):
        logger.info(
            f"Connecting to Bybit ({'Testnet' if self.testnet else 'Mainnet'})..."
        )
        self.session = aiohttp.ClientSession()

        timestamp = int(time.time() * 1000)
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
        sub_msg = {"op": "subscribe", "args": ["order", "execution", "position"]}
        await self.ws_private.send_json(sub_msg)

    async def _ping_loop(self):
        while self.connected:
            await asyncio.sleep(20)
            if self.ws_private:
                await self.ws_private.send_json({"op": "ping"})

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
        filled_size = float(data.get("cumExecQty", 0))
        avg_price = float(data.get("avgPrice", 0))

        if exchange_id in self.orders:
            self.orders[exchange_id] = replace(
                self.orders[exchange_id], status=status, filled_size=filled_size
            )

        if self.order_callback:
            await self.order_callback(exchange_id, status, filled_size, avg_price)

    async def _handle_position_update(self, data: Dict):
        symbol = data.get("symbol")
        if symbol:
            amount = float(data.get("size", 0))
            entry_price = float(data.get("entryPrice", 0))
            self.positions[symbol] = Position(
                symbol, amount, entry_price, exchange=self.name
            )

            if self.position_callback:
                await self.position_callback(symbol, amount, entry_price)

    async def place_order(self, order: Order) -> Order:
        payload = {
            "category": "linear",
            "symbol": order.symbol,
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

        payload = {"category": "linear", "symbol": order.symbol, "orderId": order_id}
        payload_str = json.dumps(payload)

        url = f"{self.base_url}/v5/order/cancel"
        async with self.session.post(
            url, data=payload_str, headers=self._get_headers(payload_str)
        ) as resp:
            data = await resp.json()
            if data.get("retCode") == 0:
                self.orders[order_id] = replace(
                    self.orders[order_id], status=OrderStatus.CANCELLED
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
        async with self.session.ws_connect(self.ws_public_url) as ws:
            sub_msg = {"op": "subscribe", "args": [f"orderbook.1.{symbol}"]}
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
                            bid=float(bids[0][0]) if bids else 0.0,
                            ask=float(asks[0][0]) if asks else 0.0,
                            bid_size=float(bids[0][1]) if bids else 0.0,
                            ask_size=float(asks[0][1]) if asks else 0.0,
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
