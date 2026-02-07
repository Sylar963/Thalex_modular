import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Callable
from dataclasses import replace

try:
    import aiohttp
    from eth_account import Account
    from eth_account.messages import encode_typed_data
except ImportError:
    raise ImportError(
        "aiohttp and eth_account are required for HyperliquidAdapter. Install with: pip install aiohttp eth_account"
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
    Balance,
)

logger = logging.getLogger(__name__)


class HyperliquidAdapter(BaseExchangeAdapter):
    REST_URL = "https://api.hyperliquid.xyz"
    REST_TESTNET_URL = "https://api.hyperliquid-testnet.xyz"
    WS_URL = "wss://api.hyperliquid.xyz/ws"
    WS_TESTNET_URL = "wss://api.hyperliquid-testnet.xyz/ws"

    def __init__(self, private_key: str, testnet: bool = True):
        super().__init__("", private_key, testnet)
        self.private_key = private_key
        self.account = Account.from_key(private_key)
        self.address = self.account.address

        self.base_url = self.REST_TESTNET_URL if testnet else self.REST_URL
        self.ws_url = self.WS_TESTNET_URL if testnet else self.WS_URL

        self.session: Optional[aiohttp.ClientSession] = None
        self.ws: Optional[aiohttp.ClientWebSocketResponse] = None

        self.positions: Dict[str, Position] = {}
        self.orders: Dict[str, Order] = {}
        self.meta: Dict = {}

        self._msg_loop_task: Optional[asyncio.Task] = None
        self._balance_task: Optional[asyncio.Task] = None
        self._time_offset_ms: int = 0
        self.balance_callback: Optional[Callable] = None

    @property
    def name(self) -> str:
        return "hyperliquid"

    async def _sync_server_time(self):
        url = f"{self.base_url}/info"
        try:
            async with self.session.post(url, json={"type": "meta"}) as resp:
                server_time = int(resp.headers.get("Date-Ms", 0))
                if server_time == 0:
                    from email.utils import parsedate_to_datetime

                    date_str = resp.headers.get("Date", "")
                    if date_str:
                        dt = parsedate_to_datetime(date_str)
                        server_time = int(dt.timestamp() * 1000)
                if server_time > 0:
                    local_time = int(time.time() * 1000)
                    self._time_offset_ms = server_time - local_time
                    logger.info(
                        f"Hyperliquid time offset: {self._time_offset_ms}ms (server ahead)"
                        if self._time_offset_ms > 0
                        else f"Hyperliquid time offset: {self._time_offset_ms}ms (local ahead)"
                    )
        except Exception as e:
            logger.warning(f"Failed to sync Hyperliquid server time: {e}")

    def _get_timestamp(self) -> int:
        return int(time.time() * 1000) + self._time_offset_ms

    async def connect(self):
        logger.info(
            f"Connecting to Hyperliquid ({'Testnet' if self.testnet else 'Mainnet'})..."
        )
        self.session = aiohttp.ClientSession()

        await self._sync_server_time()

        await self._fetch_meta()

        self.ws = await self.session.ws_connect(self.ws_url)
        self.connected = True
        self._msg_loop_task = asyncio.create_task(self._msg_loop())

        await self._subscribe_user_events()
        self._balance_task = asyncio.create_task(self._balance_loop())
        logger.info("Hyperliquid Adapter connected.")

    async def disconnect(self):
        self.connected = False
        if self._msg_loop_task:
            self._msg_loop_task.cancel()
        if self._balance_task:
            self._balance_task.cancel()
        if self.ws:
            await self.ws.close()
        if self.session:
            await self.session.close()
        logger.info("Hyperliquid Adapter disconnected.")

    async def _fetch_meta(self):
        url = f"{self.base_url}/info"
        async with self.session.post(
            url, data=self._fast_json_encode({"type": "meta"})
        ) as resp:
            # Hyperliquid returns JSON, but we use our fast decoder on text/bytes
            self.meta = self._fast_json_decode(await resp.read())

    def _get_asset_index(self, symbol: str) -> int:
        universe = self.meta.get("universe", [])
        for i, asset in enumerate(universe):
            if asset.get("name") == symbol:
                return i
        return 0

    async def _subscribe_user_events(self):
        sub_msg = {
            "method": "subscribe",
            "subscription": {"type": "userEvents", "user": self.address},
        }
        await self.ws.send_str(self._fast_json_encode(sub_msg))

    async def _msg_loop(self):
        while self.connected and self.ws:
            try:
                # Use receive() to get raw message, then decode fast
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
                logger.error(f"Error in Hyperliquid msg loop: {e}")

    async def _handle_message(self, msg: Dict):
        channel = msg.get("channel")
        if channel == "userEvents":
            data = msg.get("data", {})
            if "fills" in data:
                for fill in data["fills"]:
                    await self._handle_fill(fill)
            if "orders" in data:
                for order_update in data["orders"]:
                    await self._handle_order_update(order_update)

    async def _handle_fill(self, data: Dict):
        if self.trade_callback:
            trade = Trade(
                id=str(data.get("tid", "")),
                order_id=str(data.get("oid", "")),
                symbol=data.get("coin", ""),
                side=OrderSide.BUY
                if data.get("dir") == "Open Long" or data.get("dir") == "Close Short"
                else OrderSide.SELL,
                price=float(data.get("px", 0)),
                size=float(data.get("sz", 0)),
                exchange=self.name,
            )
            await self.trade_callback(trade)

    async def _handle_order_update(self, data: Dict):
        exchange_id = str(data.get("oid", ""))
        status_map = {
            "open": OrderStatus.OPEN,
            "filled": OrderStatus.FILLED,
            "canceled": OrderStatus.CANCELLED,
            "rejected": OrderStatus.REJECTED,
        }
        status = status_map.get(data.get("status", "").lower(), OrderStatus.PENDING)

        if exchange_id in self.orders:
            self.orders[exchange_id] = replace(self.orders[exchange_id], status=status)

        if self.order_callback:
            await self.order_callback(
                exchange_id,
                status,
                float(data.get("filledSz", 0)),
                float(data.get("avgPx", 0)),
            )

    def _sign_action(self, action: Dict, nonce: int) -> str:
        typed_data = {
            "types": {
                "EIP712Domain": [
                    {"name": "name", "type": "string"},
                    {"name": "version", "type": "string"},
                    {"name": "chainId", "type": "uint256"},
                ],
                "HyperliquidTransaction:Agent": [
                    {"name": "source", "type": "string"},
                    {"name": "connectionId", "type": "bytes32"},
                ],
            },
            "primaryType": "HyperliquidTransaction:Agent",
            "domain": {
                "name": "Hyperliquid",
                "version": "1",
                "chainId": 421614 if self.testnet else 42161,
            },
            "message": action,
        }

        import json

        action_hash = encode_typed_data(full_message=typed_data)
        signed = self.account.sign_message(action_hash)
        return signed.signature.hex()

    async def place_order(self, order: Order) -> Order:
        mapped_symbol = InstrumentService.get_exchange_symbol(order.symbol, self.name)
        asset_index = self._get_asset_index(mapped_symbol)
        nonce = self._get_timestamp()

        action = {
            "type": "order",
            "orders": [
                {
                    "asset": asset_index,
                    "isBuy": order.side == OrderSide.BUY,
                    "limitPx": str(order.price).rstrip("0").rstrip("."),
                    "sz": str(order.size).rstrip("0").rstrip("."),
                    "reduceOnly": False,
                    "orderType": {"limit": {"tif": "Gtc"}},
                }
            ],
            "grouping": "na",
        }

        signature = self._sign_action(action, nonce)

        payload = {
            "action": action,
            "nonce": nonce,
            "signature": {
                "r": signature[:66],
                "s": "0x" + signature[66:130],
                "v": int(signature[130:], 16),
            },
        }

        url = f"{self.base_url}/exchange"
        async with self.session.post(url, json=payload) as resp:
            data = await resp.json()
            if data.get("status") == "ok":
                resting = (
                    data.get("response", {}).get("data", {}).get("statuses", [{}])[0]
                )
                exchange_id = str(resting.get("resting", {}).get("oid", ""))
                updated = replace(
                    order,
                    exchange_id=exchange_id,
                    status=OrderStatus.OPEN,
                    exchange=self.name,
                )
                self.orders[exchange_id] = updated
                return updated
            else:
                logger.error(f"Hyperliquid order error: {data}")
                return replace(order, status=OrderStatus.REJECTED, exchange=self.name)

    async def cancel_order(self, order_id: str) -> bool:
        order = self.orders.get(order_id)
        if not order:
            return False

        mapped_symbol = InstrumentService.get_exchange_symbol(order.symbol, self.name)
        asset_index = self._get_asset_index(mapped_symbol)
        nonce = self._get_timestamp()

        action = {
            "type": "cancel",
            "cancels": [{"asset": asset_index, "oid": int(order_id)}],
        }
        signature = self._sign_action(action, nonce)

        payload = {
            "action": action,
            "nonce": nonce,
            "signature": {
                "r": signature[:66],
                "s": "0x" + signature[66:130],
                "v": int(signature[130:], 16),
            },
        }

        url = f"{self.base_url}/exchange"
        async with self.session.post(url, json=payload) as resp:
            data = await resp.json()
            if data.get("status") == "ok":
                self.orders[order_id] = replace(
                    self.orders[order_id], status=OrderStatus.CANCELLED
                )
                return True
            logger.error(f"Hyperliquid cancel error: {data}")
            return False

    async def place_orders_batch(self, orders: List[Order]) -> List[Order]:
        return [await self.place_order(o) for o in orders]

    async def cancel_orders_batch(self, order_ids: List[str]) -> List[bool]:
        return [await self.cancel_order(oid) for oid in order_ids]

    async def subscribe_ticker(self, symbol: str):
        asyncio.create_task(self._ticker_stream(symbol))

    async def _ticker_stream(self, symbol: str):
        mapped_symbol = InstrumentService.get_exchange_symbol(symbol, self.name)
        sub_msg = {
            "method": "subscribe",
            "subscription": {"type": "l2Book", "coin": mapped_symbol},
        }
        await self.ws.send_str(self._fast_json_encode(sub_msg))

    async def get_balances(self) -> List[Balance]:
        url = f"{self.base_url}/info"
        async with self.session.post(
            url,
            data=self._fast_json_encode({"type": "userState", "user": self.address}),
        ) as resp:
            data = self._fast_json_decode(await resp.read())
            margin_summary = data.get("marginSummary", {})
            account_value = float(margin_summary.get("accountValue", 0.0))
            total_margin_used = float(margin_summary.get("totalMarginUsed", 0.0))
            withdrawable = float(data.get("withdrawable", 0.0))

            from ...domain.entities import Balance

            bal = Balance(
                exchange=self.name,
                asset="USDC",
                total=account_value,
                available=withdrawable,
                margin_used=total_margin_used,
                equity=account_value,
            )
            return [bal]

    async def get_position(self, symbol: str) -> Position:
        url = f"{self.base_url}/info"
        async with self.session.post(
            url,
            data=self._fast_json_encode({"type": "userState", "user": self.address}),
        ) as resp:
            data = self._fast_json_decode(await resp.read())
            for pos in data.get("assetPositions", []):
                p = pos.get("position", {})
                if p.get("coin") == symbol:
                    amount = float(p.get("szi", 0))
                    entry_price = float(p.get("entryPx", 0))
                    return Position(symbol, amount, entry_price, exchange=self.name)
        return Position(symbol, 0.0, 0.0, exchange=self.name)

    async def _balance_loop(self):
        while self.connected:
            try:
                url = f"{self.base_url}/info"
                async with self.session.post(
                    url,
                    data=self._fast_json_encode(
                        {"type": "userState", "user": self.address}
                    ),
                ) as resp:
                    data = self._fast_json_decode(await resp.read())
                    margin_summary = data.get("marginSummary", {})
                    account_value = float(margin_summary.get("accountValue", 0.0))
                    total_margin_used = float(
                        margin_summary.get("totalMarginUsed", 0.0)
                    )
                    total_ncn = float(margin_summary.get("totalNtlPos", 0.0))
                    withdrawable = float(data.get("withdrawable", 0.0))

                    from ...domain.entities import Balance

                    bal = Balance(
                        exchange=self.name,
                        asset="USDC",  # Hyperliquid uses USDC
                        total=account_value,
                        available=withdrawable,
                        margin_used=total_margin_used,
                        equity=account_value,
                    )

                    if self.balance_callback:
                        await self.balance_callback(bal)

            except Exception as e:
                logger.error(f"Hyperliquid balance polling error: {e}")

            await asyncio.sleep(5)  # Poll every 5 seconds

    def set_balance_callback(self, callback: Callable):
        self.balance_callback = callback
