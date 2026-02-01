import asyncio
import logging
import json
import time
import os
from typing import Dict, Optional, Callable, List, Any
from dataclasses import replace

# Native Thalex client
try:
    from thalex.thalex import Thalex, Network, OrderType as ThOrderType
except ImportError:
    # Fail fast if lib missing
    raise ImportError("Thalex package not found. Ensure 'thalex' is in PYTHONPATH.")

from ...domain.interfaces import ExchangeGateway
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


class TokenBucket:
    """
    Token Bucket algorithm for rate limiting.
    Replenishes tokens at a fixed rate up to a maximum capacity.
    """

    def __init__(self, capacity: int, fill_rate: float):
        self.capacity = float(capacity)
        self._tokens = float(capacity)
        self.fill_rate = fill_rate  # tokens per second
        self.last_update = time.time()

    def consume(self, tokens: int = 1) -> bool:
        now = time.time()
        # Replenish
        delta = now - self.last_update
        self._tokens = min(self.capacity, self._tokens + delta * self.fill_rate)
        self.last_update = now

        if self._tokens >= tokens:
            self._tokens -= tokens
            return True
        return False


class ThalexAdapter(ExchangeGateway):
    """
    Adapter for the Thalex exchange using the official python library.
    Implements robust connection handling, heartbeat monitoring, and RPC correlation.
    """

    def __init__(self, api_key: str, api_secret: str, testnet: bool = True):
        self.api_key = api_key
        self.api_secret = api_secret
        self.network = Network.TEST if testnet else Network.PROD
        self.client = Thalex(network=self.network)

        self.connected = False
        self.ticker_callback: Optional[Callable] = None
        self.trade_callback: Optional[Callable] = None

        # Local state cache
        self.positions: Dict[str, Position] = {}
        self.orders: Dict[str, Order] = {}

        # Async Request Management
        self.pending_requests: Dict[int, asyncio.Future] = {}
        self.request_id_counter = int(time.time() * 1000)

        self.msg_loop_task: Optional[asyncio.Task] = None
        self.heartbeat_task: Optional[asyncio.Task] = None
        self.last_msg_time = time.time()
        self.tick_size: float = 1.0  # Default, will be updated from API

        # Rate Limiters
        # Matching Engine: 50 req/s (enabled via Cancel on Disconnect). Target 45 (90%)
        self.me_rate_limiter = TokenBucket(capacity=50, fill_rate=45.0)

        # Cancel Requests: 1000 req/s. Target 900 (90%)
        self.cancel_rate_limiter = TokenBucket(capacity=1000, fill_rate=900.0)

    def _get_next_id(self) -> int:
        self.request_id_counter += 1
        return self.request_id_counter

    async def _rpc_request(self, func, **kwargs) -> Dict:
        """Helper to send request and await response based on ID correlation"""
        req_id = self._get_next_id()
        future = asyncio.Future()
        self.pending_requests[req_id] = future

        try:
            # Call the client function (buy, sell, cancel, etc) passing 'id'
            # Note: The client methods must be awaited as they are async
            await func(id=req_id, **kwargs)

            # Wait for response
            response = await asyncio.wait_for(future, timeout=5.0)
            return response
        except asyncio.TimeoutError:
            logger.warning(f"RPC Timeout for request {req_id}")
            if req_id in self.pending_requests:
                del self.pending_requests[req_id]
            raise TimeoutError(f"RPC Timeout for request {req_id}")
        except Exception as e:
            if req_id in self.pending_requests:
                del self.pending_requests[req_id]
            raise e

    async def connect(self):
        logger.info("Connecting to Thalex...")
        # Use initialize() to start batcher and msg processing loop
        await self.client.initialize()

        if not self.client.connected():
            raise ConnectionError("Failed to connect to Thalex")

        # Start processing loops BEFORE login to receive the login response
        self.connected = True
        self.msg_loop_task = asyncio.create_task(self._msg_loop())
        self.heartbeat_task = asyncio.create_task(self._heartbeat_monitor())

        logger.info("Logging in...")

        try:
            # Proper RPC Login verification
            login_id = self._get_next_id()
            future = asyncio.Future()
            self.pending_requests[login_id] = future

            await self.client.login(self.api_key, self.api_secret, id=login_id)

            # This should now work with the thalex.py library fix!
            await asyncio.wait_for(future, timeout=10.0)
            logger.info("Logged in successfully.")

            # Enable Cancel On Disconnect to get higher rate limits (50 req/s vs 10 req/s)
            cod_id = self._get_next_id()
            future_cod = asyncio.Future()
            self.pending_requests[cod_id] = future_cod

            # Note: The library expects 'timeout_secs' to enable cancel on disconnect
            await self._rpc_request(
                self.client.set_cancel_on_disconnect, timeout_secs=15
            )
            logger.info("Enabled Cancel On Disconnect (Rate Limit boosted).")

            # Fetch instrument details (tick size)
            await self._fetch_instrument_details()
        except Exception as e:
            logger.error(f"Login/Setup failed: {e}")
            # Cleanup
            self.connected = False
            if self.msg_loop_task:
                self.msg_loop_task.cancel()
            if self.heartbeat_task:
                self.heartbeat_task.cancel()
            await self.client.disconnect()
            raise ConnectionError("Failed to login/setup Thalex")

        logger.info("Thalex Adapter connected and listening.")

    async def disconnect(self):
        if self.msg_loop_task:
            self.msg_loop_task.cancel()
        if self.heartbeat_task:
            self.heartbeat_task.cancel()

        self.connected = False
        await self.client.disconnect()
        logger.info("Thalex Adapter disconnected.")

    async def _fetch_instrument_details(self):
        """Fetch instrument details like tick_size from the API"""
        try:
            # We use a dummy symbol or the primary instrument from env
            symbol = os.getenv("PRIMARY_INSTRUMENT", "BTC-PERPETUAL")
            response = await self._rpc_request(
                self.client.instrument, instrument_name=symbol
            )
            if "result" in response:
                res = response["result"]
                # Assuming 'tick_size' exists in the instrument result
                self.tick_size = float(res.get("tick_size", 0.5))
                logger.info(f"Fetched tick size for {symbol}: {self.tick_size}")
        except Exception as e:
            logger.error(f"Failed to fetch instrument details: {e}")

    async def place_order(self, order: Order) -> Order:
        if not self.connected:
            raise ConnectionError("Not connected to exchange")

        # Rate Limit Check (Matching Engine)
        if not self.me_rate_limiter.consume(1):
            logger.warning("Rate limit hit (ME). Dropping order.")
            return replace(order, status=OrderStatus.REJECTED)

        th_type = (
            ThOrderType.LIMIT if order.type == OrderType.LIMIT else ThOrderType.MARKET
        )

        try:
            # Label must be a string
            label = str(order.id)

            # Select function
            func = self.client.buy if order.side == OrderSide.BUY else self.client.sell

            # Use RPC wrapper
            response = await self._rpc_request(
                func,
                instrument_name=order.symbol,
                amount=order.size,
                price=order.price,
                order_type=th_type,
                label=label,
                post_only=order.post_only,
            )

            if "error" in response:
                logger.error(f"Order error: {response['error']}")
                return replace(order, status=OrderStatus.REJECTED)

            result = response.get("result", {})
            exchange_id = str(result.get("order_id", result.get("id", "")))

            status = OrderStatus.OPEN

            updated_order = replace(order, exchange_id=exchange_id, status=status)

            # Cache order
            self.orders[exchange_id] = updated_order
            return updated_order

        except Exception as e:
            logger.error(f"Failed to place order {order}: {e}")
            return replace(order, status=OrderStatus.REJECTED)

    async def cancel_order(self, order_id: str) -> bool:
        if not self.connected:
            return False

        # Rate Limit Check (Cancel)
        if not self.cancel_rate_limiter.consume(1):
            logger.warning("Rate limit hit (Cancel). Dropping cancel request.")
            return False

        try:
            # Try to convert to int if possible
            oid = int(order_id) if str(order_id).isdigit() else order_id

            response = await self._rpc_request(self.client.cancel, order_id=oid)

            if "error" in response:
                logger.error(f"Cancel error: {response['error']}")
                return False

            if order_id in self.orders:
                self.orders[order_id] = replace(
                    self.orders[order_id], status=OrderStatus.CANCELLED
                )
            return True
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return False

    async def subscribe_ticker(self, symbol: str):
        channel = f"ticker.{symbol}.raw"
        trade_channel = f"trades.{symbol}"
        orders_channel = "orders"  # Account orders channel

        logger.info(f"Subscribing to {channel}, {trade_channel}, and {orders_channel}")
        try:
            # Note: orders is technically a private channel but 'public_subscribe' often handles both if session is auth'd on some exchanges.
            # On Thalex, it's 'private/subscribe' for account-related things if following strict RPC.
            # But the adapter has been using public_subscribe for ticker/trades.
            # Let's check if we should use private/subscribe.
            await self._rpc_request(
                self.client.public_subscribe, channels=[channel, trade_channel]
            )
            # Try private subscribe for orders and portfolio
            await self._rpc_request(
                self.client.private_subscribe, channels=[orders_channel, "portfolio"]
            )
        except Exception as e:
            logger.error(f"Subscription failed: {e}")
        try:
            # Verification for sub is nice too
            await self._rpc_request(
                self.client.public_subscribe, channels=[channel, trade_channel]
            )
        except Exception as e:
            logger.error(f"Subscription failed: {e}")

    async def get_position(self, symbol: str) -> Position:
        return self.positions.get(symbol, Position(symbol, 0.0, 0.0))

    def set_ticker_callback(self, callback):
        self.ticker_callback = callback

    def set_trade_callback(self, callback):
        self.trade_callback = callback

    async def _msg_loop(self):
        logger.debug("Starting adapter message loop...")  # Debug level
        while self.connected:
            try:
                # Add timeout to loop to allow heartbeat check
                msg = await asyncio.wait_for(self.client.receive(), timeout=1.0)
                self.last_msg_time = time.time()

                if not msg:
                    continue

                if isinstance(msg, str):
                    try:
                        msg = json.loads(msg)
                    except json.JSONDecodeError:
                        logger.warning(f"Received invalid JSON: {msg}")
                        continue

                if not isinstance(msg, dict):
                    # Handle Batch Responses (List of Dicts)
                    if isinstance(msg, list):
                        for m in msg:
                            if isinstance(m, dict):
                                # recursive process? or just inline
                                await self._handle_single_msg(m)
                        continue

                    logger.warning(f"Received non-dict msg: {msg}")
                    continue

                await self._handle_single_msg(msg)

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error in msg loop: {e}")
                await asyncio.sleep(1)

    async def _handle_single_msg(self, msg: Dict):
        # Check for RPC response
        msg_id = msg.get("id")
        if msg_id is not None and msg_id in self.pending_requests:
            self.pending_requests[msg_id].set_result(msg)
            if msg_id in self.pending_requests:  # check again just in case
                del self.pending_requests[msg_id]
            return

        # Delegate processing
        asyncio.create_task(self._process_message(msg))

    async def _process_message(self, msg: Dict):
        # Handle new format: {'channel_name': ..., 'notification': ...}
        channel = msg.get("channel_name")
        if channel:
            data = msg.get("notification", {})
            if channel.startswith("ticker."):
                part = channel.split(".")
                if len(part) >= 2:
                    symbol = part[1]
                    ticker = self._map_ticker(symbol, data)
                    if self.ticker_callback:
                        await self.ticker_callback(ticker)

            elif channel.startswith("trades."):
                part = channel.split(".")
                if len(part) >= 2:
                    symbol = part[1]
                    # trades data is usually a list of trades
                    trade_data_list = data if isinstance(data, list) else [data]
                    for t_data in trade_data_list:
                        trade = self._map_trade(symbol, t_data)
                        if self.trade_callback:
                            await self.trade_callback(trade)
            elif channel == "orders":
                # Handle order updates (fills/cancellations)
                order_data_list = data if isinstance(data, list) else [data]
                for o_data in order_data_list:
                    exchange_id = str(o_data.get("order_id", o_data.get("id", "")))
                    if not exchange_id:
                        continue

                    status_str = o_data.get("status", "").lower()
                    status = OrderStatus.OPEN
                    if status_str in ["cancelled", "canceled"]:
                        status = OrderStatus.CANCELLED
                    elif status_str in ["filled", "closed"]:
                        status = OrderStatus.FILLED
                    elif status_str in ["rejected"]:
                        status = OrderStatus.REJECTED

                    if exchange_id in self.orders:
                        self.orders[exchange_id] = replace(
                            self.orders[exchange_id],
                            status=status,
                            filled_size=float(
                                o_data.get(
                                    "amount_filled",
                                    self.orders[exchange_id].filled_size,
                                )
                            ),
                        )
                        logger.info(
                            f"Updated order {exchange_id} status to {status} from notification"
                        )
            elif channel == "portfolio":
                # Handle portfolio/position updates
                positions_data = data if isinstance(data, list) else [data]
                for p_data in positions_data:
                    symbol = p_data.get("instrument_name")
                    if symbol:
                        amount = float(p_data.get("amount", 0.0))
                        entry_price = float(p_data.get("average_price", 0.0))
                        self.positions[symbol] = Position(symbol, amount, entry_price)
                        logger.debug(f"Position update for {symbol}: {amount}")
            return

        # Fallback to old format (if any)
        method = msg.get("method")
        if method == "subscription":
            params = msg.get("params", {})
            channel = params.get("channel", "")
            data = params.get("data", {})

            if channel.startswith("ticker."):
                part = channel.split(".")
                if len(part) >= 2:
                    symbol = part[1]
                    ticker = self._map_ticker(symbol, data)
                    if self.ticker_callback:
                        await self.ticker_callback(ticker)

            elif channel.startswith("trades."):
                part = channel.split(".")
                if len(part) >= 2:
                    symbol = part[1]
                    # trades data is usually a list of trades
                    trade_data_list = (
                        data
                        if isinstance(data, list)
                        else (
                            data.get("trades", []) if isinstance(data, dict) else [data]
                        )
                    )
                    for t_data in trade_data_list:
                        trade = self._map_trade(symbol, t_data)
                        if self.trade_callback:
                            await self.trade_callback(trade)

    async def _heartbeat_monitor(self):
        while self.connected:
            await asyncio.sleep(5)
            if time.time() - self.last_msg_time > 60:
                logger.warning("No messages for 60s. Potential stale connection.")

    def _map_ticker(self, symbol: str, data: Dict) -> Ticker:
        ts = float(data.get("timestamp", time.time()))
        if ts > 1e11:
            ts /= 1000.0

        return Ticker(
            symbol=symbol,
            bid=float(data.get("best_bid_price", 0)),
            ask=float(data.get("best_ask_price", 0)),
            bid_size=float(data.get("best_bid_amount", 0)),
            ask_size=float(data.get("best_ask_amount", 0)),
            last=float(data.get("last_price", 0)),
            volume=float(data.get("volume_24h", 0)),
            timestamp=ts,
        )

    def _map_trade(self, symbol: str, data: Dict) -> Trade:
        ts = float(data.get("timestamp", time.time()))
        if ts > 1e11:
            ts /= 1000.0

        return Trade(
            id=str(data.get("id", "")),
            order_id="",
            symbol=symbol,
            side=OrderSide.BUY
            if data.get("direction", data.get("side")) == "buy"
            else OrderSide.SELL,
            price=float(data.get("price", 0)),
            size=float(data.get("amount", 0)),
            timestamp=ts,
        )
