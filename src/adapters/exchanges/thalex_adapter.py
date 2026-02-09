import asyncio
import logging
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

from ...domain.interfaces import ExchangeGateway, TimeSyncManager
from .base_adapter import TokenBucket, BaseExchangeAdapter
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


class ThalexAdapter(BaseExchangeAdapter):
    """
    Adapter for the Thalex exchange using the official python library.
    Implements robust connection handling, heartbeat monitoring, and RPC correlation.
    """

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        testnet: bool = True,
        time_sync_manager: Optional[TimeSyncManager] = None,
        me_rate_limit: float = 45.0,
        cancel_rate_limit: float = 900.0,
    ):
        super().__init__(api_key, api_secret, testnet, time_sync_manager)

        self.positions: Dict[str, Position] = {}
        self.orders: Dict[str, Order] = {}
        self._order_timestamps: Dict[str, float] = {}

        self.pending_requests: Dict[int, asyncio.Future] = {}
        self.request_id_counter = int(time.time() * 1000)

        self.msg_loop_task: Optional[asyncio.Task] = None
        self.heartbeat_task: Optional[asyncio.Task] = None
        self._prune_task: Optional[asyncio.Task] = None
        self.last_msg_time = time.time()
        self.tick_size: float = 1.0

        self.last_sequence: int = 0

        self.sequence_callback: Optional[Callable] = None

        # Use optimized token buckets
        self.me_rate_limiter = TokenBucket(
            capacity=int(me_rate_limit * 1.1), fill_rate=me_rate_limit
        )
        self.cancel_rate_limiter = TokenBucket(
            capacity=int(cancel_rate_limit * 1.1), fill_rate=cancel_rate_limit
        )

        network = Network.TEST if testnet else Network.PROD
        self.client = Thalex(network)

        # Flag to indicate if the adapter is currently reconnecting
        self.is_reconnecting = False

        # Pre-allocated buffers for serialization to reduce allocations
        self._serialization_buffer = bytearray(4096)  # Pre-allocated buffer

    @property
    def name(self) -> str:
        return "thalex"

    async def get_server_time(self) -> int:
        """Fetch current Thalex server time. Using local time as fallback."""
        return int(time.time() * 1000)

    def _get_timestamp(self) -> int:
        return super()._get_timestamp()

    def _get_next_id(self) -> int:
        self.request_id_counter += 1
        return self.request_id_counter

    def _fast_json_encode_optimized(self, data: Any) -> str:
        """
        Ultra-fast JSON encoding using orjson with pre-allocated buffers where possible.
        This is a key optimization for reducing serialization overhead.
        """
        try:
            # Use orjson for fastest serialization
            import orjson
            serialized_bytes = orjson.dumps(data, option=orjson.OPT_SERIALIZE_NUMPY)
            # Decode to string - this is still faster than standard json in most cases
            return serialized_bytes.decode("utf-8")
        except ImportError:
            # Fallback to standard json if orjson not available
            import json
            return json.dumps(data)

    async def _rpc_request(self, func, **kwargs) -> Dict:
        """Optimized helper to send request and await response based on ID correlation"""
        req_id = self._get_next_id()
        future = asyncio.Future()
        self.pending_requests[req_id] = future

        # Determine timeout based on request type
        # Subscription requests may take longer
        method_name = func.__name__.lower()
        if 'subscribe' in method_name:
            timeout_val = 30.0  # Increased timeout for subscriptions
        else:
            timeout_val = 15.0  # Increased timeout for other requests

        try:
            # Check connection status before sending request
            if not self.client.connected():
                raise ConnectionError("Client not connected")
                
            # Call the client function (buy, sell, cancel, etc) passing 'id'
            # Note: The client methods must be awaited as they are async
            await func(id=req_id, **kwargs)

            # Wait for response with adaptive timeout
            response = await asyncio.wait_for(future, timeout=timeout_val)
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
        self._ping_task = asyncio.create_task(self._ping_loop())

        logger.info("Logging in...")

        try:
            # Proper RPC Login verification
            login_id = self._get_next_id()
            future = asyncio.Future()
            self.pending_requests[login_id] = future

            await self.client.login(self.api_key, self.api_secret, id=login_id)

            # This should now work with the thalex.py library fix!
            await asyncio.wait_for(future, timeout=20.0)  # Increased from 15.0
            logger.info("Logged in successfully.")

            # Enable Cancel On Disconnect to get higher rate limits (50 req/s vs 10 req/s)
            # Note: The library expects 'timeout_secs' to enable cancel on disconnect
            await self._rpc_request(
                self.client.set_cancel_on_disconnect, timeout_secs=15
            )
            logger.info("Enabled Cancel On Disconnect (Rate Limit boosted).")

            # Fetch instrument details (tick size)
            await self.fetch_instrument_info(
                os.getenv("PRIMARY_INSTRUMENT", "BTC-PERPETUAL")
            )
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

        self._prune_task = asyncio.create_task(self._prune_order_cache())
        logger.info("Thalex Adapter connected and listening.")

    def set_order_callback(self, callback: Callable):
        self.order_callback = callback

    def set_position_callback(self, callback: Callable):
        self.position_callback = callback

    def set_balance_callback(self, callback: Callable):
        self.balance_callback = callback

    def set_sequence_callback(self, callback: Callable):
        self.sequence_callback = callback

    async def disconnect(self):
        if self.msg_loop_task:
            self.msg_loop_task.cancel()
        if self.heartbeat_task:
            self.heartbeat_task.cancel()
        if hasattr(self, "_ping_task") and self._ping_task:
            self._ping_task.cancel()
        if self._prune_task:
            self._prune_task.cancel()

        self.connected = False
        await self.client.disconnect()
        logger.info("Thalex Adapter disconnected.")

    async def fetch_instrument_info(self, symbol: str):
        """Fetch instrument details like tick_size from the API"""
        try:
            response = await self._rpc_request(
                self.client.instrument, instrument_name=symbol
            )
            if "result" in response:
                res = response["result"]
                # Assuming 'tick_size' exists in the instrument result
                self.tick_size = self._safe_float(res.get("tick_size", 0.5))
                logger.info(f"Fetched tick size for {symbol}: {self.tick_size}")
        except Exception as e:
            logger.error(f"Failed to fetch instrument details: {e}")

    async def get_balances(self) -> List[Balance]:
        try:
            # Try to use get_account_summary or fall back to requesting subscription update
            # Since Thalex lib might not expose get_account_summary directly if not defined,
            # check availability. Assuming it exists or we can rely on msg stream updates.
            # But let's try calling it.
            if hasattr(self.client, "account_summary"):
                response = await self._rpc_request(self.client.account_summary)
                if "result" in response:
                    res = response["result"]
                    total = self._safe_float(
                        res.get("equity")
                        or res.get("margin")
                        or res.get("cash_collateral"),
                        0.0,
                    )
                    available = self._safe_float(
                        res.get("available_funds") or res.get("remaining_margin"), 0.0
                    )
                    margin_used = self._safe_float(
                        res.get("margin_used") or res.get("required_margin"), 0.0
                    )

                    bal = Balance(
                        exchange=self.name,
                        asset="USD",
                        total=total,
                        available=available,
                        margin_used=margin_used,
                        equity=total,
                    )
                    if self.balance_callback:
                        await self.balance_callback(bal)
                    return [bal]
        except Exception as e:
            logger.error(f"Failed to fetch balances: {e}")
        return []

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

            self.orders[exchange_id] = updated_order
            self._order_timestamps[exchange_id] = time.time()
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
                err_msg = response["error"].get("message", "").lower()
                if "order not found" in err_msg or "doesn't exist" in err_msg:
                    logger.warning(
                        f"Order {order_id} not found, treating as cancelled."
                    )
                    # Treat as success to stop retrying
                else:
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
        # Ensure we're properly connected before subscribing
        if not self.connected or not self.client.connected():
            logger.warning(f"Cannot subscribe to {symbol}: Not connected.")
            return

        channel = f"ticker.{symbol}.raw"
        trade_channel = f"trades.{symbol}"
        orders_channel = "orders"  # Account orders channel

        logger.info(f"Subscribing to {channel}, {trade_channel}, and {orders_channel}")

        # Subscribe to public channels first with retry
        try:
            await self._rpc_request_with_retry(
                self.client.public_subscribe, channels=[channel, trade_channel], max_retries=3
            )
            logger.info(f"Successfully subscribed to public channels: {channel}, {trade_channel}")
        except Exception as e:
            logger.error(f"Public subscription failed after retries: {e}")

        # Subscribe to private channels with retry
        try:
            await self._rpc_request_with_retry(
                self.client.private_subscribe,
                channels=[orders_channel, "portfolio", "account"],
                max_retries=3
            )
            logger.info("Successfully subscribed to private channels: orders, portfolio, account")
        except Exception as e:
            logger.error(f"Private subscription failed after retries: {e}")

    async def _rpc_request_with_retry(self, func, max_retries=3, **kwargs):
        """Wrapper for _rpc_request with retry logic"""
        last_error = None
        
        for attempt in range(max_retries):
            try:
                return await self._rpc_request(func, **kwargs)
            except (TimeoutError, ConnectionError) as e:
                last_error = e
                logger.warning(f"RPC request failed (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:  # Don't sleep after the last attempt
                    await asyncio.sleep(0.5 * (2 ** attempt))  # Exponential backoff
                continue
            except Exception as e:
                # For other exceptions, don't retry
                raise e
        
        # If all retries failed, raise the last error
        raise last_error

    async def _send_batch(self, requests: List[Dict]) -> List[Any]:
        """
        Send a batch of JSON-RPC requests in a single WebSocket frame.
        Optimized for performance with reduced allocations and faster serialization.
        """
        if not requests:
            return []

        # Check if the adapter is currently reconnecting
        if self.is_reconnecting:
            logger.warning(f"Cannot send batch of {len(requests)} requests: Adapter is reconnecting.")
            return [ConnectionError("Adapter is reconnecting")] * len(requests)

        # Ensure we're properly connected and authenticated
        if not self.connected or not self.client.connected():
            logger.warning(f"Cannot send batch of {len(requests)} requests: Not connected.")
            return [ConnectionError("Not connected")] * len(requests)

        futures = []
        batch_payload = []

        for req in requests:
            method = req.get("method")
            params = req.get("params", {})

            req_id = self._get_next_id()
            future = asyncio.Future()
            self.pending_requests[req_id] = future
            futures.append(future)

            # Construct JSON-RPC 2.0 request object
            payload = {
                "jsonrpc": "2.0",
                "method": method,
                "params": params,
                "id": req_id,
            }
            batch_payload.append(payload)

        try:
            # Bypass thalex library's wrapper and use websocket directly
            if not self.client.connected():
                raise ConnectionError("Not connected")

            # Send requests individually (Pipelining) with optimized serialization
            # Thalex might not support JSON-RPC Batch Arrays. Pipelining achieves similar throughput.
            for payload in batch_payload:
                # Use optimized serialization
                req_str = self._fast_json_encode_optimized(payload)
                logger.debug(f"TX PIPE: {req_str}")
                await self.client.ws.send(req_str)

            # Wait with Timeout - increased timeout for batch operations
            results = await asyncio.wait_for(
                asyncio.gather(*futures, return_exceptions=True), timeout=35.0  # Increased from 25.0
            )
            return results

        except asyncio.TimeoutError:
            logger.error("Batch send timed out")
            # Properly clean up pending requests on timeout
            for req_id, future in zip([payload["id"] for payload in batch_payload], futures):
                if req_id in self.pending_requests:
                    del self.pending_requests[req_id]
                if not future.done():
                    future.cancel()
            return [TimeoutError("Batch Timeout")] * len(requests)

        except Exception as e:
            logger.error(f"Batch send failed: {e}")
            # Properly clean up pending requests on failure
            for req_id, future in zip([payload["id"] for payload in batch_payload], futures):
                if req_id in self.pending_requests:
                    del self.pending_requests[req_id]
                if not future.done():
                    future.set_exception(e)
            return [e] * len(requests)

    async def place_orders_batch(self, orders: List[Order]) -> List[Order]:
        """Place multiple orders in a single batch request with optimizations."""
        if not self.connected:
            logger.warning(f"Cannot place {len(orders)} orders: Not connected.")
            return [replace(o, status=OrderStatus.REJECTED) for o in orders]

        # Check if the adapter is currently reconnecting
        if self.is_reconnecting:
            logger.warning(f"Cannot place {len(orders)} orders: Adapter is reconnecting.")
            return [replace(o, status=OrderStatus.REJECTED) for o in orders]

        # Additional check: ensure client is connected and ready
        if not self.client.connected():
            logger.warning(f"Cannot place {len(orders)} orders: Client not connected.")
            return [replace(o, status=OrderStatus.REJECTED) for o in orders]

        # Optimized rate limiting with early exit
        if not self.me_rate_limiter.consume(len(orders)):
            logger.info(f"Insufficient tokens for {len(orders)} orders. Waiting...")
            await self.me_rate_limiter.consume_wait(len(orders))

        logger.debug(f"Rate limit cleared, placing {len(orders)} orders.")

        requests = []
        for o in orders:
            th_type = (
                ThOrderType.LIMIT if o.type == OrderType.LIMIT else ThOrderType.MARKET
            )
            label = str(o.id)
            method = "private/buy" if o.side == OrderSide.BUY else "private/sell"

            params = {
                "instrument_name": o.symbol,
                "amount": o.size,
                "price": o.price,
                "order_type": th_type.value,
                "label": label,
                "post_only": o.post_only,
            }
            requests.append({"method": method, "params": params})

        logger.debug(f"Sending batch request with {len(requests)} orders")
        results = await self._send_batch(requests)
        logger.debug(f"Received {len(results)} results from batch send.")

        updated_orders = []
        for order, res in zip(orders, results):
            if isinstance(res, Exception):
                logger.warning(f"Order {order.id} failed with exception: {res}")
                updated_orders.append(replace(order, status=OrderStatus.REJECTED))
            elif isinstance(res, dict) and "error" in res:
                logger.warning(f"Order {order.id} rejected: {res.get('error')}")
                updated_orders.append(replace(order, status=OrderStatus.REJECTED))
            else:
                # Handle successful response
                if isinstance(res, dict):
                    result_data = res.get("result", {})
                    exchange_id = str(
                        result_data.get("order_id", result_data.get("id", ""))
                    )
                    if exchange_id:
                        logger.info(
                            f"Order {order.id} placed successfully as {exchange_id}"
                        )
                        updated_orders.append(
                            replace(order, exchange_id=exchange_id, status=OrderStatus.OPEN)
                        )
                        self.orders[exchange_id] = updated_orders[-1]
                    else:
                        logger.error(
                            f"Order {order.id} response missing exchange_id: {res}"
                        )
                        updated_orders.append(replace(order, status=OrderStatus.REJECTED))
                else:
                    # Response is not a dict and not an exception, treat as failure
                    logger.warning(f"Order {order.id} unexpected response type: {type(res)}")
                    updated_orders.append(replace(order, status=OrderStatus.REJECTED))

        # Log the outcome of the batch operation
        successful_orders = sum(1 for order in updated_orders if order.status != OrderStatus.REJECTED)
        logger.info(f"Batch order placement: {successful_orders}/{len(orders)} orders successful")

        return updated_orders

    async def cancel_orders_batch(self, order_ids: List[str]) -> List[bool]:
        """Cancel multiple orders in a single batch with optimizations."""
        if not self.connected or not order_ids:
            return [False] * len(order_ids)

        # Check if the adapter is currently reconnecting
        if self.is_reconnecting:
            logger.warning(f"Cannot cancel {len(order_ids)} orders: Adapter is reconnecting.")
            return [False] * len(order_ids)

        # Additional check: ensure client is connected and ready
        if not self.client.connected():
            logger.warning(f"Cannot cancel {len(order_ids)} orders: Client not connected.")
            return [False] * len(order_ids)

        # Optimized rate limiting with early exit
        if not self.cancel_rate_limiter.consume(len(order_ids)):
            logger.warning(f"Insufficient tokens for {len(order_ids)} cancels. Waiting...")
            await self.cancel_rate_limiter.consume_wait(len(order_ids))

        requests = []
        for oid in order_ids:
            # clean oid
            final_oid = int(oid) if str(oid).isdigit() else oid
            requests.append(
                {"method": "private/cancel", "params": {"order_id": final_oid}}
            )

        results = await self._send_batch(requests)

        outcomes = []
        for oid_str, res in zip(order_ids, results):
            success = False
            if not isinstance(res, Exception) and "error" not in res:
                success = True
            elif isinstance(res, dict) and "error" in res:
                err_msg = res["error"].get("message", "").lower()
                if "order not found" in err_msg or "doesn't exist" in err_msg:
                    # Treat as success/already cancelled
                    success = True
                    logger.debug(
                        f"Batch cancel: Order {oid_str} not found, marking cancelled."
                    )

            if success:
                if oid_str in self.orders:
                    self.orders[oid_str] = replace(
                        self.orders[oid_str], status=OrderStatus.CANCELLED
                    )
            outcomes.append(success)
        return outcomes

    async def cancel_all_orders(self, symbol: str = None) -> bool:
        if not self.connected:
            return False

        try:
            result = await self._rpc_request(self.client.cancel_all)
            self.orders.clear()
            logger.info(f"Cancelled all orders. Result: {result}")
            return True
        except Exception as e:
            logger.error(f"Failed to cancel all orders: {e}")
            return False

    async def get_open_orders(self, symbol: str) -> List[Order]:
        if not self.connected:
            return []

        try:
            # Call open_orders on the client
            response = await self._rpc_request(
                self.client.open_orders, instrument_name=symbol
            )

            if "result" in response:
                orders_data = response["result"]
                mapped_orders = []
                for o_data in orders_data:
                    # Safe float parsing
                    price = self._safe_float(o_data.get("price"))
                    size = self._safe_float(o_data.get("amount"))
                    filled = self._safe_float(o_data.get("amount_filled", 0.0))

                    side = (
                        OrderSide.BUY
                        if o_data.get("direction", "").lower() == "buy"
                        else OrderSide.SELL
                    )
                    exchange_id = str(o_data.get("order_id", o_data.get("id", "")))
                    label = str(o_data.get("label", ""))

                    mapped_orders.append(
                        Order(
                            id=label if label else f"EXT_{exchange_id}",
                            symbol=symbol,
                            side=side,
                            price=price,
                            size=size,
                            filled_size=filled,
                            type=OrderType.LIMIT,
                            status=OrderStatus.OPEN,
                            exchange_id=exchange_id,
                            timestamp=time.time(),
                        )
                    )
                return mapped_orders
            return []
        except Exception as e:
            logger.error(f"Failed to fetch open orders: {e}")
            return []

    async def get_recent_trades(self, symbol: str, limit: int = 100) -> List[Trade]:
        """Fetch recent trades from Thalex REST API."""
        try:
            # Thalex API: public/last_trades
            response = await self._rpc_request(
                self.client.last_trades, instrument_name=symbol, limit=limit
            )

            if "result" in response:
                trades_data = response["result"]
                trades = []
                for t_data in trades_data:
                    trades.append(self._map_trade(symbol, t_data))
                return trades
            return []
        except Exception as e:
            logger.error(f"Failed to fetch recent trades from Thalex: {e}")
            return []

    async def get_position(self, symbol: str) -> Position:
        return self.positions.get(symbol, Position(symbol, 0.0, 0.0))

    def set_ticker_callback(self, callback):
        self.ticker_callback = callback

    def set_trade_callback(self, callback):
        self.trade_callback = callback

    async def _msg_loop(self):
        logger.debug("Starting adapter message loop...")
        while self.connected:
            try:
                msg = await asyncio.wait_for(self.client.receive(), timeout=1.0)
                self.last_msg_time = time.time()

                if not msg:
                    continue

                if isinstance(msg, str):
                    try:
                        msg = self._fast_json_decode(msg)
                    except ValueError:
                        logger.warning(f"Received invalid JSON: {msg}")
                        continue

                if isinstance(msg, list):
                    for m in msg:
                        if isinstance(m, dict):
                            await self._handle_single_msg(m)
                    continue

                if not isinstance(msg, dict):
                    logger.warning(f"Received non-dict msg: {msg}")
                    continue

                # Process message efficiently via handle_single_msg
                await self._handle_single_msg(msg)

            except asyncio.TimeoutError:
                # Normal timeout, continue loop
                # Check if the connection is still alive by checking if the client is connected
                if not self.client.connected():
                    logger.warning("Client reports disconnected, ending message loop")
                    self.connected = False
                    break
                continue
            except Exception as e:
                error_msg = str(e).lower()
                logger.error(f"Message loop error: {e}")
                
                # Check for specific connection-related errors
                if "connection closed" in error_msg or "invalid state" in error_msg or "websocket" in error_msg:
                    logger.info(f"Connection issue detected ({error_msg}), initiating reconnection")
                    # Trigger reconnection by setting connected to False
                    self.connected = False
                    break
                # Brief pause before continuing to prevent tight loop on error
                await asyncio.sleep(0.1)

        logger.warning("Message loop ended (Disconnected). Failing pending requests.")
        self._fail_pending_requests(ConnectionError("Disconnected"))

    async def _attempt_reconnect(self):
        # Set the reconnection flag to prevent other operations during reconnection
        self.is_reconnecting = True

        max_retries = 10
        base_delay = 1.0
        for attempt in range(max_retries):
            delay = min(base_delay * (2**attempt), 60.0)
            logger.warning(
                f"Reconnection attempt {attempt + 1}/{max_retries} in {delay:.1f}s..."
            )
            await asyncio.sleep(delay)

            try:
                # Cancel existing tasks before reconnecting
                if self.msg_loop_task:
                    self.msg_loop_task.cancel()
                if self.heartbeat_task:
                    self.heartbeat_task.cancel()
                if hasattr(self, "_ping_task") and self._ping_task:
                    self._ping_task.cancel()
                    
                # Disconnect and reconnect
                await self.client.disconnect()
                await self.client.initialize()

                if not self.client.connected():
                    continue

                # Restart message loop and heartbeat before login
                self.msg_loop_task = asyncio.create_task(self._msg_loop())
                self.heartbeat_task = asyncio.create_task(self._heartbeat_monitor())
                self._ping_task = asyncio.create_task(self._ping_loop())

                login_id = self._get_next_id()
                future = asyncio.Future()
                self.pending_requests[login_id] = future

                await self.client.login(self.api_key, self.api_secret, id=login_id)
                await asyncio.wait_for(future, timeout=15.0)  # Use same timeout as connect

                await self._rpc_request(
                    self.client.set_cancel_on_disconnect, timeout_secs=15
                )

                # Fetch instrument details again after reconnection
                await self.fetch_instrument_info(
                    os.getenv("PRIMARY_INSTRUMENT", "BTC-PERPETUAL")
                )

                logger.info("Reconnection successful.")
                self.last_msg_time = time.time()

                # Reset the reconnection flag after successful reconnection
                self.is_reconnecting = False
                return

            except Exception as e:
                logger.error(f"Reconnection attempt {attempt + 1} failed: {e}")

        logger.error("All reconnection attempts failed. Giving up.")
        self.connected = False

        # Reset the reconnection flag even if reconnection failed
        self.is_reconnecting = False

    def _fail_pending_requests(self, exc: Exception):
        """Cancel all pending requests with exception"""
        for req_id, future in list(self.pending_requests.items()):
            if not future.done():
                future.set_exception(exc)
        self.pending_requests.clear()

    async def _handle_single_msg(self, msg: Dict):
        # Check for RPC response
        msg_id = msg.get("id")
        if msg_id is not None:
            logger.debug(f"RX RPC: {msg}")
            
            # Try direct lookup first
            if msg_id in self.pending_requests:
                self.pending_requests[msg_id].set_result(msg)
                del self.pending_requests[msg_id]
                return
            
            # Try casting to int if it's a string (or vice versa, but keys are ints)
            try:
                int_id = int(msg_id)
                if int_id in self.pending_requests:
                    self.pending_requests[int_id].set_result(msg)
                    del self.pending_requests[int_id]
                    return
            except (ValueError, TypeError):
                pass
                
            logger.warning(f"Received RPC response with unknown ID: {msg_id}")
            return

        # Delegate processing
        await self._process_message(msg)

    async def _ping_loop(self):
        """Send periodic application-level pings to keep connection healthy."""
        while self.connected:
            try:
                await asyncio.sleep(20)
                if self.connected and self.client.connected():
                    # Thalex uses public/ping or just standard ping frame.
                    # Library has ping() which sends protocol ping.
                    # Let's send a lightweight public request as app-level ping
                    await self._rpc_request(self.client.system_info)
            except Exception as e:
                logger.debug(f"Ping failed: {e}")

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
                    # Update ticker liveness timestamp
                    self.last_ticker_time = time.time()
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
                            asyncio.create_task(self.trade_callback(trade))
            elif channel == "orders":
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

                    filled_size = self._safe_float(o_data.get("amount_filled", 0.0))
                    avg_price = self._safe_float(o_data.get("average_price", 0.0))
                    label = str(o_data.get("label", ""))

                    if exchange_id in self.orders:
                        self.orders[exchange_id] = replace(
                            self.orders[exchange_id],
                            status=status,
                            filled_size=filled_size,
                        )
                        self._order_timestamps[exchange_id] = time.time()
                        logger.info(
                            f"Updated order {exchange_id} (label={label}) status to {status} from notification"
                        )

                    asyncio.create_task(self.notify_order_update(
                        exchange_id, status, filled_size, avg_price, local_id=label
                    ))

            elif channel == "portfolio":
                positions_data = data if isinstance(data, list) else [data]
                for p_data in positions_data:
                    symbol = p_data.get("instrument_name")
                    if symbol:
                        amount = self._safe_float(p_data.get("amount", 0.0))
                        entry_price = self._safe_float(p_data.get("average_price", 0.0))
                        mark_price = self._safe_float(p_data.get("mark_price", 0.0))
                        unrealized_pnl = self._safe_float(
                            p_data.get("floating_profit_loss", 0.0)
                        )
                        delta = self._safe_float(p_data.get("delta", 0.0))
                        gamma = self._safe_float(p_data.get("gamma", 0.0))
                        theta = self._safe_float(p_data.get("theta", 0.0))

                        self.positions[symbol] = Position(
                            symbol=symbol,
                            size=amount,
                            entry_price=entry_price,
                            mark_price=mark_price,
                            unrealized_pnl=unrealized_pnl,
                            delta=delta,
                            gamma=gamma,
                            theta=theta,
                        )
                        logger.debug(
                            f"Position update for {symbol}: {amount} @ {entry_price}"
                        )

                        if self.position_callback:
                            asyncio.create_task(self.position_callback(symbol, amount, entry_price))

            elif channel == "account":
                acc_data = data if isinstance(data, list) else [data]

                for acc in acc_data:
                    # Map fields - adjusting based on assumed Thalex API response for account
                    # total_equity, available_funds, etc.
                    # Fallback if specific fields aren't found, log for debugging
                    total = self._safe_float(
                        acc.get("equity")
                        or acc.get("margin")
                        or acc.get("cash_collateral"),
                        0.0,
                    )
                    available = self._safe_float(
                        acc.get("available_funds") or acc.get("remaining_margin"), 0.0
                    )
                    margin_used = self._safe_float(
                        acc.get("margin_used") or acc.get("required_margin"), 0.0
                    )

                    bal = Balance(
                        exchange=self.name,
                        asset="USD",  # Defaulting to USD/USDC for Thalex usually
                        total=total,
                        available=available,
                        margin_used=margin_used,
                        equity=total,
                    )
                    if self.balance_callback:
                        asyncio.create_task(self.balance_callback(bal))
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
                    # Update ticker liveness timestamp
                    self.last_ticker_time = time.time()
                    if self.ticker_callback:
                        asyncio.create_task(self.ticker_callback(ticker))

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
                            asyncio.create_task(self.trade_callback(trade))

    async def _heartbeat_monitor(self):
        # Use the standardized last_ticker_time from the base class
        self.last_ticker_time = time.time()

        while self.connected:
            await asyncio.sleep(10)  # Check every 10 seconds for more responsive monitoring
            time_since_last_msg = time.time() - self.last_msg_time

            # Check both the internal timer and the client's connection status
            if time_since_last_msg > 60 or not self.client.connected():  # Increase timeout to 60 seconds to avoid premature disconnections
                logger.warning("Heartbeat timeout detected or client disconnected, connection may be dead")
                # Trigger reconnection logic
                self.connected = False
                break

    def _map_ticker(self, symbol: str, data: Dict) -> Ticker:
        ts = self._safe_float(data.get("timestamp", time.time()))
        if ts > 1e11:
            ts /= 1000.0

        return Ticker(
            symbol=symbol,
            bid=self._safe_float(data.get("best_bid_price", 0)),
            ask=self._safe_float(data.get("best_ask_price", 0)),
            bid_size=self._safe_float(data.get("best_bid_amount", 0)),
            ask_size=self._safe_float(data.get("best_ask_amount", 0)),
            last=self._safe_float(data.get("last_price", 0)),
            volume=self._safe_float(data.get("volume_24h", 0)),
            mark_price=self._safe_float(data.get("mark_price", 0)),
            timestamp=ts,
        )

    def _map_trade(self, symbol: str, data: Dict) -> Trade:
        ts = self._safe_float(data.get("timestamp", time.time()))
        if ts > 1e11:
            ts /= 1000.0

        return Trade(
            id=str(data.get("id", "")),
            order_id="",
            symbol=symbol,
            side=OrderSide.BUY
            if data.get("direction", data.get("side")) == "buy"
            else OrderSide.SELL,
            price=self._safe_float(data.get("price", 0)),
            size=self._safe_float(data.get("amount", 0)),
            timestamp=ts,
        )

    async def _prune_order_cache(self):
        """Prune old orders from cache with optimized timing."""
        while self.connected:
            try:
                now = time.time()
                # Prune orders older than 10 minutes that are filled/cancelled
                old_orders = [
                    oid for oid, order in self.orders.items()
                    if (now - self._order_timestamps.get(oid, 0)) > 600
                    and order.status in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED]
                ]

                for oid in old_orders:
                    self.orders.pop(oid, None)
                    self._order_timestamps.pop(oid, None)

                if old_orders:
                    logger.debug(f"Pruned {len(old_orders)} old orders from cache")

            except Exception as e:
                logger.error(f"Error in prune task: {e}")

            await asyncio.sleep(120)  # Run every 2 minutes instead of 60
