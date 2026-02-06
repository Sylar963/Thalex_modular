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
from .base_adapter import TokenBucket
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


class ThalexAdapter(ExchangeGateway):
    """
    Adapter for the Thalex exchange using the official python library.
    Implements robust connection handling, heartbeat monitoring, and RPC correlation.
    """

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        testnet: bool = True,
        me_rate_limit: float = 45.0,
        cancel_rate_limit: float = 900.0,
    ):
        self.api_key = api_key
        self.api_secret = api_secret
        self.network = Network.TEST if testnet else Network.PROD
        self.client = Thalex(network=self.network)

        self.connected = False
        self.ticker_callback: Optional[Callable] = None
        self.trade_callback: Optional[Callable] = None
        self.order_callback: Optional[Callable] = None
        self.position_callback: Optional[Callable] = None
        self.balance_callback: Optional[Callable] = None

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

        self.me_rate_limiter = TokenBucket(
            capacity=int(me_rate_limit * 1.1), fill_rate=me_rate_limit
        )
        self.cancel_rate_limiter = TokenBucket(
            capacity=int(cancel_rate_limit * 1.1), fill_rate=cancel_rate_limit
        )

    def _safe_float(self, value, default: float = 0.0) -> float:
        if value is None:
            return default
        if isinstance(value, str):
            value = value.strip()
            if not value:
                return default
        try:
            return float(value)
        except (ValueError, TypeError):
            return default

    @property
    def name(self) -> str:
        return "thalex"

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
        if self._prune_task:
            self._prune_task.cancel()

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
            if hasattr(self.client, "get_account_summary"):
                response = await self._rpc_request(self.client.get_account_summary)
                if "result" in response:
                    res = response["result"]
                    total = self._safe_float(res.get("equity", 0.0))
                    available = self._safe_float(res.get("available_funds", 0.0))
                    margin_used = self._safe_float(res.get("margin_used", 0.0))

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
                self.client.private_subscribe,
                channels=[orders_channel, "portfolio", "account"],
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

    async def _send_batch(self, requests: List[Dict]) -> List[Any]:
        """
        Send a batch of JSON-RPC requests in a single WebSocket frame.
        """
        if not requests:
            return []

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

            # Send requests individually (Pipelining)
            # Thalex might not support JSON-RPC Batch Arrays. Pipelining achieves similar throughput.
            for payload in batch_payload:
                req_str = json.dumps(payload)
                logger.debug(f"TX PIPE: {req_str}")
                await self.client.ws.send(req_str)

            # Wait with Timeout
            results = await asyncio.wait_for(
                asyncio.gather(*futures, return_exceptions=True), timeout=5.0
            )
            return results

        except asyncio.TimeoutError:
            logger.error("Batch send timed out")
            for f in futures:
                if not f.done():
                    f.cancel()
            return [TimeoutError("Batch Timeout")] * len(requests)

        except Exception as e:
            logger.error(f"Batch send failed: {e}")
            # Fail all futures
            for f in futures:
                if not f.done():
                    f.set_exception(e)
            return [e] * len(requests)

    async def place_orders_batch(self, orders: List[Order]) -> List[Order]:
        """Place multiple orders in a single batch request."""
        if not self.connected:
            logger.warning(f"Cannot place {len(orders)} orders: Not connected.")
            return [replace(o, status=OrderStatus.REJECTED) for o in orders]

        logger.debug(f"Waiting for rate limit tokens for {len(orders)} orders...")
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

        results = await self._send_batch(requests)
        logger.debug(f"Received {len(results)} results from batch send.")

        updated_orders = []
        for order, res in zip(orders, results):
            if isinstance(res, Exception):
                logger.warning(f"Order {order.id} failed with exception: {res}")
                updated_orders.append(replace(order, status=OrderStatus.REJECTED))
            elif "error" in res:
                logger.warning(f"Order {order.id} rejected: {res.get('error')}")
                updated_orders.append(replace(order, status=OrderStatus.REJECTED))
            else:
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

        return updated_orders

    async def cancel_orders_batch(self, order_ids: List[str]) -> List[bool]:
        """Cancel multiple orders in a single batch."""
        if not self.connected or not order_ids:
            return [False] * len(order_ids)

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
                        msg = json.loads(msg)
                    except json.JSONDecodeError:
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

                await self._handle_single_msg(msg)

            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                import traceback

                logger.error(f"Error in msg loop: {e}\n{traceback.format_exc()}")
                if self.connected:
                    await self._attempt_reconnect()
                await asyncio.sleep(1)

        logger.warning("Message loop ended (Disconnected). Failing pending requests.")
        self._fail_pending_requests(ConnectionError("Disconnected"))

    async def _attempt_reconnect(self):
        max_retries = 10
        base_delay = 1.0
        for attempt in range(max_retries):
            delay = min(base_delay * (2**attempt), 60.0)
            logger.warning(
                f"Reconnection attempt {attempt + 1}/{max_retries} in {delay:.1f}s..."
            )
            await asyncio.sleep(delay)

            try:
                await self.client.disconnect()
                await self.client.initialize()

                if not self.client.connected():
                    continue

                login_id = self._get_next_id()
                future = asyncio.Future()
                self.pending_requests[login_id] = future

                await self.client.login(self.api_key, self.api_secret, id=login_id)
                await asyncio.wait_for(future, timeout=10.0)

                await self._rpc_request(
                    self.client.set_cancel_on_disconnect, timeout_secs=15
                )

                logger.info("Reconnection successful.")
                self.last_msg_time = time.time()
                return

            except Exception as e:
                logger.error(f"Reconnection attempt {attempt + 1} failed: {e}")

        logger.error("All reconnection attempts failed. Giving up.")
        self.connected = False

    def _fail_pending_requests(self, exc: Exception):
        """Cancel all pending requests with exception"""
        for req_id, future in list(self.pending_requests.items()):
            if not future.done():
                future.set_exception(exc)
        self.pending_requests.clear()

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

                    if exchange_id in self.orders:
                        self.orders[exchange_id] = replace(
                            self.orders[exchange_id],
                            status=status,
                            filled_size=filled_size,
                        )
                        self._order_timestamps[exchange_id] = time.time()
                        logger.info(
                            f"Updated order {exchange_id} status to {status} from notification"
                        )

                    if self.order_callback:
                        await self.order_callback(
                            exchange_id, status, filled_size, avg_price
                        )

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
                            await self.position_callback(symbol, amount, entry_price)

            elif channel == "account":
                acc_data = data if isinstance(data, list) else [data]

                for acc in acc_data:
                    # Map fields - adjusting based on assumed Thalex API response for account
                    # total_equity, available_funds, etc.
                    # Fallback if specific fields aren't found, log for debugging
                    total = self._safe_float(acc.get("equity", 0.0))
                    available = self._safe_float(acc.get("available_funds", 0.0))
                    margin_used = self._safe_float(acc.get("margin_used", 0.0))

                    bal = Balance(
                        exchange=self.name,
                        asset="USD",  # Defaulting to USD/USDC for Thalex usually
                        total=total,
                        available=available,
                        margin_used=margin_used,
                        equity=total,
                    )
                    if self.balance_callback:
                        await self.balance_callback(bal)
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
        while self.connected:
            await asyncio.sleep(60)
            now = time.time()
            cutoff = now - 600.0
            stale_ids = [
                oid
                for oid, ts in self._order_timestamps.items()
                if ts < cutoff
                and oid in self.orders
                and self.orders[oid].status
                in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED]
            ]
            for oid in stale_ids:
                del self.orders[oid]
                del self._order_timestamps[oid]
            if stale_ids:
                logger.debug(f"Pruned {len(stale_ids)} stale orders from cache")
