import enum
import json
import logging
import asyncio
from asyncio import Queue
import jwt
import time
import functools
import orjson  # Faster JSON parsing
import ujson  # Fast JSON serialization
from typing import Optional, List, Union, Dict, Any, Callable
import numpy as np
from collections import deque

import websockets
from websockets.protocol import State as WsState
from websockets.exceptions import ConnectionClosedError, ConnectionClosedOK


def _make_auth_token(kid, private_key):
    return jwt.encode(
        {"iat": time.time()},
        private_key,
        algorithm="RS512",
        headers={"kid": kid},
    )


class Network(enum.Enum):
    TEST = "wss://testnet.thalex.com/ws/api/v2"
    PROD = "wss://thalex.com/ws/api/v2"


class Direction(enum.Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(enum.Enum):
    LIMIT = "limit"
    MARKET = "market"


class TimeInForce(enum.Enum):
    GTC = "good_till_cancelled"
    IOC = "immediate_or_cancel"


class Collar(enum.Enum):
    IGNORE = "ignore"
    REJECT = "reject"
    CLAMP = "clamp"


class Target(enum.Enum):
    LAST = "last"
    MARK = "mark"
    INDEX = "index"


class Product(enum.Enum):
    BTC_FUTURES = "FBTCUSD"
    BTC_OPTIONS = "OBTCUSD"
    ETH_FUTURES = "FETHUSD"
    ETH_OPTIONS = "OETHUSD"


# Circuit breaker states for rate limiting
class CircuitState(enum.Enum):
    CLOSED = 0  # Normal operation
    OPEN = 1  # Tripped - not allowing requests
    HALF_OPEN = 2  # Testing if system has recovered


class RfqLeg:
    def __init__(self, amount: float, instrument_name: str):
        self.instrument_name = instrument_name
        self.amount = amount

    def dumps(self):
        return {"amount": self.amount, "instrument_name": self.instrument_name}


class SideQuote:
    __slots__ = ["p", "a"]  # Memory optimization

    def __init__(
        self,
        price: float,
        amount: float,
    ):
        self.p = price
        self.a = amount

    def dumps(self):
        return {"a": self.a, "p": self.p}

    def __repr__(self):
        return f"{self.a}@{self.p}"


class Quote:
    __slots__ = ["i", "b", "a"]  # Memory optimization

    def __init__(
        self,
        instrument_name: str = "",
        bid: Optional[SideQuote] = None,
        ask: Optional[SideQuote] = None,
    ):
        self.i = instrument_name
        self.b = bid
        self.a = ask

    def dumps(self):
        d = {"i": self.i}
        if self.b is not None:
            d["b"] = self.b.dumps()
        if self.a is not None:
            d["a"] = self.a.dumps()
        return d

    def __repr__(self):
        return f"(b: {self.b}, a: {self.a})"


class Asset:
    __slots__ = ["asset_name", "amount"]  # Memory optimization

    def __init__(
        self,
        asset_name: float,
        amount: float,
    ):
        self.asset_name = asset_name
        self.amount = amount

    def dumps(self):
        return {"asset_name": self.asset_name, "amount": self.amount}


class Position:
    __slots__ = ["instrument_name", "amount"]  # Memory optimization

    def __init__(
        self,
        instrument_name: float,
        amount: float,
    ):
        self.instrument_name = instrument_name
        self.amount = amount

    def dumps(self):
        return {"instrument_name": self.instrument_name, "amount": self.amount}


# Create an object pool for frequently created objects
class ObjectPool:
    def __init__(self, factory, initial_size=100):
        self.factory = factory
        self.pool = Queue(maxsize=1000)

        # Initialize the pool
        for _ in range(initial_size):
            self.pool.put_nowait(factory())

    async def get(self):
        try:
            return self.pool.get_nowait()
        except asyncio.QueueEmpty:
            return self.factory()

    async def put(self, obj):
        try:
            self.pool.put_nowait(obj)
        except asyncio.QueueFull:
            pass


class RequestBatcher:
    """Batches requests to improve throughput"""

    def __init__(self, thalex, batch_size=10, batch_interval=0.05):
        self.thalex = thalex
        self.batch_size = batch_size
        self.batch_interval = batch_interval
        self.batch_queue = asyncio.Queue()
        self.running = False
        self.batch_task = None

    async def start(self):
        """Start the batch processor"""
        self.running = True
        self.batch_task = asyncio.create_task(self._process_batches())

    async def stop(self):
        """Stop the batch processor"""
        self.running = False
        if self.batch_task:
            self.batch_task.cancel()
            try:
                await self.batch_task
            except asyncio.CancelledError:
                pass

    async def _process_batches(self):
        """Process batches of requests"""
        batch = []
        last_send_time = time.time()

        while self.running:
            try:
                # Get a request with timeout
                try:
                    request = await asyncio.wait_for(
                        self.batch_queue.get(), timeout=self.batch_interval
                    )
                    batch.append(request)
                except asyncio.TimeoutError:
                    # Interval elapsed, check if we need to send batch
                    pass

                # Send if batch is full or interval elapsed
                current_time = time.time()
                if len(batch) >= self.batch_size or (
                    batch and current_time - last_send_time >= self.batch_interval
                ):
                    if batch:
                        # Mass quote batching - special handling for quotes
                        quotes = []
                        other_requests = []

                        for req in batch:
                            method, params, future = req
                            if method == "private/mass_quote":
                                quotes.extend(params.get("quotes", []))
                                # Keep only the last future to resolve all quote requests
                                quote_future = future
                            else:
                                other_requests.append(req)

                        # If we have quotes, batch them into a single request
                        if quotes:
                            quote_params = {"quotes": quotes}
                            # Add last used label, post_only, etc. from the last quote request
                            for param in ["label", "post_only"]:
                                if param in params:
                                    quote_params[param] = params[param]
                            await self.thalex._send_raw(
                                "private/mass_quote", quote_params
                            )
                            quote_future.set_result(True)

                        # Send other requests normally
                        for method, params, future in other_requests:
                            await self.thalex._send_raw(method, params)
                            future.set_result(True)

                        batch = []
                        last_send_time = current_time
            except Exception as e:
                logging.error(f"Error in batch processing: {str(e)}")
                # Clear the batch on error
                for _, _, future in batch:
                    future.set_exception(e)
                batch = []

    async def add(self, method, params):
        """Add a request to the batch queue"""
        future = asyncio.Future()
        await self.batch_queue.put((method, params, future))
        return await future


class CircuitBreaker:
    """Implements the circuit breaker pattern for API rate limiting"""

    def __init__(self, failure_threshold=10, recovery_time=30, test_calls=3):
        self.failure_threshold = failure_threshold
        self.recovery_time = recovery_time
        self.test_calls = test_calls

        self.state = CircuitState.CLOSED
        self.failures = 0
        self.last_failure_time = 0
        self.test_counter = 0

    def record_success(self):
        """Record a successful API call"""
        if self.state == CircuitState.HALF_OPEN:
            self.test_counter += 1
            if self.test_counter >= self.test_calls:
                self.state = CircuitState.CLOSED
                self.failures = 0
                self.test_counter = 0
                logging.info("Circuit breaker reset to CLOSED state")

    def record_failure(self):
        """Record a failed API call"""
        current_time = time.time()
        self.last_failure_time = current_time

        if self.state == CircuitState.CLOSED:
            self.failures += 1
            if self.failures >= self.failure_threshold:
                self.state = CircuitState.OPEN
                logging.warning(
                    f"Circuit breaker tripped to OPEN state. Will retry in {self.recovery_time} seconds."
                )

    def allow_request(self):
        """Check if a request should be allowed"""
        current_time = time.time()

        if self.state == CircuitState.OPEN:
            # Check if recovery time has elapsed
            if current_time - self.last_failure_time >= self.recovery_time:
                self.state = CircuitState.HALF_OPEN
                self.test_counter = 0
                logging.info("Circuit breaker state changed to HALF_OPEN for testing")
                return True
            return False

        return True


class Thalex:
    def __init__(self, network: Network):
        self.net: Network = network
        self.ws: websockets.client = None

        # Connection management
        self.max_reconnect_attempts = 5
        self.reconnect_delay = 1.0  # Initial delay in seconds
        self.max_reconnect_delay = 30.0
        self.connection_pool = []
        self.pool_size = 1  # Number of connections to maintain

        # Message processing
        self.request_batcher = RequestBatcher(self)
        self.message_queue = asyncio.Queue()
        self.processing_task = None

        # Rate limiting
        self.circuit_breaker = CircuitBreaker()
        self.rate_limit = 60  # Default - 60 requests per minute
        self.request_timestamps = deque(maxlen=100)

        # Request prioritization
        self.high_priority_queue = asyncio.PriorityQueue()  # For order operations
        self.low_priority_queue = asyncio.Queue()  # For market data operations

        # Object pools for memory optimization
        self.quote_pool = ObjectPool(Quote)

        # Caching
        self.instrument_cache = {}  # Cache for instrument data
        self._cache_lock = asyncio.Lock()

        # For streaming parsing
        self._partial_data = bytearray()

        # Async event for connection status
        self.connected_event = asyncio.Event()

    async def initialize(self):
        """Initialize the client with all optimizations"""
        # Start the message batcher
        await self.request_batcher.start()

        # Start the message processing task
        self.processing_task = asyncio.create_task(self._process_messages())

        # Connect with initial pool
        await self.connect()

    async def receive(self):
        """Get the next message from the queue"""
        message = await self.message_queue.get()
        return message

    def connected(self):
        """Check if WebSocket is connected"""
        return self.ws is not None and self.ws.state in [
            WsState.CONNECTING,
            WsState.OPEN,
        ]

    async def connect(self):
        """Connect with exponential backoff retry"""
        attempt = 0
        delay = self.reconnect_delay

        while attempt < self.max_reconnect_attempts:
            try:
                self.ws = await websockets.connect(self.net.value, ping_interval=5)
                logging.info(f"Connected to {self.net.value}")
                self.connected_event.set()
                return True
            except Exception as e:
                attempt += 1
                logging.error(f"Connection attempt {attempt} failed: {str(e)}")

                if attempt < self.max_reconnect_attempts:
                    logging.info(f"Retrying in {delay} seconds...")
                    await asyncio.sleep(delay)
                    delay = min(
                        delay * 2, self.max_reconnect_delay
                    )  # Exponential backoff
                else:
                    logging.error("Max reconnection attempts reached")
                    self.connected_event.clear()
                    return False

    async def disconnect(self):
        """Disconnect and cleanup resources"""
        # Stop the batcher
        await self.request_batcher.stop()

        # Stop message processing
        if self.processing_task:
            self.processing_task.cancel()
            try:
                await self.processing_task
            except asyncio.CancelledError:
                pass

        # Close websocket
        if self.ws:
            await self.ws.close()
            self.ws = None

        self.connected_event.clear()

    async def ping(self):
        """Send a ping to keep the connection alive"""
        if self.ws and self.ws.state == WsState.OPEN:
            try:
                await self.ws.ping()
                return True
            except Exception as e:
                logging.error(f"Error sending ping: {str(e)}")
                self.connected_event.clear()
                return False
        return False

    async def _process_messages(self):
        """Background task to process incoming WebSocket messages"""
        while True:
            try:
                # Ensure connection before proceeding
                if not self.connected():
                    await asyncio.sleep(0.1)
                    continue

                # Receive and process message
                raw_message = await self.ws.recv()

                # Use faster JSON parser
                try:
                    message = orjson.loads(raw_message)

                    # Add to queue for processing by application
                    await self.message_queue.put(ujson.dumps(message))
                except Exception as e:
                    logging.error(f"Error parsing message: {str(e)}")
                    await self.message_queue.put(raw_message)  # Fall back to original
            except ConnectionClosedError as e:
                logging.error(f"WebSocket connection closed with error: {str(e)}")
                self.connected_event.clear()
                await self.reconnect()
            except ConnectionClosedOK:
                logging.info("WebSocket connection closed normally")
                self.connected_event.clear()
                break
            except Exception as e:
                logging.error(f"Error in message processing: {str(e)}")
                await asyncio.sleep(0.1)  # Avoid tight loop on persistent errors

    async def reconnect(self):
        """Reconnect the WebSocket with exponential backoff"""
        await self.connect()

        # Resubscribe to any active subscriptions
        # (This would require tracking active subscriptions,
        # which could be added in a future enhancement)

    async def _check_rate_limit(self):
        """Check if we're under rate limit"""
        current_time = time.time()

        # Clean old timestamps
        while (
            self.request_timestamps and current_time - self.request_timestamps[0] > 60
        ):
            self.request_timestamps.popleft()

        # Check if we're at the limit
        return len(self.request_timestamps) < self.rate_limit

    def _update_rate_limit(self):
        """Record a new request for rate limiting"""
        self.request_timestamps.append(time.time())

    async def _send_raw(self, method: str, params: dict = None):
        """Low-level send without batching or circuit breaking"""
        if not params:
            params = {}

        # Fix: Extract 'id' from params if present and put it at top level
        # This fixes the issue where ID was being sent inside params{} instead of root
        req_id = params.pop("id", None)

        request = {"method": method, "params": params}
        if req_id is not None:
            request["id"] = req_id

        request_str = json.dumps(request)  # Use standard json to avoid escaping issues

        # DEBUG LOG
        # logging.info(f"TX RAW: {request_str}")

        if self.ws and self.ws.state == WsState.OPEN:
            await self.ws.send(request_str)
            return True
        return False

    async def _send(self, method: str, id: Optional[int], **kwargs):
        """Enhanced send with batching, circuit breaking, and priority"""
        # Add ID to request if provided
        params = {}
        if id is not None:
            params["id"] = id

        # Build parameters
        for key, value in kwargs.items():
            if value is not None:
                params[key] = value

        # Check if circuit breaker allows this request
        if not self.circuit_breaker.allow_request():
            logging.warning(f"Circuit breaker prevents sending {method}")
            raise Exception("Rate limit circuit breaker is open")

        # Check if we're under rate limit
        if not await self._check_rate_limit():
            self.circuit_breaker.record_failure()
            logging.warning("Rate limit exceeded")
            raise Exception("Rate limit exceeded")

        try:
            # Attempt to send via batcher for most requests
            # Critical order operations bypass batching
            if method.startswith(("private/insert", "private/amend", "private/cancel")):
                # Priority sending for order operations
                result = await self._send_raw(method, params)
            else:
                # Use batching for other operations
                result = await self.request_batcher.add(method, params)

            # Update rate limit counter
            self._update_rate_limit()

            # Record success with circuit breaker
            self.circuit_breaker.record_success()

            return result
        except Exception as e:
            # Record failure with circuit breaker
            self.circuit_breaker.record_failure()
            logging.error(f"Error sending {method}: {str(e)}")
            raise

    # Cache invalidation utility
    async def invalidate_cache(self, cache_key=None):
        """Invalidate specific cache or entire cache"""
        async with self._cache_lock:
            if cache_key:
                if cache_key in self.instrument_cache:
                    del self.instrument_cache[cache_key]
            else:
                self.instrument_cache.clear()

    # Original API methods with minimal changes
    async def login(
        self,
        key_id: str,
        private_key: str,
        account: Optional[str] = None,
        id: Optional[int] = None,
    ):
        """Login

        :key_id:  The key id to use
        :private_key:  Private key to use
        :account:  Number of an account to select for use in this session. Optional, if not specified,
        default account for the API key is selected.
        """
        await self._send(
            "public/login",
            id,
            token=_make_auth_token(key_id, private_key),
            account=account,
        )

    async def set_cancel_on_disconnect(
        self, timeout_secs: int, id: Optional[int] = None
    ):
        """Set cancel on disconnect

        :timeout_secs:  Heartbeat interval
        """
        await self._send(
            "private/set_cancel_on_disconnect",
            id,
            timeout_secs=timeout_secs,
        )

    async def instruments(self, id: Optional[int] = None):
        """Active instruments"""
        await self._send("public/instruments", id)

    async def all_instruments(self, id: Optional[int] = None):
        """All instruments"""
        await self._send("public/all_instruments", id)

    async def instrument(self, instrument_name: str, id: Optional[int] = None):
        """Single instrument

        :instrument_name:  Name of the instrument to query.
        """
        await self._send("public/instrument", id, instrument_name=instrument_name)

    async def ticker(self, instrument_name: str, id: Optional[int] = None):
        """Single ticker value

        :instrument_name:  Name of the instrument to query.
        """
        await self._send("public/ticker", id, instrument_name=instrument_name)

    async def index(self, underlying: str, id: Optional[int] = None):
        """Single index value

        :underlying:  The underlying (e.g. `BTCUSD`).
        """
        await self._send("public/index", id, underlying=underlying)

    async def book(self, instrument_name: str, id: Optional[int] = None):
        """Single order book

        :instrument_name:  Name of the instrument to query.
        """
        await self._send("public/book", id, instrument_name=instrument_name)

    async def insert(
        self,
        direction: Direction,
        instrument_name: str,
        amount: float,
        client_order_id: Optional[int] = None,
        price: Optional[float] = None,
        label: Optional[str] = None,
        order_type: Optional[OrderType] = None,
        time_in_force: Optional[TimeInForce] = None,
        post_only: Optional[bool] = None,
        reject_post_only: Optional[bool] = None,
        reduce_only: Optional[bool] = None,
        collar: Optional[Collar] = None,
        id: Optional[int] = None,
    ):
        """Insert order

        :direction:  Direction
        :client_order_id:  Session-local identifier for this order. Only valid for websocket sessions. If set,
        must be an integer between 0 and 2^64-1, inclusive. When using numbers larger than 2^32,
        please beware of implicit floating point conversions in some JSON libraries.
        :instrument_name:  Instrument name
        :price:  Limit price; required for limit orders.
        :amount:  Amount of currency to trade (e.g. BTC for futures).
        :label: {'type': 'string'},
        :order_type:  OrderType, default': 'limit'
        :time_in_force:  Note that for limit orders, the default `time_in_force` is `good_till_cancelled`,
        while for market orders, the default is `immediate_or_cancel`.
        It is illegal to send a GTC market order, or an IOC post order.
        :post_only:  If the order price is in cross with the current best price on the opposite side in the
        order book, then the price is adjusted to one tick away from that price, ensuring that
        the order will never trade on insert. If the adjusted price of a buy order falls at or
        below zero where not allowed, then the order is cancelled with delete reason 'immediate_cancel'.
        :reject_post_only:  This flag is only effective in combination with post_only.
        If set, then instead of adjusting the order price, the order will be cancelled with delete reason 'immediate_cancel'.
        The combination of post_only and reject_post_only is effectively a book-or-cancel order.
        :reduce_only:  An order marked `reduce_only` will have its amount reduced to the open position.
        If there is no open position, or if the order direction would cause an increase of the open position,
        the order is rejected. If the order is placed in the book, it will be subsequently monitored,
        and reduced to the open position if the position changes through other means (best effort).
        Multiple reduce-only orders will all be reduced individually.
        :collar:  If the instrument has a safety price collar set, and the limit price of the order
        (infinite for market orders) is in cross with (more aggressive than) this collar, how to handle.
        If set to `ignore`, the order will proceed as requested. If `reject`,\nthe order fails early.
        If `clamp`, the price is adjusted to the collar.
        The default is `clamp` for market orders and `reject` for everything else.
        Collar `ignore` is forbidden for market orders.
        """
        await self._send(
            "private/insert",
            id,
            direction=direction.value,
            instrument_name=instrument_name,
            amount=amount,
            client_order_id=client_order_id,
            price=price,
            label=label,
            order_type=order_type.value if order_type is not None else None,
            time_in_force=time_in_force.value if time_in_force is not None else None,
            post_only=post_only,
            reject_post_only=reject_post_only,
            reduce_only=reduce_only,
            collar=collar,
        )

    async def buy(
        self,
        instrument_name: str,
        amount: float,
        client_order_id: Optional[int] = None,
        price: Optional[float] = None,
        label: Optional[str] = None,
        order_type: Optional[OrderType] = None,
        time_in_force: Optional[TimeInForce] = None,
        post_only: Optional[bool] = None,
        reject_post_only: Optional[bool] = None,
        reduce_only: Optional[bool] = None,
        collar: Optional[Collar] = None,
        id: Optional[int] = None,
    ):
        """Insert buy order

        :client_order_id:  Session-local identifier for this order. Only valid for websocket sessions. If set,
        must be an integer between 0 and 2^64-1, inclusive. When using numbers larger than 2^32,
        please beware of implicit floating point conversions in some JSON libraries.
        :instrument_name:  Instrument name
        :price:  Limit price; required for limit orders.
        :amount:  Amount of currency to trade (e.g. BTC for futures).
        :label: {'type': 'string'},
        :order_type:  OrderType, default': 'limit'
        :time_in_force:  Note that for limit orders, the default `time_in_force` is `good_till_cancelled`,
        while for market orders, the default is `immediate_or_cancel`.
        It is illegal to send a GTC market order, or an IOC post order.
        :post_only:  If the order price is in cross with the current best price on the opposite side in the
        order book, then the price is adjusted to one tick away from that price, ensuring that
        the order will never trade on insert. If the adjusted price of a buy order falls at or
        below zero where not allowed, then the order is cancelled with delete reason 'immediate_cancel'.
        :reject_post_only:  This flag is only effective in combination with post_only.
        If set, then instead of adjusting the order price, the order will be cancelled with delete reason 'immediate_cancel'.
        The combination of post_only and reject_post_only is effectively a book-or-cancel order.
        :reduce_only:  An order marked `reduce_only` will have its amount reduced to the open position.
        If there is no open position, or if the order direction would cause an increase of the open position,
        the order is rejected. If the order is placed in the book, it will be subsequently monitored,
        and reduced to the open position if the position changes through other means (best effort).
        Multiple reduce-only orders will all be reduced individually.
        :collar:  If the instrument has a safety price collar set, and the limit price of the order
        (infinite for market orders) is in cross with (more aggressive than) this collar, how to handle.
        If set to `ignore`, the order will proceed as requested. If `reject`,\nthe order fails early.
        If `clamp`, the price is adjusted to the collar.
        The default is `clamp` for market orders and `reject` for everything else.
        Collar `ignore` is forbidden for market orders.
        """
        await self._send(
            "private/buy",
            id,
            instrument_name=instrument_name,
            amount=amount,
            client_order_id=client_order_id,
            price=price,
            label=label,
            order_type=order_type.value if order_type is not None else None,
            time_in_force=time_in_force.value if time_in_force is not None else None,
            post_only=post_only,
            reject_post_only=reject_post_only,
            reduce_only=reduce_only,
            collar=collar,
        )

    async def sell(
        self,
        instrument_name: str,
        amount: float,
        client_order_id: Optional[int] = None,
        price: Optional[float] = None,
        label: Optional[str] = None,
        order_type: Optional[OrderType] = None,
        time_in_force: Optional[TimeInForce] = None,
        post_only: Optional[bool] = None,
        reject_post_only: Optional[bool] = None,
        reduce_only: Optional[bool] = None,
        collar: Optional[Collar] = None,
        id: Optional[int] = None,
    ):
        """Insert sell order

        :client_order_id:  Session-local identifier for this order. Only valid for websocket sessions. If set,
        must be an integer between 0 and 2^64-1, inclusive. When using numbers larger than 2^32,
        please beware of implicit floating point conversions in some JSON libraries.
        :instrument_name:  Instrument name
        :price:  Limit price; required for limit orders.
        :amount:  Amount of currency to trade (e.g. BTC for futures).
        :label: {'type': 'string'},
        :order_type:  OrderType, default': 'limit'
        :time_in_force:  Note that for limit orders, the default `time_in_force` is `good_till_cancelled`,
        while for market orders, the default is `immediate_or_cancel`.
        It is illegal to send a GTC market order, or an IOC post order.
        :post_only:  If the order price is in cross with the current best price on the opposite side in the
        order book, then the price is adjusted to one tick away from that price, ensuring that
        the order will never trade on insert. If the adjusted price of a buy order falls at or
        below zero where not allowed, then the order is cancelled with delete reason 'immediate_cancel'.
        :reject_post_only:  This flag is only effective in combination with post_only.
        If set, then instead of adjusting the order price, the order will be cancelled with delete reason 'immediate_cancel'.
        The combination of post_only and reject_post_only is effectively a book-or-cancel order.
        :reduce_only:  An order marked `reduce_only` will have its amount reduced to the open position.
        If there is no open position, or if the order direction would cause an increase of the open position,
        the order is rejected. If the order is placed in the book, it will be subsequently monitored,
        and reduced to the open position if the position changes through other means (best effort).
        Multiple reduce-only orders will all be reduced individually.
        :collar:  If the instrument has a safety price collar set, and the limit price of the order
        (infinite for market orders) is in cross with (more aggressive than) this collar, how to handle.
        If set to `ignore`, the order will proceed as requested. If `reject`,\nthe order fails early.
        If `clamp`, the price is adjusted to the collar.
        The default is `clamp` for market orders and `reject` for everything else.
        Collar `ignore` is forbidden for market orders.
        """
        await self._send(
            "private/sell",
            id,
            instrument_name=instrument_name,
            amount=amount,
            client_order_id=client_order_id,
            price=price,
            label=label,
            order_type=order_type.value if order_type is not None else None,
            time_in_force=time_in_force.value if time_in_force is not None else None,
            post_only=post_only,
            reject_post_only=reject_post_only,
            reduce_only=reduce_only,
            collar=collar,
        )

    async def amend(
        self,
        amount: float,
        price: float,
        order_id: Optional[str] = None,
        client_order_id: Optional[int] = None,
        collar: Optional[Collar] = None,
        id: Optional[int] = None,
    ):
        """Amend order

        :client_order_id:  Exactly one of `client_order_id` or `order_id` must be specified.
        :order_id:  Exactly one of `client_order_id` or `order_id` must be specified.
        :price: number
        :amount: number
        :collar:  If the instrument has a safety price collar set, and the new limit price
        is in cross with (more aggressive than) this collar,
        how to handle. If set to `ignore`, the amend will proceed as requested. If `reject`,
        the request fails early. If `clamp`, the price is adjusted to the collar.

        The default is `reject`.
        """
        await self._send(
            "private/amend",
            id,
            amount=amount,
            price=price,
            order_id=order_id,
            client_order_id=client_order_id,
            collar=collar,
        )

    async def cancel(
        self,
        order_id: Optional[str] = None,
        client_order_id: Optional[int] = None,
        id: Optional[int] = None,
    ):
        """Cancel order

        :client_order_id:  Exactly one of `client_order_id` or `order_id` must be specified.
        :order_id:  Exactly one of `client_order_id` or `order_id` must be specified.
        """
        await self._send(
            "private/cancel",
            id,
            order_id=order_id,
            client_order_id=client_order_id,
        )

    async def cancel_all(
        self,
        id: Optional[int] = None,
    ):
        """Bulk cancel all orders"""
        await self._send(
            "private/cancel_all",
            id,
        )

    async def cancel_session(
        self,
        id: Optional[int] = None,
    ):
        """Bulk cancel all orders in session"""
        await self._send(
            "private/cancel_session",
            id,
        )

    async def create_rfq(
        self,
        legs: List[RfqLeg],
        label: Optional[str] = None,
        id: Optional[int] = None,
    ):
        """Create a request for quote

        :legs:  Specify any number of legs that you'd like to trade in a single package. Leg amounts
        may be positive (long) or negative (short), and must adhere to the regular volume tick size for the
        respective instrument. At least one leg must be long.

        :label:  User label for this RFQ, which will be reflected in eventual trades.
        """
        await self._send(
            "private/create_rfq", id, legs=[leg.dumps() for leg in legs], label=label
        )

    async def cancel_rfq(
        self,
        rfq_id,
        id: Optional[int] = None,
    ):
        """Cancel an RFQ

        :rfq_id:  The ID of the RFQ to be cancelled
        """
        await self._send("private/cancel_rfq", id, rfq_id=rfq_id)

    async def trade_rfq(
        self,
        rfq_id: str,
        direction: Direction,
        limit_price: float,
        id: Optional[int] = None,
    ):
        """Trade an RFQ

        :rfq_id:  The ID of the RFQ
        :direction:  Whether to buy or sell. *Important*: this relates to the combination as created by the system, *not* the
        package as originally requested (although they should be equal).

        :limit_price:  The maximum (for buy) or minimum (for sell) price to trade at. This is the price for one combination, not
        for the entire package.
        """
        await self._send(
            "private/trade_rfq",
            id,
            rfq_id=rfq_id,
            direction=direction.value,
            limit_price=limit_price,
        )

    async def open_rfqs(self, id: Optional[int] = None):
        """Retrieves a list of open RFQs created by this account."""
        await self._send("private/open_rfqs", id)

    async def mm_rfqs(
        self,
        id: Optional[int] = None,
    ):
        """Retrieves a list of open RFQs that this account has access to."""
        await self._send(
            "private/mm_rfqs",
            id,
        )

    async def mm_rfq_insert_quote(
        self,
        direction: Direction,
        amount: float,
        price: float,
        rfq_id: str,
        client_order_id: Optional[int] = None,
        label: Optional[str] = None,
        id: Optional[int] = None,
    ):
        """Quote on an RFQ

        :rfq_id:  The ID of the RFQ this quote is for.
        :client_order_id:  Session-local identifier for this order. Only valid for websocket sessions. If set, must be a
        number between 0 and 2^64-1, inclusive. When using numbers larger than 2^32, please beware of implicit
        floating point conversions in some JSON libraries.

        :direction:  The side of the quote.

        :price:  Limit price for the quote (for one combination).
        :amount:  Number of combinations to quote. Anything over the requested amount will not be visible to the requester.

        :label:  A label to attach to eventual trades.
        """
        await self._send(
            "private/mm_rfq_insert_quote",
            id,
            rfq_id=rfq_id,
            direction=direction.value,
            amount=amount,
            price=price,
            client_order_id=client_order_id,
            label=label,
        )

    async def mm_rfq_amend_quote(
        self,
        amount: float,
        price: float,
        order_id: Optional[int] = None,
        client_order_id: Optional[int] = None,
        id: Optional[int] = None,
    ):
        """Amend quote

        :client_order_id:  Exactly one of `client_order_id` or `order_id` must be specified.
        :order_id:  Exactly one of `client_order_id` or `order_id` must be specified.
        :price:  Limit price for the quote (for one combination).
        :amount:  Number of combinations to quote. Anything over the requested amount will not be visible to the requester.
        """
        await self._send(
            "private/mm_rfq_amend_quote",
            id,
            amount=amount,
            price=price,
            order_id=order_id,
            client_order_id=client_order_id,
        )

    async def mm_rfq_delete_quote(
        self,
        order_id: Optional[int] = None,
        client_order_id: Optional[int] = None,
        id: Optional[int] = None,
    ):
        """Delete quote

        :client_order_id:  Exactly one of `client_order_id` or `order_id` must be specified.
        :order_id:  Exactly one of `client_order_id` or `order_id` must be specified.
        """
        await self._send(
            "private/mm_rfq_delete_quote",
            id,
            order_id=order_id,
            client_order_id=client_order_id,
        )

    async def mm_rfq_quotes(
        self,
        id: Optional[int] = None,
    ):
        """List of active quotes"""
        await self._send(
            "private/mm_rfq_quotes",
            id,
        )

    async def portfolio(
        self,
        id: Optional[int] = None,
    ):
        """Portfolio"""
        await self._send(
            "private/portfolio",
            id,
        )

    async def open_orders(
        self,
        id: Optional[int] = None,
    ):
        """Open orders"""
        await self._send(
            "private/open_orders",
            id,
        )

    async def order_history(
        self,
        limit: Optional[int] = None,
        time_low: Optional[int] = None,
        time_high: Optional[int] = None,
        bookmark: Optional[str] = None,
        id: Optional[int] = None,
    ):
        """Order history

        :limit:  Max results to return.
        :time_low:  Start time (UNIX timestamp) defaults to zero.
        :time_high:  End time (UNIX timestamp) defaults to now.
        :bookmark:  Set to bookmark from previous call to get next page.
        """
        await self._send(
            "private/order_history",
            id,
            limit=limit,
            time_low=time_low,
            time_high=time_high,
            bookmark=bookmark,
        )

    async def trade_history(
        self,
        limit: Optional[int] = None,
        time_low: Optional[int] = None,
        time_high: Optional[int] = None,
        bookmark: Optional[str] = None,
        id: Optional[int] = None,
    ):
        """Trade history

        :limit:  Max results to return.
        :time_low:  Start time (UNIX timestamp) defaults to zero.
        :time_high:  End time (UNIX timestamp) defaults to now.
        :bookmark:  Set to bookmark from previous call to get next page.
        """
        await self._send(
            "private/trade_history",
            id,
            limit=limit,
            time_low=time_low,
            time_high=time_high,
            bookmark=bookmark,
        )

    async def transaction_history(
        self,
        limit: Optional[int] = None,
        time_low: Optional[int] = None,
        time_high: Optional[int] = None,
        bookmark: Optional[str] = None,
        id: Optional[int] = None,
    ):
        """Transaction history

        :limit:  Max results to return.
        :time_low:  Start time (UNIX timestamp) defaults to zero.
        :time_high:  End time (UNIX timestamp) defaults to now.
        :bookmark:  Set to bookmark from previous call to get next page.
        """
        await self._send(
            "private/transaction_history",
            id,
            limit=limit,
            time_low=time_low,
            time_high=time_high,
            bookmark=bookmark,
        )

    async def rfq_history(
        self,
        limit: Optional[int] = None,
        time_low: Optional[int] = None,
        time_high: Optional[int] = None,
        bookmark: Optional[str] = None,
        id: Optional[int] = None,
    ):
        """RFQ history

        :limit:  Max results to return.
        :time_low:  Start time (UNIX timestamp) defaults to zero.
        :time_high:  End time (UNIX timestamp) defaults to now.
        :bookmark:  Set to bookmark from previous call to get next page.
        """
        await self._send(
            "private/rfq_history",
            id,
            limit=limit,
            time_low=time_low,
            time_high=time_high,
            bookmark=bookmark,
        )

    async def account_breakdown(
        self,
        id: Optional[int] = None,
    ):
        """Account breakdown"""
        await self._send(
            "private/account_breakdown",
            id,
        )

    async def account_summary(
        self,
        id: Optional[int] = None,
    ):
        """Account summary"""
        await self._send(
            "private/account_summary",
            id,
        )

    async def required_margin_breakdown(
        self,
        id: Optional[int] = None,
    ):
        """Margin breakdown"""
        await self._send(
            "private/required_margin_breakdown",
            id,
        )

    async def required_margin_for_order(
        self,
        instrument_name: str,
        price: float,
        amount: float,
        id: Optional[int] = None,
    ):
        """Margin breakdown with order

        :instrument_name:  The name of the instrument of this hypothetical order with which the margin is to be broken down with.
        :price:  The price of the hypothetical order.
        :amount:  The amount that would be traded.
        """
        await self._send(
            "private/required_margin_for_order",
            id,
            instrument_name=instrument_name,
            amount=amount,
            price=price,
        )

    async def private_subscribe(self, channels: [str], id: Optional[int] = None):
        """Subscribe to private channels

        :channels:  List of channels to subscribe to.
        """
        await self._send("private/subscribe", id, channels=channels)

    async def public_subscribe(self, channels: [str], id: Optional[int] = None):
        """Subscribe to public channels

        :channels:  List of channels to subscribe to.
        """
        await self._send("public/subscribe", id, channels=channels)

    async def unsubscribe(self, channels: [str], id: Optional[int] = None):
        """Unsubscribe

        :channels:  List of channels to unsubscribe from. Public and private channels may be mixed.
        """
        await self._send("unsubscribe", id, channels=channels)

    async def conditional_orders(
        self,
        id: Optional[int] = None,
    ):
        """Conditional orders"""
        await self._send(
            "private/conditional_orders",
            id,
        )

    async def create_conditional_order(
        self,
        direction: Direction,
        instrument_name: str,
        amount: float,
        stop_price: float,
        limit_price: Optional[float] = None,
        bracket_price: Optional[float] = None,
        trailing_stop_callback_rate: Optional[float] = None,
        label: Optional[str] = None,
        reduce_only: Optional[bool] = None,
        target: Optional[Target] = None,
        id: Optional[int] = None,
    ):
        """Create conditional order

        :direction: enum
        :instrument_name: string
        :amount: number
        :limit_price:  If set, creates a stop limit order
        :target:  The trigger target that `stop_price` and `bracket_price` refer to.
        :stop_price:  Trigger price
        :bracket_price:  If set, creates a bracket order
        :trailing_stop_callback_rate:  If set, creates a trailing stop order
        :label:  Label will be set on the activated order
        :reduce_only:  Activated order will be reduce-only
        """
        await self._send(
            "private/create_conditional_order",
            id,
            direction=direction.value,
            instrument_name=instrument_name,
            amount=amount,
            label=label,
            reduce_only=reduce_only,
            stop_price=stop_price,
            limit_price=limit_price,
            bracket_price=bracket_price,
            trailing_stop_callback_rate=trailing_stop_callback_rate,
            target=target,
        )

    async def cancel_conditional_order(
        self,
        order_id: Optional[int] = None,
        id: Optional[int] = None,
    ):
        """Cancel conditional order

        :order_id: string
        """
        await self._send(
            "private/cancel_conditional_order",
            id,
            order_id=order_id,
        )

    async def cancel_all_conditional_orders(
        self,
        id: Optional[int] = None,
    ):
        """Bulk cancel conditional orders"""
        await self._send(
            "private/cancel_all_conditional_orders",
            id,
        )

    async def notifications_inbox(
        self,
        limit: Optional[int] = None,
        id: Optional[int] = None,
    ):
        """Notifications inbox

        :limit:  Max results to return.
        """
        await self._send("private/notifications_inbox", id, limit=limit)

    async def mark_inbox_notification_as_read(
        self,
        notification_id: str,
        read: Optional[bool] = None,
        id: Optional[int] = None,
    ):
        """Marking notification as read

        :notification_id:  ID of the notification to mark.
        :read:  Set to `true` to mark as read, `false` to mark as not read.
        """
        await self._send(
            "private/mark_inbox_notification_as_read",
            id,
            notification_id=notification_id,
            read=read,
        )

    async def mass_quote(
        self,
        quotes: List[Quote],
        label: Optional[str] = None,
        post_only: Optional[bool] = None,
        id: Optional[int] = None,
    ):
        """Send a mass quote

        :quotes:  List of quotes (maximum 100).

        Each item is a double sided quote on a single instrument. A quote atomically replace a previous quote. Both
        bid and ask price may be specified. If either bid or ask is not specified, that side is *not* replaced or
        removed. If a double-sided quote for an instrument that was specified in an earlier call is omitted from the next
        call, that quote is *not* removed or replaced. To remove a quote, set the amount to zero.

        To replace only some of the quotes you have, send only the quotes (sides) you need to replace.

        Sending a quote with the exact same price and amount as in the previous call *will* replace the quote, which
        will result in the quote losing priority. It is thus advised to avoid sending duplicate quotes.

        Note that mass quoting only allows for a one level quote on each side on the instrument. I.e. if you specify
        two or more double sided quotes on the same instrument then the quotes occurring earlier in the list will be
        replaced by the quotes occurring later in the list, as if all the double sided quotes for the same instrument
        were sent in separate API calls.

        Note that market maker protection must have been configured for the instrument's product group, and both bid
        and ask amount must not exceed the most recent protection configuration amount.

        :label:  Optional user label to apply to every quote side.
        :post_only:  If set, price may be widened so it will not cross an existing order in the book.
        If the adjusted price for any bid falls at or below zero where not allowed, then
        that side will be removed with delete reason 'immediate_cancel'.
        """
        await self._send(
            "private/mass_quote",
            id,
            quotes=[q.dumps() for q in quotes],
            label=label,
            post_only=post_only,
        )

    async def set_mm_protection(
        self,
        product: Union[Product, str],
        amount: float,
        id: Optional[int] = None,
    ):
        """Market maker protection configuration

        :product:  Product group ('F' + index or 'O' + index)
        :amount:  Amount to execute before remaining mass quotes are cancelled
        """
        if isinstance(product, Product):
            product = product.value
        await self._send(
            "private/set_mm_protection", id, product=product, amount=amount
        )

    async def verify_withdrawal(
        self,
        asset_name: str,
        amount: float,
        target_address: str,
        id: Optional[int] = None,
    ):
        """Verify if withdrawal is possible

        :asset_name:  Asset name.
        :amount:  Amount to withdraw.
        :target_address:  Target address.
        """
        await self._send(
            "private/verify_withdrawal",
            id,
            asset_name=asset_name,
            amount=amount,
            target_address=target_address,
        )

    async def withdraw(
        self,
        asset_name: str,
        amount: float,
        target_address: str,
        label: Optional[str] = None,
        id: Optional[int] = None,
    ):
        """Withdraw assets

        :asset_name:  Asset name.
        :amount:  Amount to withdraw.
        :target_address:  Target address.
        :label:  Optional label to attach to the withdrawal request.
        """
        await self._send(
            "private/withdraw",
            id,
            asset_name=asset_name,
            amount=amount,
            target_address=target_address,
            label=label,
        )

    async def crypto_withdrawals(
        self,
        id: Optional[int] = None,
    ):
        """Withdrawals"""
        await self._send(
            "public/crypto_withdrawals",
            id,
        )

    async def crypto_deposits(
        self,
        id: Optional[int] = None,
    ):
        """Deposits"""
        await self._send(
            "public/crypto_deposits",
            id,
        )

    async def btc_deposit_address(
        self,
        id: Optional[int] = None,
    ):
        """Bitcoin deposit address"""
        await self._send(
            "public/btc_deposit_address",
            id,
        )

    async def eth_deposit_address(
        self,
        id: Optional[int] = None,
    ):
        """Ethereum deposit address"""
        await self._send(
            "public/eth_deposit_address",
            id,
        )

    async def verify_internal_transfer(
        self,
        destination_account_number: str,
        assets: Optional[List[Asset]] = None,
        positions: Optional[List[Position]] = None,
        id: Optional[int] = None,
    ):
        """Verify internal transfer

        :destination_account_number:  Destination account number.
        :assets: array
        :positions: array
        """
        await self._send(
            "private/verify_internal_transfer",
            id,
            destination_account_number=destination_account_number,
            assets=assets,
            positions=positions,
        )

    async def internal_transfer(
        self,
        destination_account_number: str,
        assets: Optional[List[Asset]] = None,
        positions: Optional[List[Position]] = None,
        label: Optional[str] = None,
        id: Optional[int] = None,
    ):
        """Internal transfer

        :destination_account_number:  Destination account number.
        :assets: array
        :positions: array
        :label:  Optional label attached to the transfer.
        """
        await self._send(
            "private/internal_transfer",
            id,
            destination_account_number=destination_account_number,
            assets=assets,
            positions=positions,
            label=label,
        )

    async def system_info(
        self,
        id: Optional[int] = None,
    ):
        """Get system info"""
        await self._send("public/system_info", id)
