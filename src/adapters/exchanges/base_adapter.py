from abc import abstractmethod
from typing import Optional, Callable, Any, Union
import time
import asyncio

try:
    import orjson
except ImportError:
    import json as orjson  # Fallback for type hinting/mocking, though functionality differs

from ...domain.interfaces import ExchangeGateway, TimeSyncManager


class TokenBucket:
    """
    Optimized Token Bucket algorithm for rate limiting.
    Uses atomic operations and reduced floating-point math for better performance.
    """

    __slots__ = (
        "capacity",
        "_tokens",
        "fill_rate",
        "last_update",
        "_capacity_int",
        "_fill_rate_per_ms",
    )

    def __init__(self, capacity: int, fill_rate: float):
        self.capacity = float(capacity)
        self._tokens = float(capacity)
        self.fill_rate = fill_rate  # tokens per second
        self.last_update = time.time()

        # Pre-computed values for optimization
        self._capacity_int = capacity
        self._fill_rate_per_ms = fill_rate / 1000.0  # tokens per millisecond

    def consume(self, tokens: int = 1) -> bool:
        """
        Consume tokens from the bucket. Thread-safe for single-threaded async usage.
        Optimized to minimize floating-point operations and time calls.
        """
        now = time.time()
        # Calculate time elapsed in milliseconds to reduce floating point operations
        time_elapsed_ms = (now - self.last_update) * 1000.0
        # Add tokens based on time elapsed
        tokens_to_add = time_elapsed_ms * self._fill_rate_per_ms
        self._tokens = min(self.capacity, self._tokens + tokens_to_add)
        self.last_update = now

        if self._tokens >= tokens:
            self._tokens -= tokens
            return True
        return False

    async def consume_wait(self, tokens: int = 1) -> bool:
        """
        Async version that waits if tokens are not available.
        Optimized to reduce sleep frequency and improve responsiveness.
        """
        while True:
            now = time.time()
            time_elapsed_ms = (now - self.last_update) * 1000.0
            tokens_to_add = time_elapsed_ms * self._fill_rate_per_ms
            self._tokens = min(self.capacity, self._tokens + tokens_to_add)
            self.last_update = now

            if self._tokens >= tokens:
                self._tokens -= tokens
                return True

            # Calculate wait time more efficiently
            tokens_needed = tokens - self._tokens
            if self.fill_rate > 0:
                wait_time = (tokens_needed / self.fill_rate) * 0.9  # Small buffer
                # Cap wait time to avoid long sleeps
                await asyncio.sleep(min(wait_time, 0.05))  # Max 50ms sleep
            else:
                # If fill_rate is 0, we'll never get tokens, so break
                return False


class BaseExchangeAdapter(ExchangeGateway):
    RECV_WINDOW = 20000  # Default 20s for most exchanges

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        testnet: bool = True,
        time_sync_manager: Optional[TimeSyncManager] = None,
    ):
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        self.time_sync_manager = time_sync_manager
        self.connected = False
        self.is_reconnecting = False
        self.last_ticker_time = time.time()

        self.ticker_callback: Optional[Callable] = None
        self.trade_callback: Optional[Callable] = None
        self.order_callback: Optional[Callable] = None
        self.position_callback: Optional[Callable] = None
        self.balance_callback: Optional[Callable] = None
        self.execution_callback: Optional[Callable] = None  # For bot fills

    def _get_timestamp(self) -> int:
        """Standardized method to get exchange-aligned timestamp in milliseconds."""
        if self.time_sync_manager:
            return self.time_sync_manager.get_timestamp(self.name)
        return int(time.time() * 1000)

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    def set_ticker_callback(self, callback: Callable):
        self.ticker_callback = callback

    def set_trade_callback(self, callback: Callable):
        self.trade_callback = callback

    def set_order_callback(self, callback: Callable):
        self.order_callback = callback

    def set_position_callback(self, callback: Callable):
        self.position_callback = callback

    def set_balance_callback(self, callback: Callable):
        self.balance_callback = callback

    def set_execution_callback(self, callback: Callable):
        self.execution_callback = callback

    async def notify_order_update(
        self,
        exchange_id: str,
        status: Any,
        filled_size: float = 0.0,
        avg_price: float = 0.0,
        local_id: Optional[str] = None,
    ):
        """Standardized method to notify the system (StateTracker) of an order update."""
        if self.order_callback:
            await self.order_callback(
                exchange_id, status, filled_size, avg_price, local_id
            )

    def _safe_float(self, value, default: float = 0.0) -> float:
        """Standardized float conversion with safety handling."""
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

    def _fast_json_encode(self, data: Any) -> str:
        # orjson dumps to bytes. fast and correct.
        # We decode to utf-8 str because some libs (like thalex py) expect str.
        return orjson.dumps(data).decode("utf-8")

    def _fast_json_decode(self, data: Union[str, bytes]) -> Any:
        return orjson.loads(data)

    @property
    def is_ready(self) -> bool:
        """Check if the adapter is connected and not currently reconnecting."""
        return self.connected and not self.is_reconnecting
