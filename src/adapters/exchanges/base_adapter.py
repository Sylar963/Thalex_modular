from abc import abstractmethod
from typing import Optional, Callable, Any, Union
import time
import asyncio

try:
    import orjson
except ImportError:
    import json as orjson  # Fallback for type hinting/mocking, though functionality differs

from ...domain.interfaces import ExchangeGateway


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

    async def consume_wait(self, tokens: int = 1) -> bool:
        while True:
            now = time.time()
            delta = now - self.last_update
            self._tokens = min(self.capacity, self._tokens + delta * self.fill_rate)
            self.last_update = now

            if self._tokens >= tokens:
                self._tokens -= tokens
                return True

            wait_time = (tokens - self._tokens) / self.fill_rate
            await asyncio.sleep(min(wait_time, 0.1))


class BaseExchangeAdapter(ExchangeGateway):
    def __init__(self, api_key: str, api_secret: str, testnet: bool = True):
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        self.connected = False

        self.ticker_callback: Optional[Callable] = None
        self.trade_callback: Optional[Callable] = None
        self.order_callback: Optional[Callable] = None
        self.position_callback: Optional[Callable] = None
        self._time_offset_ms: int = 0
        self._last_sync_time: float = 0

    def _get_timestamp(self) -> int:
        """Standardized method to get exchange-aligned timestamp in milliseconds."""
        return int(time.time() * 1000) + self._time_offset_ms

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
