from abc import abstractmethod
from typing import Optional, Callable
import time
import asyncio
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
