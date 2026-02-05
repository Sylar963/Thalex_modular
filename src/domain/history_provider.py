from abc import ABC, abstractmethod
from typing import List, Dict, Any, AsyncGenerator
from .entities.history import HistoryConfig
from .entities import Ticker, Trade


class IHistoryProvider(ABC):
    @abstractmethod
    async def get_tickers(self, config: HistoryConfig) -> AsyncGenerator[Ticker, None]:
        pass

    @abstractmethod
    async def get_trades(self, config: HistoryConfig) -> AsyncGenerator[Trade, None]:
        pass

    @abstractmethod
    async def fetch_and_persist(self, config: HistoryConfig) -> int:
        pass
