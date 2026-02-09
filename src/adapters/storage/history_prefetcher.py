import time
import logging
from typing import Optional

from src.domain.interfaces import IHistoryPrefetchService
from src.domain.entities.history import HistoryConfig
from src.adapters.storage.timescale_adapter import TimescaleDBAdapter
from src.adapters.storage.bybit_history_adapter import BybitHistoryAdapter

logger = logging.getLogger(__name__)


class BybitHistoryPrefetcher(IHistoryPrefetchService):
    __slots__ = ("_provider", "_lookback_days", "_venue")

    def __init__(
        self,
        db_adapter: TimescaleDBAdapter,
        lookback_days: int = 15,
        venue: str = "bybit",
    ):
        self._provider = BybitHistoryAdapter(db_adapter)
        self._lookback_days = lookback_days
        self._venue = venue

    async def prefetch(self, symbol: str) -> int:
        now = time.time()
        start_ts = now - (self._lookback_days * 24 * 3600)
        config = HistoryConfig(
            symbol=symbol, venue=self._venue, start_time=start_ts, end_time=now
        )
        logger.info(f"Starting prefetch for {symbol} ({self._lookback_days} days)...")
        count = await self._provider.fetch_and_persist(config)
        logger.info(f"Prefetch complete for {symbol}: {count} bars fetched.")
        return count

    async def close(self) -> None:
        await self._provider.close()
