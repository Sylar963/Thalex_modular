import logging
import asyncio
import time
from typing import Optional, List
from datetime import datetime, timedelta

from ..adapters.storage.timescale_adapter import TimescaleDBAdapter
from ..adapters.storage.bybit_history_adapter import BybitHistoryAdapter
from ..domain.entities.history import HistoryConfig
from ..domain.entities import Ticker

logger = logging.getLogger(__name__)


class DataIngestor:
    def __init__(self, db_adapter: TimescaleDBAdapter):
        self.db = db_adapter
        self.bybit_adapter = BybitHistoryAdapter(db_adapter)

    async def sync_symbol(
        self, symbol: str, venue: str = "bybit", lookback_days: int = 1
    ):
        """
        Ensures data is continuous up to 'now'.
        Checks DB for latest timestamp. If missing or old, backfills.
        """
        logger.info(f"Starting sync for {venue}:{symbol}")

        # 1. Get latest timestamp from DB
        latest_ts = await self._get_latest_timestamp(symbol)

        now = time.time()
        start_time = latest_ts

        # If no data exists, start from lookback period
        if start_time == 0:
            start_time = now - (lookback_days * 86400)
            logger.info(
                f"No existing data for {symbol}. Starting fresh backfill from {datetime.fromtimestamp(start_time)}"
            )
        else:
            # Add 1ms to avoid duplicate primary key if resolution is high,
            # though kline fetching usually handles start/end inclusive carefully.
            start_time += 1.0
            logger.info(
                f"Found existing data for {symbol} until {datetime.fromtimestamp(latest_ts)}. Gap: {now - latest_ts:.2f}s"
            )

        # If gap is small (e.g. < 1 minute), might skip or just let it run
        if now - start_time < 60:
            logger.info("Data is up to date (gap < 60s).")
            return

        config = HistoryConfig(
            symbol=symbol, venue=venue, start_time=start_time, end_time=now
        )

        try:
            count = await self.bybit_adapter.fetch_and_persist(config)
            logger.info(f"Sync complete for {symbol}. Fetched {count} records.")
        except Exception as e:
            logger.error(f"Failed to sync {symbol}: {e}")

    async def _get_latest_timestamp(self, symbol: str) -> float:
        """
        Query DB for the most recent ticker time.
        """
        try:
            # We use the generic generic get_recent_tickers or a specific MAX query
            # Since get_recent_tickers orders by DESC, the first one is the latest.
            tickers = await self.db.get_recent_tickers(symbol, limit=1)
            if tickers:
                return tickers[0].timestamp
            return 0.0
        except Exception as e:
            logger.error(f"Error checking latest timestamp: {e}")
            return 0.0

    async def close(self):
        await self.bybit_adapter.close()
