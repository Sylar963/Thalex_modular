import logging
import time
from typing import Dict, Optional, List
from ..interfaces import StorageGateway

logger = logging.getLogger(__name__)


class HistoricalTrendService:
    """
    Service to calculate long-term trends from historical data stored in TimescaleDB.
    """

    def __init__(self, storage: StorageGateway):
        self.storage = storage
        self._trend_cache: Dict[str, float] = {}
        self._last_update: Dict[str, float] = {}
        self._last_warning: Dict[str, float] = {}
        self.update_interval = 3600

    async def get_trend_14d(self, symbol: str) -> float:
        """
        Calculates the 14-day trend: (Price_now - Price_14d) / Price_14d.
        Returns the percentage change as a float (e.g., 0.05 for 5%).
        """
        now = time.time()
        if (
            symbol in self._trend_cache
            and (now - self._last_update.get(symbol, 0)) < self.update_interval
        ):
            return self._trend_cache[symbol]

        try:
            # Fetch OHLCV for the last 15 days (to ensure we have a start point 14 days ago)
            start_ts = now - (15 * 24 * 3600)
            history = await self.storage.get_history(
                symbol, start_ts, now, resolution="1h"
            )

            if not history or len(history) < 24 * 14:
                if now - self._last_warning.get(symbol, 0) > 3600:
                    logger.warning(
                        f"Insufficient history for 14d trend calculation for {symbol} ({len(history) if history else 0} bars, need {24 * 14})"
                    )
                    self._last_warning[symbol] = now
                return 0.0

            # Get price from 14 days ago (approximate by index or searching timestamp)
            target_ts = now - (14 * 24 * 3600)
            price_14d = None
            for bar in history:
                if bar["time"] >= target_ts:
                    price_14d = bar["close"]
                    break

            if price_14d is None:
                price_14d = history[0]["close"]

            current_price = history[-1]["close"]
            trend = (current_price - price_14d) / price_14d

            self._trend_cache[symbol] = trend
            self._last_update[symbol] = now

            logger.info(
                f"Updated 14d trend for {symbol}: {trend:.4f} (Price 14d ago: {price_14d:.2f}, Current: {current_price:.2f})"
            )
            return trend

        except Exception as e:
            logger.error(f"Error calculating 14d trend for {symbol}: {e}")
            return 0.0

    def get_trend_side(self, trend_value: float) -> str:
        """Categorize trend into UP, DOWN, or FLAT."""
        if trend_value > 0.01:
            return "UP"
        elif trend_value < -0.01:
            return "DOWN"
        return "FLAT"
