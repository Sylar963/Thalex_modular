import logging
import time
import numpy as np
from typing import Dict, Optional, Deque, List
from collections import deque
from ..interfaces import SignalEngine
from ..entities import Ticker

logger = logging.getLogger(__name__)


class FairPriceService(SignalEngine):
    """
    Lead-Lag Fair Price Engine (Generic Oracle).

    Calculates a 'Fair Price' for a Target instrument based on a Reference (Oracle) instrument.
    Strategy:
        1. Maintain a rolling window of Offset = (Target_Mid - Oracle_Mid).
        2. Calculate the Median Offset (robust to outliers).
        3. FairPrice = Oracle_Mid + Median_Offset.

    This allows the Target price to 'jump' immediately when the Oracle moves,
    even if the Target order book hasn't updated yet.
    """

    def __init__(
        self,
        oracle_symbol: str,
        target_symbol: str,
        window_duration: int = 300,
        min_samples: int = 10,
    ):
        """
        Args:
            oracle_symbol: The symbol acting as the leader (e.g., BTCUSDT).
            target_symbol: The symbol being traded (e.g., BTC-PERPETUAL).
            window_duration: Window size in seconds for offset calculation.
            min_samples: Minimum samples required to broadcast a valid fair price.
        """
        self.oracle_symbol = oracle_symbol
        self.target_symbol = target_symbol
        self.window_duration = window_duration
        self.min_samples = min_samples

        # State
        self.oracle_price: float = 0.0
        self.target_price: float = 0.0
        self.last_oracle_time: float = 0.0
        self.last_target_time: float = 0.0

        # Rolling window of (timestamp, offset)
        self.offset_window: Deque[Dict] = deque()

        # Latest computed fair price
        self.fair_price: Optional[float] = None
        self.median_offset: Optional[float] = None

    def update_oracle(self, price: float, timestamp: float):
        """Update from Reference Exchange."""
        self.oracle_price = price
        self.last_oracle_time = timestamp
        self._recalc()

    def update_target(self, price: float, timestamp: float):
        """Update from Target Exchange."""
        self.target_price = price
        self.last_target_time = timestamp
        self._add_sample(timestamp)
        self._recalc()

    def _add_sample(self, timestamp: float):
        """Record the current spread (Alpha) between Target and Oracle."""
        if self.oracle_price > 0 and self.target_price > 0:
            # We want to capture the "Stable" difference.
            # Ideally sampled when market is quiet, but median filter handles noise.
            offset = self.target_price - self.oracle_price
            self.offset_window.append({"ts": timestamp, "offset": offset})

            # Prune old samples
            cutoff = timestamp - self.window_duration
            while self.offset_window and self.offset_window[0]["ts"] < cutoff:
                self.offset_window.popleft()

    def _recalc(self):
        """Update the Fair Price Estimate."""
        if len(self.offset_window) < self.min_samples:
            self.fair_price = None
            return

        if self.oracle_price <= 0:
            self.fair_price = None
            return

        # Calculate Median Offset
        offsets = [s["offset"] for s in self.offset_window]
        self.median_offset = float(np.median(offsets))

        # Fair Price = Oracle + Historical_Premium
        self.fair_price = self.oracle_price + self.median_offset

    def get_signal(self) -> Dict:
        """Return the current Fair Price and Metadata."""
        return {
            "fair_price": self.fair_price,
            "oracle_price": self.oracle_price,
            "median_offset": self.median_offset,
            "samples": len(self.offset_window),
            "ready": self.fair_price is not None,
        }

    # SignalEngine Interface (Optional compliance)
    def update(self, ticker: Ticker):
        # This is a bit complex because we need TWO inputs.
        # This Service likely needs to be fed explicitly by StrategyManager
        # or wired to both Adapters.
        pass

    def get_signals(self) -> Dict[str, float]:
        if self.fair_price:
            return {"fair_price": self.fair_price}
        return {}
