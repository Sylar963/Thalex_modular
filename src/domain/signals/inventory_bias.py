import logging
from typing import Dict, Optional
from ..interfaces import SignalEngine
from ..entities import Ticker, Trade

logger = logging.getLogger(__name__)


class InventoryBiasEngine(SignalEngine):
    __slots__ = (
        "_or_weight",
        "_vamp_weight",
        "_suppression_threshold",
        "_current_position",
        "_or_signals",
        "_vamp_signals",
        "_suppress_bids",
        "_suppress_asks",
        "_bias_direction",
        "_last_log_time",
    )

    def __init__(
        self,
        or_weight: float = 0.4,
        vamp_weight: float = 0.6,
        suppression_threshold: float = 0.3,
    ):
        self._or_weight = or_weight
        self._vamp_weight = vamp_weight
        self._suppression_threshold = suppression_threshold

        self._current_position: float = 0.0
        self._or_signals: Dict[str, float] = {}
        self._vamp_signals: Dict[str, float] = {}

        self._suppress_bids: float = 0.0
        self._suppress_asks: float = 0.0
        self._bias_direction: float = 0.0
        self._last_log_time: float = 0.0

    def update(self, ticker: Ticker) -> None:
        pass

    def update_trade(self, trade: Trade) -> None:
        pass

    def update_position(self, position_size: float) -> None:
        self._current_position = position_size

    def update_signals(
        self,
        or_signals: Optional[Dict[str, float]] = None,
        vamp_signals: Optional[Dict[str, float]] = None,
    ) -> None:
        if or_signals:
            self._or_signals = or_signals
        if vamp_signals:
            self._vamp_signals = vamp_signals

        self._calculate_bias()

    def get_signals(self, symbol: Optional[str] = None) -> Dict[str, float]:
        return {
            "suppress_bids": self._suppress_bids,
            "suppress_asks": self._suppress_asks,
            "bias_direction": self._bias_direction,
        }

    def _calculate_bias(self) -> None:
        or_trend = self._calculate_or_trend()
        vamp_impact = self._vamp_signals.get("market_impact", 0.0)

        combined_bias = (or_trend * self._or_weight) + (vamp_impact * self._vamp_weight)
        self._bias_direction = max(-1.0, min(1.0, combined_bias))

        self._suppress_bids = 0.0
        self._suppress_asks = 0.0

        pos = self._current_position
        threshold = self._suppression_threshold

        if pos > 0:
            if self._bias_direction < -threshold:
                suppression_strength = min(1.0, 0.5 + abs(self._bias_direction))
                self._suppress_bids = suppression_strength
                logger.debug(
                    f"LONG pos + bearish bias ({self._bias_direction:.2f}): "
                    f"suppress_bids={self._suppress_bids:.2f}"
                )
        elif pos < 0:
            if self._bias_direction > threshold:
                suppression_strength = min(1.0, 0.5 + self._bias_direction)
                self._suppress_asks = suppression_strength
                logger.debug(
                    f"SHORT pos + bullish bias ({self._bias_direction:.2f}): "
                    f"suppress_asks={self._suppress_asks:.2f}"
                )

        import time

        now = time.time()
        if now - self._last_log_time > 30:
            if self._suppress_bids > 0 or self._suppress_asks > 0:
                logger.info(
                    f"InventoryBias: pos={pos:.4f}, bias={self._bias_direction:.2f}, "
                    f"suppress_bids={self._suppress_bids:.2f}, suppress_asks={self._suppress_asks:.2f}"
                )
            self._last_log_time = now

    def _calculate_or_trend(self) -> float:
        day_dir = self._or_signals.get("day_dir", 0.0)
        breakout = self._or_signals.get("breakout_signal", 0.0)
        orm = self._or_signals.get("orm", 0.0)
        current_price = self._or_signals.get("current_price", 0.0)

        or_trend = 0.0

        if day_dir != 0:
            or_trend += day_dir * 0.4

        if breakout != 0:
            or_trend += breakout * 0.4

        if orm > 0 and current_price > 0:
            if current_price > orm:
                or_trend += 0.2
            elif current_price < orm:
                or_trend -= 0.2

        return max(-1.0, min(1.0, or_trend))
