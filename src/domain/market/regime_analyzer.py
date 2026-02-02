from dataclasses import dataclass, field
from typing import Dict, Any, List, Deque
from collections import deque
import numpy as np

from ..interfaces import RegimeAnalyzer
from ..entities import Ticker


@dataclass(slots=True)
class RegimeState:
    name: str = "Quiet"
    trading_mode: str = "market_making"
    rv_fast: float = 0.0
    rv_mid: float = 0.0
    rv_slow: float = 0.0
    trend_fast: float = 0.0
    trend_mid: float = 0.0
    trend_slow: float = 0.0
    liquidity_score: float = 1.0
    expected_move_pct: float = 0.0
    atm_iv: float = 0.0
    vol_delta: float = 0.0
    is_overpriced: bool = False


class MultiWindowRegimeAnalyzer(RegimeAnalyzer):
    WINDOW_FAST = 20
    WINDOW_MID = 100
    WINDOW_SLOW = 500

    RV_HIGH_THRESHOLD = 0.50
    RV_LOW_THRESHOLD = 0.20
    TREND_THRESHOLD = 0.015
    VOL_DELTA_OVERPRICED = 0.10

    def __init__(self):
        self._prices_fast: Deque[float] = deque(maxlen=self.WINDOW_FAST)
        self._prices_mid: Deque[float] = deque(maxlen=self.WINDOW_MID)
        self._prices_slow: Deque[float] = deque(maxlen=self.WINDOW_SLOW)
        self._last_ticker: Ticker = None
        self._em_pct: float = 0.0
        self._atm_iv: float = 0.0
        self._state = RegimeState()

    def update(self, ticker: Ticker) -> None:
        if not ticker:
            return

        price = ticker.mid_price
        self._prices_fast.append(price)
        self._prices_mid.append(price)
        self._prices_slow.append(price)
        self._last_ticker = ticker

        self._recalculate()

    def set_option_data(self, em_pct: float, atm_iv: float) -> None:
        self._em_pct = em_pct
        self._atm_iv = atm_iv

    def get_regime(self) -> Dict[str, Any]:
        return {
            "name": self._state.name,
            "trading_mode": self._state.trading_mode,
            "rv_fast": self._state.rv_fast,
            "rv_mid": self._state.rv_mid,
            "rv_slow": self._state.rv_slow,
            "trend_fast": self._state.trend_fast,
            "trend_mid": self._state.trend_mid,
            "trend_slow": self._state.trend_slow,
            "liquidity_score": self._state.liquidity_score,
            "expected_move_pct": self._state.expected_move_pct,
            "atm_iv": self._state.atm_iv,
            "vol_delta": self._state.vol_delta,
            "is_overpriced": self._state.is_overpriced,
        }

    def _recalculate(self) -> None:
        s = self._state
        s.rv_fast = self._calc_rv(self._prices_fast)
        s.rv_mid = self._calc_rv(self._prices_mid)
        s.rv_slow = self._calc_rv(self._prices_slow)
        s.trend_fast = self._calc_trend(self._prices_fast)
        s.trend_mid = self._calc_trend(self._prices_mid)
        s.trend_slow = self._calc_trend(self._prices_slow)

        if self._last_ticker:
            spread_bps = (
                (self._last_ticker.ask - self._last_ticker.bid)
                / self._last_ticker.mid_price
                * 10000
            )
            s.liquidity_score = 1.0 / (spread_bps + 1e-6)

        s.expected_move_pct = self._em_pct
        s.atm_iv = self._atm_iv
        s.vol_delta = self._em_pct - s.rv_mid if s.rv_mid > 0 else 0.0
        s.is_overpriced = s.vol_delta > self.VOL_DELTA_OVERPRICED

        s.name = self._classify_regime(s)

    def _classify_regime(self, s: RegimeState) -> str:
        if s.rv_fast > self.RV_HIGH_THRESHOLD:
            return "Volatile"

        if (
            abs(s.trend_fast) > self.TREND_THRESHOLD
            and abs(s.trend_mid) > self.TREND_THRESHOLD
        ):
            return "Trending"

        if s.liquidity_score < 0.05:
            return "Illiquid"

        if s.is_overpriced:
            return "OverpricedVol"

        return "Quiet"

    def _calc_rv(self, prices: Deque[float]) -> float:
        if len(prices) < 10:
            return 0.0
        arr = np.array(prices)
        returns = np.diff(arr) / arr[:-1]
        return float(np.std(returns) * np.sqrt(365 * 24 * 60) * 100)

    def _calc_trend(self, prices: Deque[float]) -> float:
        if len(prices) < 2:
            return 0.0
        return (prices[-1] - prices[0]) / prices[0]
