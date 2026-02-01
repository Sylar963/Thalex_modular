from dataclasses import dataclass, field
from typing import Dict, Optional, List
import numpy as np
import logging

from ..entities import MarketState, Ticker

logger = logging.getLogger(__name__)


@dataclass
class MarketRegime:
    name: str  # "Quiet", "Volatile", "Trending", "Illiquid"
    trading_mode: str = "market_making"  # "market_making", "rescue"
    volatility_score: float = 0.0
    trend_score: float = 0.0
    liquidity_score: float = 0.0
    expected_move: float = 0.0
    is_overpriced: bool = False  # If IV > RV


class RegimeDetector:
    """
    Analyzes MarketState to classify the current market regime.
    """

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.price_history: List[float] = []
        self.current_regime = MarketRegime("Quiet")

    def update(self, market_state: MarketState) -> MarketRegime:
        if not market_state.ticker:
            return self.current_regime

        price = market_state.ticker.mid_price
        self.price_history.append(price)

        if len(self.price_history) > self.window_size:
            self.price_history.pop(0)

        self.current_regime = self._detect_regime(market_state)
        return self.current_regime

    def set_expected_move(self, em: float, is_overpriced: bool):
        self.current_regime.expected_move = em
        self.current_regime.is_overpriced = is_overpriced

    def _detect_regime(self, market_state: MarketState) -> MarketRegime:
        if len(self.price_history) < 10:
            return self.current_regime

        # 1. Volatility (Std Dev of returns) - Realized Volatility (RV)
        returns = np.diff(self.price_history) / self.price_history[:-1]
        rv = np.std(returns) * np.sqrt(365 * 24 * 60) * 100  # Annualized RV approx

        # 2. Trend
        trend = (self.price_history[-1] - self.price_history[0]) / self.price_history[0]

        # 3. Liquidity
        ticker = market_state.ticker
        spread_bps = (ticker.ask - ticker.bid) / ticker.mid_price * 10000
        liquidity_score = 1.0 / (spread_bps + 1e-6)

        # Classification Logic
        name = "Quiet"
        if abs(trend) > 0.02:  # Trending if > 2% move in window
            name = "Trending"
        elif rv > 40.0:  # Volatile if RV > 40%
            name = "Volatile"
        elif spread_bps > 20:
            name = "Illiquid"

        # Edge cases for overpriced/underpriced relative to EM
        if self.current_regime.is_overpriced:
            # If options are overpriced, we might want to be more aggressive in MM
            pass

        return MarketRegime(
            name=name,
            volatility_score=rv,
            trend_score=trend,
            liquidity_score=liquidity_score,
            expected_move=self.current_regime.expected_move,
            is_overpriced=self.current_regime.is_overpriced,
        )

    def _calculate_expected_move(self, market_state: MarketState) -> float:
        """
        Calculates the expected move using ATM Straddle price from options data.

        Logic to be implemented based on user instructions:
        - Fetch ATM straddle for options > 3 days expiry
        - Expected Move = Straddle Price * 0.85 (approx)
        """
        # TODO: Implement option data fetching and calculation
        return 0.0
