import numpy as np
import time
from typing import Dict, List, Optional, Deque
from collections import deque
from dataclasses import dataclass, field
from ..interfaces import SignalEngine
from ..entities import Ticker, Trade, OrderSide


@dataclass
class VolumeCandle:
    open: float = 0.0
    high: float = 0.0
    low: float = float("inf")
    close: float = 0.0
    volume: float = 0.0
    buy_volume: float = 0.0
    sell_volume: float = 0.0
    start_time: float = 0.0
    end_time: float = 0.0
    trade_count: int = 0

    @property
    def delta(self) -> float:
        return self.buy_volume - self.sell_volume

    @property
    def delta_ratio(self) -> float:
        if self.volume == 0:
            return 0.0
        return self.delta / self.volume


class VolumeCandleSignalEngine(SignalEngine):
    """
    Signal Engine based on Volume Candles with predictive indicators.
    """

    def __init__(self, volume_threshold: float = 1.0, max_candles: int = 100):
        self.volume_threshold = volume_threshold
        self.max_candles = max_candles

        self.current_candle = VolumeCandle()
        self.candles: Deque[VolumeCandle] = deque(maxlen=max_candles)

        self.signals: Dict[str, float] = {
            "momentum": 0.0,
            "reversal": 0.0,
            "volatility": 0.0,
            "exhaustion": 0.0,
            "gamma_adjustment": 0.0,
            "reservation_price_offset": 0.0,
            "volatility_adjustment": 0.0,
        }

        # EMA State
        self.ema_periods = {"fast": 8, "med": 21, "slow": 55}
        self.ema_values = {name: 0.0 for name in self.ema_periods}
        self.delta_ema = {name: 0.0 for name in self.ema_periods}

    def update(self, ticker: Ticker):
        # Ticker updates might be used for mark price reference, but
        # this engine is driven by trades.
        pass

    def update_trade(self, trade: Trade):
        """Process a new trade and update candles/signals."""
        # 1. Update current candle
        if self.current_candle.trade_count == 0:
            self.current_candle.open = trade.price
            self.current_candle.high = trade.price
            self.current_candle.low = trade.price
            self.current_candle.start_time = trade.timestamp
        else:
            self.current_candle.high = max(self.current_candle.high, trade.price)
            self.current_candle.low = min(self.current_candle.low, trade.price)

        self.current_candle.close = trade.price
        self.current_candle.end_time = trade.timestamp
        self.current_candle.volume += trade.size
        self.current_candle.trade_count += 1

        if trade.side == OrderSide.BUY:
            self.current_candle.buy_volume += trade.size
        else:
            self.current_candle.sell_volume += trade.size

        # 2. Check for completion
        if self.current_candle.volume >= self.volume_threshold:
            self._complete_candle()

    def get_signals(self) -> Dict[str, float]:
        return self.signals.copy()

    def _complete_candle(self):
        self.candles.append(self.current_candle)
        self._update_indicators(self.current_candle)
        self._calculate_signals()

        # Reset current candle
        self.current_candle = VolumeCandle()

    def _update_indicators(self, candle: VolumeCandle):
        for name, period in self.ema_periods.items():
            alpha = 2.0 / (period + 1)
            if self.ema_values[name] == 0.0:
                self.ema_values[name] = candle.close
                self.delta_ema[name] = candle.delta_ratio
            else:
                self.ema_values[name] = candle.close * alpha + self.ema_values[name] * (
                    1 - alpha
                )
                self.delta_ema[name] = candle.delta_ratio * alpha + self.delta_ema[
                    name
                ] * (1 - alpha)

    def _calculate_signals(self):
        if len(self.candles) < 3:
            return

        recent = list(self.candles)[-5:]

        # --- 1. Enhanced VAMP (Volume Adjusted Market Pressure) ---
        # Legacy logic: VAMP = Weighted Avg of Buy/Sell VWAP based on Volume Ratio

        # Calculate Aggressive Volume Sums
        agg_buy_vol = sum(c.buy_volume for c in recent)
        agg_sell_vol = sum(c.sell_volume for c in recent)
        total_vol = agg_buy_vol + agg_sell_vol

        # Calculate VWAPs
        buy_vwap = (
            sum(c.close * c.buy_volume for c in recent) / agg_buy_vol
            if agg_buy_vol > 0
            else 0
        )
        sell_vwap = (
            sum(c.close * c.sell_volume for c in recent) / agg_sell_vol
            if agg_sell_vol > 0
            else 0
        )

        vamp_value = 0.0
        vamp_impact = 0.0

        if total_vol > 0:
            buy_ratio = agg_buy_vol / total_vol
            sell_ratio = 1.0 - buy_ratio

            # Weighted VAMP
            if buy_vwap > 0 and sell_vwap > 0:
                vamp_value = (buy_vwap * buy_ratio) + (sell_vwap * sell_ratio)
            elif buy_vwap > 0:
                vamp_value = buy_vwap
            elif sell_vwap > 0:
                vamp_value = sell_vwap

            # Impact
            vamp_impact = (agg_buy_vol - agg_sell_vol) / total_vol  # -1 to 1

        # --- 2. Market Regime Detection ---
        # Volatility
        price_range = np.mean([c.high / c.low - 1.0 for c in recent])
        volatility = min(1.0, price_range * 50)

        # Trend / Regime
        price_change = (recent[-1].close - recent[0].open) / recent[0].open
        trend_strength = abs(price_change) * 100

        market_regime = "ranging"
        if trend_strength > 0.1:  # Threshold
            market_regime = "trending"

        if volatility > 0.005:  # High vol
            market_regime = "volatile"

        # --- 3. Signal Generation (Parity with Legacy) ---

        self.signals["volatility"] = volatility
        self.signals["market_impact"] = vamp_impact
        # Pass VAMP impact as reservation offset helper
        # Logic: If high buy pressure (impact > 0), skew prices up (higher res price)
        self.signals["reservation_price_offset"] = vamp_impact * 0.0005  # Sensitivity

        # Gamma/Vol adjustments based on Regime
        if market_regime == "volatile":
            self.signals["gamma_adjustment"] = 0.5
            self.signals["volatility_adjustment"] = 0.5
        elif market_regime == "trending":
            self.signals[
                "gamma_adjustment"
            ] = -0.2  # Looser spread to capture trend? Or tighter?
            # Legacy usually widened spread in trend to avoid adverse selection
            self.signals["gamma_adjustment"] = 0.2
        else:
            self.signals["gamma_adjustment"] = 0.0
            self.signals["volatility_adjustment"] = 0.0

        self.signals["vamp_value"] = vamp_value
