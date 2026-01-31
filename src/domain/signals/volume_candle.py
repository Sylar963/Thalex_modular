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

        recent = list(self.candles)[-3:]

        # 1. Momentum
        price_momentum = (recent[-1].close / recent[0].open) - 1.0
        delta_momentum = np.mean([c.delta_ratio for c in recent])
        self.signals["momentum"] = np.clip(
            price_momentum * 100 * np.sign(delta_momentum), -1, 1
        )

        # 2. Reversal (Simple divergence)
        price_dir = np.sign(recent[-1].close - recent[0].open)
        delta_dir = np.sign(self.delta_ema["fast"] - self.delta_ema["slow"])
        self.signals["reversal"] = (
            1.0
            if (price_dir != 0 and delta_dir != 0 and price_dir != delta_dir)
            else 0.0
        )

        # 3. Volatility
        price_range = np.mean([c.high / c.low - 1.0 for c in recent])
        self.signals["volatility"] = min(1.0, price_range * 50)

        # 4. Parameter Adjustments (Simple Logic)
        self.signals["gamma_adjustment"] = self.signals["volatility"] * 0.4
        self.signals["volatility_adjustment"] = self.signals["volatility"] * 0.4
        self.signals["reservation_price_offset"] = self.signals["momentum"] * 0.0003
