import numpy as np
import time
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime
import logging
from .fast_ringbuffer import FastRingBuffer
from ..thalex_logging import LoggerFactory

class VolumeCandle:
    """Simple container for volume-based candle data"""
    def __init__(self):
        self.open_price = 0.0
        self.high_price = 0.0
        self.low_price = float('inf')
        self.close_price = 0.0
        self.volume = 0.0
        self.buy_volume = 0.0
        self.sell_volume = 0.0
        self.volume_delta = 0.0  # buy_volume - sell_volume
        self.delta_ratio = 0.0   # normalized delta (-1 to 1)
        self.trade_count = 0
        self.start_time = 0
        self.end_time = 0
        self.is_complete = False

    def update(self, price: float, volume: float, is_buy: bool, timestamp: int) -> None:
        """Update candle with new trade data"""
        # Update prices
        if self.trade_count == 0:
            self.open_price = price
            self.high_price = price
            self.low_price = price
            self.start_time = timestamp
        else:
            self.high_price = max(self.high_price, price)
            self.low_price = min(self.low_price, price)
        
        self.close_price = price
        self.end_time = timestamp
        
        # Update volumes
        self.volume += volume
        if is_buy:
            self.buy_volume += volume
        else:
            self.sell_volume += volume
        
        # Update volume delta
        self.volume_delta = self.buy_volume - self.sell_volume
        if self.volume > 0:
            self.delta_ratio = self.volume_delta / self.volume
        
        self.trade_count += 1

    def __repr__(self) -> str:
        return (f"VolumeCandle(O:{self.open_price:.2f}, H:{self.high_price:.2f}, "
                f"L:{self.low_price:.2f}, C:{self.close_price:.2f}, V:{self.volume:.4f}, "
                f"Δ:{self.volume_delta:.4f}, Ratio:{self.delta_ratio:.2f})")


class VolumeBasedCandleBuffer:
    """Buffer for volume-based candles with predictive indicators"""
    
    def __init__(
        self,
        volume_threshold: float = 1.0,     # Volume required to complete a candle
        max_candles: int = 100,            # Maximum candles to store
        max_time_seconds: int = 300,       # Maximum time before forcing candle close
        ema_periods: Dict[str, int] = None # EMA periods for tracking
    ):
        # Configure logging
        self.logger = LoggerFactory.configure_component_logger(
            "volume_candle_buffer",
            log_file="volume_candles.log",
            high_frequency=True
        )
        self.logger.info(f"Initializing volume candle buffer: threshold={volume_threshold}, "
                         f"max_candles={max_candles}, max_time={max_time_seconds}s")
        
        # Configuration
        self.volume_threshold = volume_threshold
        self.max_time_seconds = max_time_seconds
        
        # Current candle
        self.current_candle = VolumeCandle()
        
        # Completed candles buffer
        self.candles = []
        self.max_candles = max_candles
        
        # Technical indicators
        self.ema_periods = ema_periods or {"fast": 8, "med": 21, "slow": 55}
        self.ema_values = {name: 0.0 for name in self.ema_periods.keys()}
        
        # Delta indicators
        self.delta_ema = {name: 0.0 for name in self.ema_periods.keys()}
        self.cumulative_delta = 0.0
        
        # Predictive signals
        self.signals = {
            "momentum": 0.0,     # -1 to 1, direction and strength
            "reversal": 0.0,     # 0 to 1, probability of reversal
            "volatility": 0.0,   # 0 to 1, expected volatility increase
            "exhaustion": 0.0    # 0 to 1, sign of buying/selling exhaustion
        }
        
        # Performance tracking
        self._last_update_time = 0
        self._prediction_accuracy = []
        
        # Statistics
        self.updates_count = 0
        self.candles_completed = 0
        self.predictions_made = 0
        self.last_signal_strength = 0.0

    def update(self, price: float, volume: float, is_buy: bool, timestamp: int = None) -> Optional[VolumeCandle]:
        """
        Update with new trade data. Returns completed candle if applicable.
        
        Args:
            price: Trade price
            volume: Trade volume
            is_buy: Whether it was a buy trade
            timestamp: Unix timestamp in milliseconds
            
        Returns:
            Completed VolumeCandle if one was finished, None otherwise
        """
        # Set timestamp if not provided
        if timestamp is None:
            timestamp = int(time.time() * 1000)
        
        self.updates_count += 1
        
        # Debug log every 100th update to avoid excessive logging
        if self.updates_count % 100 == 0:
            self.logger.debug(f"Volume candle update #{self.updates_count}: price={price:.2f}, "
                          f"volume={volume:.6f}, is_buy={is_buy}, current_candle_volume={self.current_candle.volume:.6f}")
            
        # Update current candle
        self.current_candle.update(price, volume, is_buy, timestamp)
        
        # Check if candle should be completed
        completed_candle = None
        if (self.current_candle.volume >= self.volume_threshold or 
                (timestamp - self.current_candle.start_time) >= self.max_time_seconds * 1000):
            
            # Mark as complete
            self.current_candle.is_complete = True
            completed_candle = self.current_candle
            self.candles_completed += 1
            
            # Add to buffer
            self.candles.append(self.current_candle)
            if len(self.candles) > self.max_candles:
                self.candles.pop(0)
                
            # Update indicators
            self._update_indicators(completed_candle)
            
            # Calculate predictive signals
            self._calculate_signals()
            
            # Log candle completion with detailed info
            reason = "volume threshold reached" if completed_candle.volume >= self.volume_threshold else "time limit reached"
            duration_seconds = (completed_candle.end_time - completed_candle.start_time) / 1000
            
            self.logger.info(
                f"Volume candle #{self.candles_completed} completed ({reason}): "
                f"O:{completed_candle.open_price:.2f}, H:{completed_candle.high_price:.2f}, "
                f"L:{completed_candle.low_price:.2f}, C:{completed_candle.close_price:.2f}, "
                f"V:{completed_candle.volume:.6f}, Δ:{completed_candle.delta_ratio:.2f}, "
                f"Trades:{completed_candle.trade_count}, Duration:{duration_seconds:.1f}s"
            )
            
            # Reset current candle
            self.current_candle = VolumeCandle()
            
        self._last_update_time = timestamp
        return completed_candle
        
    def _update_indicators(self, candle: VolumeCandle) -> None:
        """Update technical indicators with new candle"""
        # Update price EMAs
        for name, period in self.ema_periods.items():
            alpha = 2.0 / (period + 1)
            
            # Initialize if first value
            if self.ema_values[name] == 0.0:
                self.ema_values[name] = candle.close_price
                self.delta_ema[name] = candle.delta_ratio
            else:
                # Update EMAs
                self.ema_values[name] = candle.close_price * alpha + self.ema_values[name] * (1 - alpha)
                self.delta_ema[name] = candle.delta_ratio * alpha + self.delta_ema[name] * (1 - alpha)
        
        # Update cumulative delta
        self.cumulative_delta += candle.volume_delta
        
        # Log indicator updates
        self.logger.debug(
            f"Updated indicators: EMAs={', '.join([f'{k}:{v:.2f}' for k, v in self.ema_values.items()])}, "
            f"Delta_EMAs={', '.join([f'{k}:{v:.2f}' for k, v in self.delta_ema.items()])}, "
            f"Cumulative_Delta={self.cumulative_delta:.6f}"
        )
        
    def _calculate_signals(self) -> None:
        """Calculate predictive signals from candle data"""
        if len(self.candles) < 5:
            return
        
        self.predictions_made += 1
            
        # Get recent candles
        recent = self.candles[-5:]
        
        # Save old signals for logging change
        old_signals = self.signals.copy()
        
        # 1. Momentum Signal - combines price trend and volume delta
        price_momentum = (recent[-1].close_price / recent[0].open_price) - 1.0
        delta_momentum = np.mean([c.delta_ratio for c in recent])
        
        # Combine price and volume momentum (volume confirming price or diverging)
        self.signals["momentum"] = np.clip(
            price_momentum * 20 * np.sign(delta_momentum) * min(1.0, abs(delta_momentum) * 2),
            -1.0, 1.0
        )
        
        # 2. Reversal Signal - looks for divergence between price and volume delta
        price_direction = np.sign(recent[-1].close_price - recent[0].open_price)
        delta_direction = np.sign(self.delta_ema["fast"] - self.delta_ema["slow"])
        
        # Potential reversal if directions differ
        if price_direction != 0 and delta_direction != 0 and price_direction != delta_direction:
            # Strength of divergence
            price_change = abs(recent[-1].close_price / recent[0].open_price - 1.0)
            delta_change = abs(self.delta_ema["fast"] - self.delta_ema["slow"])
            
            self.signals["reversal"] = min(1.0, price_change * 10) * min(1.0, delta_change * 5)
        else:
            self.signals["reversal"] = 0.0
        
        # 3. Volatility Signal - predicts increasing volatility
        price_range = np.mean([c.high_price/c.low_price - 1.0 for c in recent])
        volume_variability = np.std([c.volume for c in recent]) / np.mean([c.volume for c in recent])
        delta_variability = np.std([c.delta_ratio for c in recent])
        
        self.signals["volatility"] = min(1.0, (price_range * 10 + volume_variability + delta_variability) / 3)
        
        # 4. Exhaustion Signal - detects potential buying/selling exhaustion
        # Delta dropping while price continues in same direction
        recent_deltas = [c.delta_ratio for c in recent]
        delta_trend = recent_deltas[-1] - recent_deltas[0]
        
        if (price_direction > 0 and delta_trend < -0.2) or (price_direction < 0 and delta_trend > 0.2):
            self.signals["exhaustion"] = min(1.0, abs(delta_trend) * 2)
        else:
            self.signals["exhaustion"] = 0.0
            
        # Calculate overall signal strength for logging
        self.last_signal_strength = abs(self.signals["momentum"]) + self.signals["reversal"] + \
                                 self.signals["volatility"] + self.signals["exhaustion"]
        
        # Log signal changes if they're significant
        signal_change = sum(abs(self.signals[k] - old_signals[k]) for k in self.signals.keys())
        
        # Always log on significant changes or periodically
        if signal_change > 0.2 or self.predictions_made % 5 == 0:
            self.logger.info(
                f"Prediction #{self.predictions_made} - Signals: "
                f"Momentum={self.signals['momentum']:.2f} ({old_signals['momentum']:.2f}), "
                f"Reversal={self.signals['reversal']:.2f} ({old_signals['reversal']:.2f}), "
                f"Volatility={self.signals['volatility']:.2f} ({old_signals['volatility']:.2f}), "
                f"Exhaustion={self.signals['exhaustion']:.2f} ({old_signals['exhaustion']:.2f}), "
                f"Strength={self.last_signal_strength:.2f}"
            )
    
    def get_predicted_parameters(self) -> Dict[str, float]:
        """
        Generate predicted parameters for the Avellaneda model based on signals
        
        Returns:
            Dictionary of predicted parameters for the Avellaneda model
        """
        if len(self.candles) < 5:
            return {
                "gamma_adjustment": 0.0,
                "kappa_adjustment": 0.0,
                "reservation_price_offset": 0.0,
                "trend_direction": 0,
                "volatility_adjustment": 0.0
            }
        
        # Fetch signals
        momentum = self.signals["momentum"]
        reversal = self.signals["reversal"]
        volatility = self.signals["volatility"]
        exhaustion = self.signals["exhaustion"]
        
        # 1. Gamma adjustment (risk aversion)
        # Increase gamma when volatility expected to rise or around reversal points
        gamma_adj = volatility * 0.4 + reversal * 0.3
        
        # 2. Kappa adjustment (market depth)
        # Reduce market depth parameter when market might be thin (high volatility)
        kappa_adj = -volatility * 0.3
        
        # 3. Reservation price offset (predictive skew)
        # Skew reservation price based on momentum and potential reversals
        reservation_offset = 0.0
        if abs(momentum) > 0.2:
            # Strong momentum - skew in direction of momentum
            if exhaustion < 0.5:  # Not showing exhaustion yet
                reservation_offset = momentum * 0.0003  # Small adjustment
        elif reversal > 0.6:
            # Strong reversal signal - skew against recent price direction
            last_candle = self.candles[-1]
            prev_candle = self.candles[-2]
            price_direction = np.sign(last_candle.close_price - prev_candle.close_price)
            reservation_offset = -price_direction * reversal * 0.0005
        
        # 4. Trend direction (-1, 0, 1)
        trend_direction = 0
        if abs(momentum) > 0.3:
            trend_direction = np.sign(momentum)
        elif reversal > 0.7:
            # Reversal signal strong enough to indicate trend change
            last_price_direction = np.sign(self.candles[-1].close_price - self.candles[-2].close_price)
            trend_direction = -last_price_direction
        
        # 5. Volatility adjustment
        vol_adj = volatility * 0.2
        
        # Log parameter predictions
        predictions = {
            "gamma_adjustment": gamma_adj,
            "kappa_adjustment": kappa_adj,
            "reservation_price_offset": reservation_offset,
            "trend_direction": trend_direction,
            "volatility_adjustment": vol_adj
        }
        
        # Log prediction with detailed explanation
        explanation = []
        if abs(gamma_adj) > 0.05:
            explanation.append(f"γ adjusted by {gamma_adj:.2f} due to volatility({volatility:.2f})/reversal({reversal:.2f})")
        if abs(kappa_adj) > 0.05:
            explanation.append(f"κ adjusted by {kappa_adj:.2f} due to volatility({volatility:.2f})")
        if abs(reservation_offset) > 0.00005:
            explanation.append(f"reservation price offset {reservation_offset:.6f} due to momentum({momentum:.2f})/reversal({reversal:.2f})")
        if trend_direction != 0:
            explanation.append(f"trend direction {trend_direction} due to momentum({momentum:.2f})/reversal({reversal:.2f})")
        if abs(vol_adj) > 0.05:
            explanation.append(f"volatility adjusted by {vol_adj:.2f}")
            
        if explanation:
            self.logger.info(f"Avellaneda parameter predictions: {', '.join(explanation)}")
        
        return predictions
    
    def get_signal_metrics(self) -> Dict[str, float]:
        """Get current signal metrics for debugging/monitoring"""
        return {
            "signals": self.signals,
            "ema_price": {k: v for k, v in self.ema_values.items()},
            "delta_ema": {k: v for k, v in self.delta_ema.items()},
            "cumulative_delta": self.cumulative_delta,
            "candle_count": len(self.candles),
            "current_volume": self.current_candle.volume if self.current_candle else 0
        }
        
    def evaluate_prediction_accuracy(self, actual_price: float, prediction_horizon: int = 5) -> float:
        """
        Evaluate prediction accuracy by comparing predicted direction with actual outcome
        
        Args:
            actual_price: The current actual price to compare with past predictions
            prediction_horizon: How many candles back to check predictions
            
        Returns:
            Accuracy score (0-1)
        """
        if len(self._prediction_accuracy) > 20:
            self._prediction_accuracy.pop(0)
            
        if len(self.candles) <= prediction_horizon:
            return 0.0
            
        # Get prediction made N candles ago
        old_prediction = self.candles[-prediction_horizon].close_price
        pred_direction = self.signals["momentum"] * prediction_horizon
        
        # Calculate if prediction was correct
        actual_change = (actual_price / old_prediction) - 1.0
        prediction_correct = np.sign(pred_direction) == np.sign(actual_change)
        
        # Record accuracy
        self._prediction_accuracy.append(1.0 if prediction_correct else 0.0)
        
        # Log accuracy
        avg_accuracy = np.mean(self._prediction_accuracy)
        self.logger.debug(f"Prediction accuracy: {avg_accuracy:.2f} (last prediction: {'correct' if prediction_correct else 'incorrect'})")
        
        # Return rolling accuracy
        return avg_accuracy 