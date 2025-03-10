import numpy as np
from collections import deque
from typing import Optional, List, Dict, Tuple
import time

from thalex_py.logs.Thalex_modular.config.market_config import TECHNICAL_PARAMS

class TechnicalAnalysis:
    def __init__(self):
        # Price history
        self.price_history = deque(maxlen=max(
            TECHNICAL_PARAMS["zscore"]["window"],
            TECHNICAL_PARAMS["atr"]["period"],
            TECHNICAL_PARAMS["momentum"]["period"],
            TECHNICAL_PARAMS["volume"]["ma_period"],
            TECHNICAL_PARAMS["trend"]["long_period"]
        ))
        
        # Volume history
        self.volume_history = deque(maxlen=TECHNICAL_PARAMS["volume"]["ma_period"])
        
        # Cache for calculations
        self._cache = {
            "zscore": {"value": 0.0, "time": 0},
            "atr": {"value": 0.0, "time": 0},
            "momentum": {"value": 0.0, "time": 0},
            "volume_ma": {"value": 0.0, "time": 0},
            "trend": {"value": 0.0, "time": 0}
        }
        self.cache_duration = 1.0  # 1 second cache duration

    def update_price(self, price: float, volume: Optional[float] = None):
        """Update price and volume history"""
        self.price_history.append(price)
        if volume is not None:
            self.volume_history.append(volume)

    def calculate_zscore(self, current_price: float) -> float:
        """Calculate Z-score with caching"""
        current_time = time.time()
        if current_time - self._cache["zscore"]["time"] < self.cache_duration:
            return self._cache["zscore"]["value"]

        if len(self.price_history) < TECHNICAL_PARAMS["zscore"]["window"]:
            return 0.0

        try:
            prices = np.array(list(self.price_history))
            mean = np.mean(prices)
            std = np.std(prices)
            
            if std == 0:
                zscore = 0.0
            else:
                zscore = (current_price - mean) / std
                
            self._cache["zscore"] = {"value": zscore, "time": current_time}
            return zscore
            
        except Exception as e:
            print(f"Error calculating zscore: {str(e)}")
            return 0.0

    def calculate_atr(self) -> float:
        """Calculate Average True Range with Wilder's smoothing"""
        current_time = time.time()
        if current_time - self._cache["atr"]["time"] < self.cache_duration:
            return self._cache["atr"]["value"]

        if len(self.price_history) < TECHNICAL_PARAMS["atr"]["period"]:
            return 0.0

        try:
            prices = np.array(list(self.price_history))
            high_low = np.abs(np.diff(prices))
            high_close = np.abs(prices[1:] - prices[:-1])
            true_ranges = np.maximum(high_low, high_close)
            
            # Apply Wilder's smoothing
            smoothing = TECHNICAL_PARAMS["atr"]["smoothing"]
            atr = np.mean(true_ranges[-TECHNICAL_PARAMS["atr"]["period"]:])
            atr = atr * TECHNICAL_PARAMS["atr"]["multiplier"]
            
            self._cache["atr"] = {"value": atr, "time": current_time}
            return atr
            
        except Exception as e:
            print(f"Error calculating ATR: {str(e)}")
            return 0.0

    def calculate_momentum(self) -> float:
        """Calculate price momentum indicator"""
        current_time = time.time()
        if current_time - self._cache["momentum"]["time"] < self.cache_duration:
            return self._cache["momentum"]["value"]

        if len(self.price_history) < TECHNICAL_PARAMS["momentum"]["period"]:
            return 0.0

        try:
            prices = np.array(list(self.price_history))
            momentum = (prices[-1] / prices[-TECHNICAL_PARAMS["momentum"]["period"]] - 1.0)
            
            self._cache["momentum"] = {"value": momentum, "time": current_time}
            return momentum
            
        except Exception as e:
            print(f"Error calculating momentum: {str(e)}")
            return 0.0

    def calculate_volume_ma(self) -> float:
        """Calculate volume moving average"""
        current_time = time.time()
        if current_time - self._cache["volume_ma"]["time"] < self.cache_duration:
            return self._cache["volume_ma"]["value"]

        if len(self.volume_history) < TECHNICAL_PARAMS["volume"]["ma_period"]:
            return 0.0

        try:
            volumes = np.array(list(self.volume_history))
            volume_ma = np.mean(volumes[-TECHNICAL_PARAMS["volume"]["ma_period"]:])
            
            self._cache["volume_ma"] = {"value": volume_ma, "time": current_time}
            return volume_ma
            
        except Exception as e:
            print(f"Error calculating volume MA: {str(e)}")
            return 0.0

    def calculate_trend(self) -> Tuple[float, bool]:
        """Calculate trend strength and direction"""
        current_time = time.time()
        if current_time - self._cache["trend"]["time"] < self.cache_duration:
            return self._cache["trend"]["value"], self._cache["trend"].get("direction", True)

        if len(self.price_history) < TECHNICAL_PARAMS["trend"]["long_period"]:
            return 0.0, True

        try:
            prices = np.array(list(self.price_history))
            short_ma = np.mean(prices[-TECHNICAL_PARAMS["trend"]["short_period"]:])
            long_ma = np.mean(prices[-TECHNICAL_PARAMS["trend"]["long_period"]:])
            
            trend_strength = abs(short_ma / long_ma - 1.0)
            trend_direction = short_ma > long_ma
            
            self._cache["trend"] = {
                "value": trend_strength,
                "direction": trend_direction,
                "time": current_time
            }
            return trend_strength, trend_direction
            
        except Exception as e:
            print(f"Error calculating trend: {str(e)}")
            return 0.0, True

    def get_market_conditions(self, current_price: float) -> Dict:
        """Get comprehensive market conditions analysis"""
        zscore = self.calculate_zscore(current_price)
        atr = self.calculate_atr()
        momentum = self.calculate_momentum()
        volume_ma = self.calculate_volume_ma()
        trend_strength, trend_direction = self.calculate_trend()
        
        # Get thresholds with defaults if missing
        atr_threshold = TECHNICAL_PARAMS.get("atr", {}).get("threshold", 0.005)
        trend_threshold = TECHNICAL_PARAMS.get("trend", {}).get("confirmation_threshold", 0.6)
        zscore_threshold = TECHNICAL_PARAMS.get("zscore", {}).get("threshold", 2.0)
        volume_threshold = TECHNICAL_PARAMS.get("volume", {}).get("threshold", 1.5)
        
        # Calculate volatility for Avellaneda-Stoikov model
        volatility = self.calculate_volatility_for_avellaneda()
        
        return {
            "zscore": zscore,
            "atr": atr,
            "momentum": momentum,
            "volume_ma": volume_ma,
            "trend_strength": trend_strength,
            "trend_direction": trend_direction,
            "is_volatile": atr > atr_threshold,
            "is_trending": trend_strength > trend_threshold,
            "mean_reverting_signal": abs(zscore) > zscore_threshold,
            "high_volume": volume_ma > volume_threshold,
            "volatility": volatility  # Add volatility to market conditions
        }
        
    def calculate_volatility_for_avellaneda(self) -> float:
        """Calculate volatility for Avellaneda-Stoikov model"""
        try:
            # Get volatility parameters
            vol_window = TECHNICAL_PARAMS.get("volatility", {}).get("window", 100)
            vol_floor = TECHNICAL_PARAMS.get("volatility", {}).get("vol_floor", 0.001)
            vol_ceiling = TECHNICAL_PARAMS.get("volatility", {}).get("vol_ceiling", 5.0)
            
            if len(self.price_history) < vol_window:
                return vol_floor
                
            prices = np.array(list(self.price_history))
            if np.any(prices <= 0):
                return vol_floor
                
            # Calculate log returns
            log_prices = np.log(prices[-vol_window:])
            returns = np.diff(log_prices)
            
            # Calculate annualized volatility (assuming 1-minute data)
            vol = np.std(returns) * np.sqrt(252 * 24 * 60)  # Annualized volatility
            
            # Apply floor and ceiling
            vol = np.clip(vol, vol_floor, vol_ceiling)
            
            # Scale volatility to be more reasonable for market making
            scaling = TECHNICAL_PARAMS.get("volatility", {}).get("scaling", 1.0)
            vol = vol * scaling
            
            return vol
            
        except Exception as e:
            print(f"Error calculating volatility: {str(e)}")
            return TECHNICAL_PARAMS.get("volatility", {}).get("vol_floor", 0.001) 