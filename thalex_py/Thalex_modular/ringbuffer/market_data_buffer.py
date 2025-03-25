import numpy as np
import time
from typing import Dict, Optional, Tuple, Union
from .fast_ringbuffer import FastRingBuffer

class MarketDataBuffer:
    """
    Specialized ring buffer for market data with built-in technical indicators.
    Uses SIMD operations for fast calculations.
    """
    def __init__(
        self,
        capacity: int = 1000,
        ema_periods: Dict[str, int] = None,
        volatility_window: int = 100,
        atr_period: int = 14
    ):
        """
        Initialize market data buffer.
        
        Args:
            capacity: Buffer size
            ema_periods: Dict of EMA periods (e.g. {"fast": 12, "slow": 26})
            volatility_window: Window for volatility calculation
            atr_period: Period for ATR calculation
        """
        # Initialize base buffers
        self.prices = FastRingBuffer(capacity)
        self.volumes = FastRingBuffer(capacity)
        self.timestamps = FastRingBuffer(capacity, dtype=np.int64)
        
        # OHLCV data
        self.opens = FastRingBuffer(capacity)
        self.highs = FastRingBuffer(capacity)
        self.lows = FastRingBuffer(capacity)
        self.closes = FastRingBuffer(capacity)
        
        # Technical indicators
        self.ema_periods = ema_periods or {"fast": 12, "slow": 26}
        self.ema_buffers = {
            name: FastRingBuffer(capacity)
            for name in self.ema_periods.keys()
        }
        
        # Volatility and ATR
        self.volatility_window = volatility_window
        self.atr_period = atr_period
        self.volatility = FastRingBuffer(capacity)
        self.atr = FastRingBuffer(capacity)
        
        # Market impact tracking
        self.impact_buffer = FastRingBuffer(capacity)
        self.buy_volume = FastRingBuffer(capacity)
        self.sell_volume = FastRingBuffer(capacity)
        
        # Performance optimization
        self._cache = {}
        self._last_update_time = 0
        
    def update(
        self,
        price: float,
        volume: float = 0.0,
        timestamp: int = None,
        is_buy: Optional[bool] = None
    ) -> None:
        """
        Update market data with new tick.
        
        Args:
            price: Current price
            volume: Trade volume
            timestamp: Unix timestamp
            is_buy: Whether the trade was a buy
        """
        # Update base data
        self.prices.append(price)
        self.volumes.append(volume)
        
        if timestamp is None:
            timestamp = int(time.time() * 1000)
        else:
            # Normalize timestamp to prevent overflow
            # If timestamp is in seconds since epoch (common format)
            if timestamp > 1e9:
                # For statistics we just need relative time differences
                # Convert to seconds or minutes since the start of data collection
                # This prevents overflow when squaring timestamps
                current_time = time.time()
                # Use relative time (minutes since start of collection)
                timestamp = (timestamp - int(current_time)) / 60.0
        self.timestamps.append(timestamp)
        
        # Update OHLCV
        if len(self.closes) == 0:
            self.opens.append(price)
            self.highs.append(price)
            self.lows.append(price)
        else:
            self.highs.append(max(self.highs.max, price))
            self.lows.append(min(self.lows.min, price))
        self.closes.append(price)
        
        # Update EMAs
        for name, period in self.ema_periods.items():
            self._update_ema(name, period, price)
            
        # Update volatility
        if len(self.prices) >= 2:
            returns = np.log(price / self.prices[-2])
            vol = self._calculate_volatility(returns)
            self.volatility.append(vol)
            
        # Update ATR
        if len(self.highs) >= self.atr_period:
            atr = self._calculate_atr()
            self.atr.append(atr)
            
        # Update volume profiles
        if is_buy is not None:
            if is_buy:
                self.buy_volume.append(volume)
                self.sell_volume.append(0.0)
            else:
                self.buy_volume.append(0.0)
                self.sell_volume.append(volume)
                
        # Update market impact
        if volume > 0:
            impact = abs(price - self.prices[-2]) / price if len(self.prices) >= 2 else 0
            self.impact_buffer.append(impact * volume)
            
        # Clear cache
        self._cache.clear()
        self._last_update_time = timestamp
        
    def _update_ema(self, name: str, period: int, price: float) -> None:
        """Update EMA calculation"""
        alpha = 2.0 / (period + 1)
        
        if len(self.ema_buffers[name]) == 0:
            self.ema_buffers[name].append(price)
        else:
            prev_ema = self.ema_buffers[name][-1]
            new_ema = price * alpha + prev_ema * (1 - alpha)
            self.ema_buffers[name].append(new_ema)
            
    def _calculate_volatility(self, returns: float) -> float:
        """Calculate rolling volatility"""
        if len(self.prices) < self.volatility_window:
            return np.nan
            
        # Get recent returns
        recent_returns = self.prices.get_last(self.volatility_window)
        returns_array = np.log(recent_returns[1:] / recent_returns[:-1])
        
        # Calculate annualized volatility
        vol = np.std(returns_array) * np.sqrt(252 * 24 * 60)  # Annualized from minute data
        return vol
        
    def _calculate_atr(self) -> float:
        """Calculate Average True Range"""
        if len(self.highs) < self.atr_period:
            return np.nan
            
        # Get recent high/low/close
        highs = self.highs.get_last(self.atr_period + 1)
        lows = self.lows.get_last(self.atr_period + 1)
        closes = self.closes.get_last(self.atr_period + 1)
        
        # Calculate True Range
        tr1 = highs[1:] - lows[1:]
        tr2 = np.abs(highs[1:] - closes[:-1])
        tr3 = np.abs(lows[1:] - closes[:-1])
        
        tr = np.maximum(np.maximum(tr1, tr2), tr3)
        
        # Calculate ATR
        atr = np.mean(tr)
        return atr
        
    def get_market_impact(self, window: int = 100) -> float:
        """Calculate market impact over specified window"""
        if len(self.impact_buffer) < window:
            return 0.0
            
        recent_impact = self.impact_buffer.get_last(window)
        recent_volume = self.volumes.get_last(window)
        
        # Volume-weighted average impact
        return np.sum(recent_impact) / np.sum(recent_volume) if np.sum(recent_volume) > 0 else 0.0
        
    def get_volume_profile(self, window: int = 100) -> Dict[str, float]:
        """Get volume profile statistics"""
        if len(self.buy_volume) < window:
            return {"buy_ratio": 0.5, "total_volume": 0.0}
            
        recent_buys = np.sum(self.buy_volume.get_last(window))
        recent_sells = np.sum(self.sell_volume.get_last(window))
        total = recent_buys + recent_sells
        
        return {
            "buy_ratio": recent_buys / total if total > 0 else 0.5,
            "total_volume": total
        }
        
    def get_trend_strength(self) -> float:
        """Calculate trend strength using EMAs"""
        if len(self.ema_buffers["fast"]) == 0 or len(self.ema_buffers["slow"]) == 0:
            return 0.0
            
        fast_ema = self.ema_buffers["fast"][-1]
        slow_ema = self.ema_buffers["slow"][-1]
        
        # Normalize trend strength
        return (fast_ema - slow_ema) / slow_ema if slow_ema != 0 else 0.0
        
    def get_volatility_regime(self) -> str:
        """Determine current volatility regime"""
        if len(self.volatility) < 100:
            return "normal"
            
        current_vol = self.volatility[-1]
        vol_percentile = np.percentile(self.volatility.get_last(100), 75)
        
        if current_vol > vol_percentile * 1.5:
            return "high"
        elif current_vol < vol_percentile * 0.5:
            return "low"
        return "normal"
        
    def get_market_state(self) -> Dict[str, Union[float, str]]:
        """Get comprehensive market state"""
        return {
            "trend_strength": self.get_trend_strength(),
            "volatility": self.volatility[-1] if len(self.volatility) > 0 else np.nan,
            "volatility_regime": self.get_volatility_regime(),
            "atr": self.atr[-1] if len(self.atr) > 0 else np.nan,
            "market_impact": self.get_market_impact(),
            "volume_profile": self.get_volume_profile()
        } 