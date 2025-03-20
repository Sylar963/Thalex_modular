import numpy as np
from collections import deque
import logging
from typing import Optional

from ..config.market_config import TECHNICAL_PARAMS, TRADING_PARAMS

class TechnicalAnalysis:
    """Technical analysis component providing market signals"""
    
    def __init__(self):
        # Price history
        self.price_history = deque(maxlen=TECHNICAL_PARAMS["zscore"]["window"])
        self.returns_history = deque(maxlen=TECHNICAL_PARAMS["zscore"]["window"])
        
        # Volume tracking
        self.volume_history = deque(maxlen=TECHNICAL_PARAMS["volume"]["ma_period"])
        self.volume_ma = 0.0
        
        # Volatility tracking
        self.volatility = TRADING_PARAMS["volatility"]["vol_floor"]
        self.atr_history = deque(maxlen=TECHNICAL_PARAMS["atr"]["period"])
        
        # Trend tracking
        self.short_ma = deque(maxlen=TECHNICAL_PARAMS["trend"]["short_period"])
        self.long_ma = deque(maxlen=TECHNICAL_PARAMS["trend"]["long_period"])
        
        # Market impact tracking
        self.trade_impact_history = deque(maxlen=100)
        self.market_impact = 0.0
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        
    def update(self, price: float, volume: Optional[float] = None) -> None:
        """Update all technical indicators with new price data"""
        try:
            if price <= 0:
                return
                
            # Update price history
            self.price_history.append(price)
            
            # Update returns if we have enough prices
            if len(self.price_history) > 1:
                ret = np.log(price / self.price_history[-2])
                self.returns_history.append(ret)
            
            # Update volume metrics
            if volume is not None and volume > 0:
                self.volume_history.append(volume)
                self.volume_ma = np.mean(self.volume_history) if len(self.volume_history) > 0 else 0
            
            # Update volatility
            self._update_volatility()
            
            # Update trend indicators
            self.short_ma.append(price)
            self.long_ma.append(price)
            
        except Exception as e:
            self.logger.error(f"Error updating technical analysis: {str(e)}")
    
    def _update_volatility(self) -> None:
        """Update volatility estimates"""
        try:
            if len(self.returns_history) < 2:
                return
                
            # Calculate returns volatility
            vol = np.std(self.returns_history) * np.sqrt(252 * 24 * 60)  # Annualized
            
            # Apply EWM smoothing
            ewm_span = TRADING_PARAMS["volatility"]["ewm_span"]
            smooth_factor = 2.0 / (ewm_span + 1.0)
            self.volatility = (
                smooth_factor * vol + 
                (1 - smooth_factor) * self.volatility
            )
            
            # Apply floor and ceiling
            self.volatility = np.clip(
                self.volatility,
                TRADING_PARAMS["volatility"]["vol_floor"],
                TRADING_PARAMS["volatility"]["vol_ceiling"]
            )
            
        except Exception as e:
            self.logger.error(f"Error updating volatility: {str(e)}")
    
    def get_volatility(self) -> float:
        """Get current volatility estimate"""
        return self.volatility
    
    def is_volatile(self) -> bool:
        """Check if market is currently volatile"""
        try:
            if len(self.returns_history) < TECHNICAL_PARAMS["atr"]["period"]:
                return False
                
            current_atr = np.mean(self.atr_history) if len(self.atr_history) > 0 else 0
            threshold = TECHNICAL_PARAMS["atr"]["threshold"]
            
            return current_atr > threshold
            
        except Exception as e:
            self.logger.error(f"Error checking volatility: {str(e)}")
            return False
    
    def get_zscore(self) -> float:
        """Calculate z-score of current price"""
        try:
            if len(self.price_history) < TECHNICAL_PARAMS["zscore"]["window"]:
                return 0.0
                
            prices = np.array(self.price_history)
            mean = np.mean(prices)
            std = np.std(prices)
            
            if std == 0:
                return 0.0
                
            current_price = prices[-1]
            return (current_price - mean) / std
            
        except Exception as e:
            self.logger.error(f"Error calculating z-score: {str(e)}")
            return 0.0
    
    def is_mean_reverting(self) -> bool:
        """Check if price shows mean reversion signal"""
        try:
            zscore = self.get_zscore()
            threshold = TECHNICAL_PARAMS["zscore"]["threshold"]
            
            return abs(zscore) > threshold
            
        except Exception as e:
            self.logger.error(f"Error checking mean reversion: {str(e)}")
            return False
    
    def is_trending(self) -> bool:
        """Check if market is trending"""
        try:
            if (len(self.short_ma) < TECHNICAL_PARAMS["trend"]["short_period"] or
                len(self.long_ma) < TECHNICAL_PARAMS["trend"]["long_period"]):
                return False
                
            short_avg = np.mean(self.short_ma)
            long_avg = np.mean(self.long_ma)
            
            # Calculate trend strength
            trend_strength = abs(short_avg - long_avg) / long_avg
            
            return trend_strength > TECHNICAL_PARAMS["trend"]["confirmation_threshold"]
            
        except Exception as e:
            self.logger.error(f"Error checking trend: {str(e)}")
            return False
    
    def get_trend_direction(self) -> bool:
        """Get current trend direction (True=up, False=down)"""
        try:
            if not self.is_trending():
                return True
                
            short_avg = np.mean(self.short_ma)
            long_avg = np.mean(self.long_ma)
            
            return short_avg > long_avg
            
        except Exception as e:
            self.logger.error(f"Error getting trend direction: {str(e)}")
            return True
    
    def get_trend_strength(self) -> float:
        """Get current trend strength"""
        try:
            if (len(self.short_ma) < TECHNICAL_PARAMS["trend"]["short_period"] or
                len(self.long_ma) < TECHNICAL_PARAMS["trend"]["long_period"]):
                return 0.0
                
            short_avg = np.mean(self.short_ma)
            long_avg = np.mean(self.long_ma)
            
            return abs(short_avg - long_avg) / long_avg
            
        except Exception as e:
            self.logger.error(f"Error getting trend strength: {str(e)}")
            return 0.0
    
    def update_market_impact(self, trade_size: float, price_change: float) -> None:
        """Update market impact tracking"""
        try:
            if trade_size <= 0:
                return
                
            impact = abs(price_change / trade_size)
            self.trade_impact_history.append(impact)
            
            # Update market impact estimate
            self.market_impact = np.mean(self.trade_impact_history) if len(self.trade_impact_history) > 0 else 0
            
        except Exception as e:
            self.logger.error(f"Error updating market impact: {str(e)}")
    
    def get_market_impact(self) -> float:
        """Get current market impact estimate"""
        return self.market_impact 