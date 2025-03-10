import logging
from typing import Optional, Tuple, Dict, List
import numpy as np
from collections import deque
import time

from ..config.market_config import (
    RISK_LIMITS,
    TRADING_PARAMS,
    TECHNICAL_PARAMS
)
from ..models.data_models import Ticker, Order, INVENTORY_CONFIG

class RiskManager:
    def __init__(self):
        # Position tracking
        self.position_size = 0.0
        self.entry_price = 0.0
        self.position_start_time = 0.0
        self.last_position_update = 0.0
        
        # Risk metrics
        self.highest_profit = 0.0
        self.max_drawdown = 0.0
        self.current_drawdown = 0.0
        self.trailing_stop_active = False
        self.trailing_stop_levels = {}
        
        # Price tracking
        self.price_history = deque(maxlen=TRADING_PARAMS["volatility"]["window"])
        self.volume_history = deque(maxlen=TECHNICAL_PARAMS["volume"]["ma_period"])
        
        # Performance tracking
        self.trade_history: List[Dict] = []
        self.pnl_history: List[float] = []
        self.drawdown_history: List[float] = []
        
        # Risk limits tracking
        self.limit_breaches = {
            "position_size": 0,
            "notional_value": 0,
            "drawdown": 0,
            "loss_streak": 0
        }
        
        # Cache
        self._cache = {
            "atr": {"value": 0.0, "time": 0},
            "zscore": {"value": 0.0, "time": 0},
            "volatility": {"value": 0.0, "time": 0}
        }
        self.cache_duration = 1.0

    def update_position(self, size: float, price: float, timestamp: Optional[float] = None) -> None:
        """Update position tracking with enhanced metrics"""
        current_time = timestamp or time.time()
        
        # Calculate PnL before update
        if self.position_size != 0:
            old_pnl = self.calculate_pnl_percentage(price)
            self.pnl_history.append(old_pnl)
            
            # Update drawdown
            if old_pnl < 0:
                self.current_drawdown = abs(old_pnl)
                self.max_drawdown = max(self.max_drawdown, self.current_drawdown)
            else:
                self.current_drawdown = 0
                
            self.drawdown_history.append(self.current_drawdown)
        
        # Update position
        if self.position_size == 0:
            self.position_start_time = current_time
            self.entry_price = price
            self.trailing_stop_levels = {}
        else:
            # Calculate new entry price based on weighted average
            total_value = (self.position_size * self.entry_price) + (size * price)
            self.entry_price = total_value / (self.position_size + size)
        
        self.position_size += size
        self.last_position_update = current_time
        
        # Store trade in history
        self.trade_history.append({
            "timestamp": current_time,
            "size": size,
            "price": price,
            "position_size": self.position_size,
            "entry_price": self.entry_price
        })
        
        # Maintain history size
        if len(self.trade_history) > 1000:
            self.trade_history = self.trade_history[-1000:]

    def update_market_data(self, price: float, volume: Optional[float] = None) -> None:
        """Update market data tracking"""
        self.price_history.append(price)
        if volume is not None:
            self.volume_history.append(volume)

    def calculate_zscore(self, current_price: float) -> float:
        """Calculate Z-score with enhanced caching"""
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
            logging.error(f"Error calculating zscore: {str(e)}")
            return 0.0

    def calculate_atr(self) -> float:
        """Calculate ATR with Wilder's smoothing"""
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
            logging.error(f"Error calculating ATR: {str(e)}")
            return 0.0

    def check_position_limits(self, mark_price: float) -> Tuple[bool, str]:
        """Check position limits with enhanced risk metrics"""
        # Position size check
        if abs(self.position_size) >= RISK_LIMITS["max_position"]:
            self.limit_breaches["position_size"] += 1
            return False, "Position size limit exceeded"

        # Notional value check
        notional = abs(self.position_size * mark_price)
        if notional >= RISK_LIMITS["max_notional"]:
            self.limit_breaches["notional_value"] += 1
            return False, "Notional value limit exceeded"

        # Drawdown check
        if self.current_drawdown >= RISK_LIMITS.get("max_drawdown", float('inf')):
            self.limit_breaches["drawdown"] += 1
            return False, "Maximum drawdown exceeded"

        # Loss streak check
        recent_trades = self.trade_history[-RISK_LIMITS.get("max_consecutive_losses", 5):]
        if len(recent_trades) >= RISK_LIMITS.get("max_consecutive_losses", 5):
            losses = sum(1 for trade in recent_trades if trade.get("pnl", 0) < 0)
            if losses >= RISK_LIMITS.get("max_consecutive_losses", 5):
                self.limit_breaches["loss_streak"] += 1
                return False, "Maximum consecutive losses reached"

        # Position holding time check
        holding_time = time.time() - self.position_start_time
        if self.position_size != 0 and holding_time > TRADING_PARAMS["position_management"].get("max_hold_time", float('inf')):
            return False, "Maximum position holding time exceeded"

        return True, ""

    def calculate_dynamic_take_profit(self) -> float:
        """Calculate dynamic take profit with market conditions"""
        base_tp = RISK_LIMITS["base_take_profit_pct"]
        
        # Adjust based on volatility
        atr = self.calculate_atr()
        volatility_scalar = min(2.0, max(0.5, atr / self.entry_price))
        
        # Adjust based on trend strength
        zscore = abs(self.calculate_zscore(self.entry_price))
        trend_scalar = min(1.5, max(0.7, zscore / 2))
        
        # Adjust based on position size
        position_ratio = abs(self.position_size) / RISK_LIMITS["max_position"]
        size_scalar = 1.0 + position_ratio * 0.5  # Increase TP for larger positions
        
        # Calculate final take profit
        dynamic_tp = base_tp * volatility_scalar * trend_scalar * size_scalar
        
        return min(
            RISK_LIMITS["max_take_profit_pct"],
            max(RISK_LIMITS["min_take_profit_pct"], dynamic_tp)
        )

    def check_stop_loss(self, current_price: float) -> bool:
        """Check stop loss with dynamic adjustment"""
        if self.position_size == 0:
            return False

        pnl_pct = self.calculate_pnl_percentage(current_price)
        
        # Base stop loss check
        if pnl_pct <= -RISK_LIMITS["stop_loss_pct"]:
            return True
            
        # Dynamic stop loss based on volatility
        atr = self.calculate_atr()
        dynamic_stop = max(
            RISK_LIMITS["stop_loss_pct"],
            atr / current_price * 2  # 2x ATR as minimum stop distance
        )
        
        return pnl_pct <= -dynamic_stop

    def check_take_profit(self, current_price: float) -> Tuple[bool, str]:
        """Check take profit with multiple levels"""
        if self.position_size == 0:
            return False, ""

        pnl_pct = self.calculate_pnl_percentage(current_price)
        
        # Update highest profit
        if pnl_pct > self.highest_profit:
            self.highest_profit = pnl_pct
            
        # Check take profit levels
        for level in RISK_LIMITS["take_profit_levels"]:
            if pnl_pct >= level["percentage"]:
                return True, f"Take profit level {level['percentage']} reached"
                
        # Check trailing stops
        if self.trailing_stop_active:
            for level in RISK_LIMITS["trailing_stop_levels"]:
                if level["activation"] <= self.highest_profit:
                    trailing_stop = self.highest_profit - level["distance"]
                    if pnl_pct < trailing_stop:
                        return True, f"Trailing stop at {level['activation']} triggered"
                        
        # Check dynamic take profit
        dynamic_tp = self.calculate_dynamic_take_profit()
        if pnl_pct >= dynamic_tp:
            return True, "Dynamic take profit reached"
            
        return False, ""

    def calculate_pnl_percentage(self, current_price: float) -> float:
        """Calculate PnL percentage with validation"""
        if self.position_size == 0 or self.entry_price <= 0:
            return 0.0

        try:
            direction = 1 if self.position_size > 0 else -1
            pnl = direction * (current_price - self.entry_price) / self.entry_price
            return pnl
            
        except Exception as e:
            logging.error(f"Error calculating PnL: {str(e)}")
            return 0.0

    def should_rebalance(self) -> Tuple[bool, str]:
        """Check if position needs rebalancing with multiple conditions"""
        # Size-based rebalancing
        if abs(self.position_size) >= RISK_LIMITS["max_position"] * RISK_LIMITS["rebalance_threshold"]:
            return True, "Position size threshold reached"
            
        # Time-based rebalancing
        holding_time = time.time() - self.position_start_time
        if holding_time > INVENTORY_CONFIG["inventory_fade_time"]:
            return True, "Position holding time exceeded"
            
        # Profit-based rebalancing
        if self.highest_profit >= INVENTORY_CONFIG["min_profit_rebalance"]:
            return True, "Profit target reached"
            
        # Inventory cost based rebalancing
        inventory_cost = abs(self.position_size) * INVENTORY_CONFIG["inventory_cost_factor"] * holding_time
        if inventory_cost > INVENTORY_CONFIG["min_profit_rebalance"]:
            return True, "Inventory cost threshold reached"
            
        return False, ""

    def calculate_rebalance_size(self) -> float:
        """Calculate optimal rebalance size"""
        target_size = RISK_LIMITS["max_position"] * 0.5
        base_reduction = abs(self.position_size) - target_size
        
        # Adjust based on market conditions
        atr = self.calculate_atr()
        volatility_factor = min(1.5, max(0.5, atr / self.entry_price))
        
        # More aggressive reduction in volatile markets
        reduction_size = base_reduction * volatility_factor
        
        # Ensure minimum reduction
        min_reduction = abs(self.position_size) * 0.1  # At least 10% reduction
        return max(reduction_size, min_reduction)

    def get_risk_metrics(self) -> Dict:
        """Get comprehensive risk metrics"""
        return {
            "position_size": self.position_size,
            "entry_price": self.entry_price,
            "position_age": time.time() - self.position_start_time,
            "highest_profit": self.highest_profit,
            "current_drawdown": self.current_drawdown,
            "max_drawdown": self.max_drawdown,
            "limit_breaches": self.limit_breaches,
            "trailing_stop_active": self.trailing_stop_active,
            "trailing_stop_levels": self.trailing_stop_levels,
            "recent_trades": self.trade_history[-10:] if self.trade_history else [],
            "volatility": self.calculate_atr(),
            "market_trend": self.calculate_zscore(self.price_history[-1] if self.price_history else 0)
        }
