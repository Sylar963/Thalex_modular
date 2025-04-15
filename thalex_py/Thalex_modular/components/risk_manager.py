from typing import Optional, Tuple, Dict, List, Callable, Any
import numpy as np
from collections import deque
import time

from ..config.market_config import (
    RISK_CONFIG,
    TRADING_CONFIG
)
from ..models.data_models import Ticker, Order
from ..thalex_logging import LoggerFactory

class RiskManager:
    def __init__(self) -> None:
        """Initialize RiskManager with simplified position tracking and risk metrics"""
        # Initialize logger
        self.logger = LoggerFactory.configure_component_logger(
            "risk_manager",
            log_file="risk_manager.log",
            high_frequency=False
        )
        
        # Position tracking (core metrics only)
        self.position_size = 0.0
        self.entry_price = 0.0
        self.position_value = 0.0
        self.position_start_time = 0.0
        
        # PnL tracking (simplified)
        self.realized_pnl = 0.0
        self.unrealized_pnl = 0.0
        self.current_drawdown = 0.0
        self.max_drawdown = 0.0
        self.peak_profit = 0.0
        
        # Risk utilization
        self.position_utilization = 0.0
        self.notional_utilization = 0.0
        
        # Core risk limits from config
        self.max_position = RISK_CONFIG["limits"]["max_position"]
        self.max_notional = RISK_CONFIG["limits"]["max_notional"]
        self.max_drawdown_pct = RISK_CONFIG["limits"]["max_drawdown"]
        self.stop_loss_pct = RISK_CONFIG["limits"]["stop_loss_pct"]
        self.take_profit_pct = RISK_CONFIG["limits"]["take_profit_pct"]
        
        # Tracking for reporting
        self.limit_breaches = {
            "position_size": 0,
            "notional_value": 0,
            "drawdown": 0
        }
        
        # Callback for risk events
        self.on_risk_limit_breached: Optional[Callable[[str, float], Any]] = None
        
        self.logger.info("Simplified risk manager initialized")
        
        # Validate core risk parameters
        self._validate_risk_parameters()

    def _validate_risk_parameters(self) -> None:
        """Validate core risk parameters to ensure they are within reasonable ranges"""
        try:
            # Check position limit
            if self.max_position <= 0:
                self.logger.warning(f"Invalid max_position: {self.max_position}, using default of 1.0")
                self.max_position = 1.0
            
            # Check notional limit
            if self.max_notional <= 0:
                self.logger.warning(f"Invalid max_notional: {self.max_notional}, using default of 50000")
                self.max_notional = 50000
            
            # Check stop loss percentage
            if self.stop_loss_pct <= 0 or self.stop_loss_pct > 0.5:
                self.logger.warning(f"Unusual stop_loss_pct: {self.stop_loss_pct}, recommended range is 0.01-0.1")
            
            # Check take profit percentage
            if self.take_profit_pct <= 0 or self.take_profit_pct > 0.5:
                self.logger.warning(f"Unusual take_profit_pct: {self.take_profit_pct}, recommended range is 0.01-0.1")
            
            # Check max drawdown
            if self.max_drawdown_pct <= 0 or self.max_drawdown_pct > 0.5:
                self.logger.warning(f"Unusual max_drawdown: {self.max_drawdown_pct}, recommended range is 0.05-0.2")
                
            self.logger.info("Risk parameters validated")
        except Exception as e:
            self.logger.error(f"Error validating risk parameters: {str(e)}")

    def update_position(self, size: float, price: float, timestamp: Optional[float] = None) -> Dict[str, Any]:
        """
        Update position tracking with simplified risk checks
        
        Args:
            size: Position size change (positive for buy, negative for sell)
            price: Price at which the position change occurred
            timestamp: Optional timestamp for the update
            
        Returns:
            Dict containing current position metrics and risk status
        """
        current_time = timestamp or time.time()
        
        # Calculate PnL before update if we have a position
        if self.position_size != 0:
            old_pnl = self.calculate_pnl_percentage(price)
            
            # Update drawdown
            if old_pnl < 0:
                self.current_drawdown = abs(old_pnl)
                self.max_drawdown = max(self.max_drawdown, self.current_drawdown)
            else:
                self.current_drawdown = 0
                # Track peak profit for take profit logic
                if old_pnl > self.peak_profit:
                    self.peak_profit = old_pnl
        
        # Update position
        old_position = self.position_size
        if self.position_size == 0:
            self.position_start_time = current_time
            self.entry_price = price
        else:
            # Calculate new entry price based on weighted average
            total_value = (self.position_size * self.entry_price) + (size * price)
            self.entry_price = total_value / (self.position_size + size)
        
        self.position_size += size
        self.position_value = abs(self.position_size * price)
        
        # Update utilization metrics
        self.position_utilization = abs(self.position_size) / self.max_position
        self.notional_utilization = self.position_value / self.max_notional
        
        # Check risk limits 
        exceeded_limit, reason = self.check_position_limits(price)
        
        # Log position update
        if exceeded_limit:
            self.logger.warning(
                f"Position updated: {old_position:.3f} -> {self.position_size:.3f} @ {price:.2f} "
                f"(util: pos={self.position_utilization:.2%}, notional={self.notional_utilization:.2%}). "
                f"Risk limit exceeded: {reason}"
            )
        else:
            self.logger.info(
                f"Position updated: {old_position:.3f} -> {self.position_size:.3f} @ {price:.2f} "
                f"(util: pos={self.position_utilization:.2%}, notional={self.notional_utilization:.2%})"
            )
        
        return {
            "position_size": self.position_size,
            "entry_price": self.entry_price,
            "position_value": self.position_value,
            "risk_limit_exceeded": exceeded_limit,
            "reason": reason if exceeded_limit else ""
        }

    def check_position_limits(self, current_price: float) -> Tuple[bool, str]:
        """
        Check if current position is within the three core risk limits
        
        Args:
            current_price: Current market price
            
        Returns:
            Tuple of (boolean indicating if any limit is exceeded, reason if exceeded)
        """
        # No position, always within limits
        if self.position_size == 0:
            return False, ""
            
        # 1. Check absolute position limit
        if abs(self.position_size) > self.max_position:
            return True, f"Position size {abs(self.position_size):.3f} exceeds max {self.max_position:.3f}"
            
        # 2. Check notional value limit
        position_value = abs(self.position_size * current_price)
        if position_value > self.max_notional:
            return True, f"Position value {position_value:.2f} exceeds max {self.max_notional:.2f}"
                          
        # 3. Check max drawdown
        pnl_pct = self.calculate_pnl_percentage(current_price)
        if pnl_pct < 0 and abs(pnl_pct) > self.max_drawdown_pct:
            return True, f"Drawdown {abs(pnl_pct):.2%} exceeds max {self.max_drawdown_pct:.2%}"
            
        # All checks passed
        return False, ""

    def calculate_pnl_percentage(self, current_price: float) -> float:
        """
        Calculate PnL as a percentage of the position value
        
        Args:
            current_price: Current market price
            
        Returns:
            Float representing PnL percentage (positive for profit, negative for loss)
        """
        # Can't calculate PnL without a position or valid entry price
        if self.position_size == 0 or self.entry_price == 0:
            return 0.0
            
        # For long positions, profit when current > entry
        if self.position_size > 0:
            return (current_price - self.entry_price) / self.entry_price
        # For short positions, profit when entry > current
        else:
            return (self.entry_price - current_price) / self.entry_price

    def check_stop_loss(self, current_price: float) -> bool:
        """
        Check if stop loss should be triggered based on current position and price
        
        Returns:
            Boolean indicating if stop loss has been triggered
        """
        if self.position_size == 0:
            return False
            
        pnl_pct = self.calculate_pnl_percentage(current_price)
        
        # Dynamic stop-loss adjustment - if in profit, widen the stop-loss
        effective_stop_loss = self.stop_loss_pct
        
        # If we're in profit, dynamically adjust the stop loss to be wider
        # This lets winners run while protecting capital
        if pnl_pct > 0:
            # Gradually widen stop loss as profit increases (up to 2x wider at take_profit level)
            profit_ratio = min(1.0, pnl_pct / self.take_profit_pct)
            stop_loss_multiplier = 1.0 + profit_ratio  # Ranges from 1.0 to 2.0
            effective_stop_loss = self.stop_loss_pct * stop_loss_multiplier
            
            # For significant profits (>2x take_profit), use a trailing stop
            if pnl_pct > self.take_profit_pct * 2:
                # Trailing stop at 50% of current profit
                trailing_stop = pnl_pct * 0.5
                effective_stop_loss = max(effective_stop_loss, trailing_stop)
        
        # Check stop loss percentage threshold with dynamic adjustment
        if pnl_pct <= -effective_stop_loss:
            self.logger.warning(
                f"Stop loss triggered: PnL {pnl_pct:.2%} below threshold -{effective_stop_loss:.2%}. "
                f"Position: {self.position_size:.3f} @ {self.entry_price:.2f}, current price: {current_price:.2f}"
            )
            return True
            
        # Check drawdown threshold 
        if self.current_drawdown >= self.max_drawdown_pct:
            self.logger.warning(
                f"Stop loss triggered: Drawdown {self.current_drawdown:.2%} exceeded max {self.max_drawdown_pct:.2%}. "
                f"Position: {self.position_size:.3f} @ {self.entry_price:.2f}, current price: {current_price:.2f}"
            )
            return True
            
        return False
        
    def check_take_profit(self, current_price: float) -> Tuple[bool, str, float]:
        """
        Check if take profit should be triggered with enhanced partial profit taking
        
        Args:
            current_price: Current market price
            
        Returns:
            Tuple of (boolean indicating if take profit triggered, reason string, portion to close)
        """
        if self.position_size == 0:
            return False, "", 0.0
            
        pnl_pct = self.calculate_pnl_percentage(current_price)
        
        # No take profit if we're in a loss
        if pnl_pct <= 0:
            return False, "", 0.0
            
        # Enhanced take profit with progressive profit-taking strategy
        # Take partial profits at different levels
        
        # Stage 1: Take 25% at the main take profit threshold
        if pnl_pct >= self.take_profit_pct:
            portion_to_close = 0.25
            reason = f"Take profit level 1: {pnl_pct:.2%} >= {self.take_profit_pct:.2%}, closing {portion_to_close:.0%}"
            return True, reason, portion_to_close
        
        # Stage 2: Take another 25% at 1.5x the take profit threshold    
        if pnl_pct >= self.take_profit_pct * 1.5:
            portion_to_close = 0.25
            reason = f"Take profit level 2: {pnl_pct:.2%} >= {self.take_profit_pct*1.5:.2%}, closing {portion_to_close:.0%}"
            return True, reason, portion_to_close
            
        # Stage 3: Take another 25% at 2x the take profit threshold
        if pnl_pct >= self.take_profit_pct * 2.0:
            portion_to_close = 0.25
            reason = f"Take profit level 3: {pnl_pct:.2%} >= {self.take_profit_pct*2.0:.2%}, closing {portion_to_close:.0%}"
            return True, reason, portion_to_close
            
        # Stage 4: Close final 25% at 3x the take profit threshold
        if pnl_pct >= self.take_profit_pct * 3.0:
            portion_to_close = 0.25  # Close the remainder
            reason = f"Take profit level 4: {pnl_pct:.2%} >= {self.take_profit_pct*3.0:.2%}, closing final {portion_to_close:.0%}"
            return True, reason, portion_to_close
        
        # Simple trailing take profit logic as a safety net
        if self.peak_profit > self.take_profit_pct * 0.5:  # Only apply trailing if we've seen significant profit
            profit_drawdown = (self.peak_profit - pnl_pct) / self.peak_profit if self.peak_profit > 0 else 0
            if profit_drawdown > 0.5:  # Don't give back more than 50% of peak profit
                reason = f"Trailing take profit: Given back {profit_drawdown:.2%} of peak profit {self.peak_profit:.2%}"
                return True, reason, 1.0  # Close the entire position to protect profit
            
        return False, "", 0.0

    async def check_risk_limits(self) -> bool:
        """
        Check all risk limits and return whether trading should continue
        
        Returns:
            Boolean indicating if risk limits are within acceptable range
        """
        # Skip if no position
        if self.position_size == 0:
            return True
            
        try:
            # 1. Position size limit
            if abs(self.position_size) > self.max_position:
                self.logger.warning(f"Position size limit exceeded: {abs(self.position_size):.3f} > {self.max_position:.3f}")
                self.limit_breaches["position_size"] += 1
                return False
                
            # 2. Notional value limit
            if self.position_value > self.max_notional:
                self.logger.warning(f"Notional value limit exceeded: {self.position_value:.2f} > {self.max_notional:.2f}")
                self.limit_breaches["notional_value"] += 1
                return False
                
            # 3. Drawdown limit
            if self.current_drawdown > self.max_drawdown_pct:
                self.logger.warning(f"Drawdown limit exceeded: {self.current_drawdown:.2%} > {self.max_drawdown_pct:.2%}")
                self.limit_breaches["drawdown"] += 1
                return False
                
            return True
            
        except Exception as e:
            self.logger.error(f"Error checking risk limits: {str(e)}")
            return False  # Default to safety in case of errors

    def register_callback(self, event_type: str, callback: Callable) -> None:
        """Register a callback function for risk limit breaches"""
        if event_type == "risk_limit":
            self.on_risk_limit_breached = callback
        else:
            self.logger.warning(f"Unknown event type for callback registration: {event_type}")
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """
        Get summary of current risk metrics
        
        Returns:
            Dictionary containing current risk metrics
        """
        return {
            "position_size": self.position_size,
            "entry_price": self.entry_price,
            "position_value": self.position_value,
            "position_utilization": self.position_utilization,
            "notional_utilization": self.notional_utilization,
            "current_drawdown": self.current_drawdown,
            "max_drawdown": self.max_drawdown,
            "limit_breaches": self.limit_breaches,
            "position_duration": time.time() - self.position_start_time if self.position_size != 0 else 0
        }

    async def update_trade_metrics(self, trade_data: Dict) -> None:
        """
        Update metrics based on trade data (simplified)
        
        Args:
            trade_data: Dictionary containing trade information
        """
        try:
            # Extract trade info
            price = trade_data.get("price", 0.0)
            amount = trade_data.get("amount", 0.0)
            direction = trade_data.get("direction", "")
            
            # Skip if invalid data
            if price <= 0 or amount <= 0 or direction not in ["buy", "sell"]:
                return
                
            # Update PnL if applicable
            if direction == "buy":
                self.realized_pnl += trade_data.get("realized_pnl", 0.0)
            elif direction == "sell":
                self.realized_pnl += trade_data.get("realized_pnl", 0.0)
                
            # Check risk limits after trades
            current_time = time.time()
            trade_record = {
                "timestamp": current_time,
                "price": price,
                "amount": amount, 
                "direction": direction,
                "realized_pnl": trade_data.get("realized_pnl", 0.0)
            }
            
            # No need to store extensive trade history
            
        except Exception as e:
            self.logger.error(f"Error updating trade metrics: {str(e)}")

    async def update_position_fill(self, direction: str, price: float, size: float, timestamp: Optional[float] = None) -> Dict[str, Any]:
        """
        Update position tracking directly from order fill events (faster, more accurate)
        
        Args:
            direction: Order direction ('buy' or 'sell')
            price: Fill price
            size: Fill size
            timestamp: Optional timestamp for the update
            
        Returns:
            Dict containing current position metrics and risk status
        """
        current_time = timestamp or time.time()
        
        # Adjust size based on direction
        position_change = size if direction.lower() == 'buy' else -size
        
        # Calculate PnL before update if we have a position
        if self.position_size != 0:
            old_pnl = self.calculate_pnl_percentage(price)
            
            # Update drawdown
            if old_pnl < 0:
                self.current_drawdown = abs(old_pnl)
                self.max_drawdown = max(self.max_drawdown, self.current_drawdown)
            else:
                self.current_drawdown = 0
                # Track peak profit for take profit logic
                if old_pnl > self.peak_profit:
                    self.peak_profit = old_pnl
        
        # Update position
        old_position = self.position_size
        new_position = self.position_size + position_change
        
        # Calculate if this is a new position, adding to position, or reducing position
        is_new_position = (old_position == 0)
        is_reducing_position = (abs(new_position) < abs(old_position) and 
                                ((old_position > 0 and new_position > 0) or 
                                 (old_position < 0 and new_position < 0)))
        is_closing_position = (new_position == 0)
        is_reversing_position = (old_position * new_position < 0)  # Sign change
        
        # Handle entry price updates
        if is_new_position:
            # New position - set entry price to fill price
            self.position_start_time = current_time
            self.entry_price = price
            self.logger.info(f"New position opened: {position_change:.4f} @ {price:.2f}")
        elif is_reducing_position or is_closing_position:
            # Reducing position - calculate realized PnL
            pnl = position_change * (price - self.entry_price) * (-1 if old_position < 0 else 1)
            self.realized_pnl += pnl
            
            if is_closing_position:
                # Position closed - reset entry price
                self.entry_price = 0
                self.logger.info(f"Position closed: realized PnL: {pnl:.2f}")
            else:
                # Keep same entry price when reducing
                self.logger.info(f"Position reduced: {old_position:.4f} -> {new_position:.4f}, realized PnL: {pnl:.2f}")
        elif is_reversing_position:
            # Position reversal - close old position, open new position
            # Calculate realized PnL on the closed portion
            pnl = old_position * (price - self.entry_price) * (-1 if old_position < 0 else 1)
            self.realized_pnl += pnl
            
            # Set new entry price
            self.entry_price = price
            self.position_start_time = current_time
            self.logger.info(f"Position reversed: {old_position:.4f} -> {new_position:.4f}, realized PnL: {pnl:.2f}")
        else:
            # Adding to position - update entry price with weighted average
            total_value = (self.position_size * self.entry_price) + (position_change * price)
            self.entry_price = total_value / new_position
            self.logger.info(f"Position increased: {old_position:.4f} -> {new_position:.4f} @ {price:.2f}, new avg: {self.entry_price:.2f}")
        
        # Update position size
        self.position_size = new_position
        self.position_value = abs(self.position_size * price)
        
        # Update utilization metrics
        self.position_utilization = abs(self.position_size) / self.max_position
        self.notional_utilization = self.position_value / self.max_notional
        
        # Check risk limits 
        exceeded_limit, reason = self.check_position_limits(price)
        
        # Log position update
        if exceeded_limit:
            self.logger.warning(
                f"Position updated from fills: {old_position:.3f} -> {self.position_size:.3f} @ {price:.2f} "
                f"(util: pos={self.position_utilization:.2%}, notional={self.notional_utilization:.2%}). "
                f"Risk limit exceeded: {reason}"
            )
        
        return {
            "position_size": self.position_size,
            "entry_price": self.entry_price,
            "position_value": self.position_value,
            "risk_limit_exceeded": exceeded_limit,
            "reason": reason if exceeded_limit else ""
        }
