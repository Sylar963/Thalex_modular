import logging
from typing import Optional, Tuple, Dict, List, Callable, Any
import numpy as np
from collections import deque
import time

from ..config.market_config import (
    RISK_LIMITS,
    TRADING_PARAMS,
    TECHNICAL_PARAMS,
    RISK_CONFIG,
    TRADING_CONFIG
)
from ..models.data_models import Ticker, Order

class RiskManager:
    def __init__(self) -> None:
        """Initialize RiskManager with position tracking and risk metrics"""
        # Position tracking
        self.position_size = 0.0
        self.entry_price = 0.0
        self.position_value = 0.0
        self.position_start_time = 0.0
        self.last_position_update = 0.0
        
        # PnL tracking
        self.trade_history = []
        self.pnl_history = []
        self.current_drawdown = 0.0
        self.max_drawdown = 0.0
        self.drawdown_history = []
        self.peak_profit = 0.0  # For trailing take-profit
        
        # Risk metrics
        self.position_utilization = 0.0
        self.notional_utilization = 0.0
        self.limit_breaches = {
            "position_size": 0,
            "notional_value": 0,
            "drawdown": 0,
            "loss_streak": 0
        }
        
        # Market data tracking
        self.price_history = []
        self.volume_history = []
        
        # Volatility and market impact
        self.current_volatility = 0.0
        self.previous_volatility = 0.0
        self.volatility_history = []
        self.market_impact = 0.0
        
        # Rebalance logic
        self.rebalance_triggered = False
        self.last_rebalance_check = 0.0
        self.rebalance_cooldown = RISK_CONFIG["limits"].get("rebalance_cooldown", 300)  # Default 5 minutes
        
        # Risk event callbacks
        self.on_stop_loss_triggered: Optional[Callable[[float, float], Any]] = None
        self.on_take_profit_triggered: Optional[Callable[[float, float], Any]] = None
        self.on_rebalance_needed: Optional[Callable[[float, str], Any]] = None
        self.on_risk_limit_breached: Optional[Callable[[str, float], Any]] = None
        
        # Circuit breaker flags
        self.emergency_stop_triggered = False
        self.circuit_breaker_active = False
        self.circuit_breaker_until = 0.0
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        
        # Validate risk parameters
        self._validate_risk_parameters()

    def _validate_risk_parameters(self) -> None:
        """Validate risk parameters to ensure they are within reasonable ranges"""
        try:
            # Check position limit
            max_position = RISK_CONFIG["limits"]["max_position"]
            if max_position <= 0:
                self.logger.warning(f"Invalid max_position: {max_position}, using default of 1.0")
                # We don't modify the config, just log the warning
            
            # Check notional limit
            max_notional = RISK_CONFIG["limits"]["max_notional"]
            if max_notional <= 0:
                self.logger.warning(f"Invalid max_notional: {max_notional}, using default of 50000")
            
            # Check stop loss percentage
            stop_loss_pct = RISK_CONFIG["limits"]["stop_loss_pct"]
            if stop_loss_pct <= 0 or stop_loss_pct > 0.5:
                self.logger.warning(f"Unusual stop_loss_pct: {stop_loss_pct}, recommended range is 0.01-0.1")
            
            # Check take profit percentage
            take_profit_pct = RISK_CONFIG["limits"]["take_profit_pct"]
            if take_profit_pct <= 0 or take_profit_pct > 0.5:
                self.logger.warning(f"Unusual take_profit_pct: {take_profit_pct}, recommended range is 0.01-0.1")
            
            # Check max drawdown
            max_drawdown = RISK_CONFIG["limits"]["max_drawdown"]
            if max_drawdown <= 0 or max_drawdown > 0.5:
                self.logger.warning(f"Unusual max_drawdown: {max_drawdown}, recommended range is 0.05-0.2")
            
            # Check volatility scaling factor
            vol_scaling_factor = RISK_CONFIG["limits"].get("vol_scaling_factor", 1.0)
            if vol_scaling_factor <= 0:
                self.logger.warning(f"Invalid vol_scaling_factor: {vol_scaling_factor}, should be positive")
                
            self.logger.info("Risk parameters validated")
        except Exception as e:
            self.logger.error(f"Error validating risk parameters: {str(e)}")

    def update_position(self, size: float, price: float, timestamp: Optional[float] = None) -> Dict[str, Any]:
        """
        Update position tracking with enhanced metrics and perform automatic risk checks.
        
        Args:
            size: Position size change (positive for buy, negative for sell)
            price: Price at which the position change occurred
            timestamp: Optional timestamp for the update (defaults to current time)
            
        Returns:
            Dict containing triggered risk events and current position metrics
        """
        current_time = timestamp or time.time()
        risk_events = {
            "stop_loss_triggered": False,
            "take_profit_triggered": False,
            "rebalance_needed": False,
            "risk_limit_breached": False,
            "limit_breach_reason": ""
        }
        
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
        
        # Update utilization metrics
        max_position = RISK_CONFIG["limits"]["max_position"]
        max_notional = RISK_CONFIG["limits"]["max_notional"]
        self.position_utilization = abs(self.position_size) / max_position
        self.notional_utilization = self.position_value / max_notional
        
        # Perform immediate risk checks if we have a position
        if self.position_size != 0:
            # Check stop loss
            if self.check_stop_loss(price):
                risk_events["stop_loss_triggered"] = True
                if self.on_stop_loss_triggered:
                    self.on_stop_loss_triggered(self.position_size, price)
            
            # Check take profit
            take_profit_triggered, reason = self.check_take_profit(price)
            if take_profit_triggered:
                risk_events["take_profit_triggered"] = True
                if self.on_take_profit_triggered:
                    self.on_take_profit_triggered(self.position_size, price)
            
            # Check if rebalance is needed
            rebalance_needed, reason = self.should_rebalance()
            if rebalance_needed:
                risk_events["rebalance_needed"] = True
                risk_events["rebalance_reason"] = reason
                if self.on_rebalance_needed:
                    self.on_rebalance_needed(self.position_size, reason)
                self.rebalance_triggered = True
                self.last_rebalance_check = current_time
            
            # Check risk limits
            position_within_limits, limit_breach_reason = self.check_position_limits(price)
            if not position_within_limits:
                risk_events["risk_limit_breached"] = True
                risk_events["limit_breach_reason"] = limit_breach_reason
                if self.on_risk_limit_breached:
                    self.on_risk_limit_breached(limit_breach_reason, self.position_size)
        
        # Log position update with risk information
        risk_alerts = []
        for event, triggered in risk_events.items():
            if triggered and event != "limit_breach_reason":
                risk_alerts.append(event)
                
        if risk_alerts:
            self.logger.warning(
                f"Position updated: {old_position:.3f} -> {self.position_size:.3f} @ {price:.2f} "
                f"(util: pos={self.position_utilization:.2%}, notional={self.notional_utilization:.2%}). "
                f"Risk alerts: {', '.join(risk_alerts)}"
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
            "risk_events": risk_events
        }

    def update_market_data(self, price: float, volume: Optional[float] = None) -> None:
        """
        Update market data tracking with new price and volume information
        
        Args:
            price: Current market price
            volume: Optional trading volume
        """
        # Check if we have previous price to calculate returns
        if self.price_history:
            prev_price = self.price_history[-1]
            # Calculate percentage return
            pct_return = (price / prev_price) - 1
            # Update volatility estimate
            self.update_volatility_estimate(pct_return)
        
        # Update price history
        self.price_history.append(price)
        
        # Keep history at manageable size
        max_history = 500
        if len(self.price_history) > max_history:
            self.price_history = self.price_history[-max_history:]
        
        # Update volume history if provided
        if volume is not None:
            self.volume_history.append(volume)
            
            # Keep history at manageable size
            if len(self.volume_history) > max_history:
                self.volume_history = self.volume_history[-max_history:]
            
            # Estimate market impact if we have a position
            if self.position_size != 0:
                # Use average daily volume or recent volume as denominator
                avg_volume = np.mean(self.volume_history) if len(self.volume_history) > 0 else volume
                self.market_impact = self.estimate_market_impact(self.position_size, avg_volume)
        
        # Log significant price movements
        if len(self.price_history) >= 2:
            short_term_change = (price / self.price_history[-2] - 1)
            if abs(short_term_change) > 0.01:  # 1% move
                self.logger.info(f"Significant price movement: {short_term_change:+.2%} to {price:.2f}")
                
                # If we have a position, calculate current PnL
                if self.position_size != 0:
                    pnl_pct = self.calculate_pnl_percentage(price)
                    self.logger.info(f"Current position PnL: {pnl_pct:.2%} on {self.position_size:.3f} contracts")

    def check_position_limits(self, current_price: float) -> Tuple[bool, str]:
        """
        Check if current position is within risk limits
        
        Args:
            current_price: Current market price
            
        Returns:
            Tuple of (boolean indicating if position is within limits, reason if exceeded)
        """
        # No position, always within limits
        if self.position_size == 0:
            return True, ""
            
        # Check absolute position limit
        max_position = RISK_CONFIG["limits"]["max_position"]
        if abs(self.position_size) > max_position:
            return False, f"Position size {abs(self.position_size):.3f} exceeds max {max_position:.3f}"
            
        # Check notional value limit
        max_notional = RISK_CONFIG["limits"]["max_notional"]
        position_value = abs(self.position_size * current_price)
        if position_value > max_notional:
            return False, f"Position value {position_value:.2f} exceeds max {max_notional:.2f}"
            
        # Check dynamic risk limits based on market conditions
        vol_adjusted_max_position = self.get_volatility_adjusted_position_limit()
        hard_vol_limit = vol_adjusted_max_position * RISK_CONFIG["limits"].get("hard_vol_limit_factor", 1.5)
        
        if abs(self.position_size) > hard_vol_limit:
            return False, (f"Position {abs(self.position_size):.3f} exceeds hard volatility limit "
                          f"{hard_vol_limit:.3f}")
                          
        # Check max drawdown
        pnl_pct = self.calculate_pnl_percentage(current_price)
        max_drawdown = RISK_CONFIG["limits"]["max_drawdown"]
        
        if pnl_pct < 0 and abs(pnl_pct) > max_drawdown:
            return False, f"Drawdown {abs(pnl_pct):.2%} exceeds max {max_drawdown:.2%}"
            
        return True, ""
        
    def get_volatility_adjusted_position_limit(self) -> float:
        """
        Calculate volatility-adjusted position limit
        
        Returns:
            Float representing the adjusted position limit
        """
        base_position_limit = RISK_CONFIG["limits"]["max_position"]
        vol_scaling_factor = RISK_CONFIG["limits"].get("vol_scaling_factor", 1.0)
        
        # If we have a volatility estimate, adjust the position limit
        if self.current_volatility > 0:
            # Calculate the reference volatility (baseline)
            baseline_vol = RISK_CONFIG["limits"].get("baseline_volatility", 0.02)  # Default 2%
            
            # Adjust position limit based on ratio of current vol to baseline
            vol_ratio = baseline_vol / self.current_volatility
            adjusted_limit = base_position_limit * vol_ratio * vol_scaling_factor
            
            # Enforce minimum position limit
            min_position_limit = base_position_limit * RISK_CONFIG["limits"].get("min_position_limit_factor", 0.2)
            adjusted_limit = max(adjusted_limit, min_position_limit)
            
            # Enforce maximum position limit
            adjusted_limit = min(adjusted_limit, base_position_limit)
            
            return adjusted_limit
        
        # If no volatility estimate, use base limit
        return base_position_limit
        
    def update_volatility_estimate(self, returns: float) -> None:
        """
        Update the current volatility estimate with a new return observation
        
        Args:
            returns: Most recent return (typically percentage change in price)
        """
        # Add return to history
        self.volatility_history.append(returns)
        
        # Keep history at manageable size
        window_size = RISK_CONFIG["limits"].get("volatility_window", 100)
        if len(self.volatility_history) > window_size:
            self.volatility_history = self.volatility_history[-window_size:]
            
        # Calculate volatility (standard deviation of returns)
        if len(self.volatility_history) >= 10:  # Need some minimum history
            self.current_volatility = np.std(self.volatility_history)
            
            # Log significant volatility changes
            if (self.previous_volatility > 0 and 
                abs(self.current_volatility - self.previous_volatility) / self.previous_volatility > 0.2):
                self.logger.info(
                    f"Volatility changed from {self.previous_volatility:.2%} to {self.current_volatility:.2%} "
                    f"({(self.current_volatility - self.previous_volatility) / self.previous_volatility:+.2%})"
                )
                
            # Store previous for change tracking
            self.previous_volatility = self.current_volatility
        
    def estimate_market_impact(self, position_size: float, current_volume: float) -> float:
        """
        Estimate market impact of a position relative to market volume
        
        Args:
            position_size: Size of position to evaluate
            current_volume: Current market volume (e.g. 24h volume)
            
        Returns:
            Float representing estimated market impact percentage
        """
        if current_volume <= 0:
            return 0.0
            
        # Simple market impact model: position size as percentage of volume
        # multiplied by an impact factor
        impact_factor = RISK_CONFIG["limits"].get("market_impact_factor", 0.5)
        market_impact = (abs(position_size) / current_volume) * impact_factor
        
        self.market_impact = market_impact
        return market_impact

    def check_stop_loss(self, current_price: float) -> bool:
        """
        Check if stop loss should be triggered based on current position and price
        
        Returns:
            Boolean indicating if stop loss has been triggered
        """
        if self.position_size == 0:
            return False
            
        # If emergency stop already triggered, return True
        if self.emergency_stop_triggered:
            return True
            
        # Check if circuit breaker is active
        current_time = time.time()
        if self.circuit_breaker_active and current_time < self.circuit_breaker_until:
            self.logger.info("Circuit breaker active, forcing stop loss")
            self.emergency_stop_triggered = True
            return True
            
        pnl_pct = self.calculate_pnl_percentage(current_price)
        
        # Check stop loss percentage threshold
        stop_loss_pct = RISK_CONFIG["limits"]["stop_loss_pct"]
        if pnl_pct <= -stop_loss_pct:
            self.logger.warning(
                f"Stop loss triggered: PnL {pnl_pct:.2%} below threshold -{stop_loss_pct:.2%}. "
                f"Position: {self.position_size:.3f} @ {self.entry_price:.2f}, current price: {current_price:.2f}"
            )
            return True
            
        # Check drawdown threshold
        max_drawdown = RISK_CONFIG["limits"]["max_drawdown"]
        if self.current_drawdown >= max_drawdown:
            self.logger.warning(
                f"Stop loss triggered: Drawdown {self.current_drawdown:.2%} exceeded max {max_drawdown:.2%}. "
                f"Position: {self.position_size:.3f} @ {self.entry_price:.2f}, current price: {current_price:.2f}"
            )
            return True
            
        # Check time-based stop loss
        max_position_duration = RISK_CONFIG["limits"].get("max_position_duration", 24 * 3600)  # Default 24h
        position_duration = time.time() - self.position_start_time
        
        if position_duration > max_position_duration and pnl_pct < 0:
            self.logger.warning(
                f"Time-based stop loss triggered: Position duration {position_duration/3600:.1f}h exceeded "
                f"max {max_position_duration/3600:.1f}h with negative PnL {pnl_pct:.2%}. "
                f"Position: {self.position_size:.3f} @ {self.entry_price:.2f}, current price: {current_price:.2f}"
            )
            return True
            
        # Check velocity of loss - if price is moving against position too quickly
        # This helps prevent large losses during market gaps or flash crashes
        if len(self.price_history) >= 2 and abs(self.position_size) > 0:
            # Calculate price velocity (change per second)
            recent_price = self.price_history[-1]
            previous_price = self.price_history[-2]
            price_change = (current_price - recent_price) / recent_price
            time_diff = min(10, current_time - self.last_position_update)  # Cap at 10 seconds
            
            if time_diff > 0:
                price_velocity = price_change / time_diff
                
                # If price is moving rapidly against our position
                # For long position, negative velocity is bad; for short position, positive velocity is bad
                velocity_threshold = -0.005 if self.position_size > 0 else 0.005  # 0.5% per second
                
                if (self.position_size > 0 and price_velocity < velocity_threshold) or \
                   (self.position_size < 0 and price_velocity > abs(velocity_threshold)):
                    self.logger.warning(
                        f"Velocity-based stop loss triggered: Price velocity {price_velocity:.4%}/s exceeded threshold "
                        f"of {abs(velocity_threshold):.4%}/s. Position: {self.position_size:.3f} @ {self.entry_price:.2f}"
                    )
                    # Activate circuit breaker to prevent immediate re-entry
                    self.circuit_breaker_active = True
                    self.circuit_breaker_until = time.time() + 300  # 5 minute circuit breaker
                    return True
            
        return False
        
    def check_take_profit(self, current_price: float) -> Tuple[bool, str]:
        """
        Check if take profit should be triggered based on current position and price
        
        Args:
            current_price: Current market price
            
        Returns:
            Tuple of (boolean indicating if take profit triggered, reason string)
        """
        if self.position_size == 0:
            return False, ""
            
        pnl_pct = self.calculate_pnl_percentage(current_price)
        
        # Basic take profit threshold
        take_profit_pct = RISK_CONFIG["limits"]["take_profit_pct"]
        if pnl_pct >= take_profit_pct:
            reason = f"Take profit threshold reached: {pnl_pct:.2%} >= {take_profit_pct:.2%}"
            self.logger.info(
                f"Take profit triggered: {reason}. "
                f"Position: {self.position_size:.3f} @ {self.entry_price:.2f}, current price: {current_price:.2f}"
            )
            return True, reason
            
        # Trailing take profit
        # Track the peak profit we've seen for this position
        if pnl_pct > self.peak_profit:
            self.peak_profit = pnl_pct
            
        # If we've given back too much profit, exit
        # Only apply if we've seen significant profit first
        min_profit_threshold = take_profit_pct * 0.5  # 50% of take profit
        if self.peak_profit > min_profit_threshold:
            profit_drawdown = (self.peak_profit - pnl_pct) / self.peak_profit if self.peak_profit > 0 else 0
            max_profit_drawdown = 0.5  # Don't give back more than 50% of peak profit
            
            if profit_drawdown > max_profit_drawdown:
                reason = (f"Trailing take profit: Given back {profit_drawdown:.2%} of peak profit {self.peak_profit:.2%} "
                         f"(current: {pnl_pct:.2%})")
                self.logger.info(
                    f"Take profit triggered: {reason}. "
                    f"Position: {self.position_size:.3f} @ {self.entry_price:.2f}, current price: {current_price:.2f}"
                )
                return True, reason
            
        # Check time-based take profit - enforce take profit if in position for too long with good PnL
        min_take_profit_pct = RISK_CONFIG["limits"].get("min_take_profit_pct", take_profit_pct / 2)
        max_profit_position_duration = RISK_CONFIG["limits"].get("max_profit_position_duration", 8 * 3600)  # Default 8h
        position_duration = time.time() - self.position_start_time
        
        if (position_duration > max_profit_position_duration and pnl_pct >= min_take_profit_pct):
            reason = (f"Time-based take-profit: Position duration {position_duration/3600:.1f}h exceeded "
                     f"max {max_profit_position_duration/3600:.1f}h with positive PnL {pnl_pct:.2%}")
            self.logger.info(
                f"Take profit triggered: {reason}. "
                f"Position: {self.position_size:.3f} @ {self.entry_price:.2f}, current price: {current_price:.2f}"
            )
            return True, reason
            
        # Check for rapid positive movement that might reverse
        if len(self.price_history) >= 5 and abs(self.position_size) > 0 and pnl_pct > min_take_profit_pct:
            # Calculate short-term price velocity
            current_time = time.time()
            price_change = (current_price - self.price_history[-5]) / self.price_history[-5]
            time_diff = min(60, current_time - self.last_position_update)  # Cap at 60 seconds
            
            if time_diff > 0:
                price_velocity = price_change / time_diff
                
                # If price is moving rapidly in our favor but may reverse
                # For long position, positive velocity is good; for short position, negative velocity is good
                velocity_threshold = 0.01 if self.position_size > 0 else -0.01  # 1% per second
                
                if ((self.position_size > 0 and price_velocity > velocity_threshold) or
                    (self.position_size < 0 and price_velocity < velocity_threshold)) and pnl_pct > min_take_profit_pct:
                    reason = (f"Velocity-based take profit: Price velocity {price_velocity:.4%}/s exceeded threshold "
                             f"of {abs(velocity_threshold):.4%}/s with PnL {pnl_pct:.2%}")
                    self.logger.info(
                        f"Take profit triggered: {reason}. "
                        f"Position: {self.position_size:.3f} @ {self.entry_price:.2f}, current price: {current_price:.2f}"
                    )
                    return True, reason
            
        return False, ""

    def calculate_pnl_percentage(self, current_price: float) -> float:
        """
        Calculate PnL as a percentage of the position value
        
        Args:
            current_price: Current market price
            
        Returns:
            Float representing PnL percentage (positive for profit, negative for loss)
        """
        if self.position_size == 0 or self.entry_price == 0:
            return 0.0
            
        # For long positions: (current_price - entry_price) / entry_price
        # For short positions: (entry_price - current_price) / entry_price
        if self.position_size > 0:
            return (current_price - self.entry_price) / self.entry_price
        else:
            return (self.entry_price - current_price) / self.entry_price

    def should_rebalance(self) -> Tuple[bool, str]:
        """
        Determines if a position rebalance is needed based on various risk metrics.
        
        Returns:
            Tuple of (boolean indicating if rebalance is needed, reason for rebalancing)
        """
        current_time = time.time()
        
        # Check cooldown period
        if self.rebalance_triggered and (current_time - self.last_rebalance_check) < self.rebalance_cooldown:
            return False, ""
            
        # Reset rebalance trigger if cooldown period has passed
        if self.rebalance_triggered and (current_time - self.last_rebalance_check) >= self.rebalance_cooldown:
            self.rebalance_triggered = False
        
        # No position, no need to rebalance
        if self.position_size == 0:
            return False, ""
        
        # Check if position has become too large relative to volatility
        vol_adjusted_max_position = self.get_volatility_adjusted_position_limit()
        if abs(self.position_size) > vol_adjusted_max_position:
            reason = (f"Position {abs(self.position_size):.3f} exceeds volatility-adjusted limit "
                     f"{vol_adjusted_max_position:.3f}")
            return True, reason
            
        # Check utilization thresholds
        position_rebalance_threshold = RISK_CONFIG["limits"].get("position_rebalance_threshold", 0.8)
        if self.position_utilization > position_rebalance_threshold:
            reason = f"Position utilization {self.position_utilization:.2%} > threshold {position_rebalance_threshold:.2%}"
            return True, reason
            
        notional_rebalance_threshold = RISK_CONFIG["limits"].get("notional_rebalance_threshold", 0.8)
        if self.notional_utilization > notional_rebalance_threshold:
            reason = f"Notional utilization {self.notional_utilization:.2%} > threshold {notional_rebalance_threshold:.2%}"
            return True, reason
            
        # Check if market impact is too high
        market_impact_threshold = RISK_CONFIG["limits"].get("market_impact_threshold", 0.0025)  # Default 0.25%
        if self.market_impact > market_impact_threshold:
            reason = f"Market impact {self.market_impact:.2%} > threshold {market_impact_threshold:.2%}"
            return True, reason
            
        # Check duration of position (periodic rebalancing)
        max_position_duration_without_rebalance = RISK_CONFIG["limits"].get("position_rebalance_interval", 4 * 3600)  # Default 4h
        position_duration = current_time - self.position_start_time
        
        if position_duration > max_position_duration_without_rebalance:
            reason = (f"Position duration {position_duration/3600:.1f}h > "
                     f"rebalance interval {max_position_duration_without_rebalance/3600:.1f}h")
            return True, reason
            
        return False, ""

    def get_risk_metrics(self) -> Dict:
        """
        Get comprehensive risk metrics dictionary
        
        Returns:
            Dictionary of current risk metrics
        """
        return {
            "position_size": self.position_size,
            "position_value": self.position_value,
            "entry_price": self.entry_price,
            "position_start_time": self.position_start_time,
            "current_drawdown": self.current_drawdown,
            "max_drawdown": self.max_drawdown,
            "position_utilization": self.position_utilization,
            "notional_utilization": self.notional_utilization,
            "volatility": self.current_volatility,
            "market_impact": self.market_impact,
            "limit_breaches": self.limit_breaches,
            "peak_profit": self.peak_profit,
            "position_duration": time.time() - self.position_start_time if self.position_size != 0 else 0
        }

    async def update_pnl(self, pnl_data: Dict) -> None:
        """
        Update PnL metrics from external source
        
        Args:
            pnl_data: Dictionary containing PnL data (typically from PositionTracker)
        """
        try:
            # Store PnL history
            if "unrealized_pnl_pct" in pnl_data:
                pnl_pct = pnl_data["unrealized_pnl_pct"]
                self.pnl_history.append(pnl_pct)
                
                # Update peak profit tracking
                if pnl_pct > 0 and pnl_pct > self.peak_profit:
                    self.peak_profit = pnl_pct
                    
                # Update drawdown tracking
                if pnl_pct < 0:
                    self.current_drawdown = abs(pnl_pct)
                    self.max_drawdown = max(self.max_drawdown, self.current_drawdown)
                else:
                    self.current_drawdown = 0
                    
                self.drawdown_history.append(self.current_drawdown)
                
            # Trim history if needed
            max_history = 1000
            if len(self.pnl_history) > max_history:
                self.pnl_history = self.pnl_history[-max_history:]
            if len(self.drawdown_history) > max_history:
                self.drawdown_history = self.drawdown_history[-max_history:]
                
        except Exception as e:
            self.logger.error(f"Error updating PnL metrics: {str(e)}")
    
    async def update_trade_metrics(self, trade_data: Dict) -> None:
        """
        Update metrics based on a new trade
        
        Args:
            trade_data: Dictionary containing trade data
        """
        try:
            # Extract trade data
            direction = trade_data.get("direction", "")
            price = float(trade_data.get("price", 0))
            amount = float(trade_data.get("amount", 0))
            timestamp = trade_data.get("timestamp", time.time())
            
            # Skip if missing critical data
            if not direction or price <= 0 or amount <= 0:
                return
                
            # Record trade in history with minimal information
            trade_record = {
                "timestamp": timestamp,
                "direction": direction,
                "price": price,
                "amount": amount,
                "position_size": self.position_size,
                "entry_price": self.entry_price
            }
            
            self.trade_history.append(trade_record)
            
            # Trim history if too large
            if len(self.trade_history) > 1000:
                self.trade_history = self.trade_history[-1000:]
                
        except Exception as e:
            self.logger.error(f"Error updating trade metrics: {str(e)}")

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
            max_position = RISK_CONFIG["limits"]["max_position"]
            if abs(self.position_size) > max_position:
                self.logger.warning(f"Position size limit exceeded: {abs(self.position_size):.3f} > {max_position:.3f}")
                self.limit_breaches["position_size"] += 1
                return False
                
            # 2. Notional value limit
            max_notional = RISK_CONFIG["limits"]["max_notional"]
            if self.position_value > max_notional:
                self.logger.warning(f"Notional value limit exceeded: {self.position_value:.2f} > {max_notional:.2f}")
                self.limit_breaches["notional_value"] += 1
                return False
                
            # 3. Drawdown limit
            max_drawdown = RISK_CONFIG["limits"]["max_drawdown"]
            if self.current_drawdown > max_drawdown:
                self.logger.warning(f"Drawdown limit exceeded: {self.current_drawdown:.2%} > {max_drawdown:.2%}")
                self.limit_breaches["drawdown"] += 1
                return False
                
            # 4. Volatility-adjusted position limit
            vol_adjusted_max_position = self.get_volatility_adjusted_position_limit()
            hard_vol_limit = vol_adjusted_max_position * RISK_CONFIG["limits"].get("hard_vol_limit_factor", 1.5)
            
            if abs(self.position_size) > hard_vol_limit:
                self.logger.warning(
                    f"Volatility-adjusted position limit exceeded: {abs(self.position_size):.3f} > {hard_vol_limit:.3f}"
                )
                self.limit_breaches["position_size"] += 1
                return False
                
            return True
            
        except Exception as e:
            self.logger.error(f"Error checking risk limits: {str(e)}")
            return False  # Default to safety

    def register_callback(self, event_type: str, callback: Callable) -> None:
        """
        Register a callback function for risk events
        
        Args:
            event_type: Type of event ("stop_loss", "take_profit", "rebalance", "risk_limit")
            callback: Function to call when event occurs
        """
        if event_type == "stop_loss":
            self.on_stop_loss_triggered = callback
        elif event_type == "take_profit":
            self.on_take_profit_triggered = callback
        elif event_type == "rebalance":
            self.on_rebalance_needed = callback
        elif event_type == "risk_limit":
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
            "volatility": self.current_volatility,
            "market_impact": self.market_impact,
            "limit_breaches": self.limit_breaches,
            "position_duration": time.time() - self.position_start_time if self.position_size != 0 else 0
        }
    
    def get_position_adjustment_recommendation(self, current_price: float) -> Dict[str, Any]:
        """
        Get recommendation for position size adjustment based on risk metrics
        
        Args:
            current_price: Current market price
            
        Returns:
            Dictionary containing recommendation information
        """
        adjustment_needed = False
        target_position = self.position_size
        reason = ""
        
        # Check if we need rebalancing
        rebalance_needed, rebalance_reason = self.should_rebalance()
        if rebalance_needed:
            adjustment_needed = True
            reason = rebalance_reason
            
            # If position is too large, reduce it
            vol_adjusted_position = self.get_volatility_adjusted_position_limit()
            if abs(self.position_size) > vol_adjusted_position:
                if self.position_size > 0:
                    target_position = vol_adjusted_position
                else:
                    target_position = -vol_adjusted_position
            
            # If utilization is too high, reduce position by 30%
            elif self.position_utilization > RISK_CONFIG["limits"].get("position_rebalance_threshold", 0.8):
                target_position = self.position_size * 0.7
            
            # If market impact is too high, reduce position by 40%
            elif self.market_impact > RISK_CONFIG["limits"].get("market_impact_threshold", 0.0025):
                target_position = self.position_size * 0.6
        
        # Check if stop loss is triggered
        elif self.check_stop_loss(current_price):
            adjustment_needed = True
            target_position = 0
            reason = "Stop loss triggered"
        
        # Check if take profit is triggered
        else:
            take_profit_triggered, take_profit_reason = self.check_take_profit(current_price)
            if take_profit_triggered:
                adjustment_needed = True
                target_position = 0
                reason = take_profit_reason
        
        return {
            "adjustment_needed": adjustment_needed,
            "current_position": self.position_size,
            "target_position": target_position,
            "adjustment_size": target_position - self.position_size,
            "reason": reason
        }
