from typing import Optional, Tuple, Dict, List, Callable, Any
import numpy as np
from collections import deque
import time
import asyncio

from ..config.market_config import (
    RISK_LIMITS,
    TRADING_CONFIG
)
from ..models.data_models import Ticker, Order
from ..thalex_logging import LoggerFactory
from ..models.position_tracker import PositionTracker

# Define a tolerance for floating point comparisons (e.g., for position size)
ZERO_THRESHOLD = 1e-9

class RiskManager:
    def __init__(self, position_tracker: PositionTracker) -> None:
        """Initialize RiskManager with simplified position tracking and risk metrics"""
        # Initialize logger
        self.logger = LoggerFactory.configure_component_logger(
            "risk_manager",
            log_file="risk_manager.log",
            high_frequency=False
        )
        
        self.position_tracker = position_tracker
        
        # Load risk parameters from config
        self.max_position_abs = RISK_LIMITS.get("max_position", float('inf'))
        self.max_notional_value = RISK_LIMITS.get("max_notional", float('inf'))
        self.stop_loss_pct = RISK_LIMITS.get("stop_loss_pct")
        self.take_profit_pct = RISK_LIMITS.get("take_profit_pct")
        self.max_drawdown = RISK_LIMITS.get("max_drawdown")
        self.max_consecutive_losses = RISK_LIMITS.get("max_consecutive_losses")
        
        # Position entry time for time-based take profit - To be revisited in Step 3
        self.position_entry_time_for_current_cycle: Optional[float] = None
        
        # Callbacks for risk events
        self.callbacks: Dict[str, Callable[[str, Optional[float]], Any]] = {}
        
        self.logger.info("Simplified risk manager initialized")
        
        # Validate core risk parameters
        self._validate_risk_parameters()

    def _validate_risk_parameters(self) -> None:
        """Validate core risk parameters to ensure they are within reasonable ranges"""
        try:
            # Check position limit
            if self.max_position_abs <= 0:
                self.logger.warning(f"Invalid max_position: {self.max_position_abs}, using default of 1.0")
                self.max_position_abs = 1.0
            
            # Check notional limit
            if self.max_notional_value <= 0:
                self.logger.warning(f"Invalid max_notional: {self.max_notional_value}, using default of 50000")
                self.max_notional_value = 50000
            
            # Check stop loss percentage
            if self.stop_loss_pct is not None and (self.stop_loss_pct <= 0 or self.stop_loss_pct > 0.5):
                self.logger.warning(f"Unusual stop_loss_pct: {self.stop_loss_pct}, recommended range is 0.01-0.1")
            
            # Check take profit percentage
            if self.take_profit_pct is not None and (self.take_profit_pct <= 0 or self.take_profit_pct > 0.5):
                self.logger.warning(f"Unusual take_profit_pct: {self.take_profit_pct}, recommended range is 0.01-0.1")
            
            # Check max drawdown
            if self.max_drawdown is not None and (self.max_drawdown <= 0 or self.max_drawdown > 0.5):
                self.logger.warning(f"Unusual max_drawdown: {self.max_drawdown}, recommended range is 0.05-0.2")
                
            self.logger.info("Risk parameters validated")
        except Exception as e:
            self.logger.error(f"Error validating risk parameters: {str(e)}")

    def check_position_limits(self, current_price: float) -> Tuple[bool, str]:
        """
        Check if current position size or notional value exceeds defined limits.
        Uses PositionTracker for current position data.
        """
        metrics = self.position_tracker.get_position_metrics()
        current_pos_size = metrics.get("position", 0.0)
        
        # Check absolute position size limit
        if abs(current_pos_size) > self.max_position_abs:
            reason = f"Absolute position size |{current_pos_size:.4f}| exceeds limit {self.max_position_abs:.4f}"
            self.logger.warning(reason)
            return True, reason

        # Check notional value limit
        if current_price > 0: # Only check if price is valid
            notional_value = abs(current_pos_size) * current_price
            if notional_value > self.max_notional_value:
                reason = f"Position notional value {notional_value:.2f} USD exceeds limit {self.max_notional_value:.2f} USD"
                self.logger.warning(reason)
                return True, reason
        
        return False, ""

    def calculate_pnl_percentage(self, current_price: float) -> float:
        """
        Calculate the P&L percentage of the current position.
        Uses PositionTracker for average entry price and position size.
        """
        metrics = self.position_tracker.get_position_metrics()
        avg_entry_price = metrics.get("average_entry")
        current_pos_size = metrics.get("position", 0.0)

        if abs(current_pos_size) < ZERO_THRESHOLD or avg_entry_price is None or avg_entry_price == 0:
            return 0.0

        if current_pos_size > 0:  # Long position
            pnl_per_unit = current_price - avg_entry_price
        else:  # Short position
            pnl_per_unit = avg_entry_price - current_price
        
        # Total PnL for the current position based on its average entry
        current_unrealized_pnl = pnl_per_unit * abs(current_pos_size)
        
        # Cost basis of the current position
        cost_basis = avg_entry_price * abs(current_pos_size)
        if abs(cost_basis) < ZERO_THRESHOLD:
            return 0.0
            
        return current_unrealized_pnl / cost_basis

    def check_stop_loss(self, current_price: float) -> bool:
        """
        Check if stop-loss condition is met based on P&L percentage.
        """
        if self.stop_loss_pct is None:
            return False # Stop loss is not configured

        metrics = self.position_tracker.get_position_metrics()
        current_pos_size = metrics.get("position", 0.0)

        if abs(current_pos_size) < ZERO_THRESHOLD:
            return False # No position to stop out

        # Update unrealized PnL in PositionTracker before checking
        self.position_tracker.update_unrealized_pnl(current_price)
        # Fetch updated metrics
        metrics_after_upnl_update = self.position_tracker.get_position_metrics()
        unrealized_pnl_value = metrics_after_upnl_update.get("unrealized_pnl", 0.0)
        avg_entry_price = metrics_after_upnl_update.get("average_entry")

        if avg_entry_price is None or abs(avg_entry_price * current_pos_size) < ZERO_THRESHOLD : # Avoid division by zero if cost basis is zero
            self.logger.debug("Stop-loss check: No valid cost basis for PnL calculation.")
            return False

        # PnL percentage is calculated as UnrealizedPnL / CostBasis
        # CostBasis = abs(position_size * average_entry_price)
        cost_basis = abs(current_pos_size * avg_entry_price)
        if cost_basis == 0: # Should be caught by above, but defensive check
             return False
        
        current_pnl_pct = unrealized_pnl_value / cost_basis
        
        # Stop loss is triggered if P&L percentage is less than or equal to negative stop_loss_pct
        if current_pnl_pct <= -self.stop_loss_pct:
            self.logger.warning(
                f"Stop-loss triggered. Position: {current_pos_size:.4f}, Entry: {avg_entry_price:.2f}, "
                f"Current Price: {current_price:.2f}, P&L %: {current_pnl_pct:.4%}, Stop Loss %: {-self.stop_loss_pct:.2%}"
            )
            return True
        return False

    def check_take_profit(self, current_price: float) -> Tuple[bool, str, float]:
        """
        Check if take-profit conditions are met.
        Currently only checks P&L percentage based take profit.
        Time-based take-profit needs to be re-evaluated in Step 3.
        Returns: (triggered, reason, portion_to_close) - portion_to_close is always 1.0 for now
        """
        if self.take_profit_pct is None:
            return False, "", 0.0 # Take profit P&L not configured

        metrics = self.position_tracker.get_position_metrics()
        current_pos_size = metrics.get("position", 0.0)

        if abs(current_pos_size) < ZERO_THRESHOLD:
            return False, "", 0.0 # No position

        # Update unrealized PnL in PositionTracker before checking
        self.position_tracker.update_unrealized_pnl(current_price)
        # Fetch updated metrics
        metrics_after_upnl_update = self.position_tracker.get_position_metrics()
        unrealized_pnl_value = metrics_after_upnl_update.get("unrealized_pnl", 0.0)
        avg_entry_price = metrics_after_upnl_update.get("average_entry")

        if avg_entry_price is None or abs(avg_entry_price * current_pos_size) < ZERO_THRESHOLD:
            self.logger.debug("Take-profit check: No valid cost basis for PnL calculation.")
            return False, "", 0.0
        
        cost_basis = abs(current_pos_size * avg_entry_price)
        if cost_basis == 0:
            return False, "", 0.0

        current_pnl_pct = unrealized_pnl_value / cost_basis

        # P&L based take profit
        if self.take_profit_pct is not None and current_pnl_pct >= self.take_profit_pct:
            reason = (
                f"P&L take profit triggered. Position: {current_pos_size:.4f}, Entry: {avg_entry_price:.2f}, "
                f"Current Price: {current_price:.2f}, P&L %: {current_pnl_pct:.4%}, Target TP %: {self.take_profit_pct:.2%}"
            )
            self.logger.info(reason)
            return True, reason, 1.0 # Close entire position

        # Time-based take profit (Placeholder - to be refined in Step 3)
        # if self.position_entry_time_for_current_cycle and self.take_profit_duration_seconds:
        #     position_duration = time.time() - self.position_entry_time_for_current_cycle
        #     if position_duration >= self.take_profit_duration_seconds:
        #         # Ensure minimum profit for time-based closure, e.g. PNL % > 0.001 (0.1%)
        #         if current_pnl_pct > RISK_LIMITS.get("min_profit_for_timed_exit_pct", 0.001):
        #             reason = (
        #                 f"Time-based take profit triggered. Duration: {position_duration:.0f}s >= {self.take_profit_duration_seconds}s "
        #                 f"with P&L %: {current_pnl_pct:.4%}"
        #             )
        #             self.logger.info(reason)
        #             return True, reason, 1.0 # Close entire position
        
        return False, "", 0.0

    async def check_risk_limits(self, current_market_price: Optional[float] = None) -> bool:
        """
        Centralized method to check all relevant risk limits.
        Returns True if NO limits are breached, False if ANY limit is breached.
        """
        if current_market_price is None:
            self.logger.warning("Risk check: current_market_price is None. Cannot perform all checks.")
            # Potentially allow some checks that don't need price, or return False to be safe.
            # For now, if price is critical for most checks, let's consider it a failure.
            if self.callbacks.get("risk_limit"):
                asyncio.create_task(self.callbacks["risk_limit"]("No market price for risk check", None))
            return False # Cannot reliably check all limits

        # 1. Check Position Limits (Size and Notional Value)
        limit_exceeded, reason = self.check_position_limits(current_market_price)
        if limit_exceeded:
            self.logger.critical(f"Risk Breach: {reason}")
            if self.callbacks.get("risk_limit"):
                 # If callback is async, ensure it's awaited or created as a task
                callback_fn = self.callbacks["risk_limit"]
                if asyncio.iscoroutinefunction(callback_fn):
                    asyncio.create_task(callback_fn(reason, current_market_price))
                else:
                    callback_fn(reason, current_market_price) # Assume sync
            return False # Limit breached

        # If in a position, check for Stop-Loss and Take-Profit
        metrics = self.position_tracker.get_position_metrics()
        current_pos_size = metrics.get("position", 0.0)

        if abs(current_pos_size) > ZERO_THRESHOLD: # Only if there's an open position
            # 2. Check Stop Loss
            if self.check_stop_loss(current_market_price):
                # The reason is logged within check_stop_loss.
                # The callback here indicates a general risk breach due to stop-loss.
                if self.callbacks.get("risk_limit"):
                    reason_sl = f"Stop-loss triggered at price {current_market_price:.2f}"
                    asyncio.create_task(self.callbacks["risk_limit"](reason_sl, current_market_price))
                return False # Stop-loss triggered, considered a risk breach

            # 3. Check Take Profit (if configured to also halt/notify on TP)
            # Current AvellanedaQuoter handles TP closures and then calls _handle_risk_breach.
            # So, we might not need to explicitly return False from here on TP,
            # but the check can be performed for logging or other actions.
            # For consistency, if a TP action leads to halting, it's a "risk management action".
            # tp_triggered, tp_reason, _ = self.check_take_profit(current_market_price)
            # if tp_triggered:
            #     self.logger.info(f"Risk Info: Take-profit condition met: {tp_reason}")
            #     # Depending on bot policy, TP might not be a "breach" that stops all quoting
            #     # but rather a desired outcome. If it implies halting, then:
            #     # if self.callbacks.get("risk_limit"):
            #     #     asyncio.create_task(self.callbacks["risk_limit"](tp_reason, current_market_price))
            #     # return False 

        # Add checks for max_drawdown, max_consecutive_losses if implemented
        # For max_drawdown, you'd need to track portfolio value over time.
        # For max_consecutive_losses, you'd need to track outcomes of individual trades/cycles.
        # These are more complex and are not fully implemented with PositionTracker alone yet.

        return True # No limits breached

    def register_callback(self, event_type: str, callback: Callable[[str, Optional[float]], Any]) -> None:
        """Register a callback for a specific risk event type."""
        self.callbacks[event_type] = callback
        self.logger.info(f"Callback registered for risk event: {event_type}")

    def get_risk_summary(self) -> Dict[str, Any]:
        """
        Provide a summary of current risk exposure and P&L.
        """
        metrics = self.position_tracker.get_position_metrics()
        current_pos_size = metrics.get("position", 0.0)
        avg_entry = metrics.get("average_entry")
        real_pnl = metrics.get("realized_pnl", 0.0)
        unreal_pnl = metrics.get("unrealized_pnl", 0.0) # Will be updated by AvellanedaQuoter

        summary = {
            "current_position_size": current_pos_size,
            "average_entry_price": avg_entry if avg_entry is not None else 0.0,
            "realized_pnl": real_pnl,
            "unrealized_pnl": unreal_pnl, # This needs current market price to be accurate
            "max_position_limit": self.max_position_abs,
            "max_notional_limit": self.max_notional_value,
            "stop_loss_threshold_pct": self.stop_loss_pct,
            "take_profit_threshold_pct": self.take_profit_pct,
            # "active_stop_loss_price": self.calculate_stop_loss_price(), # Would need current_price to calc
            # "active_take_profit_price": self.calculate_take_profit_price() # Would need current_price
        }
        return summary

    # Helper method to potentially get/set the position_entry_time_for_current_cycle
    # This is a placeholder concept for how time-based logic might be managed.
    def set_position_entry_time_for_cycle(self, entry_time: Optional[float]):
        """
        Sets the entry time for the current position cycle.
        This is a conceptual method for time-based rule management and will be refined.
        """
        self.position_entry_time_for_current_cycle = entry_time
        if entry_time:
            self.logger.info(f"Position entry time for current cycle set to: {time.ctime(entry_time)}")
        else:
            self.logger.info("Position entry time for current cycle cleared.")

    def get_position_entry_time_for_cycle(self) -> Optional[float]:
        """Gets the entry time for the current position cycle."""
        return self.position_entry_time_for_current_cycle
