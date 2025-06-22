import time
import threading
from typing import Dict, List, Optional, Any, NamedTuple
from datetime import datetime

from ..thalex_logging import LoggerFactory

class Fill:
    """
    Represents a trade fill for an order.
    """
    def __init__(self, order_id: str, fill_price: float, fill_size: float, 
                 fill_time: datetime, side: str, is_maker: bool = True):
        self.order_id = order_id
        self.fill_price = fill_price
        self.fill_size = fill_size
        self.fill_time = fill_time
        self.side = side.lower()  # Normalize to lowercase ("buy" or "sell")
        self.is_maker = is_maker
        self.pnl = 0.0


class PositionTracker:
    """
    Tracks position, average entry price, and realized PnL.
    Manages position accounting in a more robust way.
    """
    def __init__(self):
        # Initialize logger
        self.logger = LoggerFactory.configure_component_logger(
            "position_tracker",
            log_file="position_tracker.log",
            high_frequency=False
        )
        
        self.current_position = 0.0
        self.average_entry_price = None
        self.weighted_entries = {}  # Dictionary to track entry prices and sizes
        self.realized_pnl = 0.0
        self.unrealized_pnl = 0.0
        self.fills_history = []     # List of all fills
        self.exit_prices = {}       # Exit prices per entry
        self.highest_profit_per_entry = {}  # Track highest profit per entry point
        self.total_volume_traded = 0.0
        self.last_validation_time = 0
        self.validation_interval = 15  # Validate every 15 seconds (reduced from 60)
        self.position_lock = threading.Lock()  # Lock for thread safety
        self.ZERO_THRESHOLD = 1e-6  # More precise zero threshold
        
        self.logger.info("Position tracker initialized")
        
    def update_on_fill(self, fill: Fill):
        """
        Update position and PnL when a fill occurs
        """
        try:
            # Acquire lock for thread safety
            with self.position_lock:
                # Store fill in history
                self.fills_history.append(fill)
                
                # Update total volume traded
                self.total_volume_traded += fill.fill_size
                
                # Determine direction (positive for buys, negative for sells)
                direction = 1.0 if fill.side == "buy" else -1.0
                position_change = direction * fill.fill_size
                
                # Process fill based on whether it's adding to or reducing position
                if (abs(self.current_position) < self.ZERO_THRESHOLD or 
                    (self.current_position > 0 and position_change > 0) or 
                    (self.current_position < 0 and position_change < 0)):
                    # Adding to position
                    self._add_to_position(fill.fill_price, position_change)
                else:
                    # Reducing position (or flipping direction)
                    self._reduce_position(fill.fill_price, position_change)
                
                # Validate position data
                self.validate_position_data()
                
        except Exception as e:
            self.logger.error(f"Error updating position on fill: {str(e)}", exc_info=True)
    
    def _add_to_position(self, price: float, size: float):
        """
        Add to existing position
        """
        try:
            # Update weighted entries
            self.weighted_entries[price] = self.weighted_entries.get(price, 0.0) + abs(size)
            
            # Initialize profit tracking for this entry
            if price not in self.highest_profit_per_entry:
                self.highest_profit_per_entry[price] = 0.0
                
            # Update total position
            old_position = self.current_position
            self.current_position += size
            
            # Recalculate average entry price
            self._recalculate_average_entry()
            
            self.logger.debug(f"Added to position: {old_position:.6f} -> {self.current_position:.6f} @ {price:.2f}")
            
        except Exception as e:
            self.logger.error(f"Error adding to position: {str(e)}", exc_info=True)
    
    def _reduce_position(self, price: float, size: float):
        """
        Reduce position or flip direction (using FIFO accounting)
        """
        try:
            # Abs size is how much to reduce
            abs_size = abs(size)
            remaining_size = abs_size
            realized_pnl = 0.0
            
            # If flipping direction, first close the existing position
            # Then add new position in opposite direction
            if ((self.current_position > 0 and size < 0 and abs_size > self.current_position) or 
                (self.current_position < 0 and size > 0 and abs_size > abs(self.current_position))):
                
                # First, close existing position and calculate P&L
                flip_size = abs(self.current_position)
                flip_direction = -1 if self.current_position > 0 else 1
                
                # Process FIFO exit for the full position
                realized_pnl = self._process_fifo_exit(price, flip_size * flip_direction)
                self.realized_pnl += realized_pnl
                
                # Update remaining size for the new position
                remaining_size = abs_size - flip_size
                
                # Reset position before adding in new direction
                self.current_position = 0
                self.weighted_entries = {}
                
                # Now add the new position in the opposite direction
                if remaining_size > self.ZERO_THRESHOLD:
                    self._add_to_position(price, size / abs_size * remaining_size)
                    
            else:
                # Standard reduction, use FIFO accounting
                realized_pnl = self._process_fifo_exit(price, size)
                self.realized_pnl += realized_pnl
                
                # Update position size
                old_position = self.current_position
                self.current_position += size
                
                # If position is effectively zero, reset tracking
                if abs(self.current_position) < self.ZERO_THRESHOLD:
                    self.reset()
                else:
                    # Recalculate average entry
                    self._recalculate_average_entry()
                    
                self.logger.debug(f"Reduced position: {old_position:.6f} -> {self.current_position:.6f} @ {price:.2f}, P&L: {realized_pnl:.2f}")
                
        except Exception as e:
            self.logger.error(f"Error reducing position: {str(e)}", exc_info=True)
    
    def _process_fifo_exit(self, exit_price: float, exit_size: float) -> float:
        """
        Process a position exit using FIFO accounting
        Returns realized P&L from the exit
        """
        try:
            # Determine if this is a buy or sell exit
            is_buy_exit = exit_size > 0
            remaining_exit = abs(exit_size)
            total_pnl = 0.0
            entries_to_remove = []
            reduced_entries = {}
            
            # Sort entries by time (FIFO)
            sorted_entries = sorted(self.weighted_entries.items())
            
            # Process each entry point
            for entry_price, entry_size in sorted_entries:
                if remaining_exit <= self.ZERO_THRESHOLD:
                    break
                    
                # Calculate how much to exit at this level
                exit_at_level = min(entry_size, remaining_exit)
                remaining_exit -= exit_at_level
                
                # Calculate P&L for this partial exit
                # For long positions: sell_price - buy_price
                # For short positions: buy_price - sell_price
                if (not is_buy_exit and self.current_position > 0) or (is_buy_exit and self.current_position < 0):
                    # Closing long position with sell OR closing short position with buy
                    pnl = (exit_price - entry_price) * exit_at_level if self.current_position > 0 else (entry_price - exit_price) * exit_at_level
                else:
                    # Should not happen in normal FIFO reduction, but handle just in case
                    pnl = 0.0
                
                total_pnl += pnl
                
                # Update or remove entry
                if exit_at_level < entry_size - self.ZERO_THRESHOLD:
                    reduced_entries[entry_price] = entry_size - exit_at_level
                else:
                    entries_to_remove.append(entry_price)
                
                # Record exit price for this entry
                self.exit_prices[entry_price] = exit_price
                
            # Update weighted entries
            for price in entries_to_remove:
                if price in self.weighted_entries:
                    del self.weighted_entries[price]
                    
            # Add back reduced entries
            for price, size in reduced_entries.items():
                self.weighted_entries[price] = size
                
            return total_pnl
            
        except Exception as e:
            self.logger.error(f"Error processing FIFO exit: {str(e)}", exc_info=True)
            return 0.0
    
    def _recalculate_average_entry(self):
        """
        Recalculate weighted average entry price
        """
        try:
            total_size = sum(self.weighted_entries.values())
            
            if total_size <= self.ZERO_THRESHOLD:
                self.average_entry_price = None
                return
                
            total_value = sum(price * size for price, size in self.weighted_entries.items())
            self.average_entry_price = total_value / total_size
            
        except Exception as e:
            self.logger.error(f"Error recalculating average entry: {str(e)}", exc_info=True)
    
    def validate_position_data(self) -> bool:
        """
        Validate position data for consistency
        """
        try:
            current_time = time.time()
            if current_time - self.last_validation_time < self.validation_interval:
                return True  # Skip validation if too recent
                
            self.last_validation_time = current_time
            
            # Check that sum of entry sizes matches position size
            total_size = sum(self.weighted_entries.values())
            if abs(total_size - abs(self.current_position)) > self.ZERO_THRESHOLD:
                self.logger.warning(f"Position size mismatch: tracked={self.current_position:.6f}, calculated={total_size:.6f}")
                
                # Attempt to fix by recalculating
                if abs(self.current_position) > self.ZERO_THRESHOLD:
                    # Normalize weighted entries to match actual position
                    scale_factor = abs(self.current_position) / total_size if total_size > self.ZERO_THRESHOLD else 0
                    if scale_factor > 0:
                        self.weighted_entries = {
                            price: size * scale_factor 
                            for price, size in self.weighted_entries.items()
                        }
                        self._recalculate_average_entry()
                    else:
                        self.reset()
                else:
                    # Reset if position should be zero
                    self.reset()
                
                return False
                
            # Perform sanity check on average entry price
            if self.current_position != 0 and self.average_entry_price is not None:
                recalculated_avg = 0
                if total_size > self.ZERO_THRESHOLD:
                    total_value = sum(price * size for price, size in self.weighted_entries.items())
                    recalculated_avg = total_value / total_size
                    
                if abs(recalculated_avg - self.average_entry_price) > 0.01:  # 1 cent difference
                    self.logger.warning(f"Average entry price mismatch: tracked={self.average_entry_price:.2f}, calculated={recalculated_avg:.2f}")
                    self.average_entry_price = recalculated_avg
                    return False
                
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating position data: {str(e)}", exc_info=True)
            return False
    
    def update_unrealized_pnl(self, current_price: float, perp_name: str = None):
        """
        Update unrealized P&L based on current market price
        
        Args:
            current_price: Current market price
            perp_name: Optional instrument name (for multi-instrument tracking)
        """
        try:
            if abs(self.current_position) < self.ZERO_THRESHOLD or self.average_entry_price is None:
                self.unrealized_pnl = 0.0
                return
                
            if self.current_position > 0:
                # Long position: current_price - entry_price
                self.unrealized_pnl = (current_price - self.average_entry_price) * self.current_position
            else:
                # Short position: entry_price - current_price
                self.unrealized_pnl = (self.average_entry_price - current_price) * abs(self.current_position)
                
        except Exception as e:
            self.logger.error(f"Error updating unrealized PnL: {str(e)}", exc_info=True)
    
    def get_total_pnl(self) -> float:
        """
        Get total P&L (realized + unrealized)
        """
        return self.realized_pnl + self.unrealized_pnl
    
    def reset(self):
        """
        Reset position tracking (but keep P&L history)
        """
        self.current_position = 0.0
        self.average_entry_price = None
        self.weighted_entries = {}
        self.unrealized_pnl = 0.0
        # Don't reset realized_pnl
        self.logger.info("Position tracking reset")
    
    def get_position_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive position metrics
        """
        return {
            "position": self.current_position,
            "average_entry": self.average_entry_price,
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl": self.unrealized_pnl,
            "total_pnl": self.get_total_pnl(),
            "total_volume": self.total_volume_traded,
            "fill_count": len(self.fills_history)
        }
    
    def update_position(self, size: float, price: float):
        """
        Update position without fill details - simpler but less accurate than update_on_fill
        """
        try:
            # Acquire lock for thread safety
            with self.position_lock:
                # Calculate position change
                position_change = size - self.current_position
                
                # Skip if no change
                if abs(position_change) < self.ZERO_THRESHOLD:
                    return
                
                # Process based on whether we're adding to or reducing position
                if (abs(self.current_position) < self.ZERO_THRESHOLD or 
                    (self.current_position > 0 and position_change > 0) or 
                    (self.current_position < 0 and position_change < 0)):
                    # Adding to position
                    self._add_to_position(price, position_change)
                else:
                    # Reducing position
                    self._reduce_position(price, position_change)
                
                # Validate position data
                self.validate_position_data()
                
        except Exception as e:
            self.logger.error(f"Error updating position: {str(e)}", exc_info=True)
            
    def update_position_size(self, size: float):
        """
        Update just the position size without changing entry prices
        Used with fill-based position tracking where entry prices are tracked elsewhere
        """
        try:
            # Acquire lock for thread safety
            with self.position_lock:
                # Skip if no change
                if abs(size - self.current_position) < self.ZERO_THRESHOLD:
                    return
                    
                old_position = self.current_position
                
                # If resetting to zero, clear all tracking data
                if abs(size) < self.ZERO_THRESHOLD:
                    self.reset()
                    self.logger.debug(f"Position reset to zero (from {old_position:.6f})")
                else:
                    # Just update the size without changing entry price info
                    self.current_position = size
                    self.logger.debug(f"Position size updated: {old_position:.6f} -> {size:.6f}")
                    
        except Exception as e:
            self.logger.error(f"Error updating position size: {str(e)}", exc_info=True)


# Thalex fee configuration constants
THALEX_FEES = {
    "maker_fee": 0.0002,  # 0.02%
    "taker_fee": 0.0005,  # 0.05%
    "minimum_fee": 0.0001  # Minimum fee per trade
}


class PortfolioTracker:
    """
    Portfolio-wide position tracker that manages multiple instruments
    using individual PositionTracker instances for each instrument.
    """
    def __init__(self):
        self.logger = LoggerFactory.configure_component_logger(
            "portfolio_tracker", log_file="portfolio_tracker.log", high_frequency=False
        )
        self.instrument_trackers: Dict[str, PositionTracker] = {}  # instrument -> PositionTracker
        self.mark_prices: Dict[str, float] = {}  # instrument -> current_mark_price
        self.trading_fees: Dict[str, float] = {}  # instrument -> accumulated_fees
        self.portfolio_lock = threading.Lock()
        
        self.logger.info("Portfolio tracker initialized")
    
    def register_instrument(self, instrument: str):
        """Create PositionTracker for new instrument"""
        try:
            with self.portfolio_lock:
                if instrument not in self.instrument_trackers:
                    self.instrument_trackers[instrument] = PositionTracker()
                    self.mark_prices[instrument] = 0.0
                    self.trading_fees[instrument] = 0.0
                    self.logger.info(f"Registered instrument: {instrument}")
        except Exception as e:
            self.logger.error(f"Error registering instrument {instrument}: {str(e)}")
    
    def update_position(self, instrument: str, size: float, price: float, fee: float = 0.0):
        """Update position and track fees using existing PositionTracker"""
        try:
            # Ensure instrument is registered
            if instrument not in self.instrument_trackers:
                self.register_instrument(instrument)
            
            # Update position without holding portfolio lock to avoid deadlock
            self.instrument_trackers[instrument].update_position(size, price)
            
            # Track accumulated fees with minimal lock
            with self.portfolio_lock:
                self.trading_fees[instrument] += fee
                
        except Exception as e:
            self.logger.error(f"Error updating position for {instrument}: {str(e)}")
    
    def update_mark_price(self, instrument: str, price: float):
        """Update current market price for P&L calculation"""
        try:
            if instrument in self.mark_prices:
                # Update mark price with lock
                with self.portfolio_lock:
                    self.mark_prices[instrument] = price
                
                # Update unrealized P&L without holding portfolio lock to avoid deadlock
                if instrument in self.instrument_trackers:
                    self.instrument_trackers[instrument].update_unrealized_pnl(price, instrument)
        except Exception as e:
            self.logger.error(f"Error updating mark price for {instrument}: {str(e)}")
    
    def get_total_pnl(self) -> float:
        """Sum of all unrealized + realized P&L across all instruments"""
        try:
            total_pnl = 0.0
            with self.portfolio_lock:
                for instrument, tracker in self.instrument_trackers.items():
                    total_pnl += tracker.get_total_pnl()
            return total_pnl
        except Exception as e:
            self.logger.error(f"Error calculating total P&L: {str(e)}")
            return 0.0
    
    def get_net_pnl_after_fees(self) -> float:
        """Total P&L minus all trading fees"""
        try:
            gross_pnl = self.get_total_pnl()
            total_fees = sum(self.trading_fees.values())
            return gross_pnl - total_fees
        except Exception as e:
            self.logger.error(f"Error calculating net P&L: {str(e)}")
            return 0.0
    
    def calculate_trade_fee(self, notional_value: float, is_maker: bool = True) -> float:
        """Calculate trading fee for a given trade"""
        try:
            if notional_value <= 0:
                return 0.0
            
            # Try to get fee rates from config first, fall back to constants
            try:
                from ..config.market_config import TRADING_CONFIG
                fees_config = TRADING_CONFIG.get("trading_fees", {})
                fee_rate = fees_config.get("maker_fee_rate", THALEX_FEES["maker_fee"]) if is_maker else fees_config.get("taker_fee_rate", THALEX_FEES["taker_fee"])
                min_fee = fees_config.get("minimum_fee_usd", THALEX_FEES["minimum_fee"])
            except ImportError:
                # Fall back to constants if config not available
                fee_rate = THALEX_FEES["maker_fee"] if is_maker else THALEX_FEES["taker_fee"]
                min_fee = THALEX_FEES["minimum_fee"]
            
            calculated_fee = notional_value * fee_rate
            return max(calculated_fee, min_fee)
        except Exception as e:
            self.logger.error(f"Error calculating trade fee: {str(e)}")
            return 0.0
    
    def estimate_closing_fees(self) -> float:
        """Estimate fees required to close all open positions"""
        try:
            total_estimated_fees = 0.0
            
            # Get fee estimation buffer from config
            try:
                from ..config.market_config import TRADING_CONFIG
                fee_buffer = 1.1  # Default fee estimation buffer
            except ImportError:
                fee_buffer = 1.1  # Default 10% buffer
            
            with self.portfolio_lock:
                for instrument, tracker in self.instrument_trackers.items():
                    position_size = tracker.current_position
                    if abs(position_size) > tracker.ZERO_THRESHOLD and instrument in self.mark_prices:
                        mark_price = self.mark_prices[instrument]
                        if mark_price > 0:
                            notional = abs(position_size * mark_price)
                            # Assume maker fees for closing (optimistic) but apply buffer
                            estimated_fee = self.calculate_trade_fee(notional, is_maker=True)
                            total_estimated_fees += estimated_fee * fee_buffer
            
            return total_estimated_fees
        except Exception as e:
            self.logger.error(f"Error estimating closing fees: {str(e)}")
            return 0.0
    
    def get_net_profit_after_all_fees(self) -> float:
        """Get total profit minus all fees (paid + estimated closing fees)"""
        try:
            gross_pnl = self.get_total_pnl()
            paid_fees = sum(self.trading_fees.values())
            estimated_closing_fees = self.estimate_closing_fees()
            return gross_pnl - paid_fees - estimated_closing_fees
        except Exception as e:
            self.logger.error(f"Error calculating net profit after all fees: {str(e)}")
            return 0.0
    
    def get_detailed_fee_breakdown(self) -> Dict[str, Any]:
        """Get comprehensive fee breakdown and profit analysis"""
        try:
            gross_pnl = self.get_total_pnl()
            paid_fees = sum(self.trading_fees.values())
            estimated_closing_fees = self.estimate_closing_fees()
            net_profit = gross_pnl - paid_fees - estimated_closing_fees
            
            breakdown = {
                "gross_pnl": gross_pnl,
                "total_paid_fees": paid_fees,
                "estimated_closing_fees": estimated_closing_fees,
                "total_fee_impact": paid_fees + estimated_closing_fees,
                "net_profit_after_all_fees": net_profit,
                "fee_percentage_of_gross": (paid_fees + estimated_closing_fees) / max(abs(gross_pnl), 0.01) * 100,
                "paid_fees_by_instrument": dict(self.trading_fees),
                "positions_requiring_closure": {}
            }
            
            # Add position-specific closing fee estimates
            with self.portfolio_lock:
                for instrument, tracker in self.instrument_trackers.items():
                    position_size = tracker.current_position
                    if abs(position_size) > tracker.ZERO_THRESHOLD and instrument in self.mark_prices:
                        mark_price = self.mark_prices[instrument]
                        if mark_price > 0:
                            notional = abs(position_size * mark_price)
                            estimated_fee = self.calculate_trade_fee(notional, is_maker=True)
                            breakdown["positions_requiring_closure"][instrument] = {
                                "position_size": position_size,
                                "mark_price": mark_price,
                                "notional_value": notional,
                                "estimated_closing_fee": estimated_fee
                            }
            
            return breakdown
        except Exception as e:
            self.logger.error(f"Error calculating detailed fee breakdown: {str(e)}")
            return {}
    
    def get_portfolio_metrics(self) -> Dict[str, Any]:
        """Get comprehensive portfolio metrics"""
        try:
            # Calculate values that don't require extended lock holding
            total_pnl = self.get_total_pnl()
            net_pnl_after_fees = self.get_net_pnl_after_fees()
            net_profit_after_all_fees = self.get_net_profit_after_all_fees()
            estimated_closing_fees = self.estimate_closing_fees()
            
            # Only hold lock for data extraction
            with self.portfolio_lock:
                metrics = {
                    "instruments": list(self.instrument_trackers.keys()),
                    "total_instruments": len(self.instrument_trackers),
                    "total_pnl": total_pnl,
                    "net_pnl_after_fees": net_pnl_after_fees,
                    "net_profit_after_all_fees": net_profit_after_all_fees,
                    "total_paid_fees": sum(self.trading_fees.values()),
                    "estimated_closing_fees": estimated_closing_fees,
                    "instrument_positions": {},
                    "instrument_pnls": {},
                    "mark_prices": self.mark_prices.copy()
                }
                
                # Add per-instrument details
                for instrument, tracker in self.instrument_trackers.items():
                    metrics["instrument_positions"][instrument] = tracker.current_position
                    metrics["instrument_pnls"][instrument] = tracker.get_total_pnl()
                
                return metrics
        except Exception as e:
            self.logger.error(f"Error getting portfolio metrics: {str(e)}")
            return {}
    
    @property
    def positions(self) -> Dict[str, float]:
        """Get current positions for all instruments"""
        try:
            with self.portfolio_lock:
                return {instrument: tracker.current_position 
                       for instrument, tracker in self.instrument_trackers.items()}
        except Exception as e:
            self.logger.error(f"Error getting positions: {str(e)}")
            return {} 