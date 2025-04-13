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
    
    def update_unrealized_pnl(self, current_price: float):
        """
        Update unrealized P&L based on current market price
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