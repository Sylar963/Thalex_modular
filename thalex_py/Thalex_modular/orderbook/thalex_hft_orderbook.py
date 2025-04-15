"""
Thalex HFT Orderbook Implementation

High-performance orderbook implementation for Thalex exchange using Numba JIT compilation.
This provides efficient processing of Level 2 market data with optimized NumPy arrays.

Key features:
- JIT compilation with Numba for maximum performance
- Normalized price/size representation for optimized calculations
- Efficient binary search for level insertions and updates
- Comprehensive analytics functions (mid price, spread, slippage, etc.)
"""

import numpy as np
from typing import Tuple, Dict, Optional, List, Any
from numba import njit, int64, uint64
from numba.types import uint32, float64, bool_
from numba.experimental import jitclass

from ..logging import LoggerFactory

# Constants for array indices
PRICE_IDX = 0
SIZE_IDX = 1

# Specification for jitclass
spec = [
    ('tick_size', float64),
    ('lot_size', float64),
    ('num_levels', uint32),
    # Using uint64 for price/size to handle larger integers after normalization
    ('asks', uint64[:, :]),
    ('bids', uint64[:, :]),
    ('warmed_up', bool_),
    # Internal state for tracking array lengths
    ('_bid_len', uint32),
    ('_ask_len', uint32),
    ('_max_levels', uint32),  # Store max levels internally
    ('_last_sequence', uint64)  # For tracking sequence numbers
]


@njit
def binary_search_descending(arr: np.ndarray, price: np.uint64, max_len: np.uint32) -> np.int64:
    """Finds index for price in a descending sorted array (bids)."""
    low = 0
    high = max_len - 1
    idx = -1  # Default to -1 (not found)

    while low <= high:
        mid = (low + high) // 2
        if arr[mid, PRICE_IDX] == price:
            return mid
        elif arr[mid, PRICE_IDX] < price:  # Target price is higher, search in the left half
            high = mid - 1
        else:  # Target price is lower, search in the right half
            idx = mid  # Potential insertion point if not found
            low = mid + 1

    # Return insertion point if not found
    return idx + 1


@njit
def binary_search_ascending(arr: np.ndarray, price: np.uint64, max_len: np.uint32) -> np.int64:
    """Finds index for price in an ascending sorted array (asks)."""
    low = 0
    high = max_len - 1
    idx = -1

    while low <= high:
        mid = (low + high) // 2
        if arr[mid, PRICE_IDX] == price:
            return mid
        elif arr[mid, PRICE_IDX] < price:  # Target price is higher, search in the right half
            idx = mid  # Potential insertion point
            low = mid + 1
        else:  # Target price is lower, search in the left half
            high = mid - 1

    # Return insertion point
    return idx + 1


@njit
def _update_side(current_levels: np.ndarray, current_len: np.uint32,
                updates: np.ndarray, is_bid: bool, max_levels: np.uint32) -> np.uint32:
    """
    Update one side of the orderbook (bids or asks).
    
    Args:
        current_levels: Current array of levels
        current_len: Current number of levels
        updates: Array of [price, size] updates
        is_bid: True if processing bids, False for asks
        max_levels: Maximum number of levels to maintain
        
    Returns:
        New length after processing updates
    """
    # Use appropriate search function based on side
    search_func = binary_search_descending if is_bid else binary_search_ascending
    
    # Pre-allocate temporary arrays for operations - speeds up insertions/deletions
    tmp_levels = np.empty((max_levels, 2), dtype=np.uint64)
    tmp_levels[:current_len] = current_levels[:current_len]
    tmp_len = current_len
    
    # Process each update in batches for better performance
    updates_len = updates.shape[0]
    batch_size = min(32, updates_len)  # Process in smaller batches for better cache usage
    
    for batch_start in range(0, updates_len, batch_size):
        batch_end = min(batch_start + batch_size, updates_len)
        
        for i in range(batch_start, batch_end):
            update_price = updates[i, PRICE_IDX]
            update_size = updates[i, SIZE_IDX]
            
            # Find index for this price level
            idx = search_func(tmp_levels, update_price, tmp_len)
            
            # Case 1: Price level exists
            if idx != -1 and idx < tmp_len and tmp_levels[idx, PRICE_IDX] == update_price:
                if update_size == 0:
                    # Delete level (size=0) - use optimized slice operations
                    if idx < tmp_len - 1:
                        tmp_levels[idx:tmp_len-1] = tmp_levels[idx+1:tmp_len]
                    tmp_len -= 1
                else:
                    # Update size for existing level - simple assignment
                    tmp_levels[idx, SIZE_IDX] = update_size
            
            # Case 2: Price level doesn't exist and size > 0 (add new level)
            elif update_size > 0 and tmp_len < max_levels:
                # Make space - use optimized slice operations
                if idx < tmp_len:
                    # Move elements in one operation for better performance
                    tmp_levels[idx+1:tmp_len+1] = tmp_levels[idx:tmp_len]
                
                # Insert new level
                tmp_levels[idx, PRICE_IDX] = update_price
                tmp_levels[idx, SIZE_IDX] = update_size
                tmp_len += 1
    
    # Copy back to original array in one operation
    if tmp_len <= current_len:
        # If the size decreased or stayed the same
        current_levels[:tmp_len] = tmp_levels[:tmp_len]
        # Zero out remaining part if book shrank
        if tmp_len < current_len:
            current_levels[tmp_len:current_len] = 0
    else:
        # If the size increased
        current_levels[:tmp_len] = tmp_levels[:tmp_len]
    
    return tmp_len


@njit
def process_thalex_l2_bids(bids_arr: np.ndarray, bid_len: np.uint32,
                         updates: np.ndarray, max_levels: np.uint32) -> np.uint32:
    """
    Process Thalex bid updates. For bids, we maintain a descending order by price.
    
    Args:
        bids_arr: The current bids array
        bid_len: Current number of bid levels
        updates: Array of [price, size] updates
        max_levels: Maximum number of levels to maintain
        
    Returns:
        New bid length after processing updates
    """
    return _update_side(bids_arr, bid_len, updates, is_bid=True, max_levels=max_levels)


@njit
def process_thalex_l2_asks(asks_arr: np.ndarray, ask_len: np.uint32,
                         updates: np.ndarray, max_levels: np.uint32) -> np.uint32:
    """
    Process Thalex ask updates. For asks, we maintain an ascending order by price.
    
    Args:
        asks_arr: The current asks array
        ask_len: Current number of ask levels
        updates: Array of [price, size] updates
        max_levels: Maximum number of levels to maintain
        
    Returns:
        New ask length after processing updates
    """
    return _update_side(asks_arr, ask_len, updates, is_bid=False, max_levels=max_levels)


@jitclass(spec)
class ThalexHFTOrderbook:
    """
    High-performance orderbook implementation for Thalex exchange.
    
    Uses Numba JIT compilation for maximum performance with Level 2 market data.
    Stores orderbook data in optimized NumPy arrays with normalized prices and sizes.
    """
    
    def __init__(self, tick_size: float, lot_size: float, num_levels: int = 2500):
        """
        Initialize the orderbook with specified parameters.
        
        Args:
            tick_size: Minimum price increment
            lot_size: Minimum size increment
            num_levels: Maximum number of price levels to track (default: 2500)
        """
        self.tick_size = float64(tick_size)
        self.lot_size = float64(lot_size)
        
        # Store as uint32 internally
        self._max_levels = uint32(num_levels)
        self.num_levels = uint32(num_levels)  # Public attribute

        # Initialize arrays with zeros - price and size (2 columns)
        # Using uint64 for potentially large normalized integers
        self.asks = np.zeros((self._max_levels, 2), dtype=np.uint64)
        self.bids = np.zeros((self._max_levels, 2), dtype=np.uint64)

        self.warmed_up = bool_(False)
        self._bid_len = uint32(0)
        self._ask_len = uint32(0)
        self._last_sequence = uint64(0)

    def reset(self):
        """Reset the orderbook state."""
        self.asks[:, :] = 0
        self.bids[:, :] = 0
        self.warmed_up = bool_(False)
        self._bid_len = uint32(0)
        self._ask_len = uint32(0)
        self._last_sequence = uint64(0)

    # ------------------------------------------------------------------------ #
    #                             Normalization                                #
    # ------------------------------------------------------------------------ #
    def _normalize_price(self, price: float) -> np.uint64:
        """Normalize price to integer based on tick size."""
        if self.tick_size == 0 or np.isnan(price) or np.isinf(price): 
            return np.uint64(0)  # Avoid division by zero or invalid values
        # Adding a small epsilon to handle potential floating point inaccuracies
        normalized = round((price + 1e-9) / self.tick_size)
        # Ensure the value is positive and within uint64 range
        if normalized < 0:
            normalized = 0
        return np.uint64(normalized)

    def _normalize_size(self, size: float) -> np.uint64:
        """Normalize size to integer based on lot size."""
        if self.lot_size == 0 or np.isnan(size) or np.isinf(size): 
            return np.uint64(0)  # Avoid division by zero or invalid values
        # Adding a small epsilon to handle potential floating point inaccuracies
        normalized = round((size + 1e-9) / self.lot_size)
        # Ensure the value is positive and within uint64 range
        if normalized < 0:
            normalized = 0
        return np.uint64(normalized)

    def _denormalize_price(self, norm_price: np.uint64) -> float:
        """Convert normalized price back to float."""
        return float64(norm_price) * self.tick_size

    def _denormalize_size(self, norm_size: np.uint64) -> float:
        """Convert normalized size back to float."""
        return float64(norm_size) * self.lot_size

    def _normalize_book_side(self, orderbook_side: np.ndarray) -> np.ndarray:
        """Normalize a list/array of [price, size] pairs."""
        # Create array for normalized data
        normalized_data = np.zeros((orderbook_side.shape[0], 2), dtype=np.uint64)
        for i in range(orderbook_side.shape[0]):
            normalized_data[i, PRICE_IDX] = self._normalize_price(orderbook_side[i, PRICE_IDX])
            normalized_data[i, SIZE_IDX] = self._normalize_size(orderbook_side[i, SIZE_IDX])
        return normalized_data

    def _denormalize_book_side(self, norm_book_side: np.ndarray, length: int) -> np.ndarray:
        """Denormalize internal integer array back to float [price, size] pairs."""
        denormalized_data = np.zeros((length, 2), dtype=np.float64)
        for i in range(length):
            denormalized_data[i, PRICE_IDX] = self._denormalize_price(norm_book_side[i, PRICE_IDX])
            denormalized_data[i, SIZE_IDX] = self._denormalize_size(norm_book_side[i, SIZE_IDX])
        return denormalized_data

    # ------------------------------------------------------------------------ #
    #                        Thalex Update Handling                            #
    # ------------------------------------------------------------------------ #
    def warmup(self, message: Dict) -> None:
        """
        Initialize the orderbook with a snapshot from Thalex.
        
        Args:
            message: The snapshot message containing initial orderbook state
        """
        # Extract data
        data = message.get('data', {})
        
        # Extract sequence if available
        if "sequence" in data:
            self._last_sequence = data["sequence"]
        
        # Extract bid and ask arrays 
        bids_raw = data.get("bids", [])
        asks_raw = data.get("asks", [])
        
        # Skip if data is missing
        if not bids_raw and not asks_raw:
            return
        
        # Process bids
        if bids_raw:
            # Pre-allocate arrays for performance
            tmp_bids = np.zeros((len(bids_raw), 2), dtype=np.float64)
            
            # Convert to normalized values in one pass
            for i, (price_str, size_str) in enumerate(bids_raw):
                price = self._normalize_price(float(price_str))
                size = self._normalize_size(float(size_str))
                
                tmp_bids[i, PRICE_IDX] = price
                tmp_bids[i, SIZE_IDX] = size
            
            # Initialize bid side - fix by passing correct parameters to _update_side
            self._bid_len = process_thalex_l2_bids(self.bids, self._bid_len, tmp_bids, self._max_levels)
        
        # Process asks
        if asks_raw:
            # Pre-allocate arrays for performance
            tmp_asks = np.zeros((len(asks_raw), 2), dtype=np.float64)
            
            # Convert to normalized values in one pass
            for i, (price_str, size_str) in enumerate(asks_raw):
                price = self._normalize_price(float(price_str))
                size = self._normalize_size(float(size_str))
                
                tmp_asks[i, PRICE_IDX] = price
                tmp_asks[i, SIZE_IDX] = size
            
            # Initialize ask side - fix by passing correct parameters to _update_side
            self._ask_len = process_thalex_l2_asks(self.asks, self._ask_len, tmp_asks, self._max_levels)
        
        # Mark as warmed up
        self.warmed_up = True

    def ingest_thalex_update(self, message: Dict) -> None:
        """
        Process a Thalex orderbook update and update the internal state.
        
        Args:
            message: Dict containing the orderbook update from Thalex API
        """
        # Fast path - check if this is an orderbook message
        if message.get('type') != 'l2_updates':
            return
        
        # Extract the data once to avoid repeated dictionary lookups
        data = message.get('data', {})
        
        # Get sequence number only once
        sequence = data.get('sequence', 0)
        
        # Fast path if we're not warmed up yet
        if not self.warmed_up:
            self.warmup(message)
            return
        
        # Check for sequence gaps
        if self._last_sequence > 0 and sequence != self._last_sequence + 1:
            # Log sequence gap and request a fresh snapshot via Exception
            gap_size = sequence - self._last_sequence - 1
            if gap_size > 0:  # Only handle positive gaps (missing messages)
                raise ValueError(f"Sequence gap detected: expected {self._last_sequence + 1}, received {sequence}, missing {gap_size} updates")
        
        # Update sequence tracking
        self._last_sequence = sequence
        
        # Extract ask and bid updates once to avoid repeated dictionary access
        asks = data.get('asks', [])
        bids = data.get('bids', [])
        
        # Process updates only if they exist - avoid function call overhead when empty
        if asks:
            self.process_thalex_l2_asks(asks)
        
        if bids:
            self.process_thalex_l2_bids(bids)

    @njit
    def process_thalex_l2_bids(self, bids: List[List[float]]) -> None:
        """
        Process bid updates from Thalex L2 message.
        
        Args:
            bids: List of [price, size] pairs
        """
        # Pre-allocate temporary arrays for better performance
        tmp = np.zeros((len(bids), 2), dtype=np.float64)
        
        # Convert directly to normalized format in one pass
        for i, (price_str, size_str) in enumerate(bids):
            # Convert strings to floats and normalize in one step
            price = self._normalize_price(float(price_str))
            size = self._normalize_size(float(size_str))
            
            tmp[i, PRICE_IDX] = price
            tmp[i, SIZE_IDX] = size
        
        # Process all updates at once for better cache locality
        self._bid_len = process_thalex_l2_bids(self.bids, self._bid_len, tmp, self._max_levels)
        
    @njit
    def process_thalex_l2_asks(self, asks: List[List[float]]) -> None:
        """
        Process ask updates from Thalex L2 message.
        
        Args:
            asks: List of [price, size] pairs
        """
        # Pre-allocate temporary arrays for better performance
        tmp = np.zeros((len(asks), 2), dtype=np.float64)
        
        # Convert directly to normalized format in one pass
        for i, (price_str, size_str) in enumerate(asks):
            # Convert strings to floats and normalize in one step
            price = self._normalize_price(float(price_str))
            size = self._normalize_size(float(size_str))
            
            tmp[i, PRICE_IDX] = price
            tmp[i, SIZE_IDX] = size
        
        # Process all updates at once for better cache locality
        self._ask_len = process_thalex_l2_asks(self.asks, self._ask_len, tmp, self._max_levels)

    # ------------------------------------------------------------------------ #
    #                             Analytics                                    #
    # ------------------------------------------------------------------------ #
    def get_bids(self) -> np.ndarray:
        """Returns the current bid levels as denormalized [price, size] pairs."""
        return self._denormalize_book_side(self.bids, self._bid_len)

    def get_asks(self) -> np.ndarray:
        """Returns the current ask levels as denormalized [price, size] pairs."""
        return self._denormalize_book_side(self.asks, self._ask_len)

    def get_best_bid(self) -> float:
        """Returns the price of the best bid level."""
        if self._bid_len == 0:
            return 0.0
        return self._denormalize_price(self.bids[0, PRICE_IDX])

    def get_best_ask(self) -> float:
        """Returns the price of the best ask level."""
        if self._ask_len == 0:
            return 0.0
        return self._denormalize_price(self.asks[0, PRICE_IDX])

    def get_best_bid_size(self) -> float:
        """Returns the size of the best bid level."""
        if self._bid_len == 0:
            return 0.0
        return self._denormalize_size(self.bids[0, SIZE_IDX])

    def get_best_ask_size(self) -> float:
        """Returns the size of the best ask level."""
        if self._ask_len == 0:
            return 0.0
        return self._denormalize_size(self.asks[0, SIZE_IDX])

    def get_mid_price(self) -> float:
        """Calculates the mid-price (average of best bid and best ask)."""
        if self._bid_len == 0 or self._ask_len == 0:
            return 0.0  # Cannot calculate if one side is empty

        best_bid_price = self.bids[0, PRICE_IDX]
        best_ask_price = self.asks[0, PRICE_IDX]

        # Denormalize before calculating average
        best_bid_float = self._denormalize_price(best_bid_price)
        best_ask_float = self._denormalize_price(best_ask_price)

        return (best_bid_float + best_ask_float) / 2.0

    def get_wmid_price(self) -> float:
        """Calculates the weighted mid-price."""
        if self._bid_len == 0 or self._ask_len == 0:
            return self.get_mid_price()  # Fallback to mid if book is empty

        best_bid_norm_price = self.bids[0, PRICE_IDX]
        best_bid_norm_size = self.bids[0, SIZE_IDX]
        best_ask_norm_price = self.asks[0, PRICE_IDX]
        best_ask_norm_size = self.asks[0, SIZE_IDX]

        if best_bid_norm_size == 0 or best_ask_norm_size == 0:
            return self.get_mid_price()  # Fallback to mid if sizes are zero

        # Denormalize prices and sizes needed for weighting
        best_bid_price = self._denormalize_price(best_bid_norm_price)
        best_bid_size = self._denormalize_size(best_bid_norm_size)
        best_ask_price = self._denormalize_price(best_ask_norm_price)
        best_ask_size = self._denormalize_size(best_ask_norm_size)

        # Weighted mid = (bid_price * ask_size + ask_price * bid_size) / (bid_size + ask_size)
        return (best_bid_price * best_ask_size + best_ask_price * best_bid_size) / (best_bid_size + best_ask_size)

    def get_spread(self) -> float:
        """Calculates the spread between best ask and best bid."""
        if self._bid_len == 0 or self._ask_len == 0:
            return 0.0  # Cannot calculate if one side is empty

        best_bid_price = self.bids[0, PRICE_IDX]
        best_ask_price = self.asks[0, PRICE_IDX]

        # Denormalize prices
        best_bid_float = self._denormalize_price(best_bid_price)
        best_ask_float = self._denormalize_price(best_ask_price)

        return best_ask_float - best_bid_float

    def get_slippage(self, is_buy: bool, quote_size: float) -> float:
        """
        Calculates the average execution price (slippage) for a market order
        of a given size (in base currency units, e.g., BTC).
        """
        if quote_size <= 0:
            return 0.0

        norm_quote_size = self._normalize_size(quote_size)
        if norm_quote_size == 0:
            return 0.0  # Quote size too small

        side_levels = self.asks if is_buy else self.bids
        side_len = self._ask_len if is_buy else self._bid_len

        if side_len == 0:
            return 0.0  # Cannot calculate if orderbook side is empty

        accum_size = np.uint64(0)
        accum_value = float64(0.0)  # Use float for value accumulation

        for i in range(side_len):
            level_norm_price = side_levels[i, PRICE_IDX]
            level_norm_size = side_levels[i, SIZE_IDX]

            # Denormalize for value calculation
            level_price = self._denormalize_price(level_norm_price)

            fill_size = min(norm_quote_size - accum_size, level_norm_size)
            fill_size_float = self._denormalize_size(fill_size)  # Denormalize fill size

            accum_value += fill_size_float * level_price
            accum_size += fill_size

            if accum_size >= norm_quote_size:
                break

        if accum_size < norm_quote_size:
            # Not enough depth to fill entire quote size
            if accum_size == 0: 
                return 0.0  # Avoid division by zero if nothing could be filled
            return accum_value / self._denormalize_size(accum_size)

        # Return average price
        return accum_value / quote_size

    def get_vamp(self, dollar_depth: float) -> Tuple[float, float]:
        """
        Calculates Volume Adjusted Mid Price (VAMP) up to a certain dollar depth.
        Returns (vamp_bid, vamp_ask)
        """
        if dollar_depth <= 0:
            return 0.0, 0.0

        vamp_bid = self._calculate_vamp_side(self.bids, self._bid_len, dollar_depth)
        vamp_ask = self._calculate_vamp_side(self.asks, self._ask_len, dollar_depth)

        return vamp_bid, vamp_ask

    def _calculate_vamp_side(self, side_levels: np.ndarray, side_len: int, dollar_depth: float) -> float:
        """Helper to calculate VAMP for one side."""
        if side_len == 0:
            return 0.0

        accum_volume_dollars = float64(0.0)
        accum_weighted_price_sum = float64(0.0)
        total_volume_considered_base = float64(0.0)

        for i in range(side_len):
            norm_price = side_levels[i, PRICE_IDX]
            norm_size = side_levels[i, SIZE_IDX]

            price = self._denormalize_price(norm_price)
            size = self._denormalize_size(norm_size)

            level_value_dollars = price * size

            # How much of this level's value can we use?
            value_to_use = min(level_value_dollars, dollar_depth - accum_volume_dollars)

            if value_to_use <= 1e-9:  # Epsilon check
                break  # Reached target dollar depth

            # Calculate the base volume corresponding to value_to_use
            volume_to_use = value_to_use / price if price > 0 else 0

            accum_weighted_price_sum += volume_to_use * price  # Sum of (volume * price)
            total_volume_considered_base += volume_to_use      # Sum of volume
            accum_volume_dollars += value_to_use

            if accum_volume_dollars >= dollar_depth - 1e-9:  # Epsilon check
                break

        # Calculate VWAP for the considered depth
        if total_volume_considered_base <= 1e-9:  # Epsilon check
            # If no volume considered (e.g., depth is smaller than first level), return best price
            return self._denormalize_price(side_levels[0, PRICE_IDX])

        return accum_weighted_price_sum / total_volume_considered_base 