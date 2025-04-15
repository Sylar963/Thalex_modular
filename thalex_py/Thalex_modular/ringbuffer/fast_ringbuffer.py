import numpy as np
from typing import Optional, Tuple, Union
import warnings

class FastRingBuffer:
    """
    High-performance ring buffer implementation using numpy arrays.
    Provides O(1) statistical operations and SIMD-optimized calculations.
    """
    def __init__(
        self,
        capacity: int,
        dtype: np.dtype = np.float64,
        fill_value: Union[int, float] = None
    ):
        """
        Initialize ring buffer with specified capacity.
        
        Args:
            capacity: Size of the buffer
            dtype: Numpy dtype for the buffer
            fill_value: Initial value to fill buffer with
        """
        self.capacity = capacity
        self.dtype = dtype
        
        # Use 0 for integral types or NaN for floating point to avoid casting errors
        if fill_value is None:
            if np.issubdtype(dtype, np.integer):
                fill_value = 0
            else:
                fill_value = np.nan
        
        # Pre-allocate buffer with appropriate fill value
        self._buffer = np.full(capacity, fill_value, dtype=dtype)
        self._index = 0
        self._is_full = False
        
        # Statistical accumulators for O(1) operations
        self._sum = 0.0
        self._sum_sq = 0.0
        self._min = np.inf
        self._max = -np.inf
        
        # Cache for expensive calculations
        self._cache = {
            'mean': None,
            'std': None,
            'var': None,
            'last_update_idx': -1
        }
        
    def append(self, value: Union[float, np.ndarray]) -> None:
        """
        Add value(s) to the buffer.
        
        Args:
            value: Single value or numpy array to append
        """
        if isinstance(value, np.ndarray):
            # Batch append
            n_values = len(value)
            if n_values > self.capacity:
                # If more values than capacity, take the last 'capacity' values
                value = value[-self.capacity:]
                n_values = self.capacity
                
            # Calculate indices for the batch
            indices = np.arange(self._index, self._index + n_values) % self.capacity
            
            # Convert to proper dtype to avoid casting warnings
            if value.dtype != self.dtype:
                value = value.astype(self.dtype, copy=True)
                
            self._buffer[indices] = value
            
            # Update statistical accumulators
            if not self._is_full:
                # Check for floating point or integer types
                if np.issubdtype(self.dtype, np.floating):
                    valid_values = value[~np.isnan(value)]
                    self._sum = np.nansum(self._buffer)
                    self._sum_sq = np.nansum(self._buffer ** 2)
                else:
                    valid_values = value
                    self._sum = np.sum(self._buffer[:self._index + n_values])
                    self._sum_sq = np.sum(self._buffer[:self._index + n_values] ** 2)
                    
                self._min = np.min(valid_values) if len(valid_values) > 0 else self._min
                self._max = np.max(valid_values) if len(valid_values) > 0 else self._max
            else:
                # Update accumulators by removing old values and adding new ones
                old_values = self._buffer[indices]
                
                # For floating point types, handle NaN values
                if np.issubdtype(self.dtype, np.floating):
                    self._sum += np.nansum(value - old_values)
                    self._sum_sq += np.nansum(value ** 2 - old_values ** 2)
                    self._min = np.nanmin(self._buffer)
                    self._max = np.nanmax(self._buffer)
                else:
                    self._sum += np.sum(value - old_values)
                    self._sum_sq += np.sum(value ** 2 - old_values ** 2)
                    self._min = np.min(self._buffer)
                    self._max = np.max(self._buffer)
            
            self._index = (self._index + n_values) % self.capacity
            if not self._is_full and self._index == 0:
                self._is_full = True
                
        else:
            # Single value append - verify proper dtype
            if not isinstance(value, self.dtype):
                try:
                    value = self.dtype(value)
                except (ValueError, TypeError):
                    warnings.warn(f"Could not convert {value} to {self.dtype}. Using default.")
                    # Use appropriate default value for the dtype
                    if np.issubdtype(self.dtype, np.floating):
                        value = np.nan
                    else:
                        value = 0
            
            # Store the old value for statistic updates
            old_value = self._buffer[self._index]
            self._buffer[self._index] = value
            
            # Update statistical accumulators
            is_valid = not np.isnan(value) if np.issubdtype(self.dtype, np.floating) else True
            
            if is_valid:
                if not self._is_full or (np.issubdtype(self.dtype, np.floating) and np.isnan(old_value)):
                    self._sum += value
                    # Handle potential overflow with large timestamp values
                    try:
                        self._sum_sq += value ** 2
                    except OverflowError:
                        # For timestamps, scale down to prevent overflow
                        if value > 1e9:  # Likely a timestamp (seconds since epoch)
                            # Store without squaring for timestamps
                            self._sum_sq = 0.0  # Reset - we'll be focusing on the mean
                else:
                    self._sum += value - old_value
                    # Handle potential overflow with large timestamp values
                    try:
                        self._sum_sq += value ** 2 - old_value ** 2
                    except OverflowError:
                        # For timestamps, scale down to prevent overflow
                        if value > 1e9:  # Likely a timestamp (seconds since epoch)
                            # Store without squaring for timestamps
                            self._sum_sq = 0.0  # Reset - we'll be focusing on the mean
                
                self._min = min(self._min, value)
                self._max = max(self._max, value)
            
            self._index = (self._index + 1) % self.capacity
            if not self._is_full and self._index == 0:
                self._is_full = True
                
        # Invalidate cache
        self._cache['last_update_idx'] = -1
        
    def extend(self, values: np.ndarray) -> None:
        """Extend buffer with multiple values"""
        self.append(values)
        
    @property
    def mean(self) -> float:
        """Calculate mean of buffer values in O(1)"""
        if not self._is_full and self._index == 0:
            return np.nan
            
        if self._cache['last_update_idx'] == self._index:
            return self._cache['mean']
            
        size = self.capacity if self._is_full else self._index
        if size == 0:
            return np.nan
            
        result = self._sum / size
        self._cache['mean'] = result
        self._cache['last_update_idx'] = self._index
        return result
        
    @property
    def std(self) -> float:
        """Calculate standard deviation in O(1)"""
        if not self._is_full and self._index == 0:
            return np.nan
            
        if self._cache['last_update_idx'] == self._index:
            return self._cache['std']
            
        size = self.capacity if self._is_full else self._index
        if size < 2:
            return np.nan
            
        # Calculate variance
        var = (self._sum_sq - (self._sum ** 2) / size) / (size - 1)
        if var < 0:  # Handle numerical precision issues
            var = 0
            
        result = np.sqrt(var)
        self._cache['std'] = result
        self._cache['last_update_idx'] = self._index
        return result
        
    @property
    def min(self) -> float:
        """Get minimum value in O(1)"""
        if not self._is_full and self._index == 0:
            return np.nan
        return self._min
        
    @property
    def max(self) -> float:
        """Get maximum value in O(1)"""
        if not self._is_full and self._index == 0:
            return np.nan
        return self._max
        
    def get_last(self, n: Optional[int] = None) -> np.ndarray:
        """
        Get last n values from buffer.
        If n is None, returns all values.
        """
        if n is None:
            n = self.capacity if self._is_full else self._index
            
        if n > self.capacity:
            n = self.capacity
            
        if self._is_full:
            if n <= self._index:
                return self._buffer[self._index - n:self._index]
            else:
                return np.concatenate([
                    self._buffer[-(n - self._index):],
                    self._buffer[:self._index]
                ])
        else:
            if n <= self._index:
                return self._buffer[self._index - n:self._index]
            else:
                return self._buffer[:self._index]
                
    def to_array(self) -> np.ndarray:
        """Convert buffer to numpy array"""
        if self._is_full:
            return np.concatenate([
                self._buffer[self._index:],
                self._buffer[:self._index]
            ])
        return self._buffer[:self._index].copy()
        
    def clear(self) -> None:
        """Clear the buffer"""
        self._buffer.fill(np.nan)
        self._index = 0
        self._is_full = False
        self._sum = 0.0
        self._sum_sq = 0.0
        self._min = np.inf
        self._max = -np.inf
        self._cache['last_update_idx'] = -1
        
    def __len__(self) -> int:
        """Get number of valid values in buffer"""
        return self.capacity if self._is_full else self._index
        
    def __getitem__(self, idx: Union[int, slice]) -> Union[float, np.ndarray]:
        """Get value(s) at specified index/slice"""
        if isinstance(idx, slice):
            return self.to_array()[idx]
        else:
            if idx < 0:
                idx = len(self) + idx
            if idx >= len(self):
                raise IndexError("Buffer index out of range")
            return self._buffer[(self._index - len(self) + idx) % self.capacity] 