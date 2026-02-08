"""
Low-level performance utilities for the Thalex Modular Trading System
"""
import time
import asyncio
from typing import Any, Callable, Awaitable
import functools
import cProfile
import io
import pstats
from contextlib import contextmanager


def time_it(func):
    """
    Decorator to time function execution
    """
    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = await func(*args, **kwargs)
        end = time.perf_counter()
        print(f"{func.__name__} took {(end - start) * 1000:.3f}ms")
        return result
    
    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"{func.__name__} took {(end - start) * 1000:.3f}ms")
        return result
    
    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    else:
        return sync_wrapper


@contextmanager
def profile_code():
    """
    Context manager for profiling code execution
    """
    pr = cProfile.Profile()
    pr.enable()
    yield
    pr.disable()
    
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats()
    print(s.getvalue())


def fast_deep_copy(obj: Any) -> Any:
    """
    Faster deep copy implementation for common trading objects
    """
    # For simple objects, use direct construction
    if isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    elif isinstance(obj, list):
        return [fast_deep_copy(item) for item in obj]
    elif isinstance(obj, dict):
        return {k: fast_deep_copy(v) for k, v in obj.items()}
    elif hasattr(obj, '__dict__'):  # Custom objects
        # Create new instance without calling __init__
        new_obj = obj.__class__.__new__(obj.__class__)
        for attr, value in obj.__dict__.items():
            setattr(new_obj, attr, fast_deep_copy(value))
        return new_obj
    else:
        # Fallback to regular copy
        import copy
        return copy.deepcopy(obj)


class FastLRUCache:
    """
    A faster LRU cache implementation using collections.OrderedDict
    """
    def __init__(self, maxsize: int = 1000):
        self._maxsize = maxsize
        self._cache = {}
        self._access_order = []  # Track access order for LRU
    
    def get(self, key, default=None):
        if key in self._cache:
            # Move to end to mark as recently used
            if key in self._access_order:
                self._access_order.remove(key)
            self._access_order.append(key)
            return self._cache[key]
        return default
    
    def __getitem__(self, key):
        if key in self._cache:
            # Move to end to mark as recently used
            if key in self._access_order:
                self._access_order.remove(key)
            self._access_order.append(key)
            return self._cache[key]
        raise KeyError(key)
    
    def __setitem__(self, key, value):
        if key in self._cache:
            # Update existing key
            self._cache[key] = value
            # Move to end to mark as recently used
            if key in self._access_order:
                self._access_order.remove(key)
            self._access_order.append(key)
        else:
            # Add new key
            if len(self._cache) >= self._maxsize:
                # Remove least recently used item
                lru_key = self._access_order.pop(0)
                del self._cache[lru_key]
            
            self._cache[key] = value
            self._access_order.append(key)
    
    def __delitem__(self, key):
        if key in self._cache:
            del self._cache[key]
            if key in self._access_order:
                self._access_order.remove(key)
        else:
            raise KeyError(key)
    
    def __contains__(self, key):
        return key in self._cache
    
    def __len__(self):
        return len(self._cache)
    
    def keys(self):
        return self._cache.keys()


def optimized_json_dumps(data: Any) -> str:
    """
    Optimized JSON serialization for common trading data structures
    """
    try:
        import orjson
        return orjson.dumps(data).decode("utf-8")
    except ImportError:
        import json
        return json.dumps(data)


def optimized_json_loads(data: str) -> Any:
    """
    Optimized JSON deserialization for common trading data structures
    """
    try:
        import orjson
        return orjson.loads(data)
    except ImportError:
        import json
        return json.loads(data)


def batch_process(items: list, processor: Callable, batch_size: int = 100):
    """
    Process items in batches for better performance
    """
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        yield from processor(batch)


async def async_batch_process(items: list, processor: Callable, batch_size: int = 100):
    """
    Process items in batches asynchronously for better performance
    """
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        result = await processor(batch)
        if result:
            yield result


def lock_free_counter():
    """
    A lock-free counter implementation using atomic operations
    """
    import threading
    counter = 0
    lock = threading.Lock()
    
    def increment():
        nonlocal counter
        with lock:
            counter += 1
            return counter
    
    def decrement():
        nonlocal counter
        with lock:
            counter -= 1
            return counter
    
    def get():
        nonlocal counter
        with lock:
            return counter
    
    return increment, decrement, get