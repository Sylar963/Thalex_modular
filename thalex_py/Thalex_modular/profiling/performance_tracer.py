import time
import contextlib
from collections import defaultdict
import threading
import numpy as np
from typing import Dict, List, Tuple, Optional, Any


class PerformanceTracer:
    """
    Performance profiler for high-frequency trading operations.
    Keeps track of execution times for critical paths and provides
    insights for dynamic optimization.
    """
    
    def __init__(self, max_samples: int = 1000):
        """
        Initialize performance tracer.
        
        Args:
            max_samples: Maximum number of samples to keep per trace point
        """
        # Dictionary to store timing data in ring buffers
        self.traces = defaultdict(lambda: np.zeros(max_samples))
        
        # Indices for ring buffers
        self.indices = defaultdict(int)
        
        # Sample counts per trace point
        self.counts = defaultdict(int)
        
        # Maximum samples to keep
        self.max_samples = max_samples
        
        # Lock for thread safety
        self.lock = threading.RLock()

    @contextlib.contextmanager
    def trace(self, name: str):
        """
        Context manager for tracing execution time of a code block.
        
        Args:
            name: Name of the trace point
        """
        start_time = time.perf_counter()
        try:
            yield
        finally:
            # Calculate elapsed time in microseconds
            elapsed = (time.perf_counter() - start_time) * 1000
            
            # Update trace data
            with self.lock:
                idx = self.indices[name]
                self.traces[name][idx] = elapsed
                self.indices[name] = (idx + 1) % self.max_samples
                self.counts[name] = min(self.counts[name] + 1, self.max_samples)

    def get_stats(self, name: str) -> Dict[str, float]:
        """
        Get statistics for a specific trace point.
        
        Args:
            name: Name of the trace point
            
        Returns:
            Dictionary with performance statistics
        """
        with self.lock:
            if name not in self.traces:
                return {"count": 0, "mean": 0, "min": 0, "max": 0, "p95": 0, "p99": 0}
            
            count = self.counts[name]
            if count == 0:
                return {"count": 0, "mean": 0, "min": 0, "max": 0, "p95": 0, "p99": 0}
            
            # Get active data
            data = self.traces[name][:count]
            
            return {
                "count": count,
                "mean": float(np.mean(data)),
                "min": float(np.min(data)),
                "max": float(np.max(data)),
                "p95": float(np.percentile(data, 95)),
                "p99": float(np.percentile(data, 99))
            }

    def get_all_stats(self) -> Dict[str, Dict[str, float]]:
        """
        Get statistics for all trace points.
        
        Returns:
            Dictionary with performance statistics for all trace points
        """
        with self.lock:
            return {name: self.get_stats(name) for name in self.traces.keys()}

    def get_critical_paths(self, threshold_ms: float = 1.0) -> List[Tuple[str, float]]:
        """
        Get list of critical paths based on performance metrics.
        
        Args:
            threshold_ms: Threshold in milliseconds for considering a path critical
            
        Returns:
            List of tuples (path_name, average_time_ms) sorted by average time
        """
        with self.lock:
            critical_paths = []
            
            for name in self.traces.keys():
                stats = self.get_stats(name)
                
                # Check if this is a critical path
                if stats["count"] >= 10 and stats["mean"] > threshold_ms:
                    critical_paths.append((name, stats["mean"]))
            
            # Sort by average time (descending)
            critical_paths.sort(key=lambda x: x[1], reverse=True)
            
            return critical_paths

    def reset(self):
        """Reset all trace data"""
        with self.lock:
            self.traces = defaultdict(lambda: np.zeros(self.max_samples))
            self.indices = defaultdict(int)
            self.counts = defaultdict(int)

    def reset_trace(self, name: str):
        """
        Reset a specific trace point.
        
        Args:
            name: Name of the trace point to reset
        """
        with self.lock:
            if name in self.traces:
                self.traces[name] = np.zeros(self.max_samples)
                self.indices[name] = 0
                self.counts[name] = 0 