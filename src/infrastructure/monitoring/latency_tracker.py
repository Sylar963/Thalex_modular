"""
Advanced Monitoring and Latency Tracking for Thalex Modular Trading System
Provides real-time performance metrics, latency tracking, and observability
"""
import asyncio
import time
import logging
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from collections import deque, defaultdict
import statistics
import json
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class LatencySample:
    """Represents a single latency measurement"""
    timestamp: float
    component: str
    operation: str
    duration_ms: float
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceMetrics:
    """Aggregated performance metrics"""
    count: int = 0
    mean: float = 0.0
    median: float = 0.0
    p95: float = 0.0
    p99: float = 0.0
    min: float = float('inf')
    max: float = 0.0
    std_dev: float = 0.0
    samples: List[float] = field(default_factory=list)
    
    def update(self, duration_ms: float):
        """Update metrics with a new sample"""
        self.samples.append(duration_ms)
        self.count = len(self.samples)
        self.mean = statistics.mean(self.samples)
        self.median = statistics.median(self.samples)
        self.p95 = sorted(self.samples)[int(0.95 * len(self.samples))] if self.samples else 0
        self.p99 = sorted(self.samples)[int(0.99 * len(self.samples))] if self.samples else 0
        self.min = min(self.samples) if self.samples else 0
        self.max = max(self.samples) if self.samples else 0
        self.std_dev = statistics.stdev(self.samples) if len(self.samples) > 1 else 0.0


class LatencyTracker:
    """
    Centralized latency tracking system that collects and aggregates performance metrics
    across all system components
    """
    
    def __init__(self, retention_minutes: int = 60, max_samples: int = 10000):
        self.retention_minutes = retention_minutes
        self.max_samples = max_samples
        self.samples: deque = deque(maxlen=max_samples)
        self.metrics: Dict[str, PerformanceMetrics] = defaultdict(PerformanceMetrics)
        self.component_metrics: Dict[str, PerformanceMetrics] = defaultdict(PerformanceMetrics)
        self.operation_metrics: Dict[str, PerformanceMetrics] = defaultdict(PerformanceMetrics)
        
        # Performance counters
        self.counters: Dict[str, int] = defaultdict(int)
        self.gauges: Dict[str, float] = defaultdict(float)
        
        # Alerting thresholds
        self.alert_thresholds: Dict[str, float] = {}
        self.alert_callbacks: List[Callable] = []
        
        # Background task
        self._monitor_task: Optional[asyncio.Task] = None
        self._running = False
        
    def start(self):
        """Start the latency tracking system"""
        self._running = True
        self._monitor_task = asyncio.create_task(self._background_monitor())
        logger.info("Latency tracker started")
        
    def stop(self):
        """Stop the latency tracking system"""
        self._running = False
        if self._monitor_task:
            self._monitor_task.cancel()
        logger.info("Latency tracker stopped")
    
    def record_latency(self, component: str, operation: str, duration_ms: float, 
                      tags: Optional[Dict[str, str]] = None, 
                      metadata: Optional[Dict[str, Any]] = None):
        """Record a latency measurement"""
        sample = LatencySample(
            timestamp=time.time(),
            component=component,
            operation=operation,
            duration_ms=duration_ms,
            tags=tags or {},
            metadata=metadata or {}
        )
        
        # Add to samples
        self.samples.append(sample)
        
        # Update aggregated metrics
        key = f"{component}.{operation}"
        self.metrics[key].update(duration_ms)
        self.component_metrics[component].update(duration_ms)
        self.operation_metrics[operation].update(duration_ms)
        
        # Check for alerts
        self._check_alerts(component, operation, duration_ms)
    
    def record_counter(self, name: str, value: int = 1):
        """Record a counter metric"""
        self.counters[name] += value
    
    def set_gauge(self, name: str, value: float):
        """Set a gauge metric"""
        self.gauges[name] = value
    
    def get_component_metrics(self, component: str) -> PerformanceMetrics:
        """Get metrics for a specific component"""
        return self.component_metrics.get(component, PerformanceMetrics())
    
    def get_operation_metrics(self, operation: str) -> PerformanceMetrics:
        """Get metrics for a specific operation"""
        return self.operation_metrics.get(operation, PerformanceMetrics())
    
    def get_all_metrics(self) -> Dict[str, PerformanceMetrics]:
        """Get all aggregated metrics"""
        return dict(self.metrics)
    
    def set_alert_threshold(self, component: str, operation: str, threshold_ms: float):
        """Set an alert threshold for a specific component and operation"""
        key = f"{component}.{operation}"
        self.alert_thresholds[key] = threshold_ms
    
    def add_alert_callback(self, callback: Callable):
        """Add a callback to be called when alerts are triggered"""
        self.alert_callbacks.append(callback)
    
    def _check_alerts(self, component: str, operation: str, duration_ms: float):
        """Check if a measurement exceeds alert thresholds"""
        key = f"{component}.{operation}"
        threshold = self.alert_thresholds.get(key)
        
        if threshold and duration_ms > threshold:
            alert_msg = f"ALERT: {component}.{operation} exceeded threshold: {duration_ms:.2f}ms > {threshold}ms"
            logger.warning(alert_msg)
            
            # Call alert callbacks
            for callback in self.alert_callbacks:
                try:
                    callback(component, operation, duration_ms, threshold)
                except Exception as e:
                    logger.error(f"Alert callback failed: {e}")
    
    async def _background_monitor(self):
        """Background task to periodically clean up old samples and log summary"""
        while self._running:
            try:
                # Clean up old samples based on retention policy
                cutoff_time = time.time() - (self.retention_minutes * 60)
                while self.samples and self.samples[0].timestamp < cutoff_time:
                    self.samples.popleft()
                
                # Log summary every 30 seconds
                await asyncio.sleep(30)
                self._log_summary()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Background monitor error: {e}")
    
    def _log_summary(self):
        """Log a summary of current metrics"""
        if not self.metrics:
            return
            
        logger.info("=== PERFORMANCE SUMMARY ===")
        for key, metrics in list(self.metrics.items())[-10:]:  # Last 10 metrics
            logger.info(f"{key}: count={metrics.count}, "
                       f"avg={metrics.mean:.3f}ms, "
                       f"p95={metrics.p95:.3f}ms, "
                       f"p99={metrics.p99:.3f}ms, "
                       f"min={metrics.min:.3f}ms, "
                       f"max={metrics.max:.3f}ms")
        
        # Log counters
        if self.counters:
            logger.info("--- COUNTERS ---")
            for name, value in self.counters.items():
                logger.info(f"{name}: {value}")
        
        # Log gauges
        if self.gauges:
            logger.info("--- GAUGES ---")
            for name, value in self.gauges.items():
                logger.info(f"{name}: {value}")


class ComponentLatencyDecorator:
    """
    Decorator to automatically track latency of component methods
    """
    
    def __init__(self, latency_tracker: LatencyTracker, component_name: str):
        self.latency_tracker = latency_tracker
        self.component_name = component_name
    
    def __call__(self, func):
        async def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                end_time = time.perf_counter()
                duration_ms = (end_time - start_time) * 1000
                self.latency_tracker.record_latency(
                    component=self.component_name,
                    operation=func.__name__,
                    duration_ms=duration_ms
                )
        return wrapper


class SystemMonitor:
    """
    Comprehensive system monitor that integrates with all major components
    """
    
    def __init__(self, latency_tracker: LatencyTracker):
        self.latency_tracker = latency_tracker
        self.start_time = time.time()
        
    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health metrics"""
        uptime = time.time() - self.start_time
        
        return {
            "uptime_seconds": uptime,
            "timestamp": datetime.utcnow().isoformat(),
            "samples_collected": len(self.latency_tracker.samples),
            "components_monitored": len(self.latency_tracker.component_metrics),
            "operations_monitored": len(self.latency_tracker.operation_metrics),
            "counters": dict(self.latency_tracker.counters),
            "gauges": dict(self.latency_tracker.gauges)
        }
    
    def get_component_health(self, component: str) -> Dict[str, Any]:
        """Get health metrics for a specific component"""
        metrics = self.latency_tracker.get_component_metrics(component)
        
        return {
            "component": component,
            "metrics": {
                "count": metrics.count,
                "mean_ms": metrics.mean,
                "median_ms": metrics.median,
                "p95_ms": metrics.p95,
                "p99_ms": metrics.p99,
                "min_ms": metrics.min,
                "max_ms": metrics.max,
                "std_dev": metrics.std_dev
            },
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def export_metrics(self, format: str = "json") -> str:
        """Export metrics in various formats"""
        health = self.get_system_health()
        
        if format.lower() == "json":
            return json.dumps(health, indent=2)
        elif format.lower() == "prometheus":
            # Basic Prometheus format
            prometheus_output = []
            for name, value in self.latency_tracker.gauges.items():
                prometheus_output.append(f"# HELP {name} Gauge metric\n{name} {value}\n")
            
            for name, count in self.latency_tracker.counters.items():
                prometheus_output.append(f"# HELP {name}_total Counter metric\n{name}_total {count}\n")
            
            return "".join(prometheus_output)
        else:
            raise ValueError(f"Unsupported format: {format}")


# Context manager for easy latency measurement
class LatencyContext:
    """Context manager for measuring and recording latency"""
    
    def __init__(self, latency_tracker: LatencyTracker, component: str, operation: str,
                 tags: Optional[Dict[str, str]] = None, 
                 metadata: Optional[Dict[str, Any]] = None):
        self.latency_tracker = latency_tracker
        self.component = component
        self.operation = operation
        self.tags = tags
        self.metadata = metadata
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            end_time = time.perf_counter()
            duration_ms = (end_time - self.start_time) * 1000
            self.latency_tracker.record_latency(
                component=self.component,
                operation=self.operation,
                duration_ms=duration_ms,
                tags=self.tags,
                metadata=self.metadata
            )


# Utility function to wrap existing functions with latency tracking
def track_latency(latency_tracker: LatencyTracker, component: str, operation: str):
    """Decorator factory to add latency tracking to any function"""
    def decorator(func):
        if asyncio.iscoroutinefunction(func):
            async def async_wrapper(*args, **kwargs):
                with LatencyContext(latency_tracker, component, operation):
                    return await func(*args, **kwargs)
            return async_wrapper
        else:
            def sync_wrapper(*args, **kwargs):
                with LatencyContext(latency_tracker, component, operation):
                    return func(*args, **kwargs)
            return sync_wrapper
    return decorator


# Example usage and integration points
def integrate_with_thalex_system():
    """
    Example of how to integrate the latency tracker with the Thalex system
    """
    # Create the latency tracker
    tracker = LatencyTracker(retention_minutes=60, max_samples=10000)
    tracker.start()
    
    # Set up alert thresholds
    tracker.set_alert_threshold("thalex_adapter", "place_order", 50.0)  # 50ms threshold
    tracker.set_alert_threshold("quoting_service", "reconcile_orders", 10.0)  # 10ms threshold
    tracker.set_alert_threshold("state_tracker", "submit_order", 1.0)  # 1ms threshold
    
    # Add alert callback
    def alert_handler(component, operation, duration, threshold):
        print(f"CRITICAL: {component}.{operation} latency {duration:.2f}ms exceeded threshold {threshold}ms!")
    
    tracker.add_alert_callback(alert_handler)
    
    # Create system monitor
    monitor = SystemMonitor(tracker)
    
    # Example of decorating methods with latency tracking
    # This would be applied to actual system methods
    # decorated_method = track_latency(tracker, "thalex_adapter", "place_order")(actual_method)
    
    return tracker, monitor


if __name__ == "__main__":
    # Example usage
    tracker = LatencyTracker()
    tracker.start()
    
    # Simulate some operations
    import random
    
    async def simulate_operations():
        for i in range(100):
            # Simulate different operations with varying latencies
            op_type = random.choice(["place_order", "cancel_order", "get_ticker"])
            comp_type = random.choice(["thalex_adapter", "bybit_adapter", "quoting_service"])
            
            # Simulate realistic latencies
            latency = random.uniform(1, 100)  # 1-100ms
            if random.random() < 0.05:  # 5% chance of high latency
                latency = random.uniform(100, 500)  # 100-500ms
            
            tracker.record_latency(comp_type, op_type, latency)
            
            # Update some counters and gauges
            tracker.record_counter(f"{comp_type}_requests")
            tracker.set_gauge(f"{comp_type}_queue_depth", random.randint(0, 10))
            
            await asyncio.sleep(0.1)
    
    async def run_example():
        task = asyncio.create_task(simulate_operations())
        
        # Let it run for a bit
        await asyncio.sleep(5)
        
        # Print some metrics
        print("Component Metrics for thalex_adapter:")
        metrics = tracker.get_component_metrics("thalex_adapter")
        print(f"Count: {metrics.count}, Mean: {metrics.mean:.2f}ms, P99: {metrics.p99:.2f}ms")
        
        # Stop the tracker
        tracker.stop()
        await task
    
    asyncio.run(run_example())