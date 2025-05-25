"""
Component Health Monitoring System - Added 2024-12-19

This module provides comprehensive health monitoring for all trading system components,
including performance metrics, error tracking, and automated recovery mechanisms.
"""

import asyncio
import logging
import time
import psutil
import traceback
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict

from .event_bus import get_event_bus, EventType, Event


class ComponentStatus(Enum):
    """Component status enumeration"""
    STARTING = "starting"
    HEALTHY = "healthy"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    STOPPED = "stopped"
    UNKNOWN = "unknown"


class HealthMetric(Enum):
    """Health metric types"""
    CPU_USAGE = "cpu_usage"
    MEMORY_USAGE = "memory_usage"
    EVENT_RATE = "event_rate"
    ERROR_RATE = "error_rate"
    RESPONSE_TIME = "response_time"
    UPTIME = "uptime"
    LAST_HEARTBEAT = "last_heartbeat"


@dataclass
class ComponentHealth:
    """Component health information"""
    name: str
    status: ComponentStatus = ComponentStatus.UNKNOWN
    last_heartbeat: float = field(default_factory=time.time)
    start_time: float = field(default_factory=time.time)
    error_count: int = 0
    warning_count: int = 0
    last_error: Optional[str] = None
    last_error_time: Optional[float] = None
    metrics: Dict[str, float] = field(default_factory=dict)
    performance_history: Dict[str, deque] = field(default_factory=lambda: defaultdict(lambda: deque(maxlen=100)))
    
    @property
    def uptime(self) -> float:
        """Get component uptime in seconds"""
        return time.time() - self.start_time
    
    @property
    def time_since_heartbeat(self) -> float:
        """Get time since last heartbeat in seconds"""
        return time.time() - self.last_heartbeat
    
    def is_healthy(self, heartbeat_timeout: float = 60.0) -> bool:
        """Check if component is healthy"""
        return (
            self.status in [ComponentStatus.HEALTHY, ComponentStatus.WARNING] and
            self.time_since_heartbeat < heartbeat_timeout
        )


class HealthMonitor:
    """
    Comprehensive health monitoring system for trading components
    
    Features:
    - Real-time component status tracking
    - Performance metrics collection
    - Automated health checks
    - Alert generation for unhealthy components
    - Recovery mechanism suggestions
    - System resource monitoring
    """
    
    def __init__(self, check_interval: float = 30.0, heartbeat_timeout: float = 60.0):
        self.check_interval = check_interval
        self.heartbeat_timeout = heartbeat_timeout
        self.components: Dict[str, ComponentHealth] = {}
        self.event_bus = get_event_bus()
        self.logger = logging.getLogger(__name__)
        self.running = False
        
        # System metrics
        self.system_metrics = {
            "cpu_percent": 0.0,
            "memory_percent": 0.0,
            "disk_usage": 0.0,
            "network_io": 0.0
        }
        
        # Alert thresholds
        self.thresholds = {
            "cpu_usage": 80.0,  # %
            "memory_usage": 85.0,  # %
            "error_rate": 0.1,  # errors per second
            "response_time": 5.0,  # seconds
            "heartbeat_timeout": heartbeat_timeout
        }
        
        # Recovery actions
        self.recovery_actions: Dict[str, Callable] = {}
        
        # Setup event subscriptions
        self._setup_event_subscriptions()
        
    def _setup_event_subscriptions(self):
        """Setup event bus subscriptions for health monitoring"""
        try:
            # Subscribe to component events
            self.event_bus.subscribe(
                EventType.COMPONENT_READY, 
                self._handle_component_ready, 
                "health_monitor"
            )
            
            self.event_bus.subscribe(
                EventType.COMPONENT_ERROR, 
                self._handle_component_error, 
                "health_monitor"
            )
            
            self.event_bus.subscribe(
                EventType.HEARTBEAT, 
                self._handle_heartbeat, 
                "health_monitor"
            )
            
            # Subscribe to performance events
            self.event_bus.subscribe(
                EventType.PERFORMANCE_METRIC_UPDATE, 
                self._handle_performance_update, 
                "health_monitor"
            )
            
            self.logger.info("Health monitor event subscriptions setup completed")
            
        except Exception as e:
            self.logger.error(f"Error setting up health monitor subscriptions: {str(e)}")
    
    async def start(self):
        """Start the health monitoring system"""
        self.running = True
        self.logger.info("Health monitor started")
        
        # Start monitoring tasks
        asyncio.create_task(self._health_check_loop())
        asyncio.create_task(self._system_metrics_loop())
        
        # Publish component ready event
        await self.event_bus.publish_simple(
            EventType.COMPONENT_READY,
            "health_monitor",
            {"component_name": "health_monitor", "status": "ready"}
        )
        
    async def stop(self):
        """Stop the health monitoring system"""
        self.running = False
        self.logger.info("Health monitor stopped")
        
    def register_component(self, name: str, recovery_action: Optional[Callable] = None):
        """
        Register a component for health monitoring
        
        Args:
            name: Component name
            recovery_action: Optional recovery function to call if component fails
        """
        if name not in self.components:
            self.components[name] = ComponentHealth(name=name)
            if recovery_action:
                self.recovery_actions[name] = recovery_action
            self.logger.info(f"Registered component: {name}")
        
    async def update_component_status(self, name: str, status: ComponentStatus, 
                              error_message: Optional[str] = None):
        """Update component status"""
        if name not in self.components:
            self.register_component(name)
            
        component = self.components[name]
        old_status = component.status
        component.status = status
        component.last_heartbeat = time.time()
        
        if status in [ComponentStatus.ERROR, ComponentStatus.CRITICAL]:
            component.error_count += 1
            if error_message:
                component.last_error = error_message
                component.last_error_time = time.time()
        elif status == ComponentStatus.WARNING:
            component.warning_count += 1
            
        # Log status changes
        if old_status != status:
            self.logger.info(f"Component {name} status changed: {old_status.value} -> {status.value}")
            
            # Trigger recovery if needed
            if status in [ComponentStatus.ERROR, ComponentStatus.CRITICAL]:
                await self._trigger_recovery(name)
                
    def update_component_metric(self, name: str, metric: str, value: float):
        """Update component performance metric"""
        if name not in self.components:
            self.register_component(name)
            
        component = self.components[name]
        component.metrics[metric] = value
        component.performance_history[metric].append((time.time(), value))
        
        # Check thresholds
        self._check_metric_thresholds(name, metric, value)
        
    async def _health_check_loop(self):
        """Main health check loop"""
        while self.running:
            try:
                await self._perform_health_checks()
                await asyncio.sleep(self.check_interval)
            except Exception as e:
                self.logger.error(f"Error in health check loop: {str(e)}")
                await asyncio.sleep(self.check_interval)
                
    async def _system_metrics_loop(self):
        """System metrics collection loop"""
        while self.running:
            try:
                await self._collect_system_metrics()
                await asyncio.sleep(10.0)  # Collect system metrics every 10 seconds
            except Exception as e:
                self.logger.error(f"Error collecting system metrics: {str(e)}")
                await asyncio.sleep(10.0)
                
    async def _perform_health_checks(self):
        """Perform health checks on all components"""
        current_time = time.time()
        
        for name, component in self.components.items():
            try:
                # Check heartbeat timeout
                if component.time_since_heartbeat > self.heartbeat_timeout:
                    if component.status != ComponentStatus.ERROR:
                        await self.update_component_status(
                            name, 
                            ComponentStatus.ERROR, 
                            f"Heartbeat timeout ({component.time_since_heartbeat:.1f}s)"
                        )
                        
                # Check error rates
                error_rate = self._calculate_error_rate(component)
                if error_rate > self.thresholds["error_rate"]:
                    if component.status not in [ComponentStatus.ERROR, ComponentStatus.CRITICAL]:
                        await self.update_component_status(
                            name, 
                            ComponentStatus.WARNING, 
                            f"High error rate: {error_rate:.3f}/s"
                        )
                        
                # Check performance metrics
                for metric_name, threshold in self.thresholds.items():
                    if metric_name in component.metrics:
                        value = component.metrics[metric_name]
                        if value > threshold and component.status == ComponentStatus.HEALTHY:
                            await self.update_component_status(
                                name, 
                                ComponentStatus.WARNING, 
                                f"High {metric_name}: {value:.2f}"
                            )
                            
            except Exception as e:
                self.logger.error(f"Error checking health of {name}: {str(e)}")
                
    async def _collect_system_metrics(self):
        """Collect system-wide metrics"""
        try:
            # CPU usage
            self.system_metrics["cpu_percent"] = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            self.system_metrics["memory_percent"] = memory.percent
            
            # Disk usage
            disk = psutil.disk_usage('/')
            self.system_metrics["disk_usage"] = (disk.used / disk.total) * 100
            
            # Network I/O (simplified)
            net_io = psutil.net_io_counters()
            self.system_metrics["network_io"] = net_io.bytes_sent + net_io.bytes_recv
            
            # Publish system metrics
            await self.event_bus.publish_simple(
                EventType.PERFORMANCE_METRIC_UPDATE,
                "health_monitor",
                {
                    "component": "system",
                    "metrics": self.system_metrics.copy()
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error collecting system metrics: {str(e)}")
            
    def _calculate_error_rate(self, component: ComponentHealth) -> float:
        """Calculate error rate for a component"""
        try:
            # Simple error rate calculation based on recent errors
            recent_window = 60.0  # 1 minute window
            current_time = time.time()
            
            if component.last_error_time and (current_time - component.last_error_time) < recent_window:
                return component.error_count / component.uptime if component.uptime > 0 else 0.0
            return 0.0
            
        except Exception:
            return 0.0
            
    def _check_metric_thresholds(self, component_name: str, metric: str, value: float):
        """Check if metric exceeds thresholds"""
        if metric in self.thresholds:
            threshold = self.thresholds[metric]
            if value > threshold:
                self.logger.warning(f"Component {component_name} metric {metric} ({value:.2f}) exceeds threshold ({threshold:.2f})")
                
    async def _trigger_recovery(self, component_name: str):
        """Trigger recovery action for a component"""
        try:
            if component_name in self.recovery_actions:
                self.logger.info(f"Triggering recovery for component: {component_name}")
                recovery_action = self.recovery_actions[component_name]
                
                if asyncio.iscoroutinefunction(recovery_action):
                    await recovery_action()
                else:
                    recovery_action()
                    
        except Exception as e:
            self.logger.error(f"Error triggering recovery for {component_name}: {str(e)}")
            
    async def _handle_component_ready(self, event: Event):
        """Handle component ready events"""
        try:
            component_name = event.data.get("component_name")
            if component_name:
                await self.update_component_status(component_name, ComponentStatus.HEALTHY)
                
        except Exception as e:
            self.logger.error(f"Error handling component ready event: {str(e)}")
            
    async def _handle_component_error(self, event: Event):
        """Handle component error events"""
        try:
            component_name = event.source
            error_message = event.data.get("error", "Unknown error")
            await self.update_component_status(component_name, ComponentStatus.ERROR, error_message)
            
        except Exception as e:
            self.logger.error(f"Error handling component error event: {str(e)}")
            
    async def _handle_heartbeat(self, event: Event):
        """Handle heartbeat events"""
        try:
            component_name = event.source
            if component_name in self.components:
                self.components[component_name].last_heartbeat = time.time()
                
                # Update status to healthy if it was in error due to heartbeat timeout
                component = self.components[component_name]
                if (component.status == ComponentStatus.ERROR and 
                    component.last_error and "timeout" in component.last_error.lower()):
                    await self.update_component_status(component_name, ComponentStatus.HEALTHY)
                    
        except Exception as e:
            self.logger.error(f"Error handling heartbeat event: {str(e)}")
            
    async def _handle_performance_update(self, event: Event):
        """Handle performance metric updates"""
        try:
            component_name = event.data.get("component", event.source)
            metrics = event.data.get("metrics", {})
            
            for metric_name, value in metrics.items():
                self.update_component_metric(component_name, metric_name, value)
                
        except Exception as e:
            self.logger.error(f"Error handling performance update: {str(e)}")
            
    def get_health_summary(self) -> Dict[str, Any]:
        """Get overall health summary"""
        healthy_count = sum(1 for c in self.components.values() if c.is_healthy(self.heartbeat_timeout))
        total_count = len(self.components)
        
        return {
            "timestamp": time.time(),
            "total_components": total_count,
            "healthy_components": healthy_count,
            "unhealthy_components": total_count - healthy_count,
            "system_metrics": self.system_metrics.copy(),
            "overall_status": "healthy" if healthy_count == total_count else "degraded"
        }
        
    def get_component_details(self, name: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific component"""
        if name not in self.components:
            return None
            
        component = self.components[name]
        return {
            "name": name,
            "status": component.status.value,
            "uptime": component.uptime,
            "last_heartbeat": component.last_heartbeat,
            "time_since_heartbeat": component.time_since_heartbeat,
            "error_count": component.error_count,
            "warning_count": component.warning_count,
            "last_error": component.last_error,
            "last_error_time": component.last_error_time,
            "metrics": component.metrics.copy(),
            "is_healthy": component.is_healthy(self.heartbeat_timeout)
        }
        
    def get_all_components(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all components"""
        return {
            name: self.get_component_details(name) 
            for name in self.components.keys()
        }


# Global health monitor instance
_global_health_monitor: Optional[HealthMonitor] = None


def get_health_monitor() -> HealthMonitor:
    """Get the global health monitor instance"""
    global _global_health_monitor
    if _global_health_monitor is None:
        _global_health_monitor = HealthMonitor()
    return _global_health_monitor


async def initialize_health_monitor():
    """Initialize and start the global health monitor"""
    monitor = get_health_monitor()
    await monitor.start()
    return monitor


# Convenience functions for component health updates
async def report_component_healthy(component_name: str):
    """Report that a component is healthy"""
    monitor = get_health_monitor()
    await monitor.update_component_status(component_name, ComponentStatus.HEALTHY)


async def report_component_error(component_name: str, error_message: str):
    """Report that a component has an error"""
    monitor = get_health_monitor()
    await monitor.update_component_status(component_name, ComponentStatus.ERROR, error_message)


async def send_heartbeat(component_name: str):
    """Send a heartbeat for a component"""
    event_bus = get_event_bus()
    await event_bus.publish_simple(
        EventType.HEARTBEAT,
        component_name,
        {"timestamp": time.time()}
    ) 