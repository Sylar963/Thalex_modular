"""
Event Bus System for Cross-Component Communication - Added 2024-12-19

This module provides a centralized event bus for communication between
different components of the trading system (PerpQuoter, RiskManager, OrderManager, etc.)
"""

import asyncio
import logging
import time
from typing import Dict, List, Callable, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque


class EventType(Enum):
    """Event types for the trading system"""
    # Market data events
    PRICE_UPDATE = "price_update"
    VOLATILITY_UPDATE = "volatility_update"
    SPREAD_UPDATE = "spread_update"
    
    # Position events
    POSITION_CHANGE = "position_change"
    POSITION_RISK_BREACH = "position_risk_breach"
    INVENTORY_IMBALANCE = "inventory_imbalance"
    
    # Order events
    ORDER_PLACED = "order_placed"
    ORDER_FILLED = "order_filled"
    ORDER_CANCELLED = "order_cancelled"
    ORDER_REJECTED = "order_rejected"
    
    # Risk events
    VAR_ALERT = "var_alert"
    RISK_LIMIT_BREACH = "risk_limit_breach"
    EMERGENCY_STOP = "emergency_stop"
    
    # Performance events
    PNL_UPDATE = "pnl_update"
    PERFORMANCE_METRIC_UPDATE = "performance_metric_update"
    
    # System events
    COMPONENT_READY = "component_ready"
    COMPONENT_ERROR = "component_error"
    HEARTBEAT = "heartbeat"


@dataclass
class Event:
    """Event data structure"""
    event_type: EventType
    source: str
    data: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    priority: int = 0  # Higher number = higher priority
    correlation_id: Optional[str] = None


class EventBus:
    """
    Centralized event bus for cross-component communication
    
    Features:
    - Asynchronous event handling
    - Priority-based event processing
    - Event filtering and routing
    - Component health monitoring
    - Event history for debugging
    """
    
    def __init__(self, max_history: int = 1000):
        self.subscribers: Dict[EventType, List[Callable]] = defaultdict(list)
        self.event_queue = asyncio.PriorityQueue()
        self.event_history = deque(maxlen=max_history)
        self.component_status: Dict[str, Dict[str, Any]] = {}
        self.running = False
        self.logger = logging.getLogger(__name__)
        self._stats = {
            "events_published": 0,
            "events_processed": 0,
            "errors": 0,
            "last_activity": time.time()
        }
        
    async def start(self):
        """Start the event bus processing loop"""
        self.running = True
        self.logger.info("Event bus started")
        asyncio.create_task(self._process_events())
        
    async def stop(self):
        """Stop the event bus"""
        self.running = False
        self.logger.info("Event bus stopped")
        
    def subscribe(self, event_type: EventType, callback: Callable, component_name: str = "unknown"):
        """
        Subscribe to events of a specific type
        
        Args:
            event_type: Type of event to subscribe to
            callback: Async function to call when event occurs
            component_name: Name of the subscribing component
        """
        self.subscribers[event_type].append(callback)
        self.logger.info(f"Component '{component_name}' subscribed to {event_type.value}")
        
        # Register component
        if component_name not in self.component_status:
            self.component_status[component_name] = {
                "status": "active",
                "last_seen": time.time(),
                "subscriptions": []
            }
        self.component_status[component_name]["subscriptions"].append(event_type.value)
        
    def unsubscribe(self, event_type: EventType, callback: Callable):
        """Unsubscribe from events"""
        if callback in self.subscribers[event_type]:
            self.subscribers[event_type].remove(callback)
            
    async def publish(self, event: Event):
        """
        Publish an event to the bus
        
        Args:
            event: Event to publish
        """
        try:
            # Add to queue with priority (negative for max-heap behavior)
            await self.event_queue.put((-event.priority, time.time(), event))
            self._stats["events_published"] += 1
            self._stats["last_activity"] = time.time()
            
            # Add to history
            self.event_history.append(event)
            
            self.logger.debug(f"Published event: {event.event_type.value} from {event.source}")
            
        except Exception as e:
            self.logger.error(f"Error publishing event: {str(e)}")
            self._stats["errors"] += 1
            
    async def publish_simple(self, event_type: EventType, source: str, data: Dict[str, Any], 
                           priority: int = 0, correlation_id: Optional[str] = None):
        """
        Convenience method to publish an event
        
        Args:
            event_type: Type of event
            source: Source component name
            data: Event data
            priority: Event priority (higher = more important)
            correlation_id: Optional correlation ID for tracking
        """
        event = Event(
            event_type=event_type,
            source=source,
            data=data,
            priority=priority,
            correlation_id=correlation_id
        )
        await self.publish(event)
        
    async def _process_events(self):
        """Process events from the queue"""
        while self.running:
            try:
                # Get event from queue (blocks if empty)
                _, _, event = await asyncio.wait_for(self.event_queue.get(), timeout=1.0)
                
                # Process event
                await self._handle_event(event)
                self._stats["events_processed"] += 1
                
            except asyncio.TimeoutError:
                # No events to process, continue
                continue
            except Exception as e:
                self.logger.error(f"Error processing event: {str(e)}")
                self._stats["errors"] += 1
                
    async def _handle_event(self, event: Event):
        """Handle a single event by notifying all subscribers"""
        subscribers = self.subscribers.get(event.event_type, [])
        
        if not subscribers:
            self.logger.debug(f"No subscribers for event type: {event.event_type.value}")
            return
            
        # Notify all subscribers concurrently
        tasks = []
        for callback in subscribers:
            try:
                if asyncio.iscoroutinefunction(callback):
                    tasks.append(asyncio.create_task(callback(event)))
                else:
                    # Handle sync callbacks
                    callback(event)
            except Exception as e:
                self.logger.error(f"Error calling subscriber: {str(e)}")
                self._stats["errors"] += 1
                
        # Wait for all async callbacks to complete
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
            
    def get_stats(self) -> Dict[str, Any]:
        """Get event bus statistics"""
        return {
            **self._stats,
            "active_subscribers": {
                event_type.value: len(callbacks) 
                for event_type, callbacks in self.subscribers.items()
            },
            "component_count": len(self.component_status),
            "queue_size": self.event_queue.qsize(),
            "history_size": len(self.event_history)
        }
        
    def get_component_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all registered components"""
        return self.component_status.copy()
        
    def update_component_heartbeat(self, component_name: str):
        """Update component heartbeat"""
        if component_name in self.component_status:
            self.component_status[component_name]["last_seen"] = time.time()
            self.component_status[component_name]["status"] = "active"
            
    def mark_component_error(self, component_name: str, error: str):
        """Mark component as having an error"""
        if component_name in self.component_status:
            self.component_status[component_name]["status"] = "error"
            self.component_status[component_name]["last_error"] = error
            self.component_status[component_name]["error_time"] = time.time()
            
    async def wait_for_event(self, event_type: EventType, timeout: float = 10.0) -> Optional[Event]:
        """
        Wait for a specific event type (useful for testing)
        
        Args:
            event_type: Event type to wait for
            timeout: Maximum time to wait
            
        Returns:
            Event if received within timeout, None otherwise
        """
        event_received = asyncio.Event()
        received_event = None
        
        async def event_waiter(event: Event):
            nonlocal received_event
            received_event = event
            event_received.set()
            
        # Subscribe temporarily
        self.subscribe(event_type, event_waiter, "event_waiter")
        
        try:
            await asyncio.wait_for(event_received.wait(), timeout=timeout)
            return received_event
        except asyncio.TimeoutError:
            return None
        finally:
            # Cleanup subscription
            self.unsubscribe(event_type, event_waiter)


# Global event bus instance
_global_event_bus: Optional[EventBus] = None


def get_event_bus() -> EventBus:
    """Get the global event bus instance"""
    global _global_event_bus
    if _global_event_bus is None:
        _global_event_bus = EventBus()
    return _global_event_bus


async def initialize_event_bus():
    """Initialize and start the global event bus"""
    bus = get_event_bus()
    await bus.start()
    return bus


# Convenience functions for common event types
async def publish_price_update(source: str, instrument: str, price: float, 
                             bid: Optional[float] = None, ask: Optional[float] = None):
    """Publish a price update event"""
    bus = get_event_bus()
    await bus.publish_simple(
        EventType.PRICE_UPDATE,
        source,
        {
            "instrument": instrument,
            "price": price,
            "bid": bid,
            "ask": ask
        },
        priority=5  # High priority for price updates
    )


async def publish_position_change(source: str, instrument: str, old_position: float, 
                                new_position: float, entry_price: Optional[float] = None):
    """Publish a position change event"""
    bus = get_event_bus()
    await bus.publish_simple(
        EventType.POSITION_CHANGE,
        source,
        {
            "instrument": instrument,
            "old_position": old_position,
            "new_position": new_position,
            "entry_price": entry_price,
            "change": new_position - old_position
        },
        priority=8  # Very high priority for position changes
    )


async def publish_var_alert(source: str, var_value: float, confidence_level: float, 
                          threshold: float, position_size: float):
    """Publish a VaR alert event"""
    bus = get_event_bus()
    await bus.publish_simple(
        EventType.VAR_ALERT,
        source,
        {
            "var_value": var_value,
            "confidence_level": confidence_level,
            "threshold": threshold,
            "position_size": position_size,
            "severity": "high" if var_value > threshold * 1.5 else "medium"
        },
        priority=9  # Highest priority for risk alerts
    )


async def publish_pnl_update(source: str, realized_pnl: float, unrealized_pnl: float, 
                           total_pnl: float, instrument: str):
    """Publish a PnL update event"""
    bus = get_event_bus()
    await bus.publish_simple(
        EventType.PNL_UPDATE,
        source,
        {
            "realized_pnl": realized_pnl,
            "unrealized_pnl": unrealized_pnl,
            "total_pnl": total_pnl,
            "instrument": instrument
        },
        priority=3  # Medium priority for PnL updates
    ) 