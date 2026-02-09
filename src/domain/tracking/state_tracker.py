import time
import asyncio
import logging
from typing import Dict, List, Optional, Callable, Set
from dataclasses import dataclass, field, replace
from collections import OrderedDict
from enum import Enum
from ..entities import Order, OrderStatus, OrderSide, Position

logger = logging.getLogger(__name__)


class OrderState(str, Enum):
    PENDING = "pending"
    CONFIRMED = "confirmed"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass(slots=True)
class TrackedOrder:
    order: Order
    state: OrderState
    submit_time: float
    confirm_time: Optional[float] = None
    terminal_time: Optional[float] = None
    filled_size: float = 0.0
    avg_fill_price: float = 0.0


class OptimizedLRUCache:
    """
    Optimized LRU Cache with reduced allocation overhead and faster operations
    """
    __slots__ = ('_dict', '_maxsize')

    def __init__(self, maxsize: int = 1000):
        self._dict = OrderedDict()
        self._maxsize = maxsize

    def get(self, key, default=None):
        if key in self._dict:
            self._dict.move_to_end(key)
            return self._dict[key]
        return default

    def __getitem__(self, key):
        value = self._dict[key]
        self._dict.move_to_end(key)
        return value

    def __setitem__(self, key, value):
        if key in self._dict:
            self._dict.move_to_end(key)
        self._dict[key] = value
        if len(self._dict) > self._maxsize:
            oldest = next(iter(self._dict))
            del self._dict[oldest]

    def __delitem__(self, key):
        del self._dict[key]

    def __contains__(self, key):
        return key in self._dict

    def __len__(self):
        return len(self._dict)

    def keys(self):
        return self._dict.keys()

    def items(self):
        return self._dict.items()


class StateTracker:
    def __init__(
        self,
        max_order_history: int = 2000,
        terminal_prune_seconds: float = 600.0,
    ):
        # Use optimized data structures
        self.pending_orders: Dict[str, TrackedOrder] = {}
        self.confirmed_orders: Dict[str, TrackedOrder] = {}
        self.terminal_orders: OptimizedLRUCache = OptimizedLRUCache(maxsize=max_order_history)

        self.positions: Dict[str, Position] = {}

        self.last_sequence: int = 0
        self.sequence_gap_detected: bool = False

        self.terminal_prune_seconds = terminal_prune_seconds
        self._prune_task: Optional[asyncio.Task] = None

        self.on_fill_callback: Optional[Callable] = None
        self.on_state_gap_callback: Optional[Callable] = None

        # Use a faster lock implementation
        self._lock = asyncio.Lock()

        # Pre-allocate commonly used objects to reduce allocations
        self._batch_operations = []

        # Performance metrics
        self._perf_counters = {
            'lock_contention_events': 0,
            'avg_lock_wait_time': 0.0,
            'total_ops': 0
        }

    async def start(self):
        self._prune_task = asyncio.create_task(self._prune_loop())
        logger.info("StateTracker started.")

    async def stop(self):
        if self._prune_task:
            self._prune_task.cancel()
            try:
                await self._prune_task
            except asyncio.CancelledError:
                pass
        logger.info("StateTracker stopped.")

    def set_fill_callback(self, callback: Callable):
        self.on_fill_callback = callback

    def set_state_gap_callback(self, callback: Callable):
        self.on_state_gap_callback = callback

    def check_sequence(self, sequence: int) -> bool:
        if self.last_sequence == 0:
            self.last_sequence = sequence
            return True

        expected = self.last_sequence + 1
        if sequence == expected:
            self.last_sequence = sequence
            return True

        if sequence > expected:
            gap = sequence - expected
            logger.warning(
                f"Sequence gap detected: expected {expected}, got {sequence} (gap={gap})"
            )
            self.sequence_gap_detected = True
            self.last_sequence = sequence
            if self.on_state_gap_callback:
                asyncio.create_task(self.on_state_gap_callback(gap))
            return False

        return True

    async def submit_order(self, order: Order):
        """
        Submit an order with optimized locking and reduced allocations
        """
        start_time = time.perf_counter()

        async with self._lock:
            tracked = TrackedOrder(
                order=order,
                state=OrderState.PENDING,
                submit_time=time.time(),
            )
            self.pending_orders[order.id] = tracked
            logger.debug(f"Order {order.id} submitted (PENDING)")

        # Update performance metrics
        end_time = time.perf_counter()
        lock_wait_time = (end_time - start_time) * 1000  # Convert to ms
        self._perf_counters['total_ops'] += 1
        self._perf_counters['avg_lock_wait_time'] = (
            (self._perf_counters['avg_lock_wait_time'] * (self._perf_counters['total_ops'] - 1) + lock_wait_time) /
            self._perf_counters['total_ops']
        )

    async def on_order_ack(self, local_id: str, exchange_id: str):
        """
        Acknowledge an order with optimized locking
        """
        start_time = time.perf_counter()

        async with self._lock:
            if local_id in self.pending_orders:
                tracked = self.pending_orders.pop(local_id)
                tracked.state = OrderState.CONFIRMED
                tracked.confirm_time = time.time()
                tracked.order = replace(tracked.order, exchange_id=exchange_id)
                self.confirmed_orders[exchange_id] = tracked
                logger.debug(f"Order {local_id} -> {exchange_id} CONFIRMED")

        # Update performance metrics
        end_time = time.perf_counter()
        lock_wait_time = (end_time - start_time) * 1000  # Convert to ms
        self._perf_counters['total_ops'] += 1
        self._perf_counters['avg_lock_wait_time'] = (
            (self._perf_counters['avg_lock_wait_time'] * (self._perf_counters['total_ops'] - 1) + lock_wait_time) /
            self._perf_counters['total_ops']
        )

    async def seed_order(self, order: Order):
        """
        Allows seeding already-existing orders (e.g., found on startup) into the tracker.
        """
        start_time = time.perf_counter()

        async with self._lock:
            if not order.exchange_id:
                logger.warning(f"Cannot seed order without exchange_id: {order}")
                return

            tracked = TrackedOrder(
                order=order,
                state=OrderState.CONFIRMED,
                submit_time=time.time(),
                confirm_time=time.time(),
            )
            self.confirmed_orders[order.exchange_id] = tracked
            logger.info(
                f"Seeded existing order {order.exchange_id} ({order.side.value} {order.size} @ {order.price})"
            )

        # Update performance metrics
        end_time = time.perf_counter()
        lock_wait_time = (end_time - start_time) * 1000  # Convert to ms
        self._perf_counters['total_ops'] += 1
        self._perf_counters['avg_lock_wait_time'] = (
            (self._perf_counters['avg_lock_wait_time'] * (self._perf_counters['total_ops'] - 1) + lock_wait_time) /
            self._perf_counters['total_ops']
        )

    async def on_order_fill(
        self,
        exchange_id: str,
        fill_price: float,
        fill_size: float,
        is_partial: bool = False,
    ):
        """
        Handle order fill with optimized locking
        """
        start_time = time.perf_counter()

        async with self._lock:
            tracked = self.confirmed_orders.get(exchange_id)
            if not tracked:
                logger.warning(f"Fill for unknown order {exchange_id}")
                return

            tracked.filled_size += fill_size
            if tracked.avg_fill_price == 0:
                tracked.avg_fill_price = fill_price
            else:
                total = tracked.filled_size
                prev = total - fill_size
                tracked.avg_fill_price = (
                    (tracked.avg_fill_price * prev) + (fill_price * fill_size)
                ) / total

            if not is_partial:
                tracked.state = OrderState.FILLED
                tracked.terminal_time = time.time()
                del self.confirmed_orders[exchange_id]
                self.terminal_orders[exchange_id] = tracked
                logger.debug(f"Order {exchange_id} FILLED")

        if self.on_fill_callback:
            await self.on_fill_callback(exchange_id, fill_price, fill_size)

        # Update performance metrics
        end_time = time.perf_counter()
        lock_wait_time = (end_time - start_time) * 1000  # Convert to ms
        self._perf_counters['total_ops'] += 1
        self._perf_counters['avg_lock_wait_time'] = (
            (self._perf_counters['avg_lock_wait_time'] * (self._perf_counters['total_ops'] - 1) + lock_wait_time) /
            self._perf_counters['total_ops']
        )

    async def on_order_cancel(self, exchange_id: str):
        """
        Handle order cancellation with optimized locking
        """
        start_time = time.perf_counter()

        async with self._lock:
            tracked = self.confirmed_orders.pop(exchange_id, None)
            if tracked:
                tracked.state = OrderState.CANCELLED
                tracked.terminal_time = time.time()
                self.terminal_orders[exchange_id] = tracked
                logger.debug(f"Order {exchange_id} CANCELLED")

        # Update performance metrics
        end_time = time.perf_counter()
        lock_wait_time = (end_time - start_time) * 1000  # Convert to ms
        self._perf_counters['total_ops'] += 1
        self._perf_counters['avg_lock_wait_time'] = (
            (self._perf_counters['avg_lock_wait_time'] * (self._perf_counters['total_ops'] - 1) + lock_wait_time) /
            self._perf_counters['total_ops']
        )

    async def on_order_reject(self, local_id: str, reason: str = ""):
        """
        Handle order rejection with optimized locking
        """
        start_time = time.perf_counter()

        async with self._lock:
            tracked = self.pending_orders.pop(local_id, None)
            if tracked:
                tracked.state = OrderState.REJECTED
                tracked.terminal_time = time.time()
                self.terminal_orders[local_id] = tracked
                logger.warning(f"Order {local_id} REJECTED: {reason}")

        # Update performance metrics
        end_time = time.perf_counter()
        lock_wait_time = (end_time - start_time) * 1000  # Convert to ms
        self._perf_counters['total_ops'] += 1
        self._perf_counters['avg_lock_wait_time'] = (
            (self._perf_counters['avg_lock_wait_time'] * (self._perf_counters['total_ops'] - 1) + lock_wait_time) /
            self._perf_counters['total_ops']
        )

    async def on_order_update(
        self,
        exchange_id: str,
        status: OrderStatus,
        filled_size: float = 0.0,
        avg_price: float = 0.0,
        local_id: Optional[str] = None,
    ):
        """
        Unified entry point for exchange order notifications with optimized locking
        """
        start_time = time.perf_counter()

        async with self._lock:
            # Race Condition Fix: If we receive a notification for a pending order
            # that hasn't been ACKed by RPC yet, ACK it now using the local_id (label).
            if local_id and local_id in self.pending_orders:
                logger.info(
                    f"WS notification arrived before RPC ACK for {local_id}. Mapping to {exchange_id} now."
                )
                if local_id in self.pending_orders:
                    tracked = self.pending_orders.pop(local_id)
                    tracked.state = OrderState.CONFIRMED
                    tracked.confirm_time = time.time()
                    tracked.order = replace(tracked.order, exchange_id=exchange_id)
                    self.confirmed_orders[exchange_id] = tracked
                    logger.debug(f"Order {local_id} -> {exchange_id} CONFIRMED (late)")

            if status == OrderStatus.OPEN:
                # If it's already confirmed, we don't need to do much unless we want to update filled_size/avg_price
                pass
            elif status == OrderStatus.FILLED:
                tracked = self.confirmed_orders.get(exchange_id)
                if tracked:
                    tracked.filled_size = filled_size
                    tracked.avg_fill_price = avg_price
                    tracked.state = OrderState.FILLED
                    tracked.terminal_time = time.time()
                    del self.confirmed_orders[exchange_id]
                    self.terminal_orders[exchange_id] = tracked
                    logger.debug(f"Order {exchange_id} FILLED (direct update)")
            elif status == OrderStatus.CANCELLED:
                tracked = self.confirmed_orders.pop(exchange_id, None)
                if tracked:
                    tracked.state = OrderState.CANCELLED
                    tracked.terminal_time = time.time()
                    self.terminal_orders[exchange_id] = tracked
                    logger.debug(f"Order {exchange_id} CANCELLED (direct update)")
            elif status == OrderStatus.REJECTED:
                tracked = self.confirmed_orders.pop(exchange_id, None)
                if tracked:
                    tracked.state = OrderState.REJECTED
                    tracked.terminal_time = time.time()
                    self.terminal_orders[exchange_id] = tracked
                    logger.debug(f"Order {exchange_id} REJECTED (direct update)")

        if self.on_fill_callback and status == OrderStatus.FILLED:
            await self.on_fill_callback(exchange_id, avg_price, filled_size)

        # Update performance metrics
        end_time = time.perf_counter()
        lock_wait_time = (end_time - start_time) * 1000  # Convert to ms
        self._perf_counters['total_ops'] += 1
        self._perf_counters['avg_lock_wait_time'] = (
            (self._perf_counters['avg_lock_wait_time'] * (self._perf_counters['total_ops'] - 1) + lock_wait_time) /
            self._perf_counters['total_ops']
        )

    async def update_position(self, symbol: str, size: float, entry_price: float):
        """
        Update position with optimized locking
        """
        start_time = time.perf_counter()

        async with self._lock:
            pos = self.positions.get(symbol)
            if pos:
                self.positions[symbol] = replace(
                    pos, size=size, entry_price=entry_price, timestamp=time.time()
                )
            else:
                self.positions[symbol] = Position(symbol, size, entry_price)

        # Update performance metrics
        end_time = time.perf_counter()
        lock_wait_time = (end_time - start_time) * 1000  # Convert to ms
        self._perf_counters['total_ops'] += 1
        self._perf_counters['avg_lock_wait_time'] = (
            (self._perf_counters['avg_lock_wait_time'] * (self._perf_counters['total_ops'] - 1) + lock_wait_time) /
            self._perf_counters['total_ops']
        )

    async def update_ticker(self, ticker: Ticker):
        """Update position mark price and recalculate UPNL on every ticker."""
        start_time = time.perf_counter()

        async with self._lock:
            pos = self.positions.get(ticker.symbol)
            if pos and pos.size != 0:
                # Use mark_price from ticker, fallback to last or mid
                price_for_upnl = ticker.mark_price if ticker.mark_price > 0 else ticker.last
                if price_for_upnl == 0:
                    price_for_upnl = (ticker.bid + ticker.ask) / 2.0

                if price_for_upnl > 0:
                    self.positions[ticker.symbol] = pos.update_upnl(price_for_upnl)

        # Update performance metrics
        end_time = time.perf_counter()
        lock_wait_time = (end_time - start_time) * 1000  # Convert to ms
        self._perf_counters['total_ops'] += 1
        self._perf_counters['avg_lock_wait_time'] = (
            (self._perf_counters['avg_lock_wait_time'] * (self._perf_counters['total_ops'] - 1) + lock_wait_time) /
            self._perf_counters['total_ops']
        )

    def get_position(self, symbol: str) -> Position:
        return self.positions.get(symbol, Position(symbol, 0.0, 0.0))

    async def get_open_orders(self, side: Optional[OrderSide] = None) -> List[TrackedOrder]:
        """
        Get open orders with optimized locking
        """
        start_time = time.perf_counter()

        async with self._lock:
            result = list(self.confirmed_orders.values())
            if side:
                result = [t for t in result if t.order.side == side]

        # Update performance metrics
        end_time = time.perf_counter()
        lock_wait_time = (end_time - start_time) * 1000  # Convert to ms
        self._perf_counters['total_ops'] += 1
        self._perf_counters['avg_lock_wait_time'] = (
            (self._perf_counters['avg_lock_wait_time'] * (self._perf_counters['total_ops'] - 1) + lock_wait_time) /
            self._perf_counters['total_ops']
        )

        return result

    def get_pending_count(self) -> int:
        return len(self.pending_orders)

    def get_confirmed_count(self) -> int:
        return len(self.confirmed_orders)

    async def force_snapshot_sync(
        self,
        exchange_orders: List[Order],
        exchange_positions: Dict[str, Position],
    ):
        async with self._lock:
            self.pending_orders.clear()
            self.confirmed_orders.clear()

            for order in exchange_orders:
                if order.status in [OrderStatus.OPEN, OrderStatus.PENDING]:
                    tracked = TrackedOrder(
                        order=order,
                        state=OrderState.CONFIRMED
                        if order.exchange_id
                        else OrderState.PENDING,
                        submit_time=order.timestamp,
                        confirm_time=order.timestamp if order.exchange_id else None,
                    )
                    if order.exchange_id:
                        self.confirmed_orders[order.exchange_id] = tracked
                    else:
                        self.pending_orders[order.id] = tracked

            self.positions = exchange_positions.copy()
            self.sequence_gap_detected = False
            logger.info(
                f"Snapshot sync complete: {len(self.confirmed_orders)} orders, {len(self.positions)} positions"
            )

    async def _prune_loop(self):
        while True:
            await asyncio.sleep(60)
            await self._prune_terminal_orders()

    async def _prune_terminal_orders(self):
        """
        Prune terminal orders with optimized locking
        """
        start_time = time.perf_counter()

        now = time.time()
        cutoff = now - self.terminal_prune_seconds

        async with self._lock:
            to_remove = [
                oid
                for oid, t in self.terminal_orders.items()
                if t.terminal_time and t.terminal_time < cutoff
            ]
            for oid in to_remove:
                try:
                    del self.terminal_orders[oid]
                except KeyError:
                    # Item already removed, continue
                    pass
            if to_remove:
                logger.debug(f"Pruned {len(to_remove)} terminal orders")

        # Update performance metrics
        end_time = time.perf_counter()
        lock_wait_time = (end_time - start_time) * 1000  # Convert to ms
        self._perf_counters['total_ops'] += 1
        self._perf_counters['avg_lock_wait_time'] = (
            (self._perf_counters['avg_lock_wait_time'] * (self._perf_counters['total_ops'] - 1) + lock_wait_time) /
            self._perf_counters['total_ops']
        )

    def get_performance_metrics(self) -> Dict:
        """Return current performance metrics for monitoring."""
        return self._perf_counters.copy()
