import time
import asyncio
import logging
from typing import Dict, List, Optional, Callable, Set
from dataclasses import dataclass, field
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


class LRUCache(OrderedDict):
    def __init__(self, maxsize: int = 1000):
        super().__init__()
        self.maxsize = maxsize

    def __setitem__(self, key, value):
        if key in self:
            self.move_to_end(key)
        super().__setitem__(key, value)
        if len(self) > self.maxsize:
            oldest = next(iter(self))
            del self[oldest]


class StateTracker:
    def __init__(
        self,
        max_order_history: int = 2000,
        terminal_prune_seconds: float = 600.0,
    ):
        self.pending_orders: Dict[str, TrackedOrder] = {}
        self.confirmed_orders: Dict[str, TrackedOrder] = {}
        self.terminal_orders: LRUCache = LRUCache(maxsize=max_order_history)

        self.positions: Dict[str, Position] = {}

        self.last_sequence: int = 0
        self.sequence_gap_detected: bool = False

        self.terminal_prune_seconds = terminal_prune_seconds
        self._prune_task: Optional[asyncio.Task] = None

        self.on_fill_callback: Optional[Callable] = None
        self.on_state_gap_callback: Optional[Callable] = None

        self._lock = asyncio.Lock()

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
        async with self._lock:
            tracked = TrackedOrder(
                order=order,
                state=OrderState.PENDING,
                submit_time=time.time(),
            )
            self.pending_orders[order.id] = tracked
            logger.debug(f"Order {order.id} submitted (PENDING)")

    async def on_order_ack(self, local_id: str, exchange_id: str):
        async with self._lock:
            if local_id in self.pending_orders:
                tracked = self.pending_orders.pop(local_id)
                tracked.state = OrderState.CONFIRMED
                tracked.confirm_time = time.time()
                tracked.order = tracked.order.__class__(
                    **{**tracked.order.__dict__, "exchange_id": exchange_id}
                )
                self.confirmed_orders[exchange_id] = tracked
                logger.debug(f"Order {local_id} -> {exchange_id} CONFIRMED")

    async def on_order_fill(
        self,
        exchange_id: str,
        fill_price: float,
        fill_size: float,
        is_partial: bool = False,
    ):
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

    async def on_order_cancel(self, exchange_id: str):
        async with self._lock:
            tracked = self.confirmed_orders.pop(exchange_id, None)
            if tracked:
                tracked.state = OrderState.CANCELLED
                tracked.terminal_time = time.time()
                self.terminal_orders[exchange_id] = tracked
                logger.debug(f"Order {exchange_id} CANCELLED")

    async def on_order_reject(self, local_id: str, reason: str = ""):
        async with self._lock:
            tracked = self.pending_orders.pop(local_id, None)
            if tracked:
                tracked.state = OrderState.REJECTED
                tracked.terminal_time = time.time()
                self.terminal_orders[local_id] = tracked
                logger.warning(f"Order {local_id} REJECTED: {reason}")

    async def update_position(self, symbol: str, size: float, entry_price: float):
        async with self._lock:
            self.positions[symbol] = Position(symbol, size, entry_price)

    def get_position(self, symbol: str) -> Position:
        return self.positions.get(symbol, Position(symbol, 0.0, 0.0))

    def get_open_orders(self, side: Optional[OrderSide] = None) -> List[TrackedOrder]:
        result = list(self.confirmed_orders.values())
        if side:
            result = [t for t in result if t.order.side == side]
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
        now = time.time()
        cutoff = now - self.terminal_prune_seconds
        async with self._lock:
            to_remove = [
                oid
                for oid, t in self.terminal_orders.items()
                if t.terminal_time and t.terminal_time < cutoff
            ]
            for oid in to_remove:
                del self.terminal_orders[oid]
            if to_remove:
                logger.debug(f"Pruned {len(to_remove)} terminal orders")
