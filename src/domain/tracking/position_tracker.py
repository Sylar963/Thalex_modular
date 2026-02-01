import time
import threading
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class Fill:
    order_id: str
    fill_price: float
    fill_size: float
    fill_time: float
    side: str
    is_maker: bool = True


class PositionTracker:
    """
    Robust Position Tracker with FIFO accounting and PnL calculation.
    Ported from legacy Thalex_modular/models/position_tracker.py
    """

    def __init__(self):
        self.current_position = 0.0
        self.average_entry_price = None
        self.weighted_entries = {}  # price -> size
        self.realized_pnl = 0.0
        self.unrealized_pnl = 0.0
        self.fills_history = []

        self.lock = threading.Lock()
        self.ZERO_THRESHOLD = 1e-6

    def update_on_fill(self, fill: Fill):
        with self.lock:
            self.fills_history.append(fill)

            direction = 1.0 if fill.side.lower() == "buy" else -1.0
            size_change = direction * fill.fill_size

            # Check if increasing or reducing/flipping
            is_increasing = (
                (abs(self.current_position) < self.ZERO_THRESHOLD)
                or (self.current_position > 0 and size_change > 0)
                or (self.current_position < 0 and size_change < 0)
            )

            if is_increasing:
                self._add_to_position(fill.fill_price, size_change)
            else:
                self._reduce_position(fill.fill_price, size_change)

    def _add_to_position(self, price: float, size: float):
        abs_size = abs(size)
        self.weighted_entries[price] = self.weighted_entries.get(price, 0.0) + abs_size
        self.current_position += size
        self._recalculate_average_entry()

    def _reduce_position(self, price: float, size: float):
        abs_size = abs(size)
        remaining_size = abs_size

        # Check for flip
        if (
            self.current_position > 0 and size < 0 and abs_size > self.current_position
        ) or (
            self.current_position < 0
            and size > 0
            and abs_size > abs(self.current_position)
        ):
            # 1. Close current position
            flip_size = abs(self.current_position)
            flip_dir = -1 if self.current_position > 0 else 1
            self.realized_pnl += self._process_fifo_exit(price, flip_size * flip_dir)

            # 2. Open new position
            rem_new_size = abs_size - flip_size
            self.current_position = 0
            self.weighted_entries = {}
            if rem_new_size > self.ZERO_THRESHOLD:
                # Add new position with correct sign
                new_sign = 1 if size > 0 else -1
                self._add_to_position(price, rem_new_size * new_sign)

        else:
            # Standard reduction
            self.realized_pnl += self._process_fifo_exit(price, size)
            self.current_position += size
            if abs(self.current_position) < self.ZERO_THRESHOLD:
                self.reset()
            else:
                self._recalculate_average_entry()

    def _process_fifo_exit(self, exit_price: float, exit_size: float) -> float:
        # FIFO Logic
        is_buy_exit = exit_size > 0
        rem_exit = abs(exit_size)
        total_pnl = 0.0

        sorted_entries = sorted(self.weighted_entries.items())
        entries_to_remove = []
        reduced_entries = {}

        for entry_price, entry_size in sorted_entries:
            if rem_exit <= self.ZERO_THRESHOLD:
                break

            used_size = min(entry_size, rem_exit)
            rem_exit -= used_size

            # Pnl Calc
            if self.current_position > 0:  # Long closing
                pnl = (exit_price - entry_price) * used_size
            else:  # Short closing
                pnl = (entry_price - exit_price) * used_size

            total_pnl += pnl

            if used_size < entry_size - self.ZERO_THRESHOLD:
                reduced_entries[entry_price] = entry_size - used_size
            else:
                entries_to_remove.append(entry_price)

        for p in entries_to_remove:
            if p in self.weighted_entries:
                del self.weighted_entries[p]

        for p, s in reduced_entries.items():
            self.weighted_entries[p] = s

        return total_pnl

    def _recalculate_average_entry(self):
        total_size = sum(self.weighted_entries.values())
        if total_size <= self.ZERO_THRESHOLD:
            self.average_entry_price = None
            return
        total_val = sum(p * s for p, s in self.weighted_entries.items())
        self.average_entry_price = total_val / total_size

    def reset(self):
        self.current_position = 0.0
        self.average_entry_price = None
        self.weighted_entries = {}


class PortfolioTracker:
    def __init__(self):
        self.trackers = {}

    def get_tracker(self, symbol: str) -> PositionTracker:
        if symbol not in self.trackers:
            self.trackers[symbol] = PositionTracker()
        return self.trackers[symbol]
