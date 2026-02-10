import logging
import time
from typing import Optional, List, Dict
from dataclasses import dataclass, field
from enum import Enum

from .orb_breakout import TradeDirection, TradeSignal

logger = logging.getLogger(__name__)


class TradeState(str, Enum):
    IDLE = "idle"
    ENTRY_PENDING = "entry_pending"
    MOMENTUM_CHECK = "momentum_check"
    ACTIVE = "active"
    CLOSED = "closed"


class ActionType(str, Enum):
    ENTER = "enter"
    SHAVE = "shave"
    ADD = "add"
    EXIT = "exit"


@dataclass(slots=True)
class TradeAction:
    action: ActionType
    symbol: str
    direction: TradeDirection
    size: float
    price: float = 0.0
    reason: str = ""


@dataclass(slots=True)
class PyramidLevel:
    level: int
    entry_price: float
    size_added: float
    target_price: float
    shaved: bool = False
    shave_profit: float = 0.0


class ORBTradeManager:
    __slots__ = [
        "symbol",
        "state",
        "direction",
        "entry_price",
        "entry_time",
        "base_unit_size",
        "current_size",
        "avg_entry",
        "break_even",
        "pyramid_levels",
        "pyramid_count",
        "retry_count",
        "max_retries",
        "momentum_check_seconds",
        "shave_pct",
        "total_shaved_profit",
        "orh",
        "orl",
        "orm",
        "orw",
        "first_up_target",
        "first_down_target",
        "subsequent_target_pct",
        "_pending_actions",
        "_entry_filled",
        "_current_target_level",
    ]

    def __init__(
        self,
        symbol: str,
        momentum_check_seconds: float = 30.0,
        shave_pct: float = 0.90,
        max_retries: int = 2,
        subsequent_target_pct: float = 0.50,
    ):
        self.symbol = symbol
        self.momentum_check_seconds = momentum_check_seconds
        self.shave_pct = shave_pct
        self.max_retries = max_retries
        self.subsequent_target_pct = subsequent_target_pct

        self.state = TradeState.IDLE
        self.direction: Optional[TradeDirection] = None
        self.entry_price = 0.0
        self.entry_time = 0.0
        self.base_unit_size = 0.0
        self.current_size = 0.0
        self.avg_entry = 0.0
        self.break_even = 0.0
        self.pyramid_levels: List[PyramidLevel] = []
        self.pyramid_count = 0
        self.retry_count = 0
        self.total_shaved_profit = 0.0

        self.orh = 0.0
        self.orl = 0.0
        self.orm = 0.0
        self.orw = 0.0
        self.first_up_target = 0.0
        self.first_down_target = 0.0

        self._pending_actions: List[TradeAction] = []
        self._entry_filled = False
        self._current_target_level = 0

    def on_signal(self, signal: TradeSignal):
        if self.state not in (TradeState.IDLE, TradeState.CLOSED):
            return

        if self.state == TradeState.CLOSED and self.retry_count >= self.max_retries:
            logger.info(
                f"[{self.symbol}] Max retries ({self.max_retries}) reached, "
                f"ignoring signal"
            )
            return

        self.direction = signal.direction
        self.entry_price = signal.entry_price
        self.base_unit_size = signal.base_unit_size
        self.orh = signal.orh
        self.orl = signal.orl
        self.orm = signal.orm
        self.orw = signal.orw
        self.first_up_target = signal.first_up_target
        self.first_down_target = signal.first_down_target

        self.state = TradeState.ENTRY_PENDING
        self.pyramid_levels.clear()
        self.pyramid_count = 0
        self._current_target_level = 0
        self.current_size = 0.0
        self.total_shaved_profit = 0.0

        self._pending_actions.append(
            TradeAction(
                action=ActionType.ENTER,
                symbol=self.symbol,
                direction=self.direction,
                size=self.base_unit_size,
                reason="OR Breakout entry",
            )
        )

        logger.info(
            f"[{self.symbol}] ORB Entry signal: {self.direction.value} "
            f"size={self.base_unit_size:.4f} price={self.entry_price:.2f}"
        )

    def on_entry_fill(self, fill_price: float, fill_size: float, timestamp: float):
        self.entry_price = fill_price
        self.entry_time = timestamp
        self.current_size = fill_size
        self.avg_entry = fill_price
        self.break_even = fill_price
        self._entry_filled = True
        self.state = TradeState.MOMENTUM_CHECK

        self.pyramid_levels.append(
            PyramidLevel(
                level=0,
                entry_price=fill_price,
                size_added=fill_size,
                target_price=self._get_next_target_price(),
            )
        )

        logger.info(
            f"[{self.symbol}] Entry filled: {fill_price:.2f} x {fill_size:.4f} "
            f"Target1={self.pyramid_levels[0].target_price:.2f}"
        )

    def _get_next_target_price(self) -> float:
        if self._current_target_level == 0:
            if self.direction == TradeDirection.LONG:
                return (
                    self.first_up_target
                    if self.first_up_target > 0
                    else (self.orm + self.orm * 0.0189)
                )
            else:
                return (
                    self.first_down_target
                    if self.first_down_target > 0
                    else (self.orm - self.orm * 0.0189)
                )

        base_target = (
            self.first_up_target
            if self.direction == TradeDirection.LONG
            else self.first_down_target
        )
        step = self.orw * self.subsequent_target_pct
        if self.direction == TradeDirection.LONG:
            return base_target + step * self._current_target_level
        else:
            return base_target - step * self._current_target_level

    def on_tick(self, price: float, timestamp: float):
        if self.state == TradeState.IDLE or self.state == TradeState.CLOSED:
            self._check_reentry(price, timestamp)
            return

        if self.state == TradeState.ENTRY_PENDING:
            return

        if self.state == TradeState.MOMENTUM_CHECK:
            self._handle_momentum_check(price, timestamp)
            return

        if self.state == TradeState.ACTIVE:
            self._handle_active(price, timestamp)
            return

    def _handle_momentum_check(self, price: float, timestamp: float):
        elapsed = timestamp - self.entry_time
        if elapsed < self.momentum_check_seconds:
            return

        in_profit = (
            (price > self.entry_price)
            if self.direction == TradeDirection.LONG
            else (price < self.entry_price)
        )

        if in_profit:
            self.state = TradeState.ACTIVE
            self.break_even = self.entry_price
            logger.info(
                f"[{self.symbol}] Momentum confirmed at {price:.2f} "
                f"(entry={self.entry_price:.2f}). ACTIVE with BE={self.break_even:.2f}"
            )
        else:
            logger.info(
                f"[{self.symbol}] Momentum FAILED at {price:.2f} "
                f"(entry={self.entry_price:.2f}). Exiting."
            )
            self._exit_position(price, "Momentum check failed")

    def _handle_active(self, price: float, timestamp: float):
        hit_be = (
            (price <= self.break_even)
            if self.direction == TradeDirection.LONG
            else (price >= self.break_even)
        )

        if hit_be and self.pyramid_count > 0:
            logger.info(
                f"[{self.symbol}] Hit break-even {self.break_even:.2f} at {price:.2f}. "
                f"Shaved profit banked: ${self.total_shaved_profit:.2f}"
            )
            self._exit_position(price, "Break-even stop hit")
            return

        if hit_be and self.pyramid_count == 0:
            logger.info(
                f"[{self.symbol}] Hit break-even {self.break_even:.2f} "
                f"(no pyramids yet). Exiting."
            )
            self._exit_position(price, "Break-even before first target")
            return

        next_target = self._get_next_target_price()
        hit_target = (
            (price >= next_target)
            if self.direction == TradeDirection.LONG
            else (price <= next_target)
        )

        if hit_target:
            self._execute_pyramid(price, next_target, timestamp)

    def _execute_pyramid(self, price: float, target_price: float, timestamp: float):
        shave_size = self.current_size * self.shave_pct
        runner_size = self.current_size - shave_size
        add_size = self.base_unit_size

        profit_per_unit = abs(price - self.avg_entry)
        shave_profit = shave_size * profit_per_unit
        self.total_shaved_profit += shave_profit

        reverse_dir = (
            TradeDirection.SHORT
            if self.direction == TradeDirection.LONG
            else TradeDirection.LONG
        )
        self._pending_actions.append(
            TradeAction(
                action=ActionType.SHAVE,
                symbol=self.symbol,
                direction=reverse_dir,
                size=shave_size,
                price=price,
                reason=f"Shave {self.shave_pct:.0%} at T{self._current_target_level + 1} "
                f"(+${shave_profit:.2f})",
            )
        )

        self._pending_actions.append(
            TradeAction(
                action=ActionType.ADD,
                symbol=self.symbol,
                direction=self.direction,
                size=add_size,
                price=price,
                reason=f"Re-add at T{self._current_target_level + 1}",
            )
        )

        new_size = runner_size + add_size
        new_avg_entry = (
            (runner_size * self.avg_entry + add_size * price) / new_size
            if new_size > 0
            else price
        )

        self.current_size = new_size
        self.avg_entry = new_avg_entry
        self.break_even = new_avg_entry

        self._current_target_level += 1
        self.pyramid_count += 1

        self.pyramid_levels.append(
            PyramidLevel(
                level=self.pyramid_count,
                entry_price=price,
                size_added=add_size,
                target_price=self._get_next_target_price(),
                shaved=True,
                shave_profit=shave_profit,
            )
        )

        logger.info(
            f"[{self.symbol}] PYRAMID L{self.pyramid_count}: "
            f"shaved {shave_size:.4f} (+${shave_profit:.2f}), "
            f"re-added {add_size:.4f}, new_size={self.current_size:.4f}, "
            f"new_BE={self.break_even:.2f}, "
            f"next_target={self._get_next_target_price():.2f}"
        )

    def _exit_position(self, price: float, reason: str):
        if self.current_size <= 0:
            self.state = TradeState.CLOSED
            return

        reverse_dir = (
            TradeDirection.SHORT
            if self.direction == TradeDirection.LONG
            else TradeDirection.LONG
        )
        self._pending_actions.append(
            TradeAction(
                action=ActionType.EXIT,
                symbol=self.symbol,
                direction=reverse_dir,
                size=self.current_size,
                price=price,
                reason=reason,
            )
        )

        self.current_size = 0.0
        self.state = TradeState.CLOSED

        logger.info(
            f"[{self.symbol}] CLOSED: {reason}. "
            f"Total shaved profit: ${self.total_shaved_profit:.2f}, "
            f"Pyramids: {self.pyramid_count}, Retry: {self.retry_count}"
        )

    def _check_reentry(self, price: float, timestamp: float):
        if self.state != TradeState.CLOSED:
            return

        if self.retry_count >= self.max_retries:
            return

        if self.direction is None:
            return

        re_broke = False
        if self.direction == TradeDirection.LONG and price > self.orh:
            re_broke = True
        elif self.direction == TradeDirection.SHORT and price < self.orl:
            re_broke = True

        if re_broke:
            self.retry_count += 1
            logger.info(
                f"[{self.symbol}] RE-ENTRY {self.retry_count}/{self.max_retries}: "
                f"Price {price:.2f} re-broke OR"
            )

            self.entry_price = price
            self.state = TradeState.ENTRY_PENDING
            self.pyramid_levels.clear()
            self.pyramid_count = 0
            self._current_target_level = 0
            self.current_size = 0.0

            self._pending_actions.append(
                TradeAction(
                    action=ActionType.ENTER,
                    symbol=self.symbol,
                    direction=self.direction,
                    size=self.base_unit_size,
                    reason=f"Re-entry #{self.retry_count}",
                )
            )

    def get_pending_actions(self) -> List[TradeAction]:
        actions = list(self._pending_actions)
        self._pending_actions.clear()
        return actions

    def reset_session(self):
        self.state = TradeState.IDLE
        self.direction = None
        self.entry_price = 0.0
        self.entry_time = 0.0
        self.base_unit_size = 0.0
        self.current_size = 0.0
        self.avg_entry = 0.0
        self.break_even = 0.0
        self.pyramid_levels.clear()
        self.pyramid_count = 0
        self.retry_count = 0
        self.total_shaved_profit = 0.0
        self._pending_actions.clear()
        self._entry_filled = False
        self._current_target_level = 0

    def get_state_snapshot(self) -> Dict:
        return {
            "symbol": self.symbol,
            "state": self.state.value,
            "direction": self.direction.value if self.direction else None,
            "entry_price": self.entry_price,
            "current_size": self.current_size,
            "avg_entry": self.avg_entry,
            "break_even": self.break_even,
            "pyramid_count": self.pyramid_count,
            "retry_count": self.retry_count,
            "total_shaved_profit": self.total_shaved_profit,
            "next_target": self._get_next_target_price() if self.direction else 0.0,
        }
