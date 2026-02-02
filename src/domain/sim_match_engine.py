import time
import asyncio
import logging
from typing import List, Dict, Optional, Callable
from dataclasses import dataclass, field
from collections import deque
from .entities import Order, OrderSide, OrderStatus, Trade
from .entities.pnl import FillEffect

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class SimOrder:
    order: Order
    submit_time: float
    active_time: float
    filled_size: float = 0.0
    avg_fill_price: float = 0.0
    status: str = "pending"


class SimMatchEngine:
    def __init__(
        self,
        latency_ms: float = 50.0,
        slippage_ticks: float = 0.0,
        tick_size: float = 0.5,
        maker_fee: float = -0.0001,
        taker_fee: float = 0.0003,
    ):
        self.latency_ms = latency_ms
        self.slippage_ticks = slippage_ticks
        self.tick_size = tick_size
        self.maker_fee = maker_fee
        self.taker_fee = taker_fee

        self.bid_book: List[SimOrder] = []
        self.ask_book: List[SimOrder] = []

        self.balance: float = 0.0
        self.position_size: float = 0.0
        self.position_entry_price: float = 0.0

        self.fills: List[FillEffect] = []
        self.fill_callback: Optional[Callable] = None

    def set_initial_state(
        self, balance: float, position_size: float = 0.0, entry_price: float = 0.0
    ):
        self.balance = balance
        self.position_size = position_size
        self.position_entry_price = entry_price

    def set_fill_callback(self, callback: Callable):
        self.fill_callback = callback

    def submit_order(self, order: Order):
        now = time.time()
        active_time = now + (self.latency_ms / 1000.0)

        sim_order = SimOrder(
            order=order,
            submit_time=now,
            active_time=active_time,
            filled_size=0.0,
            avg_fill_price=0.0,
            status="pending",
        )

        if order.side == OrderSide.BUY:
            self.bid_book.append(sim_order)
            self.bid_book.sort(key=lambda x: -x.order.price)
        else:
            self.ask_book.append(sim_order)
            self.ask_book.sort(key=lambda x: x.order.price)

    def cancel_order(self, order_id: str) -> bool:
        for book in [self.bid_book, self.ask_book]:
            for i, sim_order in enumerate(book):
                if sim_order.order.id == order_id:
                    book.pop(i)
                    return True
        return False

    def cancel_all(self):
        self.bid_book.clear()
        self.ask_book.clear()

    def on_trade(self, trade: Trade):
        now = time.time()
        trade_price = trade.price
        trade_size = trade.size

        self._match_bids(trade_price, trade_size, now)
        self._match_asks(trade_price, trade_size, now)

    def _match_bids(self, trade_price: float, trade_size: float, now: float):
        remaining = trade_size
        filled_orders = []

        for sim_order in self.bid_book:
            if remaining <= 0:
                break
            if sim_order.active_time > now:
                continue
            if sim_order.status == "filled":
                continue

            if trade_price <= sim_order.order.price:
                fill_price = sim_order.order.price - (
                    self.slippage_ticks * self.tick_size
                )
                fill_size = min(sim_order.order.size - sim_order.filled_size, remaining)

                self._execute_fill(sim_order, fill_price, fill_size, now, is_maker=True)
                remaining -= fill_size

                if sim_order.filled_size >= sim_order.order.size:
                    sim_order.status = "filled"
                    filled_orders.append(sim_order)

        for fo in filled_orders:
            if fo in self.bid_book:
                self.bid_book.remove(fo)

    def _match_asks(self, trade_price: float, trade_size: float, now: float):
        remaining = trade_size
        filled_orders = []

        for sim_order in self.ask_book:
            if remaining <= 0:
                break
            if sim_order.active_time > now:
                continue
            if sim_order.status == "filled":
                continue

            if trade_price >= sim_order.order.price:
                fill_price = sim_order.order.price + (
                    self.slippage_ticks * self.tick_size
                )
                fill_size = min(sim_order.order.size - sim_order.filled_size, remaining)

                self._execute_fill(sim_order, fill_price, fill_size, now, is_maker=True)
                remaining -= fill_size

                if sim_order.filled_size >= sim_order.order.size:
                    sim_order.status = "filled"
                    filled_orders.append(sim_order)

        for fo in filled_orders:
            if fo in self.ask_book:
                self.ask_book.remove(fo)

    def _execute_fill(
        self,
        sim_order: SimOrder,
        fill_price: float,
        fill_size: float,
        timestamp: float,
        is_maker: bool,
    ):
        fee_rate = self.maker_fee if is_maker else self.taker_fee
        fee = fill_price * fill_size * fee_rate

        old_pos = self.position_size
        side = sim_order.order.side
        sign = 1 if side == OrderSide.BUY else -1
        new_pos = old_pos + (sign * fill_size)

        realized_pnl = 0.0
        if (old_pos > 0 and side == OrderSide.SELL) or (
            old_pos < 0 and side == OrderSide.BUY
        ):
            closed_size = min(abs(old_pos), fill_size)
            dir_mult = 1 if old_pos > 0 else -1
            realized_pnl = (
                (fill_price - self.position_entry_price) * closed_size * dir_mult
            )

        self.balance -= fee
        self.balance += realized_pnl

        if new_pos == 0:
            self.position_entry_price = 0.0
        elif abs(new_pos) > abs(old_pos):
            total_val = (
                abs(old_pos) * self.position_entry_price + fill_size * fill_price
            )
            self.position_entry_price = (
                total_val / abs(new_pos) if abs(new_pos) > 0 else 0.0
            )

        self.position_size = new_pos

        sim_order.filled_size += fill_size
        if sim_order.avg_fill_price == 0:
            sim_order.avg_fill_price = fill_price
        else:
            total_filled = sim_order.filled_size
            prev_filled = total_filled - fill_size
            sim_order.avg_fill_price = (
                (sim_order.avg_fill_price * prev_filled) + (fill_price * fill_size)
            ) / total_filled

        fill_effect = FillEffect(
            timestamp=timestamp,
            symbol=sim_order.order.symbol,
            side=side.value if hasattr(side, "value") else str(side),
            price=fill_price,
            size=fill_size,
            fee=fee,
            realized_pnl=realized_pnl,
            balance_after=self.balance,
        )

        self.fills.append(fill_effect)

        if self.fill_callback:
            asyncio.create_task(self._notify_fill(fill_effect))

        logger.info(
            f"SIM FILL: {side.value} {fill_size:.4f} @ {fill_price:.2f} | "
            f"PNL: {realized_pnl:.4f} | Balance: {self.balance:.4f} | Pos: {self.position_size:.4f}"
        )

    async def _notify_fill(self, fill: FillEffect):
        if self.fill_callback:
            await self.fill_callback(fill)

    def get_equity(self, current_price: float) -> float:
        unrealized = (
            (current_price - self.position_entry_price) * self.position_size
            if self.position_size != 0
            else 0.0
        )
        return self.balance + unrealized

    def get_state(self) -> Dict:
        return {
            "balance": self.balance,
            "position_size": self.position_size,
            "position_entry_price": self.position_entry_price,
            "open_bids": len(self.bid_book),
            "open_asks": len(self.ask_book),
            "total_fills": len(self.fills),
        }
