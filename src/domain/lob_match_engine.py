import time
import asyncio
import logging
from typing import List, Dict, Optional, Callable
from dataclasses import dataclass
from .entities import Order, OrderSide, OrderStatus, Trade, Ticker
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


class LOBMatchEngine:
    def __init__(
        self,
        latency_ms: float = 50.0,
        maker_fee: float = -0.0001,
        taker_fee: float = 0.0003,
        slippage_ticks: float = 1.0,
        tick_size: float = 0.5,
    ):
        self.latency_ms = latency_ms
        self.maker_fee = maker_fee
        self.taker_fee = taker_fee
        self.slippage_ticks = slippage_ticks
        self.tick_size = tick_size

        self.bid_book: List[SimOrder] = []
        self.ask_book: List[SimOrder] = []

        self.balance: float = 0.0
        self.position_size: float = 0.0
        self.position_entry_price: float = 0.0
        self.fills: List[FillEffect] = []
        self.fill_callback: Optional[Callable] = None

    def on_ticker(self, ticker: Ticker):
        now = ticker.timestamp
        self._match_orders(ticker.bid, ticker.ask, now)

    def _match_orders(self, bid: float, ask: float, now: float):
        filled_bids = []
        for o in self.bid_book:
            if o.active_time > now:
                continue
            if ask <= o.order.price:
                self._execute_fill(o, o.order.price, o.order.size, now, is_maker=True)
                filled_bids.append(o)

        for o in filled_bids:
            self.bid_book.remove(o)

        filled_asks = []
        for o in self.ask_book:
            if o.active_time > now:
                continue
            if bid >= o.order.price:
                self._execute_fill(o, o.order.price, o.order.size, now, is_maker=True)
                filled_asks.append(o)

        for o in filled_asks:
            self.ask_book.remove(o)

    def submit_order(self, order: Order, timestamp: float):
        active_time = timestamp + (self.latency_ms / 1000.0)
        sim_order = SimOrder(order, timestamp, active_time)

        if order.side == OrderSide.BUY:
            self.bid_book.append(sim_order)
            self.bid_book.sort(key=lambda x: -x.order.price)
        else:
            self.ask_book.append(sim_order)
            self.ask_book.sort(key=lambda x: x.order.price)

    def _execute_fill(
        self, sim_order: SimOrder, price: float, size: float, ts: float, is_maker: bool
    ):
        fee_rate = self.maker_fee if is_maker else self.taker_fee
        fee = price * size * fee_rate

        old_pos = self.position_size
        side = sim_order.order.side
        sign = 1 if side == OrderSide.BUY else -1
        new_pos = old_pos + (sign * size)

        realized_pnl = 0.0
        if (old_pos > 0 and side == OrderSide.SELL) or (
            old_pos < 0 and side == OrderSide.BUY
        ):
            closed_size = min(abs(old_pos), size)
            dir_mult = 1 if old_pos > 0 else -1
            realized_pnl = (price - self.position_entry_price) * closed_size * dir_mult

        self.balance -= fee
        self.balance += realized_pnl

        if new_pos == 0:
            self.position_entry_price = 0.0
        elif abs(new_pos) > abs(old_pos):
            total_val = abs(old_pos) * self.position_entry_price + size * price
            self.position_entry_price = total_val / abs(new_pos)

        self.position_size = new_pos

        fill = FillEffect(
            timestamp=ts,
            symbol=sim_order.order.symbol,
            side=side.value if hasattr(side, "value") else str(side),
            price=price,
            size=size,
            fee=fee,
            realized_pnl=realized_pnl,
            balance_after=self.balance,
        )
        self.fills.append(fill)

    def get_equity(self, current_price: float) -> float:
        unrealized = (
            (current_price - self.position_entry_price) * self.position_size
            if self.position_size != 0
            else 0.0
        )
        return self.balance + unrealized
