import pytest
import asyncio
import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.domain.entities import (
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Ticker,
    Position,
    Balance,
)
from src.domain.lob_match_engine import LOBMatchEngine


_counter = 0


def _uid():
    global _counter
    _counter += 1
    return f"test_{_counter}"


class TestLOBMatchEngine:
    def setup_method(self):
        self.engine = LOBMatchEngine(
            latency_ms=0.0,
            maker_fee=-0.0001,
            taker_fee=0.0003,
            slippage_ticks=0.0,
            tick_size=0.01,
        )
        self.engine.balance = 1000.0

    def _make_order(self, side, price, size=1.0, symbol="TEST"):
        return Order(
            id=_uid(),
            symbol=symbol,
            side=side,
            price=price,
            size=size,
        )

    def _make_ticker(self, bid, ask, ts=None):
        return Ticker(
            symbol="TEST",
            bid=bid,
            ask=ask,
            bid_size=10.0,
            ask_size=10.0,
            last=(bid + ask) / 2,
            volume=0,
            timestamp=ts or time.time(),
        )

    def test_bid_fill_on_ask_cross(self):
        order = self._make_order(OrderSide.BUY, 100.0)
        self.engine.submit_order(order, time.time())
        assert len(self.engine.bid_book) == 1

        ticker = self._make_ticker(99.0, 100.0)
        self.engine.on_ticker(ticker)

        assert len(self.engine.bid_book) == 0
        assert len(self.engine.fills) == 1
        assert self.engine.position_size == 1.0

    def test_ask_fill_on_bid_cross(self):
        order = self._make_order(OrderSide.SELL, 100.0)
        self.engine.submit_order(order, time.time())
        assert len(self.engine.ask_book) == 1

        ticker = self._make_ticker(100.0, 101.0)
        self.engine.on_ticker(ticker)

        assert len(self.engine.ask_book) == 0
        assert len(self.engine.fills) == 1
        assert self.engine.position_size == -1.0

    def test_no_fill_when_price_doesnt_cross(self):
        order = self._make_order(OrderSide.BUY, 99.0)
        self.engine.submit_order(order, time.time())

        ticker = self._make_ticker(98.0, 100.0)
        self.engine.on_ticker(ticker)

        assert len(self.engine.bid_book) == 1
        assert len(self.engine.fills) == 0

    def test_latency_prevents_immediate_fill(self):
        engine = LOBMatchEngine(latency_ms=1000.0, tick_size=0.01)
        engine.balance = 1000.0

        now = time.time()
        order = self._make_order(OrderSide.BUY, 100.0)
        engine.submit_order(order, now)

        ticker = self._make_ticker(99.0, 100.0, ts=now + 0.1)
        engine.on_ticker(ticker)
        assert len(engine.fills) == 0

        ticker_later = self._make_ticker(99.0, 100.0, ts=now + 2.0)
        engine.on_ticker(ticker_later)
        assert len(engine.fills) == 1

    def test_cancel_order(self):
        order = self._make_order(OrderSide.BUY, 100.0)
        self.engine.submit_order(order, time.time())
        assert len(self.engine.bid_book) == 1

        result = self.engine.cancel_order(order.id)
        assert result is True
        assert len(self.engine.bid_book) == 0

    def test_cancel_nonexistent_order(self):
        result = self.engine.cancel_order("nonexistent")
        assert result is False

    def test_cancel_all(self):
        self.engine.submit_order(self._make_order(OrderSide.BUY, 99.0), time.time())
        self.engine.submit_order(self._make_order(OrderSide.SELL, 101.0), time.time())
        assert len(self.engine.bid_book) == 1
        assert len(self.engine.ask_book) == 1

        self.engine.cancel_all()
        assert len(self.engine.bid_book) == 0
        assert len(self.engine.ask_book) == 0

    def test_get_open_orders(self):
        o1 = self._make_order(OrderSide.BUY, 99.0)
        o2 = self._make_order(OrderSide.SELL, 101.0)
        self.engine.submit_order(o1, time.time())
        self.engine.submit_order(o2, time.time())

        open_orders = self.engine.get_open_orders()
        assert len(open_orders) == 2

    def test_get_state(self):
        state = self.engine.get_state()
        assert state["balance"] == 1000.0
        assert state["position_size"] == 0.0
        assert state["open_bids"] == 0
        assert state["open_asks"] == 0

    def test_roundtrip_pnl(self):
        buy = self._make_order(OrderSide.BUY, 100.0, size=1.0)
        self.engine.submit_order(buy, time.time())
        self.engine.on_ticker(self._make_ticker(99.0, 100.0))

        sell = self._make_order(OrderSide.SELL, 105.0, size=1.0)
        self.engine.submit_order(sell, time.time())
        self.engine.on_ticker(self._make_ticker(105.0, 106.0))

        assert self.engine.position_size == 0.0
        assert self.engine.position_entry_price == 0.0
        assert len(self.engine.fills) == 2
        total_pnl = sum(f.realized_pnl for f in self.engine.fills)
        assert total_pnl == pytest.approx(5.0, abs=0.1)

    def test_equity_calculation(self):
        buy = self._make_order(OrderSide.BUY, 100.0, size=1.0)
        self.engine.submit_order(buy, time.time())
        self.engine.on_ticker(self._make_ticker(99.0, 100.0))

        equity = self.engine.get_equity(110.0)
        expected_unrealized = (110.0 - 100.0) * 1.0
        assert equity == pytest.approx(1000.0 + expected_unrealized, abs=0.5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
