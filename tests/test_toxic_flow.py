import asyncio
from unittest.mock import AsyncMock, MagicMock
import time

from src.use_cases.quoting_service import QuotingService
from src.domain.signals.volume_candle import VolumeCandleSignalEngine
from src.domain.entities import (
    Order,
    OrderSide,
    OrderStatus,
    Trade,
    Position,
    OrderType,
)
from src.domain.interfaces import ExchangeGateway, Strategy, RiskManager


async def test_toxic_flow_defense_buys_cancel_asks():
    """
    Simulate a rapid sequence of BUY trades (lifting offers).
    Verify that the QuotingService detects 'Toxic BUY' flow and cancels ASKS.
    """
    # 1. Setup Mocks
    gateway = MagicMock(spec=ExchangeGateway)
    gateway.cancel_order = AsyncMock()
    strategy = MagicMock(spec=Strategy)
    risk_manager = MagicMock(spec=RiskManager)

    # 2. Setup Real Signal Engine (logic under test)
    signal_engine = VolumeCandleSignalEngine(volume_threshold=10.0, max_candles=10)

    # 3. Setup Service
    service = QuotingService(gateway, strategy, signal_engine, risk_manager)
    service.running = True
    service.symbol = "BTC-PERPETUAL"

    # Note: on_ticker_update usually populates active_orders via reconcile.
    # Here we manually seed active orders.

    # Seed Active ASKS (which should be cancelled on Toxic Buy flow)
    active_ask_1 = Order(
        id="ask1",
        symbol="BTC-PERPETUAL",
        side=OrderSide.SELL,
        price=100,
        size=1,
        exchange_id="exch_ask1",
        status=OrderStatus.OPEN,
    )
    active_ask_2 = Order(
        id="ask2",
        symbol="BTC-PERPETUAL",
        side=OrderSide.SELL,
        price=101,
        size=1,
        exchange_id="exch_ask2",
        status=OrderStatus.OPEN,
    )
    service.active_orders[OrderSide.SELL] = [active_ask_1, active_ask_2]

    # Seed Active BIDS (should remain or be handled separately)
    active_bid_1 = Order(
        id="bid1",
        symbol="BTC-PERPETUAL",
        side=OrderSide.BUY,
        price=90,
        size=1,
        exchange_id="exch_bid1",
        status=OrderStatus.OPEN,
    )
    service.active_orders[OrderSide.BUY] = [active_bid_1]

    # 4. Simulate Toxic BUY Flow
    # Send multiple BUY trades quickly.
    # Total volume = 10 (threshold)
    # Buy Volume = 9, Sell Volume = 1 -> Impact ~0.8 -> Toxic > 0.7

    ts = time.time()

    # Trade 1: Small Buy
    t1 = Trade(
        id="t1",
        order_id="o1",
        symbol="BTC-PERPETUAL",
        side=OrderSide.BUY,
        price=100,
        size=5.0,
        timestamp=ts,
    )
    await service.on_trade_update(t1)

    # Immediate flow calc: 5 buy / 5 total = 1.0 impact * (5/10 significance) = 0.5 signal.
    # Threshold 0.7 not met yet.
    assert gateway.cancel_order.call_count == 0

    # Trade 2: Another Buy
    t2 = Trade(
        id="t2",
        order_id="o2",
        symbol="BTC-PERPETUAL",
        side=OrderSide.BUY,
        price=100,
        size=4.0,
        timestamp=ts,
    )
    await service.on_trade_update(t2)

    # Current Candle: 9 buy. Volume 9.
    # Impact 1.0 * 0.9 = 0.9 signal. > 0.7 triggers defense!

    # 5. Verify Cancellations
    # Should cancel ASKS
    assert gateway.cancel_order.call_count >= 2

    # Verify arguments
    calls = [c.args[0] for c in gateway.cancel_order.call_args_list]
    assert "exch_ask1" in calls
    assert "exch_ask2" in calls
    # Bids should NOT be cancelled by BUY toxic flow
    assert "exch_bid1" not in calls

    # Verify active orders cleared
    assert len(service.active_orders[OrderSide.SELL]) == 0
    assert len(service.active_orders[OrderSide.BUY]) == 1


if __name__ == "__main__":
    asyncio.run(test_toxic_flow_defense_buys_cancel_asks())
