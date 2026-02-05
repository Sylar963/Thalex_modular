import pytest
import asyncio
import time
from unittest.mock import MagicMock, AsyncMock
from src.use_cases.strategy_manager import MultiExchangeStrategyManager, ExchangeConfig
from src.domain.sim_match_engine import SimMatchEngine
from src.domain.signals.volume_candle import VolumeCandleSignalEngine
from src.domain.signals.open_range import OpenRangeSignalEngine
from src.domain.risk.basic_manager import BasicRiskManager
from src.domain.tracking.sync_engine import SyncEngine
from src.domain.entities import Ticker, OrderSide, Trade, Position
from tests.e2e.mock_gateway import MockExchangeGateway
from src.domain.interfaces import Strategy, MarketState


# Local Mock Strategy because we need one to pass to Manager
class MockStrategy(Strategy):
    async def setup(self, config):
        pass

    def calculate_quotes(self, market_state: MarketState, position, tick_size=0.5):
        # We don't place standard quotes in this test, we test the manager's overrides
        return []


@pytest.mark.asyncio
async def test_momentum_add_flow():
    # 1. Setup Simulation Environment
    sim_engine = SimMatchEngine(latency_ms=0)
    sim_engine.set_initial_state(balance=10000.0, position_size=0.0)

    gateway = MockExchangeGateway("mock_exchange", sim_engine)

    # 2. Setup Components
    config = ExchangeConfig(gateway=gateway, symbol="BTC-PERP")

    risk_manager = BasicRiskManager(max_position=100.0)
    sync_engine = SyncEngine()

    vamp_engine = VolumeCandleSignalEngine(volume_threshold=100.0)
    or_engine = OpenRangeSignalEngine(
        session_start_utc="00:00", session_end_utc="23:59"
    )

    # Mock storage for trend service
    mock_storage = MagicMock()
    mock_storage.save_ticker = AsyncMock()
    mock_storage.save_trade = AsyncMock()
    mock_storage.save_signal = AsyncMock()

    manager = MultiExchangeStrategyManager(
        exchanges=[config],
        strategy=MockStrategy(),
        risk_manager=risk_manager,
        sync_engine=sync_engine,
        signal_engine=vamp_engine,
        or_engine=or_engine,
        storage=mock_storage,
        dry_run=False,
    )

    # Mock Trend Service to return UP trend
    manager.trend_service = MagicMock()
    manager.trend_service.get_trend_side = MagicMock(return_value="UP")
    manager.trend_service.get_trend_14d = AsyncMock(return_value=0.05)
    manager._venue_trends["BTC-PERP"] = 0.05

    await manager.start()

    # 3. Simulate Market Data to Trigger Momentum Add
    # Conditions: VAMP > 0.7, Price > ORM, Trend UP

    # Set ORM
    base_ts = time.time()
    or_engine.state.orm = 10000.0

    # Push Ticker above ORM
    ticker = Ticker(
        symbol="BTC-PERP",
        bid=10100,
        ask=10102,
        bid_size=1,
        ask_size=1,
        last=10101,
        volume=100,
        timestamp=base_ts,
        exchange="mock_exchange",
    )
    await gateway.push_ticker(ticker)

    # Push Trades to spike VAMP Impact
    # Create strong buying pressure
    for i in range(10):
        t = Trade(
            id=f"t_{i}",
            order_id=f"o_{i}",
            symbol="BTC-PERP",
            price=10100 + i,
            size=150.0,
            side=OrderSide.BUY,
            timestamp=base_ts + i,
        )
        await gateway.push_trade(t)
        # Also need to trigger manager update loop via ticker
        await gateway.push_ticker(ticker)
        print(
            f"[DEBUG_LOOP] i={i}, Impact={vamp_engine.get_signals().get('market_impact', 0.0)}"
        )

    # Allow async tasks to process
    await asyncio.sleep(0.5)

    # 4. Assertions
    print(f"\n[DEBUG] VAMP Signals: {vamp_engine.get_signals()}")
    print(f"\n[DEBUG] Manager Adds: {manager._momentum_adds}")

    # Check if Momentum Add was placed in SimEngine
    # It should be a MARKET BUY
    fills = sim_engine.fills
    open_bids = sim_engine.bid_book

    # Either filled or in book (SimEngine treats Market as Aggressive Limit)
    assert len(fills) > 0 or len(open_bids) > 0, "No orders placed for Momentum Add"

    if len(fills) > 0:
        assert fills[0].side == "buy" or fills[0].side == OrderSide.BUY
    else:
        assert open_bids[0].order.side == OrderSide.BUY

    # Verify Manager State
    adds = manager._momentum_adds.get("mock_exchange:BTC-PERP", [])
    assert len(adds) == 1
    assert adds[0]["side"] == OrderSide.BUY

    print("\n[SUCCESS] Momentum Add Triggered and Executed")

    await manager.stop()


@pytest.mark.asyncio
async def test_smart_reducing_mode():
    # 1. Setup
    sim_engine = SimMatchEngine(latency_ms=0)
    # Start with a SHORT position (Counter-Flow to UP trend)
    sim_engine.set_initial_state(
        balance=10000.0, position_size=-5.0, entry_price=10000.0
    )

    gateway = MockExchangeGateway("mock_exchange", sim_engine)
    config = ExchangeConfig(gateway=gateway, symbol="BTC-PERP")

    risk_manager = BasicRiskManager(max_position=4.0)  # Limit is 4, we have 5 -> BREACH
    risk_manager.update_position(
        Position("BTC-PERP", -5.0, 10000.0, exchange="mock_exchange")
    )

    sync_engine = SyncEngine()
    vamp_engine = VolumeCandleSignalEngine()
    or_engine = OpenRangeSignalEngine()

    sync_engine = SyncEngine()
    vamp_engine = VolumeCandleSignalEngine()
    or_engine = OpenRangeSignalEngine()

    manager = MultiExchangeStrategyManager(
        exchanges=[config],
        strategy=MockStrategy(),
        risk_manager=risk_manager,
        sync_engine=sync_engine,
        signal_engine=vamp_engine,
        or_engine=or_engine,
        storage=None,  # No storage for this test to avoid mock complexity if not needed
        dry_run=False,
    )

    # Trend is UP (so our Short -5 is Counter-Trend)
    manager.trend_service = MagicMock()
    manager.trend_service.get_trend_side = MagicMock(return_value="UP")
    manager.trend_service.get_trend_14d = AsyncMock(return_value=0.05)
    manager._venue_trends["BTC-PERP"] = 0.05

    # Inject initial position into portfolio manually to match sim
    manager.portfolio.set_position(
        Position("BTC-PERP", -5.0, 10000.0, exchange="mock_exchange")
    )

    await manager.start()

    # 2. Trigger Logic
    # Push ticker to trigger strategy loop
    ticker = Ticker(
        symbol="BTC-PERP",
        bid=10200,
        ask=10205,
        bid_size=1,
        ask_size=1,
        last=10202,
        volume=100,
        timestamp=time.time(),
        exchange="mock_exchange",
    )
    await gateway.push_ticker(ticker)

    await asyncio.sleep(0.5)

    # 3. Assertions
    # Should have triggered Aggressive Reduce (Market Buy to Close) or Limit Buy at Ask
    # Logic: if counter-trend, place Limit Order at Ask (Active Aggressive)
    # Check Sim Orders
    bids = sim_engine.bid_book
    assert len(bids) > 0 or len(sim_engine.fills) > 0

    if len(bids) > 0:
        # Check price is aggressive (at ask)
        assert bids[0].order.price >= 10205
        assert bids[0].order.size == 5.0
        print("\n[SUCCESS] Aggressive Reduce Order Placed")

    await manager.stop()
