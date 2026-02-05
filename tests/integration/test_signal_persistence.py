import pytest
import asyncio
import time
import os
from dotenv import load_dotenv
from src.adapters.storage.timescale_adapter import TimescaleDBAdapter
from src.domain.signals.volume_candle import VolumeCandleSignalEngine
from src.domain.entities import Trade, OrderSide


def get_test_dsn():
    load_dotenv()
    db_user = os.getenv("DATABASE_USER", "postgres")
    db_pass = os.getenv("DATABASE_PASSWORD", "password")
    db_host = os.getenv("DATABASE_HOST", "localhost")
    db_name = os.getenv("DATABASE_NAME", "thalex_trading")
    db_port = os.getenv("DATABASE_PORT", "5432")
    return f"postgresql://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}"


@pytest.mark.asyncio
async def test_signal_persistence_vamp():
    # Setup Storage
    adapter = TimescaleDBAdapter(get_test_dsn())
    await adapter.connect()

    # Setup Engine
    engine = VolumeCandleSignalEngine(volume_threshold=100.0, max_candles=5)

    # Simulate trade resulting in candle completion
    symbol = "BTC-PERP"
    trade = Trade(
        id="test_t1",
        order_id="test_oid1",
        symbol=symbol,
        price=10000,
        size=150.0,
        side=OrderSide.BUY,
        timestamp=time.time(),
    )

    # This usually happens in StrategyManager, but we test the persistency directly
    engine.update_trade(trade)
    if engine.pop_completed_candle():
        signals = engine.get_signals()
        await adapter.save_signal(symbol, "vamp", signals)

    # Verify persistence
    history = await adapter.get_signal_history(
        symbol, time.time() - 60, time.time() + 60, signal_type="vamp"
    )

    assert len(history) >= 1
    latest = history[-1]
    assert latest["symbol"] == symbol
    assert latest["signal_type"] == "vamp"

    await adapter.disconnect()


@pytest.mark.asyncio
async def test_signal_persistence_open_range():
    adapter = TimescaleDBAdapter(get_test_dsn())
    await adapter.connect()

    symbol = "BTC-PERP-OR"
    signals = {
        "orh": 10500.0,
        "orl": 10400.0,
        "orm": 10450.0,
        "breakout_direction": "UP",
    }

    await adapter.save_signal(symbol, "open_range", signals)

    history = await adapter.get_signal_history(
        symbol, time.time() - 60, time.time() + 60, signal_type="open_range"
    )
    assert len(history) >= 1
    assert history[-1]["orh"] == 10500.0

    await adapter.disconnect()
