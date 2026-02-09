import pytest
import time
from datetime import datetime, timezone
from src.domain.signals.open_range import OpenRangeSignalEngine
from src.domain.entities import Ticker


@pytest.fixture
def or_engine():
    return OpenRangeSignalEngine(
        session_start_utc="20:00",
        session_end_utc="20:15",
        target_pct_from_mid=1.0,
        subsequent_target_pct_of_range=100.0,
    )


def test_or_session_logic(or_engine):
    symbol = "BTC-USD"

    ts_before = datetime(2026, 2, 5, 19, 59, 0, tzinfo=timezone.utc).timestamp()
    or_engine.update(
        Ticker(
            symbol=symbol,
            bid=10000,
            ask=10000,
            bid_size=1,
            ask_size=1,
            last=10000,
            volume=0,
            timestamp=ts_before,
        )
    )
    state = or_engine.states.get(symbol)
    assert state is not None
    assert not state.or_sesh

    ts_during = datetime(2026, 2, 5, 20, 0, 1, tzinfo=timezone.utc).timestamp()
    or_engine.update(
        Ticker(
            symbol=symbol,
            bid=10100,
            ask=10200,
            bid_size=1,
            ask_size=1,
            last=10150,
            volume=10,
            timestamp=ts_during,
        )
    )
    state = or_engine.states[symbol]
    assert state.or_sesh
    assert state.orh == 10200
    assert state.orl == 10100

    ts_after = datetime(2026, 2, 5, 20, 15, 1, tzinfo=timezone.utc).timestamp()
    or_engine.update(
        Ticker(
            symbol=symbol,
            bid=10150,
            ask=10150,
            bid_size=1,
            ask_size=1,
            last=10150,
            volume=5,
            timestamp=ts_after,
        )
    )
    state = or_engine.states[symbol]
    assert not state.or_sesh
    assert or_engine.is_session_just_completed()
    assert state.orm == 10150
    assert state.orw == 100


@pytest.mark.asyncio
async def test_or_session_reconstruction(or_engine):
    symbol = "BTC-USD"
    tickers = []
    base_ts = datetime(2026, 2, 5, 19, 55, 0, tzinfo=timezone.utc).timestamp()

    for i in range(30):
        ts = base_ts + (i * 60)
        price = 10000 + i
        tickers.append(
            Ticker(
                symbol=symbol,
                bid=price - 1,
                ask=price + 1,
                bid_size=1,
                ask_size=1,
                last=price,
                volume=1,
                timestamp=ts,
            )
        )

    await or_engine.update_batch(tickers)

    state = or_engine.states.get(symbol)
    assert state is not None
    assert state.orm > 0
    assert state.orh > state.orl
    assert state.session_date == "2026-02-05"


def test_or_breakout_signals(or_engine):
    symbol = "BTC-USD"

    ts_start = datetime(2026, 2, 5, 20, 0, 0, tzinfo=timezone.utc).timestamp()
    or_engine.update(Ticker(symbol, 10000, 10010, 1, 1, 10005, 10, timestamp=ts_start))
    ts_mid = ts_start + 60
    or_engine.update(Ticker(symbol, 10005, 10015, 1, 1, 10010, 10, timestamp=ts_mid))

    ts_end = datetime(2026, 2, 5, 20, 16, 0, tzinfo=timezone.utc).timestamp()
    or_engine.update(Ticker(symbol, 10010, 10010, 1, 1, 10010, 0, timestamp=ts_end))

    ts_break = ts_end + 60
    or_engine.update(Ticker(symbol, 10100, 10100, 1, 1, 10100, 1, timestamp=ts_break))

    signals = or_engine.get_signals()
    symbol_signals = signals.get(symbol, {})
    assert symbol_signals.get("breakout_signal") == 1.0


def test_or_signals_structure(or_engine):
    symbol = "BTC-USD"
    ts = datetime(2026, 2, 5, 20, 5, 0, tzinfo=timezone.utc).timestamp()
    or_engine.update(Ticker(symbol, 10000, 10100, 1, 1, 10050, 10, timestamp=ts))

    signals = or_engine.get_signals()
    assert symbol in signals
    assert "orh" in signals[symbol]
    assert "orl" in signals[symbol]
    assert "orm" in signals[symbol]
    assert "day_dir" in signals[symbol]
