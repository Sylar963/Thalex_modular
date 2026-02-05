import pytest
import time
from datetime import datetime, timezone
from src.domain.signals.open_range import OpenRangeSignalEngine
from src.domain.entities import Ticker


@pytest.fixture
def or_engine():
    # Session from 20:00 to 20:15 UTC
    return OpenRangeSignalEngine(
        session_start_utc="20:00",
        session_end_utc="20:15",
        target_pct_from_mid=1.0,  # 1%
        subsequent_target_pct_of_range=100.0,  # 100%
    )


def test_or_session_logic(or_engine):
    # 1. Before session
    ts_before = datetime(2026, 2, 5, 19, 59, 0, tzinfo=timezone.utc).timestamp()
    or_engine.update(
        Ticker(
            symbol="BTC-USD",
            bid=10000,
            ask=10000,
            bid_size=1,
            ask_size=1,
            last=10000,
            volume=0,
            timestamp=ts_before,
        )
    )
    assert not or_engine.state.or_sesh

    # 2. During session
    ts_during = datetime(2026, 2, 5, 20, 0, 1, tzinfo=timezone.utc).timestamp()
    or_engine.update(
        Ticker(
            symbol="BTC-USD",
            bid=10100,
            ask=10200,
            bid_size=1,
            ask_size=1,
            last=10150,
            volume=10,
            timestamp=ts_during,
        )
    )
    assert or_engine.state.or_sesh
    assert or_engine.state.orh == 10200
    assert or_engine.state.orl == 10100

    # 3. End of session
    ts_after = datetime(2026, 2, 5, 20, 15, 1, tzinfo=timezone.utc).timestamp()
    or_engine.update(
        Ticker(
            symbol="BTC-USD",
            bid=10150,
            ask=10150,
            bid_size=1,
            ask_size=1,
            last=10150,
            volume=5,
            timestamp=ts_after,
        )
    )
    assert not or_engine.state.or_sesh
    assert or_engine.is_session_just_completed()
    assert or_engine.state.orm == 10150  # (10200 + 10100) / 2
    assert or_engine.state.orw == 100


@pytest.mark.asyncio
async def test_or_session_reconstruction(or_engine):
    # Simulate a sequence of tickers across session start and end
    tickers = []
    base_ts = datetime(2026, 2, 5, 19, 55, 0, tzinfo=timezone.utc).timestamp()

    for i in range(30):  # 30 minutes, 1 minute interval
        ts = base_ts + (i * 60)
        price = 10000 + i
        tickers.append(
            Ticker(
                symbol="BTC-USD",
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

    # Session was 20:00 to 20:15
    # Tickers at index 5 to 20 should be in-session
    # High should be around 10015+1 = 10016? or something like that.
    # Let's check state
    assert or_engine.state.orm > 0
    assert or_engine.state.orh > or_engine.state.orl
    assert or_engine.state.session_date == "2026-02-05"


def test_or_breakout_signals(or_engine):
    # Set up a completed session
    ts_start = datetime(2026, 2, 5, 20, 0, 0, tzinfo=timezone.utc).timestamp()
    or_engine.update(
        Ticker("BTC-USD", 10000, 10010, 1, 1, 10005, 10, timestamp=ts_start)
    )
    ts_mid = ts_start + 60
    or_engine.update(Ticker("BTC-USD", 10005, 10015, 1, 1, 10010, 10, timestamp=ts_mid))
    ts_end = datetime(2026, 2, 5, 20, 16, 0, tzinfo=timezone.utc).timestamp()
    or_engine.update(Ticker("BTC-USD", 10010, 10010, 1, 1, 10010, 0, timestamp=ts_end))

    # Mid 10007.5, ORH 10015, ORL 10000
    # Current Close 10010 (Inside)

    # Breakout UP
    ts_break = ts_end + 60
    or_engine.update(
        Ticker("BTC-USD", 10100, 10100, 1, 1, 10100, 1, timestamp=ts_break)
    )
    signals = or_engine.get_signals()
    assert signals["breakout_signal"] == 1.0
