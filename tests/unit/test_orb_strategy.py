import pytest
from src.domain.strategies.orb_breakout import (
    ORBBreakoutStrategy,
    TradeDirection,
    TradeSignal,
)


@pytest.fixture
def strategy():
    return ORBBreakoutStrategy(risk_pct=17.0, leverage=10, max_concurrent=5)


def test_compute_unit_size(strategy):
    size = strategy.compute_unit_size(equity=234.0, price=29.4)
    assert abs(size - 13.5) < 0.1


def test_compute_unit_size_zero_price(strategy):
    assert strategy.compute_unit_size(equity=234.0, price=0.0) == 0.0


def test_compute_unit_size_zero_equity(strategy):
    assert strategy.compute_unit_size(equity=0.0, price=29.4) == 0.0


def test_evaluate_no_signal(strategy):
    signals = {
        "session_active": 0.0,
        "or_token": 1.0,
        "breakout_signal": 0.0,
        "orh": 100.0,
        "orl": 99.0,
        "orm": 99.5,
        "orw": 1.0,
    }
    result = strategy.evaluate("BTCUSDT", signals, 100.5, 1000.0, 234.0)
    assert result is None


def test_evaluate_session_active(strategy):
    signals = {
        "session_active": 1.0,
        "or_token": 1.0,
        "breakout_signal": 1.0,
        "orh": 100.0,
        "orl": 99.0,
        "orm": 99.5,
        "orw": 1.0,
    }
    result = strategy.evaluate("BTCUSDT", signals, 100.5, 1000.0, 234.0)
    assert result is None


def test_evaluate_bullish_breakout(strategy):
    signals = {
        "session_active": 0.0,
        "or_token": 1.0,
        "breakout_signal": 1.0,
        "orh": 100.0,
        "orl": 99.0,
        "orm": 99.5,
        "orw": 1.0,
        "first_up_target": 101.38,
        "first_down_target": 97.62,
    }
    result = strategy.evaluate("BTCUSDT", signals, 100.5, 1000.0, 234.0)
    assert result is not None
    assert result.direction == TradeDirection.LONG
    assert result.symbol == "BTCUSDT"
    assert result.base_unit_size > 0


def test_evaluate_bearish_breakout(strategy):
    signals = {
        "session_active": 0.0,
        "or_token": 1.0,
        "breakout_signal": -1.0,
        "orh": 100.0,
        "orl": 99.0,
        "orm": 99.5,
        "orw": 1.0,
        "first_up_target": 101.38,
        "first_down_target": 97.62,
    }
    result = strategy.evaluate("BTCUSDT", signals, 98.5, 1000.0, 234.0)
    assert result is not None
    assert result.direction == TradeDirection.SHORT


def test_evaluate_max_concurrent(strategy):
    strategy.active_count = 5
    signals = {
        "session_active": 0.0,
        "or_token": 1.0,
        "breakout_signal": 1.0,
        "orh": 100.0,
        "orl": 99.0,
        "orm": 99.5,
        "orw": 1.0,
        "first_up_target": 101.38,
        "first_down_target": 97.62,
    }
    result = strategy.evaluate("BTCUSDT", signals, 100.5, 1000.0, 234.0)
    assert result is None


def test_evaluate_duplicate_signal_ignored(strategy):
    signals = {
        "session_active": 0.0,
        "or_token": 1.0,
        "breakout_signal": 1.0,
        "orh": 100.0,
        "orl": 99.0,
        "orm": 99.5,
        "orw": 1.0,
        "first_up_target": 101.38,
        "first_down_target": 97.62,
    }
    result1 = strategy.evaluate("BTCUSDT", signals, 100.5, 1000.0, 234.0)
    assert result1 is not None

    result2 = strategy.evaluate("BTCUSDT", signals, 100.8, 1001.0, 234.0)
    assert result2 is None


def test_active_count_tracking(strategy):
    assert strategy.active_count == 0
    strategy.on_trade_opened()
    assert strategy.active_count == 1
    strategy.on_trade_closed()
    assert strategy.active_count == 0
    strategy.on_trade_closed()
    assert strategy.active_count == 0
