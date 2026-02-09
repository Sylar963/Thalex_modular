import pytest
from src.domain.signals.inventory_bias import InventoryBiasEngine


@pytest.fixture
def bias_engine():
    return InventoryBiasEngine(
        or_weight=0.4, vamp_weight=0.6, suppression_threshold=0.3
    )


def test_suppress_asks_when_short_bullish_momentum(bias_engine):
    bias_engine.update_position(-1.0)
    bias_engine.update_signals(
        or_signals={
            "day_dir": 1.0,
            "breakout_signal": 1.0,
            "orm": 95000,
            "current_price": 96000,
        },
        vamp_signals={"market_impact": 0.8},
    )
    signals = bias_engine.get_signals()
    assert signals["suppress_asks"] > 0.5
    assert signals["suppress_bids"] == 0.0
    assert signals["bias_direction"] > 0.3


def test_suppress_bids_when_long_bearish_momentum(bias_engine):
    bias_engine.update_position(1.0)
    bias_engine.update_signals(
        or_signals={
            "day_dir": -1.0,
            "breakout_signal": -1.0,
            "orm": 95000,
            "current_price": 94000,
        },
        vamp_signals={"market_impact": -0.7},
    )
    signals = bias_engine.get_signals()
    assert signals["suppress_bids"] > 0.5
    assert signals["suppress_asks"] == 0.0
    assert signals["bias_direction"] < -0.3


def test_no_suppression_when_mean_reverting(bias_engine):
    bias_engine.update_position(0.5)
    bias_engine.update_signals(
        or_signals={
            "day_dir": 0.0,
            "breakout_signal": 0.0,
            "orm": 95000,
            "current_price": 95000,
        },
        vamp_signals={"market_impact": 0.1},
    )
    signals = bias_engine.get_signals()
    assert signals["suppress_bids"] < 0.1
    assert signals["suppress_asks"] < 0.1


def test_no_suppression_when_flat_position(bias_engine):
    bias_engine.update_position(0.0)
    bias_engine.update_signals(
        or_signals={
            "day_dir": 1.0,
            "breakout_signal": 1.0,
            "orm": 95000,
            "current_price": 96000,
        },
        vamp_signals={"market_impact": 0.9},
    )
    signals = bias_engine.get_signals()
    assert signals["suppress_bids"] == 0.0
    assert signals["suppress_asks"] == 0.0


def test_or_weight_dominance(bias_engine):
    bias_engine.update_position(-1.0)
    bias_engine.update_signals(
        or_signals={
            "day_dir": 1.0,
            "breakout_signal": 1.0,
            "orm": 95000,
            "current_price": 96000,
        },
        vamp_signals={"market_impact": 0.0},
    )
    signals = bias_engine.get_signals()
    assert signals["bias_direction"] > 0.2


def test_vamp_weight_dominance(bias_engine):
    bias_engine.update_position(-1.0)
    bias_engine.update_signals(
        or_signals={"day_dir": 0.0, "breakout_signal": 0.0},
        vamp_signals={"market_impact": 0.9},
    )
    signals = bias_engine.get_signals()
    assert signals["bias_direction"] > 0.4
