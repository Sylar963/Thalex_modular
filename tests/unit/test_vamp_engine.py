import pytest
import time
from src.domain.signals.volume_candle import VolumeCandleSignalEngine
from src.domain.entities import Trade, OrderSide


@pytest.fixture
def vamp_engine():
    return VolumeCandleSignalEngine(volume_threshold=100.0, max_candles=5)


def test_vamp_impact_skew(vamp_engine):
    # Simulate buying pressure
    for i in range(10):
        vamp_engine.update_trade(
            Trade(
                id=f"buy_{i}",
                order_id=f"oid_{i}",
                symbol="BTC-USD",
                price=10000 + i,
                size=15.0,  # Total 150
                side=OrderSide.BUY,
                timestamp=time.time() + i,
            )
        )

    signals = vamp_engine.get_signals()
    assert signals["market_impact"] > 0.5
    # Since we exceeded 100 volume, at least one candle should be completed
    assert vamp_engine.pop_completed_candle() is not None


def test_ema_momentum_alignment(vamp_engine):
    # Manually set EMA values to force alignment
    vamp_engine.ema_values["fast"] = 10200
    vamp_engine.ema_values["med"] = 10100
    vamp_engine.ema_values["slow"] = 10000

    # Complete 3 candles to trigger calculation
    for i in range(3):
        vamp_engine.update_trade(
            Trade(
                id=f"t{i}",
                order_id=f"oid{i}",
                symbol="BTC-USD",
                price=10200,
                size=150.0,  # Threshold is 100
                side=OrderSide.BUY,
                timestamp=time.time() + i,
            )
        )

    signals = vamp_engine.get_signals()
    # combined_momentum = (ema_momentum * 0.4) + (vamp_impact * 0.6)
    # With fast > med > slow, ema_momentum = 1.0.
    # With one buy trade, vamp_impact should be positive.
    assert signals["momentum"] > 0.0


def test_vamp_reversal(vamp_engine):
    # 1. Strong BUY
    for i in range(10):
        vamp_engine.update_trade(
            Trade(
                id=f"buy_{i}",
                order_id=f"oid_b_{i}",
                symbol="BTC-USD",
                price=10000,
                size=20.0,
                side=OrderSide.BUY,
                timestamp=time.time() + i,
            )
        )
    assert vamp_engine.get_signals()["market_impact"] > 0.8

    # 2. Strong SELL (Reversal)
    for i in range(10):
        vamp_engine.update_trade(
            Trade(
                id=f"sell_{i}",
                order_id=f"oid_s_{i}",
                symbol="BTC-USD",
                price=10000,
                size=20.0,
                side=OrderSide.SELL,
                timestamp=time.time() + 20 + i,
            )
        )

    impact = vamp_engine.get_signals()["market_impact"]
    assert impact < 0.2  # Should have dropped significantly
