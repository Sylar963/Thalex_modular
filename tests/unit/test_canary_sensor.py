import pytest
import time
from src.domain.sensors.canary_sensor import CanarySensor
from src.domain.entities import Ticker, Trade, OrderSide


class TestCanarySensor:
    def test_initial_state_has_zero_toxicity(self):
        sensor = CanarySensor(window_ms=5000)
        signals = sensor.get_signals()

        assert signals["toxicity_score"] == 0.0
        assert signals["pull_rate"] == 0.0
        assert signals["quote_stability"] == 1.0
        assert signals["size_asymmetry"] == 0.0

    def test_stable_book_produces_low_toxicity(self):
        sensor = CanarySensor(window_ms=5000)
        now = time.time()

        for i in range(20):
            ticker = Ticker(
                symbol="BTCUSDT",
                bid=100000.0,
                ask=100001.0,
                bid_size=10.0,
                ask_size=10.0,
                last=100000.5,
                volume=1000.0,
                timestamp=now + i * 0.1,
            )
            sensor.update(ticker)

        signals = sensor.get_signals()
        assert signals["toxicity_score"] < 0.1
        assert signals["quote_stability"] > 0.8

    def test_pull_detection_increases_toxicity(self):
        sensor = CanarySensor(window_ms=5000)
        now = time.time()

        for i in range(10):
            ticker = Ticker(
                symbol="BTCUSDT",
                bid=100000.0,
                ask=100001.0,
                bid_size=50.0,
                ask_size=50.0,
                last=100000.5,
                volume=1000.0,
                timestamp=now + i * 0.1,
            )
            sensor.update(ticker)

        for i in range(10):
            ticker = Ticker(
                symbol="BTCUSDT",
                bid=100000.0,
                ask=100001.0,
                bid_size=5.0,
                ask_size=5.0,
                last=100000.5,
                volume=1000.0,
                timestamp=now + 1.0 + i * 0.1,
            )
            sensor.update(ticker)

        signals = sensor.get_signals()
        assert signals["pull_rate"] > 0.01

    def test_trade_updates_size_tracking(self):
        sensor = CanarySensor(window_ms=5000)
        now = time.time()

        for i in range(10):
            ticker = Ticker(
                symbol="BTCUSDT",
                bid=100000.0,
                ask=100001.0,
                bid_size=100.0,
                ask_size=100.0,
                last=100000.5,
                volume=1000.0,
                timestamp=now + i * 0.1,
            )
            sensor.update(ticker)

        trade = Trade(
            id="t1",
            order_id="o1",
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            price=100000.5,
            size=1.0,
            timestamp=now + 1.0,
        )
        sensor.update_trade(trade)

        signals = sensor.get_signals()
        assert isinstance(signals["size_asymmetry"], float)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
