import pytest
from src.domain.risk.basic_manager import BasicRiskManager
from src.domain.entities import Position


class TestBreachReset:
    def test_breach_flag_auto_resets_when_position_reduced(self):
        rm = BasicRiskManager(max_position=10.0, max_order_size=1.0)
        rm.setup({"max_position": 5.0, "venue_limits": {"bybit": 5.0}})

        pos_breach = Position(
            symbol="BTCUSDT", size=10.0, entry_price=50000.0, exchange="bybit"
        )
        rm.update_position(pos_breach)
        assert rm.has_breached() is True

        pos_within = Position(
            symbol="BTCUSDT", size=3.0, entry_price=50000.0, exchange="bybit"
        )
        rm.update_position(pos_within)
        assert rm.has_breached() is False

    def test_breach_flag_persists_with_multiple_venues_if_one_still_breached(self):
        rm = BasicRiskManager(max_position=10.0, max_order_size=1.0)
        rm.setup({"venue_limits": {"bybit": 5.0, "thalex": 5.0}})

        bybit_breach = Position(
            symbol="BTCUSDT", size=10.0, entry_price=50000.0, exchange="bybit"
        )
        rm.update_position(bybit_breach)
        assert rm.has_breached() is True

        thalex_breach = Position(
            symbol="BTC-PERPETUAL", size=8.0, entry_price=50000.0, exchange="thalex"
        )
        rm.update_position(thalex_breach)
        assert rm.has_breached() is True

        bybit_fixed = Position(
            symbol="BTCUSDT", size=2.0, entry_price=50000.0, exchange="bybit"
        )
        rm.update_position(bybit_fixed)
        assert rm.has_breached() is True

        thalex_fixed = Position(
            symbol="BTC-PERPETUAL", size=3.0, entry_price=50000.0, exchange="thalex"
        )
        rm.update_position(thalex_fixed)
        assert rm.has_breached() is False

    def test_breach_flag_not_set_if_position_within_limits(self):
        rm = BasicRiskManager(max_position=10.0, max_order_size=1.0)
        rm.setup({"venue_limits": {"bybit": 5.0}})

        pos = Position(
            symbol="BTCUSDT", size=3.0, entry_price=50000.0, exchange="bybit"
        )
        rm.update_position(pos)
        assert rm.has_breached() is False
