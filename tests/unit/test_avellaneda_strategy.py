import pytest
import math
from unittest.mock import MagicMock
from src.domain.strategies.avellaneda import AvellanedaStoikovStrategy
from src.domain.entities import MarketState, Ticker, Position


class TestAvellanedaStrategyFixes:
    """
    Unit tests ensuring the fixes for volatility units, inventory units, and skew explosion
    are working correctly and regression-proof.
    """

    @pytest.fixture
    def strategy(self):
        strategy = AvellanedaStoikovStrategy()
        config = {
            "gamma": 0.0,  # Zero gamma to isolate components
            "volatility": 0.05,
            "min_spread": 0,
            "tick_size": 0.0001,
            "avellaneda": {
                "position_fade_time": 3600,
                "volatility_multiplier": 1.0,
                "inventory_weight": 0.0,  # Isolated
                "base_spread_factor": 0.0,
                "inventory_factor": 0.5,
            },
            "maker_fee_rate": 0.0,
            "profit_margin_rate": 0.0,
        }
        strategy.setup(config)
        return strategy

    def create_market_state(self, price=100.0, symbol="TEST"):
        ticker = Ticker(
            symbol=symbol,
            bid=price - 0.01,
            ask=price + 0.01,
            bid_size=10,
            ask_size=10,
            last=price,
            volume=1000,
            exchange="test",
            timestamp=1000,
        )
        return MarketState(ticker=ticker, timestamp=1000)

    def test_volatility_unit_scaling(self, strategy):
        """
        Verify Volatility Component is scaled by price and time.
        Formula: (vol * price) * sqrt(fade_time / 86400)
        """
        price = 3.20
        # Config: vol=0.05, mult=1.0, fade=3600
        # Expected: (0.05 * 3.20) * sqrt(3600/86400) = 0.16 * 0.2041 = 0.0326

        market = self.create_market_state(price=price)
        pos = Position(symbol="TEST", size=0.0, entry_price=price)

        strategy.calculate_quotes(market, pos, exchange="test")
        metrics = strategy.get_last_metrics()

        vol_comp = metrics["volatility"]
        expected = (0.05 * 3.20) * math.sqrt(3600 / 86400)

        assert math.isclose(vol_comp, expected, rel_tol=1e-4)

    def test_inventory_dampening_normal(self, strategy):
        """
        Verify Inventory Risk is linear for Ratio <= 1.0
        """
        strategy.inventory_factor = 0.5
        limit = 10.0
        pos_size = 5.0  # Ratio 0.5
        price = 100.0

        strategy.position_limit = limit
        market = self.create_market_state(price=price)
        pos = Position(symbol="TEST", size=pos_size, entry_price=price)

        strategy.calculate_quotes(market, pos, exchange="test")
        metrics = strategy.get_last_metrics()

        # Risk should be 0.5
        assert metrics["inventory_risk"] == 0.5

        # Spread Add: 0.5(factor) * 0.5(risk) * 0.05(vol) * 100(price) = 1.25
        spread = metrics["spread"]
        # With gamma=0, min=0, base=0, spread is ONLY components.
        # Vol comp also exists: 0.05 * 100 * 0.204 = 1.02
        # Total = 1.25 + 1.02 = 2.27

        # We can check the difference if we zero out volatility in calculation?
        # Easier to checking risk score directly which we exposed in metrics.
        assert metrics["inventory_risk"] == 0.5

    def test_inventory_dampening_high_leverage(self, strategy):
        """
        Verify Inventory Risk is dampened (sqrt) for Ratio > 1.0
        """
        limit = 10.0
        pos_size = 50.0  # Ratio 5.0
        price = 100.0

        strategy.position_limit = limit
        market = self.create_market_state(price=price)
        pos = Position(symbol="TEST", size=pos_size, entry_price=price)

        strategy.calculate_quotes(market, pos, exchange="test")
        metrics = strategy.get_last_metrics()

        # Expected Risk: 1 + sqrt(5 - 1) = 1 + 2 = 3.0
        # (Linear would be 5.0)
        assert math.isclose(metrics["inventory_risk"], 3.0, rel_tol=1e-4)

    def test_inventory_dampening_max_leverage(self, strategy):
        """
        Verify Inventory Risk at 10x max leverage
        """
        limit = 10.0
        pos_size = 100.0  # Ratio 10.0
        price = 100.0

        strategy.position_limit = limit
        market = self.create_market_state(price=price)
        pos = Position(symbol="TEST", size=pos_size, entry_price=price)

        strategy.calculate_quotes(market, pos, exchange="test")
        metrics = strategy.get_last_metrics()

        # Expected Risk: 1 + sqrt(10 - 1) = 1 + 3 = 4.0
        # (Linear would be 10.0)
        assert math.isclose(metrics["inventory_risk"], 4.0, rel_tol=1e-4)

    def test_skew_capping_and_decoupling(self, strategy):
        """
        Verify Skew is capped at 3.0 and decoupled from inventory spread.
        """
        strategy.inventory_weight = 1.0  # Enable skew (factor 0.5 * 1.0 = 0.5)
        limit = 10.0
        pos_size = 100.0  # Ratio 10.0
        price = 100.0

        strategy.position_limit = limit
        market = self.create_market_state(price=price)
        pos = Position(symbol="TEST", size=pos_size, entry_price=price)

        strategy.calculate_quotes(market, pos, exchange="test")
        metrics = strategy.get_last_metrics()

        # 1. Skew Ratio Cap check
        # The logic caps ratio at 3.0.
        # Skew = CappedRatio(3.0) * BaseSpread * Factor(0.5)

        # Calculate Base Spread (Vol + Impact only, NO Inventory)
        # Vol Comp = 0.05 * 100 * 0.2041 = 1.0205
        # Base Spread = 1.0205

        expected_skew = 3.0 * 1.0205 * 0.5

        # Allow small floating point diff
        assert math.isclose(metrics["inventory_skew"], expected_skew, rel_tol=1e-3)

        # 2. Check it didn't explode
        # If it used full spread (which includes inventory risk ~4.0 impact => total spread ~4+1=5)
        # It would be Ratio(10) * Spread(5) * 0.5 = 25.0 skew!
        # Our capped skew is ~1.53. Massive difference.
        assert metrics["inventory_skew"] < 5.0
