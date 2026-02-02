import unittest
import time
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.domain.market.regime_analyzer import MultiWindowRegimeAnalyzer, RegimeState
from src.domain.entities import Ticker


class TestMultiWindowRegimeAnalyzer(unittest.TestCase):
    def setUp(self):
        self.analyzer = MultiWindowRegimeAnalyzer()

    def _create_ticker(self, mid_price: float, spread_bps: float = 10.0) -> Ticker:
        half_spread = mid_price * (spread_bps / 2 / 10000)
        return Ticker(
            symbol="BTC-PERP",
            bid=mid_price - half_spread,
            ask=mid_price + half_spread,
            bid_size=1.0,
            ask_size=1.0,
            last=mid_price,
            volume=100.0,
            timestamp=time.time(),
        )

    def test_initial_regime_is_quiet(self):
        regime = self.analyzer.get_regime()
        self.assertEqual(regime["name"], "Quiet")

    def test_volatile_regime_detection(self):
        base_price = 100000.0
        for i in range(50):
            price = base_price + (i % 2) * 5000
            self.analyzer.update(self._create_ticker(price))

        regime = self.analyzer.get_regime()
        self.assertIn(regime["name"], ["Volatile", "Trending"])
        self.assertGreater(regime["rv_fast"], 0.0)

    def test_trending_regime_detection(self):
        base_price = 100000.0
        for i in range(120):
            price = base_price + (i * 200)
            self.analyzer.update(self._create_ticker(price))

        regime = self.analyzer.get_regime()
        self.assertIn(regime["name"], ["Trending", "Volatile"])
        self.assertGreater(abs(regime["trend_fast"]), 0.0)

    def test_option_data_integration(self):
        self.analyzer.set_option_data(em_pct=0.08, atm_iv=0.65)
        self.analyzer.update(self._create_ticker(100000.0))

        regime = self.analyzer.get_regime()
        self.assertEqual(regime["expected_move_pct"], 0.08)
        self.assertEqual(regime["atm_iv"], 0.65)

    def test_overpriced_vol_detection(self):
        base_price = 100000.0
        for i in range(30):
            self.analyzer.update(self._create_ticker(base_price))

        self.analyzer.set_option_data(em_pct=0.50, atm_iv=0.70)
        self.analyzer.update(self._create_ticker(base_price))

        regime = self.analyzer.get_regime()
        self.assertEqual(regime["expected_move_pct"], 0.50)
        self.assertEqual(regime["atm_iv"], 0.70)


if __name__ == "__main__":
    unittest.main()
