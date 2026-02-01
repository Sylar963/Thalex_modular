import unittest
import time
from src.domain.entities import MarketState, Position, Ticker, Order
from src.domain.strategies.avellaneda import AvellanedaStoikovStrategy


class TestAvellaneda(unittest.TestCase):
    def test_calculation_jit_warmup(self):
        """Test calculation produces orders and warms up JIT."""
        strategy = AvellanedaStoikovStrategy()
        strategy.setup({})

        market = MarketState(
            ticker=Ticker("BTC-PERP", 9999.0, 10001.0, 1.0, 1.0, 10000.0, 1000.0),
            signals={},
        )
        position = Position("BTC-PERP", 0.0, 0.0)

        # First pass (compiles kernel)
        start = time.time()
        orders = strategy.calculate_quotes(market, position)
        end = time.time()
        print(f"Checking JIT compilation lag: {(end - start) * 1000:.2f}ms")

        self.assertEqual(len(orders), 4)
        bid = orders[0]
        ask = orders[1]

        self.assertTrue(bid.price < ask.price)
        self.assertTrue(
            bid.side == "buy" or bid.side == OrderSide.BUY
        )  # Handle str/enum

        # Second pass (fast)
        start = time.time()
        orders2 = strategy.calculate_quotes(market, position)
        end = time.time()
        print(f"Checking JIT execution speed: {(end - start) * 1000:.2f}ms")

        # Verify result consistency
        self.assertEqual(orders[0].price, orders2[0].price)


if __name__ == "__main__":
    unittest.main()
