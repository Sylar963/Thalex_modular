import unittest
from src.domain.risk.basic_manager import BasicRiskManager
from src.domain.entities import Order, Position, OrderSide, OrderType


class TestRiskRobustness(unittest.TestCase):
    def setUp(self):
        self.risk = BasicRiskManager(max_position=0.01, max_order_size=0.1)
        self.position = Position("BTC-PERP", 0.0, 10000.0)

    def test_basic_validation(self):
        # Allow small order
        order = Order("1", "BTC-PERP", OrderSide.BUY, 9000, 0.001)
        self.assertTrue(self.risk.validate_order(order, self.position))

        # Reject large order
        order_large = Order("2", "BTC-PERP", OrderSide.BUY, 9000, 0.1)
        self.assertFalse(self.risk.validate_order(order_large, self.position))

    def test_projected_exposure(self):
        # Position 0.005
        # Active Buy 0.004
        # New Buy 0.002 -> Total 0.011 > 0.01 -> Reject

        pos = Position("BTC-PERP", 0.005, 10000.0)
        active = [Order("a1", "BTC-PERP", OrderSide.BUY, 9000, 0.004)]

        new_order = Order("n1", "BTC-PERP", OrderSide.BUY, 9000, 0.002)

        # Should be rejected
        self.assertFalse(self.risk.validate_order(new_order, pos, active_orders=active))

        # Valid case:
        # Active Sell 0.004 (Reduces exposure if we treat them as net? No, distinct sides)
        # Risk Manager Logic:
        # If checking BUY: Pos + Active Buys + New Buy
        # If checking SELL: Pos - Active Sells - New Sell (abs)

        # Lets check SELL side
        # Pos 0.005 Long.
        # Active Sell 0.010.
        # New Sell 0.002.
        # Net projected: 0.005 - 0.010 - 0.002 = -0.007. Abs(0.007) < 0.01. Safe.

        active_sell = [Order("s1", "BTC-PERP", OrderSide.SELL, 11000, 0.010)]
        new_sell = Order("n2", "BTC-PERP", OrderSide.SELL, 11000, 0.002)
        self.assertTrue(
            self.risk.validate_order(new_sell, pos, active_orders=active_sell)
        )

    def test_active_orders_accumulation(self):
        # Ensure it sums up ALL active orders of same side
        active = [
            Order("a1", "BTC-PERP", OrderSide.BUY, 9000, 0.002),
            Order("a2", "BTC-PERP", OrderSide.BUY, 8900, 0.003),
        ]
        # Total Active Buy: 0.005
        # Current Pos: 0.004
        # New Order: 0.002
        # Total: 0.011 > 0.01 -> Fail

        pos = Position("BTC-PERP", 0.004, 10000.0)
        new_order = Order("n1", "BTC-PERP", OrderSide.BUY, 9000, 0.002)

        self.assertFalse(self.risk.validate_order(new_order, pos, active_orders=active))


if __name__ == "__main__":
    unittest.main()
