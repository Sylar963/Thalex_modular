import unittest
from unittest.mock import MagicMock, AsyncMock
import asyncio
from src.use_cases.simulation_engine import SimulationEngine
from src.domain.strategies.avellaneda import AvellanedaStoikovStrategy
from src.domain.risk.basic_manager import BasicRiskManager
from src.domain.entities import Order, OrderSide


class TestSimulationEngine(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.strategy = AvellanedaStoikovStrategy()
        self.strategy.setup({"quote_levels": 1, "order_size": 1.0, "min_spread": 10.0})

        self.risk = BasicRiskManager(max_position=10.0, max_order_size=10.0)
        self.storage = MagicMock()

        self.engine = SimulationEngine(
            strategy=self.strategy,
            risk_manager=self.risk,
            data_provider=self.storage,
            initial_balance=1000.0,
            maker_fee=0.0,
        )

    async def test_simple_fill_loop(self):
        self.storage.get_history = AsyncMock(
            return_value=[
                {
                    "time": 100,
                    "close": 1000.0,
                    "low": 999.0,
                    "high": 1001.0,
                    "volume": 10.0,
                },
                {
                    "time": 200,
                    "close": 990.0,
                    "low": 990.0,
                    "high": 995.0,
                    "volume": 10.0,
                },
                {
                    "time": 300,
                    "close": 1010.0,
                    "low": 1005.0,
                    "high": 1010.0,
                    "volume": 10.0,
                },
            ]
        )

        result = await self.engine.run_simulation("BTC-PERP", 0, 400)

        # Verify Fills are processed
        self.assertGreaterEqual(len(result.fills), 2)

        # Verify Math Consistency
        # Equity must equal Initial + Sum of Trade PNLs
        expected_equity = 1000.0 + result.stats.total_pnl
        self.assertAlmostEqual(
            result.equity_curve[-1].equity, expected_equity, delta=0.01
        )

        # Check Stats mapping
        self.assertEqual(result.stats.total_trades, len(result.fills))


if __name__ == "__main__":
    unittest.main()
