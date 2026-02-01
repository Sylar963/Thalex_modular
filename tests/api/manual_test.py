import unittest
import asyncio
from src.api.v1.endpoints.market import get_instruments, get_history, get_recent_ticker
from src.api.v1.endpoints.portfolio import (
    get_summary,
    get_positions,
    get_history as get_portfolio_history,
)
from src.api.v1.endpoints.simulation import get_runs, start_simulation, SimulationConfig
from src.api.v1.endpoints.config import get_config, update_config
from src.api.repositories import (
    MarketRepository,
    PortfolioRepository,
    SimulationRepository,
    ConfigRepository,
)


# Mock Repositories
class MockMarketRepo(MarketRepository):
    def __init__(self):
        super().__init__(None)

    async def get_instruments(self):
        return [{"symbol": "BTC-PERPETUAL", "type": "future"}]

    async def get_history(self, *args, **kwargs):
        return [{"time": 1000, "close": 50000.0}]

    async def get_recent_tickers(self, *args, **kwargs):
        from src.domain.entities import Ticker

        return [Ticker("BTC-PERP", 99.0, 101.0, 1.0, 1.0, 100.0, 10.0)]


class MockPortfolioRepo(PortfolioRepository):
    def __init__(self):
        super().__init__(None)

    async def get_summary(self):
        return {"equity": 1000.0}

    async def get_positions(self):
        return [{"symbol": "BTC", "size": 1.0}]

    async def get_history(self):
        return [{"pnl": 10.0}]


class MockSimulationRepo(SimulationRepository):
    def __init__(self):
        super().__init__(None)

    async def get_runs(self):
        return [{"id": "1"}]

    async def start_simulation(self, params):
        return {"status": "started"}


class MockConfigRepo(ConfigRepository):
    def __init__(self):
        super().__init__(None)
        self._config = {"foo": "bar"}

    async def get_config(self):
        return self._config

    async def update_config(self, updates):
        self._config.update(updates)
        return self._config


class TestEndpointsDirectly(unittest.TestCase):
    def setUp(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

    def tearDown(self):
        self.loop.close()

    def test_market_endpoints(self):
        repo = MockMarketRepo()
        inst = self.loop.run_until_complete(get_instruments(repo))
        self.assertEqual(inst[0]["symbol"], "BTC-PERPETUAL")

        hist = self.loop.run_until_complete(get_history("BTC", 0, 0, "1m", repo))
        self.assertEqual(hist[0]["close"], 50000.0)

        tick = self.loop.run_until_complete(get_recent_ticker("BTC", 1, repo))
        self.assertEqual(tick[0]["symbol"], "BTC-PERP")

    def test_portfolio_endpoints(self):
        repo = MockPortfolioRepo()
        summ = self.loop.run_until_complete(get_summary(repo))
        self.assertEqual(summ["equity"], 1000.0)

        pos = self.loop.run_until_complete(get_positions(repo))
        self.assertEqual(pos[0]["symbol"], "BTC")

    def test_simulation_endpoints(self):
        repo = MockSimulationRepo()
        runs = self.loop.run_until_complete(get_runs(repo))
        self.assertEqual(runs[0]["id"], "1")

        cfg = SimulationConfig(start_date="2023", end_date="2024", strategy_config={})
        res = self.loop.run_until_complete(start_simulation(cfg, repo))
        self.assertEqual(res["status"], "started")

    def test_config_endpoints(self):
        repo = MockConfigRepo()
        val = self.loop.run_until_complete(get_config(repo))
        self.assertEqual(val["foo"], "bar")

        updated = self.loop.run_until_complete(update_config({"foo": "baz"}, repo))
        self.assertEqual(updated["foo"], "baz")


if __name__ == "__main__":
    unittest.main()
