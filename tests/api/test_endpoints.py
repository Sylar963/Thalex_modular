from fastapi.testclient import TestClient
from src.api.main import app
from src.api.dependencies import (
    get_market_repo,
    get_portfolio_repo,
    get_simulation_repo,
    get_config_repo,
)
from src.api.repositories import (
    MarketRepository,
    PortfolioRepository,
    SimulationRepository,
    ConfigRepository,
)
import pytest

client = TestClient(app)


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

    async def get_config(self):
        return {"foo": "bar"}


# Override Dependencies
app.dependency_overrides[get_market_repo] = lambda: MockMarketRepo()
app.dependency_overrides[get_portfolio_repo] = lambda: MockPortfolioRepo()
app.dependency_overrides[get_simulation_repo] = lambda: MockSimulationRepo()
app.dependency_overrides[get_config_repo] = lambda: MockConfigRepo()


def test_health():
    response = client.get("/health")
    assert response.status_code == 200


def test_market_instruments():
    response = client.get("/api/v1/market/instruments")
    assert response.status_code == 200
    assert response.json()[0]["symbol"] == "BTC-PERPETUAL"


def test_market_ticker():
    response = client.get("/api/v1/market/ticker/BTC-PERP")
    assert response.status_code == 200
    assert response.json()[0]["symbol"] == "BTC-PERP"


def test_portfolio_summary():
    response = client.get("/api/v1/portfolio/summary")
    assert response.status_code == 200
    assert response.json()["equity"] == 1000.0


def test_simulation_runs():
    response = client.get("/api/v1/simulation/runs")
    assert response.status_code == 200
    assert len(response.json()) > 0


def test_config_get():
    response = client.get("/api/v1/config")
    assert response.status_code == 200
    assert response.json()["foo"] == "bar"
