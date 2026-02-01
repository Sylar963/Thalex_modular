import os
from typing import Optional
from fastapi import Request
from .repositories import (
    MarketRepository,
    PortfolioRepository,
    SimulationRepository,
    ConfigRepository,
)
from ..adapters.storage.timescale_adapter import TimescaleDBAdapter


# Singleton instance holder
class GlobalState:
    db_adapter: Optional[TimescaleDBAdapter] = None


_state = GlobalState()


async def init_dependencies():
    """Initialize global dependencies (DB connection, etc)."""
    db_user = os.getenv("DATABASE_USER", "postgres")
    db_pass = os.getenv("DATABASE_PASSWORD", "password")
    db_host = os.getenv("DATABASE_HOST", "localhost")
    db_port = os.getenv("DATABASE_PORT", "5432")
    db_name = os.getenv("DATABASE_NAME", "thalex_trading")
    db_dsn = f"postgresql://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}"

    _state.db_adapter = TimescaleDBAdapter(db_dsn)
    await _state.db_adapter.connect()


async def close_dependencies():
    """Cleanup global dependencies."""
    if _state.db_adapter:
        await _state.db_adapter.disconnect()


# Dependency Providers


def get_db_adapter() -> Optional[TimescaleDBAdapter]:
    return _state.db_adapter


def get_market_repo() -> MarketRepository:
    return MarketRepository(get_db_adapter())


def get_portfolio_repo() -> PortfolioRepository:
    return PortfolioRepository(get_db_adapter())


def get_simulation_repo() -> SimulationRepository:
    return SimulationRepository(get_db_adapter())


def get_config_repo() -> ConfigRepository:
    return ConfigRepository(get_db_adapter())
