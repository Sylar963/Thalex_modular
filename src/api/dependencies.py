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


from dotenv import load_dotenv


async def init_dependencies():
    """Initialize global dependencies (DB connection, etc)."""
    # Ensure environment variables are loaded
    load_dotenv()

    db_user = os.getenv("DATABASE_USER", "postgres")
    db_pass = os.getenv("DATABASE_PASSWORD", "password")
    db_host = os.getenv("DATABASE_HOST", "localhost")
    db_name = os.getenv("DATABASE_NAME", "thalex_trading")

    # Robust Port Detection
    # 1. Check DATABASE_PORT (preferred)
    # 2. Check DB_PORT (fallback)
    # 3. Default to 5433 (Thalex default) with warning
    db_port = os.getenv("DATABASE_PORT")
    if not db_port:
        db_port = os.getenv("DB_PORT")

    if not db_port:
        print("WARNING: Neither DATABASE_PORT nor DB_PORT set. Defaulting to 5433.")
        db_port = "5433"

    # Log configuration for verification (masking password)
    print(f"Initializing DB Connection: {db_user}@{db_host}:{db_port}/{db_name}")

    db_dsn = f"postgresql://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}"

    try:
        _state.db_adapter = TimescaleDBAdapter(db_dsn)
        await _state.db_adapter.connect()
    except Exception as e:
        print(
            f"CRITICAL: Failed to connect to database at {db_host}:{db_port}. Error: {e}"
        )
        raise e


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
