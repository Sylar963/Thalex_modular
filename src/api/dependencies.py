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
    market_feed: Optional["MarketFeedService"] = None


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
    db_port = os.getenv("DATABASE_PORT", "5433")

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

    # --- Market Feed Service Initialization ---
    try:
        from src.services.market_feed import MarketFeedService

        _state.market_feed = MarketFeedService(db_dsn)
        await _state.market_feed.start()
        print("Market Feed Service started.")

    except Exception as e:
        print(f"Failed to start Market Feed Service: {e}")


async def close_dependencies():
    """Cleanup global dependencies."""
    if _state.market_feed:
        await _state.market_feed.stop()
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
