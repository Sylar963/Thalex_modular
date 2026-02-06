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
    data_manager: Optional["DataStreamManager"] = None


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

    # --- Data Engine Initialization ---
    try:
        from src.infrastructure.config_factory import ConfigFactory
        from src.use_cases.data_stream_manager import DataStreamManager

        # Load Config
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
        config_path = os.path.join(project_root, "config.json")
        bot_config = ConfigFactory.load_config(config_path)

        # Create Adapters in Monitor Mode (force enabled=True for data, strategy=None)
        exchange_configs = ConfigFactory.create_exchange_configs(
            bot_config, force_monitor_mode=True
        )

        if exchange_configs:
            print(
                f"Initializing Data Stream Manager for {len(exchange_configs)} venues..."
            )
            _state.data_manager = DataStreamManager(exchange_configs, _state.db_adapter)
            # Start in background task? The lifespan handler in main.py awaits this.
            # But await start() blocks until everything is connected. That's fine.
            # However, DataStreamManager.start() currently just launches tasks and returns.
            # Make sure it doesn't block forever.
            await _state.data_manager.start()
        else:
            print("No valid exchange configurations found for Data Engine.")

    except Exception as e:
        print(f"Failed to start Data Engine: {e}")
        # We process without it if it fails, or should we crash?
        # For now, print error but allow API to start (dashboard might just be empty)


async def close_dependencies():
    """Cleanup global dependencies."""
    if _state.data_manager:
        await _state.data_manager.stop()
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
