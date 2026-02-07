import os
from typing import Optional
from dotenv import load_dotenv
from .repositories import (
    MarketRepository,
    PortfolioRepository,
    SimulationRepository,
    ConfigRepository,
)
from ..adapters.storage.timescale_adapter import TimescaleDBAdapter
from src.services.data_ingestor import DataIngestionService
from src.domain.signals.open_range import OpenRangeSignalEngine


# Singleton instance holder
class GlobalState:
    db_adapter: Optional[TimescaleDBAdapter] = None
    data_ingestor: Optional[DataIngestionService] = None
    or_engine: Optional[OpenRangeSignalEngine] = None


_state = GlobalState()


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

    # --- Signal Engines ---
    try:
        import json

        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "config.json"
        )
        with open(config_path, "r") as f:
            config = json.load(f)

        or_config = config.get("signals", {}).get("open_range", {})

        _state.or_engine = OpenRangeSignalEngine(
            session_start_utc=or_config.get("session_start_utc", "20:00"),
            session_end_utc=or_config.get("session_end_utc", "20:15"),
            target_pct_from_mid=or_config.get("target_pct_from_mid", 1.49),
            subsequent_target_pct_of_range=or_config.get(
                "subsequent_target_pct_of_range", 220
            ),
            timezone=or_config.get("timezone", "UTC"),
        )
        print(
            f"Initialized OpenRangeEngine with session: {or_config.get('session_start_utc')} - {or_config.get('session_end_utc')} {or_config.get('timezone')}"
        )
    except Exception as e:
        print(f"Failed to init Signal Engines: {e}")
        # Fallback to defaults
        _state.or_engine = OpenRangeSignalEngine()

    # --- Data Ingestion Service Initialization ---
    try:
        _state.data_ingestor = DataIngestionService(db_dsn, _state.or_engine)
        await _state.data_ingestor.start()
        print("Data Ingestion Service started.")

    except Exception as e:
        print(f"Failed to start Data Ingestion Service: {e}")


async def close_dependencies():
    """Cleanup global dependencies."""
    if _state.data_ingestor:
        await _state.data_ingestor.stop()
    if _state.db_adapter:
        await _state.db_adapter.disconnect()


# Dependency Providers


def get_db_adapter() -> Optional[TimescaleDBAdapter]:
    return _state.db_adapter


def get_market_repo() -> MarketRepository:
    return MarketRepository(get_db_adapter(), _state.or_engine)


def get_portfolio_repo() -> PortfolioRepository:
    return PortfolioRepository(get_db_adapter())


def get_simulation_repo() -> SimulationRepository:
    return SimulationRepository(get_db_adapter())


def get_config_repo() -> ConfigRepository:
    return ConfigRepository(get_db_adapter())
