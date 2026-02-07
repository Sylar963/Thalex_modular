import asyncio
import os
import sys
import logging
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.adapters.storage.timescale_adapter import TimescaleDBAdapter
from src.api.repositories.simulation_repository import SimulationRepository

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def run_integration_test():
    # 0. Load Environment Variables
    load_dotenv()

    # 1. Setup DB Connection
    user = os.getenv("DATABASE_USER", "postgres")
    password = os.getenv("DATABASE_PASSWORD", "password")
    host = os.getenv("DATABASE_HOST", "localhost")
    port = os.getenv("DATABASE_PORT", "5432")
    dbname = os.getenv("DATABASE_NAME", "thalex_trading")

    dsn = f"postgresql://{user}:{password}@{host}:{port}/{dbname}"
    masked_dsn = f"postgresql://{user}:***@{host}:{port}/{dbname}"
    logger.info(f"Connecting to DB: {masked_dsn}")

    db = TimescaleDBAdapter(dsn)
    try:
        await db.connect()
    except Exception as e:
        logger.error(f"Failed to connect to DB: {e}")
        return

    # 2. Initialize Repository
    repo = SimulationRepository(db)

    # 3. Define Simulation Params
    # 4 Hour simulation on existing data
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(hours=4)

    symbol = "BTCUSDT"
    venue = "bybit"

    sim_params = {
        "symbol": symbol,
        "venue": venue,
        "start_date": start_time.timestamp(),
        "end_date": end_time.timestamp(),
        "initial_balance": 10000.0,
        "strategy_config": {
            "gamma": 0.5,
            "volatility": 0.05,
            "position_limit": 1.0,  # 1 BTC max
            "min_spread": 20,
        },
        "risk_config": {"max_drawdown": 0.1},
    }

    logger.info(
        f"Starting Simulation for {symbol} ({venue}) from {start_time.isoformat()} to {end_time.isoformat()}"
    )

    # 4. Run Simulation
    try:
        result = await repo.start_simulation(sim_params)

        logger.info("Simulation Completed!")
        logger.info(f"Run ID: {result['run_id']}")
        logger.info(f"Status: {result['status']}")

        stats = result.get("stats", {})
        logger.info("-" * 40)
        logger.info(f"Total PnL: {stats.get('total_pnl', 0):.2f}")
        logger.info(f"Total Trades: {stats.get('total_trades', 0)}")
        logger.info(f"Sharpe Ratio: {stats.get('sharpe_ratio', 0):.2f}")
        logger.info(f"Max Drawdown: {stats.get('max_drawdown', 0):.4f}")
        logger.info("-" * 40)

        if stats.get("total_trades", 0) == 0:
            logger.warning(
                "No trades executed. This might be due to missing TICKER data (BBO) needed for the strategy."
            )

    except Exception as e:
        logger.error(f"Simulation Failed: {e}", exc_info=True)
    finally:
        await db.disconnect()


if __name__ == "__main__":
    asyncio.run(run_integration_test())
