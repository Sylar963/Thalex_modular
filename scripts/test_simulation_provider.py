import asyncio
import os
import sys
import logging
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.adapters.storage.timescale_adapter import TimescaleDBAdapter
from src.adapters.storage.timescale_history_provider import TimescaleHistoryProvider
from src.domain.entities.history import HistoryConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main():
    # 0. Load Environment Variables
    load_dotenv()

    # 1. Setup DB Connection
    user = os.getenv("DATABASE_USER", "postgres")
    password = os.getenv("DATABASE_PASSWORD", "password")
    host = os.getenv("DATABASE_HOST", "localhost")
    port = os.getenv("DATABASE_PORT", "5432")
    dbname = os.getenv("DATABASE_NAME", "thalex_trading")

    dsn = f"postgresql://{user}:{password}@{host}:{port}/{dbname}"
    # Mask password in logs
    masked_dsn = f"postgresql://{user}:***@{host}:{port}/{dbname}"
    logger.info(f"Connecting to DB: {masked_dsn}")

    db = TimescaleDBAdapter(dsn)
    try:
        await db.connect()
    except Exception as e:
        logger.error(f"Failed to connect to DB: {e}")
        return

    # 2. Initialize Provider
    provider = TimescaleHistoryProvider(db)

    # 3. Define Config
    # Look back 24 hours to ensure we find some data
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(hours=24)

    # We suspect DataIngestion service is populating market_trades
    # Let's try querying what we know exists: "BTCUSDT" on "bybit"
    symbol = "BTCUSDT"
    venue = "bybit"

    config = HistoryConfig(
        symbol=symbol,
        venue=venue,
        start_time=start_time.timestamp(),
        end_time=end_time.timestamp(),
    )

    logger.info(
        f"Fetching tickers for {symbol} on {venue} from {start_time} to {end_time}..."
    )

    # 4. Fetch and Print Tickers
    count = 0
    try:
        async for ticker in provider.get_tickers(config):
            count += 1
            if count <= 5:
                print(f"[{count}] Ticker: {ticker}")

            if count >= 100:
                print("... (limit reached)")
                break
    except Exception as e:
        logger.error(f"Error fetching tickers: {e}")

    if count == 0:
        logger.warning(
            "No tickers found! Check if data ingestion is running and symbol/venue are correct."
        )
    else:
        logger.info(f"Successfully fetched {count}+ tickers.")

    # 5. Fetch and Print Trades
    logger.info(f"Fetching trades for {symbol} on {venue}...")
    count = 0
    try:
        async for trade in provider.get_trades(config):
            count += 1
            if count <= 5:
                print(f"[{count}] Trade: {trade}")

            if count >= 100:
                print("... (limit reached)")
                break
    except Exception as e:
        logger.error(f"Error fetching trades: {e}")

    if count == 0:
        logger.warning("No trades found!")
    else:
        logger.info(f"Successfully fetched {count}+ trades.")

    await db.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
