import asyncio
import logging
import sys
import os

# Ensure src module is importable
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

from src.services.data_ingestor import DataIngestor
from src.adapters.storage.timescale_adapter import TimescaleDBAdapter
from src.adapters.storage.bybit_history_adapter import BybitHistoryAdapter

# Configure logging
logging.basicConfig(level=logging.INFO)


async def verify_ingestion():
    # Credentials from env (already loaded or hardcoded for test)
    # Ideally reuse existing config loading or env vars

    db_user = os.getenv("DATABASE_USER", "postgres")
    db_pass = os.getenv("DATABASE_PASSWORD", "password")
    db_host = os.getenv("DATABASE_HOST", "localhost")
    db_port = os.getenv("DATABASE_PORT", "5432")
    db_name = os.getenv("DATABASE_NAME", "thalex_trading")
    dsn = f"postgresql://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}"

    print(f"Connecting to DB: {dsn}")
    db = TimescaleDBAdapter(dsn)
    await db.connect()

    ingestor = DataIngestor(db)

    symbol = "BTCUSDT"  # Bybit linear symbol
    print(f"Syncing trades for {symbol}...")

    try:
        await ingestor.sync_trades(symbol, limit=20)  # Small limit for test
    except Exception as e:
        print(f"Sync failed: {e}")

    # Verify count
    async with db.pool.acquire() as conn:
        count = await conn.fetchval("SELECT COUNT(*) FROM market_trades")
        print(f"Total rows in market_trades: {count}")

        rows = await conn.fetch(
            "SELECT * FROM market_trades ORDER BY time DESC LIMIT 5"
        )
        for r in rows:
            print(f"Saved Trade: {r['time']} {r['price']} {r['size']} {r['exchange']}")

    await ingestor.close()
    await db.disconnect()


if __name__ == "__main__":
    asyncio.run(verify_ingestion())
