import asyncio
import os
import sys

# Fix path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)

from src.adapters.storage.timescale_adapter import TimescaleDBAdapter


async def check_db_counts():
    # Setup
    db_user = os.getenv("DATABASE_USER", "postgres")
    db_pass = os.getenv("DATABASE_PASSWORD", "password")
    db_host = os.getenv("DATABASE_HOST", "localhost")
    db_port = os.getenv("DATABASE_PORT", "5433")
    db_name = os.getenv("DATABASE_NAME", "thalex_trading")
    dsn = f"postgresql://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}"

    adapter = TimescaleDBAdapter(dsn)
    await adapter.connect()

    tables = ["market_tickers", "market_trades", "market_candles", "market_signals"]

    print(f"Checking DB: {db_name} at {db_host}:{db_port}")

    for table in tables:
        try:
            async with adapter.pool.acquire() as conn:
                count = await conn.fetchval(f"SELECT COUNT(*) FROM {table}")
                print(f"Table '{table}': {count} rows")

                if count > 0:
                    # Show a sample
                    rows = await conn.fetch(
                        f"SELECT * FROM {table} ORDER BY time DESC LIMIT 1"
                    )
                    print(f"  Sample: {rows[0] if rows else 'None'}")

        except Exception as e:
            print(f"Error checking '{table}': {e}")

    await adapter.disconnect()


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    asyncio.run(check_db_counts())
