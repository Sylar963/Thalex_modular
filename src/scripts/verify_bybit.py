import asyncio
import logging
import os
import sys
from datetime import datetime, timedelta, timezone

# Add src to path
sys.path.append(os.getcwd())

from src.adapters.storage.timescale_adapter import TimescaleDBAdapter


async def verify_bybit_mapping():
    # Setup connection
    host = os.getenv("DATABASE_HOST", "localhost")
    port = os.getenv("DATABASE_PORT", "5433")
    user = os.getenv("DATABASE_USER", "postgres")
    password = os.getenv("DATABASE_PASSWORD", "password")
    dbname = os.getenv("DATABASE_NAME", "thalex_trading")

    dsn = f"postgres://{user}:{password}@{host}:{port}/{dbname}"
    adapter = TimescaleDBAdapter(dsn)
    await adapter.connect()

    end = datetime.now(timezone.utc).timestamp()
    start = end - 86400  # 24 hours

    print(f"\nFetching history for Bybit...")

    # Test: Requesting BTC-PERPETUAL for BYBIT
    # This should internally map to BTCUSDT (trades) and BTC-PERPETUAL (historical tickers)
    candles = await adapter.get_history("BTC-PERPETUAL", start, end, "1m", "bybit")

    print(f"--- Bybit Verification ---")
    print(f"Returned {len(candles)} candles.")

    if candles:
        print(f"First: {candles[0]}")
        print(f"Last:  {candles[-1]}")

    await adapter.close()


if __name__ == "__main__":
    asyncio.run(verify_bybit_mapping())
