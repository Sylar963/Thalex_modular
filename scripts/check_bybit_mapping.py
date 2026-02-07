# Imports should work natively now via pip install -e .
import asyncio
import logging
import os

from src.adapters.storage.timescale_adapter import TimescaleDBAdapter
from datetime import datetime, timedelta, timezone


async def test_bybit_history():
    # Setup connection
    host = os.getenv("DATABASE_HOST", "localhost")
    port = os.getenv("DATABASE_PORT", "5433")  # Using 5433 as verified
    user = os.getenv("DATABASE_USER", "postgres")
    password = os.getenv("DATABASE_PASSWORD", "password")
    dbname = os.getenv("DATABASE_NAME", "thalex_trading")

    dsn = f"postgres://{user}:{password}@{host}:{port}/{dbname}"
    adapter = TimescaleDBAdapter(dsn)
    await adapter.connect()

    end = datetime.now(timezone.utc).timestamp()
    start = end - 3600 * 4  # 4 hours

    print("\n--- Testing Bybit History Fetch ---")

    # Test 1: BTC-PERPETUAL (Thalex Symbol) requesting Bybit
    print(f"\n1. Requesting 'BTC-PERPETUAL' with venue='bybit'...")
    candles_1 = await adapter.get_history("BTC-PERPETUAL", start, end, "5m", "bybit")
    print(f"Result count: {len(candles_1)}")

    # Test 2: BTCUSDT (Bybit Symbol) requesting Bybit
    print(f"\n2. Requesting 'BTCUSDT' with venue='bybit'...")
    candles_2 = await adapter.get_history("BTCUSDT", start, end, "5m", "bybit")
    print(f"Result count: {len(candles_2)}")

    await adapter.close()


if __name__ == "__main__":
    asyncio.run(test_bybit_history())
