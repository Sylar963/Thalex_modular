import asyncio
import os
import sys
from datetime import datetime, timedelta

# Fix path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)

from src.adapters.storage.timescale_adapter import TimescaleDBAdapter


async def verify_history():
    # Setup
    db_user = os.getenv("DATABASE_USER", "postgres")
    db_pass = os.getenv("DATABASE_PASSWORD", "password")
    db_host = os.getenv("DATABASE_HOST", "localhost")
    db_port = os.getenv("DATABASE_PORT", "5433")
    db_name = os.getenv("DATABASE_NAME", "thalex_trading")
    dsn = f"postgresql://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}"

    adapter = TimescaleDBAdapter(dsn)
    await adapter.connect()

    symbol = "BTC-PERPETUAL"
    end = datetime.utcnow().timestamp()
    start = end - 3600 * 24  # Last 24 hours

    print(
        f"Fetching history for {symbol} from {datetime.fromtimestamp(start)} to {datetime.fromtimestamp(end)}"
    )

    try:
        # Test 1: Thalex only
        print("\n--- Test 1: Venue = thalex ---")
        history_thalex = await adapter.get_history(
            symbol, start, end, resolution="1m", exchange="thalex"
        )
        print(f"Returned {len(history_thalex)} candles.")
        if history_thalex:
            print(f"First: {history_thalex[0]}")
            print(f"Last: {history_thalex[-1]}")

        # Test 2: ALL
        print("\n--- Test 2: Venue = all ---")
        history_all = await adapter.get_history(
            symbol, start, end, resolution="1m", exchange="all"
        )
        print(f"Returned {len(history_all)} candles.")
        if history_all:
            print(f"First: {history_all[0]}")
            print(f"Last: {history_all[-1]}")

        # Comparison
        print(f"\nDifference in count: {len(history_all) - len(history_thalex)}")

    except Exception as e:
        print(f"Error: {e}")

    await adapter.disconnect()


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    asyncio.run(verify_history())
