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
        history = await adapter.get_history(symbol, start, end, resolution="1m")
        print(f"Returned {len(history)} candles.")
        if len(history) > 0:
            print(f"First: {history[0]}")
            print(f"Last: {history[-1]}")
        else:
            print("No history found. Trying BTCUSDT...")
            history = await adapter.get_history("BTCUSDT", start, end, resolution="1m")
            print(f"Returned {len(history)} candles for BTCUSDT.")

    except Exception as e:
        print(f"Error: {e}")

    await adapter.disconnect()


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    asyncio.run(verify_history())
