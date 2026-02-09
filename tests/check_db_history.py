import asyncio
import asyncpg
import time
from datetime import datetime, timedelta


async def check_history():
    dsn = "postgresql://postgres:password@localhost:5432/thalex_trading"
    try:
        conn = await asyncpg.connect(dsn)
        print("Connected to DB")
    except Exception as e:
        print(f"Failed to connect to 5432: {e}")
        dsn = "postgresql://postgres:password@localhost:5433/thalex_trading"
        conn = await asyncpg.connect(dsn)
        print("Connected to DB on 5433")

    now = datetime.now()
    start_15d = now - timedelta(days=15)

    symbol = "BTC-PERPETUAL"

    # Check count in market_tickers
    rows = await conn.fetch(
        """
        SELECT 
            time_bucket('1 hour', time) AS bucket,
            COUNT(*) as count
        FROM market_tickers
        WHERE symbol = $1 AND time >= $2
        GROUP BY bucket
        ORDER BY bucket ASC
    """,
        symbol,
        start_15d,
    )

    print(f"Buckets found for {symbol} (last 15d): {len(rows)}")

    if len(rows) > 0:
        print(f"Earliest bucket: {rows[0]['bucket']}")
        print(f"Latest bucket: {rows[-1]['bucket']}")

        # Identify missing ranges
        found_buckets = {r["bucket"] for r in rows}
        current = rows[0]["bucket"]
        end = rows[-1]["bucket"]

        missing_start = None

        print("\nMissing Ranges:")
        while current <= end:
            if current not in found_buckets:
                if missing_start is None:
                    missing_start = current
            else:
                if missing_start is not None:
                    print(f"  {missing_start} to {current - timedelta(hours=1)}")
                    missing_start = None
            current += timedelta(hours=1)

        if missing_start is not None:
            print(f"  {missing_start} to {end}")
    else:
        print("No data found in market_tickers")

    await conn.close()


if __name__ == "__main__":
    asyncio.run(check_history())
