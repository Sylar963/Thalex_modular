import asyncio
import os
import sys
import asyncpg


async def verify_ingestion():
    # Use the same DSN as the app
    db_dsn = "postgresql://postgres:password@localhost:5433/thalex_trading"
    print("\n--- Checking ETHUSDT Data ---")
    try:
        conn = await asyncpg.connect(db_dsn)

        # Check market_trades
        count = await conn.fetchval(
            "SELECT count(*) FROM market_trades WHERE symbol = 'ETHUSDT'"
        )
        print(f"ETHUSDT Trade Count: {count}")

        if count > 0:
            latest = await conn.fetchrow(
                "SELECT * FROM market_trades WHERE symbol = 'ETHUSDT' ORDER BY time DESC LIMIT 1"
            )
            print(f"Latest Trade: {latest}")
            print("SUCCESS: Data Ingestion is working for non-traded symbol.")
        else:
            print(
                "WARNING: No data found for ETHUSDT yet. Verify service is running or wait for trades."
            )

        await conn.close()

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    asyncio.run(verify_ingestion())
