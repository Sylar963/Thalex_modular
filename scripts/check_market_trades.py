import asyncio
import asyncpg
import os
from datetime import datetime

# Credentials from .env
DB_USER = "postgres"
DB_PASS = "password"
DB_HOST = "localhost"
DB_PORT = "5433"
DB_NAME = "thalex_trading"

DSN = f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"


async def check_db():
    print(f"Connecting to {DSN}...")
    try:
        conn = await asyncpg.connect(DSN)
        print("Connected.")

        # Check total trades
        count = await conn.fetchval("SELECT COUNT(*) FROM market_trades")
        print(f"Total rows in market_trades: {count}")

        # Check trades by exchange
        rows = await conn.fetch(
            "SELECT exchange, COUNT(*) as c FROM market_trades GROUP BY exchange"
        )
        print("\nTrades by exchange:")
        for r in rows:
            print(f"- {r['exchange']}: {r['c']}")

        # Check recent trades
        rows = await conn.fetch(
            "SELECT time, symbol, price, size, exchange FROM market_trades ORDER BY time DESC LIMIT 5"
        )
        print("\nRecent 5 trades:")
        for r in rows:
            print(
                f"{r['time']} | {r['symbol']} | {r['price']} | {r['size']} | {r['exchange']}"
            )

        await conn.close()
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    asyncio.run(check_db())
