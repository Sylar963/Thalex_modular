import asyncio
import os
import asyncpg
from dotenv import load_dotenv


async def check_db():
    load_dotenv()
    db_user = os.getenv("DATABASE_USER", "postgres")
    db_pass = os.getenv("DATABASE_PASSWORD", "password")
    db_host = os.getenv("DATABASE_HOST", "localhost")
    db_port = os.getenv("DATABASE_PORT", "5433")
    db_name = os.getenv("DATABASE_NAME", "thalex_trading")
    dsn = f"postgresql://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}"

    print(f"Connecting to {dsn}...")
    try:
        conn = await asyncpg.connect(dsn)

        print("\n--- Account Balances ---")
        rows = await conn.fetch("SELECT * FROM account_balances")
        for r in rows:
            print(dict(r))

        print("\n--- Portfolio Positions ---")
        rows = await conn.fetch("SELECT * FROM portfolio_positions")
        for r in rows:
            print(dict(r))

        print("\n--- Market Tickers (Last 5) ---")
        rows = await conn.fetch(
            "SELECT symbol, exchange, bid, ask, last, time FROM market_tickers ORDER BY time DESC LIMIT 5"
        )
        for r in rows:
            print(
                f"{r['time']} {r['symbol']} {r['exchange']} Last:{r['last']} Bid:{r['bid']} Ask:{r['ask']}"
            )

        await conn.close()
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    asyncio.run(check_db())
