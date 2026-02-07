import asyncio
import os
import asyncpg


async def main():
    db_user = os.getenv("DATABASE_USER", "postgres")
    db_pass = os.getenv("DATABASE_PASSWORD", "password")
    db_host = os.getenv("DATABASE_HOST", "localhost")
    db_name = os.getenv("DATABASE_NAME", "thalex_trading")
    db_port = os.getenv("DATABASE_PORT", "5433")

    dsn = f"postgresql://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}"
    print(f"Connecting to {dsn}")

    try:
        conn = await asyncpg.connect(dsn)
        rows = await conn.fetch("SELECT * FROM account_balances")
        if not rows:
            print("No rows found in account_balances.")
        else:
            print(f"Found {len(rows)} rows:")
            for row in rows:
                print(dict(row))
        await conn.close()
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
