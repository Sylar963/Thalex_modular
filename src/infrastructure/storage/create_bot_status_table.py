import asyncio
import os
import asyncpg
from dotenv import load_dotenv

load_dotenv()


async def create_table():
    host = os.getenv("DATABASE_HOST", "localhost")
    user = os.getenv("DATABASE_USER", "postgres")
    password = os.getenv("DATABASE_PASSWORD", "password")
    db_name = os.getenv("DATABASE_NAME", "thalex_trading")
    port = os.getenv("DATABASE_PORT", "5432")

    print(f"Connecting to {db_name} at {host}:{port} as {user}...")

    try:
        conn = await asyncpg.connect(
            host=host, user=user, password=password, database=db_name, port=port
        )

        print("Creating bot_status table...")
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS bot_status (
                time TIMESTAMPTZ NOT NULL,
                symbol TEXT NOT NULL,
                exchange TEXT NOT NULL,
                risk_state TEXT NOT NULL,
                trend_state TEXT NOT NULL,
                execution_mode TEXT NOT NULL,
                active_signals TEXT,
                risk_breach BOOLEAN DEFAULT FALSE,
                metadata JSONB DEFAULT '{}'::jsonb
            );
        """)

        # Convert to hypertable if not already
        try:
            await conn.execute(
                "SELECT create_hypertable('bot_status', 'time', if_not_exists => TRUE);"
            )
            print("Converted to hypertable.")
        except Exception as e:
            print(f"Hypertable creation info: {e}")

        print("Table bot_status created successfully.")
        await conn.close()

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    asyncio.run(create_table())
