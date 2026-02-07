import asyncio
import os
import sys

# Fix path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)

from src.api.repositories.portfolio_repository import PortfolioRepository
from src.adapters.storage.timescale_adapter import TimescaleDBAdapter


async def verify_api_response():
    # Setup
    db_user = os.getenv("DATABASE_USER", "postgres")
    db_pass = os.getenv("DATABASE_PASSWORD", "password")
    db_host = os.getenv("DATABASE_HOST", "localhost")
    db_port = os.getenv("DATABASE_PORT", "5433")
    db_name = os.getenv("DATABASE_NAME", "thalex_trading")
    dsn = f"postgresql://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}"

    adapter = TimescaleDBAdapter(dsn)
    await adapter.connect()
    repo = PortfolioRepository(adapter)

    print("Fetching Summary for 'thalex'...")
    summary = await repo.get_summary(exchange="thalex")
    print(f"Summary Response: {summary}")

    print("Fetching Summary for 'Thalex' (Title Case)...")
    summary_cap = await repo.get_summary(exchange="Thalex")
    print(f"Summary (Title Case): {summary_cap}")

    await adapter.disconnect()


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    asyncio.run(verify_api_response())
