import asyncio
import sys
import os

# Add root to path
sys.path.append(os.getcwd())

from src.data_ingestion.historical_options_loader import HistoricalOptionsLoader


async def main():
    print("Starting minimal loader test...")
    loader = HistoricalOptionsLoader()
    # Run for just 1 day to verify symbol fix
    await loader.run(days_back=1)
    print("Loader test complete.")


if __name__ == "__main__":
    asyncio.run(main())
