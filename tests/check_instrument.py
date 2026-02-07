import asyncio
import aiohttp
import json
from datetime import datetime


async def check():
    url = "https://thalex.com/api/v2/public/mark_price_historical_data"

    # Also check instruments list to see valid names
    print("\nChecking available instruments...")
    async with aiohttp.ClientSession() as session:
        async with session.get("https://thalex.com/api/v2/public/instruments") as resp:
            data = await resp.json()
            result = data.get("result", [])
            print(f"Found {len(result)} instruments.")

            names = [i.get("instrument_name") for i in result[:10]]
            print(f"Sample Names: {names}")


asyncio.run(check())
