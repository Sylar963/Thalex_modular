import asyncio
import aiohttp
import json


async def verify_orb():
    url = "http://127.0.0.1:8000/api/v1/market/signals/open-range/levels?symbol=BTC-PERPETUAL"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    print("\n--- Open Range Levels ---")
                    print(json.dumps(data, indent=2))

                    if data.get("session_start_timestamp"):
                        print(
                            f"\nSUCCESS: session_start_timestamp is present: {data['session_start_timestamp']}"
                        )
                    else:
                        print("\nWARNING: session_start_timestamp is MISSING or NULL")
                else:
                    print(f"Failed to fetch ORB levels: {response.status}")
                    print(await response.text())
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    asyncio.run(verify_orb())
