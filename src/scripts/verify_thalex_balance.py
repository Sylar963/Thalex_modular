import asyncio
import os
import sys

# Fix path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "thalex_py"))

from src.adapters.exchanges.thalex_adapter import ThalexAdapter
from dotenv import load_dotenv


async def verify():
    load_dotenv()

    # Check if we have keys
    api_key = os.getenv("THALEX_PROD_API_KEY_ID") or os.getenv("THALEX_KEY_ID")
    api_secret = os.getenv("THALEX_PROD_PRIVATE_KEY") or os.getenv("THALEX_PRIVATE_KEY")

    if not api_key:
        print("No API Key found. Cannot verify.")
        return

    print("Initializing Adapter...")
    adapter = ThalexAdapter(api_key, api_secret, testnet=False)
    # Check if we are in testnet mode really
    is_testnet = os.getenv("TRADING_MODE", "testnet").lower() == "testnet"
    if is_testnet:
        print("Using Testnet mode")
        adapter = ThalexAdapter(api_key, api_secret, testnet=True)

    try:
        await adapter.connect()
        print("Connected.")

        print("Fetching balances...")
        balances = await adapter.get_balances()
        print(f"Balances returned: {balances}")

        if balances and len(balances) > 0:
            print(f"Equity: {balances[0].equity}")
            if balances[0].equity == 0:
                print(
                    "WARNING: Equity is 0. If this is a real account with funds, check mapping."
                )
            else:
                print("SUCCESS: Non-zero equity found.")
        else:
            print("FAILURE: No balances returned.")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        await adapter.disconnect()


if __name__ == "__main__":
    asyncio.run(verify())
