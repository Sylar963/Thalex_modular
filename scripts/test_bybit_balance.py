import asyncio
import os
import aiohttp
import time
import hmac
import hashlib
import json
from dotenv import load_dotenv

# Load env variables
load_dotenv()

API_KEY = os.getenv("BYBIT_API_KEY")
API_SECRET = os.getenv("BYBIT_API_SECRET")
TESTNET = False  # Set to True if testing on testnet, based on config.json it is false

# URLS
REST_URL = "https://api.bybit.com"
if TESTNET:
    REST_URL = "https://api-testnet.bybit.com"


def get_signature(api_key, api_secret, payload):
    timestamp = str(int(time.time() * 1000))
    recv_window = "10000"
    param_str = f"{timestamp}{api_key}{recv_window}{payload}"
    signature = hmac.new(
        api_secret.encode("utf-8"),
        param_str.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    return timestamp, signature


async def fetch_balance():
    url = f"{REST_URL}/v5/account/wallet-balance"
    params = {"accountType": "UNIFIED"}
    query_str = "accountType=UNIFIED"

    timestamp, signature = get_signature(API_KEY, API_SECRET, query_str)

    headers = {
        "X-BAPI-API-KEY": API_KEY,
        "X-BAPI-TIMESTAMP": timestamp,
        "X-BAPI-RECV-WINDOW": "10000",
        "X-BAPI-SIGN": signature,
    }

    print(f"Fetching balances from {url} with params {params}")

    async with aiohttp.ClientSession() as session:
        async with session.get(url, params=params, headers=headers) as resp:
            data = await resp.json()
            print(json.dumps(data, indent=2))


if __name__ == "__main__":
    if not API_KEY or not API_SECRET:
        print("Error: API Keys not found in environment")
    else:
        asyncio.run(fetch_balance())
