# Imports should work natively now via pip install -e .
import asyncio
import json
import logging
import os

from thalex.thalex import Thalex, Network
from dotenv import load_dotenv


async def verify_raw():
    load_dotenv()

    api_key = os.getenv("THALEX_PROD_API_KEY_ID") or os.getenv("THALEX_KEY_ID")
    api_secret = os.getenv("THALEX_PROD_PRIVATE_KEY") or os.getenv("THALEX_PRIVATE_KEY")
    is_testnet = os.getenv("TRADING_MODE", "testnet").lower() == "testnet"

    if not api_key:
        print("No API Key.")
        return

    net = Network.TEST if is_testnet else Network.PROD
    client = Thalex(network=net)

    await client.initialize()
    try:
        req_id = 1
        await client.login(api_key, api_secret, id=req_id)
        # Wait for login response? Library handles it?
        # Thalex.login sends command. We need to receive response.
        # But we can just try calling account_summary

        # We need a small loop to consume login response first
        resp = await client.receive()
        print(f"Login Response: {resp}")

        print("Calling account_summary...")
        req_id += 1
        await client.account_summary(id=req_id)

        # Wait for response
        while True:
            msg = await asyncio.wait_for(client.receive(), timeout=5.0)
            if isinstance(msg, str):
                msg = json.loads(msg)

            print(f"Received: {msg}")
            if msg.get("id") == req_id:
                break

    except Exception as e:
        print(f"Error: {e}")
    finally:
        await client.disconnect()


if __name__ == "__main__":
    asyncio.run(verify_raw())
