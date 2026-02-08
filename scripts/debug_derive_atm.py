import asyncio
import os
import logging
from src.adapters.exchanges.derive_adapter import DeriveAdapter

logging.basicConfig(level=logging.INFO)

async def test():
    adapter = DeriveAdapter(testnet=False)
    await adapter.connect()
    
    # Try one specific instrument from the targeted expiry (20260227)
    # Strike 32 C
    name = "HYPE-20260227-32-C"
    
    print(f"Fetching ticker for {name}...")
    try:
        # Use get_ticker (singular) which calls public/get_ticker
        # Let's check if the adapter has get_ticker or we should use _rpc_request_ws
        # The adapter has a _rpc_request_ws method.
        
        payload = {
            "instrument_name": name
        }
        response = await adapter._rpc_request_ws("public/get_ticker", payload)
        print("Response:")
        import json
        print(json.dumps(response, indent=2))
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        await adapter.disconnect()

if __name__ == "__main__":
    asyncio.run(test())
