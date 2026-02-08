import asyncio
import websockets
import json

async def test():
    uri = "wss://api.lyra.finance/ws"
    async with websockets.connect(uri) as websocket:
        print(f"Connected to {uri}")
        payload = {
            "jsonrpc": "2.0",
            "method": "public/get_tickers",
            "params": {
                "currency": "HYPE",
                "instrument_type": "option"
            },
            "id": 1
        }
        await websocket.send(json.dumps(payload))
        print("Sent request public/get_tickers")
        
        try:
            response = await asyncio.wait_for(websocket.recv(), timeout=5)
            print("Received response:")
            data = json.loads(response)
            if "error" in data:
                print(json.dumps(data, indent=2))
            else:
                # Print only first 2 results to save space
                result = data.get("result", [])
                print(f"Result count: {len(result)}")
                if result:
                    print(json.dumps(result[:2], indent=2))
        except Exception as e:
            print(f"Error receiving: {e}")

if __name__ == "__main__":
    asyncio.run(test())
