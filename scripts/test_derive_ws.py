import asyncio
import websockets
import json

async def test():
    uri = "wss://api.lyra.finance/ws"
    async with websockets.connect(uri) as websocket:
        print(f"Connected to {uri}")
        payload = {
            "jsonrpc": "2.0",
            "method": "public/get_instruments",
            "params": {
                "currency": "HYPE",
                "kind": "option",
                "expired": False
            },
            "id": 1
        }
        await websocket.send(json.dumps(payload))
        print("Sent request")
        
        try:
            response = await asyncio.wait_for(websocket.recv(), timeout=5)
            print("Received response:")
            print(json.dumps(json.loads(response), indent=2))
        except Exception as e:
            print(f"Error receiving: {e}")

if __name__ == "__main__":
    asyncio.run(test())
