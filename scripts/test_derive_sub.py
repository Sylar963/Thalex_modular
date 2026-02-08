import asyncio
import websockets
import json

async def test():
    uri = "wss://api.lyra.finance/ws"
    async with websockets.connect(uri) as websocket:
        print(f"Connected to {uri}")
        
        # Subscribe to HYPE-PERP
        payload = {
            "jsonrpc": "2.0",
            "method": "subscribe",
            "params": {
                "channels": ["ticker_slim.HYPE-PERP"]
            },
            "id": 1
        }
        await websocket.send(json.dumps(payload))
        print("Sent subscription request")
        
        # Wait for messages
        for _ in range(5):
            try:
                response = await asyncio.wait_for(websocket.recv(), timeout=10)
                data = json.loads(response)
                print("Received message:")
                print(json.dumps(data, indent=2))
            except Exception as e:
                print(f"Error: {e}")
                break

if __name__ == "__main__":
    asyncio.run(test())
