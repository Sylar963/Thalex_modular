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
                "instrument_type": "option",
                "expired": False
            },
            "id": 1
        }
        await websocket.send(json.dumps(payload))
        
        try:
            response = await asyncio.wait_for(websocket.recv(), timeout=5)
            data = json.loads(response)
            result = data.get("result", [])
            print(f"Result count: {len(result)}")
            if result:
                expiries = {}
                for inst in result:
                    name = inst['instrument_name']
                    parts = name.split('-')
                    if len(parts) >= 2:
                        name_expiry = parts[1]
                        ts_expiry = inst['option_details']['expiry']
                        expiries[name_expiry] = ts_expiry
                
                print("Unique expiries (name -> timestamp):")
                for name, ts in sorted(expiries.items()):
                    print(f"  {name} -> {ts}")
                
                # Try to find what field represents the expiry date used in tickers
                # Maybe it's an integer timestamp?
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(test())
