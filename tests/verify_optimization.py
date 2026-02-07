import sys
import os
import asyncio

# Imports work natively now via pip install -e .


async def verify():
    print("Verifying optimizations...")

    # 1. Check uvloop
    try:
        import uvloop

        print("[OK] uvloop installed")
    except ImportError:
        print("[FAIL] uvloop not found")
        return

    # 2. Check orjson
    try:
        import orjson

        print("[OK] orjson installed")
    except ImportError:
        print("[FAIL] orjson not found")
        return

    # 3. Check Adapters
    from src.adapters.exchanges.thalex_adapter import ThalexAdapter
    from src.adapters.exchanges.bybit_adapter import BybitAdapter

    try:
        # Instantiate to check for syntax errors in __init__ or usage
        # We pass dummy keys, just checking import and init logic
        thalex = ThalexAdapter("key", "secret")
        print("[OK] ThalexAdapter instantiated")

        # Check if helper works
        json_str = thalex._fast_json_encode({"test": 123})
        if '{"test":123}' in json_str:  # orjson might satisfy this
            print(f"[OK] ThalexAdapter JSON Encode: {json_str}")
        else:
            print(f"[FAIL] ThalexAdapter JSON Encode unexpected: {json_str}")

        bybit = BybitAdapter("key", "secret")
        print("[OK] BybitAdapter instantiated")

        from src.adapters.exchanges.hyperliquid_adapter import HyperliquidAdapter

        hl = HyperliquidAdapter("0x" + "a" * 64, testnet=True)  # Mock private key
        print("[OK] HyperliquidAdapter instantiated")
        hl_json = hl._fast_json_encode({"test": 1})
        print(f"[OK] Hyperliquid JSON: {hl_json}")

        from src.adapters.exchanges.binance_adapter import BinanceAdapter

        bn = BinanceAdapter("key", "secret")
        print("[OK] BinanceAdapter instantiated")
        bn_json = bn._fast_json_encode({"test": 2})
        print(f"[OK] Binance JSON: {bn_json}")

    except Exception as e:
        print(f"[FAIL] Adapter instantiation failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    loop.run_until_complete(verify())
