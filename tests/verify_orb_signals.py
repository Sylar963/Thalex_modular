import sys
import os
import asyncio
from datetime import datetime, timezone

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.domain.signals.open_range import OpenRangeSignalEngine
from src.domain.signals.volume_candle import VolumeCandleSignalEngine
from src.domain.sensors.canary_sensor import CanarySensor
from src.domain.signals.inventory_bias import InventoryBiasEngine
from src.domain.entities import Ticker


def test_orb_signals():
    print("Testing OpenRangeSignalEngine...")
    or_engine = OpenRangeSignalEngine()
    symbol = "BTC-USD"

    # Initialize state
    ts = datetime(2026, 2, 5, 20, 5, 0, tzinfo=timezone.utc).timestamp()
    or_engine.update(Ticker(symbol, 10000, 10100, 1, 1, 10050, 10, timestamp=ts))

    # Test get_signals with symbol
    try:
        signals = or_engine.get_signals(symbol)
        print(
            f"✅ OpenRangeSignalEngine.get_signals('{symbol}') returned: {list(signals.keys())}"
        )
        if "orh" not in signals:
            print("❌ Failed: 'orh' missing in signals")
            sys.exit(1)
    except Exception as e:
        print(f"❌ OpenRangeSignalEngine.get_signals('{symbol}') crashed: {e}")
        sys.exit(1)

    # Test get_signals without symbol (should be empty but safe)
    try:
        signals = or_engine.get_signals()
        print(f"✅ OpenRangeSignalEngine.get_signals() returned: {signals}")
    except Exception as e:
        print(f"❌ OpenRangeSignalEngine.get_signals() crashed: {e}")
        sys.exit(1)


def test_other_engines():
    print("\nTesting other engines...")

    # VolumeCandle
    try:
        vc = VolumeCandleSignalEngine()
        vc.get_signals("BTC-USD")
        print("✅ VolumeCandleSignalEngine.get_signals('BTC-USD') works")
    except Exception as e:
        print(f"❌ VolumeCandleSignalEngine crashed: {e}")
        sys.exit(1)

    # Canary
    try:
        cs = CanarySensor()
        cs.get_signals("BTC-USD")
        print("✅ CanarySensor.get_signals('BTC-USD') works")
    except Exception as e:
        print(f"❌ CanarySensor crashed: {e}")
        sys.exit(1)

    # InventoryBias
    try:
        ib = InventoryBiasEngine()
        ib.get_signals("BTC-USD")
        print("✅ InventoryBiasEngine.get_signals('BTC-USD') works")
    except Exception as e:
        print(f"❌ InventoryBiasEngine crashed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    test_orb_signals()
    test_other_engines()
    print("\n🎉 All Verification Tests Passed!")
