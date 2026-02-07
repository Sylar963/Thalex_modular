import sys
import importlib.util

try:
    if importlib.util.find_spec("src.adapters.exchanges.thalex_adapter") is not None:
        print("[OK] src.adapters.exchanges.thalex_adapter is available")
    else:
        print("[FAIL] src.adapters.exchanges.thalex_adapter not found")
except Exception as e:
    print(f"[ERROR] testing src: {e}")

try:
    import thalex

    print("[OK] import thalex")
    print(f"Thalex file: {thalex.__file__}")
except ImportError as e:
    print(f"[FAIL] import thalex: {e}")

print("Path:", sys.path)
