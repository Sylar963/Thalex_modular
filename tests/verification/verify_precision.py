import math
import json

try:
    import orjson
except ImportError:
    orjson = None


def _format_number(value: float, precision: int = 10) -> str:
    """Format a float to a decimal string, avoiding scientific notation."""
    if value == 0:
        return "0"
    s = "{:.{p}f}".format(value, p=precision)
    return s.rstrip("0").rstrip(".") if "." in s else s


def check_precision(tick_size, price_target):
    print(f"\n--- Testing Tick Size: {tick_size} ---")

    # Simulate strategy rounding
    raw_price = price_target
    rounded = round(raw_price / tick_size) * tick_size

    print(f"Raw: {raw_price}")
    print(f"Rounded (float): {rounded}")
    print(f"Rounded (str): {str(rounded)}")

    # Test _format_number
    formatted = _format_number(rounded)
    print(f"Formatted (_format_number): {formatted}")

    # Test standard json dumps
    json_dump = json.dumps({"p": rounded})
    print(f"json.dumps: {json_dump}")

    # Test orjson dumps
    if orjson:
        orjson_dump = orjson.dumps({"p": rounded}).decode("utf-8")
        print(f"orjson.dumps: {orjson_dump}")
        orjson_numpy_dump = orjson.dumps(
            {"p": rounded}, option=orjson.OPT_SERIALIZE_NUMPY
        ).decode("utf-8")
        print(f"orjson.dumps (numpy opt): {orjson_numpy_dump}")
    else:
        print("orjson not available")

    # Check for scientific notation in custom format
    if "e" in formatted.lower():
        print("FAIL: Scientific notation detected in _format_number output!")
    else:
        print("PASS: No scientific notation in _format_number")

    # Check for scientific notation in json
    if "e" in json_dump.lower():
        print("WARNING: json.dumps produces scientific notation")

    if orjson and "e" in orjson_dump.lower():
        print("WARNING: orjson.dumps produces scientific notation")


if __name__ == "__main__":
    # Test SUI scenarios
    sui_price = 3.20123

    print("=== SCENARIO 1: Standard SUI Price ===")
    check_precision(0.0001, sui_price)

    print("\n=== SCENARIO 2: Small Tick (Scientific Range) ===")
    check_precision(0.000001, 0.000025123)

    print("\n=== SCENARIO 3: Very Small Number (1e-5) ===")
    check_precision(0.00001, 0.00001)

    print("\n=== SCENARIO 4: SHIB/PEPE Low Price ===")
    check_precision(1e-8, 0.00000123)
