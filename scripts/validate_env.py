#!/usr/bin/env python3
import os
import sys

REQUIRED_VARS = [
    ("DATABASE_HOST", "Database"),
    ("DATABASE_PORT", "Database"),
    ("DATABASE_NAME", "Database"),
    ("DATABASE_USER", "Database"),
    ("DATABASE_PASSWORD", "Database"),
]

VENUE_VARS = {
    "thalex": [
        ("THALEX_PROD_API_KEY_ID", "Thalex Production"),
        ("THALEX_PROD_PRIVATE_KEY", "Thalex Production"),
    ],
    "bybit": [
        ("BYBIT_API_KEY", "Bybit"),
        ("BYBIT_API_SECRET", "Bybit"),
    ],
    "binance": [
        ("BINANCE_API_KEY", "Binance"),
        ("BINANCE_API_SECRET", "Binance"),
    ],
    "hyperliquid": [
        ("HYPERLIQUID_PRIVATE_KEY", "Hyperliquid"),
    ],
}


def check_env():
    from dotenv import load_dotenv

    load_dotenv()

    errors = []
    warnings = []

    for var, category in REQUIRED_VARS:
        val = os.getenv(var)
        if not val or val.startswith("your_"):
            errors.append(f"[{category}] Missing or placeholder: {var}")

    for venue, vars_list in VENUE_VARS.items():
        venue_ok = True
        for var, category in vars_list:
            val = os.getenv(var)
            if not val or val.startswith("your_"):
                venue_ok = False
        if not venue_ok:
            warnings.append(
                f"[{venue.upper()}] Credentials not configured (venue will be disabled)"
            )

    print("=" * 60)
    print("Thalex Modular - Environment Validation")
    print("=" * 60)

    if errors:
        print("\n❌ ERRORS (must fix before starting):")
        for e in errors:
            print(f"   {e}")

    if warnings:
        print("\n⚠️  WARNINGS (optional venues not configured):")
        for w in warnings:
            print(f"   {w}")

    if not errors and not warnings:
        print("\n✅ All environment variables are configured!")

    print("\n" + "=" * 60)

    if errors:
        sys.exit(1)
    return True


if __name__ == "__main__":
    check_env()
