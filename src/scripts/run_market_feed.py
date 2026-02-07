# Imports should work natively now via pip install -e .
import sys
import os
import asyncio

from src.services.market_feed import main

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
