import sys
import os
import asyncio

# Fix path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "thalex_py"))

from src.services.market_feed import main

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
