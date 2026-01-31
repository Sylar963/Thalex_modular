import asyncio
import logging
import signal
from typing import Dict, Any
import sys
import os
from dotenv import load_dotenv

# Ensure project root is in path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)
# Add thalex_py to path for legacy lib
sys.path.append(os.path.join(project_root, "thalex_py"))

from src.adapters.exchanges.thalex_adapter import ThalexAdapter
from src.domain.strategies.avellaneda import AvellanedaStoikovStrategy
from src.domain.signals.volume_candle import VolumeCandleSignalEngine
from src.domain.risk.basic_manager import BasicRiskManager
from src.use_cases.quoting_service import QuotingService

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("Main")


async def main():
    load_dotenv()

    # Determined logic for keys based on env
    testnet = os.getenv("TRADING_MODE", "testnet").lower() == "testnet"
    if testnet:
        api_key = os.getenv("THALEX_TEST_API_KEY_ID")
        api_secret = os.getenv("THALEX_TEST_PRIVATE_KEY")
    else:
        api_key = os.getenv("THALEX_PROD_API_KEY_ID") or os.getenv("THALEX_KEY_ID")
        api_secret = os.getenv("THALEX_PROD_PRIVATE_KEY") or os.getenv(
            "THALEX_PRIVATE_KEY"
        )

    symbol = os.getenv("PRIMARY_INSTRUMENT", "BTC-PERPETUAL")

    if not api_key or not api_secret:
        logger.warning("API credentials not found in environment. Using mock/empty.")
        # In production this should likely exit or ask for input

    # 2. Dependency Injection
    gateway = ThalexAdapter(api_key, api_secret, testnet=testnet)

    strategy = AvellanedaStoikovStrategy()
    strategy.setup(
        {
            "gamma": 0.5,
            "kappa": 2.0,
            "volatility": 0.05,
            "position_limit": 5.0,
            "base_spread": 0.001,
        }
    )

    signal_engine = VolumeCandleSignalEngine(volume_threshold=10.0, max_candles=50)

    risk_manager = BasicRiskManager(max_position=10.0, max_order_size=1.0)

    # 3. Service Assembly
    service = QuotingService(gateway, strategy, signal_engine, risk_manager)

    # 4. Graceful Shutdown Handler
    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()

    def signal_handler():
        logger.info("Signal received, stopping...")
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, signal_handler)

    # 5. Start Application
    try:
        await service.start(symbol)

        # Keep running until stop signal
        await stop_event.wait()

    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
    finally:
        await service.stop()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass  # Handle separately if needed, but signal handler covers it
