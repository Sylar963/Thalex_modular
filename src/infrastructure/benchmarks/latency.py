import asyncio
import logging
import time
import os
import sys
from statistics import mean, stdev

# Ensure project root is in path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.append(project_root)
# Add thalex_py to path
sys.path.append(os.path.join(project_root, "thalex_py"))

from src.adapters.exchanges.thalex_adapter import ThalexAdapter
from src.domain.entities import Order, OrderSide, OrderType, OrderStatus
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("Benchmark")


async def run_latency_test():
    load_dotenv()

    api_key = os.getenv("THALEX_PROD_API_KEY_ID") or os.getenv("THALEX_KEY_ID")
    api_secret = os.getenv("THALEX_PROD_PRIVATE_KEY") or os.getenv("THALEX_PRIVATE_KEY")

    # Check for testnet vs prod key logic in .env
    # Assuming user put PROD keys but we might want to check network
    network_env = os.getenv("NETWORK", "test")
    testnet = network_env == "test"

    # If keys are missing, try load from specific env vars present in .example.env structure
    if not api_key:
        logger.error("API Keys not found. Please ensure .env is set.")
        return

    logger.info(f"Starting Latency Benchmark on {'TESTNET' if testnet else 'MAINNET'}")

    adapter = ThalexAdapter(api_key, api_secret, testnet=testnet)

    try:
        await adapter.connect()
        logger.info("Connected.")

        # Subscribe to ensure connection is warm
        symbol = "BTC-PERPETUAL"
        await adapter.subscribe_ticker(symbol)
        await asyncio.sleep(2)  # Warmup

        latencies = []
        cancel_latencies = []

        iterations = 5
        logger.info(f"Running {iterations} iterations...")

        for i in range(iterations):
            # 1. Place Order (Limit, very low price to avoid fill)
            # Fetch generic price or assume safe low price
            safe_bid = 1000.0  # BTC @ 1000 is safe from accidentally filling

            order = Order(
                id=f"bench_{int(time.time() * 1000)}",
                symbol=symbol,
                side=OrderSide.BUY,
                type=OrderType.LIMIT,
                price=safe_bid,
                size=0.01,
                timestamp=time.time(),
            )

            t0 = time.time()
            placed_order = await adapter.place_order(order)
            t1 = time.time()

            rtt_ms = (t1 - t0) * 1000
            latencies.append(rtt_ms)
            logger.info(
                f"Order {i + 1} placed in {rtt_ms:.2f}ms. Status: {placed_order.status}"
            )

            if placed_order.status == OrderStatus.REJECTED:
                logger.warning("Order rejected, check logs.")
                continue

            # 2. Cancel Order
            t2 = time.time()
            canceled = await adapter.cancel_order(placed_order.exchange_id)
            t3 = time.time()

            cancel_ms = (t3 - t2) * 1000
            cancel_latencies.append(cancel_ms)
            logger.info(
                f"Order {i + 1} canceled in {cancel_ms:.2f}ms. Success: {canceled}"
            )

            await asyncio.sleep(0.5)

        if latencies:
            logger.info("\n--- Results ---")
            logger.info(
                f"Placement Latency: Avg={mean(latencies):.2f}ms, Min={min(latencies):.2f}ms, Max={max(latencies):.2f}ms"
            )
            logger.info(f"Cancel Latency:    Avg={mean(cancel_latencies):.2f}ms")
        else:
            logger.error("No valid latency data collected.")

    except Exception as e:
        logger.error(f"Benchmark failed: {e}", exc_info=True)
    finally:
        await adapter.disconnect()


if __name__ == "__main__":
    asyncio.run(run_latency_test())
