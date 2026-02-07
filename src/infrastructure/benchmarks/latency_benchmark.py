import asyncio
import logging
import os
import time
import statistics
from typing import List, Dict, Any
from dataclasses import dataclass
from dotenv import load_dotenv

from src.adapters.exchanges.thalex_adapter import ThalexAdapter
from src.adapters.exchanges.bybit_adapter import BybitAdapter
from src.domain.entities import Order, OrderSide, OrderType, OrderStatus, Ticker

# Load env vars
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("LatencyBenchmark")

# Configuration
load_dotenv()

# Determine Network Mode
# Default to TESTNET unless TRADING_MODE is explicitly 'production' or 'prod'
TRADING_MODE = os.getenv("TRADING_MODE", "testnet").lower()
USE_TESTNET = TRADING_MODE not in ["production", "prod"]

logger.info(f"Running in {TRADING_MODE.upper()} mode (Testnet={USE_TESTNET})")

# Thalex Credentials
if USE_TESTNET:
    THALEX_API_KEY = os.getenv("THALEX_TEST_API_KEY_ID")
    THALEX_API_SECRET = os.getenv("THALEX_TEST_PRIVATE_KEY")
else:
    THALEX_API_KEY = os.getenv("THALEX_PROD_API_KEY_ID") or os.getenv("THALEX_KEY_ID")
    THALEX_API_SECRET = os.getenv("THALEX_PROD_PRIVATE_KEY") or os.getenv(
        "THALEX_PRIVATE_KEY"
    )

# Bybit Credentials
BYBIT_API_KEY = os.getenv("BYBIT_API_KEY")
BYBIT_API_SECRET = os.getenv("BYBIT_API_SECRET")

# Benchmark Parameters
ITERATIONS = 10
WARMUP_ROUNDS = 2
DEEP_OTM_FACTOR = 0.5  # Place buy orders at 50% of market price

# Benchmark Parameters
ITERATIONS = 10
WARMUP_ROUNDS = 2
DEEP_OTM_FACTOR = 0.5  # Place buy orders at 50% of market price


@dataclass
class LatencyResult:
    exchange: str
    round: int
    order_latency_ms: float
    cancel_latency_ms: float
    success: bool
    error: str = ""


class LatencyBenchmark:
    def __init__(self):
        self.results: List[LatencyResult] = []
        self.thalex: ThalexAdapter = None
        self.bybit: BybitAdapter = None

    async def setup(self):
        """Initialize adapters and connect"""
        logger.info(f"Initializing adapters (Testnet={USE_TESTNET})...")

        # Thalex Setup
        if THALEX_API_KEY and THALEX_API_SECRET:
            self.thalex = ThalexAdapter(
                api_key=THALEX_API_KEY,
                api_secret=THALEX_API_SECRET,
                testnet=USE_TESTNET,
            )
            # Thalex connects asynchronously
            try:
                await self.thalex.connect()
                logger.info("Thalex connected.")
            except Exception as e:
                logger.error(f"Failed to connect to Thalex: {e}")
                self.thalex = None
        else:
            logger.warning("Thalex API keys missing. Skipping Thalex.")

        # Bybit Setup
        if BYBIT_API_KEY and BYBIT_API_SECRET:
            self.bybit = BybitAdapter(
                api_key=BYBIT_API_KEY, api_secret=BYBIT_API_SECRET, testnet=USE_TESTNET
            )
            try:
                await self.bybit.connect()
                logger.info("Bybit connected.")
            except Exception as e:
                logger.error(f"Failed to connect to Bybit: {e}")
                self.bybit = None
        else:
            logger.warning("Bybit API keys missing. Skipping Bybit.")

    async def teardown(self):
        """Disconnect adapters"""
        logger.info("Tearing down adapters...")
        if self.thalex:
            await self.thalex.disconnect()
        if self.bybit:
            await self.bybit.disconnect()

    async def run_thalex_benchmark(self):
        if not self.thalex or not self.thalex.connected:
            return

        exchange_name = "Thalex"
        symbol = "BTC-PERPETUAL"  # Standard Thalex instrument
        logger.info(f"Starting {exchange_name} benchmark on {symbol}...")

        # Get Ticker for Reference Price
        # We need to implement a way to get ticker since subscribe is async callback based
        # For simplicity, we can reuse the ticker update callback mechanism or try to fetch if available.
        # But ThalexAdapter relies on ws subscription.
        # Let's subscribe and wait for first ticker.

        price_event = asyncio.Event()
        current_price = 0.0

        async def on_ticker(ticker: Ticker):
            nonlocal current_price
            if ticker.symbol == symbol and ticker.last > 0:
                current_price = ticker.last
                price_event.set()

        self.thalex.set_ticker_callback(on_ticker)
        await self.thalex.subscribe_ticker(symbol)

        logger.info("Waiting for market data...")
        try:
            await asyncio.wait_for(price_event.wait(), timeout=10.0)
        except asyncio.TimeoutError:
            logger.error("Timeout waiting for Thalex ticker data.")
            return

        buy_price = current_price * DEEP_OTM_FACTOR
        # Ensure tick size precision (assuming 0.5 for BTC-PERP on Thalex roughly, or standard 0.01)
        # We should use instrument info but for benchmark we can be safe with int for BTC
        buy_price = int(buy_price)

        logger.info(f"Market Price: {current_price}, Benchmark Buy Price: {buy_price}")

        # Warmup
        logger.info(f"Warming up ({WARMUP_ROUNDS} rounds)...")
        for _ in range(WARMUP_ROUNDS):
            await self._execute_round(self.thalex, symbol, buy_price, -1)

        # Benchmark
        logger.info(f"Running {ITERATIONS} benchmark rounds...")
        for i in range(ITERATIONS):
            await self._execute_round(self.thalex, symbol, buy_price, i)
            await asyncio.sleep(0.5)  # Small delay between rounds

    async def run_bybit_benchmark(self):
        if not self.bybit or not self.bybit.connected:
            return

        exchange_name = "Bybit"
        symbol = "BTCUSDT"
        logger.info(f"Starting {exchange_name} benchmark on {symbol}...")

        # Bybit Ticker
        # BybitAdapter also uses callback for ticker
        price_event = asyncio.Event()
        current_price = 0.0

        async def on_ticker(ticker: Ticker):
            nonlocal current_price
            # Bybit ticker might come as orderbook or different format, let's check
            # The adapter normalizes to Ticker
            if (
                ticker.symbol == symbol or ticker.symbol == "BTCUSDT"
            ) and ticker.ask > 0:
                # Use ask price for safety reference, or last
                current_price = ticker.ask
                price_event.set()

        self.bybit.set_ticker_callback(on_ticker)
        await self.bybit.subscribe_ticker(symbol)

        logger.info("Waiting for market data...")
        try:
            await asyncio.wait_for(price_event.wait(), timeout=10.0)
        except asyncio.TimeoutError:
            logger.error("Timeout waiting for Bybit ticker data.")
            return

        buy_price = current_price * DEEP_OTM_FACTOR
        # Bybit BTCUSDT tick size is usually 0.10 or 0.50 depending on market,
        # but let's round to 1 decimal to be safe for a deep OTM order
        buy_price = round(buy_price, 1)

        logger.info(f"Market Price: {current_price}, Benchmark Buy Price: {buy_price}")

        # Warmup
        logger.info(f"Warming up ({WARMUP_ROUNDS} rounds)...")
        for _ in range(WARMUP_ROUNDS):
            await self._execute_round(self.bybit, symbol, buy_price, -1)

        # Benchmark
        logger.info(f"Running {ITERATIONS} benchmark rounds...")
        for i in range(ITERATIONS):
            await self._execute_round(self.bybit, symbol, buy_price, i)
            # await asyncio.sleep(0.2)

    async def _execute_round(self, adapter, symbol: str, price: float, round_idx: int):
        exchange_name = adapter.name

        # 1. Prepare Order
        # Generate a unique client order ID if needed, but adapter handles mapping usually.
        # Thalex adapter uses order.id as label. Bybit adapter uses order.id as orderLinkId.
        # We need a unique ID for every order.
        cl_order_id = f"bench_{int(time.time() * 1000)}_{round_idx + 10}"

        order = Order(
            id=cl_order_id,
            symbol=symbol,
            side=OrderSide.BUY,
            type=OrderType.LIMIT,
            price=price,
            size=0.001,  # Min size usually safe
            timestamp=time.time(),
        )

        try:
            # 2. Place Order (Measure Latency)
            t0 = time.perf_counter()
            placed_order = await adapter.place_order(order)
            t1 = time.perf_counter()

            place_latency = (t1 - t0) * 1000.0

            if placed_order.status == OrderStatus.REJECTED:
                logger.warning(f"Order rejected on {exchange_name}")
                if round_idx >= 0:
                    self.results.append(
                        LatencyResult(
                            exchange_name,
                            round_idx,
                            place_latency,
                            0,
                            False,
                            "Rejected",
                        )
                    )
                return

            exchange_id = placed_order.exchange_id
            if not exchange_id:
                logger.warning(f"No exchange ID returned on {exchange_name}")
                # Force cancel using client id if possible?
                # Adapters implemented cancel_order(order_id) where order_id checks internal map
                exchange_id = cl_order_id

            # 3. Cancel Order (Measure Latency)
            t2 = time.perf_counter()
            cancelled = await adapter.cancel_order(exchange_id)
            t3 = time.perf_counter()

            cancel_latency = (t3 - t2) * 1000.0

            if round_idx >= 0:
                self.results.append(
                    LatencyResult(
                        exchange=exchange_name,
                        round=round_idx,
                        order_latency_ms=place_latency,
                        cancel_latency_ms=cancel_latency,
                        success=cancelled,
                    )
                )
                logger.info(
                    f"Round {round_idx}: Place={place_latency:.2f}ms, Cancel={cancel_latency:.2f}ms"
                )

        except Exception as e:
            logger.error(f"Error in round {round_idx} on {exchange_name}: {e}")
            if round_idx >= 0:
                self.results.append(
                    LatencyResult(exchange_name, round_idx, 0, 0, False, str(e))
                )

    def print_report(self):
        print("\n" + "=" * 60)
        print("LATENCY BENCHMARK REPORT")
        print("=" * 60)

        for exchange in ["thalex", "bybit"]:
            results = [
                r for r in self.results if r.exchange.lower() == exchange and r.success
            ]
            if not results:
                print(f"\nNo successful results for {exchange.upper()}")
                continue

            place_times = [r.order_latency_ms for r in results]
            cancel_times = [r.cancel_latency_ms for r in results]

            print(f"\n{exchange.upper()} ({len(results)} samples):")
            print("-" * 30)
            print(
                f"{'Metric':<15} | {'Min':<8} | {'Max':<8} | {'Avg':<8} | {'Median':<8}"
            )
            print("-" * 30)

            def print_stats(name, data):
                if not data:
                    return
                _min = min(data)
                _max = max(data)
                _avg = statistics.mean(data)
                _med = statistics.median(data)
                print(
                    f"{name:<15} | {_min:8.2f} | {_max:8.2f} | {_avg:8.2f} | {_med:8.2f}"
                )

            print_stats("Place Order", place_times)
            print_stats("Cancel Order", cancel_times)
            print("-" * 30)


async def main():
    benchmark = LatencyBenchmark()
    try:
        await benchmark.setup()

        # Run benchmarks sequentially
        if benchmark.thalex:
            await benchmark.run_thalex_benchmark()

        if benchmark.bybit:
            await benchmark.run_bybit_benchmark()

    except KeyboardInterrupt:
        logger.info("Benchmark interrupted.")
    finally:
        await benchmark.teardown()
        benchmark.print_report()


if __name__ == "__main__":
    asyncio.run(main())
