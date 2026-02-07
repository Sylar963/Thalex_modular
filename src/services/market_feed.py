import asyncio
import logging
import os
import signal
import time
from typing import List, Dict

from ..adapters.storage.timescale_adapter import TimescaleDBAdapter
from ..adapters.exchanges.bybit_adapter import BybitAdapter
from ..adapters.exchanges.thalex_adapter import ThalexAdapter
from ..domain.entities import Trade, Balance

logger = logging.getLogger("MarketFeed")


class MarketFeedService:
    def __init__(self, db_url: str, or_engine=None):
        self.db_url = db_url
        self.storage = TimescaleDBAdapter(db_url)
        self.or_engine = or_engine
        self.adapters: List = []
        self.running = False
        self.tasks: List[asyncio.Task] = []

    async def start(self):
        self.running = True
        logger.info("Starting MarketFeedService...")

        # Connect to DB
        try:
            await self.storage.connect()
            logger.info("Connected to TimescaleDB")
        except Exception as e:
            logger.error(f"Failed to connect to DB: {e}")
            return

        # Initialize Adapters based on ENV or Config
        # Thalex
        thalex_key = os.getenv("THALEX_PROD_API_KEY_ID") or os.getenv("THALEX_KEY_ID")
        thalex_secret = os.getenv("THALEX_PROD_PRIVATE_KEY") or os.getenv(
            "THALEX_PRIVATE_KEY"
        )
        if thalex_key and thalex_secret:
            thalex = ThalexAdapter(
                thalex_key, thalex_secret, testnet=False
            )  # Production typically involves collecting public data
            # Override testnet flag if env says so? Market Data usually same endpoint but good to be precise.
            # Assuming prod for market data feed unless specified.
            is_testnet = os.getenv("TRADING_MODE", "testnet").lower() == "testnet"
            thalex.testnet = is_testnet
            thalex.network = (
                thalex.network if not is_testnet else thalex.network
            )  # Logic handled in adapter init
            # Re-init adapter with correct testnet flag if needed, but passing it in constructor is better.
            thalex = ThalexAdapter(thalex_key, thalex_secret, testnet=is_testnet)

            thalex.set_trade_callback(self.on_trade)
            self.adapters.append(thalex)
            logger.info("Initialized Thalex Adapter")

        # Bybit
        bybit_key = os.getenv("BYBIT_API_KEY")  # Optional for public data strictly?
        # BybitAdapter requires key/secret even for public streams currently (due to auth flow in connect).
        # We'll use env vars.
        bybit_key = os.getenv("BYBIT_API_KEY", "")
        bybit_secret = os.getenv("BYBIT_API_SECRET", "")
        if bybit_key:  # Only add if we have some creds or modify adapter to allow anon
            bybit = BybitAdapter(
                bybit_key, bybit_secret, testnet=True
            )  # Defaulting to testnet matching bot?
            # Let's check env
            is_testnet = os.getenv("TRADING_MODE", "testnet").lower() == "testnet"
            bybit = BybitAdapter(bybit_key, bybit_secret, testnet=is_testnet)
            bybit.set_trade_callback(self.on_trade)
            self.adapters.append(bybit)
            logger.info("Initialized Bybit Adapter")

        # Connect and Subscribe
        for adapter in self.adapters:
            try:
                # Wire up callbacks
                adapter.set_trade_callback(self.on_trade)
                if hasattr(adapter, "set_balance_callback"):
                    adapter.set_balance_callback(self.on_balance)

                await adapter.connect()

                # Initial Balance Fetch
                if hasattr(adapter, "get_balances"):
                    try:
                        await adapter.get_balances()
                        logger.info(f"Fetched initial balances for {adapter.name}")
                    except Exception as e:
                        logger.error(
                            f"Failed to fetch initial balances for {adapter.name}: {e}"
                        )

                # Determine symbols
                # For now hardcoded or from env.
                symbols = os.getenv("MARKET_SYMBOLS", "BTC-PERPETUAL,BTCUSDT").split(
                    ","
                )

                for sym in symbols:
                    sym = sym.strip()
                    if not sym:
                        continue

                    # Try subscribing. Adapters should handle symbol mapping internally.
                    # Thalex uses BTC-PERPETUAL, Bybit uses BTCUSDT.
                    # We might need to be smart about which symbol goes to which adapter.
                    # Simple heuristic:
                    if adapter.name == "thalex" and "PERPETUAL" in sym:
                        await adapter.subscribe_ticker(
                            sym
                        )  # Ticker usually subscribes to trades too in ThalexAdapter
                    elif adapter.name == "bybit" and "USDT" in sym:
                        await adapter.subscribe_trades(sym)
                    elif (
                        adapter.name == "bybit" and "PERP" in sym
                    ):  # Bybit also has USDC perps
                        pass

                logger.info(f"Started {adapter.name}")
            except Exception as e:
                logger.error(f"Failed to start adapter {adapter.name}: {e}")

        # Adapters run in background
        logger.info("MarketFeedService started.")

        # Warmup Signal Engine
        if self.or_engine:
            asyncio.create_task(self._warmup_engine())

        self.loop_task = asyncio.create_task(self._run_loop())

    async def _warmup_engine(self):
        """Load historical data to prime the engine."""
        try:
            # Since MarketFeedService has self.storage (TimescaleDBAdapter), we can use it.
            # Fetch last 48h to ensure we capture at least one full session even with gaps
            end = time.time()
            start = end - 172800

            # Which symbol? iterating all enabled.
            # For now hardcoded or primary
            symbol = os.getenv("TRADING_SYMBOL", "BTC-PERPETUAL")

            # logger.info(f"Fetching history for {symbol} warmup...")
            history = await self.storage.get_history(
                symbol, start, end, resolution="1m"
            )
            if not history:
                logger.warning(f"No history found for {symbol} warmup.")
                return

            logger.info(
                f"Warming up OR Engine with {len(history)} candles for {symbol}"
            )

            for candle in history:
                # Ensure we have floats and timestamp
                # TimescaleDB usually returns dict with 'time' as datetime or valid string
                ts = (
                    candle["time"].timestamp()
                    if hasattr(candle["time"], "timestamp")
                    else candle["time"]
                )

                # If ts is datetime string, we might need parsing, but adapter should handle it.
                # Assuming adapter returns proper python objects.

                self.or_engine.update_candle(
                    symbol=symbol,
                    timestamp=float(ts),
                    open_=float(candle["open"]),
                    high=float(candle["high"]),
                    low=float(candle["low"]),
                    close=float(candle["close"]),
                )

            logger.info(
                f"Warmup complete. ORB Levels: {self.or_engine.get_chart_levels(symbol)}"
            )

        except Exception as e:
            logger.error(f"Warmup failed: {e}")

    async def _run_loop(self):
        while self.running:
            await asyncio.sleep(1)

    async def stop(self):
        self.running = False
        if hasattr(self, "loop_task"):
            self.loop_task.cancel()
            try:
                await self.loop_task
            except asyncio.CancelledError:
                pass

        logger.info("Stopping MarketFeedService...")
        for adapter in self.adapters:
            await adapter.disconnect()
        await self.storage.disconnect()

    async def on_trade(self, trade: Trade):
        # Persist to DB
        try:
            # logger.debug(f"Saving trade {trade}")
            asyncio.create_task(self.storage.save_trade(trade))

            # Update Signal Engine
            if self.or_engine:
                self.or_engine.update_trade(trade)

        except Exception as e:
            logger.error(f"Failed to save trade: {e}")

    async def on_balance(self, balance: Balance):
        try:
            logger.info(f"Saving balance update: {balance}")
            asyncio.create_task(self.storage.save_balance(balance))
        except Exception as e:
            logger.error(f"Failed to save balance: {e}")


async def main():
    # Setup Logging
    logging.basicConfig(level=logging.INFO)

    # Load Env
    from dotenv import load_dotenv

    load_dotenv()

    db_user = os.getenv("DATABASE_USER", "postgres")
    db_pass = os.getenv("DATABASE_PASSWORD", "password")
    db_host = os.getenv("DATABASE_HOST", "localhost")
    db_port = os.getenv("DATABASE_PORT", "5433")
    db_name = os.getenv("DATABASE_NAME", "thalex_trading")
    db_dsn = f"postgresql://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}"

    service = MarketFeedService(db_dsn)

    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()

    def signal_handler():
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, signal_handler)

    task = asyncio.create_task(service.start())

    await stop_event.wait()
    await service.stop()
    task.cancel()


if __name__ == "__main__":
    asyncio.run(main())
