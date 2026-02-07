import asyncio
import logging
import os
import signal
import time
import json
from typing import List, Dict, Optional

from ..adapters.storage.timescale_adapter import TimescaleDBAdapter
from ..adapters.exchanges.bybit_adapter import BybitAdapter
from ..adapters.exchanges.thalex_adapter import ThalexAdapter
from ..domain.entities import Trade
from ..domain.signals.open_range import OpenRangeSignalEngine

logger = logging.getLogger("DataIngestor")


class DataIngestionService:
    def __init__(self, db_url: str, or_engine: Optional[OpenRangeSignalEngine] = None):
        self.db_url = db_url
        self.storage = TimescaleDBAdapter(db_url)
        self.or_engine = or_engine
        self.adapters: List = []
        self.running = False
        self.tasks: List[asyncio.Task] = []
        self.config = self._load_json_config()

    def _load_json_config(self) -> Dict:
        """Load data_ingestion config from config.json"""
        try:
            config_path = os.path.join(os.getcwd(), "config.json")
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    return json.load(f)
            return {}
        except Exception as e:
            logger.error(f"Failed to load config.json: {e}")
            return {}

    async def start(self):
        self.running = True
        logger.info("Starting DataIngestionService...")

        # Connect to DB
        try:
            await self.storage.connect()
            logger.info("Connected to TimescaleDB")
        except Exception as e:
            logger.error(f"Failed to connect to DB: {e}")
            return

        ingestion_config = self.config.get("data_ingestion", {})
        if not ingestion_config.get("enabled", False):
            logger.warning("Data Ingestion is DISABLED in config.")
            return

        sources = ingestion_config.get("sources", [])

        # 1. Initialize Adapters
        # Note: We reuse existing adapters. Ideally, we should support public-only connections.
        # For now, we grab keys from ENV to ensure connection succeeds.

        # Thalex
        thalex_source = next((s for s in sources if s["venue"] == "thalex"), None)
        if thalex_source:
            thalex_key = os.getenv("THALEX_PROD_API_KEY_ID") or os.getenv(
                "THALEX_KEY_ID"
            )
            thalex_secret = os.getenv("THALEX_PROD_PRIVATE_KEY") or os.getenv(
                "THALEX_PRIVATE_KEY"
            )
            is_testnet = os.getenv("TRADING_MODE", "testnet").lower() == "testnet"

            if thalex_key and thalex_secret:
                try:
                    thalex = ThalexAdapter(
                        thalex_key, thalex_secret, testnet=is_testnet
                    )
                    thalex.set_trade_callback(self.on_trade)
                    await thalex.connect()
                    self.adapters.append(thalex)

                    # Subscribe
                    for sym in thalex_source.get("symbols", []):
                        await thalex.subscribe_ticker(sym)
                        logger.info(f"Subscribed to {sym} on Thalex")
                except Exception as e:
                    logger.error(f"Failed to init Thalex ingestion: {e}")

        # Bybit
        bybit_source = next((s for s in sources if s["venue"] == "bybit"), None)
        if bybit_source:
            bybit_key = os.getenv("BYBIT_API_KEY", "")
            bybit_secret = os.getenv("BYBIT_API_SECRET", "")
            is_testnet = os.getenv("TRADING_MODE", "testnet").lower() == "testnet"

            if bybit_key:
                try:
                    bybit = BybitAdapter(bybit_key, bybit_secret, testnet=is_testnet)
                    bybit.set_trade_callback(self.on_trade)
                    await bybit.connect()
                    self.adapters.append(bybit)

                    # Subscribe
                    for sym in bybit_source.get("symbols", []):
                        # Simple Mapping for Bybit topics
                        await bybit.subscribe_trades(sym)
                        logger.info(f"Subscribed to {sym} on Bybit")
                except Exception as e:
                    logger.error(f"Failed to init Bybit ingestion: {e}")

        # Warmup Signal Engine
        if self.or_engine:
            asyncio.create_task(self._warmup_engine())

        self.loop_task = asyncio.create_task(self._run_loop())
        logger.info(f"DataIngestionService running with {len(self.adapters)} adapters.")

    async def _warmup_engine(self):
        """Load historical data to prime the engine."""
        try:
            # Fetch last 48h to ensure we capture at least one full session even with gaps
            end = time.time()
            start = end - 172800
            # For warmup, we might want to iterate ALL symbols in the watchlist?
            # Or just the primary one for the ORB engine.
            # ORB Engine currently assumes single symbol or manages multiple?
            # It manages multiple states by symbol.

            ingestion_config = self.config.get("data_ingestion", {})
            sources = ingestion_config.get("sources", [])

            all_symbols = []
            for s in sources:
                all_symbols.extend(s.get("symbols", []))

            for symbol in all_symbols:
                # Map specific symbols if needed (e.g. BTC-PERPETUAL)
                # But get_history handles "all" or specific?
                # We need specific history for each symbol.

                # Check config or adapter for venue?
                # We can try fetching history for the symbol.
                # get_history signature: symbol, start, end, resolution, exchange
                # We should probably pass exchange='all' or let adapter figure it out?
                # or pass specific venue if we know it.

                # Simple approach: pass exchange='all' to leverage universal mapping/aggregation
                # or pass None/Null.
                if "USDT" in symbol:
                    # Bybit symbols
                    exchange = "bybit"
                else:
                    exchange = "thalex"  # Default

                history = await self.storage.get_history(
                    symbol, start, end, resolution="1m", exchange=exchange
                )

                if not history and "PERPETUAL" in symbol:
                    # Try All if specific failed
                    history = await self.storage.get_history(
                        symbol, start, end, resolution="1m", exchange="all"
                    )

                if not history:
                    # logger.warning(f"No history found for {symbol} warmup.")
                    continue

                for candle in history:
                    ts = (
                        candle["time"].timestamp()
                        if hasattr(candle["time"], "timestamp")
                        else candle["time"]
                    )
                    self.or_engine.update_candle(
                        symbol=symbol,
                        timestamp=float(ts),
                        open_=float(candle["open"]),
                        high=float(candle["high"]),
                        low=float(candle["low"]),
                        close=float(candle["close"]),
                    )
                logger.info(f"Warmed up {symbol} with {len(history)} candles.")

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

        logger.info("Stopping DataIngestionService...")
        for adapter in self.adapters:
            await adapter.disconnect()
        await self.storage.disconnect()

    async def on_trade(self, trade: Trade):
        # Persist to DB
        try:
            asyncio.create_task(self.storage.save_trade(trade))

            # Update Signal Engine
            if self.or_engine:
                self.or_engine.update_trade(trade)

        except Exception as e:
            logger.error(f"Failed to process trade: {e}")

    async def on_balance(self, balance):
        # Persist Balance
        try:
            logger.info(f"Saving balance update: {balance}")
            asyncio.create_task(self.storage.save_balance(balance))
        except Exception as e:
            logger.error(f"Failed to process balance: {e}")
