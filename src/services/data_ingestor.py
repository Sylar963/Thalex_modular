import asyncio
import logging
import os
import signal
import time

try:
    import orjson
except ImportError:
    import json as orjson
from typing import List, Dict, Optional

from ..adapters.storage.timescale_adapter import TimescaleDBAdapter
from ..adapters.exchanges.bybit_adapter import BybitAdapter
from ..adapters.exchanges.thalex_adapter import ThalexAdapter
from ..domain.entities import Trade, Ticker
from ..domain.signals.open_range import OpenRangeSignalEngine
from ..use_cases.sim_state_manager import sim_state_manager
from ..domain.services.market_validator import MarketDataValidator

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
        self.validator = MarketDataValidator()

    def _load_json_config(self) -> Dict:
        """Load data_ingestion config from config.json"""
        try:
            # Robust path resolution relative to project root
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            config_path = os.path.join(base_dir, "config.json")
            
            if os.path.exists(config_path):
                with open(config_path, "rb") as f:
                    return orjson.loads(f.read())
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

            # Inject Storage into SimStateManager for persistence
            sim_state_manager.set_storage(self.storage)

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
                    thalex.set_ticker_callback(self.on_ticker)
                    thalex.set_balance_callback(self.on_balance)
                    thalex.set_position_callback(self.on_position)
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
                    bybit.set_ticker_callback(self.on_ticker)
                    bybit.set_balance_callback(self.on_balance)
                    bybit.set_position_callback(self.on_position)
                    await bybit.connect()
                    self.adapters.append(bybit)

                    # Subscribe
                    for sym in bybit_source.get("symbols", []):
                        # Subscribe to tickers as well for simulation
                        await bybit.subscribe_ticker(sym)
                        await bybit.subscribe_trades(sym)
                        logger.info(f"Subscribed to {sym} on Bybit (Ticker+Trades)")
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
        # Validate first
        valid_trade = self.validator.validate_trade(trade)
        if not valid_trade:
            return

        # Persist to DB
        try:
            asyncio.create_task(self.storage.save_trade(valid_trade))

            # Update Signal Engine
            if self.or_engine:
                self.or_engine.update_trade(valid_trade)

        except Exception as e:
            logger.error(f"Failed to process trade: {e}")

    async def on_balance(self, balance):
        # Persist Balance
        try:
            logger.info(f"Saving balance update: {balance}")
            asyncio.create_task(self.storage.save_balance(balance))
        except Exception as e:
            logger.error(f"Failed to process balance: {e}")

    async def on_position(self, symbol: str, size: float, entry_price: float):
        # Persist Position
        try:
            from ..domain.entities import Position
            # We try to get the full position from adapter if possible, but basic is enough for snapshot
            # Actually, most adapters store the full Position in self.positions
            # But the callback only sends symbol, size, entry_price for simplicity.
            
            # Find the adapter that sent this
            position = None
            for adapter in self.adapters:
                if hasattr(adapter, "positions") and symbol in adapter.positions:
                    position = adapter.positions[symbol]
                    break
            
            if not position:
                position = Position(symbol=symbol, size=size, entry_price=entry_price)
                
            asyncio.create_task(self.storage.save_position(position))
            logger.info(f"Saved position update: {symbol} {size} @ {entry_price}")
        except Exception as e:
            logger.error(f"Failed to process position: {e}")

    async def on_ticker(self, ticker: Ticker):
        # Validate first
        valid_ticker = self.validator.validate_ticker(ticker)
        if not valid_ticker:
            return

        # 1. Persist Ticker
        try:
            # We can optionally sample decimation here if needed
            asyncio.create_task(self.storage.save_ticker(valid_ticker))
        except Exception as e:
            logger.error(f"Failed to save ticker: {e}")

        # 2. Forward to Live Simulation
        try:
            await sim_state_manager.on_ticker(valid_ticker)
        except Exception as e:
            logger.error(f"Failed to update Live Sim with ticker: {e}")

    async def sync_symbol(self, symbol: str, venue: str = "thalex"):
        """Manual sync of recent data for a symbol (gap-fill)."""
        logger.info(f"Syncing {symbol} from {venue}...")
        
        adapter = None
        if venue.lower() == "thalex":
             thalex_key = os.getenv("THALEX_PROD_API_KEY_ID") or os.getenv("THALEX_KEY_ID")
             thalex_secret = os.getenv("THALEX_PROD_PRIVATE_KEY") or os.getenv("THALEX_PRIVATE_KEY")
             is_testnet = os.getenv("TRADING_MODE", "testnet").lower() == "testnet"
             if thalex_key and thalex_secret:
                 adapter = ThalexAdapter(thalex_key, thalex_secret, testnet=is_testnet)
             else:
                 # Fallback to public if no keys? ThalexAdapter requires keys for connect/login currently.
                 # For sync, we might only need public. But adapter.connect() does login.
                 logger.warning("Thalex keys missing, sync might fail if adapter requires auth.")

        elif venue.lower() == "bybit":
             bybit_key = os.getenv("BYBIT_API_KEY", "")
             bybit_secret = os.getenv("BYBIT_API_SECRET", "")
             is_testnet = os.getenv("TRADING_MODE", "testnet").lower() == "testnet"
             adapter = BybitAdapter(bybit_key, bybit_secret, testnet=is_testnet)

        if not adapter:
            raise ValueError(f"No valid adapter/keys for venue {venue}")

        try:
            await adapter.connect()
            # Fetch last 1000 trades to cover the ORB window (usually 15-30m)
            trades = await adapter.get_recent_trades(symbol, limit=1000)
            if trades:
                logger.info(f"Fetched {len(trades)} recent trades for {symbol}")
                # Save trades to DB
                for t in trades:
                    await self.storage.save_trade(t)
                    # Prime signal engine if active
                    if self.or_engine:
                        self.or_engine.update_trade(t)
            else:
                logger.warning(f"No recent trades returned for {symbol} on {venue}")
        finally:
            await adapter.disconnect()
