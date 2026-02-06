import asyncio
import logging
from typing import List, Dict, Callable
from dataclasses import replace

from ..domain.interfaces import (
    ExchangeGateway,
    StorageGateway,
)
from ..domain.tracking.state_tracker import StateTracker
from ..domain.entities import (
    Ticker,
    Trade,
    Position,
    MarketState,
)
from src.use_cases.strategy_manager import ExchangeConfig, VenueContext

logger = logging.getLogger(__name__)


class DataStreamManager:
    """
    Manages connections to multiple exchanges purely for data ingestion.
    Does NOT execute any trading strategies or risk checks.
    """

    def __init__(
        self,
        exchanges: List[ExchangeConfig],
        storage: StorageGateway,
    ):
        self.venues: Dict[str, VenueContext] = {}
        for cfg in exchanges:
            # We enforce enabled=True context for data streaming if passed here
            # But we rely on the caller to have set cfg.enabled if they want it
            self.venues[cfg.gateway.name] = VenueContext(cfg)

        self.storage = storage
        self._running = False
        self._tasks: List[asyncio.Task] = []

    async def start(self):
        logger.info("Starting DataStreamManager (Monitoring Mode)...")
        self._running = True

        for name, venue in self.venues.items():
            if venue.config.enabled:
                logger.info(f"Connecting to {name}...")
                await self._connect_exchange(venue)
            else:
                logger.info(f"Skipping {name} (disabled in config)")

        logger.info("DataStreamManager started.")

    async def stop(self):
        logger.info("Stopping DataStreamManager...")
        self._running = False

        for name, venue in self.venues.items():
            try:
                if venue.config.enabled:
                    await venue.config.gateway.disconnect()
            except Exception as e:
                logger.error(f"Error disconnecting {name}: {e}")

    async def _connect_exchange(self, venue: VenueContext):
        cfg = venue.config
        gw = cfg.gateway

        # Set Callbacks - Direct to Storage
        gw.set_ticker_callback(self._make_ticker_callback(gw.name))
        gw.set_trade_callback(self._make_trade_callback(gw.name))
        gw.set_position_callback(self._make_position_callback(gw.name))
        gw.set_balance_callback(self._handle_balance_update)

        # We also listen to orders to update local state, though valid for 'monitor' mode only if
        # the user is trading manually or via another bot instance.
        # But generally useful to have order history.
        gw.set_order_callback(self._make_order_callback(gw.name))

        await gw.connect()

        # Initial Fetch
        try:
            if hasattr(gw, "get_balances"):
                await gw.get_balances()
        except Exception as e:
            logger.warning(f"Failed to fetch initial balances for {gw.name}: {e}")

        if hasattr(gw, "fetch_instrument_info"):
            await gw.fetch_instrument_info(cfg.symbol)

        # Initial Position
        try:
            initial_pos = await gw.get_position(cfg.symbol)
            await self._persist_position(initial_pos, gw.name)
        except Exception as e:
            logger.warning(f"Could not fetch initial position for {gw.name}: {e}")

        await gw.subscribe_ticker(cfg.symbol)
        logger.info(f"Connected to {gw.name} for {cfg.symbol}")

    def _make_ticker_callback(self, exchange: str) -> Callable:
        async def callback(ticker: Ticker):
            if not self._running:
                return

            ticker = replace(ticker, exchange=exchange)
            if self.storage:
                # Fire and forget persistence
                asyncio.create_task(self.storage.save_ticker(ticker))

        return callback

    def _make_trade_callback(self, exchange: str) -> Callable:
        async def callback(trade: Trade):
            trade = replace(trade, exchange=exchange)
            if self.storage:
                asyncio.create_task(self.storage.save_trade(trade))

        return callback

    def _make_position_callback(self, exchange: str) -> Callable:
        async def callback(symbol: str, size: float, entry_price: float):
            pos = Position(
                symbol=symbol, size=size, entry_price=entry_price, exchange=exchange
            )
            await self._persist_position(pos, exchange)

        return callback

    def _make_order_callback(self, exchange: str) -> Callable:
        async def callback(order_id, status, filled_size, avg_price):
            # In pure monitor mode, simple order updates might not have the full Order object context
            # if we didn't place them.
            # If the adapter sends full order info, great. If not, we might be limited.
            # Most persistence of orders happens via `save_execution` or similar,
            # but usually we need the full Order object.
            # For now, let's leave this blank unless we want to track 'shadow' orders or query open orders periodically.
            pass

        return callback

    async def _handle_balance_update(self, balance):
        if self.storage:
            await self.storage.save_balance(balance)

    async def _persist_position(self, pos: Position, exchange: str):
        if self.storage:
            asyncio.create_task(self.storage.save_position(pos))
