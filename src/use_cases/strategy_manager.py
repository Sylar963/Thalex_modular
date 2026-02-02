import asyncio
import logging
from typing import List, Dict, Optional, Callable
from dataclasses import dataclass

from ..domain.interfaces import ExchangeGateway, Strategy, RiskManager
from ..domain.tracking.sync_engine import SyncEngine, GlobalState
from ..domain.entities import Ticker, Trade, Position, Order

logger = logging.getLogger(__name__)


@dataclass
class ExchangeConfig:
    gateway: ExchangeGateway
    symbol: str
    enabled: bool = True


class MultiExchangeStrategyManager:
    def __init__(
        self,
        exchanges: List[ExchangeConfig],
        strategy: Strategy,
        risk_manager: RiskManager,
        sync_engine: SyncEngine,
    ):
        self.exchanges = {cfg.gateway.name: cfg for cfg in exchanges}
        self.strategy = strategy
        self.risk_manager = risk_manager
        self.sync_engine = sync_engine

        self._running = False
        self._tasks: List[asyncio.Task] = []

        self.sync_engine.on_state_change = self._on_global_state_change

    async def start(self):
        logger.info("Starting MultiExchangeStrategyManager...")
        self._running = True

        for name, cfg in self.exchanges.items():
            if cfg.enabled:
                await self._connect_exchange(cfg)

        logger.info(f"Started with {len(self.exchanges)} exchanges")

    async def stop(self):
        logger.info("Stopping MultiExchangeStrategyManager...")
        self._running = False

        for task in self._tasks:
            task.cancel()

        for name, cfg in self.exchanges.items():
            try:
                await cfg.gateway.disconnect()
            except Exception as e:
                logger.error(f"Error disconnecting {name}: {e}")

        logger.info("MultiExchangeStrategyManager stopped")

    async def _connect_exchange(self, cfg: ExchangeConfig):
        gw = cfg.gateway
        gw.set_ticker_callback(self._make_ticker_callback(gw.name))
        gw.set_trade_callback(self._make_trade_callback(gw.name))
        gw.set_position_callback(self._make_position_callback(gw.name))

        await gw.connect()
        await gw.subscribe_ticker(cfg.symbol)

        logger.info(f"Connected to {gw.name} for {cfg.symbol}")

    def _make_ticker_callback(self, exchange: str) -> Callable:
        async def callback(ticker: Ticker):
            await self.sync_engine.update_ticker(exchange, ticker.symbol, ticker)

        return callback

    def _make_trade_callback(self, exchange: str) -> Callable:
        async def callback(trade: Trade):
            logger.debug(
                f"[{exchange}] Trade: {trade.side.value} {trade.size} @ {trade.price}"
            )

        return callback

    def _make_position_callback(self, exchange: str) -> Callable:
        async def callback(symbol: str, size: float, entry_price: float):
            position = Position(
                symbol=symbol, size=size, entry_price=entry_price, exchange=exchange
            )
            await self.sync_engine.update_position(exchange, symbol, position)

        return callback

    def _on_global_state_change(self, state: GlobalState):
        net_position = state.net_position
        logger.info(f"Global Net Position: {net_position:.4f}")

        arb = self.sync_engine.get_arb_opportunity("BTCUSDT")
        if arb:
            logger.warning(
                f"Arbitrage opportunity detected: Buy on {arb['buy_exchange']}, Sell on {arb['sell_exchange']}, Spread: {arb['spread']:.2f}"
            )

    def get_state_snapshot(self) -> Dict:
        return {
            "net_position": self.sync_engine.state.net_position,
            "global_best_bid": self.sync_engine.state.global_best_bid,
            "global_best_ask": self.sync_engine.state.global_best_ask,
            "positions": {
                k: {"symbol": p.symbol, "size": p.size, "exchange": p.exchange}
                for k, p in self.sync_engine.state.positions.items()
            },
            "tickers": {
                k: {"bid": t.bid, "ask": t.ask, "exchange": t.exchange}
                for k, t in self.sync_engine.state.tickers.items()
            },
        }
