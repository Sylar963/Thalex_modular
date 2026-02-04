import asyncio
import logging
from typing import List, Dict, Optional, Callable
from dataclasses import dataclass, replace

from ..domain.interfaces import (
    ExchangeGateway,
    Strategy,
    RiskManager,
    StorageGateway,
    SignalEngine,
)
from ..domain.tracking.sync_engine import SyncEngine, GlobalState
from ..domain.tracking.state_tracker import StateTracker
from ..domain.entities import (
    Ticker,
    Trade,
    Position,
    Order,
    OrderSide,
    OrderStatus,
    Portfolio,
    MarketState,
)

logger = logging.getLogger(__name__)


@dataclass
class ExchangeConfig:
    gateway: ExchangeGateway
    symbol: str
    enabled: bool = True
    tick_size: float = 0.5
    strategy: Optional[Strategy] = None


class VenueContext:
    def __init__(self, config: ExchangeConfig):
        self.config = config
        self.state_tracker = StateTracker()
        self.market_state = MarketState()
        self.last_mid_price = 0.0


class MultiExchangeStrategyManager:
    def __init__(
        self,
        exchanges: List[ExchangeConfig],
        strategy: Strategy,
        risk_manager: RiskManager,
        sync_engine: SyncEngine,
        signal_engine: Optional[SignalEngine] = None,
        storage: Optional[StorageGateway] = None,
        dry_run: bool = False,
    ):
        self.venues: Dict[str, VenueContext] = {}
        for cfg in exchanges:
            self.venues[cfg.gateway.name] = VenueContext(cfg)

        self.strategy = strategy
        self.risk_manager = risk_manager
        self.sync_engine = sync_engine
        self.signal_engine = signal_engine
        self.storage = storage
        self.dry_run = dry_run

        self.portfolio = Portfolio()
        self._running = False
        self._tasks: List[asyncio.Task] = []
        self._reconcile_lock = asyncio.Lock()
        self.min_edge_threshold = 0.5

        self.sync_engine.on_state_change = self._on_global_state_change

    async def start(self):
        logger.info("Starting MultiExchangeStrategyManager...")
        self._running = True

        for name, venue in self.venues.items():
            if venue.config.enabled:
                await venue.state_tracker.start()
                await self._connect_exchange(venue)

        logger.info(
            f"Started with {len([v for v in self.venues.values() if v.config.enabled])} venues"
        )

    async def stop(self):
        logger.info("Stopping MultiExchangeStrategyManager...")
        self._running = False

        for task in self._tasks:
            task.cancel()

        for name, venue in self.venues.items():
            try:
                await self._cancel_all_venue(venue)
                await venue.config.gateway.disconnect()
            except Exception as e:
                logger.error(f"Error disconnecting {name}: {e}")

        logger.info("MultiExchangeStrategyManager stopped")

    async def _connect_exchange(self, venue: VenueContext):
        cfg = venue.config
        gw = cfg.gateway

        gw.set_ticker_callback(self._make_ticker_callback(gw.name))
        gw.set_trade_callback(self._make_trade_callback(gw.name))
        gw.set_order_callback(venue.state_tracker.on_order_update)
        gw.set_position_callback(self._make_position_callback(gw.name))

        await gw.connect()

        if hasattr(gw, "cancel_all_orders"):
            await gw.cancel_all_orders(cfg.symbol)
            venue.state_tracker.pending_orders.clear()
            venue.state_tracker.confirmed_orders.clear()

        try:
            initial_pos = await gw.get_position(cfg.symbol)
            await venue.state_tracker.update_position(
                cfg.symbol, initial_pos.size, initial_pos.entry_price
            )
            pos = Position(
                symbol=cfg.symbol,
                size=initial_pos.size,
                entry_price=initial_pos.entry_price,
                exchange=gw.name,
            )
            self.portfolio.set_position(pos)
            if self.storage:
                await self.storage.save_position(pos)
        except Exception as e:
            logger.warning(f"Could not fetch initial position for {gw.name}: {e}")

        await gw.subscribe_ticker(cfg.symbol)
        logger.info(f"Connected to {gw.name} for {cfg.symbol}")

    def _make_ticker_callback(self, exchange: str) -> Callable:
        async def callback(ticker: Ticker):
            if not self._running:
                return

            venue = self.venues.get(exchange)
            if not venue:
                return

            ticker = replace(ticker, exchange=exchange)
            venue.market_state = MarketState(ticker=ticker, timestamp=ticker.timestamp)

            await self.sync_engine.update_ticker(exchange, ticker.symbol, ticker)

            if self.storage and not self.dry_run:
                asyncio.create_task(self.storage.save_ticker(ticker))

            await self._run_strategy_for_venue(venue)

        return callback

    def _make_trade_callback(self, exchange: str) -> Callable:
        async def callback(trade: Trade):
            if self.signal_engine:
                self.signal_engine.update_trade(trade)
            logger.debug(
                f"[{exchange}] Trade: {trade.side.value} {trade.size} @ {trade.price}"
            )

        return callback

    def _make_position_callback(self, exchange: str) -> Callable:
        async def callback(symbol: str, size: float, entry_price: float):
            venue = self.venues.get(exchange)
            if venue:
                await venue.state_tracker.update_position(symbol, size, entry_price)

            position = Position(
                symbol=symbol, size=size, entry_price=entry_price, exchange=exchange
            )
            self.portfolio.set_position(position)
            await self.sync_engine.update_position(exchange, symbol, position)

            if self.storage and not self.dry_run:
                asyncio.create_task(self.storage.save_position(position))

        return callback

    async def _run_strategy_for_venue(self, venue: VenueContext):
        if self._reconcile_lock.locked():
            return

        if not venue.market_state.ticker:
            return

        current_mid = venue.market_state.ticker.mid_price
        tick_size = venue.config.tick_size
        min_edge = tick_size * self.min_edge_threshold

        if (
            venue.last_mid_price > 0
            and abs(current_mid - venue.last_mid_price) < min_edge
        ):
            return

        venue.last_mid_price = current_mid

        async with self._reconcile_lock:
            exchange = venue.config.gateway.name
            position = self.portfolio.get_position(venue.config.symbol, exchange)

            if self.signal_engine:
                venue.market_state.signals.update(self.signal_engine.get_signals())

            active_strategy = (
                venue.config.strategy if venue.config.strategy else self.strategy
            )

            desired_orders = active_strategy.calculate_quotes(
                venue.market_state, position, tick_size=tick_size
            )

            for i in range(len(desired_orders)):
                o = desired_orders[i]
                if o.price:
                    rounded = round(o.price / tick_size) * tick_size
                    desired_orders[i] = replace(o, price=rounded, exchange=exchange)

            valid_orders = []
            for o in desired_orders:
                if self.risk_manager.validate_order(
                    o, self.portfolio, active_orders=valid_orders
                ):
                    valid_orders.append(o)
                else:
                    logger.warning(f"Risk rejected order on {exchange}: {o}")

            await self._reconcile_orders_venue(venue, valid_orders)

    async def _reconcile_orders_venue(
        self, venue: VenueContext, desired_orders: List[Order]
    ):
        desired_by_side = {OrderSide.BUY: [], OrderSide.SELL: []}
        for o in desired_orders:
            desired_by_side[o.side].append(o)

        to_cancel_ids = []
        to_place_orders = []
        tick_size = venue.config.tick_size

        for side in [OrderSide.BUY, OrderSide.SELL]:
            desired_list = desired_by_side.get(side, [])
            active_list = [
                t.order for t in venue.state_tracker.get_open_orders(side=side)
            ]

            active_map = [{"order": act, "matched": False} for act in active_list]

            for des in desired_list:
                found = False
                for item in active_map:
                    if item["matched"]:
                        continue
                    act = item["order"]
                    price_diff = abs(des.price - act.price)
                    size_diff = abs(des.size - act.size)
                    if price_diff < 1e-9 and size_diff < 1e-9:
                        item["matched"] = True
                        found = True
                        break
                    if price_diff <= (1.1 * tick_size) and size_diff < 1e-9:
                        item["matched"] = True
                        found = True
                        break
                if not found:
                    to_place_orders.append(des)

            for item in active_map:
                if not item["matched"] and item["order"].exchange_id:
                    to_cancel_ids.append(item["order"].exchange_id)

        gw = venue.config.gateway
        exchange = gw.name

        if to_cancel_ids or to_place_orders:
            logger.info(
                f"[{exchange}] Reconcile: Cancel {len(to_cancel_ids)}, Place {len(to_place_orders)}"
            )

        if to_cancel_ids:
            results = await gw.cancel_orders_batch(to_cancel_ids)
            for oid, success in zip(to_cancel_ids, results):
                if success:
                    await venue.state_tracker.on_order_cancel(oid)

        if to_place_orders:
            new_orders = await gw.place_orders_batch(to_place_orders)
            for o in new_orders:
                if o.status == OrderStatus.OPEN and o.exchange_id:
                    await venue.state_tracker.submit_order(o)
                    await venue.state_tracker.on_order_ack(o.id, o.exchange_id)

    async def _cancel_all_venue(self, venue: VenueContext):
        all_active = venue.state_tracker.get_open_orders()
        if not all_active:
            return
        tasks = [
            venue.config.gateway.cancel_order(t.order.exchange_id)
            for t in all_active
            if t.order.exchange_id
        ]
        await asyncio.gather(*tasks, return_exceptions=True)

    def _on_global_state_change(self, state: GlobalState):
        net_position = state.net_position
        logger.debug(f"Global Net Position: {net_position:.4f}")

        arb = self.sync_engine.get_arb_opportunity("BTCUSDT")
        if arb:
            logger.warning(
                f"Arb detected: Buy {arb['buy_exchange']}, Sell {arb['sell_exchange']}, Spread: {arb['spread']:.2f}"
            )

    def get_portfolio(self) -> Portfolio:
        return self.portfolio

    def get_state_snapshot(self) -> Dict:
        return {
            "net_position": self.sync_engine.state.net_position,
            "global_best_bid": self.sync_engine.state.global_best_bid,
            "global_best_ask": self.sync_engine.state.global_best_ask,
            "portfolio": {
                k: {"symbol": p.symbol, "size": p.size, "exchange": p.exchange}
                for k, p in self.portfolio.positions.items()
            },
            "tickers": {
                k: {"bid": t.bid, "ask": t.ask, "exchange": t.exchange}
                for k, t in self.sync_engine.state.tickers.items()
            },
        }
