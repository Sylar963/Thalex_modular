import asyncio
import logging
import time
from typing import List, Dict, Optional, Callable
import json
from dataclasses import dataclass, replace

from ..domain.interfaces import (
    ExchangeGateway,
    Strategy,
    RiskManager,
    StorageGateway,
    SignalEngine,
)
from ..domain.market.trend_service import HistoricalTrendService
from ..domain.tracking.sync_engine import SyncEngine, GlobalState
from ..domain.tracking.state_tracker import StateTracker
from ..domain.entities import (
    Ticker,
    Trade,
    Position,
    Order,
    OrderType,
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
        or_engine: Optional[SignalEngine] = None,
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
        self.or_engine = or_engine
        self.storage = storage
        self.dry_run = dry_run

        self.trend_service = HistoricalTrendService(storage) if storage else None
        self._venue_trends: Dict[str, float] = {}  # symbol -> trend_value
        self._last_trend_update = 0.0
        self._last_status_persist: Dict[str, float] = {}

        self._momentum_adds: Dict[
            str, List[Dict]
        ] = {}  # venue_key (exchange:symbol) -> list of adds

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
        gw.set_balance_callback(self._handle_balance_update)

        await gw.connect()

        # Fetch initial balances
        try:
            if hasattr(gw, "get_balances"):
                await gw.get_balances()
        except Exception as e:
            logger.warning(f"Failed to fetch initial balances for {gw.name}: {e}")

        if hasattr(gw, "fetch_instrument_info"):
            await gw.fetch_instrument_info(cfg.symbol)
            cfg.tick_size = getattr(gw, "tick_size", cfg.tick_size)
            logger.info(f"Dynamic tick_size for {cfg.symbol}: {cfg.tick_size}")

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
            if self.risk_manager:
                self.risk_manager.update_position(pos)
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

            if self.signal_engine:
                self.signal_engine.update(ticker)

            if self.or_engine:
                self.or_engine.update(ticker)
                # Persist Open Range signals on session end OR breakout
                if (
                    hasattr(self.or_engine, "is_session_just_completed")
                    and self.or_engine.is_session_just_completed()
                ):
                    or_signals = self.or_engine.get_signals()
                    if self.storage and not self.dry_run:
                        asyncio.create_task(
                            self.storage.save_signal(
                                ticker.symbol, "open_range", or_signals
                            )
                        )

            await self.sync_engine.update_ticker(exchange, ticker.symbol, ticker)

            if self.storage and not self.dry_run:
                asyncio.create_task(self.storage.save_ticker(ticker))

            await self._run_strategy_for_venue(venue)

        return callback

    def _make_trade_callback(self, exchange: str) -> Callable:
        async def callback(trade: Trade):
            # Ensure trade has correct exchange context
            trade = replace(trade, exchange=exchange)

            if self.signal_engine:
                self.signal_engine.update_trade(trade)
                # Persist VAMP signals on candle completion
                if hasattr(self.signal_engine, "pop_completed_candle"):
                    completed = self.signal_engine.pop_completed_candle()
                    if completed and self.storage and not self.dry_run:
                        signals = self.signal_engine.get_signals()
                        asyncio.create_task(
                            self.storage.save_signal(trade.symbol, "vamp", signals)
                        )

            logger.debug(
                f"[{exchange}] Trade: {trade.side.value} {trade.size} @ {trade.price}"
            )

            if self.storage and not self.dry_run:
                asyncio.create_task(self.storage.save_trade(trade))

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

    async def _handle_balance_update(self, balance):
        await self.sync_engine.update_balance(balance)
        if self.storage:
            await self.storage.save_balance(balance)

    async def _update_trends_if_needed(self, symbol: str):
        if not self.trend_service:
            return

        now = time.time()
        if now - self._last_trend_update > 3600:  # Refresh once per hour
            trend = await self.trend_service.get_trend_14d(symbol)
            self._venue_trends[symbol] = trend
            self._last_trend_update = now

    async def _run_strategy_for_venue(self, venue: VenueContext):
        if self._reconcile_lock.locked():
            return

        # 1. Update Long-Term Trend
        await self._update_trends_if_needed(venue.config.symbol)

        # 1.5 Persist Bot Status (Every 1M approx, throttled by update_trends logic or separate timer)
        # For simplicity, we pigggyback on trend check or add explicit check
        now = time.time()
        venue_key = f"{venue.config.gateway.name}:{venue.config.symbol}"
        last_persist = self._last_status_persist.get(venue_key, 0.0)
        if now - last_persist > 60:
            await self._persist_bot_status(venue)
            self._last_status_persist[venue_key] = now

        # 2. Check for Risk Breach
        if self.risk_manager.has_breached():
            logger.critical("RISK BREACH DETECTED - ENTERING SMART REDUCING MODE")
            await self._run_smart_reducing_mode(venue)
            return

        if not venue.market_state.ticker:
            return

        current_mid = venue.market_state.ticker.mid_price
        tick_size = venue.config.tick_size
        min_edge = tick_size * self.min_edge_threshold

        # 3. Handle Momentum Sub-strategy (Adds and Exits)
        await self._manage_momentum_strategy(venue)

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
            "risk": self.risk_manager.get_risk_state(),
            "trends": self._venue_trends,
        }

    async def _run_smart_reducing_mode(self, venue: VenueContext):
        """
        Intelligently liquidates or manages positions during a risk breach.
        """
        async with self._reconcile_lock:
            exchange = venue.config.gateway.name
            symbol = venue.config.symbol
            position = self.portfolio.get_position(symbol, exchange)

            if abs(position.size) < 1e-9:
                return  # No position to reduce

            trend_val = self._venue_trends.get(symbol, 0.0)
            trend_side = (
                self.trend_service.get_trend_side(trend_val)
                if self.trend_service
                else "FLAT"
            )

            is_counter_trend = False
            if position.size > 0 and trend_side == "DOWN":
                is_counter_trend = True
            elif position.size < 0 and trend_side == "UP":
                is_counter_trend = True

            # Calculate Exit Quote
            ticker = venue.market_state.ticker
            if not ticker:
                return

            desired_orders = []
            if is_counter_trend:
                # AGGRESSIVE EXIT: Counter-trend trades are dangerous
                logger.warning(
                    f"Counter-trend position detected in breach ({trend_side}). Exiting aggressively."
                )
                price = ticker.bid if position.size > 0 else ticker.ask
                desired_orders.append(
                    Order(
                        id=f"exit_{int(time.time())}",
                        symbol=symbol,
                        side=OrderSide.SELL if position.size > 0 else OrderSide.BUY,
                        price=price,
                        size=abs(position.size),
                        type=OrderType.LIMIT,
                        exchange=exchange,
                    )
                )
            else:
                # DEFENSIVE EXIT: Trend-following trades get a chance to break-even
                logger.info(
                    f"Trend-following position in breach ({trend_side}). Setting break-even TP."
                )
                fee_bps = 5  # 0.05% approx
                break_even = position.entry_price * (
                    1 + (fee_bps / 10000 if position.size > 0 else -fee_bps / 10000)
                )

                # Ensure break_even is not worse than current market significantly?
                # For now, just use it.
                desired_orders.append(
                    Order(
                        id=f"tp_{int(time.time())}",
                        symbol=symbol,
                        side=OrderSide.SELL if position.size > 0 else OrderSide.BUY,
                        price=break_even,
                        size=abs(position.size),
                        type=OrderType.LIMIT,
                        exchange=exchange,
                    )
                )

            await self._reconcile_orders_venue(venue, desired_orders)

    async def _manage_momentum_strategy(self, venue: VenueContext):
        """
        Handles adding to momentum and managing time-based exits.
        """
        if not self.signal_engine or not self.or_engine:
            return

        exchange = venue.config.gateway.name
        symbol = venue.config.symbol
        venue_key = f"{exchange}:{symbol}"

        vamp_signals = self.signal_engine.get_signals()
        or_signals = self.or_engine.get_signals()
        trend_val = self._venue_trends.get(symbol, 0.0)
        trend_side = (
            self.trend_service.get_trend_side(trend_val)
            if self.trend_service
            else "FLAT"
        )

        vamp_impact = vamp_signals.get("market_impact", 0.0)
        or_mid = or_signals.get("orm", 0.0)
        current_price = venue.market_state.ticker.mid_price

        # 1. Check for Exits (10-minute cap)
        now = time.time()
        active_adds = self._momentum_adds.get(venue_key, [])
        remaining_adds = []
        for add in active_adds:
            should_exit = False
            # Time cap (10 mins)
            if now - add["timestamp"] > 600:
                logger.warning("Momentum ADD timed out (10 mins). Exiting.")
                should_exit = True
            # VAMP reversal
            elif (add["side"] == OrderSide.BUY and vamp_impact < 0.2) or (
                add["side"] == OrderSide.SELL and vamp_impact > -0.2
            ):
                logger.info("Momentum ADD signal reversal detected. Exiting.")
                should_exit = True

            if should_exit:
                await self._exit_momentum_add(venue, add)
            else:
                remaining_adds.append(add)
        self._momentum_adds[venue_key] = remaining_adds

        # 2. Check for New Adds
        if len(remaining_adds) >= 1:  # Max 1 add for now
            return

        add_side = None
        # Condition: Impact > 0.7 + Price > OR Mid + Trend matches VAMP
        if vamp_impact > 0.7 and current_price > or_mid and trend_side == "UP":
            add_side = OrderSide.BUY
        elif vamp_impact < -0.7 and current_price < or_mid and trend_side == "DOWN":
            add_side = OrderSide.SELL

        if add_side:
            logger.critical(
                f"MOMENTUM TRIGGERED: {add_side} Add on {exchange} {symbol}"
            )
            # Execute Market Add
            # We use a small fixed size for now or % of max
            add_size = 0.1  # Placeholder
            order = Order(
                id=f"mom_add_{int(now)}",
                symbol=symbol,
                side=add_side,
                price=0.0,  # Market
                size=add_size,
                type=OrderType.MARKET,
                exchange=exchange,
            )
            resp = await venue.config.gateway.place_order(order)
            if resp.status != OrderStatus.REJECTED:
                self._momentum_adds.setdefault(venue_key, []).append(
                    {
                        "side": add_side,
                        "size": add_size,
                        "timestamp": now,
                        "order_id": resp.exchange_id,
                    }
                )

    async def _exit_momentum_add(self, venue: VenueContext, add_info: Dict):
        """Executes a market order to exit a momentum addition."""
        logger.info(f"Exiting Momentum Add ({add_info['side']})")
        exit_order = Order(
            id=f"mom_exit_{int(time.time())}",
            symbol=venue.config.symbol,
            side=OrderSide.SELL if add_info["side"] == OrderSide.BUY else OrderSide.BUY,
            price=0.0,  # Market
            size=add_info["size"],
            type=OrderType.MARKET,
            exchange=venue.config.gateway.name,
        )
        if hasattr(venue.config.gateway, "place_order"):
            await venue.config.gateway.place_order(exit_order)

    async def _persist_bot_status(self, venue: VenueContext):
        if not self.storage:
            return

        exchange = venue.config.gateway.name
        symbol = venue.config.symbol

        # Determine Execution Mode
        mode = "QUOTING"
        if self.risk_manager.has_breached():
            mode = "SMART_REDUCING"
        elif len(self._momentum_adds.get(f"{exchange}:{symbol}", [])) > 0:
            mode = "ADD_MOMENTUM"

        # Determine Trend
        trend_val = self._venue_trends.get(symbol, 0.0)
        trend_side = (
            self.trend_service.get_trend_side(trend_val)
            if self.trend_service
            else "FLAT"
        )

        status = {
            "symbol": symbol,
            "exchange": exchange,
            "risk_state": "BREACHED" if self.risk_manager.has_breached() else "NORMAL",
            "trend_state": trend_side,
            "execution_mode": mode,
            "active_signals": list(venue.market_state.signals.keys()),
            "risk_breach": self.risk_manager.has_breached(),
            "metadata": {"trend_value": trend_val, "mid_price": venue.last_mid_price},
        }

        if hasattr(self.storage, "save_bot_status"):
            await self.storage.save_bot_status(status)
