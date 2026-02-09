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
    RiskManager,
    StorageGateway,
    SignalEngine,
    SafetyComponent,
)
from ..domain.market.trend_service import HistoricalTrendService
from ..domain.signals.inventory_bias import InventoryBiasEngine
from ..domain.tracking.sync_engine import SyncEngine, GlobalState
from ..domain.tracking.state_tracker import StateTracker
from ..domain.tracking.position_tracker import PortfolioTracker, Fill
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


class OptimizedVenueContext:
    """
    Optimized venue context with per-venue state management to reduce global lock contention
    """

    def __init__(self, config: ExchangeConfig):
        self.config = config
        self.state_tracker = StateTracker()
        self.market_state = MarketState()
        self.last_mid_price = 0.0
        self._venue_lock = asyncio.Lock()
        self._last_strategy_run = 0.0
        self._perf_metrics = {
            "strategy_runs": 0,
            "avg_run_time": 0.0,
            "total_run_time": 0.0,
        }
        self._consecutive_failures = 0
        self._failure_backoff_until = 0.0
        self._reducing_order_id: Optional[str] = None
        self._reducing_order_timestamp = 0.0
        self._first_ticker_logged = False


class MultiExchangeStrategyManager:
    def __init__(
        self,
        exchanges: List[ExchangeConfig],
        strategy: Strategy,
        risk_manager: RiskManager,
        sync_engine: SyncEngine,
        signal_engine: Optional[SignalEngine] = None,
        or_engine: Optional[SignalEngine] = None,
        canary_sensor: Optional[SignalEngine] = None,
        inventory_bias_engine: Optional[InventoryBiasEngine] = None,
        storage: Optional[StorageGateway] = None,
        safety_components: Optional[List[SafetyComponent]] = None,
        dry_run: bool = False,
    ):
        self.venues: Dict[str, OptimizedVenueContext] = {}
        for cfg in exchanges:
            self.venues[cfg.gateway.name] = OptimizedVenueContext(cfg)

        self.strategy = strategy
        self.risk_manager = risk_manager
        self.signal_engine = signal_engine
        self.sync_engine = sync_engine
        self.or_engine = or_engine
        self.canary_sensor = canary_sensor
        self.inventory_bias_engine = inventory_bias_engine
        self.storage = storage
        self.safety_components = safety_components or []
        self.dry_run = dry_run

        self.trend_service = HistoricalTrendService(storage) if storage else None
        self._venue_trends: Dict[str, float] = {}  # symbol -> trend_value
        self._last_trend_update = 0.0
        self._last_status_persist: Dict[str, float] = {}
        self._last_hft_persist: Dict[str, float] = {}

        self._momentum_adds: Dict[
            str, List[Dict]
        ] = {}  # venue_key (exchange:symbol) -> list of adds

        self.portfolio = Portfolio()
        self.pnl_tracker = PortfolioTracker()
        self._running = False
        self._tasks: List[asyncio.Task] = []
        # Removed global reconcile lock - now using per-venue locks
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

    async def _connect_exchange(self, venue: OptimizedVenueContext):
        cfg = venue.config
        gw = cfg.gateway

        gw.set_ticker_callback(self._make_ticker_callback(gw.name))
        gw.set_trade_callback(self._make_trade_callback(gw.name))
        gw.set_order_callback(venue.state_tracker.on_order_update)
        gw.set_position_callback(self._make_position_callback(gw.name))
        gw.set_balance_callback(self._handle_balance_update)
        gw.set_execution_callback(self._make_execution_callback(gw.name))

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

        if hasattr(gw, "get_open_orders"):
            try:
                open_orders = await gw.get_open_orders(cfg.symbol)
                for o in open_orders:
                    await venue.state_tracker.seed_order(o)
                logger.info(
                    f"Synchronized {len(open_orders)} existing orders for {gw.name}"
                )
            except Exception as e:
                logger.warning(
                    f"Failed to fetch initial open orders for {gw.name}: {e}"
                )

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
                logger.warning(f"Ticker received for unknown exchange: {exchange}")
                return

            # Symbol filtering
            if ticker.symbol != venue.config.symbol:
                return

            # log every 100th ticker or so if needed, but for now let's just log arrival once
            if not hasattr(venue, "_first_ticker_logged"):
                logger.info(f"First ticker received for {exchange}:{ticker.symbol}")
                venue._first_ticker_logged = True

            ticker = replace(ticker, exchange=exchange)
            venue.market_state = MarketState(ticker=ticker, timestamp=ticker.timestamp)

            # Ensure state tracker is updated for real-time UPNL
            await venue.state_tracker.update_ticker(ticker)

            if self.signal_engine:
                self.signal_engine.update(ticker)

            if self.or_engine:
                self.or_engine.update(ticker)
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

            if self.canary_sensor:
                self.canary_sensor.update(ticker)
                canary_signals = self.canary_sensor.get_signals()
                venue.market_state.signals.update(canary_signals)

                venue_key = f"{exchange}:{ticker.symbol}"
                now = ticker.timestamp
                last_persist = self._last_hft_persist.get(venue_key, 0.0)
                if self.storage and not self.dry_run and (now - last_persist) >= 1.0:
                    self._last_hft_persist[venue_key] = now
                    asyncio.create_task(
                        self.storage.save_hft_signal(
                            ticker.symbol,
                            exchange,
                            canary_signals,
                            ticker.bid,
                            ticker.ask,
                        )
                    )

            if self.inventory_bias_engine:
                position = self.portfolio.get_position(ticker.symbol, exchange)
                or_sigs = (
                    self.or_engine.get_signals().get(ticker.symbol, {})
                    if self.or_engine
                    else {}
                )
                vamp_sigs = (
                    self.signal_engine.get_signals() if self.signal_engine else {}
                )

                if isinstance(or_sigs, dict):
                    or_sigs["current_price"] = ticker.mid_price

                self.inventory_bias_engine.update_position(
                    position.size if position else 0.0
                )
                self.inventory_bias_engine.update_signals(or_sigs, vamp_sigs)
                venue.market_state.signals.update(
                    self.inventory_bias_engine.get_signals()
                )

            await self.sync_engine.update_ticker(exchange, ticker.symbol, ticker)

            if self.storage and not self.dry_run:
                asyncio.create_task(self.storage.save_ticker(ticker))

            # --- Safety Checks ---
            safety_context = {"timestamp": ticker.timestamp, "ticker": ticker}
            is_healthy = True
            for component in self.safety_components:
                if not component.check_health(safety_context):
                    component.record_failure()
                    logger.warning(
                        f"Safety Check FAILED: {component.__class__.__name__}. skipping cycle."
                    )
                    is_healthy = False
                    # Don't break immediately, let all record failure if needed?
                    # Or break to fail fast? Fail fast is better for latency.
                    break
                else:
                    # Only record success if we actually passed?
                    # Circuit breaker needs explicit success to reset/heal.
                    # We should probably do this at end of successful cycle, or here if check passed?
                    # If we record success here, it might be granular.
                    # For CB, success means "Time passed without failure" or "Operation succeeded".
                    # Let's simple record success if check passes for now.
                    component.record_success()

            if is_healthy:
                # Fix for Deadlock: Decouple strategy run from WebSocket loop
                # If we await here, the WS loop blocks, preventing RPC responses (Order Acks) from being processed.
                # This causes the strategy to timeout waiting for the very messages this loop is supposed to deliver.
                if not venue._venue_lock.locked():
                    asyncio.create_task(self._run_strategy_for_venue(venue))
                else:
                    # Strategy is already running, skip this tick to prevent pile-up
                    pass

        return callback

    def _make_trade_callback(self, exchange: str) -> Callable:
        async def callback(trade: Trade):
            # Ensure trade has correct exchange context
            trade = replace(trade, exchange=exchange)

            if self.signal_engine:
                self.signal_engine.update_trade(trade)
                if hasattr(self.signal_engine, "pop_completed_candle"):
                    completed = self.signal_engine.pop_completed_candle()
                    if completed and self.storage and not self.dry_run:
                        signals = self.signal_engine.get_signals()
                        asyncio.create_task(
                            self.storage.save_signal(trade.symbol, "vamp", signals)
                        )

            if self.canary_sensor:
                self.canary_sensor.update_trade(trade)

            logger.debug(
                f"[{exchange}] Trade: {trade.side.value} {trade.size} @ {trade.price}"
            )

            if self.storage and not self.dry_run:
                asyncio.create_task(self.storage.save_trade(trade))

        return callback

    def _make_execution_callback(self, exchange: str) -> Callable:
        async def callback(trade: Trade):
            trade = replace(trade, exchange=exchange)
            logger.info(
                f"[{exchange}] FILL: {trade.side.value} {trade.size} @ {trade.price}"
            )
            if self.storage and not self.dry_run:
                asyncio.create_task(self.storage.save_execution(trade))

            fill = Fill(
                order_id=trade.order_id,
                fill_price=trade.price,
                fill_size=trade.size,
                fill_time=trade.timestamp,
                side=trade.side.value,
            )
            tracker = self.pnl_tracker.get_tracker(trade.symbol)
            tracker.update_on_fill(fill)
            realized = tracker.realized_pnl
            logger.info(
                f"[{exchange}] PnL: realized=${realized:.4f}, pos={tracker.current_position}"
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
            if self.risk_manager:
                self.risk_manager.update_position(position)

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

    async def _run_strategy_for_venue(self, venue: OptimizedVenueContext):
        """
        Run strategy for a specific venue with reduced lock contention
        """
        # Health check: Skip if the venue is not ready
        if (
            not hasattr(venue.config.gateway, "is_ready")
            or not venue.config.gateway.is_ready
        ):
            logger.debug(
                f"Skipping strategy cycle for {venue.config.gateway.name}: Not ready"
            )
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

        # Log venue health periodically
        if now % 30 < 1:  # Log approximately every 30 seconds
            logger.info(
                f"Venue {venue.config.gateway.name} healthy - Ticker: {venue.market_state.ticker}"
            )

        # 2. Synchronize Position Before Risk Check (CRITICAL for accurate state)
        # Fetch fresh position data from exchange to prevent stale state during rapid fills
        try:
            gw = venue.config.gateway
            symbol = venue.config.symbol
            fresh_position = await gw.get_position(symbol)

            await venue.state_tracker.update_position(
                symbol, fresh_position.size, fresh_position.entry_price
            )

            pos = Position(
                symbol=symbol,
                size=fresh_position.size,
                entry_price=fresh_position.entry_price,
                exchange=gw.name,
            )
            self.portfolio.set_position(pos)
            if self.risk_manager:
                self.risk_manager.update_position(pos)

        except Exception as e:
            logger.warning(
                f"Failed to sync position for {venue.config.gateway.name}: {e}"
            )

        # 3. Check for Risk Breach
        if self.risk_manager.has_breached():
            logger.critical("RISK BREACH DETECTED - ENTERING SMART REDUCING MODE")
            await self._run_smart_reducing_mode(venue)
            return

        if not venue.market_state.ticker:
            return

        current_mid = venue.market_state.ticker.mid_price
        tick_size = venue.config.tick_size

        # Adjust min_edge based on venue characteristics to prevent stale behavior
        # For Thalex with larger tick sizes, we need to be more sensitive to price changes
        if tick_size >= 1.0:  # Thalex typically has 1.0 tick size
            # Use a more dynamic calculation based on both tick size and market volatility
            min_edge = tick_size * 0.1  # More sensitive for coarse tick sizes
        else:
            min_edge = (
                tick_size * self.min_edge_threshold
            )  # Standard sensitivity for fine tick sizes

        # Further adjust min_edge based on exchange-specific characteristics
        exchange_name = venue.config.gateway.name
        if exchange_name.lower() == "thalex":
            # For Thalex, be slightly more responsive to price movements
            min_edge = min_edge * 0.8  # 20% more responsive
        elif exchange_name.lower() == "bybit":
            min_edge = min_edge * 1.0  # Standard responsiveness

        # Use per-venue lock instead of global lock
        async with venue._venue_lock:
            start_time = time.perf_counter()

            # 3. Handle Momentum Sub-strategy (Adds and Exits) - NOW INSIDE LOCK
            await self._manage_momentum_strategy(venue)

            exchange = venue.config.gateway.name
            position = self.portfolio.get_position(venue.config.symbol, exchange)

            if self.signal_engine:
                venue.market_state.signals.update(self.signal_engine.get_signals())

            active_strategy = (
                venue.config.strategy if venue.config.strategy else self.strategy
            )

            desired_orders = active_strategy.calculate_quotes(
                venue.market_state, position, tick_size=tick_size, exchange=exchange
            )

            for i in range(len(desired_orders)):
                o = desired_orders[i]
                if o.price:
                    rounded = round(o.price / tick_size) * tick_size
                    desired_orders[i] = replace(o, price=rounded, exchange=exchange)

            # Fetch ALL open orders (confirmed + pending) for risk validation
            all_open_tracked = await venue.state_tracker.get_open_orders(
                include_pending=True
            )
            all_open_orders = [t.order for t in all_open_tracked]

            valid_orders = []
            for o in desired_orders:
                # pass BOTH existing open orders AND newly calculated ones for this batch
                if self.risk_manager.validate_order(
                    o, self.portfolio, active_orders=all_open_orders + valid_orders
                ):
                    valid_orders.append(o)
                else:
                    logger.warning(f"Risk rejected order on {exchange}: {o}")

            # Log detailed metrics for Thalex specifically
            if exchange.lower() == "thalex":
                logger.info(
                    f"Thalex strategy run - Position: {position.size}, Desired orders: {len(desired_orders)}, Valid orders: {len(valid_orders)}"
                )
                if venue.market_state.ticker:
                    logger.info(
                        f"Thalex ticker - Bid: {venue.market_state.ticker.bid}, Ask: {venue.market_state.ticker.ask}, Mid: {venue.market_state.ticker.mid_price}"
                    )

                # Additional Thalex-specific metrics
                spread = (
                    venue.market_state.ticker.ask - venue.market_state.ticker.bid
                    if venue.market_state.ticker
                    else 0
                )
                logger.info(
                    f"Thalex spread: {spread}, Position utilization: {abs(position.size) / venue.config.strategy.position_limit if venue.config.strategy and venue.config.strategy.position_limit > 0 else 0}"
                )

            await self._reconcile_orders_venue(venue, valid_orders)

            # Update performance metrics
            end_time = time.perf_counter()
            run_time = (end_time - start_time) * 1000  # Convert to ms
            venue._perf_metrics["strategy_runs"] += 1
            venue._perf_metrics["total_run_time"] += run_time
            venue._perf_metrics["avg_run_time"] = (
                venue._perf_metrics["total_run_time"]
                / venue._perf_metrics["strategy_runs"]
            )

            # Log performance metrics for Thalex
            if exchange.lower() == "thalex":
                logger.info(
                    f"Thalex strategy performance - Run time: {run_time:.2f}ms, Avg: {venue._perf_metrics['avg_run_time']:.2f}ms"
                )

    async def _reconcile_orders_venue(
        self, venue: OptimizedVenueContext, desired_orders: List[Order]
    ):
        """
        Reconcile orders for a specific venue with optimized diff algorithm
        """
        if not self._running:
            return

        start_time = time.perf_counter()

        desired_by_side = {OrderSide.BUY: [], OrderSide.SELL: []}
        for o in desired_orders:
            desired_by_side[o.side].append(o)

        to_cancel_ids = []
        to_place_orders = []
        tick_size = venue.config.tick_size

        for side in [OrderSide.BUY, OrderSide.SELL]:
            desired_list = desired_by_side.get(side, [])
            active_list = [
                t.order
                for t in await venue.state_tracker.get_open_orders(
                    side=side, include_pending=True
                )
            ]

            # Use optimized diff algorithm
            active_by_price_size = {}
            for act in active_list:
                key = (
                    round(act.price / tick_size),
                    act.size,
                )  # Normalize price to tick boundaries
                active_by_price_size[key] = act

            desired_by_price_size = {}
            for des in desired_list:
                key = (
                    round(des.price / tick_size),
                    des.size,
                )  # Normalize price to tick boundaries
                desired_by_price_size[key] = des

            # Find orders to cancel (in active but not in desired)
            for key, act in active_by_price_size.items():
                if key not in desired_by_price_size and act.exchange_id:
                    to_cancel_ids.append(act.exchange_id)

            # Find orders to place (in desired but not in active)
            for key, des in desired_by_price_size.items():
                if key not in active_by_price_size:
                    to_place_orders.append(des)

        gw = venue.config.gateway
        exchange = gw.name

        if to_cancel_ids or to_place_orders:
            logger.info(
                f"[{exchange}] Reconcile: Cancel {len(to_cancel_ids)}, Place {len(to_place_orders)}"
            )

        if to_cancel_ids:
            try:
                results = await gw.cancel_orders_batch(to_cancel_ids)
                for oid, success in zip(to_cancel_ids, results):
                    if success:
                        await venue.state_tracker.on_order_cancel(oid)
            except ConnectionError as e:
                logger.warning(
                    f"Connection error during cancel_orders_batch for {exchange}: {e}"
                )
            except Exception as e:
                logger.error(
                    f"Unexpected error during cancel_orders_batch for {exchange}: {e}"
                )

        if to_place_orders:
            now = time.time()
            if now < venue._failure_backoff_until:
                remaining = venue._failure_backoff_until - now
                logger.warning(
                    f"[{exchange}] Skipping order placement due to circuit breaker. "
                    f"Backoff ends in {remaining:.1f}s (consecutive failures: {venue._consecutive_failures})"
                )
                return

            try:
                new_orders = await gw.place_orders_batch(to_place_orders)
                for o in new_orders:
                    if o.status == OrderStatus.OPEN and o.exchange_id:
                        await venue.state_tracker.submit_order(o)
                        await venue.state_tracker.on_order_ack(o.id, o.exchange_id)

                venue._consecutive_failures = 0
                venue._failure_backoff_until = 0.0

            except ConnectionError as e:
                venue._consecutive_failures += 1
                backoff_duration = min(2 ** (venue._consecutive_failures - 1), 16)
                venue._failure_backoff_until = now + backoff_duration
                logger.warning(
                    f"Connection error during place_orders_batch for {exchange}: {e}. "
                    f"Circuit breaker activated. Backoff: {backoff_duration}s (failure #{venue._consecutive_failures})"
                )
            except Exception as e:
                venue._consecutive_failures += 1
                backoff_duration = min(2 ** (venue._consecutive_failures - 1), 16)
                venue._failure_backoff_until = now + backoff_duration
                logger.error(
                    f"Unexpected error during place_orders_batch for {exchange}: {e}. "
                    f"Circuit breaker activated. Backoff: {backoff_duration}s (failure #{venue._consecutive_failures})"
                )

        end_time = time.perf_counter()
        logger.debug(
            f"_reconcile_orders_venue for {exchange} took {(end_time - start_time) * 1000:.3f}ms"
        )

    async def _cancel_all_venue(self, venue: OptimizedVenueContext):
        all_active = await venue.state_tracker.get_open_orders()
        if not all_active:
            return

        tasks = []
        for t in all_active:
            if t.order.exchange_id:
                task = asyncio.create_task(
                    self._safe_cancel_order(venue.config.gateway, t.order.exchange_id)
                )
                tasks.append(task)

        await asyncio.gather(*tasks, return_exceptions=True)

    async def _safe_cancel_order(self, gateway, order_id):
        """Safely cancel an order, handling connection errors."""
        try:
            return await gateway.cancel_order(order_id)
        except ConnectionError as e:
            logger.warning(
                f"Connection error during cancel_order for {gateway.name}: {e}"
            )
            return False
        except Exception as e:
            logger.error(
                f"Unexpected error during cancel_order for {gateway.name}: {e}"
            )
            return False

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

    async def _run_smart_reducing_mode(self, venue: OptimizedVenueContext):
        async with venue._venue_lock:
            exchange = venue.config.gateway.name
            symbol = venue.config.symbol
            position = self.portfolio.get_position(symbol, exchange)

            if abs(position.size) < 1e-9:
                venue._reducing_order_id = None
                venue._reducing_order_timestamp = 0.0
                return

            now = time.time()

            if venue._reducing_order_id:
                all_open = await venue.state_tracker.get_open_orders(
                    include_pending=True
                )
                order_still_active = any(
                    t.order.id == venue._reducing_order_id for t in all_open
                )

                if order_still_active:
                    time_since_placed = now - venue._reducing_order_timestamp
                    if time_since_placed < 30:
                        logger.debug(
                            f"Reducing order {venue._reducing_order_id} still active "
                            f"({time_since_placed:.1f}s ago). Skipping duplicate placement."
                        )
                        return
                    else:
                        logger.warning(
                            f"Reducing order {venue._reducing_order_id} stale "
                            f"(placed {time_since_placed:.1f}s ago). Replacing..."
                        )
                        venue._reducing_order_id = None
                else:
                    venue._reducing_order_id = None

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

            ticker = venue.market_state.ticker
            if not ticker:
                return

            desired_orders = []
            order_id = f"{'exit' if is_counter_trend else 'tp'}_{int(now)}"

            if is_counter_trend:
                logger.warning(
                    f"Counter-trend position detected in breach ({trend_side}). Exiting aggressively."
                )
                price = ticker.bid if position.size > 0 else ticker.ask
                desired_orders.append(
                    Order(
                        id=order_id,
                        symbol=symbol,
                        side=OrderSide.SELL if position.size > 0 else OrderSide.BUY,
                        price=price,
                        size=abs(position.size),
                        type=OrderType.LIMIT,
                        exchange=exchange,
                    )
                )
            else:
                logger.info(
                    f"Trend-following position in breach ({trend_side}). Setting break-even TP."
                )
                fee_bps = 5
                break_even = position.entry_price * (
                    1 + (fee_bps / 10000 if position.size > 0 else -fee_bps / 10000)
                )

                desired_orders.append(
                    Order(
                        id=order_id,
                        symbol=symbol,
                        side=OrderSide.SELL if position.size > 0 else OrderSide.BUY,
                        price=break_even,
                        size=abs(position.size),
                        type=OrderType.LIMIT,
                        exchange=exchange,
                    )
                )

            venue._reducing_order_id = order_id
            venue._reducing_order_timestamp = now

            await self._reconcile_orders_venue(venue, desired_orders)

    async def _manage_momentum_strategy(self, venue: OptimizedVenueContext):
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

            # Check Risk Before Placing
            # We pass current portfolio position for validation
            current_pos = self.portfolio.get_position(symbol, exchange)
            if self.risk_manager and not self.risk_manager.validate_order(
                order, current_pos
            ):
                logger.warning(f"Momentum Add REJECTED by Risk Manager: {order}")
                return

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

    async def _exit_momentum_add(self, venue: OptimizedVenueContext, add_info: Dict):
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

    async def _persist_bot_status(self, venue: OptimizedVenueContext):
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
