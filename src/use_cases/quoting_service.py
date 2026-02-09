import asyncio
import logging
import time
from dataclasses import replace
from typing import List, Optional, Dict, TYPE_CHECKING
from ..domain.interfaces import (
    ExchangeGateway,
    Strategy,
    SignalEngine,
    RiskManager,
    StorageGateway,
    RegimeAnalyzer,
)
from ..domain.entities import (
    MarketState,
    Position,
    Ticker,
    Order,
    OrderStatus,
    OrderSide,
)
from ..domain.entities.pnl import EquitySnapshot
from ..domain.tracking.state_tracker import StateTracker

if TYPE_CHECKING:
    from ..domain.sim_match_engine import SimMatchEngine
    from .sim_state_manager import SimStateManager

logger = logging.getLogger(__name__)


class QuotingService:
    def __init__(
        self,
        gateway: ExchangeGateway,
        strategy: Strategy,
        signal_engine: SignalEngine,
        risk_manager: RiskManager,
        storage_gateway: Optional[StorageGateway] = None,
        dry_run: bool = False,
        sim_engine: Optional["SimMatchEngine"] = None,
        sim_state: Optional["SimStateManager"] = None,
        regime_analyzer: Optional[RegimeAnalyzer] = None,
        state_tracker: Optional[StateTracker] = None,
        or_engine: Optional[SignalEngine] = None,
    ):
        self.gateway = gateway
        self.strategy = strategy
        self.signal_engine = signal_engine
        self.risk_manager = risk_manager
        self.storage = storage_gateway
        self.dry_run = dry_run
        self.sim_engine = sim_engine
        self.sim_state = sim_state
        self.regime_analyzer = regime_analyzer
        self.state_tracker = state_tracker or StateTracker()
        self.or_engine = or_engine

        self.symbol: str = ""
        self.market_state = MarketState()
        self.running = False
        # Use a boolean flag for zero-overhead "try-lock" behavior
        self._is_reconciling = False
        self._last_equity_snapshot_time = 0.0
        self._last_mid_price = 0.0
        self.min_edge_threshold = 0.5
        self.tick_size = 0.5

        # Pre-allocate commonly used objects to reduce allocations
        self._reconciliation_cache = {}
        self._perf_metrics = {
            'reconcile_calls': 0,
            'total_reconcile_time': 0.0,
            'avg_reconcile_time': 0.0
        }

    async def start(self, symbol: str):
        self.symbol = symbol
        self.running = True

        logger.info(f"Starting Quoting Service for {symbol}")

        try:
            # 1. Setup Callbacks
            self.gateway.set_ticker_callback(self.on_ticker_update)
            self.gateway.set_trade_callback(self.on_trade_update)
            self.gateway.set_order_callback(self.state_tracker.on_order_update)
            # Fix: Use wrapper to ensure persistence
            self.gateway.set_position_callback(self.on_position_update)
            self.gateway.set_balance_callback(self.on_balance_update)

            # Wire reactive events
            self.state_tracker.set_fill_callback(self.on_fill_event)
            self.state_tracker.set_state_gap_callback(self.on_sequence_gap)

            # Start state tracker
            await self.state_tracker.start()

            # 2. Connect
            await self.gateway.connect()

            # 2.5 Clean Slate: Robust "Detect & Destroy" Strategy
            if hasattr(self.gateway, "cancel_all_orders"):
                logger.info("Cancelling all existing orders on startup...")
                cancel_success = await self.gateway.cancel_all_orders(symbol)

                if not cancel_success:
                    logger.warning(
                        "Bulk cancel failed. Attempting individual cleanup (Detect & Destroy)..."
                    )
                    try:
                        open_orders = await self.gateway.get_open_orders(symbol)
                        if open_orders:
                            logger.info(
                                f"Found {len(open_orders)} stale orders. Cancelling individually..."
                            )
                            tasks = [
                                self.gateway.cancel_order(o.exchange_id)
                                for o in open_orders
                                if o.exchange_id
                            ]
                            results = await asyncio.gather(
                                *tasks, return_exceptions=True
                            )

                            # Verify cleanup
                            failed_cleanups = [
                                r
                                for r in results
                                if isinstance(r, Exception) or r is False
                            ]
                            if failed_cleanups:
                                logger.error(
                                    f"Failed to clean up {len(failed_cleanups)} orders."
                                )
                                # Final check
                                remaining = await self.gateway.get_open_orders(symbol)
                                if remaining:
                                    raise RuntimeError(
                                        f"Startup aborted: Unable to clean {len(remaining)} stale orders."
                                    )
                        else:
                            logger.info(
                                "No stale orders found after bulk cancel failure."
                            )
                    except Exception as e:
                        logger.error(f"Detect & Destroy failed: {e}")
                        raise RuntimeError(
                            "Startup aborted: Failed to ensure clean state."
                        ) from e

                self.state_tracker.pending_orders.clear()
                self.state_tracker.confirmed_orders.clear()
                logger.info("StateTracker order state cleared.")

            # 3. Initial State Fetch & Sync
            try:
                initial_pos = await self.gateway.get_position(symbol)
                await self.state_tracker.update_position(
                    symbol, initial_pos.size, initial_pos.entry_price
                )
                if self.storage:
                    await self.storage.save_position(initial_pos)
                logger.info(f"Initial Position: {initial_pos}")
            except Exception as e:
                logger.warning(f"Could not fetch initial position: {e}")

            # 4. Subscribe
            await self.gateway.subscribe_ticker(symbol)
            logger.info(f"Subscribed to {symbol}")

        except Exception as e:
            logger.error(f"Failed to start QuotingService: {e}")
            self.running = False
            raise e

    async def stop(self):
        logger.info("Stopping Quoting Service...")
        self.running = False
        await self._cancel_all()
        await self.gateway.disconnect()
        logger.info("Quoting Service Stopped.")

    async def on_position_update(self, symbol: str, size: float, entry_price: float):
        """Callback for position updates from Gateway."""
        # 1. Update In-Memory Tracker
        await self.state_tracker.update_position(symbol, size, entry_price)

        # 2. Persist to DB
        if self.storage and not self.dry_run:
            # Fetch the full position object from gateway if available
            if hasattr(self.gateway, "positions") and symbol in self.gateway.positions:
                position = self.gateway.positions[symbol]
            else:
                # Fallback to basic position if full data not available
                position = Position(symbol, size, entry_price)

            asyncio.create_task(self.storage.save_position(position))

    async def on_balance_update(self, balance):
        """Callback for account balance updates from Gateway."""
        if self.storage and not self.dry_run:
            asyncio.create_task(self.storage.save_balance(balance))
        logger.debug(f"Balance update received: {balance}")

    async def on_ticker_update(self, ticker: Ticker):
        if not self.running:
            return

        # DEBUG TRACE
        logger.debug(f"QS received ticker: {ticker.symbol} {ticker.bid}/{ticker.ask}")

        # 1. Update Internal State
        self.market_state.ticker = ticker
        self.market_state.timestamp = ticker.timestamp

        # 1.1 Real-time UPNL and Mark Price Sync in StateTracker
        if self.state_tracker:
            await self.state_tracker.update_ticker(ticker)

        if self.dry_run and self.sim_engine:
            await self.state_tracker.update_position(
                self.symbol,
                self.sim_engine.position_size,
                self.sim_engine.position_entry_price,
            )
            if self.sim_state:
                await self.sim_state.update_price(ticker.mid_price)
                await self.sim_state.update_position(
                    self.sim_engine.position_size,
                    self.sim_engine.position_entry_price,
                    self.sim_engine.balance,
                )
                now = time.time()
                if now - self._last_equity_snapshot_time >= 1.0:
                    self._last_equity_snapshot_time = now
                    equity = self.sim_engine.get_equity(ticker.mid_price)
                    snapshot = EquitySnapshot(
                        timestamp=now,
                        balance=self.sim_engine.balance,
                        position_value=abs(self.sim_engine.position_size)
                        * ticker.mid_price,
                        equity=equity,
                        unrealized_pnl=equity - self.sim_engine.balance,
                    )
                    await self.sim_state.record_equity_snapshot(snapshot)
        else:
            # We rely on callback for updates, but fetching here ensures consistency
            # However, aggressive polling AND callback might be duplicate.
            # Let's keep the persistence here just in case callback misses,
            # BUT make it non-blocking.
            pass

        if not self.dry_run and not self.risk_manager.can_trade():
            await self._cancel_all()
            return

        if self.regime_analyzer:
            self.regime_analyzer.update(ticker)

        regime = self.regime_analyzer.get_regime() if self.regime_analyzer else None

        if regime and self.storage and not self.dry_run:
            asyncio.create_task(self.storage.save_regime(self.symbol, regime))

        if self.storage and not self.dry_run:
            asyncio.create_task(self.storage.save_ticker(ticker))

        if self.or_engine:
            self.or_engine.update(ticker)
            or_signals = self.or_engine.get_signals()
            self.market_state.signals.update(or_signals)

            if (
                hasattr(self.or_engine, "is_session_just_completed")
                and self.or_engine.is_session_just_completed()
            ):
                if self.storage and not self.dry_run:
                    asyncio.create_task(
                        self.storage.save_signal(self.symbol, "open_range", or_signals)
                    )

        tick_size = getattr(self.gateway, "tick_size", 0.5)

        await self._run_strategy(regime=regime, tick_size=tick_size)

    async def on_trade_update(self, trade):
        if not self.running:
            return

        if self.dry_run and self.sim_engine:
            self.sim_engine.on_trade(trade)
            if self.sim_state and self.sim_engine.fills:
                last_fill = self.sim_engine.fills[-1]
                if (
                    last_fill.timestamp == trade.timestamp
                    or abs(last_fill.timestamp - trade.timestamp) < 0.1
                ):
                    await self.sim_state.record_fill(last_fill)

        self.signal_engine.update_trade(trade)
        signals = self.signal_engine.get_signals()

        if hasattr(self.signal_engine, "pop_completed_candle"):
            completed = self.signal_engine.pop_completed_candle()
            if completed and self.storage and not self.dry_run:
                asyncio.create_task(
                    self.storage.save_signal(self.symbol, "vamp", signals)
                )

        immediate_flow = signals.get("immediate_flow", 0.0)
        TOXIC_THRESHOLD = 0.7

        if abs(immediate_flow) > TOXIC_THRESHOLD and not self.dry_run:
            if immediate_flow > 0:
                logger.warning(
                    f"Toxic BUY flow detected ({immediate_flow:.2f}). Pulling ASKS."
                )
                await self._cancel_side(OrderSide.SELL)
            elif immediate_flow < 0:
                logger.warning(
                    f"Toxic SELL flow detected ({immediate_flow:.2f}). Pulling BIDS."
                )
                await self._cancel_side(OrderSide.BUY)
            return

        if abs(immediate_flow) > 0.3:
            if not self._is_reconciling:
                self.market_state.signals.update(signals)
                asyncio.create_task(self._fast_reconcile())

        if self.storage and not self.dry_run:
            logger.debug(f"Saving market trade {trade.id} to DB")
            asyncio.create_task(self.storage.save_trade(trade))

    async def _cancel_side(self, side: OrderSide):
        """Emergency cancel for a specific side."""
        orders_to_cancel = await self.state_tracker.get_open_orders(side=side)
        if not orders_to_cancel:
            return

        tasks = [
            self.gateway.cancel_order(t.order.exchange_id)
            for t in orders_to_cancel
            if t.order.exchange_id
        ]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _fast_reconcile(self):
        """Wrapper to run reconciliation logic triggered by trades/fills."""
        if not self.market_state.ticker:
            return

        tick_size = getattr(self.gateway, "tick_size", 0.5)
        await self._run_strategy(tick_size=tick_size)

    async def _run_strategy(self, regime=None, tick_size=0.5):
        """Centralized strategy execution pipeline with performance optimizations."""
        if self._is_reconciling:
            return

        current_mid = (
            self.market_state.ticker.mid_price if self.market_state.ticker else 0.0
        )

        MIN_EDGE_THRESHOLD = tick_size * self.min_edge_threshold
        mid_price_move = abs(current_mid - self._last_mid_price)

        if self._last_mid_price > 0 and mid_price_move < MIN_EDGE_THRESHOLD:
            return

        self._last_mid_price = current_mid
        self._is_reconciling = True

        try:
            start_time = time.perf_counter()

            # 2. Strategy Calculation
            position = self.state_tracker.get_position(self.symbol)
            desired_orders = self.strategy.calculate_quotes(
                self.market_state, position, regime=regime, tick_size=tick_size
            )

            # 3. Tick Alignment - optimized in-place
            for i in range(len(desired_orders)):
                o = desired_orders[i]
                if o.price:
                    rounded_price = round(o.price / tick_size) * tick_size
                    desired_orders[i] = replace(o, price=rounded_price)

            # 4. Risk Validation - optimized with early exit
            valid_orders = []
            # Note: We validate against *open* orders in the tracker + current pos
            active_orders_ref = [t.order for t in await self.state_tracker.get_open_orders()]

            for o in desired_orders:
                if self.risk_manager.validate_order(
                    o, position, active_orders=valid_orders
                ):
                    valid_orders.append(o)
                else:
                    logger.warning(f"Risk Manager rejected order: {o}")

            # 5. Execution
            await self._reconcile_orders(valid_orders)

            # Update performance metrics
            end_time = time.perf_counter()
            reconcile_time = (end_time - start_time) * 1000  # Convert to ms
            self._perf_metrics['reconcile_calls'] += 1
            self._perf_metrics['total_reconcile_time'] += reconcile_time
            self._perf_metrics['avg_reconcile_time'] = (
                self._perf_metrics['total_reconcile_time'] /
                self._perf_metrics['reconcile_calls']
            )

            if self._perf_metrics['reconcile_calls'] % 100 == 0:
                logger.debug(f"Reconciliation performance: avg={self._perf_metrics['avg_reconcile_time']:.3f}ms over {self._perf_metrics['reconcile_calls']} calls")
        finally:
            self._is_reconciling = False

    async def _reconcile_orders(self, desired_orders: List[Order]):
        """
        Optimized diff of desired orders against active orders and execute changes.
        Uses efficient algorithms to minimize latency and reduce allocations.
        """
        start_time = time.perf_counter()

        # Pre-partition by side for efficiency
        desired_buy = [o for o in desired_orders if o.side == OrderSide.BUY]
        desired_sell = [o for o in desired_orders if o.side == OrderSide.SELL]

        to_cancel_ids = []
        to_place_orders = []

        # Process both sides in parallel
        buy_tasks = asyncio.create_task(self._diff_side(OrderSide.BUY, desired_buy))
        sell_tasks = asyncio.create_task(self._diff_side(OrderSide.SELL, desired_sell))

        buy_result = await buy_tasks
        sell_result = await sell_tasks

        to_cancel_ids.extend(buy_result[0])
        to_cancel_ids.extend(sell_result[0])
        to_place_orders.extend(buy_result[1])
        to_place_orders.extend(sell_result[1])

        if to_cancel_ids or to_place_orders:
            logger.info(
                f"Smart Reconcile: Cancelling {len(to_cancel_ids)}, Placing {len(to_place_orders)}"
            )

        # Execute cancellations first to free up limits
        if to_cancel_ids:
            if self.dry_run and self.sim_engine:
                for oid in to_cancel_ids:
                    self.sim_engine.cancel_order(oid)
            else:
                results = await self.gateway.cancel_orders_batch(to_cancel_ids)
                for oid, success in zip(to_cancel_ids, results):
                    if success:
                        await self.state_tracker.on_order_cancel(oid)

        # Then place new orders
        if to_place_orders:
            if self.dry_run and self.sim_engine:
                for o in to_place_orders:
                    await self.state_tracker.submit_order(o)
                    self.sim_engine.submit_order(o)
            else:
                new_orders = await self.gateway.place_orders_batch(to_place_orders)
                for o in new_orders:
                    if o.status == OrderStatus.OPEN and o.exchange_id:
                        await self.state_tracker.submit_order(o)
                        await self.state_tracker.on_order_ack(o.id, o.exchange_id)
                    elif o.status == OrderStatus.REJECTED:
                        logger.warning(
                            f"Order {o.id} was rejected by exchange, not tracking."
                        )

        end_time = time.perf_counter()
        logger.debug(f"_reconcile_orders took {(end_time - start_time) * 1000:.3f}ms")

    async def _diff_side(self, side: OrderSide, desired_list: List[Order]) -> tuple:
        """
        Efficiently compute the difference for a single side (BUY or SELL).
        Returns (to_cancel_ids, to_place_orders) tuple.
        """
        active_list = [
            t.order for t in await self.state_tracker.get_open_orders(side=side)
        ]

        # Use a more efficient matching algorithm
        to_cancel_ids = []
        to_place_orders = []

        # Create lookup dictionaries for O(1) access
        active_by_price_size = {}
        for act in active_list:
            key = (round(act.price / self.tick_size), act.size)  # Normalize price to tick boundaries
            active_by_price_size[key] = act

        desired_by_price_size = {}
        for des in desired_list:
            key = (round(des.price / self.tick_size), des.size)  # Normalize price to tick boundaries
            desired_by_price_size[key] = des

        # Find orders to cancel (in active but not in desired)
        for key, act in active_by_price_size.items():
            if key not in desired_by_price_size and act.exchange_id:
                to_cancel_ids.append(act.exchange_id)

        # Find orders to place (in desired but not in active)
        for key, des in desired_by_price_size.items():
            if key not in active_by_price_size:
                to_place_orders.append(des)

        return to_cancel_ids, to_place_orders

    async def _cancel_all(self):
        all_active = await self.state_tracker.get_open_orders()
        if not all_active:
            return

        tasks = [
            self.gateway.cancel_order(t.order.exchange_id)
            for t in all_active
            if t.order.exchange_id
        ]
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _sync_active_orders(self):
        """Force a snapshot sync if state is suspected to be drifted."""
        try:
            exchange_orders = await self.gateway.get_open_orders(self.symbol)
            pos = await self.gateway.get_position(self.symbol)

            await self.state_tracker.force_snapshot_sync(
                exchange_orders, {self.symbol: pos}
            )
            logger.info("Reactive State Sync Complete.")
        except Exception as e:
            logger.error(f"Failed to sync state: {e}")

    async def on_fill_event(self, exchange_id: str, price: float, size: float):
        """Reactive fill handler."""
        logger.info(f"FILL RECEIVED: {exchange_id} at {price} ({size} units)")

        # PERSISTENCE: Record Bot Execution
        if self.storage and not self.dry_run:
            # Try to resolve order details
            found_order = None
            known_orders = await self.state_tracker.get_open_orders()

            for tracker_item in known_orders:
                if tracker_item.order.exchange_id == exchange_id:
                    found_order = tracker_item.order
                    break

            if found_order:
                # Create Trade Object for storage
                from ..domain.entities import Trade
                import time

                exec_trade = Trade(
                    id=f"exec_{int(time.time() * 1000)}_{exchange_id}",
                    order_id=found_order.id,  # Internal ID
                    symbol=found_order.symbol,
                    side=found_order.side,
                    price=price,
                    size=size,
                    exchange="thalex",
                    fee=0.0,
                    timestamp=time.time(),
                )
                asyncio.create_task(self.storage.save_execution(exec_trade))
            else:
                logger.warning(
                    f"Could not find order for fill {exchange_id} - execution not saved."
                )
        # Trigger immediate re-quotes or signal updates if needed
        # This allows the bot to react to fills faster than the next ticker
        if not self._is_reconciling:
            # Trigger a fast reconcile cycle
            pass

    async def on_sequence_gap(self, gap_size: int):
        """Reactive gap handler: force state sync."""
        logger.warning(
            f"CRITICAL: State Gap Detected ({gap_size} msgs). Forcing re-sync."
        )
        await self._sync_active_orders()

    def get_performance_metrics(self) -> Dict:
        """Return current performance metrics for monitoring."""
        return self._perf_metrics.copy()
