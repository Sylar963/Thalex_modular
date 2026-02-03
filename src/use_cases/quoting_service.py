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
    """
    Core Application Service for the Trading Bot.
    Orchestrates the flow of data: Market Data -> Signals -> Strategy -> Execution.
    """

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

        self.symbol: str = ""
        self.market_state = MarketState()
        self.running = False
        self.tick_size = 1.0
        self._reconcile_lock = asyncio.Lock()
        self._last_equity_snapshot_time = 0.0

    async def start(self, symbol: str):
        self.symbol = symbol
        self.running = True

        logger.info(f"Starting Quoting Service for {symbol}")

        # 1. Setup Callbacks
        self.gateway.set_ticker_callback(self.on_ticker_update)
        self.gateway.set_trade_callback(self.on_trade_update)
        self.gateway.set_order_callback(self.state_tracker.on_order_update)
        self.gateway.set_position_callback(self.state_tracker.update_position)

        # Wire reactive events
        self.state_tracker.set_fill_callback(self.on_fill_event)
        self.state_tracker.set_state_gap_callback(self.on_sequence_gap)

        # Start state tracker
        await self.state_tracker.start()

        # 2. Connect
        await self.gateway.connect()

        # 3. Initial State Fetch & Sync
        try:
            initial_pos = await self.gateway.get_position(symbol)
            await self.state_tracker.update_position(
                symbol, initial_pos.size, initial_pos.entry_price
            )
            logger.info(f"Initial Position: {initial_pos}")
        except Exception as e:
            logger.warning(f"Could not fetch initial position: {e}")

        # 4. Subscribe
        await self.gateway.subscribe_ticker(symbol)
        logger.info(f"Subscribed to {symbol}")

    async def stop(self):
        logger.info("Stopping Quoting Service...")
        self.running = False
        await self._cancel_all()
        await self.gateway.disconnect()
        logger.info("Quoting Service Stopped.")

    async def on_ticker_update(self, ticker: Ticker):
        if not self.running:
            return

        # 1. Update Internal State
        self.market_state.ticker = ticker
        self.market_state.timestamp = ticker.timestamp
        # tick_size remains 1.0 for now, could be dynamic later

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
            new_position = await self.gateway.get_position(self.symbol)
            await self.state_tracker.update_position(
                new_position.symbol,
                new_position.size,
                new_position.entry_price,
            )

        if not self.dry_run and not self.risk_manager.can_trade():
            await self._cancel_all()
            return

        if self.regime_analyzer:
            self.regime_analyzer.update(ticker)

        regime = self.regime_analyzer.get_regime() if self.regime_analyzer else None

        if regime and self.storage and not self.dry_run:
            asyncio.create_task(self.storage.save_regime(self.symbol, regime))

        tick_size = getattr(self.gateway, "tick_size", 0.5)

        # 2. Strategy Logic & Execution
        # We delegate to _run_strategy to ensure consistency with _fast_reconcile
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
            if not self._reconcile_lock.locked():
                self.market_state.signals = signals
                asyncio.create_task(self._fast_reconcile())

        if self.storage and not self.dry_run:
            asyncio.create_task(self.storage.save_trade(trade))

    async def _cancel_side(self, side: OrderSide):
        """Emergency cancel for a specific side."""
        orders_to_cancel = self.state_tracker.get_open_orders(side=side)
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
        """Centralized strategy execution pipeline."""
        # 1. Check Lock (Early Exit)
        if self._reconcile_lock.locked():
            return

        async with self._reconcile_lock:
            # 2. Strategy Calculation
            position = self.state_tracker.get_position(self.symbol)
            desired_orders = self.strategy.calculate_quotes(
                self.market_state, position, regime=regime, tick_size=tick_size
            )

            # 3. Tick Alignment
            for i in range(len(desired_orders)):
                o = desired_orders[i]
                if o.price:
                    rounded_price = round(o.price / tick_size) * tick_size
                    desired_orders[i] = replace(o, price=rounded_price)

            # 4. Risk Validation
            valid_orders = []
            # Note: We validate against *open* orders in the tracker + current pos
            active_orders_ref = [t.order for t in self.state_tracker.get_open_orders()]

            for o in desired_orders:
                if self.risk_manager.validate_order(
                    o, position, active_orders=valid_orders
                ):
                    valid_orders.append(o)
                else:
                    logger.warning(f"Risk Manager rejected order: {o}")

            # 5. Execution
            await self._reconcile_orders(valid_orders)

    async def _reconcile_orders(self, desired_orders: List[Order]):
        """
        Diff desired orders against active orders and execute changes.
        Optimized for margin: cancel all unnecessary/moved orders FIRST, then place new ones.
        Utilizes BATCHING to strictly adhere to rate limits and minimize latency.
        """
        desired_by_side = {OrderSide.BUY: [], OrderSide.SELL: []}
        for o in desired_orders:
            desired_by_side[o.side].append(o)

        to_cancel_ids = []
        to_place_orders = []

        # Budget Calculation
        # Max orders per second = 45 (90% of 50)
        # We want to be safe. Let's assume we can do ~5 full batches per second or ~50 ops.
        # But here we are in a single tick.
        # We should limit the number of "places" this cycle if we are low on tokens?
        # The adapter checks tokens. We should prioritize.

        for side in [OrderSide.BUY, OrderSide.SELL]:
            desired_list = desired_by_side.get(side, [])
            active_list = [
                t.order for t in self.state_tracker.get_open_orders(side=side)
            ]

            # Smart Diff Strategy:
            # 1. Map active by ID
            # 2. Iterate desired.
            #    If we find a "close enough" active order, keep it (don't cancel, don't place).
            #    "Close enough": Price diff <= threshold AND Size diff <= threshold

            # Thresholds
            PRICE_THRESHOLD = self.tick_size * 1.5  # 1 tick tolerance? or 0.1?
            # User asked for "Advance technique": "Smart Delta"
            # If price changed by < 2 ticks, keep old order, UNLESS it's the Top Level (Best Bid/Ask).
            # We want Top Level to be precise.

            active_map = []
            for act in active_list:
                active_map.append({"order": act, "matched": False})

            desired_to_place_indices = []

            for i, des in enumerate(desired_list):
                is_top_level = (
                    i == 0
                )  # Assumes desired_list is sorted by aggressiveness?
                # Ideally desired_orders should be sorted. Strategy usually returns them sorted?
                # Let's assume yes or sort them.
                # But 'des' is just an order. We don't know it's rank easily without sorting.
                # Avellaneda strategy usually returns [Bid1, Bid2...] and [Ask1, Ask2...]

                found = False
                best_match_idx = -1

                for j, item in enumerate(active_map):
                    if item["matched"]:
                        continue

                    act = item["order"]

                    # Strict check for top level?
                    # Or just general check?

                    price_diff = abs(des.price - act.price)
                    size_diff = abs(des.size - act.size)

                    # MATCH LOGIC
                    # If same price/size (float tolerance), it's a perfect match.
                    if price_diff < 1e-9 and size_diff < 1e-9:
                        active_map[j]["matched"] = True
                        found = True
                        break

                    # "Lazy" Match
                    # If NOT top level, allow small deviation
                    if not is_top_level:  # We need to confirm I is index of closeness.
                        # Assuming strategy returns levels in order.
                        # We can enforce sorting: Bids descending, Asks ascending.
                        pass

                    # If price is within 2 ticks and size same...
                    # update: User says "increase refresh interval".
                    # We can effectively increase interval for deep orders by tolerating drift.
                    if price_diff <= (1.1 * self.tick_size) and size_diff < 1e-9:
                        # It's "good enough", don't churn it
                        active_map[j]["matched"] = True
                        found = True
                        # We keep the ACTIVE order in our records, even though we wanted DESIRED.
                        # This means our internal state remains the OLD order.
                        # We must update desired_by_side or similar?
                        # No, we just don't place 'des', and we don't cancel 'act'.
                        # effectively 'act' becomes the realization of 'des'.
                        break

                if not found:
                    desired_to_place_indices.append(i)

            # Collect cancels
            for item in active_map:
                if not item["matched"] and item["order"].exchange_id:
                    to_cancel_ids.append(item["order"].exchange_id)

            # Collect places
            for i in desired_to_place_indices:
                to_place_orders.append(desired_list[i])

        if to_cancel_ids or to_place_orders:
            logger.info(
                f"Smart Reconcile: Cancelling {len(to_cancel_ids)}, Placing {len(to_place_orders)}"
            )

        if to_cancel_ids:
            if self.dry_run and self.sim_engine:
                for oid in to_cancel_ids:
                    self.sim_engine.cancel_order(oid)
            else:
                await self.gateway.cancel_orders_batch(to_cancel_ids)

        if to_place_orders:
            for o in to_place_orders:
                await self.state_tracker.submit_order(o)

            if self.dry_run and self.sim_engine:
                for o in to_place_orders:
                    self.sim_engine.submit_order(o)
            else:
                new_orders = await self.gateway.place_orders_batch(to_place_orders)
                for o in new_orders:
                    if o.status == OrderStatus.OPEN and o.exchange_id:
                        await self.state_tracker.on_order_ack(o.id, o.exchange_id)

    async def _cancel_all(self):
        all_active = self.state_tracker.get_open_orders()
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
        # Trigger immediate re-quotes or signal updates if needed
        # This allows the bot to react to fills faster than the next ticker
        if not self._reconcile_lock.locked():
            # Trigger a fast reconcile cycle
            pass

    async def on_sequence_gap(self, gap_size: int):
        """Reactive gap handler: force state sync."""
        logger.warning(
            f"CRITICAL: State Gap Detected ({gap_size} msgs). Forcing re-sync."
        )
        await self._sync_active_orders()
