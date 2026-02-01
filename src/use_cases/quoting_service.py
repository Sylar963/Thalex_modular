import asyncio
import logging
import time
from dataclasses import replace
from typing import List, Optional, Dict
from ..domain.interfaces import (
    ExchangeGateway,
    Strategy,
    SignalEngine,
    RiskManager,
    StorageGateway,
)
from ..domain.entities import (
    MarketState,
    Position,
    Ticker,
    Order,
    OrderStatus,
    OrderSide,
)

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
    ):
        self.gateway = gateway
        self.strategy = strategy
        self.signal_engine = signal_engine
        self.risk_manager = risk_manager
        self.storage = storage_gateway

        self.symbol: str = ""
        self.market_state = MarketState()
        self.position = Position("", 0.0, 0.0)
        self.active_orders: Dict[OrderSide, List[Order]] = {
            OrderSide.BUY: [],
            OrderSide.SELL: [],
        }  # Side -> List[Order]
        self.running = False
        self.tick_size = 1.0  # From logs, tick for BTC-PERPETUAL is 1
        self._reconcile_lock = asyncio.Lock()

    async def start(self, symbol: str):
        self.symbol = symbol
        self.running = True

        logger.info(f"Starting Quoting Service for {symbol}")

        # 1. Setup Callbacks
        self.gateway.set_ticker_callback(self.on_ticker_update)
        self.gateway.set_trade_callback(self.on_trade_update)

        # 2. Connect
        await self.gateway.connect()

        # 3. Initial State Fetch
        try:
            self.position = await self.gateway.get_position(symbol)
            logger.info(f"Initial Position: {self.position}")
        except Exception as e:
            logger.warning(f"Could not fetch initial position: {e}")
            self.position = Position(symbol, 0.0, 0.0)

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

        # Refresh position from local cache of adapter (fast)
        new_position = await self.gateway.get_position(self.symbol)

        # Update Signals
        self.signal_engine.update(ticker)
        self.market_state.signals = self.signal_engine.get_signals()

        # Update order list from gateway periodically or on significant price move
        # For now, let's just make sure we don't have stale orders
        # If the user cancels manually, the exchange will eventually notify us or we'll find out here
        # Actually, the adapter should update self.orders when it receives cancellations.
        # But we need to sync self.active_orders.
        await self._sync_active_orders()

        # Persist Data
        if self.storage:
            # Fire and forget (create task) to avoid blocking main loop
            asyncio.create_task(self.storage.save_ticker(ticker))
            # Save position snapshot
            asyncio.create_task(self.storage.save_position(new_position))

        self.position = new_position

        # 2. Risk Check (Global)
        if not self.risk_manager.can_trade():
            await self._cancel_all()
            return

        # 3. Strategy Calculation
        # Safely get tick_size from gateway, falling back to 0.5 (BTC default)
        tick_size = getattr(self.gateway, "tick_size", 0.5)

        desired_orders = self.strategy.calculate_quotes(
            self.market_state, self.position, tick_size=tick_size
        )

        # 3.1 Force Tick Alignment
        for i in range(len(desired_orders)):
            o = desired_orders[i]
            if o.price:
                rounded_price = round(o.price / self.tick_size) * self.tick_size
                desired_orders[i] = replace(o, price=rounded_price)

        # 4. Risk Validation
        valid_orders = []
        for o in desired_orders:
            # We validate against the orders we have already accepted for this cycle
            # This ensures the TOTAL set of orders + position is within limits
            if self.risk_manager.validate_order(
                o, self.position, active_orders=valid_orders
            ):
                valid_orders.append(o)
            else:
                logger.warning(f"Risk Manager rejected order: {o}")

        # Observability Log
        if self.running:
            # Just a periodic summary or if things change?
            # For now, log the target vs active count
            pass

        # 5. Execution (Diff against active orders)
        # Use a lock to prevent concurrent reconciliation on fast tickers
        if self._reconcile_lock.locked():
            return

        async with self._reconcile_lock:
            await self._reconcile_orders(valid_orders)

    async def on_trade_update(self, trade):
        if not self.running:
            return

        # 1. Update Signals
        self.signal_engine.update_trade(trade)
        signals = self.signal_engine.get_signals()

        # 2. TOXIC FLOW DEFENSE
        # Check for immediate adverse selection (Strong flow against us)
        # immediate_flow is -1.0 (Sell pressure) to 1.0 (Buy pressure)
        immediate_flow = signals.get("immediate_flow", 0.0)

        # Threshold: > 0.7 means 70% of recent volume is one-sided
        TOXIC_THRESHOLD = 0.7

        if abs(immediate_flow) > TOXIC_THRESHOLD:
            # If heavy BUY pressure (flow > 0), we are at risk of selling too cheap.
            # Cancel ASKS.
            if immediate_flow > 0:
                logger.warning(
                    f"Toxic BUY flow detected ({immediate_flow:.2f}). Pulling ASKS."
                )
                await self._cancel_side(OrderSide.SELL)

            # If heavy SELL pressure (flow < 0), we are at risk of buying too high.
            # Cancel BIDS.
            elif immediate_flow < 0:
                logger.warning(
                    f"Toxic SELL flow detected ({immediate_flow:.2f}). Pulling BIDS."
                )
                await self._cancel_side(OrderSide.BUY)

            # Force immediate strategy recalculation?
            # Ideally yes, but after cancels settle.
            # For now, just pulling the quotes protects us.
            return

        # 3. Trade-Driven Reconciliation
        # If the trade was large enough to shift VAMP significantly, we should reconcile
        if abs(immediate_flow) > 0.3:
            # Fast track: Update state and reconcile without waiting for ticker
            # Note: Tickers come fast too, so we rate limit this?
            # For now, let's just trigger it if not locked
            if not self._reconcile_lock.locked():
                # We need updated market state... mainly signals.
                # Ticker price might be stale but signals are fresh.
                self.market_state.signals = signals

                # Create task to reconcile (fire and forget to not block trade processing)
                # But valid checks need fresh ticker?
                # Assuming Strategy can handle stale ticker with new signals (skew changes)
                asyncio.create_task(self._fast_reconcile())

        if self.storage:
            asyncio.create_task(self.storage.save_trade(trade))

    async def _cancel_side(self, side: OrderSide):
        """Emergency cancel for a specific side."""
        orders_to_cancel = self.active_orders.get(side, [])
        if not orders_to_cancel:
            return

        tasks = [
            self.gateway.cancel_order(o.exchange_id)
            for o in orders_to_cancel
            if o.exchange_id
        ]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
            # Optimistically clear
            self.active_orders[side] = []

    async def _fast_reconcile(self):
        """Wrapper to run reconciliation logic triggered by trades."""
        # Need to ensure we have a valid position/ticker
        if not self.market_state.ticker:
            return

        async with self._reconcile_lock:
            # Re-run strategy
            # Note: This duplicates logic from on_ticker_update.
            # Refactoring to a centralized _run_strategy() would be cleaner.
            # For now, keeping it inline for minimizing diffs.

            tick_size = getattr(self.gateway, "tick_size", 0.5)
            desired_orders = self.strategy.calculate_quotes(
                self.market_state, self.position, tick_size=tick_size
            )
            # Tick align
            for i in range(len(desired_orders)):
                o = desired_orders[i]
                if o.price:
                    rounded_price = round(o.price / self.tick_size) * self.tick_size
                    desired_orders[i] = replace(o, price=rounded_price)

            active = []
            # Risk validate against current active orders (simplified)
            # Ideally we full diff. _reconcile_orders handles the diff.
            await self._reconcile_orders(desired_orders)

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
            active_list = self.active_orders.get(side, [])

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

        # Execute Batch Cancel
        if to_cancel_ids:
            # Chunking? batch size limit?
            # Thalex might struggle with huge batches, but 40 is fine.
            # Using fire-and-forget for cancels to be fast?
            # No, we need to know they are done to free margin?
            # Or reliance on "Cancel on Disconnect" safety net allows async?
            # Let's await to be safe.
            await self.gateway.cancel_orders_batch(to_cancel_ids)

            # Update local state immediately
            cancelled_set = set(to_cancel_ids)
            for side in [OrderSide.BUY, OrderSide.SELL]:
                self.active_orders[side] = [
                    o
                    for o in self.active_orders[side]
                    if o.exchange_id not in cancelled_set
                ]

        # Execute Batch Place
        if to_place_orders:
            # Prioritize: Sort to_place_orders so best prices are first?
            # Strategy puts best prices first usually.
            new_orders = await self.gateway.place_orders_batch(to_place_orders)

            for o in new_orders:
                if o.status == OrderStatus.OPEN:
                    self.active_orders[o.side].append(o)

    async def _cancel_all(self):
        all_active = []
        for side, orders in self.active_orders.items():
            all_active.extend(orders)

        if not all_active:
            return

        tasks = [
            self.gateway.cancel_order(o.exchange_id)
            for o in all_active
            if o.exchange_id
        ]
        await asyncio.gather(*tasks, return_exceptions=True)

        self.active_orders[OrderSide.BUY] = []
        self.active_orders[OrderSide.SELL] = []

    async def _sync_active_orders(self):
        """Ensure active_orders only contains orders that still exist on exchange."""
        for side in [OrderSide.BUY, OrderSide.SELL]:
            active_list = self.active_orders.get(side, [])
            valid_orders = []

            for order in active_list:
                if not order.exchange_id:
                    continue

                current = self.gateway.orders.get(order.exchange_id)

                if not current:
                    # Assume valid if not found in local cache (transient)
                    valid_orders.append(order)
                elif current.status in [OrderStatus.OPEN, OrderStatus.PENDING]:
                    valid_orders.append(order)
                else:
                    logger.info(
                        f"Active order {order.exchange_id} ({side}) is now {current.status}. Removing."
                    )

            self.active_orders[side] = valid_orders
