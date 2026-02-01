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
        self.position = await self.gateway.get_position(self.symbol)

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
        self.signal_engine.update_trade(trade)
        if self.storage:
            asyncio.create_task(self.storage.save_trade(trade))

    async def _reconcile_orders(self, desired_orders: List[Order]):
        """
        Diff desired orders against active orders and execute changes.
        Optimized for margin: cancel all unnecessary/moved orders FIRST, then place new ones.
        """
        desired_by_side = {OrderSide.BUY: [], OrderSide.SELL: []}
        for o in desired_orders:
            desired_by_side[o.side].append(o)

        to_cancel = []
        to_place = []

        # 1. Identify diff per side
        for side in [OrderSide.BUY, OrderSide.SELL]:
            desired_list = desired_by_side.get(side, [])
            active_list = self.active_orders.get(side, [])

            # Simple diff strategy:
            # Match strictly by price and size.
            # Anything in Active NOT in Desired -> Cancel
            # Anything in Desired NOT in Active -> Place

            # We need to match objects.
            # Let's map active orders by "signature" (price, size)
            # Warning: Float comparison needs tolerance

            active_map = []  # List of (order, is_matched)
            for act in active_list:
                active_map.append({"order": act, "matched": False})

            desired_to_place_indices = []

            for i, des in enumerate(desired_list):
                found = False
                for j, item in enumerate(active_map):
                    if item["matched"]:
                        continue
                    act = item["order"]
                    if (
                        abs(des.price - act.price) < 1e-9
                        and abs(des.size - act.size) < 1e-9
                    ):
                        # Match found, keep it
                        active_map[j]["matched"] = True
                        found = True
                        break

                if not found:
                    desired_to_place_indices.append(i)

            # Collect cancels
            for item in active_map:
                if not item["matched"]:
                    to_cancel.append((side, item["order"].exchange_id))

            # Collect places
            for i in desired_to_place_indices:
                to_place.append((side, desired_list[i]))

        # Logging for Observability
        if to_cancel or to_place:
            logger.info(
                f"Reconcile: Cancelling {len(to_cancel)}, Placing {len(to_place)} (Active: {sum(len(l) for l in self.active_orders.values())})"
            )

        # 2. Cancel ALL required sides in parallel to free up margin fast
        if to_cancel:
            cancel_tasks = [
                self.gateway.cancel_order(eid) for side, eid in to_cancel if eid
            ]
            if cancel_tasks:
                results = await asyncio.gather(*cancel_tasks, return_exceptions=True)
                for (side, eid), res in zip(to_cancel, results):
                    if isinstance(res, Exception):
                        logger.error(f"Error cancelling {side} order {eid}: {res}")

            # Remove from active_orders immediately (optimistic/authoritative)
            # We need to rebuild the active lists without the cancelled ones
            ids_to_remove = set(eid for side, eid in to_cancel)
            for side in [OrderSide.BUY, OrderSide.SELL]:
                self.active_orders[side] = [
                    o
                    for o in self.active_orders[side]
                    if o.exchange_id not in ids_to_remove
                ]

        # 3. Place new orders
        # We can also do this in parallel per side or globally? Globally is faster.
        if to_place:
            place_tasks = [
                self.gateway.place_order(desired) for side, desired in to_place
            ]
            results = await asyncio.gather(*place_tasks, return_exceptions=True)

            for (side, desired), res in zip(to_place, results):
                if isinstance(res, Exception):
                    logger.error(f"Error placing {side} order {desired}: {res}")
                elif isinstance(res, Order):
                    # Update status
                    if res.status != OrderStatus.REJECTED:
                        self.active_orders[side].append(res)

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
