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
        self.active_orders: Dict[
            OrderSide, Order
        ] = {}  # Side -> Order (Assuming simple 1-level quoting)
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
        desired_orders = self.strategy.calculate_quotes(
            self.market_state, self.position
        )

        # 3.1 Force Tick Alignment
        for i in range(len(desired_orders)):
            o = desired_orders[i]
            if o.price:
                rounded_price = round(o.price / self.tick_size) * self.tick_size
                desired_orders[i] = replace(o, price=rounded_price)

        # 4. Risk Validation
        valid_orders = [
            o
            for o in desired_orders
            if self.risk_manager.validate_order(o, self.position)
        ]

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
        Optimized for margin: cancel all necessary sides in parallel FIRST, then place new ones.
        """
        desired_by_side = {o.side: o for o in desired_orders}
        to_cancel = []
        to_place = []

        # 1. Identify what needs to be cancelled vs placed
        for side in [OrderSide.BUY, OrderSide.SELL]:
            desired = desired_by_side.get(side)
            active = self.active_orders.get(side)

            if desired and active:
                if (
                    abs(desired.price - active.price) < 1e-9
                    and abs(desired.size - active.size) < 1e-9
                ):
                    continue  # side is ok

                # Update needed
                to_cancel.append((side, active.exchange_id))
                to_place.append((side, desired))

            elif desired and not active:
                to_place.append((side, desired))

            elif not desired and active:
                to_cancel.append((side, active.exchange_id))

        # 2. Cancel ALL required sides in parallel to free up margin fast
        if to_cancel:
            cancel_tasks = [self.gateway.cancel_order(eid) for side, eid in to_cancel]
            results = await asyncio.gather(*cancel_tasks, return_exceptions=True)
            for (side, eid), res in zip(to_cancel, results):
                if isinstance(res, Exception):
                    logger.error(f"Error cancelling {side} order {eid}: {res}")
                # Even if cancel failed (e.g. already filled), we remove from active
                if side in self.active_orders:
                    del self.active_orders[side]

        # 3. Place new orders
        for side, desired in to_place:
            new_order = await self.gateway.place_order(desired)
            if new_order.status != OrderStatus.REJECTED:
                self.active_orders[side] = new_order
            else:
                # If placement failed, ensure we don't think it's still active
                if side in self.active_orders:
                    del self.active_orders[side]

    async def _cancel_all(self):
        sides = list(self.active_orders.keys())
        for side in sides:
            order = self.active_orders[side]
            if order.exchange_id:
                await self.gateway.cancel_order(order.exchange_id)
            del self.active_orders[side]

    async def _sync_active_orders(self):
        """Ensure active_orders only contains orders that still exist on exchange."""
        sides = list(self.active_orders.keys())
        for side in sides:
            order = self.active_orders[side]
            if not order.exchange_id:
                continue

            # Check gateway's view
            current = self.gateway.orders.get(order.exchange_id)
            if current and current.status not in [
                OrderStatus.OPEN,
                OrderStatus.PENDING,
            ]:
                logger.info(
                    f"Active order {order.exchange_id} ({side}) is now {current.status}. Removing."
                )
                del self.active_orders[side]
            elif not current:
                # This could happen if we reconnected and lost local cache.
                pass
