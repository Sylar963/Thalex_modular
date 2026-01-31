import asyncio
import logging
import time
from typing import List, Optional, Dict
from ..domain.interfaces import ExchangeGateway, Strategy, SignalEngine, RiskManager
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
    ):
        self.gateway = gateway
        self.strategy = strategy
        self.signal_engine = signal_engine
        self.risk_manager = risk_manager

        self.symbol: str = ""
        self.market_state = MarketState()
        self.position = Position("", 0.0, 0.0)
        self.active_orders: Dict[
            OrderSide, Order
        ] = {}  # Side -> Order (Assuming simple 1-level quoting)
        self.running = False
        self.tick_size = 0.5  # Default, should update from symbol info

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
        self.tick_size = 0.5  # TODO: Fetch from config or instrument info

        # Refresh position from local cache of adapter (fast)
        self.position = await self.gateway.get_position(self.symbol)

        # Update Signals
        self.signal_engine.update(ticker)
        self.market_state.signals = self.signal_engine.get_signals()

        # 2. Risk Check (Global)
        if not self.risk_manager.can_trade():
            await self._cancel_all()
            return

        # 3. Strategy Calculation
        desired_orders = self.strategy.calculate_quotes(
            self.market_state, self.position
        )

        # 4. Risk Validation
        valid_orders = [
            o
            for o in desired_orders
            if self.risk_manager.validate_order(o, self.position)
        ]

        # 5. Execution (Diff against active orders)
        await self._reconcile_orders(valid_orders)

    async def on_trade_update(self, trade):
        if not self.running:
            return
        self.signal_engine.update_trade(trade)

    async def _reconcile_orders(self, desired_orders: List[Order]):
        """
        Diff desired orders against active orders and execute changes.
        Assumes at most one order per side (simple MM).
        """
        desired_by_side = {o.side: o for o in desired_orders}

        # Sides to process: Buy and Sell
        for side in [OrderSide.BUY, OrderSide.SELL]:
            desired = desired_by_side.get(side)
            active = self.active_orders.get(side)

            if desired and active:
                # Compare functionality
                # Tolerance check (avoid churn if price is close)
                # But for MM, price exactness matters.
                if (
                    abs(desired.price - active.price) < 1e-9
                    and abs(desired.size - active.size) < 1e-9
                ):
                    continue  # No change needed

                # Update needed -> Cancel old, Place new
                # (Ideally use 'amend_order' if supported)
                await self.gateway.cancel_order(active.exchange_id)
                new_order = await self.gateway.place_order(desired)
                if new_order.status != OrderStatus.REJECTED:
                    self.active_orders[side] = new_order
                else:
                    del self.active_orders[side]  # Failed to replace

            elif desired and not active:
                # Place new
                new_order = await self.gateway.place_order(desired)
                if new_order.status != OrderStatus.REJECTED:
                    self.active_orders[side] = new_order

            elif not desired and active:
                # Cancel existing
                await self.gateway.cancel_order(active.exchange_id)
                del self.active_orders[side]

    async def _cancel_all(self):
        sides = list(self.active_orders.keys())
        for side in sides:
            order = self.active_orders[side]
            if order.exchange_id:
                await self.gateway.cancel_order(order.exchange_id)
            del self.active_orders[side]
