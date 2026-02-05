from typing import List, Callable, Optional, Dict
import asyncio
import time
from src.domain.interfaces import ExchangeGateway
from src.domain.entities import Order, OrderStatus, Ticker, Trade, Position, Balance
from src.domain.sim_match_engine import SimMatchEngine


class MockExchangeGateway(ExchangeGateway):
    def __init__(self, name: str, sim_engine: SimMatchEngine):
        self.name = name
        self.sim_engine = sim_engine
        self.connected = False

        self.ticker_callback: Optional[Callable] = None
        self.trade_callback: Optional[Callable] = None
        self.order_callback: Optional[Callable] = None
        self.position_callback: Optional[Callable] = None
        self.balance_callback: Optional[Callable] = None

        # Hook into SimEngine fills
        self.sim_engine.set_fill_callback(self._on_sim_fill)

    async def connect(self):
        self.connected = True

    async def disconnect(self):
        self.connected = False

    async def get_time(self) -> int:
        return int(time.time() * 1000)

    async def get_position(self, symbol: str) -> Position:
        return Position(
            symbol=symbol,
            size=self.sim_engine.position_size,
            entry_price=self.sim_engine.position_entry_price,
            exchange=self.name,
        )

    async def get_balances(self) -> List[Balance]:
        return [
            Balance(
                exchange=self.name,
                asset="USDT",
                total=self.sim_engine.balance,
                available=self.sim_engine.balance,
                timestamp=time.time(),
            )
        ]

    async def place_order(self, order: Order) -> Order:
        order.exchange_id = f"sim_{order.id}"
        order.status = OrderStatus.OPEN
        order.exchange = self.name

        # Submit to Sim Engine
        self.sim_engine.submit_order(order)

        # Ack immediately
        if self.order_callback:
            await self.order_callback(order.exchange_id, OrderStatus.OPEN, 0.0, 0.0)

        return order

    async def cancel_order(self, order_id: str) -> bool:
        success = self.sim_engine.cancel_order(order_id)
        if success and self.order_callback:
            await self.order_callback(order_id, OrderStatus.CANCELLED, 0.0, 0.0)
        return success

    async def cancel_all_orders(self, symbol: str) -> List[str]:
        self.sim_engine.cancel_all()
        return []

    async def place_orders_batch(self, orders: List[Order]) -> List[Order]:
        return [await self.place_order(o) for o in orders]

    async def cancel_orders_batch(self, order_ids: List[str]) -> List[bool]:
        return [await self.cancel_order(oid) for oid in order_ids]

    async def subscribe_ticker(self, symbol: str):
        pass  # Tickers driven externally for tests

    # --- Test Helpers ---

    async def push_ticker(self, ticker: Ticker):
        if self.ticker_callback:
            await self.ticker_callback(ticker)

    async def push_trade(self, trade: Trade):
        if self.trade_callback:
            await self.trade_callback(trade)
        # Also feed sim engine for matching
        self.sim_engine.on_trade(trade)

    async def _on_sim_fill(self, fill):
        # Notify StrategyManager of Fill
        # We need to map Sim fill to OrderStatus update
        # SimMatchEngine uses internal SimOrder, but we need exchange_id
        # For simplicity in this mock, we assume 1:1 mapping if possible or just generic update
        pass
        # Note: In a full mock, we'd map callbacks properly.
        # For this E2E, we might rely on strategy querying portfolio or basic ack.
        # But let's try to be proper:
        # We need the order_id. SimMatchEngine doesn't pass it back easily in fill struct
        # unless we modify it or lookup.
        # For now, we will rely on checking SimEngine state in assertions.

    def set_ticker_callback(self, callback: Callable):
        self.ticker_callback = callback

    def set_trade_callback(self, callback: Callable):
        self.trade_callback = callback

    def set_order_callback(self, callback: Callable):
        self.order_callback = callback

    def set_position_callback(self, callback: Callable):
        self.position_callback = callback

    def set_balance_callback(self, callback: Callable):
        self.balance_callback = callback
