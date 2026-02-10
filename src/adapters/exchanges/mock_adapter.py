import time
import uuid
import asyncio
import logging
from typing import List, Optional, Callable
from dataclasses import replace

from ...domain.interfaces import ExchangeGateway
from ...domain.entities import (
    Order,
    OrderSide,
    OrderStatus,
    Position,
    Ticker,
    Trade,
    Balance,
)
from ...domain.lob_match_engine import LOBMatchEngine
from ...domain.entities.pnl import FillEffect

logger = logging.getLogger(__name__)


class MockExchangeGateway(ExchangeGateway):
    def __init__(
        self,
        real_adapter: ExchangeGateway,
        initial_balance: float = 10000.0,
        latency_ms: float = 50.0,
        slippage_ticks: float = 0.5,
        maker_fee: float = -0.0001,
        taker_fee: float = 0.0003,
        tick_size: float = 0.01,
    ):
        self._real = real_adapter
        self._initial_balance = initial_balance

        self._engine = LOBMatchEngine(
            latency_ms=latency_ms,
            slippage_ticks=slippage_ticks,
            tick_size=tick_size,
            maker_fee=maker_fee,
            taker_fee=taker_fee,
        )
        self._engine.balance = initial_balance
        self._engine.fill_callback = self._on_sim_fill

        self._ticker_callback: Optional[Callable] = None
        self._trade_callback: Optional[Callable] = None
        self._order_callback: Optional[Callable] = None
        self._position_callback: Optional[Callable] = None
        self._balance_callback: Optional[Callable] = None
        self._execution_callback: Optional[Callable] = None

        self._last_ticker: Optional[Ticker] = None
        self._order_map: dict = {}
        self._tick_size = tick_size

    @property
    def name(self) -> str:
        return f"mock_{self._real.name}"

    @property
    def tick_size(self) -> float:
        return self._tick_size

    @property
    def connected(self) -> bool:
        return self._real.connected

    @connected.setter
    def connected(self, value: bool):
        pass

    @property
    def is_ready(self) -> bool:
        return self._real.is_ready

    @property
    def is_reconnecting(self) -> bool:
        return getattr(self._real, "is_reconnecting", False)

    @is_reconnecting.setter
    def is_reconnecting(self, value: bool):
        pass

    async def connect(self):
        self._real.set_ticker_callback(self._on_real_ticker)
        self._real.set_trade_callback(self._on_real_trade)
        await self._real.connect()
        logger.info(
            f"MockExchangeGateway connected via {self._real.name} | "
            f"Balance: {self._initial_balance:.2f} | "
            f"Latency: {self._engine.latency_ms}ms | "
            f"Tick: {self._tick_size}"
        )

    async def disconnect(self):
        state = self._engine.get_state()
        logger.info(
            f"MockExchangeGateway disconnecting | "
            f"Final Balance: {state['balance']:.4f} | "
            f"Position: {state['position_size']:.4f} | "
            f"Total Fills: {state['total_fills']}"
        )
        await self._real.disconnect()

    async def get_server_time(self) -> int:
        return await self._real.get_server_time()

    async def subscribe_ticker(self, symbol: str):
        await self._real.subscribe_ticker(symbol)

    async def get_balances(self) -> List[Balance]:
        return [
            Balance(
                exchange=self._real.name if hasattr(self._real, "name") else "mock",
                asset="USDT",
                total=self._engine.balance,
                available=self._engine.balance,
            )
        ]

    async def get_position(self, symbol: str) -> Position:
        return Position(
            symbol=symbol,
            size=self._engine.position_size,
            entry_price=self._engine.position_entry_price,
        )

    async def place_order(self, order: Order) -> Order:
        sim_id = f"sim_{uuid.uuid4().hex[:12]}"
        now = time.time()

        placed = replace(
            order,
            exchange_id=sim_id,
            status=OrderStatus.OPEN,
        )

        self._order_map[sim_id] = placed
        self._engine.submit_order(placed, now)

        if self._last_ticker:
            self._engine.on_ticker(self._last_ticker)

        logger.debug(
            f"MOCK ORDER: {placed.side.value} {placed.size:.4f} @ {placed.price:.2f} [{sim_id}]"
        )
        return placed

    async def cancel_order(self, order_id: str) -> bool:
        success = self._engine.cancel_order(order_id)
        if success:
            self._order_map.pop(order_id, None)
        return success

    async def place_orders_batch(self, orders: List[Order]) -> List[Order]:
        results = []
        for order in orders:
            placed = await self.place_order(order)
            results.append(placed)
        return results

    async def cancel_orders_batch(self, order_ids: List[str]) -> List[bool]:
        results = []
        for oid in order_ids:
            results.append(await self.cancel_order(oid))
        return results

    async def get_open_orders(self, symbol: str) -> List[Order]:
        return [o for o in self._engine.get_open_orders() if o.symbol == symbol]

    def set_ticker_callback(self, callback: Callable):
        self._ticker_callback = callback

    def set_trade_callback(self, callback: Callable):
        self._trade_callback = callback

    def set_order_callback(self, callback: Callable):
        self._order_callback = callback

    def set_position_callback(self, callback: Callable):
        self._position_callback = callback

    def set_balance_callback(self, callback: Callable):
        self._balance_callback = callback

    def set_execution_callback(self, callback: Callable):
        self._execution_callback = callback

    async def _on_real_ticker(self, ticker: Ticker):
        self._last_ticker = ticker

        fills_before = len(self._engine.fills)
        self._engine.on_ticker(ticker)
        new_fills = self._engine.fills[fills_before:]

        for fill in new_fills:
            await self._dispatch_fill_events(fill)

        if self._ticker_callback:
            await self._ticker_callback(ticker)

    async def _on_real_trade(self, trade: Trade):
        if self._trade_callback:
            await self._trade_callback(trade)

    async def _on_sim_fill(self, fill: FillEffect):
        await self._dispatch_fill_events(fill)

    async def _dispatch_fill_events(self, fill: FillEffect):
        if self._position_callback:
            await self._position_callback(
                fill.symbol,
                self._engine.position_size,
                self._engine.position_entry_price,
            )

        if self._execution_callback:
            sim_id = f"sim_fill_{uuid.uuid4().hex[:8]}"
            await self._execution_callback(sim_id, fill.price, fill.size)

    def get_sim_state(self) -> dict:
        state = self._engine.get_state()
        if self._last_ticker:
            mid = (self._last_ticker.bid + self._last_ticker.ask) / 2
            state["equity"] = self._engine.get_equity(mid)
            state["unrealized_pnl"] = state["equity"] - state["balance"]
        return state
