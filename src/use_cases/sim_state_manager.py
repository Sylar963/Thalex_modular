import asyncio
import time
import uuid
import logging
from typing import List, Dict, Optional, AsyncGenerator
from dataclasses import dataclass, field
from ..domain.entities.pnl import EquitySnapshot, FillEffect
from ..domain.entities import Ticker, Trade, OrderSide
from ..domain.interfaces import StorageGateway

logger = logging.getLogger(__name__)


@dataclass
class LiveSimState:
    running: bool = False
    mode: str = "shadow"
    start_time: float = 0.0
    symbol: str = ""
    initial_balance: float = 0.0
    current_balance: float = 0.0
    position_size: float = 0.0
    position_entry_price: float = 0.0
    equity_history: List[EquitySnapshot] = field(default_factory=list)
    fills: List[FillEffect] = field(default_factory=list)
    last_price: float = 0.0
    mark_price: float = 0.0


class SimStateManager:
    def __init__(self, max_history: int = 10000):
        self.state = LiveSimState()
        self.max_history = max_history
        self._subscribers: List[asyncio.Queue] = []
        self._lock = asyncio.Lock()
        self.storage: Optional[StorageGateway] = None
        self._snapshot_task: Optional[asyncio.Task] = None

    def set_storage(self, storage: StorageGateway):
        self.storage = storage

    async def start(self, symbol: str, initial_balance: float, mode: str = "shadow"):
        async with self._lock:
            if self.state.running:
                logger.warning("Live Simulation already running")
                return

            self.state = LiveSimState(
                running=True,
                mode=mode,
                start_time=time.time(),
                symbol=symbol,
                initial_balance=initial_balance,
                current_balance=initial_balance,
            )

            self._snapshot_task = asyncio.create_task(self._periodic_snapshot())
            logger.info(
                f"SimStateManager started: {symbol} | "
                f"balance={initial_balance} | mode={mode}"
            )

    async def stop(self):
        async with self._lock:
            self.state.running = False
            if self._snapshot_task:
                self._snapshot_task.cancel()
                try:
                    await self._snapshot_task
                except asyncio.CancelledError:
                    pass
                self._snapshot_task = None

            for q in self._subscribers:
                await q.put(None)
            self._subscribers.clear()
            logger.info("SimStateManager stopped")

    async def _periodic_snapshot(self):
        try:
            while self.state.running:
                await asyncio.sleep(1.0)
                if self.state.last_price > 0:
                    price = (
                        self.state.mark_price
                        if self.state.mark_price > 0
                        else self.state.last_price
                    )
                    unrealized = (
                        (price - self.state.position_entry_price)
                        * self.state.position_size
                        if self.state.position_size != 0
                        else 0.0
                    )
                    equity = self.state.current_balance + unrealized

                    snapshot = EquitySnapshot(
                        timestamp=time.time(),
                        balance=self.state.current_balance,
                        position_value=abs(self.state.position_size) * price,
                        equity=equity,
                        unrealized_pnl=unrealized,
                    )
                    await self._record_equity_snapshot(snapshot)
        except asyncio.CancelledError:
            pass

    async def on_ticker(self, ticker: Ticker):
        if not self.state.running or ticker.symbol != self.state.symbol:
            return
        self.state.last_price = ticker.last
        self.state.mark_price = (
            ticker.mark_price if ticker.mark_price > 0 else ticker.last
        )

    async def on_position_update(self, symbol: str, size: float, entry_price: float):
        if not self.state.running or symbol != self.state.symbol:
            return
        self.state.position_size = size
        self.state.position_entry_price = entry_price

    async def on_balance_update(self, balance: float):
        if not self.state.running:
            return
        self.state.current_balance = balance

    async def record_fill(self, fill: FillEffect):
        self.state.fills.append(fill)
        self.state.current_balance = fill.balance_after

        if len(self.state.fills) > self.max_history:
            self.state.fills = self.state.fills[-self.max_history :]

        logger.info(
            f"Shadow Fill: {fill.side} {fill.size:.4f} @ {fill.price:.2f} | PNL: {fill.realized_pnl:.4f}"
        )

        if self.storage:
            try:
                trade = Trade(
                    id=str(uuid.uuid4()),
                    order_id=str(uuid.uuid4()),
                    symbol=fill.symbol,
                    price=fill.price,
                    size=fill.size,
                    side=OrderSide(fill.side),
                    timestamp=fill.timestamp,
                    exchange=f"sim_{self.state.mode}",
                    fee=fill.fee,
                )
                asyncio.create_task(self.storage.save_execution(trade))
            except Exception as e:
                logger.error(f"Failed to persist sim execution: {e}")

    async def _record_equity_snapshot(self, snapshot: EquitySnapshot):
        self.state.equity_history.append(snapshot)

        if len(self.state.equity_history) > self.max_history:
            self.state.equity_history = self.state.equity_history[-self.max_history :]

        for q in self._subscribers:
            try:
                q.put_nowait(snapshot)
            except asyncio.QueueFull:
                pass

    def get_status(self) -> Dict:
        s = self.state
        unrealized = 0.0
        price_for_upnl = s.mark_price if s.mark_price > 0 else s.last_price

        if s.position_size != 0 and price_for_upnl > 0:
            unrealized = (price_for_upnl - s.position_entry_price) * s.position_size

        return {
            "running": s.running,
            "mode": s.mode,
            "symbol": s.symbol,
            "start_time": s.start_time,
            "uptime_seconds": time.time() - s.start_time if s.running else 0,
            "initial_balance": s.initial_balance,
            "current_balance": s.current_balance,
            "position_size": s.position_size,
            "position_entry_price": s.position_entry_price,
            "unrealized_pnl": unrealized,
            "equity": s.current_balance + unrealized,
            "total_fills": len(s.fills),
            "last_price": s.last_price,
            "mark_price": s.mark_price,
        }

    def get_fills(self, limit: int = 100) -> List[Dict]:
        fills = self.state.fills[-limit:] if limit else self.state.fills
        return [
            {
                "timestamp": f.timestamp,
                "symbol": f.symbol,
                "side": f.side,
                "price": f.price,
                "size": f.size,
                "fee": f.fee,
                "realized_pnl": f.realized_pnl,
                "balance_after": f.balance_after,
            }
            for f in fills
        ]

    def get_equity_history(self, limit: int = 1000) -> List[Dict]:
        history = (
            self.state.equity_history[-limit:] if limit else self.state.equity_history
        )
        return [
            {
                "timestamp": e.timestamp,
                "balance": e.balance,
                "position_value": e.position_value,
                "equity": e.equity,
                "unrealized_pnl": e.unrealized_pnl,
            }
            for e in history
        ]

    async def subscribe_equity(self) -> AsyncGenerator[EquitySnapshot, None]:
        queue: asyncio.Queue = asyncio.Queue(maxsize=100)
        self._subscribers.append(queue)
        try:
            while True:
                snapshot = await queue.get()
                if snapshot is None:
                    break
                yield snapshot
        finally:
            if queue in self._subscribers:
                self._subscribers.remove(queue)


sim_state_manager = SimStateManager()
