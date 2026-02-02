import asyncio
import time
import logging
from typing import List, Dict, Optional, AsyncGenerator
from dataclasses import dataclass, field
from ..domain.entities.pnl import EquitySnapshot, FillEffect

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


class SimStateManager:
    def __init__(self, max_history: int = 10000):
        self.state = LiveSimState()
        self.max_history = max_history
        self._subscribers: List[asyncio.Queue] = []
        self._lock = asyncio.Lock()

    async def start(self, symbol: str, initial_balance: float, mode: str = "shadow"):
        async with self._lock:
            self.state = LiveSimState(
                running=True,
                mode=mode,
                start_time=time.time(),
                symbol=symbol,
                initial_balance=initial_balance,
                current_balance=initial_balance,
                position_size=0.0,
                position_entry_price=0.0,
                equity_history=[],
                fills=[],
                last_price=0.0,
            )
            logger.info(
                f"SimStateManager started: {symbol} with {initial_balance} balance in {mode} mode"
            )

    async def stop(self):
        async with self._lock:
            self.state.running = False
            for q in self._subscribers:
                await q.put(None)
            self._subscribers.clear()
            logger.info("SimStateManager stopped")

    async def update_price(self, price: float):
        async with self._lock:
            self.state.last_price = price

    async def update_position(self, size: float, entry_price: float, balance: float):
        async with self._lock:
            self.state.position_size = size
            self.state.position_entry_price = entry_price
            self.state.current_balance = balance

    async def record_fill(self, fill: FillEffect):
        async with self._lock:
            self.state.fills.append(fill)
            self.state.current_balance = fill.balance_after

            if len(self.state.fills) > self.max_history:
                self.state.fills = self.state.fills[-self.max_history :]

    async def record_equity_snapshot(self, snapshot: EquitySnapshot):
        async with self._lock:
            self.state.equity_history.append(snapshot)

            if len(self.state.equity_history) > self.max_history:
                self.state.equity_history = self.state.equity_history[
                    -self.max_history :
                ]

            for q in self._subscribers:
                try:
                    q.put_nowait(snapshot)
                except asyncio.QueueFull:
                    pass

    def get_status(self) -> Dict:
        s = self.state
        unrealized = 0.0
        if s.position_size != 0 and s.last_price > 0:
            unrealized = (s.last_price - s.position_entry_price) * s.position_size

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
