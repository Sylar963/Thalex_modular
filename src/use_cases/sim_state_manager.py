import asyncio
import time
import uuid
import logging
from typing import List, Dict, Optional, AsyncGenerator
from dataclasses import dataclass, field
from ..domain.entities.pnl import EquitySnapshot, FillEffect
from ..domain.entities import Ticker, MarketState, Position, Order, Trade, OrderSide
from ..domain.lob_match_engine import LOBMatchEngine
from ..domain.strategies.avellaneda import AvellanedaStoikovStrategy
from ..domain.risk.basic_manager import BasicRiskManager
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

        # Modular Components
        self.match_engine: Optional[LOBMatchEngine] = None
        self.strategy: Optional[AvellanedaStoikovStrategy] = None
        self.risk_manager: Optional[BasicRiskManager] = None
        self.storage: Optional[StorageGateway] = None
        self._loop_task: Optional[asyncio.Task] = None

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
                position_size=0.0,
                position_entry_price=0.0,
                equity_history=[],
                fills=[],
                last_price=0.0,
                mark_price=0.0,
            )

            # Initialize Engines
            self.match_engine = LOBMatchEngine()
            self.match_engine.balance = initial_balance
            self.match_engine.fill_callback = self.record_fill

            self.strategy = AvellanedaStoikovStrategy()
            # Default config, should ideally come from params
            self.strategy.setup(
                {"gamma": 0.5, "volatility": 0.01, "position_limit": 5.0}
            )

            self.risk_manager = BasicRiskManager()
            self.risk_manager.setup({"max_position": 5.0})

            # Start the processing loop
            self._loop_task = asyncio.create_task(self._simulation_loop())

            logger.info(
                f"SimStateManager started: {symbol} with {initial_balance} balance in {mode} mode"
            )

    async def stop(self):
        async with self._lock:
            self.state.running = False
            if self._loop_task:
                self._loop_task.cancel()
                try:
                    await self._loop_task
                except asyncio.CancelledError:
                    pass
                self._loop_task = None

            for q in self._subscribers:
                await q.put(None)
            self._subscribers.clear()
            logger.info("SimStateManager stopped")

    async def _simulation_loop(self):
        """
        Main loop to verify liveliness.
        """
        logger.info("Live Simulation Loop Started")
        try:
            while self.state.running:
                await asyncio.sleep(1.0)
        except asyncio.CancelledError:
            logger.info("Live Simulation Loop Cancelled")
        except Exception as e:
            logger.error(f"Error in Live Simulation Loop: {e}")

    async def on_ticker(self, ticker: Ticker):
        """
        Called by MarketFeedService when a new ticker arrives.
        This drives the simulation.
        """
        if not self.state.running or ticker.symbol != self.state.symbol:
            return

        async with self._lock:
            # 1. Update Market State
            self.state.last_price = ticker.last
            # Prefer mark_price from ticker if available
            self.state.mark_price = ticker.mark_price if ticker.mark_price > 0 else ticker.last

            self.match_engine.on_ticker(ticker)

            # 2. Strategy Logic
            market_state = MarketState(ticker=ticker, timestamp=ticker.timestamp)
            pos = Position(
                self.state.symbol,
                self.match_engine.position_size,
                self.match_engine.position_entry_price,
            )

            quotes = self.strategy.calculate_quotes(market_state, pos)

            # 3. Submit Orders to Match Engine
            self.match_engine.bid_book.clear()
            self.match_engine.ask_book.clear()

            for q in quotes:
                if self.risk_manager.validate_order(q, pos):
                    self.match_engine.submit_order(q, ticker.timestamp)

            # 4. Update Equity Snapshot
            # Use mid price or mark price for equity calc? 
            # Usually equity is balance + UPNL. UPNL uses mark price.
            # But get_equity in match_engine might use mid.
            # Let's standardize on mark price if available for UPNL part.
            
            current_price = self.state.mark_price if self.state.mark_price > 0 else (ticker.bid + ticker.ask) / 2
            current_equity = self.match_engine.balance + (current_price - self.match_engine.position_entry_price) * self.match_engine.position_size
            
            snapshot = EquitySnapshot(
                timestamp=ticker.timestamp,
                balance=self.match_engine.balance,
                position_value=0,  # Simplified
                equity=current_equity,
                unrealized_pnl=current_equity - self.match_engine.balance,
            )

            # Update internal state (lighter update)
            self.state.position_size = self.match_engine.position_size
            self.state.position_entry_price = self.match_engine.position_entry_price
            self.state.current_balance = self.match_engine.balance

            await self.record_equity_snapshot(snapshot)

    async def record_fill(self, fill: FillEffect):
        """
        Callback from LOBMatchEngine.
        Persists fill as a mock Trade in bot_executions.
        """
        self.state.fills.append(fill)
        self.state.current_balance = fill.balance_after

        if len(self.state.fills) > self.max_history:
            self.state.fills = self.state.fills[-self.max_history :]

        logger.info(f"Live Sim Fill: {fill.side} {fill.size} @ {fill.price}")

        # Persist to DB if storage is available
        if self.storage:
            try:
                # Map FillEffect to Trade
                trade = Trade(
                    id=str(uuid.uuid4()),
                    order_id=str(uuid.uuid4()),  # Mock Order ID
                    symbol=fill.symbol,
                    price=fill.price,
                    size=fill.size,
                    side=OrderSide(fill.side),
                    timestamp=fill.timestamp,
                    exchange=f"sim_{self.state.mode}",  # Distinguish from real fills
                    fee=fill.fee,
                )
                asyncio.create_task(self.storage.save_execution(trade))
            except Exception as e:
                logger.error(f"Failed to persist sim execution: {e}")

    async def record_equity_snapshot(self, snapshot: EquitySnapshot):
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
        # Prefer mark_price for UPNL
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
