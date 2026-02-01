from dataclasses import dataclass, field
from typing import List, Dict, Optional
from . import OrderSide


@dataclass(slots=True)
class EquitySnapshot:
    timestamp: float
    balance: float
    position_value: float
    equity: float
    unrealized_pnl: float


@dataclass(slots=True)
class FillEffect:
    timestamp: float
    symbol: str
    side: str
    price: float
    size: float
    fee: float
    realized_pnl: float
    balance_after: float


@dataclass(slots=True)
class SimStats:
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl: float
    sharpe_ratio: float
    max_drawdown: float
    profit_factor: float
    avg_trade_pnl: float


@dataclass(slots=True)
class SimulationResult:
    run_id: str
    start_time: float
    end_time: float
    config: Dict
    equity_curve: List[EquitySnapshot] = field(default_factory=list)
    fills: List[FillEffect] = field(default_factory=list)
    stats: Optional[SimStats] = None
