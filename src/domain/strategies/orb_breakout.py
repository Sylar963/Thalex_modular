import logging
from typing import Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class TradeDirection(str, Enum):
    LONG = "long"
    SHORT = "short"


@dataclass(slots=True)
class TradeSignal:
    symbol: str
    direction: TradeDirection
    entry_price: float
    orh: float
    orl: float
    orm: float
    orw: float
    base_unit_size: float
    timestamp: float
    first_up_target: float = 0.0
    first_down_target: float = 0.0


class ORBBreakoutStrategy:
    __slots__ = [
        "risk_pct",
        "leverage",
        "max_concurrent",
        "active_count",
        "_last_signals",
    ]

    def __init__(
        self,
        risk_pct: float = 17.0,
        leverage: int = 10,
        max_concurrent: int = 5,
    ):
        self.risk_pct = risk_pct / 100.0
        self.leverage = leverage
        self.max_concurrent = max_concurrent
        self.active_count = 0
        self._last_signals: Dict[str, float] = {}

    def compute_unit_size(self, equity: float, price: float) -> float:
        if price <= 0 or equity <= 0:
            return 0.0
        return (equity * self.risk_pct * self.leverage) / price

    def evaluate(
        self,
        symbol: str,
        or_signals: Dict[str, Any],
        current_price: float,
        timestamp: float,
        equity: float,
    ) -> Optional[TradeSignal]:
        if self.active_count >= self.max_concurrent:
            return None

        session_active = or_signals.get("session_active", 0.0)
        if session_active == 1.0:
            return None

        or_token = or_signals.get("or_token", 0.0)
        if or_token != 1.0:
            return None

        breakout_signal = or_signals.get("breakout_signal", 0.0)

        prev_signal = self._last_signals.get(symbol, 0.0)
        self._last_signals[symbol] = breakout_signal

        if breakout_signal == prev_signal:
            return None

        if breakout_signal == 0.0:
            return None

        orh = or_signals.get("orh", 0.0)
        orl = or_signals.get("orl", 0.0)
        orm = or_signals.get("orm", 0.0)
        orw = or_signals.get("orw", 0.0)

        if orh <= 0 or orl <= 0 or orm <= 0 or orw <= 0:
            return None

        direction = TradeDirection.LONG if breakout_signal > 0 else TradeDirection.SHORT
        unit_size = self.compute_unit_size(equity, current_price)

        if unit_size <= 0:
            return None

        first_up = or_signals.get("first_up_target", 0.0)
        first_down = or_signals.get("first_down_target", 0.0)

        logger.info(
            f"ORB Signal: {direction.value.upper()} {symbol} "
            f"size={unit_size:.4f} price={current_price:.2f} "
            f"ORH={orh:.2f} ORL={orl:.2f}"
        )

        return TradeSignal(
            symbol=symbol,
            direction=direction,
            entry_price=current_price,
            orh=orh,
            orl=orl,
            orm=orm,
            orw=orw,
            base_unit_size=unit_size,
            timestamp=timestamp,
            first_up_target=first_up,
            first_down_target=first_down,
        )

    def on_trade_opened(self):
        self.active_count += 1

    def on_trade_closed(self):
        self.active_count = max(0, self.active_count - 1)

    def setup(self, config: Dict[str, Any]):
        self.risk_pct = config.get("risk_pct_per_trade", 17.0) / 100.0
        self.leverage = config.get("leverage", 10)
        self.max_concurrent = config.get("max_concurrent_positions", 5)
        logger.info(
            f"ORB Strategy: risk={self.risk_pct:.0%} lever={self.leverage}x "
            f"max_concurrent={self.max_concurrent}"
        )
