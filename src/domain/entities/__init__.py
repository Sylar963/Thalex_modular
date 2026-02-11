from dataclasses import dataclass, field, replace
from typing import Optional, List, Dict
from enum import Enum
import time


class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    LIMIT = "limit"
    MARKET = "market"


class OrderStatus(str, Enum):
    PENDING = "pending"
    OPEN = "open"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass(slots=True)
class Instrument:
    symbol: str
    base_currency: str
    quote_currency: str
    tick_size: float
    min_size: float
    contract_size: float = 1.0


@dataclass(slots=True)
class Ticker:
    symbol: str
    bid: float
    ask: float
    bid_size: float
    ask_size: float
    last: float
    volume: float
    exchange: str = "thalex"
    mark_price: float = 0.0
    index_price: float = 0.0
    timestamp: float = field(default_factory=time.time)

    @property
    def mid_price(self) -> float:
        return (self.bid + self.ask) / 2.0

    @property
    def spread(self) -> float:
        return self.ask - self.bid


@dataclass(slots=True)
class Order:
    id: str
    symbol: str
    side: OrderSide
    price: float
    size: float
    type: OrderType = OrderType.LIMIT
    status: OrderStatus = OrderStatus.PENDING
    exchange: str = "thalex"
    exchange_id: Optional[str] = None
    filled_size: float = 0.0
    post_only: bool = False
    timestamp: float = field(default_factory=time.time)


@dataclass(slots=True)
class Trade:
    id: str
    order_id: str
    symbol: str
    side: OrderSide
    price: float
    size: float
    exchange: str = "thalex"
    fee: float = 0.0
    timestamp: float = field(default_factory=time.time)


@dataclass(slots=True)
class Position:
    symbol: str
    size: float
    entry_price: float
    exchange: str = "thalex"
    mark_price: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0
    timestamp: float = field(default_factory=time.time)

    def update_upnl(self, current_price: float) -> "Position":
        """
        Updates mark_price and recalculates unrealized_pnl.
        Returns a NEW Position object (immutable pattern).
        """
        if current_price <= 0:
            return self

        new_upnl = 0.0
        if self.size != 0:
            # Linear PnL Formula
            new_upnl = (current_price - self.entry_price) * self.size

        return replace(
            self,
            mark_price=current_price,
            unrealized_pnl=new_upnl,
            timestamp=time.time(),
        )


@dataclass(slots=True)
class MarketState:
    ticker: Optional[Ticker] = None
    orderbook: Optional[Dict] = None
    signals: Dict[str, float] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass(slots=True)
class Balance:
    exchange: str
    asset: str
    total: float
    available: float
    margin_used: float = 0.0
    equity: float = 0.0
    timestamp: float = field(default_factory=time.time)


@dataclass
class Portfolio:
    positions: Dict[str, Position] = field(default_factory=dict)
    balances: Dict[str, Balance] = field(default_factory=dict)

    @property
    def total_equity(self) -> float:
        """
        Returns the total equity across all exchanges.
        Note: This assumes all balance equities are in the same currency (e.g. USD/USDT)
        or that non-USD equities are negligible for this calculation.
        """
        return sum(b.equity for b in self.balances.values())

    def update_balance(self, balance: Balance):
        # Key by exchange (assuming one main account/asset per exchange for now)
        # or key by (exchange, asset).
        # Given StrategyManager uses total_equity, we probably want to key by exchange+asset if possible
        # but to keep it simple and consistent with how we might use it:
        # If we just overwrite by exchange, we lose multi-asset tracking.
        # But if the bot is mono-asset (USDT), it's fine.
        # Let's use a composite key to be safe.
        key = f"{balance.exchange}:{balance.asset}"
        self.balances[key] = balance

    def get_position(self, symbol: str, exchange: str = "") -> Position:
        key = f"{exchange}:{symbol}" if exchange else symbol
        return self.positions.get(key, Position(symbol, 0.0, 0.0, exchange=exchange))

    def set_position(self, position: Position):
        key = (
            f"{position.exchange}:{position.symbol}"
            if position.exchange
            else position.symbol
        )
        self.positions[key] = position

    def get_aggregate_exposure(self, symbol: str) -> float:
        total = 0.0
        for key, pos in self.positions.items():
            if pos.symbol == symbol:
                total += pos.size
        return total

    def get_exposure_by_exchange(self, exchange: str) -> float:
        total = 0.0
        for key, pos in self.positions.items():
            if pos.exchange == exchange:
                total += abs(pos.size)
        return total

    def all_positions(self) -> List[Position]:
        return list(self.positions.values())
