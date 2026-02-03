from dataclasses import dataclass, field
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


@dataclass(slots=True)
class MarketState:
    ticker: Optional[Ticker] = None
    orderbook: Optional[Dict] = None  # Placeholder for full orderbook
    signals: Dict[str, float] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
