from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from .entities import Order, Ticker, Position, Trade, Instrument, MarketState


class ExchangeGateway(ABC):
    """Abstract interface for exchange connectivity."""

    @abstractmethod
    async def connect(self):
        """Establish connection to the exchange."""
        pass

    @abstractmethod
    async def disconnect(self):
        """Close connection to the exchange."""
        pass

    @abstractmethod
    async def place_order(self, order: Order) -> Order:
        """Place an order and return the updated order with exchange ID."""
        pass

    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order by ID."""
        pass

    @abstractmethod
    async def place_orders_batch(self, orders: List[Order]) -> List[Order]:
        """Place multiple orders in a single batch request."""
        pass

    @abstractmethod
    async def cancel_orders_batch(self, order_ids: List[str]) -> List[bool]:
        """Cancel multiple orders in a single batch."""
        pass

    @abstractmethod
    async def subscribe_ticker(self, symbol: str):
        """Subscribe to ticker updates for a symbol."""
        pass

    @abstractmethod
    async def get_position(self, symbol: str) -> Position:
        """Get current position for a symbol."""
        pass

    @abstractmethod
    def set_ticker_callback(self, callback):
        """Set callback for ticker updates: async def cb(ticker: Ticker)."""
        pass

    @abstractmethod
    def set_trade_callback(self, callback):
        """Set callback for trade updates: async def cb(trade: Trade)."""
        pass

    @abstractmethod
    def set_order_callback(self, callback):
        pass

    @abstractmethod
    def set_position_callback(self, callback):
        pass


class Strategy(ABC):
    """Abstract interface for trading strategies."""

    @abstractmethod
    def calculate_quotes(
        self, market_state: MarketState, position: Position, tick_size: float = 0.5
    ) -> List[Order]:
        """Calculates the list of orders to place based on market state."""
        pass

    @abstractmethod
    def setup(self, config: Dict[str, Any]):
        """Initialize strategy with configuration."""
        pass


class SignalEngine(ABC):
    """Abstract interface for signal generation."""

    @abstractmethod
    def update(self, ticker: Ticker):
        """Update internal state with new market data."""
        pass

    @abstractmethod
    def update_trade(self, trade: Trade):
        """Update internal state with new trade execution."""
        pass

    @abstractmethod
    def get_signals(self) -> Dict[str, float]:
        """Return current signal values."""
        pass


class RiskManager(ABC):
    """Abstract interface for risk management."""

    @abstractmethod
    def validate_order(
        self,
        order: Order,
        position: Position,
        active_orders: Optional[List[Order]] = None,
    ) -> bool:
        """Check if an order is safe to place."""
        pass

    @abstractmethod
    def check_position_limits(self, position: Position) -> bool:
        """Check if position is within limits."""
        pass

    @abstractmethod
    def can_trade(self) -> bool:
        """Global switch to check if trading is allowed."""
        pass


class StorageGateway(ABC):
    """Abstract interface for data persistence."""

    @abstractmethod
    async def save_ticker(self, ticker: Ticker):
        """Save ticker data to storage."""
        pass

    @abstractmethod
    async def save_trade(self, trade: Trade):
        """Save trade data to storage."""
        pass

    @abstractmethod
    async def save_position(self, position: Position):
        """Save current position snapshot."""
        pass

    @abstractmethod
    async def get_recent_tickers(self, symbol: str, limit: int = 100) -> List[Ticker]:
        """Retrieve recent ticker history."""
        pass

    @abstractmethod
    async def get_history(
        self, symbol: str, start: float, end: float, resolution: str
    ) -> List[Dict]:
        """Retrieve OHLCV history."""
        pass
