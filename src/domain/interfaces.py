from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from .entities import Order, Ticker, Position, Trade, Instrument, MarketState, Balance


class ExchangeGateway(ABC):
    """Abstract interface for exchange connectivity."""

    @abstractmethod
    async def get_balances(self) -> List[Balance]:
        """Fetch current account balances."""
        pass

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
    async def get_open_orders(self, symbol: str) -> List[Order]:
        """Fetch all open orders for a specific symbol."""
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

    @abstractmethod
    def set_balance_callback(self, callback):
        """Set callback for balance updates: async def cb(balance: Balance)."""
        pass

    @abstractmethod
    async def get_server_time(self) -> int:
        """Fetch current exchange server time in milliseconds."""
        pass


class TimeSyncManager(ABC):
    """Interface for managing time synchronization across multiple venues."""

    @abstractmethod
    def get_timestamp(self, exchange: str) -> int:
        """Get exchange-aligned timestamp in milliseconds."""
        pass

    @abstractmethod
    def get_offset(self, exchange: str) -> int:
        """Get the current time offset for the venue in milliseconds."""
        pass

    @abstractmethod
    async def sync_all(self):
        """Perform synchronization with all managed exchanges."""
        pass

    @abstractmethod
    async def sync_venue(self, exchange: str):
        """Sync a specific venue immediately."""
        pass

    @abstractmethod
    async def start_sync_loop(self):
        """Start periodic background synchronization."""
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


class RegimeAnalyzer(ABC):
    @abstractmethod
    def update(self, ticker: Ticker) -> None:
        pass

    @abstractmethod
    def set_option_data(self, em_pct: float, atm_iv: float) -> None:
        pass

    @abstractmethod
    def get_regime(self) -> Dict[str, Any]:
        pass


class RiskManager(ABC):
    @abstractmethod
    def validate_order(
        self,
        order: Order,
        position,
        active_orders: Optional[List[Order]] = None,
    ) -> bool:
        pass

    @abstractmethod
    def update_position(self, position: Position) -> None:
        """Update internal position state for risk tracking."""
        pass

    @abstractmethod
    def check_position_limits(self, position) -> bool:
        pass

    @abstractmethod
    def can_trade(self) -> bool:
        pass

    @abstractmethod
    def has_breached(self) -> bool:
        """Check if a risk limit has been breached."""
        pass

    @abstractmethod
    def get_risk_state(self) -> Dict[str, Any]:
        """Return the current state of risk management."""
        pass

    @abstractmethod
    def reset_breach(self) -> None:
        """Manually reset the risk breach status."""
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

    @abstractmethod
    async def save_regime(self, symbol: str, regime: Dict[str, Any]):
        """Save computed market regime metrics."""
        pass

    @abstractmethod
    async def get_regime_history(
        self, symbol: str, start: float, end: float
    ) -> List[Dict]:
        """Retrieve historical regime metrics."""
        pass

    @abstractmethod
    async def save_balance(self, balance: Any):
        """Save account balance snapshot."""
        pass

    @abstractmethod
    async def get_latest_balances(self) -> List[Any]:
        """Retrieve latest balances for all exchanges."""
        pass


class SafetyComponent(ABC):
    """Abstract interface for safety plugins."""

    @abstractmethod
    def check_health(self, context: Dict[str, Any]) -> bool:
        """
        Check if the component allows trading to proceed.
        Returns True if healthy, False if trading should stop.
        Context dictionary provides necessary data (e.g., timestamps, prices).
        """
        pass

    @abstractmethod
    def record_failure(self) -> None:
        """Record a failure event (e.g., API error, latency breach)."""
        pass

    @abstractmethod
    def record_success(self) -> None:
        """Record a success event (e.g., successful cycle)."""
        pass
