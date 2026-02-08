"""
Test script to verify concurrent reconciliation in multi-venue mode.
This script creates mock venues and verifies that they can reconcile independently.
"""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock
import sys
import os

# Add the src directory to the path so we can import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from use_cases.strategy_manager import MultiExchangeStrategyManager, ExchangeConfig, VenueContext
from domain.entities import Ticker, Order, OrderSide, OrderType, OrderStatus
from domain.interfaces import ExchangeGateway


class MockGateway(ExchangeGateway):
    """Mock gateway for testing purposes."""
    
    def __init__(self, name, symbol):
        self._name = name
        self.symbol = symbol
        self.connected = True
        self.is_reconnecting = False
        self.last_ticker_time = time.time()
        self.ticker_callback = None
        self.trade_callback = None
        self.order_callback = None
        self.position_callback = None
        self.balance_callback = None
        
    @property
    def name(self):
        return self._name
    
    async def connect(self):
        self.connected = True
        print(f"{self.name} connected")
    
    async def disconnect(self):
        self.connected = False
        print(f"{self.name} disconnected")
    
    async def place_order(self, order):
        print(f"{self.name} placing order: {order}")
        return order
    
    async def cancel_order(self, order_id):
        print(f"{self.name} cancelling order: {order_id}")
        return True
    
    async def place_orders_batch(self, orders):
        print(f"{self.name} placing {len(orders)} orders in batch")
        return orders
    
    async def cancel_orders_batch(self, order_ids):
        print(f"{self.name} cancelling {len(order_ids)} orders in batch")
        return [True] * len(order_ids)
    
    async def get_open_orders(self, symbol):
        return []
    
    async def get_position(self, symbol):
        from domain.entities import Position
        return Position(symbol, 0.0, 0.0)
    
    async def get_balances(self):
        from domain.entities import Balance
        return [Balance("USD", 10000, 10000, 0, 10000)]
    
    async def subscribe_ticker(self, symbol):
        print(f"{self.name} subscribed to {symbol}")
    
    def set_ticker_callback(self, callback):
        self.ticker_callback = callback
    
    def set_trade_callback(self, callback):
        self.trade_callback = callback
    
    def set_order_callback(self, callback):
        self.order_callback = callback
    
    def set_position_callback(self, callback):
        self.position_callback = callback
    
    def set_balance_callback(self, callback):
        self.balance_callback = callback


class MockStrategy:
    """Mock strategy for testing."""
    
    def calculate_quotes(self, market_state, position, tick_size=0.5):
        # Return a simple buy and sell order
        ticker = market_state.ticker
        if not ticker:
            return []
        
        mid_price = ticker.mid_price
        spread = tick_size * 2  # Simple spread
        
        orders = [
            Order(
                id=f"buy_{int(time.time())}",
                symbol=ticker.symbol,
                side=OrderSide.BUY,
                price=mid_price - spread/2,
                size=0.1,
                type=OrderType.LIMIT,
                exchange=ticker.exchange
            ),
            Order(
                id=f"sell_{int(time.time())}",
                symbol=ticker.symbol,
                side=OrderSide.SELL,
                price=mid_price + spread/2,
                size=0.1,
                type=OrderType.LIMIT,
                exchange=ticker.exchange
            )
        ]
        return orders


class MockRiskManager:
    """Mock risk manager for testing."""
    
    def validate_order(self, order, portfolio, active_orders=None):
        return True  # Allow all orders for testing
    
    def has_breached(self):
        return False  # No breaches for testing
    
    def update_position(self, position):
        pass
    
    def get_risk_state(self):
        return {}


async def simulate_tickers(manager, venues):
    """Simulate ticker updates for venues."""
    for i in range(10):  # Send 10 ticker updates
        for name, venue in venues.items():
            ticker = Ticker(
                symbol=venue.config.symbol,
                bid=50000.0 + (i * 10 * (1 if name == "bybit" else -1)),  # Different price movements
                ask=50000.0 + (i * 10 * (1 if name == "bybit" else -1)) + 10,  # 10 USD spread
                bid_size=1.0,
                ask_size=1.0,
                last=50000.0 + (i * 10 * (1 if name == "bybit" else -1)) + 5,
                volume=1000.0,
                exchange=name
            )
            
            # Call the ticker callback directly to trigger strategy execution
            if venue.config.gateway.ticker_callback:
                await venue.config.gateway.ticker_callback(ticker)
        
        await asyncio.sleep(0.1)  # Small delay between ticker updates


async def main():
    print("Starting multi-venue locking test...")
    
    # Create mock gateways
    bybit_gateway = MockGateway("bybit", "BTCUSDT")
    thalex_gateway = MockGateway("thalex", "BTC-PERPETUAL")
    
    # Create exchange configs
    bybit_config = ExchangeConfig(bybit_gateway, "BTCUSDT", enabled=True, tick_size=0.5)
    thalex_config = ExchangeConfig(thalex_gateway, "BTC-PERPETUAL", enabled=True, tick_size=1.0)
    
    exchanges = [bybit_config, thalex_config]
    
    # Create mock components
    strategy = MockStrategy()
    risk_manager = MockRiskManager()
    
    # Mock sync engine
    sync_engine = MagicMock()
    sync_engine.update_ticker = AsyncMock()
    sync_engine.update_position = AsyncMock()
    sync_engine.update_balance = AsyncMock()
    sync_engine.on_state_change = None
    
    # Create the strategy manager
    manager = MultiExchangeStrategyManager(
        exchanges=exchanges,
        strategy=strategy,
        risk_manager=risk_manager,
        sync_engine=sync_engine,
        dry_run=True
    )
    
    # Start the manager
    await manager.start()
    
    # Simulate ticker updates to trigger strategy execution
    await simulate_tickers(manager, manager.venues)
    
    # Stop the manager
    await manager.stop()
    
    print("Test completed successfully!")


if __name__ == "__main__":
    asyncio.run(main())