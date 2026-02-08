"""
Benchmark tests for critical components of the Thalex Modular Trading System
"""
import asyncio
import time
import statistics
from typing import List, Dict
import numpy as np

from src.use_cases.quoting_service import QuotingService
from src.use_cases.strategy_manager import MultiExchangeStrategyManager
from src.domain.tracking.state_tracker import StateTracker
from src.domain.entities import Order, OrderSide, OrderType, Ticker


def benchmark_quoting_service():
    """Benchmark the QuotingService performance"""
    print("Benchmarking QuotingService...")
    
    # Mock components for testing
    class MockGateway:
        def __init__(self):
            self.name = "mock"
            
        async def connect(self):
            pass
            
        async def disconnect(self):
            pass
            
        async def subscribe_ticker(self, symbol):
            pass
            
        async def cancel_all_orders(self, symbol):
            return True
            
        async def get_position(self, symbol):
            from src.domain.entities import Position
            return Position(symbol, 0.0, 0.0)
            
        def set_ticker_callback(self, callback):
            pass
            
        def set_trade_callback(self, callback):
            pass
            
        def set_order_callback(self, callback):
            pass
            
        def set_position_callback(self, callback):
            pass
            
        def set_balance_callback(self, callback):
            pass
    
    class MockStrategy:
        def calculate_quotes(self, market_state, position, regime=None, tick_size=0.5):
            from src.domain.entities import Order
            return [
                Order(
                    id="test_order_1",
                    symbol="BTC-PERPETUAL",
                    side=OrderSide.BUY,
                    price=market_state.ticker.bid - 10,
                    size=0.001,
                    type=OrderType.LIMIT
                ),
                Order(
                    id="test_order_2",
                    symbol="BTC-PERPETUAL",
                    side=OrderSide.SELL,
                    price=market_state.ticker.ask + 10,
                    size=0.001,
                    type=OrderType.LIMIT
                )
            ]
    
    class MockSignalEngine:
        def get_signals(self):
            return {}
        
        def update(self, ticker):
            pass
    
    class MockRiskManager:
        def validate_order(self, order, position, active_orders=None):
            return True
        
        def can_trade(self):
            return True
    
    # Initialize components
    gateway = MockGateway()
    strategy = MockStrategy()
    signal_engine = MockSignalEngine()
    risk_manager = MockRiskManager()
    
    qs = QuotingService(
        gateway=gateway,
        strategy=strategy,
        signal_engine=signal_engine,
        risk_manager=risk_manager
    )
    
    # Create mock ticker
    ticker = Ticker(
        symbol="BTC-PERPETUAL",
        bid=40000.0,
        ask=40010.0,
        bid_size=1.0,
        ask_size=1.0,
        last=40005.0,
        volume=1000.0,
        timestamp=time.time()
    )
    
    # Benchmark _run_strategy method
    iterations = 1000
    times = []
    
    # Prepare market state
    qs.market_state.ticker = ticker
    
    for i in range(iterations):
        start_time = time.perf_counter()
        asyncio.run(qs._run_strategy())
        end_time = time.perf_counter()
        times.append((end_time - start_time) * 1000)  # Convert to milliseconds
    
    print(f"QuotingService._run_strategy(): {statistics.mean(times):.3f}ms avg, "
          f"{min(times):.3f}ms min, {max(times):.3f}ms max over {iterations} iterations")
    
    # Show performance metrics if available
    if hasattr(qs, 'get_performance_metrics'):
        metrics = qs.get_performance_metrics()
        print(f"Performance metrics: {metrics}")
    
    print()


def benchmark_state_tracker():
    """Benchmark the StateTracker performance"""
    print("Benchmarking StateTracker...")
    
    tracker = StateTracker()
    
    # Create mock orders for testing
    orders = []
    for i in range(100):
        order = Order(
            id=f"test_order_{i}",
            symbol="BTC-PERPETUAL",
            side=OrderSide.BUY if i % 2 == 0 else OrderSide.SELL,
            price=40000.0 + (i * 10),
            size=0.001,
            type=OrderType.LIMIT
        )
        orders.append(order)
    
    # Benchmark submit_order
    iterations = 1000
    times_submit = []
    
    for i in range(iterations):
        order = Order(
            id=f"benchmark_order_{i}",
            symbol="BTC-PERPETUAL",
            side=OrderSide.BUY,
            price=40000.0 + (i * 0.1),
            size=0.001,
            type=OrderType.LIMIT
        )
        start_time = time.perf_counter()
        asyncio.run(tracker.submit_order(order))
        end_time = time.perf_counter()
        times_submit.append((end_time - start_time) * 1000)  # Convert to milliseconds
    
    print(f"StateTracker.submit_order(): {statistics.mean(times_submit):.3f}ms avg, "
          f"{min(times_submit):.3f}ms min, {max(times_submit):.3f}ms max over {iterations} iterations")
    
    # Benchmark on_order_ack
    times_ack = []
    for i in range(min(500, iterations)):  # Fewer iterations to avoid index out of range
        start_time = time.perf_counter()
        asyncio.run(tracker.on_order_ack(f"benchmark_order_{i}", f"exch_{i}"))
        end_time = time.perf_counter()
        times_ack.append((end_time - start_time) * 1000)  # Convert to milliseconds
    
    print(f"StateTracker.on_order_ack(): {statistics.mean(times_ack):.3f}ms avg, "
          f"{min(times_ack):.3f}ms min, {max(times_ack):.3f}ms max over {len(times_ack)} iterations")
    
    # Show performance metrics if available
    if hasattr(tracker, 'get_performance_metrics'):
        metrics = tracker.get_performance_metrics()
        print(f"Performance metrics: {metrics}")
    
    print()


def benchmark_order_reconciliation():
    """Benchmark order reconciliation performance"""
    print("Benchmarking Order Reconciliation...")
    
    # Mock components for testing
    class MockGateway:
        def __init__(self):
            self.name = "mock"
            
        async def place_orders_batch(self, orders):
            return [order for order in orders]
            
        async def cancel_orders_batch(self, order_ids):
            return [True] * len(order_ids)
    
    class MockStateTracker:
        def __init__(self):
            self.orders = []
            
        def get_open_orders(self, side=None):
            from src.domain.tracking.state_tracker import TrackedOrder
            from src.domain.tracking.state_tracker import OrderState
            return [TrackedOrder(order=o, state=OrderState.CONFIRMED, submit_time=time.time()) 
                   for o in self.orders]
        
        async def submit_order(self, order):
            self.orders.append(order)
            
        async def on_order_ack(self, local_id, exchange_id):
            pass
        
        async def on_order_cancel(self, exchange_id):
            self.orders = [o for o in self.orders if o.id != exchange_id.replace("exch_", "")]
    
    # Initialize components
    gateway = MockGateway()
    state_tracker = MockStateTracker()
    
    # Create the optimized quoting service
    class OptimizedQuotingService:
        def __init__(self):
            self.gateway = gateway
            self.state_tracker = state_tracker
            self.tick_size = 0.5
        
        async def _diff_side(self, side, desired_list):
            """
            Efficiently compute the difference for a single side (BUY or SELL).
            Returns (to_cancel_ids, to_place_orders) tuple.
            """
            active_list = [t.order for t in self.state_tracker.get_open_orders(side=side)]

            # Use a more efficient matching algorithm
            to_cancel_ids = []
            to_place_orders = []

            # Create lookup dictionaries for O(1) access
            active_by_price_size = {}
            for act in active_list:
                key = (round(act.price / self.tick_size), act.size)  # Normalize price to tick boundaries
                active_by_price_size[key] = act

            desired_by_price_size = {}
            for des in desired_list:
                key = (round(des.price / self.tick_size), des.size)  # Normalize price to tick boundaries
                desired_by_price_size[key] = des

            # Find orders to cancel (in active but not in desired)
            for key, act in active_by_price_size.items():
                if key not in desired_by_price_size and act.exchange_id:
                    to_cancel_ids.append(act.exchange_id)

            # Find orders to place (in desired but not in active)
            for key, des in desired_by_price_size.items():
                if key not in active_by_price_size:
                    to_place_orders.append(des)

            return to_cancel_ids, to_place_orders
    
    qs = OptimizedQuotingService()
    
    # Create test orders
    test_orders = []
    for i in range(50):
        order = Order(
            id=f"test_order_{i}",
            symbol="BTC-PERPETUAL",
            side=OrderSide.BUY if i % 2 == 0 else OrderSide.SELL,
            price=40000.0 + (i * 10),
            size=0.001,
            type=OrderType.LIMIT
        )
        test_orders.append(order)
    
    # Benchmark the diff algorithm
    iterations = 1000
    times = []
    
    for i in range(iterations):
        start_time = time.perf_counter()
        buy_result = await qs._diff_side(OrderSide.BUY, [o for o in test_orders if o.side == OrderSide.BUY])
        sell_result = await qs._diff_side(OrderSide.SELL, [o for o in test_orders if o.side == OrderSide.SELL])
        end_time = time.perf_counter()
        times.append((end_time - start_time) * 1000)  # Convert to milliseconds
    
    print(f"Order reconciliation (_diff_side): {statistics.mean(times):.3f}ms avg, "
          f"{min(times):.3f}ms min, {max(times):.3f}ms max over {iterations} iterations")
    
    print()


async def run_all_benchmarks():
    """Run all benchmarks"""
    print("Running Performance Benchmarks for Thalex Modular Trading System\n")
    
    benchmark_quoting_service()
    benchmark_state_tracker()
    benchmark_order_reconciliation()
    
    print("Benchmarking complete!")


if __name__ == "__main__":
    asyncio.run(run_all_benchmarks())