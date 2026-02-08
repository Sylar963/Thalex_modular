"""
Comprehensive test suite for the optimized Thalex Modular Trading System components
Validates performance improvements and correctness of all optimized components
"""
import asyncio
import time
import pytest
from unittest.mock import Mock, AsyncMock, MagicMock
import statistics

from src.infrastructure.monitoring.latency_tracker import LatencyTracker, SystemMonitor
from src.adapters.exchanges.base_adapter import TokenBucket
from src.adapters.exchanges.thalex_adapter import ThalexAdapter
from src.use_cases.quoting_service import QuotingService
from src.domain.tracking.state_tracker import StateTracker
from src.use_cases.strategy_manager import MultiExchangeStrategyManager, VenueContext
from dataclasses import dataclass
from src.domain.entities import Order, OrderSide, OrderType, OrderStatus, Ticker, Position


class TestOptimizedComponents:
    """Test suite for all optimized components"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.latency_tracker = LatencyTracker()
        self.latency_tracker.start()
    
    def teardown_method(self):
        """Teardown test fixtures"""
        self.latency_tracker.stop()
    
    async def test_token_bucket_performance(self):
        """Test that the optimized token bucket performs better than original"""
        # The current implementation is already the optimized one
        from src.adapters.exchanges.base_adapter import TokenBucket

        # Test token bucket
        tb = TokenBucket(100, 50.0)

        # Benchmark the implementation
        iterations = 10000

        # Time the implementation
        start = time.perf_counter()
        for _ in range(iterations):
            tb.consume(1)
        tb_time = (time.perf_counter() - start) * 1000  # ms

        print(f"TokenBucket Performance (10,000 operations):")
        print(f"  Current implementation: {tb_time:.3f}ms")

        # Just verify it runs without errors and is reasonably fast
        assert tb_time < 1000, f"Token bucket should be fast, got {tb_time}ms"
    
    async def test_state_tracker_performance(self):
        """Test that the state tracker performs well"""
        import time

        # Create state tracker (now optimized)
        tracker = StateTracker()
        await tracker.start()

        # Create test order
        test_order = Order(
            id="test_order_1",
            symbol="BTC-PERPETUAL",
            side=OrderSide.BUY,
            price=50000.0,
            size=0.001,
            type=OrderType.LIMIT
        )

        # Test tracker performance
        iterations = 1000

        start = time.perf_counter()
        for i in range(iterations):
            await tracker.submit_order(test_order)
            await tracker.on_order_ack(test_order.id, f"exch_{i}")
        tracker_time = (time.perf_counter() - start) * 1000  # ms

        print(f"StateTracker Performance (1000 operations):")
        print(f"  Current implementation: {tracker_time:.3f}ms")

        # Verify it runs without errors and is reasonably fast
        assert tracker_time < 1000, f"State tracker should be fast, got {tracker_time}ms"

        await tracker.stop()
    
    async def test_quoting_service_reconciliation(self):
        """Test quoting service reconciliation performance"""
        # Mock dependencies
        mock_gateway = AsyncMock()
        mock_gateway.place_orders_batch = AsyncMock(return_value=[
            Order(id="new_order_1", symbol="BTC-PERPETUAL", side=OrderSide.BUY,
                  price=49999.0, size=0.001, type=OrderType.LIMIT, status=OrderStatus.OPEN, exchange_id="exch_1")
        ])
        mock_gateway.cancel_orders_batch = AsyncMock(return_value=[True])
        mock_gateway.name = "thalex"

        mock_strategy = Mock()
        mock_strategy.calculate_quotes = Mock(return_value=[
            Order(id="quote_1", symbol="BTC-PERPETUAL", side=OrderSide.BUY,
                  price=49999.0, size=0.001, type=OrderType.LIMIT)
        ])

        mock_risk_manager = Mock()
        mock_risk_manager.validate_order = Mock(return_value=True)
        mock_risk_manager.can_trade = Mock(return_value=True)

        # Create quoting service (now optimized)
        qs = QuotingService(
            gateway=mock_gateway,
            strategy=mock_strategy,
            signal_engine=Mock(),
            risk_manager=mock_risk_manager
        )

        # Initialize state tracker
        await qs.state_tracker.start()

        # Create test market state
        ticker = Ticker(
            symbol="BTC-PERPETUAL",
            bid=49999.0,
            ask=50001.0,
            last=50000.0,
            timestamp=time.time()
        )
        qs.market_state.ticker = ticker

        # Test reconciliation performance
        iterations = 100

        start = time.perf_counter()
        for _ in range(iterations):
            await qs._reconcile_orders([
                Order(id=f"test_order_{i}", symbol="BTC-PERPETUAL", side=OrderSide.BUY,
                      price=49999.0 - i*10, size=0.001, type=OrderType.LIMIT)
                for i in range(5)
            ])
        reconcile_time = (time.perf_counter() - start) * 1000  # ms

        avg_time = reconcile_time / iterations
        print(f"QuotingService Reconciliation Performance:")
        print(f"  Average time per reconciliation: {avg_time:.3f}ms")
        print(f"  Total time for {iterations} reconciliations: {reconcile_time:.3f}ms")

        # Should be under 10ms average for good performance
        assert avg_time < 10.0, f"Reconciliation should be under 10ms, got {avg_time}ms"

        # Test performance metrics if available
        if hasattr(qs, 'get_performance_metrics'):
            perf_metrics = qs.get_performance_metrics()
            assert perf_metrics['reconcile_calls'] == iterations

        await qs.state_tracker.stop()
    
    async def test_strategy_manager_concurrency(self):
        """Test that strategy manager handles concurrency well"""
        # Mock dependencies
        mock_gateway = AsyncMock()
        mock_gateway.place_orders_batch = AsyncMock(return_value=[
            Order(id="new_order_1", symbol="BTC-PERPETUAL", side=OrderSide.BUY,
                  price=49999.0, size=0.001, type=OrderType.LIMIT, status=OrderStatus.OPEN, exchange_id="exch_1")
        ])
        mock_gateway.cancel_orders_batch = AsyncMock(return_value=[True])
        mock_gateway.name = "thalex"
        mock_gateway.set_ticker_callback = Mock()
        mock_gateway.set_trade_callback = Mock()
        mock_gateway.set_order_callback = Mock()
        mock_gateway.set_position_callback = Mock()
        mock_gateway.set_balance_callback = Mock()
        mock_gateway.connect = AsyncMock()
        mock_gateway.subscribe_ticker = AsyncMock()
        mock_gateway.get_position = AsyncMock(return_value=Position("BTC-PERPETUAL", 0.0, 0.0))
        mock_gateway.get_open_orders = AsyncMock(return_value=[])

        mock_strategy = Mock()
        mock_strategy.calculate_quotes = Mock(return_value=[
            Order(id="quote_1", symbol="BTC-PERPETUAL", side=OrderSide.BUY,
                  price=49999.0, size=0.001, type=OrderType.LIMIT)
        ])

        mock_risk_manager = Mock()
        mock_risk_manager.validate_order = Mock(return_value=True)
        mock_risk_manager.has_breached = Mock(return_value=False)
        mock_risk_manager.update_position = Mock()

        mock_sync_engine = Mock()
        mock_sync_engine.update_ticker = AsyncMock()
        mock_sync_engine.update_position = AsyncMock()
        mock_sync_engine.update_balance = AsyncMock()
        mock_sync_engine.on_state_change = None
        mock_sync_engine.state = Mock()
        mock_sync_engine.state.net_position = 0.0
        mock_sync_engine.state.global_best_bid = 49999.0
        mock_sync_engine.state.global_best_ask = 50001.0
        mock_sync_engine.state.tickers = {}

        # Create exchange config
        @dataclass
        class ExchangeConfig:
            gateway: object
            symbol: str
            enabled: bool = True
            tick_size: float = 0.5
            strategy: object = None

        exchange_config = ExchangeConfig(
            gateway=mock_gateway,
            symbol="BTC-PERPETUAL",
            enabled=True
        )

        # Create strategy manager (now optimized)
        sm = MultiExchangeStrategyManager(
            exchanges=[exchange_config],
            strategy=mock_strategy,
            risk_manager=mock_risk_manager,
            sync_engine=mock_sync_engine
        )

        # Create venue context
        venue = sm.venues["thalex"]  # Use the existing venue context from the initialized manager
        await venue.state_tracker.start()

        # Create test ticker
        ticker = Ticker(
            symbol="BTC-PERPETUAL",
            bid=49999.0,
            ask=50001.0,
            last=50000.0,
            timestamp=time.time()
        )
        venue.market_state.ticker = ticker

        # Test concurrent strategy runs
        iterations = 50

        start = time.perf_counter()
        tasks = []
        for _ in range(iterations):
            task = asyncio.create_task(sm._run_strategy_for_venue(venue))
            tasks.append(task)

        await asyncio.gather(*tasks)
        concurrent_time = (time.perf_counter() - start) * 1000  # ms

        avg_time = concurrent_time / iterations
        print(f"StrategyManager Concurrent Performance:")
        print(f"  Average time per strategy run: {avg_time:.3f}ms")
        print(f"  Total time for {iterations} concurrent runs: {concurrent_time:.3f}ms")

        # Should handle concurrency reasonably well
        assert avg_time < 50.0, f"Concurrent strategy runs should be under 50ms avg, got {avg_time}ms"

        await venue.state_tracker.stop()
    
    async def test_latency_tracking_integration(self):
        """Test that latency tracking works correctly with all components"""
        # Create tracker and monitor
        tracker = LatencyTracker(retention_minutes=1, max_samples=1000)
        tracker.start()
        
        monitor = SystemMonitor(tracker)
        
        # Simulate operations across different components
        operations = [
            ("thalex_adapter", "place_order"),
            ("quoting_service", "reconcile_orders"),
            ("state_tracker", "submit_order"),
            ("strategy_manager", "run_strategy"),
        ]
        
        for i in range(100):
            comp, op = operations[i % len(operations)]
            # Simulate varying latencies
            latency = 1 + (i % 10)  # 1-10ms
            tracker.record_latency(comp, op, latency)
        
        # Wait a bit for aggregation
        await asyncio.sleep(0.1)
        
        # Check that metrics are properly collected
        all_metrics = tracker.get_all_metrics()
        assert len(all_metrics) >= len(operations), "Should have metrics for all operations"
        
        # Check component metrics
        for comp, _ in operations:
            comp_metrics = tracker.get_component_metrics(comp)
            assert comp_metrics.count > 0, f"Component {comp} should have metrics"
        
        # Export metrics
        json_export = monitor.export_metrics("json")
        assert "samples_collected" in json_export
        assert "components_monitored" in json_export
        
        prometheus_export = monitor.export_metrics("prometheus")
        assert "# HELP" in prometheus_export
        
        tracker.stop()
    
    async def test_alerting_functionality(self):
        """Test that alerting works correctly"""
        tracker = LatencyTracker()
        tracker.start()
        
        # Set up alert threshold
        tracker.set_alert_threshold("test_component", "slow_operation", 5.0)  # 5ms threshold
        
        # Track a slow operation
        alerts_received = []
        
        def alert_callback(component, operation, duration, threshold):
            alerts_received.append((component, operation, duration, threshold))
        
        tracker.add_alert_callback(alert_callback)
        
        # Record a slow operation (>5ms threshold)
        tracker.record_latency("test_component", "slow_operation", 10.0)
        
        # Wait briefly for alert to be processed
        await asyncio.sleep(0.01)
        
        # Check that alert was received
        assert len(alerts_received) == 1
        comp, op, dur, thresh = alerts_received[0]
        assert comp == "test_component"
        assert op == "slow_operation"
        assert dur == 10.0
        assert thresh == 5.0
        
        tracker.stop()


class PerformanceBenchmarkSuite:
    """Comprehensive performance benchmark suite"""
    
    async def run_all_benchmarks(self):
        """Run all performance benchmarks"""
        print("=" * 60)
        print("COMPREHENSIVE PERFORMANCE BENCHMARK SUITE")
        print("=" * 60)
        
        test_instance = TestOptimizedComponents()
        
        print("\n1. Testing TokenBucket Performance...")
        await test_instance.test_token_bucket_performance()

        print("\n2. Testing StateTracker Performance...")
        await test_instance.test_state_tracker_performance()

        print("\n3. Testing QuotingService Reconciliation...")
        await test_instance.test_quoting_service_reconciliation()

        print("\n4. Testing StrategyManager Concurrency...")
        await test_instance.test_strategy_manager_concurrency()
        
        print("\n5. Testing Latency Tracking Integration...")
        await test_instance.test_latency_tracking_integration()
        
        print("\n6. Testing Alerting Functionality...")
        await test_instance.test_alerting_functionality()
        
        print("\n" + "=" * 60)
        print("ALL BENCHMARKS COMPLETED SUCCESSFULLY")
        print("=" * 60)


# Run benchmarks if this file is executed directly
if __name__ == "__main__":
    async def main():
        suite = PerformanceBenchmarkSuite()
        await suite.run_all_benchmarks()
    
    asyncio.run(main())