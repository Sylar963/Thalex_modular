import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock
from src.use_cases.strategy_manager import (
    MultiExchangeStrategyManager,
    VenueContext,
    ExchangeConfig,
)
from src.domain.interfaces import SafetyComponent, RiskManager, Strategy
from src.domain.entities import Ticker


class MockSafetyComponent(SafetyComponent):
    def __init__(self, should_pass=True):
        self.should_pass = should_pass
        self.fail_count = 0
        self.success_count = 0

    def check_health(self, context) -> bool:
        return self.should_pass

    def record_failure(self):
        self.fail_count += 1

    def record_success(self):
        self.success_count += 1


@pytest.mark.asyncio
async def test_strategy_manager_respects_safety():
    # Setup dependencies
    mock_strategy = MagicMock(spec=Strategy)
    mock_risk = MagicMock(spec=RiskManager)
    mock_risk.has_breached.return_value = False
    mock_sync = MagicMock()
    mock_gateway = MagicMock()
    mock_gateway.name = "test_exchange"

    # Create venue config
    config = ExchangeConfig(gateway=mock_gateway, symbol="BTC-USD", enabled=True)

    # 1. Healthy Component
    healthy_safety = MockSafetyComponent(should_pass=True)

    manager = MultiExchangeStrategyManager(
        exchanges=[config],
        strategy=mock_strategy,
        risk_manager=mock_risk,
        sync_engine=mock_sync,
        safety_components=[healthy_safety],
        dry_run=True,
    )

    # Create a wrapper to access the private callback
    callback = manager._make_ticker_callback("test_exchange")
    ticker = Ticker(
        symbol="BTC-USD",
        bid=100,
        ask=101,
        mid_price=100.5,
        timestamp=1000,
        exchange="test_exchange",
    )

    # Initialize venue state
    manager.venues["test_exchange"].market_state.ticker = ticker
    manager._running = True

    # Run callback
    await callback(ticker)

    # Assertions for Healthy Case
    assert healthy_safety.success_count == 1
    # Strategy should be called
    mock_strategy.calculate_quotes.assert_called()


@pytest.mark.asyncio
async def test_strategy_manager_halts_on_safety_fail():
    # Setup dependencies
    mock_strategy = MagicMock(spec=Strategy)
    mock_risk = MagicMock(spec=RiskManager)
    mock_sync = MagicMock()
    mock_gateway = MagicMock()
    mock_gateway.name = "test_exchange"
    config = ExchangeConfig(gateway=mock_gateway, symbol="BTC-USD", enabled=True)

    # 2. Unhealthy Component
    failing_safety = MockSafetyComponent(should_pass=False)

    manager = MultiExchangeStrategyManager(
        exchanges=[config],
        strategy=mock_strategy,
        risk_manager=mock_risk,
        sync_engine=mock_sync,
        safety_components=[failing_safety],
        dry_run=True,
    )

    callback = manager._make_ticker_callback("test_exchange")
    ticker = Ticker(
        symbol="BTC-USD",
        bid=100,
        ask=101,
        mid_price=100.5,
        timestamp=1000,
        exchange="test_exchange",
    )
    manager._running = True

    # Reset mock strategy calls
    mock_strategy.calculate_quotes.reset_mock()

    # Run callback
    await callback(ticker)

    # Assertions for Unhealthy Case
    assert failing_safety.fail_count == 1
    # Strategy should NOT be called
    mock_strategy.calculate_quotes.assert_not_called()
