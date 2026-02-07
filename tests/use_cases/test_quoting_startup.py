import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock
from src.use_cases.quoting_service import QuotingService
from src.domain.entities import (
    Order,
    OrderSide,
    OrderType,
    OrderStatus,
    Position,
    Ticker,
)
from src.domain.tracking.state_tracker import StateTracker


@pytest.mark.asyncio
async def test_start_cancel_all_success():
    """Test standard startup where cancel_all_orders succeeds."""
    gateway = AsyncMock()
    # Fix RuntimeWarning by making setters synchronous
    gateway.set_ticker_callback = MagicMock()
    gateway.set_trade_callback = MagicMock()
    gateway.set_order_callback = MagicMock()
    gateway.set_position_callback = MagicMock()

    # Mock cancel_all_orders output
    gateway.cancel_all_orders.return_value = True

    # Mock position fetch
    gateway.get_position.return_value = Position("BTC-PERP", 0, 0, 0)

    service = QuotingService(
        gateway=gateway,
        strategy=MagicMock(),
        signal_engine=MagicMock(),
        risk_manager=MagicMock(),
        state_tracker=StateTracker(),
    )

    await service.start("BTC-PERP")

    gateway.cancel_all_orders.assert_called_once()
    gateway.get_open_orders.assert_not_called()
    assert service.running is True
    await service.stop()


@pytest.mark.asyncio
async def test_detect_and_destroy_success():
    """Test cleanup when cancel_all_orders fails but individual cleanup succeeds."""
    gateway = AsyncMock()
    gateway.set_ticker_callback = MagicMock()
    gateway.set_trade_callback = MagicMock()
    gateway.set_order_callback = MagicMock()
    gateway.set_position_callback = MagicMock()

    gateway.cancel_all_orders.return_value = False

    # Mock get_open_orders returning 2 stale orders
    stale_order_1 = Order(
        id="1",
        symbol="BTC-PERP",
        side=OrderSide.BUY,
        price=100.0,
        size=1.0,
        type=OrderType.LIMIT,
        status=OrderStatus.OPEN,
        exchange="thalex",
        exchange_id="ex1",
    )
    stale_order_2 = Order(
        id="2",
        symbol="BTC-PERP",
        side=OrderSide.SELL,
        price=100.0,
        size=1.0,
        type=OrderType.LIMIT,
        status=OrderStatus.OPEN,
        exchange="thalex",
        exchange_id="ex2",
    )

    # First call returns orders to clean, second call returns empty (check)
    gateway.get_open_orders.side_effect = [[stale_order_1, stale_order_2], []]

    # Mock individual cancel
    gateway.cancel_order.return_value = True
    gateway.get_position.return_value = Position("BTC-PERP", 0, 0, 0)

    service = QuotingService(
        gateway=gateway,
        strategy=MagicMock(),
        signal_engine=MagicMock(),
        risk_manager=MagicMock(),
        state_tracker=StateTracker(),
    )

    await service.start("BTC-PERP")

    gateway.cancel_all_orders.assert_called_once()
    assert gateway.get_open_orders.call_count >= 1
    assert gateway.cancel_order.call_count == 2
    assert service.running is True
    await service.stop()


@pytest.mark.asyncio
async def test_startup_fails_if_cleanup_fails():
    """Test that startup aborts if both bulk and individual cleanup fail."""
    gateway = AsyncMock()
    gateway.set_ticker_callback = MagicMock()
    gateway.set_trade_callback = MagicMock()
    gateway.set_order_callback = MagicMock()
    gateway.set_position_callback = MagicMock()

    gateway.cancel_all_orders.return_value = False

    stale_order = Order(
        id="1",
        symbol="BTC-PERP",
        side=OrderSide.BUY,
        price=100.0,
        size=1.0,
        type=OrderType.LIMIT,
        status=OrderStatus.OPEN,
        exchange="thalex",
        exchange_id="ex1",
    )

    # First call returns order, second call (verification) still returns it
    gateway.get_open_orders.side_effect = [[stale_order], [stale_order]]

    gateway.cancel_order.return_value = False  # Cancel fails
    gateway.get_position.return_value = Position("BTC-PERP", 0, 0, 0)

    service = QuotingService(
        gateway=gateway,
        strategy=MagicMock(),
        signal_engine=MagicMock(),
        risk_manager=MagicMock(),
        state_tracker=StateTracker(),
    )

    with pytest.raises(RuntimeError, match="Startup aborted"):
        await service.start("BTC-PERP")

    assert service.running is False
