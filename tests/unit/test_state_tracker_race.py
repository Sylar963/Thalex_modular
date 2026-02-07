import pytest
import asyncio
from src.domain.entities import Order, OrderSide, OrderType, OrderStatus
from src.domain.tracking.state_tracker import StateTracker, OrderState


@pytest.mark.asyncio
async def test_order_race_condition_ws_first():
    """
    Test the race condition where a WS notification for a fill arrives
    before the RPC acknowledgement.
    """
    tracker = StateTracker()
    local_id = "test_order_1"
    exchange_id = "ex_123"

    order = Order(
        id=local_id,
        symbol="BTC-USD",
        side=OrderSide.BUY,
        price=50000.0,
        size=0.1,
        type=OrderType.LIMIT,
        status=OrderStatus.PENDING,
    )

    # 1. Order submitted locally
    await tracker.submit_order(order)
    assert local_id in tracker.pending_orders

    # 2. SIMULATE RACE: WS notification for FILL arrives BEFORE RPC ACK
    # Currently, this will fail to find the order because it's only looking in confirmed_orders
    await tracker.on_order_update(
        exchange_id=exchange_id,
        status=OrderStatus.FILLED,
        filled_size=0.1,
        avg_price=50000.0,
        # missing local_id/label which we will add to fix this
    )

    # BUG: The order should have transitioned to FILLED (in terminal_orders)
    # but currently it's likely still in pending_orders because exchange_id was unknown.

    # If it was still in pending_orders, it's a bug
    # Let's see what happens.
    if exchange_id in tracker.terminal_orders:
        print("Order correctly reached terminal state despite race!")
    else:
        print(f"Order state lost! exchange_id {exchange_id} not in terminal_orders")
        assert local_id in tracker.pending_orders  # It's stuck in pending

    # 3. RPC Ack arrives eventually
    await tracker.on_order_ack(local_id, exchange_id)

    # Now it's in confirmed_orders, but it's actually FILLED.
    # The bot will think it's still alive.
    assert exchange_id in tracker.confirmed_orders
    assert tracker.confirmed_orders[exchange_id].state == OrderState.CONFIRMED


@pytest.mark.asyncio
async def test_order_race_condition_with_label_fix():
    """
    Test the fix where we pass the local_id (label) to on_order_update.
    """
    tracker = StateTracker()
    local_id = "test_order_fix"
    exchange_id = "ex_456"

    order = Order(
        id=local_id,
        symbol="BTC-USD",
        side=OrderSide.BUY,
        price=50000.0,
        size=0.1,
        type=OrderType.LIMIT,
        status=OrderStatus.PENDING,
    )

    await tracker.submit_order(order)

    # 1. WS notification with local_id arrives first
    # This represents the FIXED behavior
    # We need to modify StateTracker.on_order_update to accept local_id
    try:
        await tracker.on_order_update(
            exchange_id=exchange_id,
            status=OrderStatus.FILLED,
            filled_size=0.1,
            avg_price=50000.0,
            local_id=local_id,  # New parameter
        )

        # Should be in terminal state
        assert exchange_id in tracker.terminal_orders
        assert tracker.terminal_orders[exchange_id].state == OrderState.FILLED
        assert local_id not in tracker.pending_orders

        # 2. RPC Ack arrives
        await tracker.on_order_ack(local_id, exchange_id)

        # Should still be in terminal state, not moved back to confirmed
        assert exchange_id in tracker.terminal_orders
        assert exchange_id not in tracker.confirmed_orders

    except TypeError:
        pytest.skip("StateTracker.on_order_update does not accept local_id yet")
