import pytest
from src.domain.strategies.orb_breakout import TradeDirection, TradeSignal
from src.domain.strategies.orb_trade_manager import (
    ORBTradeManager,
    TradeState,
    ActionType,
)


def _make_signal(
    direction=TradeDirection.LONG,
    price=100.0,
    unit_size=1.0,
    orh=100.0,
    orl=99.0,
    orm=99.5,
    orw=1.0,
):
    return TradeSignal(
        symbol="BTCUSDT",
        direction=direction,
        entry_price=price,
        orh=orh,
        orl=orl,
        orm=orm,
        orw=orw,
        base_unit_size=unit_size,
        timestamp=1000.0,
        first_up_target=orm + orm * 0.0189,
        first_down_target=orm - orm * 0.0189,
    )


@pytest.fixture
def manager():
    return ORBTradeManager(
        symbol="BTCUSDT",
        momentum_check_seconds=30.0,
        shave_pct=0.90,
        max_retries=2,
        subsequent_target_pct=0.50,
    )


def test_initial_state(manager):
    assert manager.state == TradeState.IDLE
    assert manager.current_size == 0.0
    assert manager.retry_count == 0


def test_entry_signal(manager):
    signal = _make_signal()
    manager.on_signal(signal)
    assert manager.state == TradeState.ENTRY_PENDING

    actions = manager.get_pending_actions()
    assert len(actions) == 1
    assert actions[0].action == ActionType.ENTER
    assert actions[0].size == 1.0


def test_momentum_check_pass(manager):
    signal = _make_signal(price=100.0)
    manager.on_signal(signal)
    manager.on_entry_fill(100.0, 1.0, 1000.0)
    assert manager.state == TradeState.MOMENTUM_CHECK

    manager.on_tick(101.0, 1031.0)
    assert manager.state == TradeState.ACTIVE
    assert manager.break_even == 100.0


def test_momentum_check_fail(manager):
    signal = _make_signal(price=100.0)
    manager.on_signal(signal)
    manager.get_pending_actions()
    manager.on_entry_fill(100.0, 1.0, 1000.0)

    manager.on_tick(99.0, 1031.0)
    assert manager.state == TradeState.CLOSED

    actions = manager.get_pending_actions()
    assert len(actions) == 1
    assert actions[0].action == ActionType.EXIT


def test_momentum_check_not_yet(manager):
    signal = _make_signal(price=100.0)
    manager.on_signal(signal)
    manager.on_entry_fill(100.0, 1.0, 1000.0)

    manager.on_tick(101.0, 1015.0)
    assert manager.state == TradeState.MOMENTUM_CHECK


def test_pyramid_at_target(manager):
    signal = _make_signal(price=100.0, unit_size=1.0, orm=99.5, orw=1.0)
    manager.on_signal(signal)
    manager.get_pending_actions()
    manager.on_entry_fill(100.0, 1.0, 1000.0)

    manager.on_tick(101.0, 1031.0)
    assert manager.state == TradeState.ACTIVE

    target = manager.pyramid_levels[0].target_price
    manager.on_tick(target + 0.1, 1100.0)

    actions = manager.get_pending_actions()
    shave_actions = [a for a in actions if a.action == ActionType.SHAVE]
    add_actions = [a for a in actions if a.action == ActionType.ADD]

    assert len(shave_actions) == 1
    assert len(add_actions) == 1
    assert shave_actions[0].size == pytest.approx(0.9, abs=0.01)
    assert add_actions[0].size == 1.0

    assert manager.current_size == pytest.approx(1.1, abs=0.01)
    assert manager.pyramid_count == 1
    assert manager.break_even > 100.0


def test_break_even_stop(manager):
    signal = _make_signal(price=100.0, unit_size=1.0, orm=99.5, orw=1.0)
    manager.on_signal(signal)
    manager.get_pending_actions()
    manager.on_entry_fill(100.0, 1.0, 1000.0)

    manager.on_tick(101.0, 1031.0)

    target = manager.pyramid_levels[0].target_price
    manager.on_tick(target + 0.1, 1100.0)
    manager.get_pending_actions()

    be = manager.break_even
    manager.on_tick(be - 0.01, 1200.0)

    assert manager.state == TradeState.CLOSED
    actions = manager.get_pending_actions()
    exit_actions = [a for a in actions if a.action == ActionType.EXIT]
    assert len(exit_actions) == 1
    assert manager.total_shaved_profit > 0


def test_multi_level_pyramid(manager):
    signal = _make_signal(price=100.0, unit_size=1.0, orm=99.5, orw=1.0)
    manager.on_signal(signal)
    manager.get_pending_actions()
    manager.on_entry_fill(100.0, 1.0, 1000.0)

    manager.on_tick(101.0, 1031.0)

    t1 = manager.pyramid_levels[0].target_price
    manager.on_tick(t1 + 0.1, 1100.0)
    manager.get_pending_actions()
    assert manager.pyramid_count == 1

    t2_price = manager._get_next_target_price()
    manager.on_tick(t2_price + 0.1, 1200.0)
    manager.get_pending_actions()
    assert manager.pyramid_count == 2


def test_reentry_after_be_stop(manager):
    signal = _make_signal(price=100.0, unit_size=1.0, orm=99.5, orw=1.0)
    manager.on_signal(signal)
    manager.get_pending_actions()
    manager.on_entry_fill(100.0, 1.0, 1000.0)

    manager.on_tick(101.0, 1031.0)

    target = manager.pyramid_levels[0].target_price
    manager.on_tick(target + 0.1, 1100.0)
    manager.get_pending_actions()

    be = manager.break_even
    manager.on_tick(be - 0.01, 1200.0)
    manager.get_pending_actions()
    assert manager.state == TradeState.CLOSED
    assert manager.retry_count == 0

    manager.on_tick(manager.orh + 0.5, 1300.0)
    assert manager.state == TradeState.ENTRY_PENDING
    assert manager.retry_count == 1

    actions = manager.get_pending_actions()
    assert len(actions) == 1
    assert actions[0].action == ActionType.ENTER


def test_max_retries_enforced(manager):
    signal = _make_signal(price=100.0, unit_size=1.0, orm=99.5, orw=1.0)
    manager.on_signal(signal)
    manager.get_pending_actions()
    manager.on_entry_fill(100.0, 1.0, 1000.0)
    manager.on_tick(99.0, 1031.0)
    manager.get_pending_actions()
    assert manager.state == TradeState.CLOSED

    manager.on_tick(101.0, 1100.0)
    assert manager.retry_count == 1
    manager.get_pending_actions()
    manager.on_entry_fill(101.0, 1.0, 1100.0)
    manager.on_tick(100.0, 1131.0)
    manager.get_pending_actions()
    assert manager.state == TradeState.CLOSED

    manager.on_tick(102.0, 1200.0)
    assert manager.retry_count == 2
    manager.get_pending_actions()
    manager.on_entry_fill(102.0, 1.0, 1200.0)
    manager.on_tick(101.0, 1231.0)
    manager.get_pending_actions()
    assert manager.state == TradeState.CLOSED

    manager.on_tick(103.0, 1300.0)
    assert manager.retry_count == 2
    assert manager.state == TradeState.CLOSED


def test_short_direction(manager):
    signal = _make_signal(direction=TradeDirection.SHORT, price=98.0)
    manager.on_signal(signal)
    manager.get_pending_actions()
    manager.on_entry_fill(98.0, 1.0, 1000.0)

    manager.on_tick(97.0, 1031.0)
    assert manager.state == TradeState.ACTIVE

    target = manager.pyramid_levels[0].target_price
    assert target < 98.0

    manager.on_tick(target - 0.1, 1100.0)
    actions = manager.get_pending_actions()
    shave_actions = [a for a in actions if a.action == ActionType.SHAVE]
    add_actions = [a for a in actions if a.action == ActionType.ADD]
    assert len(shave_actions) == 1
    assert len(add_actions) == 1
    assert shave_actions[0].direction == TradeDirection.LONG


def test_reset_session(manager):
    signal = _make_signal()
    manager.on_signal(signal)
    manager.on_entry_fill(100.0, 1.0, 1000.0)

    manager.reset_session()
    assert manager.state == TradeState.IDLE
    assert manager.current_size == 0.0
    assert manager.retry_count == 0
    assert manager.pyramid_count == 0


def test_state_snapshot(manager):
    signal = _make_signal()
    manager.on_signal(signal)
    manager.on_entry_fill(100.0, 1.0, 1000.0)

    snap = manager.get_state_snapshot()
    assert snap["symbol"] == "BTCUSDT"
    assert snap["state"] == "momentum_check"
    assert snap["current_size"] == 1.0
    assert snap["direction"] == "long"
