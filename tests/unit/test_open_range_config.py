import pytest
from src.domain.signals.open_range import OpenRangeSignalEngine


def test_per_symbol_config():
    engine = OpenRangeSignalEngine(
        target_pct_from_mid=1.5,
        subsequent_target_pct_of_range=200,
        symbol_configs={
            "BTCUSDT": {"target_pct_from_mid": 2.0},
            "ETHUSDT": {"subsequent_target_pct_of_range": 300},
        },
    )

    # Check default (SOLUSDT)
    state_sol = engine._get_state("SOLUSDT")
    assert state_sol.target_pct_from_mid == pytest.approx(0.015)
    assert state_sol.subsequent_target_pct_of_range == pytest.approx(2.0)

    # Check BTC override (target_pct_from_mid changed)
    state_btc = engine._get_state("BTCUSDT")
    assert state_btc.target_pct_from_mid == pytest.approx(0.02)
    assert state_btc.subsequent_target_pct_of_range == pytest.approx(2.0)

    # Check ETH override (subsequent_target_pct_of_range changed)
    state_eth = engine._get_state("ETHUSDT")
    assert state_eth.target_pct_from_mid == pytest.approx(0.015)
    assert state_eth.subsequent_target_pct_of_range == pytest.approx(3.0)
