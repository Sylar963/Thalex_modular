import asyncio
from unittest.mock import MagicMock
from src.use_cases.strategy_manager import MultiExchangeStrategyManager, ExchangeConfig
from src.domain.interfaces import ExchangeGateway, Strategy, RiskManager
from src.domain.tracking.sync_engine import SyncEngine


async def test_signal_engine_assignment():
    # Mock dependencies
    gateway = MagicMock(spec=ExchangeGateway)
    gateway.name = "test_exchange"
    config = ExchangeConfig(gateway=gateway, symbol="BTCUSDT")

    strategy = MagicMock(spec=Strategy)
    risk_manager = MagicMock(spec=RiskManager)
    sync_engine = MagicMock(spec=SyncEngine)
    signal_engine = MagicMock()  # Should not be None

    # Initialize manager
    manager = MultiExchangeStrategyManager(
        exchanges=[config],
        strategy=strategy,
        risk_manager=risk_manager,
        sync_engine=sync_engine,
        signal_engine=signal_engine,
    )

    # Check if signal_engine is assigned
    if hasattr(manager, "signal_engine") and manager.signal_engine is not None:
        print("SUCCESS: signal_engine is correctly assigned.")
    else:
        print("FAILURE: signal_engine is NOT assigned.")
        exit(1)


if __name__ == "__main__":
    asyncio.run(test_signal_engine_assignment())
