import asyncio
import logging
import sys
import os
import time
from dataclasses import replace

# Project root
sys.path.append(os.getcwd())

from src.infrastructure.config_factory import ConfigFactory
from src.use_cases.strategy_manager import MultiExchangeStrategyManager, ExchangeConfig
from src.domain.services.fair_price_service import FairPriceService
from src.domain.entities import Ticker
from src.adapters.exchanges.mock_adapter import MockExchangeGateway

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Verification")


# MOCK STRATEGY to avoid NoneType error
class MockStrategy:
    def setup(self, cfg):
        pass

    def calculate_quotes(self, *args, **kwargs):
        return []


async def test_lead_lag_logic():
    logger.info("Starting Lead-Lag Logic Verification...")

    # 1. Setup Config
    bot_config = ConfigFactory.load_config("config.json")

    fp_service = FairPriceService(
        oracle_symbol="HYPEUSDT",
        target_symbol="HYPEUSDT",
        oracle_exchange="binance",
        window_duration=60,
        min_samples=3,
    )

    # Mock Gateways
    class MockGateway:
        def __init__(self, name):
            self.name = name
            self.is_ready = True

        async def connect(self):
            pass

        async def disconnect(self):
            pass

        async def subscribe_ticker(self, s):
            pass

        def set_ticker_callback(self, cb):
            self.ticker_cb = cb

        def set_trade_callback(self, cb):
            pass

        def set_order_callback(self, cb):
            pass

        def set_position_callback(self, cb):
            pass

        def set_balance_callback(self, cb):
            pass

        def set_execution_callback(self, cb):
            pass

        async def get_position(self, s):
            class Pos:
                size = 0.0
                entry_price = 0.0

            return Pos()

        async def get_open_orders(self, s):
            return []

        async def start(self):
            pass

    binance_gw = MockGateway("binance")
    bybit_gw = MockGateway("bybit")

    configs = [
        ExchangeConfig(gateway=binance_gw, symbol="HYPEUSDT", enabled=True),
        ExchangeConfig(gateway=bybit_gw, symbol="HYPEUSDT", enabled=True),
    ]

    # 2. Initialize Manager
    from src.domain.risk.basic_manager import BasicRiskManager
    from src.domain.tracking.sync_engine import SyncEngine

    manager = MultiExchangeStrategyManager(
        exchanges=configs,
        strategy=MockStrategy(),  # inject mock strategy
        risk_manager=BasicRiskManager(),
        sync_engine=SyncEngine(),
        fair_price_service=fp_service,
        dry_run=True,
    )

    await manager.start()

    # 3. Simulate Data Flow
    # Binance (Oracle) moves UP, Bybit (Target) stays lagging

    # T0: Both at 100
    ts = time.time()
    logger.info("Injecting T0: Price 100")

    # Note: We must ensure mid_price is calculated correctly.
    # Ticker(bid=99.5, ask=100.5) -> mid=100.0

    await binance_gw.ticker_cb(
        Ticker(
            symbol="HYPEUSDT",
            bid=99.5,
            ask=100.5,
            last=100,
            timestamp=ts,
            exchange="binance",
            bid_size=1.0,
            ask_size=1.0,
            volume=1000.0,
        )
    )
    await bybit_gw.ticker_cb(
        Ticker(
            symbol="HYPEUSDT",
            bid=99.5,
            ask=100.5,
            last=100,
            timestamp=ts,
            exchange="bybit",
            bid_size=1.0,
            ask_size=1.0,
            volume=1000.0,
        )
    )

    logger.info(f"Fair Price: {fp_service.get_signal()}")
    logger.info(f"Window Size: {len(fp_service.offset_window)}")

    # Needs min samples (3)
    for i in range(10):  # Increased loop count
        ts += 1
        await binance_gw.ticker_cb(
            Ticker(
                symbol="HYPEUSDT",
                bid=99.5,
                ask=100.5,
                last=100,
                timestamp=ts,
                exchange="binance",
                bid_size=1.0,
                ask_size=1.0,
                volume=1000.0,
            )
        )
        await bybit_gw.ticker_cb(
            Ticker(
                symbol="HYPEUSDT",
                bid=99.5,
                ask=100.5,
                last=100,
                timestamp=ts,
                exchange="bybit",
                bid_size=1.0,
                ask_size=1.0,
                volume=1000.0,
            )
        )

    logger.info(f"Fair Price after warm-up: {fp_service.get_signal()}")
    logger.info(f"Window Size: {len(fp_service.offset_window)}")

    if fp_service.fair_price is None:
        logger.error("Fair Price is STILL None! Check logic.")
        # Debug internal state
        logger.info(f"Oracle Price: {fp_service.oracle_price}")
        logger.info(f"Target Price: {fp_service.target_price}")

    # T1: Binance Moves to 110 (Leader Pumping)
    logger.info("Injecting T1: Binance Pumps to 110")
    ts += 1

    # Bid 109.5, Ask 110.5 -> Mid 110.0
    await binance_gw.ticker_cb(
        Ticker(
            symbol="HYPEUSDT",
            bid=109.5,
            ask=110.5,
            last=110,
            timestamp=ts,
            exchange="binance",
            bid_size=1.0,
            ask_size=1.0,
            volume=1000.0,
        )
    )

    # Target still at 100
    # Expected: Offset was ~0. Now Target-Oracle = 100 - 110 = -10.
    # But Median Offset should still be ~0 because we have history of 0s.
    # Fair Price = Oracle (110) + MedianOffset (0) = 110.
    # So Fair Price should jump to 110 IMMEDIATELY, effectively leading the target.

    signal = fp_service.get_signal()
    logger.info(f"Fair Price Signal: {signal}")

    if signal["fair_price"] and abs(signal["fair_price"] - 110.0) < 1.0:
        logger.info("SUCCESS: Fair Price tracked Oracle move!")
    else:
        logger.error(
            f"FAILURE: Fair Price {signal.get('fair_price')} did not match expected ~110"
        )

    await manager.stop()


if __name__ == "__main__":
    asyncio.run(test_lead_lag_logic())
