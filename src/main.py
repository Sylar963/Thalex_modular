import asyncio
import logging
import signal
import argparse
import sys
import os
import json
from dotenv import load_dotenv

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "thalex_py"))

from src.adapters.exchanges.thalex_adapter import ThalexAdapter
from src.domain.strategies.avellaneda import AvellanedaStoikovStrategy
from src.domain.signals.volume_candle import VolumeCandleSignalEngine
from src.domain.risk.basic_manager import BasicRiskManager
from src.domain.market.regime_analyzer import MultiWindowRegimeAnalyzer
from src.services.options_volatility_service import OptionsVolatilityService
from src.domain.tracking.state_tracker import StateTracker
from src.use_cases.quoting_service import QuotingService

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("Main")


def parse_args():
    parser = argparse.ArgumentParser(description="Thalex Trading Bot")
    parser.add_argument(
        "--mode",
        choices=["live", "shadow", "backtest"],
        default="live",
        help="Execution mode: live (real orders), shadow (simulated fills), backtest (historical)",
    )
    parser.add_argument(
        "--initial-balance",
        type=float,
        default=1000.0,
        help="Initial balance for shadow/backtest mode",
    )
    parser.add_argument(
        "--latency-ms",
        type=float,
        default=50.0,
        help="Simulated latency in milliseconds for shadow mode",
    )
    parser.add_argument(
        "--slippage-ticks",
        type=float,
        default=0.0,
        help="Simulated slippage in ticks for shadow mode",
    )
    return parser.parse_args()


async def main():
    args = parse_args()
    load_dotenv()

    mode = args.mode
    logger.info(f"Starting in {mode.upper()} mode")

    testnet = os.getenv("TRADING_MODE", "testnet").lower() == "testnet"
    if testnet:
        api_key = os.getenv("THALEX_TEST_API_KEY_ID")
        api_secret = os.getenv("THALEX_TEST_PRIVATE_KEY")
    else:
        api_key = os.getenv("THALEX_PROD_API_KEY_ID") or os.getenv("THALEX_KEY_ID")
        api_secret = os.getenv("THALEX_PROD_PRIVATE_KEY") or os.getenv(
            "THALEX_PRIVATE_KEY"
        )

    # Load Config from config.json
    config_path = os.path.join(project_root, "config.json")
    try:
        with open(config_path, "r") as f:
            bot_config = json.load(f)
        logger.info(f"Loaded configuration from {config_path}")
    except FileNotFoundError:
        logger.warning(f"config.json not found at {config_path}, using empty config")
        bot_config = {}

    symbol = os.getenv("PRIMARY_INSTRUMENT") or bot_config.get(
        "primary_instrument", "BTC-PERPETUAL"
    )

    if not api_key or not api_secret:
        logger.warning("API credentials not found in environment.")

    exchange_config = bot_config.get("exchange", {})
    gateway = ThalexAdapter(
        api_key,
        api_secret,
        testnet=testnet,
        me_rate_limit=exchange_config.get("me_rate_limit", 45.0),
        cancel_rate_limit=exchange_config.get("cancel_rate_limit", 900.0),
    )

    db_user = os.getenv("DATABASE_USER", "postgres")
    db_pass = os.getenv("DATABASE_PASSWORD", "password")
    db_host = os.getenv("DATABASE_HOST", "localhost")
    db_port = os.getenv("DATABASE_PORT", "5432")
    db_name = os.getenv("DATABASE_NAME", "thalex_trading")
    db_dsn = f"postgresql://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}"

    storage = None
    try:
        from src.adapters.storage.timescale_adapter import TimescaleDBAdapter

        storage = TimescaleDBAdapter(db_dsn)
        logger.info(
            f"Initialized TimescaleDB Adapter (DSN: ...@{db_host}:{db_port}/...)"
        )
    except ImportError:
        logger.warning("asyncpg not installed, skipping DB storage.")

    strategy = AvellanedaStoikovStrategy()
    strategy_params = bot_config.get("strategy", {}).get("params", {})
    strategy.setup(strategy_params)

    signal_params = bot_config.get("signals", {}).get("volume_candle", {})
    signal_engine = VolumeCandleSignalEngine(
        volume_threshold=signal_params.get("volume_threshold", 10.0),
        max_candles=signal_params.get("max_candles", 50),
    )

    or_params = bot_config.get("signals", {}).get("open_range", {})
    or_engine = None
    if or_params.get("enabled", False):
        from src.domain.signals.open_range import OpenRangeSignalEngine

        or_engine = OpenRangeSignalEngine(
            session_start_utc=or_params.get("session_start_utc", "20:00"),
            session_end_utc=or_params.get("session_end_utc", "20:15"),
            target_pct_from_mid=or_params.get("target_pct_from_mid", 1.49),
            subsequent_target_pct_of_range=or_params.get(
                "subsequent_target_pct_of_range", 220
            ),
        )
        logger.info("OpenRangeSignalEngine initialized")

    risk_params = bot_config.get("risk", {})
    risk_manager = BasicRiskManager()
    risk_manager.setup(risk_params)

    sim_engine = None
    sim_state = None
    dry_run = mode == "shadow"

    if dry_run:
        from src.domain.sim_match_engine import SimMatchEngine
        from src.use_cases.sim_state_manager import sim_state_manager

        sim_engine = SimMatchEngine(
            latency_ms=args.latency_ms,
            slippage_ticks=args.slippage_ticks,
            tick_size=getattr(gateway, "tick_size", 0.5),
        )
        sim_engine.set_initial_state(args.initial_balance)
        sim_state = sim_state_manager
        await sim_state.start(symbol, args.initial_balance, mode="shadow")
        logger.info(f"Shadow mode initialized with balance: {args.initial_balance}")

    regime_analyzer = MultiWindowRegimeAnalyzer()

    # Initialize Options Volatility Service (if not backtesting)
    if mode != "backtest":
        # Pass the underlying Thalex client to the service
        vol_service = OptionsVolatilityService(gateway.client, symbol.split("-")[0])

        async def _poll_vol_data():
            while True:
                try:
                    em_pct, atm_iv = await vol_service.get_expected_move()
                    if em_pct > 0:
                        regime_analyzer.set_option_data(em_pct, atm_iv)
                        logger.debug(
                            f"Updated Regime with Options Data: EM={em_pct:.2%}, IV={atm_iv:.2%}"
                        )
                except Exception as e:
                    logger.error(f"Failed to poll options data: {e}")
                await asyncio.sleep(15)  # Poll every 15s

        # Create background task
        asyncio.create_task(_poll_vol_data())

    state_tracker = StateTracker()

    service = QuotingService(
        gateway,
        strategy,
        signal_engine,
        risk_manager,
        storage_gateway=storage,
        dry_run=dry_run,
        sim_engine=sim_engine,
        sim_state=sim_state,
        regime_analyzer=regime_analyzer,
        state_tracker=state_tracker,
        or_engine=or_engine,
    )

    # Apply throttling config if present
    throttling_params = bot_config.get("throttling", {})
    if "min_edge_threshold" in throttling_params:
        service.min_edge_threshold = throttling_params["min_edge_threshold"]

    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()

    def signal_handler():
        logger.info("Signal received, stopping...")
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, signal_handler)

    try:
        if storage:
            await storage.connect()

        await service.start(symbol)
        await stop_event.wait()

    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
    finally:
        await service.stop()
        if sim_state:
            await sim_state.stop()
        if storage:
            await storage.disconnect()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
