import os
import json
import logging
from typing import List, Optional
from dataclasses import dataclass

from src.adapters.exchanges.thalex_adapter import ThalexAdapter
from src.adapters.exchanges.bybit_adapter import BybitAdapter
from src.adapters.exchanges.binance_adapter import BinanceAdapter
from src.adapters.exchanges.hyperliquid_adapter import HyperliquidAdapter
from src.use_cases.strategy_manager import ExchangeConfig
from src.domain.strategies.avellaneda import AvellanedaStoikovStrategy
from src.domain.interfaces import SafetyComponent, TimeSyncManager
from src.domain.safety.latency_monitor import LatencyMonitor
from src.domain.safety.circuit_breaker import CircuitBreaker

logger = logging.getLogger(__name__)


class ConfigFactory:
    @staticmethod
    def load_config(path: str = "config.json") -> dict:
        try:
            with open(path, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"Config file not found at {path}, returning empty config.")
            return {}

    @staticmethod
    def create_exchange_configs(
        bot_config: dict,
        force_monitor_mode: bool = False,
        time_sync_manager: Optional[TimeSyncManager] = None,
        dry_run: bool = False,
    ) -> List[ExchangeConfig]:
        """
        Creates a list of ExchangeConfig objects based on the provided configuration.

        :param bot_config: Dictionary containing the bot configuration.
        :param force_monitor_mode: If True, enabled=True is forced for all valid venues,
                                   but strategy is set to None (Monitoring only).
        """
        venues_config = bot_config.get("venues", {})
        exchange_configs = []
        strategy_params = bot_config.get("strategy", {}).get("params", {})

        for venue_name, venue_cfg in venues_config.items():
            # If monitoring, we ignore the 'enabled' flag for connection purposes
            # (effectively treating it as enabled for data), OR we respect it?
            # User said: "All option when select exchange... those 2 will work only currently"
            # Implication: We should connect to anything properly configured.
            # But let's respect 'enabled' unless force_monitor_mode is True.

            is_enabled = venue_cfg.get("enabled", False)
            if not is_enabled and not force_monitor_mode:
                continue

            # If force_monitor_mode is True, we treat it as enabled for data streaming
            effective_enabled = True if force_monitor_mode else is_enabled

            venue_testnet = venue_cfg.get("testnet", True)
            venue_symbols = venue_cfg.get("symbols", [])
            venue_tick_size = venue_cfg.get("tick_size", 0.5)

            # Custom URLs for flexible environment setup
            api_url = venue_cfg.get("api_url")
            ws_url = venue_cfg.get("ws_url")
            ws_public_url = venue_cfg.get("ws_public_url")
            ws_private_url = venue_cfg.get("ws_private_url")

            gw = None

            try:
                if venue_name == "thalex":
                    key = (
                        os.getenv("THALEX_TEST_API_KEY_ID")
                        if venue_testnet
                        else (
                            os.getenv("THALEX_PROD_API_KEY_ID")
                            or os.getenv("THALEX_KEY_ID")
                        )
                    )
                    secret = (
                        os.getenv("THALEX_TEST_PRIVATE_KEY")
                        if venue_testnet
                        else (
                            os.getenv("THALEX_PROD_PRIVATE_KEY")
                            or os.getenv("THALEX_PRIVATE_KEY")
                        )
                    )
                    if key and secret:
                        gw = ThalexAdapter(
                            key,
                            secret,
                            testnet=venue_testnet,
                            time_sync_manager=time_sync_manager,
                            ws_url=ws_url,
                        )

                elif venue_name == "bybit":
                    key = os.getenv("BYBIT_API_KEY")
                    secret = os.getenv("BYBIT_API_SECRET")
                    if key and secret:
                        gw = BybitAdapter(
                            key,
                            secret,
                            testnet=venue_testnet,
                            time_sync_manager=time_sync_manager,
                            api_url=api_url,
                            ws_public_url=ws_public_url,
                            ws_private_url=ws_private_url,
                        )

                elif venue_name == "binance":
                    key = os.getenv("BINANCE_API_KEY")
                    secret = os.getenv("BINANCE_API_SECRET")
                    if key and secret:
                        gw = BinanceAdapter(
                            key,
                            secret,
                            testnet=venue_testnet,
                            time_sync_manager=time_sync_manager,
                        )

                elif venue_name == "hyperliquid":
                    key = os.getenv("HYPERLIQUID_PRIVATE_KEY")
                    if key:
                        gw = HyperliquidAdapter(
                            key,
                            testnet=venue_testnet,
                            time_sync_manager=time_sync_manager,
                        )

                else:
                    logger.warning(f"Unknown venue type: {venue_name}")
                    continue

                if not gw:
                    logger.warning(f"Skipping {venue_name}: Missing credentials.")
                    continue

                if dry_run:
                    from src.adapters.exchanges.mock_adapter import MockExchangeGateway

                    sim_config = bot_config.get("simulation", {})
                    gw = MockExchangeGateway(
                        real_adapter=gw,
                        initial_balance=sim_config.get("starting_balance", 10000.0),
                        latency_ms=sim_config.get("latency_ms", 50.0),
                        slippage_ticks=sim_config.get("slippage_ticks", 0.5),
                        maker_fee=sim_config.get("maker_fee", -0.0001),
                        taker_fee=sim_config.get("taker_fee", 0.0003),
                        tick_size=venue_tick_size,
                    )
                    logger.info(
                        f"Shadow Mode: Wrapped {venue_name} in MockExchangeGateway"
                    )

                for sym in venue_symbols:
                    venue_strategy = None
                    if not force_monitor_mode:
                        v_strat_params = venue_cfg.get("strategy_params")
                        if v_strat_params:
                            venue_strategy = AvellanedaStoikovStrategy()
                            merged = strategy_params.copy()
                            merged.update(v_strat_params)
                            venue_strategy.setup(merged)

                    exchange_configs.append(
                        ExchangeConfig(
                            gateway=gw,
                            symbol=sym,
                            enabled=effective_enabled,
                            tick_size=venue_tick_size,
                            strategy=venue_strategy,
                        )
                    )
            except Exception as e:
                logger.error(f"Failed to configure {venue_name}: {e}")

        return exchange_configs

    @staticmethod
    def create_safety_components(bot_config: dict) -> List[SafetyComponent]:
        """
        Instantiates protection plugins based on 'safety' config section.
        """
        safety_config = bot_config.get("safety", {})
        components = []

        # 1. Latency Monitor
        lat_conf = safety_config.get("latency_monitor", {})
        if lat_conf.get("enabled", True):  # Default ON is safer
            max_lat = lat_conf.get("max_latency", 1.0)
            components.append(LatencyMonitor(max_latency=max_lat))
            logger.info(f"Safety: LatencyMonitor enabled (max={max_lat}s).")

        # 2. Circuit Breaker
        cb_conf = safety_config.get("circuit_breaker", {})
        if cb_conf.get("enabled", True):  # Default ON
            threshold = cb_conf.get("failure_threshold", 5)
            timeout = cb_conf.get("recovery_timeout", 60)
            components.append(
                CircuitBreaker(
                    failure_threshold=threshold,
                    recovery_timeout=timeout,
                    name="GlobalCB",
                )
            )
            logger.info(
                f"Safety: CircuitBreaker enabled (thresh={threshold}, timeout={timeout}s)."
            )

        return components

    @staticmethod
    def create_history_prefetchers(
        bot_config: dict,
        db_adapter,
    ) -> dict:
        from src.adapters.storage.history_prefetcher import BybitHistoryPrefetcher

        prefetchers = {}
        venues_config = bot_config.get("venues", {})
        for venue_name, venue_cfg in venues_config.items():
            if not venue_cfg.get("enabled", False):
                continue
            if venue_name == "bybit":
                prefetchers["bybit"] = BybitHistoryPrefetcher(db_adapter)
                logger.info("History prefetcher created for Bybit.")
        return prefetchers

    @staticmethod
    def create_canary_sensor(bot_config: dict):
        from src.domain.sensors.canary_sensor import CanarySensor

        sensors_config = bot_config.get("sensors", {})
        canary_config = sensors_config.get("canary", {})

        if not canary_config.get("enabled", True):
            logger.info("Canary sensor disabled by config.")
            return None

        window_ms = canary_config.get("window_ms", 5000)
        sensor = CanarySensor(window_ms=window_ms)
        logger.info(f"Canary sensor enabled (window={window_ms}ms).")
        return sensor
