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
from src.domain.interfaces import SafetyComponent
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
        bot_config: dict, force_monitor_mode: bool = False
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
                        gw = ThalexAdapter(key, secret, testnet=venue_testnet)

                elif venue_name == "bybit":
                    key = os.getenv("BYBIT_API_KEY")
                    secret = os.getenv("BYBIT_API_SECRET")
                    if key and secret:
                        gw = BybitAdapter(key, secret, testnet=venue_testnet)

                elif venue_name == "binance":
                    key = os.getenv("BINANCE_API_KEY")
                    secret = os.getenv("BINANCE_API_SECRET")
                    if key and secret:
                        gw = BinanceAdapter(key, secret, testnet=venue_testnet)

                elif venue_name == "hyperliquid":
                    key = os.getenv("HYPERLIQUID_PRIVATE_KEY")
                    if key:
                        gw = HyperliquidAdapter(key, testnet=venue_testnet)

                else:
                    logger.warning(f"Unknown venue type: {venue_name}")
                    continue

                if not gw:
                    logger.warning(f"Skipping {venue_name}: Missing credentials.")
                    continue

                for sym in venue_symbols:
                    # Strategy Setup
                    venue_strategy = None
                    if (
                        not force_monitor_mode
                    ):  # Only setup strategy if NOT in monitor mode
                        v_strat_params = venue_cfg.get("strategy_params")
                        if v_strat_params:
                            venue_strategy = AvellanedaStoikovStrategy()
                            # Merge global and local params (simplified version of main.py logic)
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
