import logging

logger = logging.getLogger(__name__)

try:
    from .thalex_adapter import ThalexAdapter
except ImportError:
    logger.debug("ThalexAdapter dependencies missing.")

try:
    from .binance_adapter import BinanceAdapter
except ImportError:
    logger.debug("BinanceAdapter dependencies missing.")

try:
    from .bybit_adapter import BybitAdapter
except ImportError:
    logger.debug("BybitAdapter dependencies missing.")

try:
    from .hyperliquid_adapter import HyperliquidAdapter
except ImportError:
    logger.debug("HyperliquidAdapter dependencies missing.")

from .base_adapter import BaseExchangeAdapter
