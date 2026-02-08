import logging
from typing import Optional
from ..entities import Ticker, Trade

logger = logging.getLogger(__name__)

class MarketDataValidator:
    """
    Centralized validation system for market data.
    Ensures data integrity across all venues before ingestion.
    """

    @staticmethod
    def validate_ticker(ticker: Ticker) -> Optional[Ticker]:
        """
        Validates and sanitizes a Ticker object.
        Returns None if the ticker is fundamentally invalid.
        """
        # 1. basic sanity check
        if not ticker.symbol:
            return None

        # 2. Fix Zero Last Price
        # Some feeds (like Bybit orderbook) might send 0 for last price.
        # We try to recover it from Bid/Ask mid-point.
        if ticker.last <= 0:
            if ticker.bid > 0 and ticker.ask > 0:
                # Use Mid-Price
                object.__setattr__(ticker, 'last', (ticker.bid + ticker.ask) / 2.0)
            elif ticker.bid > 0:
                object.__setattr__(ticker, 'last', ticker.bid)
            elif ticker.ask > 0:
                object.__setattr__(ticker, 'last', ticker.ask)
            else:
                # Completely dead ticker (0/0/0)
                # logger.debug(f"Dropping invalid ticker for {ticker.symbol}: {ticker}")
                return None

        # 3. Spread Sanity (Optional, prevent crossed markets if needed)
        # if ticker.bid > ticker.ask: ...

        return ticker

    @staticmethod
    def validate_trade(trade: Trade) -> Optional[Trade]:
        """
        Validates a Trade object.
        """
        if trade.price <= 0:
            logger.warning(f"Dropping zero-price trade for {trade.symbol}: {trade.price}")
            return None
        
        if trade.size <= 0:
            return None

        return trade
