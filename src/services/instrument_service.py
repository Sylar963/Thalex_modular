from typing import Dict, Optional


class InstrumentService:
    """
    Centralized service for mapping internal symbols (Thalex-style)
    to exchange-specific symbols.

    Internal: BTC-PERPETUAL, ETH-PERPETUAL
    Bybit: BTCUSDT, ETHUSDT
    Binance: BTCUSDT, ETHUSDT
    Hyperliquid: BTC, ETH
    Thalex: BTC-PERPETUAL, ETH-PERPETUAL (no change)
    """

    _MAPPINGS = {
        "bybit": {"BTC-PERPETUAL": "BTCUSDT", "ETH-PERPETUAL": "ETHUSDT"},
        "binance": {"BTC-PERPETUAL": "BTCUSDT", "ETH-PERPETUAL": "ETHUSDT"},
        "hyperliquid": {"BTC-PERPETUAL": "BTC", "ETH-PERPETUAL": "ETH"},
        "thalex": {
            # Identity mapping
            "BTC-PERPETUAL": "BTC-PERPETUAL",
            "ETH-PERPETUAL": "ETH-PERPETUAL",
        },
    }

    @classmethod
    def get_exchange_symbol(cls, internal_symbol: str, exchange: str) -> str:
        exchange = exchange.lower()
        internal_symbol = internal_symbol.upper()

        # Default to internal symbol if no mapping found (pass-through)
        if exchange not in cls._MAPPINGS:
            return internal_symbol

        return cls._MAPPINGS[exchange].get(internal_symbol, internal_symbol)

    @classmethod
    def get_internal_symbol(cls, exchange_symbol: str, exchange: str) -> str:
        # Reverse lookup (O(N) but map is small)
        exchange = exchange.lower()
        if exchange not in cls._MAPPINGS:
            return exchange_symbol

        for internal, external in cls._MAPPINGS[exchange].items():
            if external == exchange_symbol:
                return internal

        return exchange_symbol
