import asyncio
import logging
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field
import time

from ..entities import Ticker, Position, OrderSide

logger = logging.getLogger(__name__)


@dataclass
class GlobalState:
    positions: Dict[str, Position] = field(default_factory=dict)
    tickers: Dict[str, Ticker] = field(default_factory=dict)
    last_update: float = field(default_factory=time.time)

    @property
    def net_position(self) -> float:
        return sum(p.size for p in self.positions.values())

    @property
    def global_best_bid(self) -> float:
        if not self.tickers:
            return 0.0
        return max(t.bid for t in self.tickers.values())

    @property
    def global_best_ask(self) -> float:
        if not self.tickers:
            return float("inf")
        return min(t.ask for t in self.tickers.values())


class SyncEngine:
    def __init__(self):
        self.state = GlobalState()
        self._lock = asyncio.Lock()
        self.on_state_change: Optional[Callable[[GlobalState], None]] = None

    async def update_position(self, exchange: str, symbol: str, position: Position):
        async with self._lock:
            key = f"{exchange}:{symbol}"
            self.state.positions[key] = position
            self.state.last_update = time.time()

            logger.debug(f"SyncEngine: Position updated for {key}: {position.size}")

            if self.on_state_change:
                self.on_state_change(self.state)

    async def update_ticker(self, exchange: str, symbol: str, ticker: Ticker):
        async with self._lock:
            key = f"{exchange}:{symbol}"
            self.state.tickers[key] = ticker
            self.state.last_update = time.time()

    def get_net_position(self, symbol: str) -> float:
        total = 0.0
        for key, pos in self.state.positions.items():
            if key.endswith(f":{symbol}"):
                total += pos.size
        return total

    def get_positions_by_exchange(self, exchange: str) -> List[Position]:
        return [
            p for k, p in self.state.positions.items() if k.startswith(f"{exchange}:")
        ]

    def get_global_mid_price(self, symbol: str) -> float:
        prices = []
        for key, ticker in self.state.tickers.items():
            if key.endswith(f":{symbol}"):
                prices.append(ticker.mid_price)
        return sum(prices) / len(prices) if prices else 0.0

    def get_arb_opportunity(self, symbol: str) -> Optional[Dict]:
        best_bid_exchange = None
        best_ask_exchange = None
        best_bid = 0.0
        best_ask = float("inf")

        for key, ticker in self.state.tickers.items():
            if key.endswith(f":{symbol}"):
                exchange = key.split(":")[0]
                if ticker.bid > best_bid:
                    best_bid = ticker.bid
                    best_bid_exchange = exchange
                if ticker.ask < best_ask:
                    best_ask = ticker.ask
                    best_ask_exchange = exchange

        if best_bid > best_ask and best_bid_exchange and best_ask_exchange:
            return {
                "buy_exchange": best_ask_exchange,
                "sell_exchange": best_bid_exchange,
                "spread": best_bid - best_ask,
            }
        return None
