from typing import List, Dict, Optional
from .base_repository import BaseRepository
from ...domain.entities import Ticker


class MarketRepository(BaseRepository):
    async def get_recent_tickers(self, symbol: str, limit: int = 100) -> List[Ticker]:
        if not self.storage:
            return []
        return await self.storage.get_recent_tickers(symbol, limit)

    async def get_history(
        self, symbol: str, start: float, end: float, resolution: str
    ) -> List[Dict]:
        if not self.storage:
            return []
        # Assuming the storage adapter has this extended method (we added it)
        if hasattr(self.storage, "get_history"):
            return await self.storage.get_history(symbol, start, end, resolution)
        return []

    async def get_regime_history(
        self, symbol: str, start: float, end: float
    ) -> List[Dict]:
        if not self.storage:
            return []
        return await self.storage.get_regime_history(symbol, start, end)

    async def get_instruments(self) -> List[Dict]:
        # Placeholder for instrument discovery.
        # Ideally fetches from Convex or Thalex Adapter cache.
        # For now return static list or query DB distinct symbols
        return [
            {"symbol": "BTC-PERPETUAL", "type": "future"},
            {"symbol": "ETH-PERPETUAL", "type": "future"},
        ]
