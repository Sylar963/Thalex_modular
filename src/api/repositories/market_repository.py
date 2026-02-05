from typing import List, Dict, Optional
from .base_repository import BaseRepository
from ...domain.entities import Ticker


class MarketRepository(BaseRepository):
    async def trigger_sync(self, symbol: str, venue: str = "bybit") -> dict:
        from ...services.data_ingestor import DataIngestor

        if not self.storage:
            return {"status": "error", "message": "Storage not available"}

        ingestor = DataIngestor(self.storage)
        try:
            await ingestor.sync_symbol(symbol, venue)
            return {"status": "success", "message": f"Sync triggered for {symbol}"}
        except Exception as e:
            return {"status": "error", "message": str(e)}
        finally:
            await ingestor.close()

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
        return [
            {"symbol": "BTC-PERPETUAL", "type": "future"},
            {"symbol": "ETH-PERPETUAL", "type": "future"},
        ]

    async def get_volume_bars(
        self, symbol: str, volume_threshold: float, limit: int
    ) -> List[Dict]:
        if not self.storage:
            return []
        if hasattr(self.storage, "get_volume_bars"):
            return await self.storage.get_volume_bars(symbol, volume_threshold, limit)
        return []

    async def get_tick_bars(
        self, symbol: str, tick_count: int, limit: int, exchange: str = "thalex"
    ) -> List[Dict]:
        if not self.storage:
            return []
        if hasattr(self.storage, "get_tick_bars"):
            return await self.storage.get_tick_bars(symbol, tick_count, limit, exchange)
        return []

    async def get_signal_history(
        self, symbol: str, start: float, end: float, signal_type: Optional[str] = None
    ) -> List[Dict]:
        if not self.storage:
            return []
        if hasattr(self.storage, "get_signal_history"):
            return await self.storage.get_signal_history(
                symbol, start, end, signal_type
            )
        return []

    async def get_open_range_levels(self) -> Dict:
        if hasattr(self, "_or_engine") and self._or_engine:
            return self._or_engine.get_chart_levels()
        return {
            "orh": None,
            "orl": None,
            "orm": None,
            "day_dir": 0,
            "up_targets": [],
            "down_targets": [],
            "up_signal": False,
            "down_signal": False,
            "session_active": False,
        }
