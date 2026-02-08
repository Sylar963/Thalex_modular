import logging
from typing import List, Dict, Optional
from .base_repository import BaseRepository
from ...domain.entities import Ticker

logger = logging.getLogger(__name__)


class MarketRepository(BaseRepository):
    def __init__(self, storage, or_engine=None):
        super().__init__(storage)
        self._or_engine = or_engine

    async def trigger_sync(self, symbol: str, venue: str = "bybit") -> dict:
        from ...services.data_ingestor import DataIngestionService
        import os

        if not self.storage:
            return {"status": "error", "message": "Storage not available"}

        # Get DSN from storage or ENV
        db_dsn = getattr(self.storage, "dsn", None)
        if not db_dsn:
            db_user = os.getenv("DATABASE_USER", "postgres")
            db_pass = os.getenv("DATABASE_PASSWORD", "password")
            db_host = os.getenv("DATABASE_HOST", "localhost")
            db_name = os.getenv("DATABASE_NAME", "thalex_trading")
            db_port = os.getenv("DATABASE_PORT", "5432")
            db_dsn = f"postgresql://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}"

        ingestor = DataIngestionService(db_dsn, self._or_engine)
        try:
            await ingestor.storage.connect()
            await ingestor.sync_symbol(symbol, venue)
            return {"status": "success", "message": f"Sync triggered for {symbol}"}
        except Exception as e:
            logger.error(f"Sync failed for {symbol}: {e}")
            return {"status": "error", "message": str(e)}
        finally:
            await ingestor.storage.disconnect()

    async def get_recent_tickers(self, symbol: str, limit: int = 100) -> List[Ticker]:
        if not self.storage:
            return []
        return await self.storage.get_recent_tickers(symbol, limit)

    async def get_history(
        self,
        symbol: str,
        start: float,
        end: float,
        resolution: str,
        exchange: str = "thalex",
    ) -> List[Dict]:
        if not self.storage:
            return []
        # Assuming the storage adapter has this extended method (we added it)
        if hasattr(self.storage, "get_history"):
            return await self.storage.get_history(
                symbol, start, end, resolution, exchange
            )
        return []

    async def get_regime_history(
        self, symbol: str, start: Optional[float] = None, end: Optional[float] = None
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
        self, symbol: str, volume_threshold: float, limit: int, exchange: str = "thalex"
    ) -> List[Dict]:
        if not self.storage:
            return []
        if hasattr(self.storage, "get_volume_bars"):
            return await self.storage.get_volume_bars(
                symbol, volume_threshold, limit, exchange
            )
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

    async def get_open_range_levels(self, symbol: str) -> Dict:
        if hasattr(self, "_or_engine") and self._or_engine:
            return self._or_engine.get_chart_levels(symbol)
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
