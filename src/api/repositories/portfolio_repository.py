from typing import List, Dict, Optional
from .base_repository import BaseRepository
from ...domain.entities import Position


class PortfolioRepository(BaseRepository):
    async def get_summary(self) -> Dict:
        # Derived summary from database positions
        # Ideally, we'd have a separate account balance table, but for now:
        positions = await self.db_adapter.get_latest_positions()
        unrealized_pnl = sum(getattr(p, "unrealized_pnl", 0.0) for p in positions)

        return {
            "equity": 150000.00,  # Mocked until balance table added
            "margin_used": 25000.00,
            "margin_available": 125000.00,
            "daily_pnl": 1250.50,
            "unrealized_pnl": unrealized_pnl,
            "positions_count": len(positions),
        }

    async def get_positions(self) -> List[Dict]:
        """Fetch active positions from DB."""
        if not self.db_adapter:
            return []

        positions = await self.db_adapter.get_latest_positions()
        return [
            {
                "symbol": p.symbol,
                "size": p.size,
                "entry_price": p.entry_price,
                "mark_price": 0.0,  # Will be filled if mark_price stored
                "unrealized_pnl": 0.0,
                "delta": 0.0,
                "gamma": 0.0,
                "theta": 0.0,
            }
            for p in positions
        ]

    async def get_history(self) -> List[Dict]:
        # Placeholder for historical PnL
        return [
            {"timestamp": 1700000000, "pnl": 0.0},
            {"timestamp": 1700086400, "pnl": 100.0},
            {"timestamp": 1700172800, "pnl": 250.0},
        ]
