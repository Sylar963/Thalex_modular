from typing import List, Dict, Optional
from .base_repository import BaseRepository
from ...domain.entities import Position


class PortfolioRepository(BaseRepository):
    async def get_summary(self) -> Dict:
        # Derived summary from database positions
        # Ideally, we'd have a separate account balance table, but for now:
        if not self.storage:
            return {
                "equity": 0.0,
                "margin_used": 0.0,
                "margin_available": 0.0,
                "daily_pnl": 0.0,
                "unrealized_pnl": 0.0,
                "positions_count": 0,
            }

        positions = await self.storage.get_latest_positions()
        unrealized_pnl = sum(getattr(p, "unrealized_pnl", 0.0) for p in positions)

        return {
            "equity": 150000.00,  # Mocked until balance table added
            "margin_used": 25000.00,
            "margin_available": 125000.00,
            "daily_pnl": 1250.50,
            "unrealized_pnl": unrealized_pnl,
            "positions_count": len(positions),
        }

    async def get_positions(self, exchange: Optional[str] = None) -> List[Dict]:
        if not self.storage:
            return []

        positions = await self.storage.get_latest_positions()

        if exchange:
            positions = [
                p for p in positions if getattr(p, "exchange", None) == exchange
            ]

        return [
            {
                "symbol": p.symbol,
                "size": p.size,
                "entry_price": p.entry_price,
                "mark_price": getattr(p, "mark_price", 0.0),
                "unrealized_pnl": getattr(p, "unrealized_pnl", 0.0),
                "exchange": getattr(p, "exchange", "unknown"),
                "delta": getattr(p, "delta", 0.0),
                "gamma": getattr(p, "gamma", 0.0),
                "theta": getattr(p, "theta", 0.0),
            }
            for p in positions
        ]

    async def get_aggregate(self) -> Dict:
        if not self.storage:
            return {
                "total_positions": 0,
                "total_unrealized_pnl": 0.0,
                "by_exchange": {},
                "by_symbol": {},
            }

        positions = await self.storage.get_latest_positions()

        by_exchange = {}
        by_symbol = {}
        total_pnl = 0.0

        for p in positions:
            exch = getattr(p, "exchange", "unknown")
            sym = p.symbol
            pnl = getattr(p, "unrealized_pnl", 0.0)

            if exch not in by_exchange:
                by_exchange[exch] = {
                    "count": 0,
                    "total_size": 0.0,
                    "unrealized_pnl": 0.0,
                }
            by_exchange[exch]["count"] += 1
            by_exchange[exch]["total_size"] += abs(p.size)
            by_exchange[exch]["unrealized_pnl"] += pnl

            if sym not in by_symbol:
                by_symbol[sym] = {"net_size": 0.0, "unrealized_pnl": 0.0}
            by_symbol[sym]["net_size"] += p.size
            by_symbol[sym]["unrealized_pnl"] += pnl

            total_pnl += pnl

        return {
            "total_positions": len(positions),
            "total_unrealized_pnl": total_pnl,
            "by_exchange": by_exchange,
            "by_symbol": by_symbol,
        }

    async def get_history(self) -> List[Dict]:
        # Placeholder for historical PnL
        return [
            {"timestamp": 1700000000, "pnl": 0.0},
            {"timestamp": 1700086400, "pnl": 100.0},
            {"timestamp": 1700172800, "pnl": 250.0},
        ]

    async def get_executions(
        self, start: float, end: float, symbol: Optional[str] = None
    ) -> List[Dict]:
        """Fetch bot executions (private fills)."""
        if not self.storage:
            return []

        # Check if storage has the method (it should now)
        if hasattr(self.storage, "get_executions"):
            return await self.storage.get_executions(start, end, symbol)
        return []
