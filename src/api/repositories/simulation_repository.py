from typing import List, Dict, Any, AsyncGenerator, Optional
from .base_repository import BaseRepository
from ...use_cases.simulation_engine import SimulationEngine
from ...use_cases.sim_state_manager import sim_state_manager
from ...domain.strategies.avellaneda import AvellanedaStoikovStrategy
from ...domain.risk.basic_manager import BasicRiskManager
from ...domain.entities.pnl import EquitySnapshot
from dataclasses import asdict


class SimulationRepository(BaseRepository):
    def __init__(self, storage):
        super().__init__(storage)
        self.runs: Dict[str, Any] = {}

    async def get_runs(self) -> List[Dict]:
        return [
            {
                "run_id": r_id,
                "status": "completed",
                "start_time": r.start_time,
                "end_time": r.end_time,
                "pnl": r.stats.total_pnl if r.stats else 0,
            }
            for r_id, r in self.runs.items()
        ]

    async def start_simulation(self, params: Dict) -> Dict:
        from ...adapters.storage.bybit_history_adapter import BybitHistoryAdapter

        strategy = AvellanedaStoikovStrategy()
        strategy.setup(params.get("strategy_config", {}))

        risk = BasicRiskManager()
        risk.setup(params.get("risk_config", {}))

        # Initialize modular components
        # Note: We assume self.storage is the TimescaleDBAdapter instance
        history_provider = BybitHistoryAdapter(self.storage)

        engine = SimulationEngine(
            strategy=strategy,
            risk_manager=risk,
            data_provider=history_provider,
            history_db=self.storage,
            initial_balance=params.get("initial_balance", 1000.0),
        )

        result = await engine.run_simulation(
            symbol=params.get("symbol", "BTC-PERPETUAL"),
            venue=params.get("venue", "bybit"),
            start_time=float(params["start_date"]),
            end_time=float(params["end_date"]),
        )

        self.runs[result.run_id] = result

        return {
            "run_id": result.run_id,
            "status": "completed",
            "stats": asdict(result.stats) if result.stats else {},
        }

    async def get_equity_curve(self, run_id: str) -> List[Dict]:
        result = self.runs.get(run_id)
        if not result:
            return []
        return [asdict(snapshot) for snapshot in result.equity_curve]

    async def get_fills(self, run_id: str) -> List[Dict]:
        result = self.runs.get(run_id)
        if not result:
            return []
        return [
            {
                "timestamp": f.timestamp,
                "symbol": f.symbol,
                "side": f.side,
                "price": f.price,
                "size": f.size,
                "fee": f.fee,
                "realized_pnl": f.realized_pnl,
                "balance_after": f.balance_after,
            }
            for f in result.fills
        ]

    async def get_stats(self, run_id: str) -> Dict:
        result = self.runs.get(run_id)
        if not result or not result.stats:
            return {"edge": [], "decay": []}

        # Map flat stats to frontend array format
        # Frontend expects arrays of metrics over time or windows
        # Our current backend stats are just averages (one point)
        # We'll create a single-item array for now to suffice the interface

        stats = result.stats

        edge_metrics = [
            {
                "timestamp": result.end_time,
                "edge": stats.avg_edge,
                "count": stats.total_trades,
            }
        ]

        decay_metrics = [
            {"window": "5s", "decay": stats.avg_adverse_selection_5s},
            {"window": "30s", "decay": stats.avg_adverse_selection_30s},
        ]

        return {"edge": edge_metrics, "decay": decay_metrics, "summary": asdict(stats)}

    def get_live_status(self) -> Dict:
        return sim_state_manager.get_status()

    async def start_live_sim(self, params: Dict) -> Dict:
        await sim_state_manager.start(
            symbol=params.get("symbol", "BTC-PERPETUAL"),
            initial_balance=params.get("initial_balance", 1000.0),
            mode="shadow",
        )
        return {
            "status": "started",
            "message": "Live simulation started in shadow mode",
        }

    async def stop_live_sim(self) -> Dict:
        await sim_state_manager.stop()
        return {"status": "stopped", "message": "Live simulation stopped"}

    def get_live_fills(self, limit: int = 100) -> List[Dict]:
        return sim_state_manager.get_fills(limit)

    def get_live_equity_history(self, limit: int = 1000) -> List[Dict]:
        return sim_state_manager.get_equity_history(limit)

    async def subscribe_live_equity(self) -> AsyncGenerator[EquitySnapshot, None]:
        async for snapshot in sim_state_manager.subscribe_equity():
            yield snapshot
