from typing import List, Dict, Any
from .base_repository import BaseRepository
from ...use_cases.simulation_engine import SimulationEngine
from ...domain.strategies.avellaneda import AvellanedaStoikovStrategy
from ...domain.risk.basic_manager import BasicRiskManager


class SimulationRepository(BaseRepository):
    def __init__(self, storage):
        super().__init__(storage)
        self.runs: Dict[str, Any] = {}  # In-memory storage for runs in this session

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
        # 1. Setup Strategy & Risk
        strategy = AvellanedaStoikovStrategy()
        strategy.setup(params.get("strategy_config", {}))

        risk = BasicRiskManager()
        risk.setup(params.get("risk_config", {}))

        # 2. Engine
        engine = SimulationEngine(strategy, risk, self.storage)

        # 3. Run (Synchronous for now, but could be backgrounded)
        result = await engine.run_simulation(
            symbol=params.get("symbol", "BTC-PERPETUAL"),
            start_time=float(
                params["start_date"]
            ),  # Expecting timestamp or parseable str
            end_time=float(params["end_date"]),
        )

        self.runs[result.run_id] = result

        return {
            "run_id": result.run_id,
            "status": "completed",
            "stats": result.stats.__dict__ if result.stats else {},
        }

    async def get_equity_curve(self, run_id: str) -> List[Dict]:
        result = self.runs.get(run_id)
        if not result:
            return []
        return [snapshot.__dict__ for snapshot in result.equity_curve]
