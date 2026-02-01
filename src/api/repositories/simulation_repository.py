from typing import List, Dict
from .base_repository import BaseRepository


class SimulationRepository(BaseRepository):
    async def get_runs(self) -> List[Dict]:
        # Placeholder for simulation runs
        return [
            {
                "run_id": "sim_001",
                "status": "completed",
                "start_time": 1700000000,
                "end_time": 1700100000,
                "sharpe": 1.5,
                "return": 12.5,
            },
            {
                "run_id": "sim_002",
                "status": "failed",
                "start_time": 1700200000,
                "end_time": 1700200100,
                "sharpe": 0.0,
                "return": 0.0,
            },
        ]

    async def start_simulation(self, params: Dict) -> Dict:
        # Placeholder to trigger simulation
        return {"run_id": "sim_003", "status": "started", "params": params}
