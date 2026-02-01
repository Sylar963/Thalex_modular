from fastapi import APIRouter, Depends
from typing import List, Dict
from ...repositories import SimulationRepository
from ...dependencies import get_simulation_repo
from pydantic import BaseModel

router = APIRouter()


class SimulationConfig(BaseModel):
    symbol: str = "BTC-PERPETUAL"
    start_date: float  # Timestamp
    end_date: float  # Timestamp
    strategy_config: Dict = {}
    risk_config: Dict = {}


@router.get("/runs", response_model=List[Dict])
async def get_runs(repo: SimulationRepository = Depends(get_simulation_repo)):
    """List previous backfill/sim runs."""
    return await repo.get_runs()


@router.post("/start", response_model=Dict)
async def start_simulation(
    config: SimulationConfig, repo: SimulationRepository = Depends(get_simulation_repo)
):
    """Trigger a new run."""
    return await repo.start_simulation(config.dict())


@router.get("/{run_id}/equity", response_model=List[Dict])
async def get_equity_curve(
    run_id: str, repo: SimulationRepository = Depends(get_simulation_repo)
):
    """Fetch equity curve for a specific run."""
    return await repo.get_equity_curve(run_id)
