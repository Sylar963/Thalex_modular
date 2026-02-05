from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from typing import List, Dict
from ...repositories import SimulationRepository
from ...dependencies import get_simulation_repo
from pydantic import BaseModel
import asyncio
import json

router = APIRouter()


class SimulationConfig(BaseModel):
    symbol: str = "BTC-PERPETUAL"
    venue: str = "bybit"
    start_date: float
    end_date: float
    strategy_config: Dict = {}
    risk_config: Dict = {}


class LiveSimConfig(BaseModel):
    symbol: str = "BTC-PERPETUAL"
    initial_balance: float = 1000.0
    latency_ms: float = 50.0
    slippage_ticks: float = 0.0
    strategy_config: Dict = {}
    risk_config: Dict = {}


@router.get("/runs", response_model=List[Dict])
async def get_runs(repo: SimulationRepository = Depends(get_simulation_repo)):
    return await repo.get_runs()


@router.post("/start", response_model=Dict)
async def start_simulation(
    config: SimulationConfig, repo: SimulationRepository = Depends(get_simulation_repo)
):
    return await repo.start_simulation(config.dict())


@router.get("/{run_id}/equity", response_model=List[Dict])
async def get_equity_curve(
    run_id: str, repo: SimulationRepository = Depends(get_simulation_repo)
):
    return await repo.get_equity_curve(run_id)


@router.get("/{run_id}/fills", response_model=List[Dict])
async def get_fills(
    run_id: str, repo: SimulationRepository = Depends(get_simulation_repo)
):
    return await repo.get_fills(run_id)


@router.get("/{run_id}/stats", response_model=Dict)
async def get_stats(
    run_id: str, repo: SimulationRepository = Depends(get_simulation_repo)
):
    return await repo.get_stats(run_id)


@router.get("/live/status", response_model=Dict)
async def get_live_status(repo: SimulationRepository = Depends(get_simulation_repo)):
    return repo.get_live_status()


@router.post("/live/start", response_model=Dict)
async def start_live_sim(
    config: LiveSimConfig, repo: SimulationRepository = Depends(get_simulation_repo)
):
    return await repo.start_live_sim(config.dict())


@router.post("/live/stop", response_model=Dict)
async def stop_live_sim(repo: SimulationRepository = Depends(get_simulation_repo)):
    return await repo.stop_live_sim()


@router.get("/live/fills", response_model=List[Dict])
async def get_live_fills(
    limit: int = 100, repo: SimulationRepository = Depends(get_simulation_repo)
):
    return repo.get_live_fills(limit)


@router.get("/live/equity")
async def stream_live_equity(repo: SimulationRepository = Depends(get_simulation_repo)):
    async def event_generator():
        async for snapshot in repo.subscribe_live_equity():
            data = {
                "timestamp": snapshot.timestamp,
                "balance": snapshot.balance,
                "equity": snapshot.equity,
                "unrealized_pnl": snapshot.unrealized_pnl,
            }
            yield f"data: {json.dumps(data)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
