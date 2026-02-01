from fastapi import APIRouter, Depends
from typing import List, Dict
from ..repositories import MetricsRepository

router = APIRouter()


def get_repository():
    return MetricsRepository()


@router.get("/market/metrics", response_model=List[Dict])
async def read_metrics(
    limit: int = 100, repo: MetricsRepository = Depends(get_repository)
):
    """
    Get latest market metrics (IV, Expected Move, Prices).
    """
    return repo.get_latest_metrics(limit)


@router.get("/simulation/report")
async def read_simulation_report(repo: MetricsRepository = Depends(get_repository)):
    """
    Get the latest simulation performance report.
    """
    return repo.get_simulation_report()
