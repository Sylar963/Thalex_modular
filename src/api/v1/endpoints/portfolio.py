from fastapi import APIRouter, Depends, Query
from typing import List, Dict, Optional
from ...repositories import PortfolioRepository
from ...dependencies import get_portfolio_repo

router = APIRouter()


@router.get("/summary", response_model=Dict)
async def get_summary(repo: PortfolioRepository = Depends(get_portfolio_repo)):
    return await repo.get_summary()


@router.get("/positions", response_model=List[Dict])
async def get_positions(
    exchange: Optional[str] = Query(None, description="Filter by exchange name"),
    repo: PortfolioRepository = Depends(get_portfolio_repo),
):
    return await repo.get_positions(exchange=exchange)


@router.get("/aggregate", response_model=Dict)
async def get_aggregate(repo: PortfolioRepository = Depends(get_portfolio_repo)):
    return await repo.get_aggregate()


@router.get("/history", response_model=List[Dict])
async def get_history(repo: PortfolioRepository = Depends(get_portfolio_repo)):
    return await repo.get_history()


@router.get("/executions", response_model=List[Dict])
async def get_executions(
    start: float,
    end: float,
    symbol: str = None,
    repo: PortfolioRepository = Depends(get_portfolio_repo),
):
    return await repo.get_executions(start, end, symbol)
