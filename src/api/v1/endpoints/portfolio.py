from fastapi import APIRouter, Depends
from typing import List, Dict
from ...repositories import PortfolioRepository
from ...dependencies import get_portfolio_repo

router = APIRouter()


@router.get("/summary", response_model=Dict)
async def get_summary(repo: PortfolioRepository = Depends(get_portfolio_repo)):
    """Global account state."""
    return await repo.get_summary()


@router.get("/positions", response_model=List[Dict])
async def get_positions(repo: PortfolioRepository = Depends(get_portfolio_repo)):
    """Active positions with Greeks."""
    return await repo.get_positions()


@router.get("/history", response_model=List[Dict])
async def get_history(repo: PortfolioRepository = Depends(get_portfolio_repo)):
    """Historical realized PNL curve."""
    return await repo.get_history()


@router.get("/executions", response_model=List[Dict])
async def get_executions(
    start: float,
    end: float,
    symbol: str = None,
    repo: PortfolioRepository = Depends(get_portfolio_repo),
):
    """Bot executions (fills)."""
    return await repo.get_executions(start, end, symbol)
