from fastapi import APIRouter, Depends
from typing import List, Dict
from ...repositories import MarketRepository
from ...dependencies import get_market_repo
from dataclasses import asdict

router = APIRouter()


@router.get("/instruments", response_model=List[Dict])
async def get_instruments(repo: MarketRepository = Depends(get_market_repo)):
    """List available trading instruments."""
    return await repo.get_instruments()


@router.get("/history/{symbol}", response_model=List[Dict])
async def get_history(
    symbol: str,
    start: float,
    end: float,
    resolution: str = "1m",
    repo: MarketRepository = Depends(get_market_repo),
):
    """OHLCV / IV History for charting."""
    return await repo.get_history(symbol, start, end, resolution)


@router.get("/ticker/{symbol}", response_model=List[Dict])
async def get_recent_ticker(
    symbol: str, limit: int = 1, repo: MarketRepository = Depends(get_market_repo)
):
    """Live Top of Book (or recent history)."""
    tickers = await repo.get_recent_tickers(symbol, limit)
    return [asdict(t) for t in tickers]
