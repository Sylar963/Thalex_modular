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


@router.get("/regime/history/{symbol}", response_model=List[Dict])
async def get_regime_history(
    symbol: str,
    start: float,
    end: float,
    repo: MarketRepository = Depends(get_market_repo),
):
    """Get historical regime metrics (RV, Trend, EM, Vol Delta)."""
    return await repo.get_regime_history(symbol, start, end)


@router.get("/chart/{symbol}/volume-bars", response_model=List[Dict])
async def get_volume_bars(
    symbol: str,
    threshold: float = 0.1,
    limit: int = 100,
    repo: MarketRepository = Depends(get_market_repo),
):
    """Get OHLCV candles aggregated by volume threshold (e.g., 0.1 BTC per candle)."""
    return await repo.get_volume_bars(symbol, threshold, limit)


@router.get("/chart/{symbol}/tick-bars", response_model=List[Dict])
async def get_tick_bars(
    symbol: str,
    tick_count: int = 2500,
    limit: int = 100,
    exchange: str = "thalex",
    repo: MarketRepository = Depends(get_market_repo),
):
    return await repo.get_tick_bars(symbol, tick_count, limit, exchange)


@router.get("/signals/history/{symbol}", response_model=List[Dict])
async def get_signal_history(
    symbol: str,
    start: float,
    end: float,
    signal_type: str = None,
    repo: MarketRepository = Depends(get_market_repo),
):
    return await repo.get_signal_history(symbol, start, end, signal_type)


@router.get("/signals/open-range/levels", response_model=Dict)
async def get_open_range_levels(repo: MarketRepository = Depends(get_market_repo)):
    return await repo.get_open_range_levels()


from pydantic import BaseModel


class SyncRequest(BaseModel):
    symbol: str
    venue: str = "bybit"


@router.post("/sync", response_model=Dict)
async def trigger_sync(
    request: SyncRequest, repo: MarketRepository = Depends(get_market_repo)
):
    """
    Trigger a manual data sync (gap-fill) for the specified symbol.
    Fetches missing historical data from the exchange and saves to DB.
    """
    return await repo.trigger_sync(request.symbol, request.venue)
