from fastapi import APIRouter, Depends, HTTPException, Query
from typing import Dict, Any, List
import logging

from ...dependencies import get_strategy_manager
from ....use_cases.strategy_manager import MultiExchangeStrategyManager

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/metrics", response_model=List[Dict[str, Any]])
async def get_strategy_metrics(
    exchange: str = Query(None, description="Filter by exchange name"),
    symbol: str = Query(None, description="Filter by symbol"),
    manager: MultiExchangeStrategyManager = Depends(get_strategy_manager),
):
    """
    Get real-time strategy metrics (Gamma, Risk, Reservation Price) from memory.
    """
    if not manager:
        raise HTTPException(status_code=503, detail="Strategy Manager not available")

    results = []

    for venue_name, venue in manager.venues.items():
        # Filtering
        if exchange and venue_name.lower() != exchange.lower():
            continue
        if symbol and venue.config.symbol != symbol:
            continue

        # Retrieve cached metrics
        metrics = getattr(venue, "_last_strategy_metrics", None)
        if metrics:
            results.append(metrics)
        else:
            # Fallback for venues without metrics yet (or not running Avellaneda)
            results.append(
                {
                    "exchange": venue_name,
                    "symbol": venue.config.symbol,
                    "status": "no_metrics_available",
                    "timestamp": venue.market_state.timestamp
                    if venue.market_state
                    else 0,
                }
            )

    return results
