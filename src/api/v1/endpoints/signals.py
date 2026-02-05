from fastapi import APIRouter, Depends, Query
from typing import List, Dict, Optional
from ...dependencies import get_db_adapter

router = APIRouter()


@router.get("/history", response_model=List[Dict])
async def get_signal_history(
    symbol: str = Query(..., description="Symbol to fetch signals for"),
    signal_type: Optional[str] = Query(
        None, description="Filter by signal type: vamp, open_range, regime"
    ),
    limit: int = Query(100, ge=1, le=1000),
    db=Depends(get_db_adapter),
):
    if not db or not db.pool:
        return []

    query = """
        SELECT time, symbol, signal_type, momentum, reversal, volatility, exhaustion,
               gamma_adjustment, reservation_price_offset, volatility_adjustment,
               vamp_value, market_impact, immediate_flow, orh, orl, orm, breakout_direction
        FROM market_signals
        WHERE symbol = $1
    """
    params = [symbol]

    if signal_type:
        query += " AND signal_type = $2"
        params.append(signal_type)

    query += " ORDER BY time DESC LIMIT $" + str(len(params) + 1)
    params.append(limit)

    rows = await db.pool.fetch(query, *params)
    return [
        {
            "time": row["time"].isoformat() if row["time"] else None,
            "symbol": row["symbol"],
            "signal_type": row["signal_type"],
            "momentum": row["momentum"],
            "reversal": row["reversal"],
            "volatility": row["volatility"],
            "exhaustion": row["exhaustion"],
            "vamp_value": row["vamp_value"],
            "orh": row["orh"],
            "orl": row["orl"],
            "orm": row["orm"],
            "breakout_direction": row["breakout_direction"],
        }
        for row in rows
    ]


@router.get("/latest", response_model=Dict)
async def get_latest_signals(
    symbol: str = Query(..., description="Symbol to fetch signals for"),
    db=Depends(get_db_adapter),
):
    if not db or not db.pool:
        return {}

    query = """
        SELECT DISTINCT ON (signal_type) signal_type, vamp_value, momentum, reversal, 
               volatility, exhaustion, orh, orl, orm, breakout_direction, time
        FROM market_signals
        WHERE symbol = $1
        ORDER BY signal_type, time DESC
    """
    rows = await db.pool.fetch(query, symbol)

    return {
        row["signal_type"]: {
            "vamp_value": row["vamp_value"],
            "momentum": row["momentum"],
            "reversal": row["reversal"],
            "volatility": row["volatility"],
            "exhaustion": row["exhaustion"],
            "orh": row["orh"],
            "orl": row["orl"],
            "orm": row["orm"],
            "breakout_direction": row["breakout_direction"],
            "time": row["time"].isoformat() if row["time"] else None,
        }
        for row in rows
    }
