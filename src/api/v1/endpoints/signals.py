from fastapi import APIRouter, Depends, Query
from typing import List, Dict, Optional
import json
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
            "market_impact": row["market_impact"],
            "immediate_flow": row["immediate_flow"],
            "orh": row["orh"],
            "orl": row["orl"],
            "orm": row["orm"],
            "breakout_direction": row["breakout_direction"],
        }
        for row in rows
    ]


@router.get("/hft", response_model=List[Dict])
async def get_hft_signals(
    symbol: str = Query(..., description="Symbol to fetch HFT signals for"),
    exchange: Optional[str] = Query(None, description="Filter by exchange"),
    limit: int = Query(100, ge=1, le=1000),
    db=Depends(get_db_adapter),
):
    if not db or not db.pool:
        return []

    query = """
        SELECT time, symbol, exchange, toxicity_score, pull_rate, quote_stability, 
               size_asymmetry, bid, ask, spread
        FROM hft_signals
        WHERE symbol = $1
    """
    params = [symbol]

    if exchange:
        query += " AND exchange = $2"
        params.append(exchange)

    query += " ORDER BY time DESC LIMIT $" + str(len(params) + 1)
    params.append(limit)

    rows = await db.pool.fetch(query, *params)
    return [
        {
            "time": row["time"].isoformat() if row["time"] else None,
            "symbol": row["symbol"],
            "exchange": row["exchange"],
            "toxicity_score": row["toxicity_score"],
            "pull_rate": row["pull_rate"],
            "quote_stability": row["quote_stability"],
            "size_asymmetry": row["size_asymmetry"],
            "bid": row["bid"],
            "ask": row["ask"],
            "spread": row["spread"],
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
            "market_impact": row["market_impact"],
            "immediate_flow": row["immediate_flow"],
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


@router.get("/bot_status", response_model=List[Dict])
async def get_bot_status(
    symbol: Optional[str] = None,
    limit: int = Query(10, ge=1, le=100),
    db=Depends(get_db_adapter),
):
    if not db or not db.pool:
        return []

    query = """
        SELECT time, symbol, exchange, risk_state, trend_state, execution_mode, risk_breach, metadata
        FROM bot_status
    """
    params = []

    if symbol:
        query += " WHERE symbol = $1"
        params.append(symbol)

    query += " ORDER BY time DESC LIMIT $" + str(len(params) + 1)
    params.append(limit)

    rows = await db.pool.fetch(query, *params)
    return [
        {
            "time": row["time"].isoformat() if row["time"] else None,
            "symbol": row["symbol"],
            "exchange": row["exchange"],
            "risk_state": row["risk_state"],
            "trend_state": row["trend_state"],
            "execution_mode": row["execution_mode"],
            "risk_breach": row["risk_breach"],
            "metadata": json.loads(row["metadata"]) if row["metadata"] else {},
        }
        for row in rows
    ]
