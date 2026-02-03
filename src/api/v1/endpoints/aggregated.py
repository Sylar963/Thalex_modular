from fastapi import APIRouter, Depends
from typing import Dict, List, Any

from ...dependencies import get_db_adapter

router = APIRouter()


@router.get("/positions")
async def get_aggregated_positions(db=Depends(get_db_adapter)) -> List[Dict[str, Any]]:
    if not db or not db.pool:
        return []
    try:
        async with db.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT symbol, exchange, size, entry_price, unrealized_pnl, last_update
                FROM portfolio_positions
                WHERE size != 0
                ORDER BY exchange, symbol
            """)
            return [dict(r) for r in rows]
    except Exception:
        return []


@router.get("/risk")
async def get_global_risk(db=Depends(get_db_adapter)) -> Dict[str, Any]:
    if not db or not db.pool:
        return {"total_delta": 0.0, "total_gamma": 0.0, "total_theta": 0.0}
    try:
        async with db.pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT 
                    COALESCE(SUM(delta), 0) as total_delta,
                    COALESCE(SUM(gamma), 0) as total_gamma,
                    COALESCE(SUM(theta), 0) as total_theta,
                    COALESCE(SUM(size), 0) as net_position
                FROM portfolio_positions
            """)
            return {
                "total_delta": row["total_delta"],
                "total_gamma": row["total_gamma"],
                "total_theta": row["total_theta"],
                "net_position": row["net_position"],
            }
    except Exception:
        return {
            "total_delta": 0.0,
            "total_gamma": 0.0,
            "total_theta": 0.0,
            "net_position": 0.0,
        }


@router.get("/pnl")
async def get_aggregated_pnl(db=Depends(get_db_adapter)) -> Dict[str, Any]:
    if not db or not db.pool:
        return {"total_unrealized_pnl": 0.0}
    try:
        async with db.pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT 
                    COALESCE(SUM(unrealized_pnl), 0) as total_unrealized_pnl
                FROM portfolio_positions
            """)
            return {"total_unrealized_pnl": row["total_unrealized_pnl"]}
    except Exception:
        return {"total_unrealized_pnl": 0.0}


@router.get("/summary")
async def get_exchange_summary(db=Depends(get_db_adapter)) -> List[Dict[str, Any]]:
    if not db or not db.pool:
        return []
    try:
        async with db.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT 
                    exchange,
                    COUNT(*) as position_count,
                    COALESCE(SUM(size), 0) as total_size,
                    COALESCE(SUM(unrealized_pnl), 0) as total_pnl
                FROM portfolio_positions
                WHERE size != 0
                GROUP BY exchange
            """)
            return [dict(r) for r in rows]
    except Exception:
        return []
