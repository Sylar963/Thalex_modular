import asyncpg
import logging
import json
import time
from typing import List, Optional
from ...domain.interfaces import StorageGateway
from ...domain.entities import Ticker, Trade, Position

logger = logging.getLogger(__name__)


class TimescaleDBAdapter(StorageGateway):
    """
    Adapter for TimescaleDB (PostgreSQL) storage.
    Handles connection pooling and efficient batch insertion of market data.
    """

    def __init__(self, dsn: str):
        self.dsn = dsn
        self.pool: Optional[asyncpg.Pool] = None

    async def connect(self):
        """Initialize connection pool and schema."""
        try:
            logger.info("Connecting to TimescaleDB...")
            self.pool = await asyncpg.create_pool(
                self.dsn, min_size=1, max_size=10, command_timeout=60
            )
            await self._init_schema()
            logger.info("Connected to TimescaleDB")
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            raise

    async def disconnect(self):
        if self.pool:
            await self.pool.close()

    async def _init_schema(self):
        """Create necessary tables and hypertables."""
        async with self.pool.acquire() as conn:
            # 1. Tickers Table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS market_tickers (
                    time TIMESTAMPTZ NOT NULL,
                    symbol TEXT NOT NULL,
                    bid DOUBLE PRECISION,
                    ask DOUBLE PRECISION,
                    last DOUBLE PRECISION,
                    volume DOUBLE PRECISION
                );
            """)
            # Convert to hypertable (if not already)
            # Try/Catch block in SQL usually needed if already exists, but "IF NOT EXISTS" in Timescale logic is separate.
            # We'll assume standard Postgres for MVP, user can enable timescaledb extension manually or we run query
            # asking 'SELECT create_hypertable' if desired. For now, simple tables.

            try:
                await conn.execute(
                    "SELECT create_hypertable('market_tickers', 'time', if_not_exists => TRUE);"
                )
            except Exception as e:
                logger.debug(
                    f"Hypertable creation skipped (might be standard Postgres): {e}"
                )

            # 2. Trades Table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS market_trades (
                    time TIMESTAMPTZ NOT NULL,
                    symbol TEXT NOT NULL,
                    price DOUBLE PRECISION,
                    size DOUBLE PRECISION,
                    side TEXT,
                    trade_id TEXT
                );
            """)
            try:
                await conn.execute(
                    "SELECT create_hypertable('market_trades', 'time', if_not_exists => TRUE);"
                )
            except Exception:
                pass

    async def save_ticker(self, ticker: Ticker):
        if not self.pool:
            return
        try:
            async with self.pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO market_tickers (time, symbol, bid, ask, last, volume)
                    VALUES (to_timestamp($1), $2, $3, $4, $5, $6)
                    """,
                    ticker.timestamp,
                    ticker.symbol,
                    ticker.bid,
                    ticker.ask,
                    ticker.last,
                    ticker.volume,
                )
        except Exception as e:
            logger.error(f"Failed to save ticker: {e}")

    async def save_trade(self, trade: Trade):
        if not self.pool:
            return
        try:
            async with self.pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO market_trades (time, symbol, price, size, side, trade_id)
                    VALUES (to_timestamp($1), $2, $3, $4, $5, $6)
                    """,
                    trade.timestamp,
                    trade.symbol,
                    trade.price,
                    trade.size,
                    trade.side.value,
                    trade.id,
                )
        except Exception as e:
            logger.error(f"Failed to save trade: {e}")

    async def save_position(self, position: Position):
        # We might store positions in a separate table
        pass

    async def get_recent_tickers(self, symbol: str, limit: int = 100) -> List[Ticker]:
        return []
