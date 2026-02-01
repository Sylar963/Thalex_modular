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
        if not self.pool:
            return []
        try:
            async with self.pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT time, symbol, bid, ask, last, volume
                    FROM market_tickers
                    WHERE symbol = $1
                    ORDER BY time DESC
                    LIMIT $2
                    """,
                    symbol,
                    limit,
                )
                return [
                    Ticker(
                        symbol=r["symbol"],
                        bid=r["bid"],
                        ask=r["ask"],
                        bid_size=0.0,  # Not stored currently
                        ask_size=0.0,  # Not stored currently
                        last=r["last"],
                        volume=r["volume"],
                        timestamp=r["time"].timestamp(),
                    )
                    for r in rows
                ]
        except Exception as e:
            logger.error(f"Failed to fetch recent tickers: {e}")
            return []

    async def get_history(
        self, symbol: str, start: float, end: float, resolution: str = "1m"
    ) -> List[Dict]:
        """
        Fetch OHLCV history for a symbol.
        Uses TimescaleDB's time_bucket for aggregation.
        """
        if not self.pool:
            return []

        # rudimentary interval mapping
        interval_map = {
            "1m": "1 minute",
            "5m": "5 minutes",
            "1h": "1 hour",
            "1d": "1 day",
        }
        bucket_interval = interval_map.get(resolution, "1 minute")

        try:
            async with self.pool.acquire() as conn:
                # We use market_trades for OHLCV if available, or tickers if that's what we have.
                # Assuming we construct candles from trades for accuracy,
                # but if trades are sparse, we might use tickers (last price).
                # For this implementation, let's use market_trades for OHLCV.

                rows = await conn.fetch(
                    f"""
                    SELECT
                        time_bucket($1, time) AS bucket,
                        FIRST(price, time) as open,
                        MAX(price) as high,
                        MIN(price) as low,
                        LAST(price, time) as close,
                        SUM(size) as volume
                    FROM market_trades
                    WHERE symbol = $2
                      AND time >= to_timestamp($3)
                      AND time <= to_timestamp($4)
                    GROUP BY bucket
                    ORDER BY bucket ASC
                    """,
                    bucket_interval,
                    symbol,
                    start,
                    end,
                )

                return [
                    {
                        "time": r["bucket"].timestamp(),
                        "open": r["open"],
                        "high": r["high"],
                        "low": r["low"],
                        "close": r["close"],
                        "volume": r["volume"] or 0.0,
                    }
                    for r in rows
                ]
        except Exception as e:
            logger.error(f"Failed to fetch history: {e}")
            return []
