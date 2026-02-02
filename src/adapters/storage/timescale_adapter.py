import asyncpg
import logging
from datetime import timedelta
from typing import List, Optional, Dict, Any
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
            try:
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

                # 3. Portfolio Positions Table (Snapshot)
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS portfolio_positions (
                        symbol TEXT PRIMARY KEY,
                        size DOUBLE PRECISION NOT NULL,
                        entry_price DOUBLE PRECISION,
                        mark_price DOUBLE PRECISION,
                        unrealized_pnl DOUBLE PRECISION,
                        delta DOUBLE PRECISION,
                        gamma DOUBLE PRECISION,
                        theta DOUBLE PRECISION,
                        last_update TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
                    );
                """)
                # 4. Market Regimes Table
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS market_regimes (
                        time TIMESTAMPTZ NOT NULL,
                        symbol TEXT NOT NULL,
                        regime_name TEXT,
                        rv_fast DOUBLE PRECISION,
                        rv_mid DOUBLE PRECISION,
                        rv_slow DOUBLE PRECISION,
                        trend_fast DOUBLE PRECISION,
                        trend_mid DOUBLE PRECISION,
                        trend_slow DOUBLE PRECISION,
                        liquidity_score DOUBLE PRECISION,
                        expected_move_pct DOUBLE PRECISION,
                        atm_iv DOUBLE PRECISION,
                        vol_delta DOUBLE PRECISION,
                        is_overpriced BOOLEAN
                    );
                """)
                try:
                    await conn.execute(
                        "SELECT create_hypertable('market_regimes', 'time', if_not_exists => TRUE);"
                    )
                except Exception:
                    pass
            except Exception as e:
                logger.error(f"Failed to initialize schema: {e}")
                raise

    async def save_regime(self, symbol: str, regime: Dict[str, Any]):
        if not self.pool:
            return
        try:
            async with self.pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO market_regimes 
                    (time, symbol, regime_name, rv_fast, rv_mid, rv_slow, trend_fast, trend_mid, trend_slow, liquidity_score, expected_move_pct, atm_iv, vol_delta, is_overpriced)
                    VALUES (NOW(), $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
                    """,
                    symbol,
                    regime.get("name"),
                    regime.get("rv_fast"),
                    regime.get("rv_mid"),
                    regime.get("rv_slow"),
                    regime.get("trend_fast"),
                    regime.get("trend_mid"),
                    regime.get("trend_slow"),
                    regime.get("liquidity_score"),
                    regime.get("expected_move_pct"),
                    regime.get("atm_iv"),
                    regime.get("vol_delta"),
                    regime.get("is_overpriced"),
                )
        except Exception as e:
            logger.error(f"Failed to save regime: {e}")

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
        """Upsert current position snapshot."""
        if not self.pool:
            return
        try:
            async with self.pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO portfolio_positions (symbol, size, entry_price, mark_price, unrealized_pnl, delta, gamma, theta, last_update)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, CURRENT_TIMESTAMP)
                    ON CONFLICT (symbol) DO UPDATE SET
                        size = EXCLUDED.size,
                        entry_price = EXCLUDED.entry_price,
                        mark_price = EXCLUDED.mark_price,
                        unrealized_pnl = EXCLUDED.unrealized_pnl,
                        delta = EXCLUDED.delta,
                        gamma = EXCLUDED.gamma,
                        theta = EXCLUDED.theta,
                        last_update = CURRENT_TIMESTAMP
                    """,
                    position.symbol,
                    position.size,
                    position.entry_price,
                    0.0,  # mark_price placeholder
                    0.0,  # unrealized_pnl placeholder
                    0.0,  # delta placeholder
                    0.0,  # gamma placeholder
                    0.0,  # theta placeholder
                )
        except Exception as e:
            logger.error(f"Failed to save position: {e}")

    async def get_latest_positions(self) -> List[Position]:
        """Fetch all non-zero positions from the database."""
        if not self.pool:
            return []
        try:
            async with self.pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT symbol, size, entry_price, mark_price, unrealized_pnl, delta, gamma, theta
                    FROM portfolio_positions
                    WHERE size != 0
                """
                )
                return [
                    Position(
                        symbol=r["symbol"],
                        size=r["size"],
                        entry_price=r["entry_price"],
                        unrealized_pnl=r["unrealized_pnl"] or 0.0,
                    )
                    for r in rows
                ]
        except Exception as e:
            logger.error(f"Failed to fetch latest positions: {e}")
            return []

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
            "1m": timedelta(minutes=1),
            "5m": timedelta(minutes=5),
            "1h": timedelta(hours=1),
            "1d": timedelta(days=1),
        }
        bucket_interval = interval_map.get(resolution, timedelta(minutes=1))

        try:
            async with self.pool.acquire() as conn:
                # We use market_trades for OHLCV if available, or tickers if that's what we have.
                # Assuming we construct candles from trades for accuracy,
                # but if trades are sparse, we might use tickers (last price).
                # For this implementation, let's use market_trades for OHLCV.

                rows = await conn.fetch(
                    """
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
            return []

    async def get_regime_history(
        self, symbol: str, start: float, end: float
    ) -> List[Dict]:
        if not self.pool:
            return []
        try:
            async with self.pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT * FROM market_regimes
                    WHERE symbol = $1
                      AND time >= to_timestamp($2)
                      AND time <= to_timestamp($3)
                    ORDER BY time ASC
                    """,
                    symbol,
                    start,
                    end,
                )
                return [dict(r) for r in rows]
        except Exception as e:
            logger.error(f"Failed to fetch regime history: {e}")
            return []
