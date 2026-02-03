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
                        exchange TEXT NOT NULL DEFAULT 'thalex',
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
                        exchange TEXT NOT NULL DEFAULT 'thalex',
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
                        symbol TEXT NOT NULL,
                        exchange TEXT NOT NULL DEFAULT 'thalex',
                        size DOUBLE PRECISION NOT NULL,
                        entry_price DOUBLE PRECISION,
                        mark_price DOUBLE PRECISION,
                        unrealized_pnl DOUBLE PRECISION,
                        delta DOUBLE PRECISION,
                        gamma DOUBLE PRECISION,
                        theta DOUBLE PRECISION,
                        last_update TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                        PRIMARY KEY (symbol, exchange)
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
                # 5. Bot Executions Table (Private Fills)
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS bot_executions (
                        time TIMESTAMPTZ NOT NULL,
                        symbol TEXT NOT NULL,
                        exchange TEXT NOT NULL DEFAULT 'thalex',
                        side TEXT NOT NULL,
                        price DOUBLE PRECISION NOT NULL,
                        size DOUBLE PRECISION NOT NULL,
                        order_id TEXT,
                        fee DOUBLE PRECISION DEFAULT 0.0
                    );
                """)
                try:
                    await conn.execute(
                        "SELECT create_hypertable('bot_executions', 'time', if_not_exists => TRUE);"
                    )
                except Exception:
                    pass

                # MIGRATION: Add exchange column if missing (for existing tables)
                tables_to_migrate = [
                    "market_tickers",
                    "market_trades",
                    "portfolio_positions",
                ]
                for table in tables_to_migrate:
                    try:
                        await conn.execute(f"""
                            ALTER TABLE {table} 
                            ADD COLUMN IF NOT EXISTS exchange TEXT DEFAULT 'thalex';
                        """)
                    except Exception as e:
                        # Ignore if column exists or other non-critical error
                        logger.warning(f"Migration for {table} exchange column: {e}")

                # MIGRATION: Fix Portfolio Positions Primary Key (symbol -> symbol, exchange)
                try:
                    # Check if PK needs update
                    # We can try to drop the old one and add the new one.
                    # If it already exists as (symbol, exchange), drop might fail if name differs,
                    # or adding might fail.
                    # Safest is to try dropping the specific default name 'portfolio_positions_pkey'
                    # and recreating it.
                    await conn.execute("""
                        ALTER TABLE portfolio_positions DROP CONSTRAINT IF EXISTS portfolio_positions_pkey;
                        ALTER TABLE portfolio_positions ADD CONSTRAINT portfolio_positions_pkey PRIMARY KEY (symbol, exchange);
                    """)
                except Exception as e:
                    logger.warning(f"Migration for portfolio_positions PK: {e}")

                # MIGRATION: Add Unique ID to bot_executions (idempotency)
                try:
                    await conn.execute("""
                        ALTER TABLE bot_executions DROP CONSTRAINT IF EXISTS bot_executions_order_time_unique;
                        ALTER TABLE bot_executions ADD CONSTRAINT bot_executions_order_time_unique UNIQUE (order_id, time);
                    """)
                except Exception as e:
                    logger.warning(f"Migration for bot_executions UNIQUE: {e}")

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
                    INSERT INTO market_tickers (time, symbol, exchange, bid, ask, last, volume)
                    VALUES (to_timestamp($1), $2, $3, $4, $5, $6, $7)
                    """,
                    ticker.timestamp,
                    ticker.symbol,
                    ticker.exchange,
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
                    INSERT INTO market_trades (time, symbol, exchange, price, size, side, trade_id)
                    VALUES (to_timestamp($1), $2, $3, $4, $5, $6, $7)
                    """,
                    trade.timestamp,
                    trade.symbol,
                    trade.exchange,
                    trade.price,
                    trade.size,
                    trade.side.value
                    if hasattr(trade.side, "value")
                    else str(trade.side),
                    trade.id,
                )
        except Exception as e:
            logger.error(f"Failed to save trade: {e}")

    async def save_execution(self, trade: Trade):
        """Save a bot execution (fill) to the private table."""
        if not self.pool:
            return
        try:
            async with self.pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO bot_executions (time, symbol, exchange, side, price, size, order_id, fee)
                    VALUES (to_timestamp($1), $2, $3, $4, $5, $6, $7, $8)
                    ON CONFLICT (order_id, time) DO NOTHING
                    """,
                    trade.timestamp,
                    trade.symbol,
                    trade.exchange,
                    trade.side.value
                    if hasattr(trade.side, "value")
                    else str(trade.side),
                    trade.price,
                    trade.size,
                    trade.order_id,
                    trade.fee,
                )
                logger.info(f"Saved execution for order {trade.order_id}")
        except Exception as e:
            logger.error(f"Failed to save execution: {e}")

    async def save_position(self, position: Position):
        if not self.pool:
            return
        try:
            async with self.pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO portfolio_positions (symbol, exchange, size, entry_price, mark_price, unrealized_pnl, delta, gamma, theta, last_update)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, CURRENT_TIMESTAMP)
                    ON CONFLICT (symbol, exchange) DO UPDATE SET
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
                    position.exchange,
                    position.size,
                    position.entry_price,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
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

    async def get_executions(
        self, start: float, end: float, symbol: Optional[str] = None
    ) -> List[Dict]:
        """Fetch bot executions (private fills)."""
        if not self.pool:
            return []
        try:
            async with self.pool.acquire() as conn:
                query = """
                    SELECT time, symbol, side, price, size, order_id, fee
                    FROM bot_executions
                    WHERE time >= to_timestamp($1)
                      AND time <= to_timestamp($2)
                """
                args = [start, end]
                if symbol:
                    query += " AND symbol = $3"
                    args.append(symbol)

                query += " ORDER BY time DESC"

                rows = await conn.fetch(query, *args)
                return [
                    {
                        "timestamp": r["time"].timestamp(),
                        "symbol": r["symbol"],
                        "side": r["side"],
                        "price": r["price"],
                        "size": r["size"],
                        "order_id": r["order_id"],
                        "fee": r["fee"],
                    }
                    for r in rows
                ]
        except Exception as e:
            logger.error(f"Failed to fetch executions: {e}")
            return []

    async def get_history(
        self, symbol: str, start: float, end: float, resolution: str = "1m"
    ) -> List[Dict]:
        """
        Fetch OHLCV history for a symbol.
        Hybrid Strategy:
        1. Attempt to aggregate from 'market_trades' (Standard OHLCV).
        2. If result is empty, fallback to 'market_tickers' (Last Price Candles).
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
                # 1. Try Market Trades
                trade_rows = await conn.fetch(
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

                if trade_rows:
                    logger.info(
                        f"Fetched {len(trade_rows)} candles from market_trades for {symbol}"
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
                        for r in trade_rows
                    ]

                # 2. Fallback to Market Tickers
                logger.warning(
                    f"No trades found for {symbol} history. Falling back to market_tickers."
                )
                ticker_rows = await conn.fetch(
                    """
                    SELECT
                        time_bucket($1, time) AS bucket,
                        FIRST(last, time) as open,
                        MAX(last) as high,
                        MIN(last) as low,
                        LAST(last, time) as close,
                        SUM(volume) as volume
                    FROM market_tickers
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
                    for r in ticker_rows
                ]

        except Exception as e:
            logger.error(f"Failed to fetch OHLCV history: {e}")
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
