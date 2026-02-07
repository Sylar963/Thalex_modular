import asyncpg
import logging
from datetime import timedelta
from typing import List, Optional, Dict, Any
import json
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

                # Uniqueness for Idempotency (Time + Symbol + Exchange)
                try:
                    await conn.execute("""
                        CREATE UNIQUE INDEX IF NOT EXISTS market_tickers_time_symbol_exchange_idx 
                        ON market_tickers (time, symbol, exchange);
                    """)
                except Exception as e:
                    logger.warning(
                        f"Failed to create unique index on market_tickers: {e}"
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

                # MIGRATION: Market Trades Unique Constraint (required for ON CONFLICT in save_trade)
                try:
                    await conn.execute("""
                        CREATE UNIQUE INDEX IF NOT EXISTS market_trades_time_exchange_trade_id_idx 
                        ON market_trades (time, exchange, trade_id);
                    """)
                except Exception as e:
                    logger.warning(f"Migration for market_trades UNIQUE: {e}")

                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS market_signals (
                        time TIMESTAMPTZ NOT NULL,
                        symbol TEXT NOT NULL,
                        signal_type TEXT NOT NULL,
                        momentum DOUBLE PRECISION,
                        reversal DOUBLE PRECISION,
                        volatility DOUBLE PRECISION,
                        exhaustion DOUBLE PRECISION,
                        gamma_adjustment DOUBLE PRECISION,
                        reservation_price_offset DOUBLE PRECISION,
                        volatility_adjustment DOUBLE PRECISION,
                        vamp_value DOUBLE PRECISION,
                        market_impact DOUBLE PRECISION,
                        immediate_flow DOUBLE PRECISION,
                        orh DOUBLE PRECISION,
                        orl DOUBLE PRECISION,
                        orm DOUBLE PRECISION,
                        breakout_direction TEXT
                    );
                """)
                try:
                    await conn.execute(
                        "SELECT create_hypertable('market_signals', 'time', if_not_exists => TRUE);"
                    )
                except Exception:
                    pass

                # 7. Account Balances Table
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS account_balances (
                        exchange TEXT NOT NULL,
                        asset TEXT NOT NULL,
                        total DOUBLE PRECISION NOT NULL,
                        available DOUBLE PRECISION NOT NULL,
                        margin_used DOUBLE PRECISION DEFAULT 0.0,
                        equity DOUBLE PRECISION DEFAULT 0.0,
                        last_update TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                        PRIMARY KEY (exchange, asset)
                    );
                """)

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

    async def save_bot_status(self, status: Dict[str, Any]):
        if not self.pool:
            return
        try:
            async with self.pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO bot_status 
                    (time, symbol, exchange, risk_state, trend_state, execution_mode, active_signals, risk_breach, metadata)
                    VALUES (NOW(), $1, $2, $3, $4, $5, $6, $7, $8)
                    """,
                    status.get("symbol"),
                    status.get("exchange"),
                    status.get("risk_state", "UNKNOWN"),
                    status.get("trend_state", "FLAT"),
                    status.get("execution_mode", "NORMAL"),
                    str(status.get("active_signals", [])),
                    status.get("risk_breach", False),
                    json.dumps(status.get("metadata", {})),
                )
        except Exception as e:
            logger.error(f"Failed to save bot status: {e}")

    async def save_signal(self, symbol: str, signal_type: str, signals: Dict[str, Any]):
        if not self.pool:
            return
        try:
            async with self.pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO market_signals 
                    (time, symbol, signal_type, momentum, reversal, volatility, exhaustion, 
                     gamma_adjustment, reservation_price_offset, volatility_adjustment, 
                     vamp_value, market_impact, immediate_flow, orh, orl, orm, breakout_direction)
                    VALUES (NOW(), $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16)
                    """,
                    symbol,
                    signal_type,
                    signals.get("momentum"),
                    signals.get("reversal"),
                    signals.get("volatility"),
                    signals.get("exhaustion"),
                    signals.get("gamma_adjustment"),
                    signals.get("reservation_price_offset"),
                    signals.get("volatility_adjustment"),
                    signals.get("vamp_value"),
                    signals.get("market_impact"),
                    signals.get("immediate_flow"),
                    signals.get("orh"),
                    signals.get("orl"),
                    signals.get("orm"),
                    signals.get("breakout_direction"),
                )
        except Exception as e:
            logger.error(f"Failed to save signal: {e}")

    async def save_tickers_bulk(self, tickers: List[Tuple]):
        """
        Batch insert tickers.
        Expected tuple: (time, symbol, exchange, bid, ask, last, volume)
        """
        if not self.pool or not tickers:
            return
        try:
            async with self.pool.acquire() as conn:
                await conn.executemany(
                    """
                    INSERT INTO market_tickers (time, symbol, exchange, bid, ask, last, volume)
                    VALUES ($1, $2, $3, $4, $5, $6, $7)
                    ON CONFLICT DO NOTHING
                    """,
                    tickers,
                )
        except Exception as e:
            logger.error(f"Failed to save tickers bulk: {e}")

    async def save_options_metrics_bulk(self, metrics: List[Tuple]):
        """
        Batch insert options metrics.
        Expected tuple: (time, underlying, strike, expiry_date, days_to_expiry, call_mark_price, put_mark_price, straddle_price, implied_vol, expected_move_pct)
        """
        if not self.pool or not metrics:
            return
        try:
            async with self.pool.acquire() as conn:
                await conn.executemany(
                    """
                    INSERT INTO options_live_metrics 
                    (time, underlying, strike, expiry_date, days_to_expiry, call_mark_price, put_mark_price, straddle_price, implied_vol, expected_move_pct)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                    ON CONFLICT DO NOTHING
                    """,
                    metrics,
                )
        except Exception as e:
            logger.error(f"Failed to save options metrics bulk: {e}")

    async def get_signal_history(
        self, symbol: str, start: float, end: float, signal_type: Optional[str] = None
    ) -> List[Dict]:
        if not self.pool:
            return []
        try:
            async with self.pool.acquire() as conn:
                query = """
                    SELECT * FROM market_signals
                    WHERE symbol = $1
                      AND time >= to_timestamp($2)
                      AND time <= to_timestamp($3)
                """
                args = [symbol, start, end]
                if signal_type:
                    query += " AND signal_type = $4"
                    args.append(signal_type)

                query += " ORDER BY time ASC"

                rows = await conn.fetch(query, *args)
                return [dict(r) for r in rows]
        except Exception as e:
            logger.error(f"Failed to fetch signal history: {e}")
            return []

    async def save_ticker(self, ticker: Ticker):
        if not self.pool:
            return

    async def save_ticker(self, ticker: Ticker):
        if not self.pool:
            return
        try:
            async with self.pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO market_tickers (time, symbol, exchange, bid, ask, last, volume)
                    VALUES (to_timestamp($1), $2, $3, $4, $5, $6, $7)
                    ON CONFLICT (time, symbol, exchange) DO UPDATE SET
                        bid = EXCLUDED.bid,
                        ask = EXCLUDED.ask,
                        last = EXCLUDED.last,
                        volume = EXCLUDED.volume
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
                    ON CONFLICT (time, exchange, trade_id) DO NOTHING
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
                    position.mark_price,
                    position.unrealized_pnl,
                    position.delta,
                    position.gamma,
                    position.theta,
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
                    SELECT symbol, size, entry_price, mark_price, unrealized_pnl, delta, gamma, theta, exchange
                    FROM portfolio_positions
                    WHERE size != 0
                """
                )
                return [
                    Position(
                        symbol=r["symbol"],
                        size=r["size"],
                        entry_price=r["entry_price"],
                        mark_price=r["mark_price"] or 0.0,
                        unrealized_pnl=r["unrealized_pnl"] or 0.0,
                        delta=r["delta"] or 0.0,
                        gamma=r["gamma"] or 0.0,
                        theta=r["theta"] or 0.0,
                        exchange=r["exchange"] or "thalex",
                    )
                    for r in rows
                ]
        except Exception as e:
            logger.error(f"Failed to fetch latest positions: {e}")
            return []

    async def get_recent_tickers(
        self, symbol: str, limit: int = 100, exchange: Optional[str] = None
    ) -> List[Ticker]:
        if not self.pool:
            return []
        try:
            async with self.pool.acquire() as conn:
                query = """
                    SELECT time, symbol, bid, ask, last, volume, exchange
                    FROM market_tickers
                    WHERE symbol = $1
                """
                args = [symbol]

                if exchange:
                    query += " AND exchange = $2"
                    args.append(exchange)

                query += """
                    ORDER BY time DESC
                    LIMIT $""" + str(len(args) + 1)

                args.append(limit)

                rows = await conn.fetch(query, *args)
                return [
                    Ticker(
                        symbol=r["symbol"],
                        bid=r["bid"],
                        ask=r["ask"],
                        bid_size=0.0,
                        ask_size=0.0,
                        last=r["last"],
                        volume=r["volume"],
                        exchange=r["exchange"],
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
        self,
        symbol: str,
        start: float,
        end: float,
        resolution: str = "1m",
        exchange: str = "thalex",
    ) -> List[Dict]:
        """
        Fetch OHLCV history for a symbol with Universal Aggregation.
        - Maps symbol to all venue aliases (e.g. BTCUSDT, BTC-PERP).
        - Fetches from BOTH market_trades (Recent) and market_tickers (Backfill).
        - Aggregates for 'all', Unions for specific venue.
        """
        if not self.pool:
            return []

        # 1. Interval mapping
        interval_map = {
            "1m": timedelta(minutes=1),
            "5m": timedelta(minutes=5),
            "1h": timedelta(hours=1),
            "1d": timedelta(days=1),
        }
        bucket_interval = interval_map.get(resolution, timedelta(minutes=1))

        # 2. Universal Symbol Mapping
        symbols = [symbol]
        is_all = exchange and exchange.lower() == "all"
        is_btc = "BTC" in symbol.upper()

        if is_all or is_btc:
            # Expand to all known aliases to ensure we catch data from all venues
            symbols = list(set([symbol, "BTC-PERPETUAL", "BTCUSDT", "BTC-PERP"]))
            if is_all:
                logger.info(f"Aggregating history for ALL venues: {symbols}")

        # Core Query Builder for Candles
        async def fetch_candles(table_name: str, price_col: str, vol_col: str):
            placeholders = ",".join([f"${i + 4}" for i in range(len(symbols))])
            query = f"""
                SELECT
                    time_bucket($1, time) AS bucket,
                    FIRST({price_col}, time) as open,
                    MAX({price_col}) as high,
                    MIN({price_col}) as low,
                    LAST({price_col}, time) as close,
                    SUM({vol_col}) as volume
                FROM {table_name}
                WHERE symbol IN ({placeholders})
                  AND time >= to_timestamp($2)
                  AND time <= to_timestamp($3)
            """
            args = [bucket_interval, start, end] + symbols

            # Filter by exchange if not ALL
            # Note: For market_tickers, we are lenient because legacy data might have NULL/Empty exchange
            if not is_all and table_name == "market_trades":
                idx = len(args) + 1
                query += f" AND LOWER(exchange) = LOWER(${idx})"
                args.append(exchange.lower())

            query += " GROUP BY bucket ORDER BY bucket ASC"

            async with self.pool.acquire() as conn:
                return await conn.fetch(query, *args)

        try:
            # 4. Fetch from both sources
            trade_rows = await fetch_candles("market_trades", "price", "size")
            ticker_rows = await fetch_candles("market_tickers", "last", "volume")

            # 5. Merge Logic
            merged = {}

            def process_rows(rows, source_name):
                for r in rows:
                    t = r["bucket"].timestamp()
                    data = {
                        "time": t,
                        "open": float(r["open"]),
                        "high": float(r["high"]),
                        "low": float(r["low"]),
                        "close": float(r["close"]),
                        "volume": float(r["volume"] or 0),
                    }

                    if t not in merged:
                        merged[t] = data
                    else:
                        existing = merged[t]
                        if is_all:
                            # Aggregation Logic: SUM volumes, MAX High, MIN Low
                            existing["volume"] += data["volume"]
                            existing["high"] = max(existing["high"], data["high"])
                            existing["low"] = min(existing["low"], data["low"])
                            # For Open/Close in 'all', we take the one with higher volume as a proxy
                            if data["volume"] > existing["volume"] * 0.5:
                                existing["open"] = data["open"]
                                existing["close"] = data["close"]
                        else:
                            # Specific Venue Logic: Prefer Ticks (trade_rows) over Snapshots (ticker_rows)
                            if source_name == "trades":
                                merged[t] = data

            process_rows(ticker_rows, "tickers")
            process_rows(trade_rows, "trades")

            # 6. Sort and Return
            final_history = sorted(merged.values(), key=lambda x: x["time"])

            if not final_history:
                logger.warning(
                    f"No history found for {symbol} ({exchange}) from {start} to {end}"
                )

            return final_history

        except Exception as e:
            logger.error(f"Failed standard history fetch for {symbol}: {e}")
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

    async def get_volume_bars(
        self,
        symbol: str,
        volume_threshold: float = 0.1,
        limit: int = 100,
        exchange: str = "thalex",
    ) -> List[Dict]:
        if not self.pool:
            return []
        try:
            async with self.pool.acquire() as conn:
                # Dynamic Query Construction
                symbols = [symbol]
                is_all = exchange and exchange.lower() == "all"
                is_bybit = exchange and exchange.lower() == "bybit"

                if is_all or is_bybit:
                    if "BTC" in symbol:
                        symbols = ["BTC-PERPETUAL", "BTCUSDT", "BTC-PERP"]

                placeholders = ",".join([f"${i + 3}" for i in range(len(symbols))])

                # Part 1: CTE
                base_query = f"""
                    WITH trades_numbered AS (
                        SELECT 
                            time,
                            price,
                            size,
                            side,
                            SUM(size) OVER (ORDER BY time) as cumulative_volume
                        FROM market_trades
                        WHERE symbol IN ({placeholders})
                """

                idx_exchange = len(symbols) + 3

                if not is_all:
                    base_query += f" AND LOWER(exchange) = LOWER(${idx_exchange})"

                base_query += """
                        ORDER BY time DESC
                        LIMIT 50000
                    ),
                    trades_with_bar_id AS (
                        SELECT *,
                            FLOOR(cumulative_volume / $1) as bar_id
                        FROM trades_numbered
                    )
                    SELECT 
                        bar_id,
                        FIRST(price, time) as open,
                        MAX(price) as high,
                        MIN(price) as low,
                        LAST(price, time) as close,
                        SUM(size) as volume,
                        COUNT(*) as trade_count,
                        SUM(CASE WHEN side = 'buy' THEN size ELSE 0 END) as buy_volume,
                        SUM(CASE WHEN side = 'sell' THEN size ELSE 0 END) as sell_volume,
                        MIN(time) as start_time,
                        MAX(time) as end_time
                    FROM trades_with_bar_id
                    GROUP BY bar_id
                    ORDER BY bar_id DESC
                    LIMIT $2
                """

                # Construct arguments list
                # Query expects: $1 (vol_thresh), $2 (limit), $3..$N (symbols), $N+1 (exchange optional)
                query_args = [volume_threshold, limit] + symbols
                if not is_all:
                    query_args.append(exchange)

                rows = await conn.fetch(base_query, *query_args)
                return [
                    {
                        "time": r["end_time"].timestamp(),
                        "open": r["open"],
                        "high": r["high"],
                        "low": r["low"],
                        "close": r["close"],
                        "volume": float(r["volume"] or 0.0),
                        "trade_count": r["trade_count"],
                        "buy_volume": float(r["buy_volume"] or 0.0),
                        "sell_volume": float(r["sell_volume"] or 0.0),
                        "delta": (float(r["buy_volume"] or 0.0))
                        - (float(r["sell_volume"] or 0.0)),
                    }
                    for r in rows
                ]
        except Exception as e:
            logger.error(f"Failed to fetch volume bars: {e}")
            return []

    async def get_tick_bars(
        self,
        symbol: str,
        tick_count: int = 2500,
        limit: int = 100,
        exchange: str = "thalex",
    ) -> List[Dict]:
        if not self.pool:
            return []
        try:
            async with self.pool.acquire() as conn:
                # Dynamic Query Construction
                symbols = [symbol]
                is_all = exchange and exchange.lower() == "all"
                is_bybit = exchange and exchange.lower() == "bybit"

                if is_all or is_bybit:
                    if "BTC" in symbol:
                        symbols = ["BTC-PERPETUAL", "BTCUSDT", "BTC-PERP"]

                placeholders = ",".join([f"${i + 3}" for i in range(len(symbols))])

                # Part 1: CTE
                base_query = f"""
                    WITH trades_numbered AS (
                        SELECT 
                            time,
                            price,
                            size,
                            side,
                            ROW_NUMBER() OVER (ORDER BY time) as row_num
                        FROM market_trades
                        WHERE symbol IN ({placeholders})
                """

                idx_exchange = len(symbols) + 3

                if not is_all:
                    base_query += f" AND LOWER(exchange) = LOWER(${idx_exchange})"

                base_query += """
                        ORDER BY time DESC
                        LIMIT 250000
                    ),
                    trades_with_bar_id AS (
                        SELECT *,
                            FLOOR((row_num - 1) / $1) as bar_id
                        FROM trades_numbered
                    )
                    SELECT 
                        bar_id,
                        FIRST(price, time) as open,
                        MAX(price) as high,
                        MIN(price) as low,
                        LAST(price, time) as close,
                        SUM(size) as volume,
                        COUNT(*) as trade_count,
                        SUM(CASE WHEN side = 'buy' THEN size ELSE 0 END) as buy_volume,
                        SUM(CASE WHEN side = 'sell' THEN size ELSE 0 END) as sell_volume,
                        MIN(time) as start_time,
                        MAX(time) as end_time
                    FROM trades_with_bar_id
                    GROUP BY bar_id
                    ORDER BY bar_id DESC
                    LIMIT $2
                """

                # Construct arguments list
                # Query expects: $1 (tick_count), $2 (limit), $3..$N (symbols), $N+1 (exchange optional)
                query_args = [tick_count, limit] + symbols
                if not is_all:
                    query_args.append(exchange)

                rows = await conn.fetch(base_query, *query_args)
                return [
                    {
                        "time": r["end_time"].timestamp(),
                        "open": r["open"],
                        "high": r["high"],
                        "low": r["low"],
                        "close": r["close"],
                        "volume": float(r["volume"] or 0.0),
                        "trade_count": r["trade_count"],
                        "buy_volume": float(r["buy_volume"] or 0.0),
                        "sell_volume": float(r["sell_volume"] or 0.0),
                        "delta": (float(r["buy_volume"] or 0.0))
                        - (float(r["sell_volume"] or 0.0)),
                    }
                    for r in rows
                ]
        except Exception as e:
            logger.error(f"Failed to fetch tick bars: {e}")
            return []

    async def _init_balances_table(self, conn):
        """Create account balances table."""
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS account_balances (
                exchange TEXT NOT NULL,
                asset TEXT NOT NULL,
                total DOUBLE PRECISION NOT NULL,
                available DOUBLE PRECISION NOT NULL,
                margin_used DOUBLE PRECISION DEFAULT 0.0,
                equity DOUBLE PRECISION DEFAULT 0.0,
                last_update TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (exchange, asset)
            );
        """)

    async def save_balance(self, balance):
        if not self.pool:
            return
        try:
            async with self.pool.acquire() as conn:
                # Ensure table exists (lazy init or we could do it in _init_schema)
                # Ideally _init_schema handles creation, but for this hot-patch we can add it here or rely on _init_schema update
                # Let's add it to _init_schema properly, but here is the save logic.
                await conn.execute(
                    """
                    INSERT INTO account_balances (exchange, asset, total, available, margin_used, equity, last_update)
                    VALUES ($1, $2, $3, $4, $5, $6, CURRENT_TIMESTAMP)
                    ON CONFLICT (exchange, asset) DO UPDATE SET
                        total = EXCLUDED.total,
                        available = EXCLUDED.available,
                        margin_used = EXCLUDED.margin_used,
                        equity = EXCLUDED.equity,
                        last_update = CURRENT_TIMESTAMP
                    """,
                    balance.exchange,
                    balance.asset,
                    balance.total,
                    balance.available,
                    balance.margin_used,
                    balance.equity,
                )
        except Exception as e:
            logger.error(f"Failed to save balance: {e}")

    async def get_latest_balances(self) -> List[Any]:
        if not self.pool:
            return []
        try:
            from ...domain.entities import (
                Balance,
            )  # Deferred import to avoid circular dependency

            async with self.pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT exchange, asset, total, available, margin_used, equity
                    FROM account_balances
                    """
                )
                return [
                    Balance(
                        exchange=r["exchange"],
                        asset=r["asset"],
                        total=r["total"],
                        available=r["available"],
                        margin_used=r["margin_used"],
                        equity=r["equity"],
                    )
                    for r in rows
                ]
        except Exception as e:
            logger.error(f"Failed to fetch latest balances: {e}")
            return []
