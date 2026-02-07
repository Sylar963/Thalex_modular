from typing import AsyncGenerator, List, Dict, Any, Optional
import logging
from datetime import datetime, timezone
import asyncpg

from ...domain.history_provider import IHistoryProvider
from ...domain.entities.history import HistoryConfig
from ...domain.entities import Ticker, Trade, OrderSide

logger = logging.getLogger(__name__)


class TimescaleHistoryProvider(IHistoryProvider):
    """
    A generic, read-only history provider that fetches market data from TimescaleDB.
    It is venue-agnostic and relies on the 'exchange' column in the database.
    """

    def __init__(self, db_adapter):
        """
        Initialize with a TimescaleDBAdapter instance.

        Args:
            db_adapter: An instance of TimescaleDBAdapter (must expose .pool or ._get_connection)
        """
        self.db = db_adapter

    async def get_tickers(self, config: HistoryConfig) -> AsyncGenerator[Ticker, None]:
        """
        Stream tickers from the database.
        Fallback: If no tickers found in 'market_tickers', stream 'market_trades' and synthesize BBO.
        """
        query_tickers = """
            SELECT time, symbol, bid, ask, last, volume, exchange
            FROM market_tickers
            WHERE symbol = $1
              AND exchange = $2
              AND time >= to_timestamp($3)
              AND time <= to_timestamp($4)
            ORDER BY time ASC
        """

        ticker_found = False

        async with self.db.pool.acquire() as conn:
            async with conn.transaction():
                async for record in conn.cursor(
                    query_tickers,
                    config.symbol,
                    config.venue.lower(),
                    config.start_time,
                    config.end_time,
                ):
                    ticker_found = True
                    yield self._record_to_ticker(record)

        if not ticker_found:
            logger.warning(
                f"No BBO (tickers) found for {config.symbol}. Synthesizing from Trades..."
            )

            query_trades = """
                SELECT time, symbol, price, size, side, exchange, trade_id
                FROM market_trades
                WHERE symbol = $1
                  AND exchange = $2
                  AND time >= to_timestamp($3)
                  AND time <= to_timestamp($4)
                ORDER BY time ASC
            """

            async with self.db.pool.acquire() as conn:
                async with conn.transaction():
                    async for record in conn.cursor(
                        query_trades,
                        config.symbol,
                        config.venue.lower(),
                        config.start_time,
                        config.end_time,
                    ):
                        yield self._trade_to_ticker(record)

    async def get_trades(self, config: HistoryConfig) -> AsyncGenerator[Trade, None]:
        """
        Stream trades from the database for the specified config.
        """
        query = """
            SELECT time, symbol, price, size, side, exchange, trade_id
            FROM market_trades
            WHERE symbol = $1
              AND exchange = $2
              AND time >= to_timestamp($3)
              AND time <= to_timestamp($4)
            ORDER BY time ASC
        """

        async with self.db.pool.acquire() as conn:
            async with conn.transaction():
                async for record in conn.cursor(
                    query,
                    config.symbol,
                    config.venue.lower(),
                    config.start_time,
                    config.end_time,
                ):
                    yield self._record_to_trade(record)

    async def fetch_and_persist(self, config: HistoryConfig) -> int:
        """
        No-op for this provider as it reads existing data.
        In the future, this could trigger a backfill job if data is missing.
        """
        logger.info(
            f"TimescaleHistoryProvider: Using existing data for {config.symbol} on {config.venue}"
        )
        return 0

    def _record_to_ticker(self, record: Any) -> Ticker:
        """Convert a DB record to a domain Ticker object."""
        ts = record["time"].replace(tzinfo=timezone.utc).timestamp()

        return Ticker(
            symbol=record["symbol"],
            bid=float(record["bid"]),
            ask=float(record["ask"]),
            bid_size=0.0,
            ask_size=0.0,
            last=float(record["last"]),
            volume=float(record["volume"]) if record["volume"] is not None else 0.0,
            timestamp=ts,
            exchange=record["exchange"],
        )

    def _trade_to_ticker(self, record: Any) -> Ticker:
        """Synthesize a Ticker from a Trade record (Mid = Price, Spread=0)."""
        ts = record["time"].replace(tzinfo=timezone.utc).timestamp()
        price = float(record["price"])

        return Ticker(
            symbol=record["symbol"],
            bid=price,
            ask=price,
            bid_size=0.0,
            ask_size=0.0,
            last=price,
            volume=float(record["size"]),
            timestamp=ts,
            exchange=record["exchange"],
        )

    def _record_to_trade(self, record: Any) -> Trade:
        """Convert a DB record to a domain Trade object."""
        ts = record["time"].replace(tzinfo=timezone.utc).timestamp()

        side_str = record["side"].upper()
        try:
            side = OrderSide(side_str)
        except ValueError:
            side = OrderSide.BUY if "BUY" in side_str else OrderSide.SELL

        return Trade(
            id=str(record["trade_id"]),
            order_id="",
            symbol=record["symbol"],
            price=float(record["price"]),
            size=float(record["size"]),
            side=side,
            timestamp=ts,
            exchange=record["exchange"],
        )
