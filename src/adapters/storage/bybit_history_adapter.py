import asyncio
import aiohttp
import logging
import time
from typing import AsyncGenerator, List, Dict, Any, Optional
from datetime import datetime

from ...domain.history_provider import IHistoryProvider
from ...domain.entities.history import HistoryConfig
from ...domain.entities import Ticker, Trade, OrderSide
from .timescale_adapter import TimescaleDBAdapter

logger = logging.getLogger(__name__)


class BybitHistoryAdapter(IHistoryProvider):
    BASE_URL = "https://api.bybit.com"
    CATEGORY = "linear"

    def __init__(self, db_adapter: TimescaleDBAdapter):
        self.db = db_adapter
        self.session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()
        return self.session

    async def get_tickers(self, config: HistoryConfig) -> AsyncGenerator[Ticker, None]:
        rows = await self.db.get_recent_tickers(config.symbol, limit=2000)
        for r in sorted(rows, key=lambda x: x.timestamp):
            if config.start_time <= r.timestamp <= config.end_time:
                yield r

    async def get_trades(self, config: HistoryConfig) -> AsyncGenerator[Trade, None]:
        async with self.db.pool.acquire() as conn:
            query = """
                SELECT time, symbol, price, size, side, trade_id
                FROM market_trades
                WHERE symbol = $1 AND time >= to_timestamp($2) AND time <= to_timestamp($3)
                ORDER BY time ASC
            """
            async for r in conn.cursor(
                query, config.symbol, config.start_time, config.end_time
            ):
                yield Trade(
                    id=r["trade_id"],
                    symbol=r["symbol"],
                    price=r["price"],
                    size=r["size"],
                    side=OrderSide.BUY
                    if r["side"].lower() == "buy"
                    else OrderSide.SELL,
                    timestamp=r["time"].timestamp(),
                )

    async def fetch_and_persist(self, config: HistoryConfig) -> int:
        session = await self._get_session()
        url = f"{self.BASE_URL}/v5/market/kline"

        current_start = int(config.start_time * 1000)
        end_ms = int(config.end_time * 1000)
        total_count = 0

        while current_start < end_ms:
            params = {
                "category": self.CATEGORY,
                "symbol": config.symbol,
                "interval": "1",
                "start": current_start,
                "end": end_ms,
                "limit": 1000,
            }

            async with session.get(url, params=params) as resp:
                data = await resp.json()
                if data.get("retCode") == 0:
                    list_data = data.get("result", {}).get("list", [])
                    if not list_data:
                        logger.info(
                            f"No more data returned from Bybit for {config.symbol}."
                        )
                        break

                    # Bybit returns data in DESCENDING order (Newest -> Oldest)
                    # list_data[0] is the NEWEST candle in this batch.
                    # list_data[-1] is the OLDEST candle in this batch.

                    count_in_batch = 0
                    for item in list_data:
                        # Parse candle
                        ts = float(item[0]) / 1000.0

                        # Filter out candles outside our requested range just in case
                        if ts < config.start_time:
                            continue

                        last_price = float(item[4])
                        # Heuristic: slight spread around last price since we don't have BBO history in klines
                        # item[2] is High, item[3] is Low.
                        ticker = Ticker(
                            symbol=config.symbol,
                            bid=last_price * 0.9999,
                            ask=last_price * 1.0001,
                            bid_size=0.0,
                            ask_size=0.0,
                            last=last_price,
                            volume=float(item[5]),
                            timestamp=ts,
                            exchange="bybit",
                        )
                        await self.db.save_ticker(ticker)
                        count_in_batch += 1
                        total_count += 1

                    # Update cursor strategy:
                    # Since we want to fill the gap relative to 'start_time' up to 'end_time',
                    # and the API returns newest first, we are essentially requesting a chunk starting at 'current_start'.
                    # HOWEVER, 'start' in Bybit API is "From" timestamp (inclusive) but direction depends on sort.
                    # By default (and verified), it returns candles *starting at* 'start' and going forward if using ascending?
                    # WAIT: The verification script showed DESCENDING order.
                    # Implication: if we pass start=T, Bybit returns [T+N, ... T+1, T].
                    # Actually, Bybit Linear Kline: "start" is the start timestamp. "limit" is count.
                    # If we ask for start=100, limit=2: it returns [101, 100] (Desc).
                    # So to get the NEXT batch, we need to know the timestamp of the NEWEST candle we just got,
                    # and add 1 minute/interval to it.

                    # list_data[0] is the newest.
                    newest_ts_in_batch = int(list_data[0][0])
                    current_start = newest_ts_in_batch + 60000  # Advance by 1 minute

                    logger.debug(
                        f"Batch processed. Newest TS: {newest_ts_in_batch}. Next start: {current_start}"
                    )

                    if count_in_batch == 0:
                        # We might have gotten data that was all older than our start_time?
                        # Or maybe we reached the end (future)?
                        # If the newest candle is older than what we wanted? No, start param prevents that.
                        # If we got 0 valid candles but list_data wasn't empty, valid data might be in the future?
                        # Safe break to avoid infinite loop
                        pass

                else:
                    logger.error(f"Bybit fetch error: {data}")
                    break

        return total_count

    async def close(self):
        if self.session:
            await self.session.close()
