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
                        break

                    for item in list_data:
                        ts = float(item[0]) / 1000.0
                        ticker = Ticker(
                            symbol=config.symbol,
                            bid=float(item[2]),
                            ask=float(item[3]),
                            bid_size=0.0,
                            ask_size=0.0,
                            last=float(item[4]),
                            volume=float(item[5]),
                            timestamp=ts,
                            exchange="bybit",
                        )
                        await self.db.save_ticker(ticker)
                        total_count += 1

                    last_ts = int(list_data[-1][0])
                    current_start = last_ts + 60000
                else:
                    logger.error(f"Bybit fetch error: {data}")
                    break

        return total_count

    async def close(self):
        if self.session:
            await self.session.close()
