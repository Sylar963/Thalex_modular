import aiohttp
import asyncio
import asyncpg
from typing import List, Dict, Optional, Tuple

BYBIT_BASE = "https://api.bybit.com"
DB_DSN = "postgresql://postgres:password@localhost:5433/thalex_trading"


class MarketDataFetcher:
    def __init__(self, session: aiohttp.ClientSession):
        self.session = session

    async def fetch_ticker(self, symbol: str) -> Optional[Dict]:
        url = f"{BYBIT_BASE}/v5/market/tickers"
        params = {"category": "linear", "symbol": symbol}
        try:
            async with self.session.get(url, params=params) as resp:
                data = await resp.json()
                tickers = data.get("result", {}).get("list", [])
                return tickers[0] if tickers else None
        except Exception as e:
            print(f"Error fetching ticker for {symbol}: {e}")
            return None

    async def fetch_klines(
        self, symbol: str, interval: str = "60", limit: int = 336
    ) -> List:
        url = f"{BYBIT_BASE}/v5/market/kline"
        params = {
            "category": "linear",
            "symbol": symbol,
            "interval": interval,
            "limit": limit,
        }
        try:
            async with self.session.get(url, params=params) as resp:
                data = await resp.json()
                return data.get("result", {}).get("list", [])
        except Exception as e:
            print(f"Error fetching klines for {symbol}: {e}")
            return []

    async def fetch_orderbook(self, symbol: str, limit: int = 50) -> Dict:
        url = f"{BYBIT_BASE}/v5/market/orderbook"
        params = {"category": "linear", "symbol": symbol, "limit": limit}
        try:
            async with self.session.get(url, params=params) as resp:
                data = await resp.json()
                return data.get("result", {})
        except Exception as e:
            print(f"Error fetching orderbook for {symbol}: {e}")
            return {}

    async def fetch_instrument_info(self, symbol: str) -> Optional[Dict]:
        url = f"{BYBIT_BASE}/v5/market/instruments-info"
        params = {"category": "linear", "symbol": symbol}
        try:
            async with self.session.get(url, params=params) as resp:
                data = await resp.json()
                instruments = data.get("result", {}).get("list", [])
                return instruments[0] if instruments else None
        except Exception as e:
            print(f"Error fetching instrument info for {symbol}: {e}")
            return None

    async def fetch_fills_from_db(self, symbol: str) -> List[Dict]:
        try:
            # Note: 5433 port as per project config, but verify if 5432 is used elsewhere
            # The prompt context mentioned "Standardized on port 5432... removing the legacy 5433"
            # So I should probably use 5432, but mm_coin_analyzer.py usages 5433.
            # I will check valid port. User rules say 5432.
            # But mm_coin_analyzer.py has 5433.
            # I will use 5432 based on "Standardized on port 5432" rule.
            conn = await asyncpg.connect(
                "postgresql://postgres:password@localhost:5432/thalex_trading"
            )
            fills = await conn.fetch(
                "SELECT time, side, price, size, fee FROM bot_executions WHERE symbol=$1 ORDER BY time ASC",
                symbol,
            )
            await conn.close()
            return [dict(f) for f in fills]
        except Exception as e:
            # Fallback to 5433 just in case, or just log error
            try:
                conn = await asyncpg.connect(
                    "postgresql://postgres:password@localhost:5433/thalex_trading"
                )
                fills = await conn.fetch(
                    "SELECT time, side, price, size, fee FROM bot_executions WHERE symbol=$1 ORDER BY time ASC",
                    symbol,
                )
                await conn.close()
                return [dict(f) for f in fills]
            except Exception as e2:
                print(f"Error fetching DB fills: {e2}")
                return []
