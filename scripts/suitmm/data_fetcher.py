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

    async def _get_db_connection(self):
        try:
            return await asyncpg.connect(
                "postgresql://postgres:password@localhost:5432/thalex_trading"
            )
        except Exception as e:
            print(f"Failed to connect to primary port 5432: {e}")
            # Fallback for legacy setups
            return await asyncpg.connect(
                "postgresql://postgres:password@localhost:5433/thalex_trading"
            )

    async def fetch_fills_from_db(self, symbol: str) -> List[Dict]:
        try:
            conn = await self._get_db_connection()
            # Try to fetch fills. Handle cases where table might use 'symbol' or 'instrument_id'
            # Assuming 'symbol' based on previous context.
            rows = await conn.fetch(
                """
                SELECT time, side, price, size, fee 
                FROM bot_executions 
                WHERE symbol=$1 
                ORDER BY time ASC
                """,
                symbol,
            )
            await conn.close()
            return [dict(r) for r in rows]
        except Exception as e:
            print(f"Error fetching DB fills: {e}")
            return []

    async def fetch_indicators(
        self, symbol: str, days: int = 14
    ) -> Dict[str, List[Dict]]:
        """
        Fetches historical indicator data from TimescaleDB tables.
        Returns a dictionary with lists of records for signals, regimes, and hft metrics.
        """
        try:
            conn = await self._get_db_connection()

            # Helper to fetch and convert to dict
            async def fetch_table(table, columns):
                query = f"""
                    SELECT time, {", ".join(columns)}
                    FROM {table}
                    WHERE symbol=$1 AND time > NOW() - INTERVAL '{days} days'
                    ORDER BY time ASC
                """
                try:
                    rows = await conn.fetch(query, symbol)
                    return [dict(r) for r in rows]
                except Exception as e:
                    print(f"  Warning: Could not fetch from {table}: {e}")
                    return []

            print(f"  Fetching indicators for {symbol}...")

            # 1. Market Signals
            signals = await fetch_table(
                "market_signals",
                [
                    "momentum",
                    "reversal",
                    "volatility",
                    "vamp_value",
                    "market_impact",
                    "immediate_flow",
                    "orh",
                    "orl",
                    "orm",
                ],
            )

            # 2. Market Regimes
            regimes = await fetch_table(
                "market_regimes",
                [
                    "trend_fast",
                    "trend_mid",
                    "trend_slow",
                    "rv_fast",
                    "rv_mid",
                    "rv_slow",
                    "liquidity_score",
                    "expected_move_pct",
                    "atm_iv",
                ],
            )

            # 3. HFT Signals
            hft = await fetch_table(
                "hft_signals", ["toxicity_score", "pull_rate", "quote_stability"]
            )

            await conn.close()
            return {"signals": signals, "regimes": regimes, "hft": hft}

        except Exception as e:
            print(f"Error fetching indicators: {e}")
            return {"signals": [], "regimes": [], "hft": []}
