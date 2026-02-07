import asyncio
import logging
import os
import aiohttp

try:
    import orjson
except ImportError:
    import json as orjson
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Tuple, Optional

# Imports work natively now via pip install -e .

# Import Adapter
try:
    from src.adapters.storage.timescale_adapter import TimescaleDBAdapter
except ImportError:
    # If run from root
    from src.adapters.storage.timescale_adapter import TimescaleDBAdapter

# Configuration from ENV
DB_HOST = os.getenv("DATABASE_HOST", "localhost")
DB_NAME = os.getenv("DATABASE_NAME", "thalex_trading")
DB_USER = os.getenv("DATABASE_USER", "postgres")
DB_PASS = os.getenv("DATABASE_PASSWORD", "password")
DB_PORT = os.getenv("DATABASE_PORT", "5433")

DB_DSN = f"postgres://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

THALEX_API_URL = "https://thalex.com/api/v2"

# Logger Setup
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("HistoricalLoader")


class HistoricalOptionsLoader:
    def __init__(self):
        self.db = TimescaleDBAdapter(DB_DSN)
        self.session = None

    async def connect(self):
        await self.db.connect()

    async def disconnect(self):
        await self.db.disconnect()

    async def _fetch_list(self, endpoint: str, params: Dict) -> List:
        url = f"{THALEX_API_URL}/{endpoint}"
        async with self.session.get(url, params=params) as resp:
            if resp.status != 200:
                logger.error(f"API Error {resp.status} for {url}: {await resp.text()}")
                return []
            data = await resp.json(loads=orjson.loads)
            result = data.get("result", [])

            # If result is a dict, attempt to find the first list within it (e.g. 'index', 'mark')
            if isinstance(result, dict):
                for val in result.values():
                    if isinstance(val, list):
                        return val
            return result if isinstance(result, list) else []

    async def get_index_history(
        self, index_name: str, start_ts: int, end_ts: int
    ) -> List:
        """Fetch index price history."""
        params = {
            "index_name": index_name,
            "from": start_ts,
            "to": end_ts,
            "resolution": "1m",
        }
        return await self._fetch_list("public/index_price_historical_data", params)

    async def get_mark_history(
        self, instrument_name: str, start_ts: int, end_ts: int
    ) -> List:
        """Fetch mark price history for an instrument."""
        params = {
            "instrument_name": instrument_name,
            "from": start_ts,
            "to": end_ts,
            "resolution": "1m",
        }
        return await self._fetch_list("public/mark_price_historical_data", params)

    async def get_perpetual_history(self, start_ts: int, end_ts: int) -> List:
        """Fetch historical data for BTC-PERPETUAL."""
        return await self._fetch_list(
            "public/mark_price_historical_data",
            {
                "instrument_name": "BTC-PERPETUAL",
                "from": start_ts,
                "to": end_ts,
                "resolution": "1m",
            },
        )

    def _generate_expirations(
        self, start_date: datetime, end_date: datetime
    ) -> List[datetime]:
        """
        Generates likely expiration dates (Fridays) between start and end.
        """
        expirations = []
        current = start_date
        while current <= end_date:
            # Find next Friday
            days_ahead = 4 - current.weekday()
            if days_ahead <= 0:
                days_ahead += 7
            next_friday = current + timedelta(days=days_ahead)
            if next_friday > end_date:
                break
            expirations.append(next_friday)
            current = next_friday
        return expirations

    def _format_date_thalex(self, date: datetime) -> str:
        """Formats date as DDMMMYY (e.g. 29MAR24)"""
        return date.strftime("%d%b%y").upper()

    async def run(self, days_back: int = 30):
        # Establish Connections
        await self.connect()
        self.session = aiohttp.ClientSession()

        try:
            now = datetime.now(timezone.utc)
            start_history = now - timedelta(days=days_back)

            # Chunking loop: process 1 day at a time
            chunk_size = timedelta(days=1)
            current_start = start_history

            while current_start < now:
                current_end = min(current_start + chunk_size, now)

                start_ts = int(current_start.timestamp())
                end_ts = int(current_end.timestamp())

                logger.info(f"Processing chunk: {current_start} to {current_end}")

                # 1. Fetch Index History (for Options metrics)
                index_history = await self.get_index_history("BTCUSD", start_ts, end_ts)

                # 2. Fetch Perpetual History (for direct insertion)
                perp_history = await self.get_perpetual_history(start_ts, end_ts)

                if not index_history and not perp_history:
                    logger.warning(f"No data for chunk {current_start}. Skipping.")
                    current_start += chunk_size
                    continue

                likely_expiries = self._generate_expirations(
                    current_start, current_end + timedelta(days=60)
                )

                # 3. Identify unique pairs needed for Option Metrics
                unique_pairs = set()
                index_map = {}  # ts -> price

                for point in index_history:
                    if isinstance(point, list):
                        ts = point[0]
                        price = float(point[4])
                    else:
                        ts = point.get("time")
                        price = float(point.get("close", 0))

                    if not ts or price <= 0:
                        continue

                    index_map[ts] = price

                    current_dt = datetime.fromtimestamp(ts, tz=timezone.utc)
                    target_date = current_dt + timedelta(days=30)
                    valid_expiries = [e for e in likely_expiries if e > current_dt]
                    if not valid_expiries:
                        continue
                    best_expiry = min(
                        valid_expiries, key=lambda x: abs((x - target_date).days)
                    )
                    strike = round(price / 1000) * 1000
                    unique_pairs.add((best_expiry, strike))

                logger.info(f"Chunk needs {len(unique_pairs)} instrument pairs.")

                # 4. Fetch Instrument History concurrently
                instrument_data = {}

                tasks = []
                for expiry, strike in unique_pairs:
                    exp_str = self._format_date_thalex(expiry)
                    for kind in ["C", "P"]:
                        instr_name = f"BTC-{exp_str}-{strike}-{kind}"
                        tasks.append(
                            (
                                instr_name,
                                self.get_mark_history(instr_name, start_ts, end_ts),
                            )
                        )

                # Limit concurrency
                semaphore = asyncio.Semaphore(10)

                async def fetch_safe(name, coro):
                    async with semaphore:
                        return name, await coro

                results = await asyncio.gather(*(fetch_safe(n, c) for n, c in tasks))

                for name, hist in results:
                    if hist:
                        # Convert to dict for fast lookup: ts -> list data
                        data_map = (
                            {h[0]: h for h in hist}
                            if isinstance(hist[0], list)
                            else {h["time"]: h for h in hist}
                        )
                        instrument_data[name] = data_map

                # 5. Prepare Batch Inserts
                metrics_to_insert = []
                tickers_to_insert = []

                # A. Options Metrics
                for ts, price in index_map.items():
                    current_dt = datetime.fromtimestamp(ts, tz=timezone.utc)

                    # Logic match
                    valid_expiries = [e for e in likely_expiries if e > current_dt]
                    if not valid_expiries:
                        continue
                    best_expiry = min(
                        valid_expiries,
                        key=lambda x: abs((x - (current_dt + timedelta(days=30))).days),
                    )
                    strike = round(price / 1000) * 1000

                    exp_str = self._format_date_thalex(best_expiry)
                    call_name = f"BTC-{exp_str}-{strike}-C"
                    put_name = f"BTC-{exp_str}-{strike}-P"

                    c_data = instrument_data.get(call_name, {}).get(ts)
                    p_data = instrument_data.get(put_name, {}).get(ts)

                    if c_data and p_data:
                        c_mark = (
                            float(c_data[4])
                            if isinstance(c_data, list)
                            else float(c_data.get("close", 0))
                        )
                        p_mark = (
                            float(p_data[4])
                            if isinstance(p_data, list)
                            else float(p_data.get("close", 0))
                        )
                        straddle = c_mark + p_mark
                        days_to_expiry = (best_expiry - current_dt).days
                        em_pct = straddle / price if price > 0 else 0

                        metrics_to_insert.append(
                            (
                                datetime.fromtimestamp(ts, tz=timezone.utc),
                                "BTC",
                                strike,
                                best_expiry.date(),
                                days_to_expiry,
                                c_mark,
                                p_mark,
                                straddle,
                                0.0,  # IV
                                em_pct,
                            )
                        )

                # B. Perpetual Tickers (REAL API DATA)
                if perp_history:
                    for point in perp_history:
                        if isinstance(point, list):
                            # [time, open, high, low, close, vol, (meta?)]
                            ts = point[0]
                            close_price = float(point[4])

                            tickers_to_insert.append(
                                (
                                    datetime.fromtimestamp(ts, tz=timezone.utc),
                                    "BTC-PERPETUAL",
                                    "thalex",
                                    close_price,  # bid proxy
                                    close_price,  # ask proxy
                                    close_price,  # last
                                    0,  # volume
                                )
                            )

                # Batch Insert via Adapter
                await self.db.save_options_metrics_bulk(metrics_to_insert)
                await self.db.save_tickers_bulk(tickers_to_insert)

                logger.info(f"Inserted {len(metrics_to_insert)} records for chunk.")
                current_start += chunk_size
                await asyncio.sleep(0.5)

        except Exception as e:
            logger.error(f"Loader failed: {e}", exc_info=True)
        finally:
            if self.session:
                await self.session.close()
            await self.disconnect()


if __name__ == "__main__":
    loader = HistoricalOptionsLoader()
    # Run for 30 days backfill
    asyncio.run(loader.run(days_back=30))
