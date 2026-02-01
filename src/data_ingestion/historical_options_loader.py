import asyncio
import logging
import os
import aiohttp
import json
from datetime import datetime, timedelta, timezone
import psycopg2
from psycopg2.extras import execute_batch
import numpy as np
from typing import List, Dict, Tuple, Optional

# Configuration
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_NAME = os.getenv("DB_NAME", "thalex_trading")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASS = os.getenv("DB_PASS", "password")

THALEX_API_URL = "https://thalex.com/api/v2"  # Verify base URL

# Logger Setup
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("HistoricalLoader")


class HistoricalOptionsLoader:
    def __init__(self):
        self.conn = None
        self.session = None

    def connect_db(self):
        try:
            self.conn = psycopg2.connect(
                host=DB_HOST, database=DB_NAME, user=DB_USER, password=DB_PASS
            )
            logger.info("Connected to TimescaleDB")
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            raise

    async def _fetch(self, endpoint: str, params: Dict) -> Dict:
        url = f"{THALEX_API_URL}/{endpoint}"
        async with self.session.get(url, params=params) as resp:
            if resp.status != 200:
                logger.error(f"API Error {resp.status} for {url}: {await resp.text()}")
                return {}
            return await resp.json()

    async def get_index_history(
        self, index_name: str, start_ts: int, end_ts: int
    ) -> List[Dict]:
        """Fetch index price history."""
        # Endpoint based on spec: public/index_price_historical_data
        # Actually standard thalex is usually /public/history or similar, sticking to user spec
        # User Spec: public/index_price_historical_data

        # NOTE: Thalex API usually uses specific endpoints.
        # For this script I will use the user provided 'public/index_price_historical_data' name
        # but fallback to likely standard 'public/history' if needed or assumes user spec is exact mapping.

        # Based on Thalex public docs, 'public/history' with type='index' might be it.
        # But let's assume valid mapping or generic 'history' endpoint for now.
        # Implied Spec: params: index_name, from, to, resolution

        params = {
            "index_name": index_name,
            "from": start_ts,
            "to": end_ts,
            "resolution": "1h",
        }
        # Trying user spec endpoint name
        data = await self._fetch("public/index_price_historical_data", params)
        return data.get("result", [])  # Assuming standard response wrapper

    async def get_mark_history(
        self, instrument_name: str, start_ts: int, end_ts: int
    ) -> List[Dict]:
        """Fetch mark price history for an instrument."""
        params = {
            "instrument_name": instrument_name,
            "from": start_ts,
            "to": end_ts,
            "resolution": "1h",
        }
        data = await self._fetch("public/mark_price_historical_data", params)
        return data.get("result", [])

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

    async def run(self, days_back: int = 7):
        self.connect_db()
        self.session = aiohttp.ClientSession()

        try:
            end_dt = datetime.now(timezone.utc)
            start_dt = end_dt - timedelta(days=days_back)

            start_ts = int(start_dt.timestamp())
            end_ts = int(end_dt.timestamp())

            logger.info(f"Fetching data from {start_dt} to {end_dt}")

            # 1. Fetch Index History
            index_history = await self.get_index_history("BTCUSD", start_ts, end_ts)
            if not index_history:
                logger.warning("No index history found. Exiting.")
                return

            # Convert to list of (ts, price)
            # Assuming result is list of dicts like {'time': ..., 'close': ...}
            # Or list of lists. Adapting to common OHLCV list-of-lists or dicts.
            # Spec implies list of objects or similar. I'll handle dicts.

            logger.info(f"Fetched {len(index_history)} index points.")

            likely_expiries = self._generate_expirations(
                start_dt, end_dt + timedelta(days=60)
            )  # Look ahead for expiries

            metrics_batch = []

            for point in index_history:
                ts = point.get("time")  # integer seconds
                if not ts:
                    continue
                price = float(point.get("close", 0))
                if price <= 0:
                    continue

                # Check timeframe logic: We need an expiry ~30 days out
                current_dt = datetime.fromtimestamp(ts, tz=timezone.utc)
                target_dte = 30

                # Find closest expiry to 30 days
                target_date = current_dt + timedelta(days=target_dte)
                valid_expiries = [e for e in likely_expiries if e > current_dt]
                if not valid_expiries:
                    continue

                best_expiry = min(
                    valid_expiries, key=lambda x: abs((x - target_date).days)
                )

                # Identify ATM Strike (Round to nearest 1000 for BTC)
                strike = round(price / 1000) * 1000

                # Construct Instrument Names
                # Spec Format: OBTCUSD-{DDMMMYY}-{STRIKE}-{C/P}
                exp_str = self._format_date_thalex(best_expiry)
                call_name = f"OBTCUSD-{exp_str}-{strike}-C"
                put_name = f"OBTCUSD-{exp_str}-{strike}-P"

                # Note: This is request-heavy (N * 2 requests per hour).
                # Optimization: Fetch full history for likely ATM candidates once and cache?
                # For this script, we'll do naive loop but limit concurrency or batching if needed.
                # To avoid spamming, let's gather all unique instrument names first?
                # No, we need time-point alignment.
                # Actually, fetching history for one instrument returns array.
                # So we can fetch history for likely instruments ONCE.
                pass  # Logic refined below

            # OPTIMIZED APPROACH:
            # 1. Identify all (Expiry, Strike) pairs needed across the timeframe.
            unique_pairs = set()
            for point in index_history:
                ts = point.get("time")
                if not ts:
                    continue
                price = float(point.get("close", 0))
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

            logger.info(f"identified {len(unique_pairs)} unique option pairs to fetch.")

            # 2. Fetch History for each pair
            instrument_data = {}  # Map name -> history_dict_by_ts

            for expiry, strike in unique_pairs:
                exp_str = self._format_date_thalex(expiry)
                for kind in ["C", "P"]:
                    # Name format check: OBTCUSD vs BTC
                    # User spec says OBTCUSD.
                    instr_name = f"OBTCUSD-{exp_str}-{strike}-{kind}"

                    hist = await self.get_mark_history(instr_name, start_ts, end_ts)
                    if hist:
                        # Convert to dict for fast lookup: ts -> price/iv
                        data_map = {h["time"]: h for h in hist}
                        instrument_data[instr_name] = data_map
                    else:
                        logger.warning(f"No history for {instr_name}")

            # 3. Join Data
            count = 0
            for point in index_history:
                ts = point.get("time")
                if not ts:
                    continue
                price = float(point.get("close", 0))

                current_dt = datetime.fromtimestamp(ts, tz=timezone.utc)
                valid_expiries = [e for e in likely_expiries if e > current_dt]
                if not valid_expiries:
                    continue
                best_expiry = min(
                    valid_expiries,
                    key=lambda x: abs((x - (current_dt + timedelta(days=30))).days),
                )
                strike = round(price / 1000) * 1000

                exp_str = self._format_date_thalex(best_expiry)
                call_name = f"OBTCUSD-{exp_str}-{strike}-C"
                put_name = f"OBTCUSD-{exp_str}-{strike}-P"

                c_data = instrument_data.get(call_name, {}).get(ts)
                p_data = instrument_data.get(put_name, {}).get(ts)

                if c_data and p_data:
                    c_mark = float(
                        c_data.get("close", 0)
                    )  # Using close of hourly bar as mark proxy
                    p_mark = float(p_data.get("close", 0))
                    # Thalex history might return 'price' or 'mark_price' or 'close'

                    straddle = c_mark + p_mark
                    days_to_expiry = (best_expiry - current_dt).days

                    # Store metrics
                    # time, underlying, strike, expiry_date, days_to_expiry, call_mark, put_mark, straddle, iv, em
                    self.save_metric(
                        ts=datetime.fromtimestamp(ts, tz=timezone.utc),
                        underlying="BTC",
                        strike=strike,
                        expiry=best_expiry.date(),
                        dte=days_to_expiry,
                        c_mark=c_mark,
                        p_mark=p_mark,
                        straddle=straddle,
                        # IV might not be in OHLCV, assumes approx or missing
                        iv=0.0,
                        em_pct=straddle / price if price > 0 else 0,
                    )
                    count += 1

            logger.info(f"Successfully processed {count} records.")

        except Exception as e:
            logger.error(f"Loader failed: {e}", exc_info=True)
        finally:
            await self.session.close()

    def save_metric(
        self, ts, underlying, strike, expiry, dte, c_mark, p_mark, straddle, iv, em_pct
    ):
        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT INTO options_live_metrics 
            (time, underlying, strike, expiry_date, days_to_expiry, call_mark_price, put_mark_price, straddle_price, implied_vol, expected_move_pct)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT DO NOTHING
            """,
            (ts, underlying, strike, expiry, dte, c_mark, p_mark, straddle, iv, em_pct),
        )
        self.conn.commit()


if __name__ == "__main__":
    loader = HistoricalOptionsLoader()
    asyncio.run(loader.run(days_back=7))
