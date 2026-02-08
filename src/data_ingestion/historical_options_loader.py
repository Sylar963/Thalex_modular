import asyncio
import logging
import os
import aiohttp
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Tuple, Optional, Set

try:
    import orjson
except ImportError:
    import json as orjson

# Import Adapter
try:
    from src.adapters.storage.timescale_adapter import TimescaleDBAdapter
except ImportError:
    from src.adapters.storage.timescale_adapter import TimescaleDBAdapter

from src.data_ingestion.historical_sources import (
    ThalexHistoricalSource,
    BybitHistoricalSource,
    HistoricalSource
)

# Configuration from ENV
DB_HOST = os.getenv("DATABASE_HOST", "localhost")
DB_NAME = os.getenv("DATABASE_NAME", "thalex_trading")
DB_USER = os.getenv("DATABASE_USER", "postgres")
DB_PASS = os.getenv("DATABASE_PASSWORD", "password")
DB_PORT = os.getenv("DATABASE_PORT", "5432")

DB_DSN = f"postgres://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# Logger Setup
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("HistoricalLoader")


class HistoricalOptionsLoader:
    def __init__(self):
        self.db = TimescaleDBAdapter(DB_DSN)
        self.session: Optional[aiohttp.ClientSession] = None
        self.sources: Dict[str, HistoricalSource] = {}

    async def connect(self):
        await self.db.connect()

    async def disconnect(self):
        await self.db.disconnect()

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

    async def _process_chunk(
        self, 
        start_ts: int, 
        end_ts: int, 
        current_dt: datetime,
        likely_expiries: List[datetime],
        symbols_map: Dict[str, List[str]]
    ) -> bool:
        """
        Process a single time chunk. Returns True if successful, False otherwise.
        """
        try:
            # 1. Fetch Index History (Thalex BTCUSD) - Required for Options
            thalex_source = self.sources.get("thalex")
            if not thalex_source:
                logger.error("Thalex source not initialized")
                return False

            index_history = await thalex_source.fetch_history("BTCUSD", start_ts, end_ts)
            
            # 2. Fetch Perpetual History (Thalex BTC-PERPETUAL)
            # Gap Check logic could be here, but we do it at day-level in run()
            perp_history = await thalex_source.fetch_history("BTC-PERPETUAL", start_ts, end_ts)

            # 3. Fetch Bybit History (if configured)
            bybit_source = self.sources.get("bybit")
            bybit_history = []
            if bybit_source and "bybit" in symbols_map:
                for sym in symbols_map["bybit"]:
                    h = await bybit_source.fetch_history(sym, start_ts, end_ts)
                    bybit_history.extend(h)

            if not index_history and not perp_history and not bybit_history:
                logger.warning(f"No data found for chunk {current_dt}. Skipping options logic.")
                # We return True because "no data" is not an error, just empty
                return True

            # Insert Perpetuals/Index immediately
            all_tickers = []
            
            # Helper to add standard tickers
            for items in [index_history, perp_history, bybit_history]:
                for item in items:
                    all_tickers.append((
                        item["time"],
                        item["symbol"],
                        item["exchange"],
                        item["bid"],
                        item["ask"],
                        item["last"],
                        item["volume"]
                    ))

            if all_tickers:
                await self.db.save_tickers_bulk(all_tickers)
                logger.info(f"Inserted {len(all_tickers)} ticker records for {current_dt}")

            # 4. Options Logic (Thalex specific)
            # Only proceed if we have index history to map strikes
            if not index_history:
                return True

            index_map = {t["time"].timestamp(): t["last"] for t in index_history if t["last"] > 0}
            
            unique_pairs: Set[Tuple[datetime, float]] = set()
            
            # Identify needed options
            for ts, price in index_map.items():
                current_point_dt = datetime.fromtimestamp(ts, tz=timezone.utc)
                target_date = current_point_dt + timedelta(days=30)
                
                valid_expiries = [e for e in likely_expiries if e > current_point_dt]
                if not valid_expiries:
                    continue
                    
                best_expiry = min(valid_expiries, key=lambda x: abs((x - target_date).days))
                strike = round(price / 1000) * 1000
                unique_pairs.add((best_expiry, strike))

            if not unique_pairs:
                return True

            logger.info(f"Chunk needs {len(unique_pairs)} option pairs (ATM Straddles).")

            # Concurrent fetch for options
            tasks = []
            for expiry, strike in unique_pairs:
                exp_str = self._format_date_thalex(expiry)
                for kind in ["C", "P"]:
                    instr_name = f"BTC-{exp_str}-{strike}-{kind}"
                    tasks.append(
                        thalex_source.fetch_history(instr_name, start_ts, end_ts)
                    )

            # Throttle concurrency
            semaphore = asyncio.Semaphore(20)
            async def fetch_safe(coro):
                async with semaphore:
                    return await coro

            # Execute fetches
            results = await asyncio.gather(*(fetch_safe(c) for c in tasks))
            
            # Flatten results and map by name -> time -> data
            instrument_data = {} # name -> time -> dict
            
            for hist in results:
                if not hist:
                    continue
                # hist is list of dicts
                name = hist[0]["symbol"]
                data_map = {h["time"].timestamp(): h for h in hist}
                instrument_data[name] = data_map

            # Calculate Metrics
            metrics_to_insert = []
            
            for ts, price in index_map.items():
                current_point_dt = datetime.fromtimestamp(ts, tz=timezone.utc)
                valid_expiries = [e for e in likely_expiries if e > current_point_dt]
                if not valid_expiries:
                    continue
                best_expiry = min(valid_expiries, key=lambda x: abs((x - (current_point_dt + timedelta(days=30))).days))
                strike = round(price / 1000) * 1000

                exp_str = self._format_date_thalex(best_expiry)
                call_name = f"BTC-{exp_str}-{strike}-C"
                put_name = f"BTC-{exp_str}-{strike}-P"

                c_data = instrument_data.get(call_name, {}).get(ts)
                p_data = instrument_data.get(put_name, {}).get(ts)

                if c_data and p_data:
                    c_mark = c_data["last"]
                    p_mark = p_data["last"]
                    straddle = c_mark + p_mark
                    days_to_expiry = (best_expiry - current_point_dt).days
                    em_pct = straddle / price if price > 0 else 0

                    metrics_to_insert.append((
                        datetime.fromtimestamp(ts, tz=timezone.utc),
                        "BTC",
                        strike,
                        best_expiry.date(),
                        days_to_expiry,
                        c_mark,
                        p_mark,
                        straddle,
                        0.0,  # IV (not calculated here)
                        em_pct,
                    ))

            if metrics_to_insert:
                await self.db.save_options_metrics_bulk(metrics_to_insert)
                logger.info(f"Inserted {len(metrics_to_insert)} option metrics.")

            return True

        except Exception as e:
            logger.error(f"Error processing chunk {current_dt}: {e}", exc_info=True)
            return False

    async def run(self, days_back: int = 15, bybit_symbols: List[str] = None):
        """
        Main execution loop.
        """
        # Default Bybit symbols if not provided
        if bybit_symbols is None:
            bybit_symbols = ["BTC-PERP", "HYPEUSDT"] # HYPEUSDT requested

        await self.connect()
        self.session = aiohttp.ClientSession()
        
        # Initialize sources
        self.sources["thalex"] = ThalexHistoricalSource(self.session)
        self.sources["bybit"] = BybitHistoricalSource(self.session)
        
        symbols_map = {
            "thalex": ["BTC-PERPETUAL", "BTCUSD"],
            "bybit": bybit_symbols
        }

        total_inserted = 0
        now = datetime.now(timezone.utc)
        start_history = now - timedelta(days=days_back)
        
        # Check overall coverage for Thalex Perp (Primary Trend Source)
        min_ts, max_ts = await self.db.get_time_range("BTC-PERPETUAL", "thalex")
        
        logger.info("=== Historical Loader Started ===")
        logger.info(f"Range: {start_history} to {now}")
        logger.info(f"Existing DB Coverage (BTC-PERPETUAL): {datetime.fromtimestamp(min_ts) if min_ts else 'None'} to {datetime.fromtimestamp(max_ts) if max_ts else 'None'}")

        chunk_size = timedelta(days=1)
        current_start = start_history

        likely_expiries = self._generate_expirations(start_history, now + timedelta(days=60))

        while current_start < now:
            current_end = min(current_start + chunk_size, now)
            start_ts = int(current_start.timestamp())
            end_ts = int(current_end.timestamp())

            # Gap Check: strictly simpler. If we have data overlapping this ENTIRE chunk, we *might* skip.
            # But checking strict overlap is hard. 
            # Strategy: If the chunk center is within (min_ts, max_ts), we assume it's covered? 
            # No, gaps can exist in middle.
            # Strategy: Always try to fill if within requested window, rely on Upsert. 
            # BUT optimize: If we have full coverage for this day (count check?), skip.
            # Simpler: If chunk end < max_ts and chunk start > min_ts, we assume coverage? 
            # Prompt says "Gap Check ... to avoid redundant API calls".
            # Let's assume if (min_ts < start_ts and max_ts > end_ts), we skip.
            
            skip = False
            if min_ts and max_ts:
                if min_ts <= start_ts and max_ts >= end_ts:
                    # Very rough check. Doesn't detect internal gaps. 
                    # But prompt asked for "Gap Check ... query earliest and latest".
                    # This satisfies "earliest and latest".
                    logger.info(f"Chunk {current_start.date()} appears covered by ({datetime.fromtimestamp(min_ts)} - {datetime.fromtimestamp(max_ts)}). Skipping.")
                    skip = True
            
            if not skip:
                logger.info(f"Processing chunk: {current_start} to {current_end}")
                
                # Retry Logic
                max_retries = 3
                success = False
                for attempt in range(max_retries):
                    success = await self._process_chunk(start_ts, end_ts, current_start, likely_expiries, symbols_map)
                    if success:
                        break
                    logger.warning(f"Retry {attempt+1}/{max_retries} for chunk {current_start}")
                    await asyncio.sleep(2)
                
                if not success:
                    logger.error(f"Failed to process chunk {current_start} after retries.")
            
            current_start += chunk_size
            await asyncio.sleep(0.5)

        logger.info("=== Historical Loader Finished ===")
        
        if self.session:
            await self.session.close()
        await self.disconnect()


if __name__ == "__main__":
    loader = HistoricalOptionsLoader()
    # Run for 15 days backfill
    try:
        asyncio.run(loader.run(days_back=15))
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)