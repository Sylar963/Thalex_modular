import asyncio
import os
import logging
from datetime import datetime, timedelta, timezone
import asyncpg
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load env
load_dotenv()


async def backfill_data():
    # 1. Connect to DB
    dsn = "postgresql://postgres:password@localhost:5432/thalex_trading"
    try:
        conn = await asyncpg.connect(dsn)
        logger.info("Connected to DB")
    except Exception as e:
        logger.error(f"Failed to connect to 5432: {e}")
        dsn = "postgresql://postgres:password@localhost:5433/thalex_trading"
        conn = await asyncpg.connect(dsn)
        logger.info("Connected to DB on 5433")

    # 2. Setup Thalex Adapter
    # We need to use the public API for historical data.
    # ThalexAdapter uses the 'thalex' library.
    # We can try to use the adapter or just use the library directly/requests if easier.
    # ThalexAdapter doesn't seem to have a 'get_historical_candles' method exposed in the snippet.
    # Let's use the thalex library directly if possible, or 'public/mark_price_historical_data' via base request.

    # Wait, ThalexAdapter inherits from BaseExchangeAdapter.
    # Let's see if we can use the library directly to fetch public data.
    from thalex.thalex import Thalex, Network

    client = Thalex(Network.PROD)
    await client.initialize()

    symbol = "BTC-PERPETUAL"

    # Missing Ranges from previous step
    missing_ranges = [
        (
            datetime.fromisoformat("2026-02-02T02:00:00+00:00"),
            datetime.fromisoformat("2026-02-02T19:00:00+00:00"),
        ),
        (
            datetime.fromisoformat("2026-02-02T21:00:00+00:00"),
            datetime.fromisoformat("2026-02-03T00:00:00+00:00"),
        ),
        (
            datetime.fromisoformat("2026-02-03T05:00:00+00:00"),
            datetime.fromisoformat("2026-02-04T06:00:00+00:00"),
        ),
        (
            datetime.fromisoformat("2026-02-07T05:00:00+00:00"),
            datetime.fromisoformat("2026-02-07T18:00:00+00:00"),
        ),
    ]

    for start, end in missing_ranges:
        logger.info(f"Backfilling range: {start} -> {end}")

        # Thalex public/mark_price_historical_data usually takes start_ts, end_ts
        # Inspecting thalex.py in adapter shows client.last_trades, instrument etc.
        # Let's assume there is a method or we construct a raw request.

        start_ts = int(start.timestamp())
        end_ts = int(end.timestamp())

        # Thalex Production API URL
        # Confirmed via historical_sources.py: https://thalex.com/api/v2/public/mark_price_historical_data
        base_url = "https://thalex.com/api/v2/public"
        endpoint = "mark_price_historical_data"
        url = f"{base_url}/{endpoint}"

        import aiohttp

        async with aiohttp.ClientSession() as session:
            logger.info(f"Trying {url}...")

            # Correct parameters based on historical_sources.py
            params = {
                "instrument_name": symbol,
                "from": start_ts,
                "to": end_ts,
                "resolution": "1m",
            }

            try:
                async with session.get(url, params=params) as resp:
                    if resp.status != 200:
                        logger.warning(
                            f"Failed {url}: {resp.status} {await resp.text()}"
                        )
                        # continue # No loop anymore
                    else:
                        data = await resp.json()
                        result = data.get("result", [])

                        # Handle nested dict response (from historical_sources.py)
                        if isinstance(result, dict):
                            for val in result.values():
                                if isinstance(val, list):
                                    result = val
                                    break

                        if not result:
                            logger.warning(f"No result from {url}")
                            # continue
                        else:
                            logger.info(f"Fetched {len(result)} points from {url}")

                            # Insert into market_tickers
                            rows_to_insert = []
                            for point in result:
                                # Thalex format: [time, open, high, low, close, volume] (or similar)
                                # From source: index 0 is time, index 4 is close/price.
                                if isinstance(point, list) and len(point) >= 5:
                                    ts = point[0]
                                    price = float(point[4])
                                    dt = datetime.fromtimestamp(ts, tz=timezone.utc)
                                    rows_to_insert.append(
                                        (dt, symbol, "thalex", price, price, price, 0.0)
                                    )
                                elif isinstance(point, dict):
                                    ts = point.get("time")
                                    price = float(
                                        point.get("close", 0) or point.get("price", 0)
                                    )
                                    if ts:
                                        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
                                        rows_to_insert.append(
                                            (
                                                dt,
                                                symbol,
                                                "thalex",
                                                price,
                                                price,
                                                price,
                                                0.0,
                                            )
                                        )

                            if rows_to_insert:
                                try:
                                    await conn.executemany(
                                        """
                                        INSERT INTO market_tickers (time, symbol, exchange, bid, ask, last, volume)
                                        VALUES ($1, $2, $3, $4, $5, $6, $7)
                                        ON CONFLICT (time, symbol, exchange) DO NOTHING
                                        """,
                                        rows_to_insert,
                                    )
                                    logger.info(f"Inserted {len(rows_to_insert)} rows")
                                except Exception as e:
                                    logger.error(f"Insert failed: {e}")
            except Exception as e:
                logger.warning(f"Error requesting {url}: {e}")

    await client.disconnect()
    await conn.close()


if __name__ == "__main__":
    asyncio.run(backfill_data())
