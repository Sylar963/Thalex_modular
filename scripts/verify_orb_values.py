import asyncio
import aiohttp
import time
import logging
from datetime import datetime, timedelta
import zoneinfo
from src.domain.signals.open_range import OpenRangeSignalEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def fetch_candles(symbol: str, start_ts: float, end_ts: float):
    url = "https://api.bybit.com/v5/market/kline"
    params = {
        "category": "linear",
        "symbol": symbol,
        "interval": "1",
        "start": int(start_ts * 1000),
        "end": int(end_ts * 1000),
        "limit": 1000,
    }

    async with aiohttp.ClientSession() as session:
        async with session.get(url, params=params) as resp:
            data = await resp.json()
            if data["retCode"] != 0:
                logger.error(f"Error fetching candles: {data}")
                return []

            candles = data["result"]["list"]
            # Sort ascending by time
            candles.sort(key=lambda x: int(x[0]))
            return candles


async def run_verification():
    # User Params updated to match their values (20:15-20:30 PST)
    symbol = "HYPEUSDT"
    target_pct = 3.49
    subsequent_pct = 320
    session_start = "20:15"
    session_end = "20:30"
    timezone = "America/Tijuana"

    logger.info(f"Verifying ORB values for {symbol} with:")
    logger.info(f"Target % from Mid: {target_pct}%")
    logger.info(f"Subsequent Target % of Range: {subsequent_pct}%")
    logger.info(f"Session: {session_start}-{session_end} {timezone}")

    engine = OpenRangeSignalEngine(
        target_pct_from_mid=target_pct,
        subsequent_target_pct_of_range=subsequent_pct,
        session_start_utc=session_start,
        session_end_utc=session_end,
        timezone=timezone,
    )

    tz = zoneinfo.ZoneInfo(timezone)
    now = datetime.now(tz)
    # Feb 9th
    target_date = now.date() - timedelta(days=1)

    # Fetch wider range to include 20:15 session
    start_dt = datetime.combine(
        target_date, datetime.strptime("19:00", "%H:%M").time(), tzinfo=tz
    )
    end_dt = datetime.combine(
        target_date, datetime.strptime("22:00", "%H:%M").time(), tzinfo=tz
    )

    start_ts = start_dt.timestamp()
    end_ts = end_dt.timestamp()

    logger.info(f"Fetching data from {start_dt} to {end_dt}")

    candles = await fetch_candles(symbol, start_ts, end_ts)
    if not candles:
        logger.error("No candles found.")
        return

    logger.info(f"Fetched {len(candles)} candles.")

    for c in candles:
        ts = int(c[0]) / 1000.0
        open_p = float(c[1])
        high = float(c[2])
        low = float(c[3])
        close = float(c[4])

        # Feed candle to engine
        engine.update_candle(symbol, ts, open_p, high, low, close)

    # Get Final State
    state = engine._get_state(symbol)

    print("\n" + "=" * 40)
    print(f"ORB RESULTS for {symbol} ({start_dt.date()})")
    print(f"ORH: {state.orh:.4f}")
    print(f"ORL: {state.orl:.4f}")
    print(f"ORM: {state.orm:.4f}")
    print(f"ORW: {state.orw:.4f}")
    print(f"Day Direction: {state.day_dir}")
    print("-" * 20)
    print(f"T1 UP: {state.first_up_target:.4f}")
    print(f"T1 MEAN: {state.first_down_target:.4f}")
    print("-" * 20)
    print("UP TARGETS:")
    for t in state.up_targets:
        print(f"  {t.label}: {t.price:.4f} (Hit: {t.hit})")
    print("DOWN TARGETS:")
    for t in state.down_targets:
        print(f"  {t.label}: {t.price:.4f} (Hit: {t.hit})")
    print("=" * 40 + "\n")


if __name__ == "__main__":
    asyncio.run(run_verification())
