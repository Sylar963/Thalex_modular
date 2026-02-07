import asyncio
import os
import sys
import time
import logging
from typing import List, Optional
from dotenv import load_dotenv

# Ensure we can import from src and thalex_py
project_root = os.getcwd()
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "thalex_py"))

from thalex.thalex import Thalex, Network
from src.adapters.storage.timescale_adapter import TimescaleDBAdapter
from src.domain.entities import Trade, OrderSide

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("BackfillHistory")


async def backfill():
    load_dotenv()

    # 1. Setup DB
    db_dsn = f"postgres://{os.getenv('DATABASE_USER')}:{os.getenv('DATABASE_PASSWORD')}@{os.getenv('DATABASE_HOST')}:{os.getenv('DATABASE_PORT')}/{os.getenv('DATABASE_NAME')}"
    db = TimescaleDBAdapter(db_dsn)
    await db.connect()

    # 2. Setup Thalex API
    mode = os.getenv("TRADING_MODE", "testnet").lower()
    network = Network.PROD if mode == "production" else Network.TEST
    key_id = (
        os.getenv("THALEX_PROD_API_KEY_ID")
        if mode == "production"
        else os.getenv("THALEX_TEST_API_KEY_ID")
    )
    private_key = (
        os.getenv("THALEX_PROD_PRIVATE_KEY")
        if mode == "production"
        else os.getenv("THALEX_TEST_PRIVATE_KEY")
    )

    if not key_id or not private_key:
        logger.error("Missing API credentials")
        return

    client = Thalex(network)
    await client.initialize()
    await client.login(key_id, private_key)

    # 3. Fetch History (Last 30 days)
    logger.info("Fetching trade history...")

    try:
        req_id = int(time.time() * 1000)
        await client.trade_history(limit=100, id=req_id)

        # Wait for response with this ID
        trades_data = []

        start_wait = time.time()
        while time.time() - start_wait < 10:
            msg_str = await client.receive()
            import json

            msg = json.loads(msg_str)

            if msg.get("id") == req_id:
                if "error" in msg:
                    logger.error(f"API Error: {msg['error']}")
                    break
                trades_data = msg.get("result", {}).get("trades", [])
                break

        logger.info(f"Fetched {len(trades_data)} trades.")

        # 4. Save to DB
        count = 0
        if trades_data:
            logger.info(f"First trade sample: {trades_data[0]}")

        for t_data in trades_data:
            # Map to Trade
            # API returns 'time' in seconds (float)
            if "time" in t_data:
                ts_sec = float(t_data["time"])
            elif "timestamp" in t_data:
                ts_sec = float(t_data["timestamp"]) / 1_000_000.0
            else:
                logger.warning(f"Skipping trade without time: {t_data.keys()}")
                continue

            trade = Trade(
                id=str(t_data["trade_id"]),  # Was 'id'
                order_id=str(t_data.get("order_id", "")),
                symbol=t_data["instrument_name"],
                side=OrderSide.BUY if t_data["direction"] == "buy" else OrderSide.SELL,
                price=float(t_data["price"]),
                size=float(t_data["amount"]),
                fee=float(t_data.get("fee", 0.0)),
                exchange="thalex",
                timestamp=ts_sec,
            )

            await db.save_execution(trade)
            count += 1

        logger.info(f"Backfilled {count} trades.")

    except Exception as e:
        logger.error(f"Backfill error: {e}")
    finally:
        await client.disconnect()
        await db.disconnect()


if __name__ == "__main__":
    asyncio.run(backfill())
