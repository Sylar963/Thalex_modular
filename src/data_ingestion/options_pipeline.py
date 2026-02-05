import asyncio
import json
import logging
import time
import os
from datetime import datetime, timedelta
import psycopg2
from psycopg2.extras import execute_batch
import numpy as np

# Thalex Library
from thalex.thalex import Thalex, Network

# Configuration
DB_HOST = os.getenv("DATABASE_HOST", "localhost")
DB_NAME = os.getenv("DATABASE_NAME", "thalex_trading")
DB_USER = os.getenv("DATABASE_USER", "postgres")
DB_PASS = os.getenv("DATABASE_PASSWORD", "password")

# Logger Setup
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("OptionsPipeline")


class OptionsDataPipeline:
    def __init__(self):
        self.thalex = Thalex(network=Network.PROD)  # Using PROD for data
        self.conn = None
        self.running = True

    def connect_db(self):
        try:
            self.conn = psycopg2.connect(
                host=DB_HOST, database=DB_NAME, user=DB_USER, password=DB_PASS
            )
            logger.info("Connected to TimescaleDB")
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            raise

    async def connect_thalex(self):
        await self.thalex.initialize()
        if not self.thalex.connected():
            raise ConnectionError("Failed to connect to Thalex WS")
        logger.info("Connected to Thalex WS")

    async def fetch_index_price(self, underlying: str) -> float:
        # Thalex usually provides index prices via ticker or dedicated endpoint
        # For this implementation, we'll try to get it from the perpetual contract ticker as a proxy
        # or specific index ticker if available.
        # Thalex ticker format: "BTC-PERP" or "BTCUSD" index?
        # Let's use the 'ticker' endpoint for the underlying index if available, e.g., ".BTC" or similar
        # Fallback: Use BTC-PERP mark price

        try:
            # Assuming we can get index via a public ticker request
            # Adjust channel/instrument based on actual API
            ticker = await self.thalex.ticker(instrument_name=f"{underlying}-PERP")
            return float(ticker["result"]["index_price"])
        except Exception as e:
            logger.error(f"Failed to fetch index price for {underlying}: {e}")
            return 0.0

    async def get_atm_options(
        self, underlying: str, index_price: float, target_dte: int = 30
    ):
        # 1. List instruments
        instruments = await self.thalex.public_instruments()
        # Filter for Options on Underlying
        options = [
            i
            for i in instruments["result"]
            if i["base_currency"] == underlying and i["type"] == "option"
        ]

        # 2. Find target expiry
        now = datetime.utcnow()
        target_date = now + timedelta(days=target_dte)

        # Find closest expiry
        expiries = sorted(list(set(o["expiration_timestamp"] for o in options)))
        closest_expiry = min(expiries, key=lambda x: abs(x - target_date.timestamp()))

        expiry_options = [
            o for o in options if o["expiration_timestamp"] == closest_expiry
        ]

        # 3. Find ATM Strike
        # Sort by distance to index_price
        strikes = sorted(list(set(o["strike_price"] for o in expiry_options)))
        atm_strike = min(strikes, key=lambda x: abs(x - index_price))

        # Get Call and Put tickers for ATM
        # Instrument names usually: BTC-28JUN24-60000-C
        # We need to construct or find the exact instrument names

        call_instr = next(
            (
                o
                for o in expiry_options
                if o["strike_price"] == atm_strike and o["option_type"] == "call"
            ),
            None,
        )
        put_instr = next(
            (
                o
                for o in expiry_options
                if o["strike_price"] == atm_strike and o["option_type"] == "put"
            ),
            None,
        )

        return call_instr, put_instr, closest_expiry

    async def run(self):
        self.connect_db()
        await self.connect_thalex()

        while self.running:
            try:
                for underlying in ["BTC", "ETH"]:
                    # 1. Fetch Index
                    index_price = await self.fetch_index_price(underlying)
                    if index_price == 0:
                        continue

                    self.save_index_price(underlying, index_price)

                    # 2. Get ATM Options (e.g., ~30 days out)
                    call_instr, put_instr, expiry_ts = await self.get_atm_options(
                        underlying, index_price, target_dte=30
                    )

                    if not call_instr or not put_instr:
                        logger.warning(f"Could not find ATM options for {underlying}")
                        continue

                    # 3. Fetch Tickers
                    call_ticker = await self.thalex.ticker(
                        instrument_name=call_instr["instrument_name"]
                    )
                    put_ticker = await self.thalex.ticker(
                        instrument_name=put_instr["instrument_name"]
                    )

                    c_mark = float(call_ticker["result"]["mark_price"])
                    p_mark = float(put_ticker["result"]["mark_price"])
                    c_iv = float(call_ticker["result"]["mark_iv"])

                    # 4. Calculate Expected Move
                    # Mark Basis
                    straddle_price = c_mark + p_mark

                    # Volatility Basis (EM = Spot * 0.8 * IV * sqrt(T))
                    days_to_expiry = (
                        datetime.fromtimestamp(expiry_ts) - datetime.utcnow()
                    ).days
                    t_years = max(days_to_expiry / 365.0, 0.001)
                    em_vol_basis = index_price * 0.8 * c_iv * np.sqrt(t_years)

                    # 5. Persist
                    self.save_metrics(
                        underlying=underlying,
                        strike=call_instr["strike_price"],
                        expiry_date=datetime.fromtimestamp(expiry_ts).date(),
                        days_to_expiry=days_to_expiry,
                        call_mark=c_mark,
                        put_mark=p_mark,
                        straddle=straddle_price,
                        iv=c_iv,
                        em_pct=straddle_price / index_price,
                    )

                await asyncio.sleep(5)  # Poll interval

            except Exception as e:
                logger.error(f"Pipeline loop error: {e}")
                await asyncio.sleep(5)

    def save_index_price(self, underlying, price):
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT INTO index_prices (time, underlying, price) VALUES (NOW(), %s, %s)",
            (underlying, price),
        )
        self.conn.commit()

    def save_metrics(
        self,
        underlying,
        strike,
        expiry_date,
        days_to_expiry,
        call_mark,
        put_mark,
        straddle,
        iv,
        em_pct,
    ):
        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT INTO options_live_metrics 
            (time, underlying, strike, expiry_date, days_to_expiry, call_mark_price, put_mark_price, straddle_price, implied_vol, expected_move_pct)
            VALUES (NOW(), %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                underlying,
                strike,
                expiry_date,
                days_to_expiry,
                call_mark,
                put_mark,
                straddle,
                iv,
                em_pct,
            ),
        )
        self.conn.commit()


if __name__ == "__main__":
    pipeline = OptionsDataPipeline()
    asyncio.run(pipeline.run())
