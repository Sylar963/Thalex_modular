import asyncio
import os
import logging
import pandas as pd
from datetime import datetime
from src.adapters.exchanges.derive_adapter import DeriveAdapter

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("DeriveExpectedMove")

async def main():
    # 1. Initialize Adapter
    adapter = DeriveAdapter(testnet=False) # Use Production
    
    try:
        # 2. Connect
        await adapter.connect()
        
        # 3. Fetch HYPE Options Chain
        logger.info("Fetching HYPE options chain...")
        chain = await adapter.build_options_chain(currency="HYPE")
        
        if chain.empty:
            logger.error("No HYPE options found. Exiting.")
            return

        logger.info(f"Found {len(chain)} HYPE option instruments.")
        
        # 4. Filter for nearest expiration
        # Convert expiry strings to dates or sort. Assuming '20MAY24' or '20240520' format?
        # Derive v2 uses 'YYYYMMDD' usually? Or 'DMMMYY'?
        # Let's inspect the first expiry format if possible, but assuming standard sort works for now if format is YYYYMMDD.
        # If it's 20240315 it sorts correctly.
        
        # Filter out expired? The API call already had expired=False.
        
        expiries = sorted(chain['expiry'].unique())
        if not expiries:
            logger.error("No expiries found.")
            return
            
        target_expiry = expiries[0] # Nearest expiry
        logger.info(f"Targeting nearest expiry: {target_expiry}")
        
        expiry_chain = chain[chain['expiry'] == target_expiry]
        
        # 5. Find ATM Strike
        # We need the underlying price. We can get it from the 'index_price' of any option ticker 
        # or just infer it from the straddle with minimum difference.
        # Alternatively, fetch the Spot Ticker.
        
        # Let's use the average of the strikes with the lowest premium spread? 
        # Or easier: Find the strike where Call Price ~ Put Price (Straddle Delta Neutral).
        # Or just get the index price.
        
        # To get index price, we can fetch "public/get_ticker" for the perpetual if available?
        # Or just use the 'underlying_price' / 'index_price' field if present in ticker. 
        # The 'get_tickers' result usually has 'index_price'.
        
        # Let's re-fetch tickers to get index price from one of them.
        tickers = await adapter.get_tickers(currency="HYPE")
        if tickers:
            index_price = float(tickers[0].get('index_price', 0))
            logger.info(f"Current HYPE Index Price: {index_price}")
        else:
            logger.warning("Could not fetch index price. Estimating from strikes.")
            index_price = expiry_chain['strike'].mean()

        # Find ATM Strike (closest to index price)
        expiry_chain['diff'] = (expiry_chain['strike'] - index_price).abs()
        atm_strike_row = expiry_chain.loc[expiry_chain['diff'].idxmin()]
        atm_strike = atm_strike_row['strike']
        
        logger.info(f"ATM Strike: {atm_strike}")
        
        # 6. Subscribe to ATM Call and Put (Optional, but good for verification of live data)
        # We can just use the snapshot from 'build_options_chain' if it populated prices.
        
        atm_call = expiry_chain[(expiry_chain['strike'] == atm_strike) & (expiry_chain['type'] == 'Call')].iloc[0]
        atm_put = expiry_chain[(expiry_chain['strike'] == atm_strike) & (expiry_chain['type'] == 'Put')].iloc[0]
        
        c_price = atm_call['mark_price']
        p_price = atm_put['mark_price']
        
        logger.info(f"ATM Call ({atm_strike}): {c_price}")
        logger.info(f"ATM Put  ({atm_strike}): {p_price}")
        
        # 7. Calculate Expected Move
        # Formula: 0.85 * (ATM Call + ATM Put) for approx 1 standard deviation move?
        # Or just Straddle Price.
        
        straddle_price = c_price + p_price
        expected_move = 0.85 * straddle_price
        
        print("\n" + "="*40)
        print(f"DERIVE HYPE OPTIONS ANALYSIS")
        print(f"Expiry: {target_expiry}")
        print(f"Index Price: ${index_price:.4f}")
        print(f"ATM Strike: {atm_strike}")
        print("-" * 40)
        print(f"ATM Straddle Price: ${straddle_price:.4f}")
        print(f"Expected Move (0.85 * Straddle): ${expected_move:.4f}")
        print("="*40 + "\n")
        
    except Exception as e:
        logger.error(f"An error occurred: {e}")
    finally:
        await adapter.disconnect()

if __name__ == "__main__":
    asyncio.run(main())
