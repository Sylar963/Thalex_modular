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
        
        # 4. Filter for 15-day minimum expiry
        now_ts = datetime.now().timestamp()
        min_expiry_ts = now_ts + (15 * 24 * 3600)
        
        # Get unique expiries and their timestamps from option_details if available, 
        # but build_options_chain just gives the string 'expiry'.
        # Let's re-fetch instruments to get timestamps.
        instruments = await adapter.get_instruments(currency="HYPE")
        valid_expiries = []
        for inst in instruments:
            expiry_ts = inst['option_details']['expiry']
            if expiry_ts >= min_expiry_ts:
                valid_expiries.append((expiry_ts, inst['instrument_name'].split('-')[1]))
        
        if not valid_expiries:
            logger.error("No expiries found matching 15-day minimum.")
            return
            
        # Sort by timestamp and pick the first one >= 15 days
        valid_expiries.sort()
        target_expiry_ts, target_expiry_str = valid_expiries[0]
        logger.info(f"Targeting 15-day+ expiry: {target_expiry_str} (TS: {target_expiry_ts})")
        
        expiry_chain = chain[chain['expiry'] == target_expiry_str]
        
        # 5. Find ATM Strike
        # Fetch HYPE-PERP ticker specifically for reliable index price
        perp_tickers = await adapter.get_tickers(currency="HYPE", instrument_type="perp")
        index_price = 0.0
        if perp_tickers:
            # Handle the specific dict-to-list format from our adapter update
            perp_data = perp_tickers[0]
            # In Derive v2, 'I' is index price in the slim/ticker data
            index_price = float(perp_data.get('I', perp_data.get('index_price', 0)))
            logger.info(f"Current HYPE Index Price: {index_price}")
        
        if index_price == 0:
            logger.warning("Could not fetch index price. Estimating from strikes.")
            index_price = expiry_chain['strike'].mean()

        # Find ATM Strike (closest to index price)
        expiry_chain['diff'] = (expiry_chain['strike'] - index_price).abs()
        atm_strike_row = expiry_chain.loc[expiry_chain['diff'].idxmin()]
        atm_strike = atm_strike_row['strike']
        
        logger.info(f"ATM Strike: {atm_strike}")
        
        atm_call = expiry_chain[(expiry_chain['strike'] == atm_strike) & (expiry_chain['type'] == 'Call')].iloc[0]
        atm_put = expiry_chain[(expiry_chain['strike'] == atm_strike) & (expiry_chain['type'] == 'Put')].iloc[0]
        
        # 6. Fetch specific ATM Call and Put tickers for pricing
        # We use public/get_ticker for these specifically as build_options_chain might have missed them
        call_name = atm_call['instrument_name']
        put_name = atm_put['instrument_name']
        
        logger.info(f"Fetching specific tickers for ATM pair: {call_name}, {put_name}")
        
        call_resp = await adapter._rpc_request_ws("public/get_ticker", {"instrument_name": call_name})
        put_resp = await adapter._rpc_request_ws("public/get_ticker", {"instrument_name": put_name})
        
        c_ticker = call_resp.get("result", {})
        p_ticker = put_resp.get("result", {})
        
        # Extract mark price from result root or option_pricing
        c_price = float(c_ticker.get('mark_price', c_ticker.get('option_pricing', {}).get('mark_price', 0)))
        p_price = float(p_ticker.get('mark_price', p_ticker.get('option_pricing', {}).get('mark_price', 0)))
        
        logger.info(f"ATM Call ({call_name}): {c_price}")
        logger.info(f"ATM Put  ({put_name}): {p_price}")
        
        # 7. Calculate Expected Move
        # Formula: 0.85 * (ATM Call + ATM Put) for approx 1 standard deviation move?
        # Or just Straddle Price.
        
        straddle_price = c_price + p_price
        expected_move = 0.85 * straddle_price
        
        print("\n" + "="*40)
        print(f"DERIVE HYPE OPTIONS ANALYSIS")
        print(f"Expiry: {target_expiry_str}")
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
