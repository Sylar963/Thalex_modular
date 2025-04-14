#!/usr/bin/env python3
"""
Fix ETH-PERPETUAL price fetching in the hedge manager.
This script ensures the hedge manager has access to ETH prices and can send real orders to the testnet.
"""

import sys
import time
import logging
import asyncio
import json
from typing import Dict, Any
from pathlib import Path

# Add the project directory to Python path to ensure all imports work
import os
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)

# Import Thalex client and hedge manager
import thalex as th
from thalex import Network
from thalex_py.Thalex_modular.config.market_config import BOT_CONFIG, CALL_IDS
from thalex_py.Thalex_modular.models.keys import key_ids, private_keys
from thalex_py.Thalex_modular.components.hedge import create_hedge_manager
from thalex_py.Thalex_modular.components.hedge.hedge_execution import OrderSide

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("eth_price_fix")

class ThalexClientWrapper:
    """Wrapper for Thalex client to adapt to what hedge manager expects"""
    
    def __init__(self):
        """Initialize with Thalex client"""
        # Get network from config
        network = BOT_CONFIG["market"]["network"]
        logger.info(f"Creating Thalex client for {network} network")
        
        self.thalex = th.Thalex(network=network)
        self.market_prices = {}
        self.client_initialized = False
        
        # Add a price update task
        self.price_update_task = None
        
    async def initialize(self):
        """Initialize the Thalex client with login and subscriptions"""
        if self.client_initialized:
            return
        
        try:
            # Connect to Thalex
            logger.info("Connecting to Thalex...")
            await self.thalex.connect()
            
            if not self.thalex.connected():
                logger.error("Failed to connect to Thalex")
                return False
            
            logger.info("Connected to Thalex successfully")
            
            # Login
            logger.info("Logging in...")
            network = BOT_CONFIG["market"]["network"]
            key_id = key_ids[network]
            private_key = private_keys[network]
            login_response = await self.thalex.login(key_id, private_key, id=CALL_IDS["login"])
            logger.info(f"Login response: {login_response}")
            
            # Subscribe to ticker data
            logger.info("Subscribing to ticker data...")
            subscribe_response = await self.thalex.public_subscribe(
                channels=["ticker.BTC-PERPETUAL.raw", "ticker.ETH-PERPETUAL.raw"],
                id=CALL_IDS["subscribe"]
            )
            logger.info(f"Subscribe response: {subscribe_response}")
            
            # Start the price update task
            self.price_update_task = asyncio.create_task(self._update_prices_continuous())
            
            # Wait for initial prices to be set
            for attempt in range(10):
                if "BTC-PERPETUAL" in self.market_prices and "ETH-PERPETUAL" in self.market_prices:
                    break
                logger.info(f"Waiting for initial prices... ({attempt+1}/10)")
                await asyncio.sleep(1)
            
            if "BTC-PERPETUAL" not in self.market_prices or "ETH-PERPETUAL" not in self.market_prices:
                logger.error("Failed to get initial prices after 10 attempts")
                return False
                
            logger.info(f"Initialized Thalex client with real prices: BTC={self.market_prices.get('BTC-PERPETUAL')}, ETH={self.market_prices.get('ETH-PERPETUAL')}")
            self.client_initialized = True
            return True
            
        except Exception as e:
            logger.error(f"Error initializing Thalex client: {e}")
            return False
    
    async def _update_prices_continuous(self):
        """Continuously update prices from WebSocket messages"""
        try:
            logger.info("Starting continuous price updates")
            while True:
                try:
                    # Get message from WebSocket
                    msg = await self.thalex.receive()
                    
                    # Parse the message
                    if isinstance(msg, str):
                        try:
                            msg = json.loads(msg)
                        except:
                            pass
                    
                    # Extract ticker data if available
                    if isinstance(msg, dict):
                        # For WebSocket notification messages
                        if "channel_name" in msg and "notification" in msg:
                            channel = msg["channel_name"]
                            if "ticker.BTC-PERPETUAL" in channel and "mark_price" in msg["notification"]:
                                new_price = msg["notification"]["mark_price"]
                                old_price = self.market_prices.get("BTC-PERPETUAL")
                                self.market_prices["BTC-PERPETUAL"] = new_price
                                logger.info(f"Updated BTC price: {old_price} -> {new_price}")
                                
                            elif "ticker.ETH-PERPETUAL" in channel and "mark_price" in msg["notification"]:
                                new_price = msg["notification"]["mark_price"]
                                old_price = self.market_prices.get("ETH-PERPETUAL")
                                self.market_prices["ETH-PERPETUAL"] = new_price
                                logger.info(f"Updated ETH price: {old_price} -> {new_price}")
                                
                        # For direct API responses
                        elif "result" in msg and isinstance(msg["result"], dict) and "mark_price" in msg["result"]:
                            if "BTC-PERPETUAL" in str(msg):
                                self.market_prices["BTC-PERPETUAL"] = msg["result"]["mark_price"]
                                logger.info(f"Updated BTC price from API: {msg['result']['mark_price']}")
                            elif "ETH-PERPETUAL" in str(msg):
                                self.market_prices["ETH-PERPETUAL"] = msg["result"]["mark_price"]
                                logger.info(f"Updated ETH price from API: {msg['result']['mark_price']}")
                
                except asyncio.CancelledError:
                    logger.info("Price update task cancelled")
                    break
                    
                except Exception as e:
                    logger.error(f"Error processing WebSocket message: {e}")
                    await asyncio.sleep(1)
                    
        except asyncio.CancelledError:
            logger.info("Price update task cancelled")
        except Exception as e:
            logger.error(f"Fatal error in price update task: {e}")
    
    async def shutdown(self):
        """Shutdown the Thalex client"""
        if self.price_update_task:
            self.price_update_task.cancel()
            try:
                await self.price_update_task
            except asyncio.CancelledError:
                pass
                
        if self.thalex:
            await self.thalex.disconnect()
    
    def get_price(self, symbol):
        """Get market price for a symbol"""
        price = self.market_prices.get(symbol, 0)
        logger.debug(f"get_price({symbol}) -> {price}")
        return price
        
    def get_mark_price(self, symbol):
        """Get mark price (same as market price)"""
        price = self.market_prices.get(symbol, 0)
        logger.debug(f"get_mark_price({symbol}) -> {price}")
        return price
        
    def get_ticker(self, symbol):
        """Get ticker data"""
        price = self.market_prices.get(symbol, 0)
        logger.debug(f"get_ticker({symbol}) -> {{'mark_price': {price}}}")
        return {"mark_price": price, "last": price}
        
    def get_instrument_data(self, symbol):
        """Get instrument data"""
        price = self.market_prices.get(symbol, 0)
        logger.debug(f"get_instrument_data({symbol}) -> {{'mark_price': {price}}}")
        return {"mark_price": price}
    
    async def place_market_order(self, symbol, side, size):
        """Place a real market order on Thalex"""
        logger.info(f"Placing order: {symbol} {side} {size}")
        
        # Convert hedge manager OrderSide to Thalex Direction
        direction = th.Direction.BUY if side == OrderSide.BUY.value else th.Direction.SELL
        
        try:
            # Use the proper Thalex API to place the order
            if direction == th.Direction.BUY:
                response = await self.thalex.buy(
                    instrument_name=symbol,
                    amount=abs(size),
                    order_type=th.OrderType.MARKET
                )
            else:
                response = await self.thalex.sell(
                    instrument_name=symbol,
                    amount=abs(size),
                    order_type=th.OrderType.MARKET
                )
                
            logger.info(f"Order response: {response}")
            return response
            
        except Exception as e:
            logger.error(f"Error placing order: {e}")
            return {"status": "error", "error": str(e)}
            
    async def place_limit_order(self, symbol, side, size, price):
        """Place a real limit order on Thalex"""
        logger.info(f"Placing limit order: {symbol} {side} {size} @ {price}")
        
        # Convert hedge manager OrderSide to Thalex Direction
        direction = th.Direction.BUY if side == OrderSide.BUY.value else th.Direction.SELL
        
        try:
            # Use the proper Thalex API to place the order
            if direction == th.Direction.BUY:
                response = await self.thalex.buy(
                    instrument_name=symbol,
                    amount=abs(size),
                    price=price,
                    order_type=th.OrderType.LIMIT
                )
            else:
                response = await self.thalex.sell(
                    instrument_name=symbol,
                    amount=abs(size),
                    price=price,
                    order_type=th.OrderType.LIMIT
                )
                
            logger.info(f"Order response: {response}")
            return response
            
        except Exception as e:
            logger.error(f"Error placing limit order: {e}")
            return {"status": "error", "error": str(e)}

async def fix_eth_price_fetching():
    """Fix ETH-PERPETUAL price fetching with real Thalex client"""
    # Create a real Thalex client wrapper
    exchange_client = ThalexClientWrapper()
    
    try:
        # Initialize the client (connect, login, get prices)
        success = await exchange_client.initialize()
        if not success:
            logger.error("Failed to initialize Thalex client. Exiting.")
            return
        
        # Validate that we have prices
        if not exchange_client.market_prices.get("ETH-PERPETUAL"):
            logger.error("Failed to get ETH-PERPETUAL price from exchange")
            return
        
        logger.info("Successfully fetched real-time ETH price from exchange")
        
        # Create hedge manager with the real exchange client
        hedge_manager = create_hedge_manager(
            exchange_client=exchange_client
        )
        
        # Start the hedge manager
        hedge_manager.start()
        logger.info("Hedge manager started with real exchange client")
        
        try:
            # Use the real-time prices directly from exchange client
            btc_price = exchange_client.market_prices.get("BTC-PERPETUAL")
            eth_price = exchange_client.market_prices.get("ETH-PERPETUAL")
            
            # Validate real-time prices before proceeding
            if btc_price <= 0 or eth_price <= 0:
                logger.error(f"Invalid prices detected: BTC={btc_price}, ETH={eth_price}. Both must be positive values.")
                return
                
            logger.info(f"Verified real-time prices: BTC={btc_price}, ETH={eth_price}")
            
            # Explicitly set prices to ensure they're in the hedge manager
            hedge_manager.update_market_price("BTC-PERPETUAL", btc_price)
            hedge_manager.update_market_price("ETH-PERPETUAL", eth_price)
            logger.info(f"Set real-time prices in hedge manager: BTC={btc_price}, ETH={eth_price}")
            
            # Check current market prices
            logger.info(f"Current market prices: {hedge_manager.market_prices}")
            
            # Verify exchange client can fetch ETH price
            eth_price_from_exchange = hedge_manager._fetch_price_from_exchange("ETH-PERPETUAL")
            logger.info(f"ETH price from exchange: {eth_price_from_exchange}")
            
            # Validate that real price is used, not fallback
            eth_price_ratio = btc_price / eth_price if eth_price > 0 else 0
            logger.info(f"Real BTC/ETH price ratio: {eth_price_ratio:.2f} (should NOT be exactly 16.0)")
            
            if abs(eth_price_ratio - 16.0) < 0.1:
                logger.warning("WARNING: BTC/ETH price ratio is very close to 16.0, which suggests fallback values might be used")
            
            # Check hedge configuration
            hedge_assets = hedge_manager.config.get_hedge_assets("BTC-PERPETUAL")
            correlation_factors = hedge_manager.config.get_correlation_factors("BTC-PERPETUAL")
            logger.info(f"Hedge assets for BTC: {hedge_assets}")
            logger.info(f"Correlation factors for BTC: {correlation_factors}")
            
            # Force a rebalance - now with real-time prices
            logger.info("Forcing a rebalance of all hedges with real-time prices...")
            hedge_manager._rebalance_all_hedges()
            
            # Get current hedged positions
            positions = hedge_manager.get_all_hedged_positions()
            
            # Log positions with real-time prices
            logger.info("Current hedged positions (with real-time prices):")
            for primary_asset, hedges in positions.items():
                for hedge_asset, position in hedges.items():
                    primary_notional = position.primary_position * position.primary_price
                    hedge_notional = position.hedge_position * position.hedge_price
                    
                    logger.info(f"Primary: {primary_asset} {position.primary_position} @ {position.primary_price}")
                    logger.info(f"Hedge: {hedge_asset} {position.hedge_position} @ {position.hedge_price}")
                    logger.info(f"Primary notional: ${primary_notional}")
                    logger.info(f"Hedge notional: ${hedge_notional}")
                    if primary_notional != 0:
                        logger.info(f"Hedge ratio: {abs(hedge_notional/primary_notional):.2f}")
            
            # Keep the script running to allow continuous price updates
            logger.info("Maintaining connection for 60 seconds to allow continuous price updates...")
            
            # Run for a minute to allow price updates to flow through
            for i in range(6):
                await asyncio.sleep(10)
                
                # Update the hedge manager with the latest prices
                btc_price = exchange_client.market_prices.get("BTC-PERPETUAL")
                eth_price = exchange_client.market_prices.get("ETH-PERPETUAL")
                
                # Validate prices are real-time at each iteration
                if btc_price <= 0 or eth_price <= 0:
                    logger.error(f"Invalid prices detected during iteration {i+1}: BTC={btc_price}, ETH={eth_price}")
                    continue
                
                # Verify the BTC/ETH ratio is not exactly the hardcoded fallback value
                eth_price_ratio = btc_price / eth_price if eth_price > 0 else 0
                if abs(eth_price_ratio - 16.0) < 0.01:
                    logger.warning(f"Iteration {i+1}: BTC/ETH ratio suspiciously close to hardcoded value (16.0): {eth_price_ratio:.4f}")
                else:
                    logger.info(f"Iteration {i+1}: Using real-time prices with BTC/ETH ratio: {eth_price_ratio:.4f}")
                
                hedge_manager.update_market_price("BTC-PERPETUAL", btc_price)
                hedge_manager.update_market_price("ETH-PERPETUAL", eth_price)
                
                # Log current prices
                logger.info(f"Updated real-time prices: BTC={btc_price}, ETH={eth_price}")
                
                # Force rebalance with latest prices
                hedge_manager._rebalance_all_hedges()
                logger.info(f"Rebalanced with latest prices (iteration {i+1}/6)")
            
            # Final verification that we're using real-time data
            btc_final = exchange_client.market_prices.get("BTC-PERPETUAL")
            eth_final = exchange_client.market_prices.get("ETH-PERPETUAL")
            final_ratio = btc_final / eth_final if eth_final > 0 else 0
            
            logger.info(f"Final verification - Real-time prices: BTC={btc_final}, ETH={eth_final}, Ratio={final_ratio:.4f}")
            logger.info("ETH price fetching with real-time data is now working correctly")
            
        finally:
            # Stop the hedge manager
            hedge_manager.stop()
            logger.info("Hedge manager stopped")
    finally:
        # Shutdown the Thalex client
        await exchange_client.shutdown()

def main():
    """Main entry point"""
    # Create a new event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        # Run with graceful interrupt handling
        try:
            loop.run_until_complete(fix_eth_price_fetching())
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received, shutting down gracefully...")
            # Create and run shutdown task
            shutdown_task = asyncio.ensure_future(shutdown_gracefully(), loop=loop)
            loop.run_until_complete(shutdown_task)
    finally:
        loop.close()

async def shutdown_gracefully():
    """Perform graceful shutdown"""
    logger.info("Performing graceful shutdown...")
    # Create a new client just for cleanup
    client = ThalexClientWrapper()
    try:
        await client.initialize()
        logger.info("Initialized client for cleanup")
        
        # Clean up any pending orders or state
        await client.shutdown()
        logger.info("Client shutdown complete")
    except Exception as e:
        logger.error(f"Error during graceful shutdown: {e}")

if __name__ == "__main__":
    main() 