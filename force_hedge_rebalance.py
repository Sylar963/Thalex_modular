#!/usr/bin/env python3
"""
Force a rebalance of hedge positions.
This script connects to a running market maker and forces it to rebalance its hedge positions.
"""

import sys
import os
import time
import json
import logging
from typing import Dict, Any
import socket
from pathlib import Path

# Import the hedge manager - fix the import path
from thalex_py.Thalex_modular.components.hedge import create_hedge_manager

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("force_hedge")

# Create a mock exchange client that can provide prices
class MockExchangeClient:
    """Mock exchange client for testing"""
    
    def __init__(self):
        """Initialize with market prices"""
        self.market_prices = {}
        
    def set_price(self, symbol, price):
        """Set market price for a symbol"""
        self.market_prices[symbol] = price
        
    def get_price(self, symbol):
        """Get market price for a symbol"""
        return self.market_prices.get(symbol, 0)
        
    def get_mark_price(self, symbol):
        """Get mark price (same as market price for mock)"""
        return self.market_prices.get(symbol, 0)
        
    def get_ticker(self, symbol):
        """Get ticker data"""
        price = self.market_prices.get(symbol, 0)
        return {"mark_price": price, "last": price}

def force_rebalance(btc_price=None, eth_price=None):
    """Force a rebalance of hedge positions"""
    # Create mock exchange client with prices
    exchange_client = MockExchangeClient()
    
    # Set prices in the mock client
    if btc_price:
        exchange_client.set_price("BTC-PERPETUAL", btc_price)
    if eth_price:
        exchange_client.set_price("ETH-PERPETUAL", eth_price)
    
    # Create hedge manager with exchange client
    hedge_manager = create_hedge_manager(
        exchange_client=exchange_client
    )
    
    # Start the hedge manager
    hedge_manager.start()
    logger.info("Hedge manager started")
    
    try:
        # Get current hedge state
        state_file = Path("hedge_state.json")
        current_positions = {}
        
        if state_file.exists():
            try:
                with open(state_file, 'r') as f:
                    state = json.load(f)
                    
                # Get market prices from state
                market_prices = state.get("market_prices", {})
                active_hedges = state.get("active_hedges", {})
                
                # Use provided prices or get from state
                btc_price = btc_price or market_prices.get("BTC-PERPETUAL", 0)
                eth_price = eth_price or market_prices.get("ETH-PERPETUAL", 0)
                
                # Extract current positions
                for primary_asset, hedges in active_hedges.items():
                    for hedge_asset, position in hedges.items():
                        current_positions[primary_asset] = {
                            "size": position["primary_position"],
                            "price": position["primary_price"]
                        }
                
                logger.info(f"Loaded current positions from state: {json.dumps(current_positions)}")
            except Exception as e:
                logger.error(f"Error loading hedge state: {e}")
        
        # Set market prices if provided
        if btc_price:
            hedge_manager.update_market_price("BTC-PERPETUAL", btc_price)
            logger.info(f"Updated BTC price: {btc_price}")
        
        if eth_price:
            hedge_manager.update_market_price("ETH-PERPETUAL", eth_price)
            logger.info(f"Updated ETH price: {eth_price}")
        
        # Force rebalance for all positions
        for asset, position in current_positions.items():
            logger.info(f"Rebalancing {asset} position: {position['size']} @ {position['price']}")
            
            # Update position to trigger rebalance
            result = hedge_manager.update_position(
                asset,
                position["size"],
                position["price"]
            )
            
            # Log hedge operations
            if result["hedges"]:
                logger.info(f"Hedge operations:")
                for hedge in result["hedges"]:
                    logger.info(f"  {hedge['side']} {hedge['size']} {hedge['hedge_asset']} @ {hedge['price']}")
            else:
                logger.info(f"No hedge operations needed")
        
        # Wait for any pending operations to complete
        time.sleep(2)
        
        # Check final state
        all_positions = hedge_manager.get_all_hedged_positions()
        pnl = hedge_manager.calculate_portfolio_pnl()
        
        logger.info(f"=== FINAL HEDGE STATUS ===")
        logger.info(f"Total PnL: ${pnl['total_pnl']:.2f}")
        
        for primary_asset, hedges in all_positions.items():
            for hedge_asset, position in hedges.items():
                primary_monetary = position.primary_position * position.primary_price
                hedge_monetary = position.hedge_position * position.hedge_price
                net_monetary = primary_monetary + hedge_monetary
                
                logger.info(f"Asset: {primary_asset}")
                logger.info(f"  Primary: {position.primary_position} @ {position.primary_price:.2f} = ${primary_monetary:.2f}")
                logger.info(f"  Hedge: {position.hedge_position} {hedge_asset} @ {position.hedge_price:.2f} = ${hedge_monetary:.2f}")
                logger.info(f"  Net monetary exposure: ${net_monetary:.2f}")
                if primary_monetary != 0:
                    logger.info(f"  Hedge ratio: {abs(hedge_monetary/primary_monetary):.2f} (target: {position.hedge_ratio:.2f})")
                logger.info(f"  PnL: ${position.pnl:.2f}")
        
    except Exception as e:
        logger.error(f"Error rebalancing hedges: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Stop the hedge manager
        hedge_manager.stop()
        logger.info("Hedge manager stopped")

def main():
    """Main function"""
    btc_price = None
    eth_price = None
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        btc_price = float(sys.argv[1])
    if len(sys.argv) > 2:
        eth_price = float(sys.argv[2])
    
    force_rebalance(btc_price, eth_price)

if __name__ == "__main__":
    main() 