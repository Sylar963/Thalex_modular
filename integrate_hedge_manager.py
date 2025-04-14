#!/usr/bin/env python3
"""
Example of integrating the hedge manager with a trading system.
This shows how to use the hedge manager in a production environment.
"""

import time
import logging
from typing import Dict, Any

# Import the hedge manager
from thalex_py.Thalex_modular.components.hedge import create_hedge_manager

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("hedge_integration")

class TradingSystem:
    """Example trading system class that uses the hedge manager"""
    
    def __init__(self, config_path=None, exchange_client=None):
        """Initialize the trading system with a hedge manager"""
        # Create the hedge manager
        self.hedge_manager = create_hedge_manager(
            config_path=config_path,
            exchange_client=exchange_client,  # Pass the exchange client explicitly
            strategy_type="notional"  # Options: "notional" or "delta_neutral"
        )
        
        # Initialize price tracking
        self.market_prices = {}
        
        # Start the hedge manager (this starts background rebalancing)
        self.hedge_manager.start()
        logger.info("Hedge manager started")
    
    def update_market_data(self, symbol: str, price: float):
        """Process market data updates"""
        # Update local price tracking
        self.market_prices[symbol] = price
        
        # Forward price updates to the hedge manager
        self.hedge_manager.update_market_price(symbol, price)
        logger.info(f"Updated market price: {symbol} @ {price}")
    
    def process_fill(self, fill_data: Dict[str, Any]):
        """Process a fill from the exchange"""
        # Example fill data structure
        # {
        #    "symbol": "BTC-PERP",
        #    "price": 85000.0,
        #    "size": 1.0,
        #    "side": "buy",  # or "sell"
        #    "timestamp": time.time()
        # }
        
        # Create a fill object that the hedge manager can process
        class Fill:
            def __init__(self, data):
                self.instrument = data["symbol"]
                self.price = data["price"]
                self.size = data["size"]
                self.is_buy = data["side"].lower() == "buy"
                self.timestamp = data.get("timestamp", time.time())
        
        # Create a fill object
        fill = Fill(fill_data)
        
        # Process the fill through the hedge manager
        self.hedge_manager.on_fill(fill)
        
        logger.info(f"Processed fill: {fill_data['symbol']} {fill_data['side']} {fill_data['size']} @ {fill_data['price']}")
    
    def update_position_directly(self, symbol: str, position: float, price: float):
        """Manually update a position and its hedge"""
        # This can be used when you know the current position without a fill event
        result = self.hedge_manager.update_position(symbol, position, price)
        
        # Log the hedge operations performed
        if result["hedges"]:
            logger.info(f"Hedge operations for {symbol} position update:")
            for hedge in result["hedges"]:
                logger.info(f"  {hedge['side']} {hedge['size']} {hedge['hedge_asset']} @ {hedge['price']}")
        else:
            logger.info(f"No hedge operations needed for {symbol} position update")
    
    def get_portfolio_status(self):
        """Get current portfolio status including all hedged positions"""
        # Get PnL information
        pnl_data = self.hedge_manager.calculate_portfolio_pnl()
        total_pnl = pnl_data["total_pnl"]
        
        # Get all hedged positions
        positions = self.hedge_manager.get_all_hedged_positions()
        
        # Create a status report
        status = {
            "total_pnl": total_pnl,
            "positions": {}
        }
        
        # Process each hedged position
        for primary_asset, hedges in positions.items():
            status["positions"][primary_asset] = []
            
            for hedge_asset, position in hedges.items():
                status["positions"][primary_asset].append({
                    "hedge_asset": hedge_asset,
                    "primary_position": position.primary_position,
                    "primary_price": position.primary_price,
                    "hedge_position": position.hedge_position,
                    "hedge_price": position.hedge_price,
                    "hedge_ratio": position.hedge_ratio,
                    "pnl": position.pnl
                })
        
        return status
    
    def shutdown(self):
        """Shutdown the trading system and hedge manager"""
        # Stop the hedge manager gracefully
        self.hedge_manager.stop()
        logger.info("Hedge manager stopped")


def main():
    """Example of running the trading system with hedge manager"""
    # Create the trading system
    trading_system = TradingSystem()
    
    try:
        # Set initial market prices - use PERPETUAL naming consistently
        trading_system.update_market_data("BTC-PERPETUAL", 85500.0)
        trading_system.update_market_data("ETH-PERPETUAL", 3200.0)
        
        # Example 1: Process a fill (BTC long position)
        fill_data = {
            "symbol": "BTC-PERPETUAL",
            "price": 85500.0,
            "size": 1.0,
            "side": "buy",
            "timestamp": time.time()
        }
        trading_system.process_fill(fill_data)
        
        # Wait a moment for hedge to process
        time.sleep(2)
        
        # Show portfolio status
        status = trading_system.get_portfolio_status()
        print("\nPortfolio Status after BTC long:")
        print(f"Total PnL: ${status['total_pnl']:.2f}")
        for asset, positions in status["positions"].items():
            print(f"Asset: {asset}")
            for pos in positions:
                print(f"  Hedged with: {pos['hedge_asset']}")
                print(f"  Primary: {pos['primary_position']} @ {pos['primary_price']}")
                print(f"  Hedge: {pos['hedge_position']} @ {pos['hedge_price']}")
                print(f"  Ratio: {pos['hedge_ratio']:.2f}")
                print(f"  PnL: ${pos['pnl']:.2f}")
        
        # Example 2: Price movement
        print("\nUpdating market prices...")
        trading_system.update_market_data("BTC-PERPETUAL", 86000.0)
        trading_system.update_market_data("ETH-PERPETUAL", 3180.0)
        
        # Calculate new PnL
        status = trading_system.get_portfolio_status()
        print(f"New Total PnL: ${status['total_pnl']:.2f}")
        
        # Example 3: Close position
        print("\nClosing BTC position...")
        fill_data = {
            "symbol": "BTC-PERPETUAL",
            "price": 86000.0,
            "size": 1.0,
            "side": "sell",
            "timestamp": time.time()
        }
        trading_system.process_fill(fill_data)
        
        # Wait a moment for hedge to process
        time.sleep(2)
        
        # Show final status
        status = trading_system.get_portfolio_status()
        print("\nFinal Portfolio Status:")
        print(f"Total PnL: ${status['total_pnl']:.2f}")
        
    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
    finally:
        # Ensure proper shutdown
        trading_system.shutdown()


if __name__ == "__main__":
    main() 