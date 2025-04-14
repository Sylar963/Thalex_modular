"""
ThalexClient implementation for the Thalex exchange.
This client adapts the native thalex API to work with the hedge execution system.
"""

import asyncio
import logging
from typing import Dict, Optional, Any, List
from enum import Enum

# Import the native Thalex client
try:
    from thalex.thalex import Thalex, Network, Direction, OrderType as ThOrderType
except ImportError:
    raise ImportError("Thalex package not installed. Please install it with 'pip install thalex'")

from ..thalex_logging import LoggerFactory

logger = LoggerFactory.configure_component_logger("thalex_client", log_file="client.log")


class ThalexClient:
    """Thalex exchange client implementation"""
    
    def __init__(self, api_key: str, api_secret: str, testnet: bool = True):
        """
        Initialize the Thalex client
        
        Args:
            api_key: API key for Thalex
            api_secret: API secret for Thalex
            testnet: Whether to use the testnet
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        
        # Network selection
        network = Network.TESTNET if testnet else Network.MAINNET
        
        # Native Thalex client
        self.thalex = Thalex(network=network)
        
        # State tracking
        self.is_initialized = False
        self.market_prices: Dict[str, float] = {}
        
        # Background tasks
        self.price_update_task = None
        
        logger.info(f"Initialized ThalexClient with testnet={testnet}")
    
    async def initialize(self) -> bool:
        """
        Initialize the client (connect, login)
        
        Returns:
            True if initialization was successful
        """
        if self.is_initialized:
            return True
            
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
            login_response = await self.thalex.login(self.api_key, self.api_secret)
            logger.info(f"Login response: {login_response}")
            
            # Subscribe to ticker data for common instruments
            instruments = ["BTC-PERPETUAL", "ETH-PERPETUAL"]
            channels = [f"ticker.{i}.raw" for i in instruments]
            
            logger.info(f"Subscribing to ticker data: {channels}")
            subscribe_response = await self.thalex.public_subscribe(channels=channels)
            logger.info(f"Subscribe response: {subscribe_response}")
            
            # Start the price update task
            self.price_update_task = asyncio.create_task(self._update_prices_continuous())
            
            self.is_initialized = True
            logger.info("ThalexClient initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing ThalexClient: {e}")
            return False
    
    async def _update_prices_continuous(self):
        """Continuously update market prices from ticker data"""
        logger.info("Starting continuous price updates")
        
        while True:
            try:
                # Get a message from the websocket
                message = await self.thalex.receive()
                
                # Process ticker updates
                if message and isinstance(message, dict):
                    if message.get('method') == 'subscription' and 'params' in message:
                        params = message['params']
                        
                        # Check if it's a ticker update
                        if params.get('channel', '').startswith('ticker.') and params.get('channel', '').endswith('.raw'):
                            # Extract the instrument name from the channel
                            parts = params['channel'].split('.')
                            if len(parts) >= 3:
                                instrument = parts[1]
                                data = params.get('data', {})
                                
                                # Update price information
                                if 'last_price' in data and data['last_price'] > 0:
                                    self.market_prices[instrument] = data['last_price']
                                    logger.debug(f"Updated {instrument} price to {data['last_price']}")
                                
                                # Also update with mark price if available
                                if 'mark_price' in data and data['mark_price'] > 0:
                                    self.market_prices[f"{instrument}_mark"] = data['mark_price']
                                    logger.debug(f"Updated {instrument} mark price to {data['mark_price']}")
            
            except asyncio.CancelledError:
                logger.info("Price update task cancelled")
                break
            except Exception as e:
                logger.error(f"Error in price update task: {e}")
                await asyncio.sleep(1)  # Prevent tight loop on error
    
    def get_price(self, instrument_name: str) -> float:
        """
        Get current price for an instrument
        
        Args:
            instrument_name: Instrument to get price for
            
        Returns:
            Current price or 0 if not available
        """
        # Return the cached price if available
        return self.market_prices.get(instrument_name, 0.0)
    
    def get_mark_price(self, instrument_name: str) -> float:
        """
        Get current mark price for an instrument
        
        Args:
            instrument_name: Instrument to get price for
            
        Returns:
            Current mark price or 0 if not available
        """
        # Return the cached mark price if available
        return self.market_prices.get(f"{instrument_name}_mark", 0.0)
    
    async def place_market_order(self, symbol: str, side: str, size: float) -> Dict:
        """
        Place a market order
        
        Args:
            symbol: Instrument to trade
            side: "buy" or "sell"
            size: Order size
            
        Returns:
            Order response dict
        """
        logger.info(f"Placing market order: {symbol} {side} {size}")
        
        # Ensure client is initialized
        if not self.is_initialized:
            success = await self.initialize()
            if not success:
                return {"status": "error", "error": "Failed to initialize client"}
        
        try:
            # Convert order direction
            direction = Direction.BUY if side.lower() == "buy" else Direction.SELL
            
            # Place the order
            if direction == Direction.BUY:
                response = await self.thalex.buy(
                    instrument_name=symbol,
                    amount=abs(size),
                    order_type=ThOrderType.MARKET
                )
            else:
                response = await self.thalex.sell(
                    instrument_name=symbol,
                    amount=abs(size),
                    order_type=ThOrderType.MARKET
                )
                
            logger.info(f"Market order response: {response}")
            
            # Format the response for hedge execution
            return {
                "status": "filled",
                "filled_size": size,
                "filled_price": self.get_price(symbol)
            }
            
        except Exception as e:
            logger.error(f"Error placing market order: {e}")
            return {"status": "error", "error": str(e)}
    
    async def place_limit_order(self, symbol: str, side: str, size: float, price: float) -> Dict:
        """
        Place a limit order
        
        Args:
            symbol: Instrument to trade
            side: "buy" or "sell"
            size: Order size
            price: Limit price
            
        Returns:
            Order response dict
        """
        logger.info(f"Placing limit order: {symbol} {side} {size} @ {price}")
        
        # Ensure client is initialized
        if not self.is_initialized:
            success = await self.initialize()
            if not success:
                return {"status": "error", "error": "Failed to initialize client"}
        
        try:
            # Convert order direction
            direction = Direction.BUY if side.lower() == "buy" else Direction.SELL
            
            # Place the order
            if direction == Direction.BUY:
                response = await self.thalex.buy(
                    instrument_name=symbol,
                    amount=abs(size),
                    price=price,
                    order_type=ThOrderType.LIMIT
                )
            else:
                response = await self.thalex.sell(
                    instrument_name=symbol,
                    amount=abs(size),
                    price=price,
                    order_type=ThOrderType.LIMIT
                )
                
            logger.info(f"Limit order response: {response}")
            
            # Format the response for hedge execution
            return {
                "status": "open",
                "filled_size": 0.0,
                "filled_price": 0.0,
                "order_id": response.get("order_id", "")
            }
            
        except Exception as e:
            logger.error(f"Error placing limit order: {e}")
            return {"status": "error", "error": str(e)}
    
    async def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an order
        
        Args:
            order_id: Order ID to cancel
            
        Returns:
            True if successful
        """
        logger.info(f"Cancelling order: {order_id}")
        
        # Ensure client is initialized
        if not self.is_initialized:
            success = await self.initialize()
            if not success:
                return False
        
        try:
            response = await self.thalex.cancel(order_id=order_id)
            logger.info(f"Cancel order response: {response}")
            return True
        except Exception as e:
            logger.error(f"Error cancelling order: {e}")
            return False 