"""
ThalexClient implementation for the Thalex exchange.
This client adapts the native thalex API to work with the hedge execution system.
"""

import asyncio
import logging
import time
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
        last_successful_update = None
        no_message_counter = 0
        error_counter = 0
        
        # Initialize prices with reasonable defaults if needed
        if "BTC-PERPETUAL" not in self.market_prices:
            self.market_prices["BTC-PERPETUAL"] = 0.0
        if "ETH-PERPETUAL" not in self.market_prices:
            self.market_prices["ETH-PERPETUAL"] = 0.0
        
        while True:
            try:
                # Get a message from the websocket with timeout
                message = await asyncio.wait_for(self.thalex.receive(), timeout=5.0)
                
                if not message:
                    no_message_counter += 1
                    if no_message_counter >= 10:
                        logger.warning(f"No messages received for {no_message_counter} attempts, checking connection...")
                        no_message_counter = 0
                        
                        # Check connection and reinitialize if needed
                        if not self.thalex.connected():
                            logger.error("Thalex client disconnected, attempting to reconnect...")
                            reconnected = await self.initialize()
                            if reconnected:
                                logger.info("Successfully reconnected to Thalex")
                            else:
                                logger.error("Failed to reconnect to Thalex")
                                await asyncio.sleep(5)  # Wait before retrying
                    continue
                
                no_message_counter = 0  # Reset counter on successful message
                
                # Try to parse message if it's a string
                if isinstance(message, str):
                    try:
                        import json
                        message = json.loads(message)
                    except json.JSONDecodeError:
                        logger.error(f"Failed to parse message as JSON: {message[:100]}...")
                        continue
                
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
                                
                                # Update price information with explicit logging
                                old_price = self.market_prices.get(instrument, 0.0)
                                
                                # Update with last price if available
                                if 'last_price' in data and data['last_price'] > 0:
                                    new_price = data['last_price']
                                    self.market_prices[instrument] = new_price
                                    logger.info(f"Updated {instrument} price: {old_price} -> {new_price}")
                                    last_successful_update = time.time()
                                
                                # Also update with mark price if available
                                if 'mark_price' in data and data['mark_price'] > 0:
                                    mark_price = data['mark_price']
                                    self.market_prices[f"{instrument}_mark"] = mark_price
                                    logger.info(f"Updated {instrument} mark price to {mark_price}")
                                    last_successful_update = time.time()
                            else:
                                logger.warning(f"Malformed channel format: {params['channel']}")
                    else:
                        # Log other message types for debugging
                        if message.get('method') != 'heartbeat':  # Skip heartbeat messages
                            logger.debug(f"Received non-ticker message: {message}")
                
                # Check if we haven't received price updates for too long
                if last_successful_update and time.time() - last_successful_update > 60:
                    logger.warning(f"No price updates for {int(time.time() - last_successful_update)} seconds")
                    # Attempt resubscription
                    try:
                        logger.info("Attempting to resubscribe to ticker data...")
                        instruments = ["BTC-PERPETUAL", "ETH-PERPETUAL"]
                        channels = [f"ticker.{i}.raw" for i in instruments]
                        subscribe_response = await self.thalex.public_subscribe(channels=channels)
                        logger.info(f"Resubscribe response: {subscribe_response}")
                        last_successful_update = time.time()  # Reset timer
                    except Exception as e:
                        logger.error(f"Error resubscribing to ticker data: {e}")
            
            except asyncio.TimeoutError:
                logger.warning("Timeout waiting for ticker data")
                continue
            except asyncio.CancelledError:
                logger.info("Price update task cancelled")
                break
            except Exception as e:
                error_counter += 1
                logger.error(f"Error in price update task: {str(e)}")
                if error_counter >= 5:
                    logger.error(f"Too many consecutive errors ({error_counter}), attempting reinitialization")
                    error_counter = 0
                    try:
                        # Try to reconnect and reinitialize
                        await self.initialize()
                    except Exception as reconnect_error:
                        logger.error(f"Failed to reinitialize after errors: {reconnect_error}")
                
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
    
    async def ensure_initialized(self) -> bool:
        """
        Ensure the client is initialized before proceeding with operations
        
        Returns:
            True if successfully initialized or already initialized
        """
        if self.is_initialized:
            # Check if the connection is still active
            if not self.thalex.connected():
                logger.warning("Client was initialized but connection lost, reconnecting...")
                self.is_initialized = False
                return await self.initialize()
            return True
            
        logger.info("Client not initialized, initializing now...")
        return await self.initialize()
    
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
        initialized = await self.ensure_initialized()
        if not initialized:
            logger.error("Failed to initialize client before placing market order")
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
                "filled_price": self.get_price(symbol) or response.get("price", 0.0)
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
        initialized = await self.ensure_initialized()
        if not initialized:
            logger.error("Failed to initialize client before placing limit order")
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
            True if cancellation was successful
        """
        logger.info(f"Cancelling order: {order_id}")
        
        # Ensure client is initialized
        initialized = await self.ensure_initialized()
        if not initialized:
            logger.error("Failed to initialize client before cancelling order")
            return False
        
        try:
            # Call Thalex cancel method
            response = await self.thalex.cancel(order_id=order_id)
            logger.info(f"Cancel response: {response}")
            return True
        except Exception as e:
            logger.error(f"Error cancelling order: {e}")
            return False 