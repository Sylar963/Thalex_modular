import asyncio
import json
import os
import socket
import time
import traceback
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Union
from collections import deque
import random

import numpy as np
import websockets
import orjson  # Fast JSON parsing

import thalex as th
from thalex import Network
from thalex_py.Thalex_modular.config.market_config import (
    BOT_CONFIG,
    MARKET_CONFIG, 
    CALL_IDS, 
    RISK_LIMITS,
    TRADING_PARAMS,
    TRADING_CONFIG
)
from thalex_py.Thalex_modular.models.data_models import Ticker, Order, OrderStatus, Quote
from thalex_py.Thalex_modular.components.risk_manager import RiskManager
from thalex_py.Thalex_modular.components.order_manager import OrderManager
from thalex_py.Thalex_modular.components.avellaneda_market_maker import AvellanedaMarketMaker
from thalex_py.Thalex_modular.models.keys import key_ids, private_keys
from thalex_py.Thalex_modular.performance_monitor import PerformanceMonitor
from thalex_py.Thalex_modular.thalex_logging import LoggerFactory
from thalex_py.Thalex_modular.ringbuffer.market_data_buffer import MarketDataBuffer

class AvellanedaQuoter:
    def __init__(self, thalex: th.Thalex):
        """Initialize the Avellaneda-Stoikov market maker"""
        # Core dependencies
        self.thalex = thalex
        self.logger = LoggerFactory.configure_component_logger(
            "avellaneda_quoter",
            log_file="avellaneda_quoter.log",
            high_frequency=True
        )
        
        # Set up the market maker components
        self.market_maker = AvellanedaMarketMaker()
        self.order_manager = OrderManager(self.thalex)
        self.risk_manager = RiskManager()
        self.performance_monitor = PerformanceMonitor()
        
        # Market data
        self.ticker = None
        self.index = None
        self.perp_name = None
        self.market_data = MarketDataBuffer()
        
        # Portfolio data
        self.portfolio = {}
        
        # Rate limiting parameters - now managed by circuit breaker in Thalex client
        self.max_requests_per_minute = BOT_CONFIG["connection"]["rate_limit"]
        self.rate_limit_warning_sent = False
        self.volatile_market_warning_sent = False  # Flag for tracking volatile market warnings
        
        # For coordination
        self.quote_cv = asyncio.Condition()
        self.condition_met = False
        self.setup_complete = asyncio.Event()
        
        # Connection parameters - now using improved connection management
        self.heartbeat_interval = BOT_CONFIG["connection"]["heartbeat_interval"]
        self.last_heartbeat = time.time()
        
        # Quoting management
        self.quoting_enabled = True
        self.cooldown_active = False
        self.cooldown_until = 0
        
        # Quote storage and metrics - use NumPy arrays for better performance
        self.price_history = np.zeros(100)  # Store last 100 prices
        self.price_history_idx = 0
        self.price_history_full = False
        self.current_quotes = ([], [])  # (bid_quotes, ask_quotes)
        
        # Order management
        self.max_orders_per_side = 5  # Maximum orders per side
        self.max_total_orders = 16  # Maximum total orders
        
        # Set default tick size
        self.tick_size = 1.0
        
        # Latest update timestamps
        self.last_ticker_time = 0
        self.last_quote_time = 0
        self.last_quote_update_time = 0
        
        # Instrument data cache to avoid frequent lookups
        self.instrument_data_cache = {}
        
        # Message processing statistics
        self.message_processing_times = np.zeros(1000)  # Last 1000 processing times
        self.message_processing_idx = 0
        
        self.logger.info("Avellaneda quoter initialized")

    async def start(self):
        """Start the quoter and initialize logging"""
        # Initialize logging system
        await LoggerFactory.initialize()
        self.logger.info("Starting Avellaneda quoter...")
        
        try:
            # Initialize required attributes
            self.rate_limit_warning_sent = False
            self.volatile_market_warning_sent = False
            self.cooldown_active = False
            self.cooldown_until = 0
            self.last_ticker_time = 0
            self.last_quote_time = 0
            
            # Set the rate limit in the Thalex client
            self.thalex.rate_limit = BOT_CONFIG["connection"]["rate_limit"]
            
            # Explicitly connect to the Thalex API
            self.logger.info("Connecting to Thalex API...")
            await self.thalex.connect()
            
            # Check if connection was successful
            if not self.thalex.connected():
                raise ConnectionError("Failed to establish WebSocket connection")
            
            self.logger.info("Successfully connected to Thalex API")
            
            # Initialize instrument details
            self.logger.info("Initializing instrument details...")
            await self.await_instruments()
            
            # Login and setup
            self.logger.info("Logging in and setting up...")
            network = MARKET_CONFIG["network"]
            key_id = key_ids[network]
            private_key = private_keys[network]
            await self.thalex.login(key_id, private_key, id=CALL_IDS["login"])
            
            # Set cancel on disconnect
            await self.thalex.set_cancel_on_disconnect(6, id=CALL_IDS["set_cod"])
            
            # Note: We're not setting up market maker protection since we're using individual limit orders instead of mass quotes
            self.logger.info("Using individual limit orders instead of mass quotes - no market maker protection needed")
            
            # Subscribe to channels
            self.logger.info("Subscribing to channels...")
            if self.perp_name:  # Only subscribe if we have the instrument name
                await self.thalex.public_subscribe(
                    channels=[f"ticker.{self.perp_name}.raw", f"price_index.{BOT_CONFIG['market']['underlying']}"],
                    id=CALL_IDS["subscribe"]
                )
                await self.thalex.private_subscribe(
                    channels=["session.orders", "account.portfolio", "account.trade_history"],
                    id=CALL_IDS["subscribe"] + 1
                )
            else:
                raise ValueError("No instrument name available for subscriptions")

            # Create and start tasks
            self.logger.info("Creating tasks...")
            tasks = [
                asyncio.create_task(self.listen_task()),
                asyncio.create_task(self.quote_task()),
                asyncio.create_task(self.heartbeat_task()),
                asyncio.create_task(self.performance_monitor.start_recording(self)),
                asyncio.create_task(self.gamma_update_task())
            ]
            
            self.logger.info("Tasks created, waiting for completion...")
            await asyncio.gather(*tasks)
            
        except Exception as e:
            self.logger.error(f"Error in main quoter loop: {str(e)}", exc_info=True)
        finally:
            # Ensure proper shutdown
            await self.shutdown()

    async def shutdown(self):
        """Shutdown the quoter and cleanup resources"""
        try:
            # Cancel all orders
            await self.order_manager.cancel_all_orders()
            
            # Shutdown thalex client if connected
            if self.thalex and self.thalex.connected():
                await self.thalex.disconnect()
            
            # Shutdown logging
            await LoggerFactory.shutdown()
            
            self.logger.info("Quoter shutdown complete")
        except Exception as e:
            self.logger.error(f"Error during shutdown: {str(e)}")

    async def await_instruments(self):
        """Initialize instrument details"""
        try:
            await self.thalex.instruments(CALL_IDS["instruments"])
            msg = await self.thalex.receive()
            
            # Use faster JSON parsing
            try:
                msg = orjson.loads(msg)
            except Exception:
                msg = json.loads(msg)
                
            # Verify the response matches our request ID
            if "id" not in msg or msg["id"] != CALL_IDS["instruments"]:
                raise ValueError(f"Unexpected response ID: got {msg.get('id')}, expected {CALL_IDS['instruments']}")
            
            if "result" not in msg or not isinstance(msg["result"], list):
                raise ValueError(f"Invalid instrument response format: missing or invalid 'result' field")
                
            # Cache all instruments for faster lookup
            for instrument in msg["result"]:
                self.instrument_data_cache[instrument["instrument_name"]] = instrument
            
            # Find our target perpetual instrument
            perpetual_found = False
            for i in msg["result"]:
                if i["type"] == "perpetual" and i["underlying"] == MARKET_CONFIG["underlying"]:
                    perpetual_found = True
                    tick_size = i["tick_size"]
                    self.perp_name = i["instrument_name"]
                    
                    # Set tick size for all components AND for the quoter itself
                    self.tick_size = tick_size
                    self.order_manager.set_tick_size(tick_size)
                    self.market_maker.set_tick_size(tick_size)
                    self.logger.info(f"Found perpetual {self.perp_name} with tick size {tick_size} - setting for all components")
                    return
                    
            if not perpetual_found:
                self.logger.error(f"Could not find perpetual with underlying {MARKET_CONFIG['underlying']}")
                raise ValueError(f"Perpetual {MARKET_CONFIG['underlying']} not found")
                
        except ValueError as e:
            # Re-raise value errors with the original message
            self.logger.error(f"Error processing instruments: {str(e)}")
            raise
        except Exception as e:
            # Convert other exceptions to a more specific type with context
            self.logger.error(f"Unexpected error initializing instruments: {str(e)}", exc_info=True)
            raise RuntimeError(f"Failed to initialize instruments: {str(e)}") from e

    async def check_rate_limit(self, priority: str = "normal") -> bool:
        """
        Check if we're under the rate limit - now delegated to Thalex client
        
        This is kept for backward compatibility but uses the client's circuit breaker
        """
        try:
            # For high priority operations, we'll attempt even if near limits
            if priority == "high":
                # Check if the client has a circuit breaker
                if hasattr(self.thalex, 'circuit_breaker') and hasattr(self.thalex, 'CircuitState'):
                    if self.thalex.circuit_breaker.state != self.thalex.CircuitState.OPEN:
                        return True
                else:
                    # Fall back to simple check if circuit breaker not available
                    return True
            
            # Try to use the client's rate limit checking if available
            if hasattr(self.thalex, '_check_rate_limit') and callable(self.thalex._check_rate_limit):
                return await self.thalex._check_rate_limit()
            else:
                # Fall back to assuming we're under limit if no check method exists
                self.logger.debug("No rate limit check method available, assuming under limit")
                return True
        except Exception as e:
            self.logger.error(f"Error in rate limit check: {str(e)}")
            # On error, be conservative and act as if we're rate limited
            return False

    async def listen_task(self):
        """Listen for websocket messages"""
        self.logger.info("Starting listen task...")
        
        consecutive_errors = 0
        max_consecutive_errors = 10
        base_error_delay = 0.1  # 100ms
        max_error_delay = 5.0   # 5 seconds
        
        while True:
            try:
                # Check if connection is alive before attempting to receive
                if hasattr(self.thalex, 'connected') and callable(self.thalex.connected) and not self.thalex.connected():
                    self.logger.warning("WebSocket connection lost, attempting to reconnect...")
                    try:
                        await self.thalex.connect()
                        self.logger.info("Successfully reconnected WebSocket")
                        # Reset error counter on successful reconnection
                        consecutive_errors = 0
                    except Exception as conn_err:
                        self.logger.error(f"Failed to reconnect: {str(conn_err)}")
                        # Use exponential backoff for reconnection attempts
                        await asyncio.sleep(min(2 ** consecutive_errors, 30))
                        consecutive_errors += 1
                        continue
                
                # Using the optimized receive method that handles reconnections
                message = await self.thalex.receive()
                
                # Reset consecutive errors counter on successful message
                consecutive_errors = 0
                
                start_time = time.time()
                
                # Use faster JSON parsing if message is still a string
                if isinstance(message, str):
                    try:
                        data = orjson.loads(message)
                    except Exception:
                        data = json.loads(message)
                else:
                    data = message
                
                # Process notification - handle both old and new API formats
                # New format uses 'channel_name' instead of 'channel'
                if "notification" in data:
                    if "channel" in data:
                        await self.handle_notification(data["channel"], data["notification"])
                    elif "channel_name" in data:
                        # New API format with channel_name
                        await self.handle_notification(data["channel_name"], data["notification"])
                    else:
                        self.logger.warning(f"Received notification with unknown channel format: {data.keys()}")
                # Process response to API call
                elif "id" in data:
                    cid = data["id"]
                    if "result" in data:
                        await self.handle_result(data["result"], cid)
                    elif "error" in data:
                        await self.handle_error(data["error"], cid)
                else:
                    self.logger.warning(f"Received unrecognized message format: {data.keys()}")
                
                # Track message processing time
                processing_time = time.time() - start_time
                self.message_processing_times[self.message_processing_idx] = processing_time
                self.message_processing_idx = (self.message_processing_idx + 1) % 1000
                
                # Log if processing took too long
                if processing_time > 0.1:  # More than 100ms
                    self.logger.warning(f"Message processing took {processing_time*1000:.2f}ms")
                
            except asyncio.CancelledError:
                self.logger.info("Listen task cancelled, exiting...")
                break
            except (ConnectionError, websockets.exceptions.ConnectionClosed) as conn_err:
                consecutive_errors += 1
                self.logger.error(f"Connection error in listen task (attempt {consecutive_errors}): {str(conn_err)}")
                # Use exponential backoff for connection errors
                delay = min(base_error_delay * (2 ** consecutive_errors), max_error_delay)
                self.logger.info(f"Waiting {delay:.2f}s before reconnection attempt")
                await asyncio.sleep(delay)
            except Exception as e:
                consecutive_errors += 1
                self.logger.error(f"Error in listen task (attempt {consecutive_errors}): {str(e)}", exc_info=True)
                
                # If too many consecutive errors, implement exponential backoff
                if consecutive_errors > max_consecutive_errors:
                    delay = min(base_error_delay * (2 ** (consecutive_errors - max_consecutive_errors)), max_error_delay)
                    self.logger.warning(f"Too many consecutive errors ({consecutive_errors}), backing off for {delay:.2f}s")
                    await asyncio.sleep(delay)
                else:
                    await asyncio.sleep(base_error_delay)  # Basic delay on occasional errors

    async def handle_notification(self, channel: str, notification: Union[Dict, List[Dict]]):
        """Handle incoming notifications from different channels.
        
        Args:
            channel: The notification channel name
            notification: The notification payload
        """
        try:
            if not isinstance(notification, (dict, list)):
                self.logger.error(f"Invalid notification format: {type(notification)}")
                return
            
            self.logger.info(f"Handling notification from channel: {channel}")
            
            # Handle different channel types - Support both old and new formats
            if channel.startswith("ticker."):
                self.logger.info(f"Processing ticker update for {channel}")
                await self.handle_ticker_update(notification)
            elif channel.startswith("price_index."):
                self.logger.info(f"Processing price index update")
                await self.handle_index_update(notification)
            elif channel == "session.orders":
                self.logger.info(f"Processing order update")
                if isinstance(notification, list):
                    for order_data in notification:
                        await self.handle_order_update(order_data)
                else:
                    await self.handle_order_update(notification)
            elif channel == "account.portfolio":
                self.logger.info(f"Processing portfolio update")
                await self.handle_portfolio_update(notification)
            elif channel == "account.trade_history":
                self.logger.info(f"Processing trade update")
                await self.handle_trade_update(notification)
            else:
                self.logger.warning(f"Notification for unknown channel: {channel}")
        except Exception as e:
            self.logger.error(f"Error processing notification: {str(e)}", exc_info=True)

    async def handle_ticker_update(self, ticker_data: Dict):
        """Handle a ticker update notification"""
        try:
            # Extract needed values
            instrument_name = ticker_data.get("instrument_name", "")
            mark_price = ticker_data.get("mark_price", 0.0)
            index_price = ticker_data.get("index_price", 0.0)
            bid_price = ticker_data.get("best_bid_price", 0.0)
            ask_price = ticker_data.get("best_ask_price", 0.0)
            timestamp = ticker_data.get("timestamp", 0) / 1e6  # Convert to seconds
            
            # Validate prices - critical for quote generation
            if mark_price <= 0:
                self.logger.warning(f"Invalid mark price received: {mark_price}, skipping update")
                return
                
            if bid_price <= 0 or ask_price <= 0 or bid_price >= ask_price:
                self.logger.warning(f"Invalid bid/ask prices: bid={bid_price}, ask={ask_price}")
                # If bid/ask are invalid but mark price is valid, we can still continue
                # Just log the warning and proceed
            
            # Make sure prices are aligned to tick size if tick size is available
            if hasattr(self, 'tick_size') and self.tick_size > 0:
                # Round prices to tick size
                if mark_price > 0:
                    mark_price = round(mark_price / self.tick_size) * self.tick_size
                if bid_price > 0:
                    bid_price = round(bid_price / self.tick_size) * self.tick_size
                if ask_price > 0:
                    ask_price = round(ask_price / self.tick_size) * self.tick_size
                if index_price > 0:
                    index_price = round(index_price / self.tick_size) * self.tick_size
                    
                self.logger.debug(f"Aligned prices to tick size {self.tick_size}: mark={mark_price}, bid={bid_price}, ask={ask_price}")
            
            # Update market data buffer using the proper update method
            self.market_data.update(
                price=mark_price,
                volume=0.0,  # Ticker updates don't have volume data
                timestamp=int(timestamp * 1000),  # Convert back to milliseconds for buffer
                is_buy=None  # No buy/sell direction in ticker data
            )
            
            self.last_ticker_time = timestamp
            
            # Check if we have a ticker to update
            if self.ticker is None:
                # Create a new ticker with the expected dictionary format
                self.ticker = Ticker({
                    "mark_price": mark_price,
                    "best_bid_price": bid_price,
                    "best_ask_price": ask_price,
                    "index": index_price,
                    "mark_timestamp": timestamp,
                    "funding_rate": 0.0  # Default value
                })
                self.logger.info(f"Created initial ticker: mark={mark_price}, bid={bid_price}, ask={ask_price}")
            else:
                # Update existing ticker (more efficient)
                old_mark = self.ticker.mark_price
                self.ticker.mark_price = mark_price
                self.ticker.best_bid_price = bid_price
                self.ticker.best_ask_price = ask_price
                self.ticker.index = index_price
                self.ticker.mark_ts = timestamp
                
                # Log significant price changes
                if abs(old_mark - mark_price) > (self.tick_size * 10):  # If price moved more than 10 ticks
                    self.logger.info(f"Significant price change: {old_mark} -> {mark_price} (Δ: {mark_price - old_mark})")
            
            # Add to price history using NumPy array for better performance
            self.price_history[self.price_history_idx] = mark_price
            self.price_history_idx = (self.price_history_idx + 1) % 100
            if self.price_history_idx == 0:
                self.price_history_full = True
            
            # Check for volatile market conditions
            if len(self.price_history) >= 5:
                mean_price = np.mean(self.price_history[max(0, self.price_history_idx - 5):self.price_history_idx])
                std_dev = np.std(self.price_history[max(0, self.price_history_idx - 5):self.price_history_idx])
                volatility = std_dev / mean_price if mean_price > 0 else 0.0
                volatility_pct = volatility * 100  # Convert to percentage
                
                # Use volatility threshold with fallback to default value
                volatility_threshold = RISK_LIMITS.get("volatility_threshold", 0.05)
                is_volatile = volatility_pct > volatility_threshold
                
                if is_volatile and not self.volatile_market_warning_sent:
                    self.logger.warning(f"Volatile market detected: {volatility_pct:.2f}% > {volatility_threshold:.2f}%")
                    self.volatile_market_warning_sent = True
                elif not is_volatile:
                    self.volatile_market_warning_sent = False
            
            # Notify quote task of price update
            async with self.quote_cv:
                self.condition_met = True
                self.quote_cv.notify()
            
        except Exception as e:
            self.logger.error(f"Error processing ticker update: {str(e)}", exc_info=True)

    async def handle_index_update(self, index_data: Dict):
        """Process index price updates"""
        try:
            self.index = float(index_data["price"])
            async with self.quote_cv:
                self.quote_cv.notify()
        except Exception as e:
            self.logger.error(f"Error handling index update: {str(e)}")

    async def handle_order_update(self, order_data: Dict):
        """Process order updates and trigger quote updates if necessary"""
        try:
            # Extract order info
            order = self.create_order_from_data(order_data)
            
            # Update order manager with the new order info
            if hasattr(self, 'order_manager') and self.order_manager is not None:
                await self.order_manager.update_order(order)
            
            # Handle fills
            if order_data.get("status") == "filled":
                # Log the fill with improved formatting
                self.logger.info(
                    f"Order filled: {order.id} - {order.direction.upper()} {order.amount:.4f} @ {order.price:.2f}"
                )
                
                # Update market maker with fill information
                self.market_maker.on_order_filled(
                    order_id=order.id,
                    fill_price=order.price,
                    fill_size=order.amount,
                    is_buy=order.direction == "buy"
                )
                
                # Update VAMP with order fill data if supported
                if hasattr(self.market_maker, 'update_vamp'):
                    self.market_maker.update_vamp(
                        price=order.price,
                        volume=order.amount,
                        is_buy=order.direction == "buy",
                        is_aggressive=False  # Assuming most of our fills are passive
                    )
                
                # Force quote update after fills based on size threshold
                significant_fill_threshold = TRADING_CONFIG["avellaneda"].get("significant_fill_threshold", 0.1)
                
                # Create comprehensive market conditions
                market_state = self.market_data.get_market_state()
                market_conditions = {
                    "volatility": market_state["yz_volatility"] if "yz_volatility" in market_state and not np.isnan(market_state["yz_volatility"]) else 
                                 (market_state["volatility"] if not np.isnan(market_state["volatility"]) else TRADING_CONFIG["avellaneda"]["fixed_volatility"]),
                    "market_impact": market_state.get("market_impact", 0.0),
                    # Add higher impact for a recent fill
                    "fill_impact": 0.5 if order.amount >= significant_fill_threshold else 0.2  # Higher impact for significant fills
                }
                
                if self.ticker:
                    if order.amount >= significant_fill_threshold:
                        # Create task with high priority for immediate update for significant fills
                        self.logger.info(f"Scheduling immediate quote update after significant fill of {order.amount:.4f}")
                        asyncio.create_task(self.update_quotes(None, market_conditions))
                    else:
                        # Still update quotes for smaller fills, but with less urgency
                        self.logger.info(f"Scheduling quote update after fill of {order.amount:.4f}")
                        # Notify quote task instead of creating a separate task
                        async with self.quote_cv:
                            self.condition_met = True
                            self.quote_cv.notify()
                else:
                    self.logger.warning("Cannot update quotes after fill: No ticker data available")
                
        except Exception as e:
            self.logger.error(f"Error handling order update: {str(e)}", exc_info=True)

    def create_order_from_data(self, data: Dict) -> Order:
        """Create an Order object from API response data"""
        try:
            # Extract essential fields with defaults
            order_id = data.get("client_order_id", data.get("order_id", 0))
            price = float(data.get("price", 0))
            amount = float(data.get("amount", 0))
            status_str = data.get("status", "pending")
            direction = data.get("direction", "")
            
            # Convert status string to OrderStatus enum
            try:
                status = OrderStatus(status_str)
            except (ValueError, KeyError):
                self.logger.warning(f"Unknown order status: {status_str}, using PENDING")
                status = OrderStatus.PENDING
                
            # Create and return Order object
            return Order(
                id=order_id,
                price=price,
                amount=amount,
                status=status,
                direction=direction
            )
        except Exception as e:
            self.logger.error(f"Error creating order from data: {str(e)}")
            # Return a minimal valid order to prevent further errors
            return Order(id=0, price=0, amount=0)

    async def handle_portfolio_update(self, portfolio_data: List[Dict]):
        """Handle a portfolio update notification"""
        try:
            self.logger.info(f"Processing portfolio update")
            
            # Default position size
            position_size = 0.0
            
            # Iterate through portfolio items
            for item in portfolio_data:
                # Extract asset data - ensure it's a dictionary before using .get()
                asset = item.get("asset", {})
                if asset and isinstance(asset, dict):  # Check if asset is a dictionary
                    asset_name = asset.get("asset_name", "")
                    amount = asset.get("amount", 0.0)
                    
                    # Store in portfolio
                    self.portfolio[asset_name] = float(amount)
                    
                    self.logger.debug(f"Updated asset {asset_name}: {float(amount):.8f}")
                elif asset and not isinstance(asset, dict):  # Handle case where asset is not a dictionary
                    # Log the unexpected format but don't crash
                    self.logger.warning(f"Received asset in unexpected format: {type(asset)}, value: {asset}")
                
                # Extract position data
                position = item.get("position", {})
                if not position:
                    continue
                    
                # Ensure position is a dictionary before using .get()
                if not isinstance(position, dict):
                    self.logger.warning(f"Received position in unexpected format: {type(position)}, value: {position}")
                    continue
                    
                instrument = position.get("instrument_name", "")
                
                # Only process our target instrument
                if self.perp_name and instrument == self.perp_name:
                    # Safely extract values with defaults
                    size = float(position.get('size', 0.0) or 0.0)
                    avg_price = float(position.get('average_price', 0.0) or 0.0)
                    
                    # Update position tracking
                    position_size = size
                    position_value = abs(size * avg_price) if avg_price > 0 else 0.0
                    
                    # Update the risk manager with position information - use async call properly
                    await self.risk_manager.update_position(
                        size=size,
                        price=avg_price,
                        timestamp=time.time()
                    )
                    
                    self.logger.info(
                        f"Updated position for {instrument}: "
                        f"Size={size:.8f}, "
                        f"Avg Price={avg_price:.2f}, "
                        f"Value={position_value:.2f}"
                    )
                    
            # Update market maker with position information for inventory management
            if hasattr(self.market_maker, 'update_position'):
                self.market_maker.update_position(position_size, avg_price)
                
        except Exception as e:
            self.logger.error(f"Error handling portfolio update: {str(e)}")
            
        # Trigger quote update after position change
        async with self.quote_cv:
            self.condition_met = True
            self.quote_cv.notify()

    async def handle_trade_update(self, trade_data: Dict):
        """
        Process trade updates
        
        Args:
            trade_data: Trade data from the exchange
        """
        try:
            # Check if trade_data is a list and handle accordingly
            if isinstance(trade_data, list):
                for trade in trade_data:
                    await self._process_single_trade(trade)
            else:
                await self._process_single_trade(trade_data)
                
        except Exception as e:
            self.logger.error(f"Error handling trade update: {str(e)}", exc_info=True)
            
    async def _process_single_trade(self, trade: Dict):
        """Process a single trade update"""
        try:
            # Extract trade details
            order_id = trade.get("order_id", "")
            instrument = trade.get("instrument_name", "")
            direction = trade.get("direction", "")
            price = float(trade.get("price", 0))
            amount = float(trade.get("amount", 0))
            timestamp = trade.get("timestamp", 0)
            
            # Log the trade
            self.logger.info(
                f"Trade executed: {direction.upper()} {amount:.4f} @ {price:.2f} "
                f"(Order ID: {order_id})"
            )
            
            # Update market maker with fill information
            if self.market_maker:
                is_buy = direction.lower() == "buy"
                self.market_maker.on_order_filled(
                    order_id=order_id,
                    fill_price=price,
                    fill_size=amount,
                    is_buy=is_buy
                )
                
            # Force a quote update after significant fills
            # This ensures we react quickly to executed trades
            if self.ticker and amount >= 0.01:  # Only for fills >= 0.01 BTC
                self.logger.info(f"Scheduling quote update after trade of {amount:.4f}")
                asyncio.create_task(self.update_quotes(None, self.get_market_conditions()))
                
        except Exception as e:
            self.logger.error(f"Error processing trade: {str(e)}", exc_info=True)

    async def handle_result(self, result: Dict, cid: Optional[int]):
        """Handle API call results"""
        try:
            if cid is None:
                return
                
            if cid == CALL_IDS["login"]:
                self.logger.info("Login successful")
            elif cid == CALL_IDS["subscribe"]:
                self.logger.info("Subscription successful")
            elif cid == CALL_IDS["mass_quote"]:
                # Note: We're using individual limit orders now, but keeping this handler for backward compatibility
                self.logger.info("Mass quote request successful (legacy handler)")
                # Update the order manager with the mass quote result
                if hasattr(self, 'order_manager') and self.order_manager is not None:
                    await self.order_manager.handle_order_result(result, cid)
            elif cid >= 100:  # Other order IDs
                await self.order_manager.handle_order_result(result, cid)
                
        except Exception as e:
            self.logger.error(f"Error handling result: {str(e)}")

    async def handle_error(self, error: Dict, cid: Optional[int]):
        """Handle API errors"""
        try:
            if cid is None:
                return
                
            # API call error handling
            if cid == CALL_IDS["login"]:
                self.logger.error(f"Login failed: {error.get('message', '')}")
            elif cid == CALL_IDS["subscribe"]:
                self.logger.error(f"Subscription failed: {error.get('message', '')}")
            elif cid == CALL_IDS["mass_quote"]:
                # Note: We're using individual limit orders now, but keeping this handler for backward compatibility
                self.logger.error(f"Mass quote error (legacy handler): {error.get('message', '')}")
                # Pass to order manager for handling
                if hasattr(self, 'order_manager') and self.order_manager is not None:
                    await self.order_manager.handle_order_error(error, cid)
            elif cid >= 100:  # Order-related errors
                await self.order_manager.handle_order_error(error, cid)
            else:
                # Generic API error
                self.logger.error(f"API error for message {cid}: {error}")
                
        except Exception as e:
            self.logger.error(f"Error handling error message: {str(e)}")

    async def update_quotes(self, price_data: Dict, market_conditions: Dict):
        """
        Update quotes based on new market data
        
        Args:
            price_data: Current ticker/price data
            market_conditions: Current market conditions (volatility, market impact)
        """
        try:
            if not self.market_maker:
                self.logger.warning("Cannot update quotes: Market maker not initialized")
                return
                
            # Create ticker from price data if provided, otherwise use the stored ticker
            ticker = None
            if price_data:
                # Make sure the price_data has the expected keys for Ticker
                ticker_data = {
                    "mark_price": price_data.get("mark_price", 0.0),
                    "best_bid_price": price_data.get("best_bid_price", price_data.get("bid_price", 0.0)),
                    "best_ask_price": price_data.get("best_ask_price", price_data.get("ask_price", 0.0)),
                    "index": price_data.get("index_price", price_data.get("index", 0.0)),
                    "mark_timestamp": price_data.get("timestamp", price_data.get("mark_timestamp", time.time())),
                    "funding_rate": price_data.get("funding_rate", 0.0)
                }
                ticker = Ticker(ticker_data)
            else:
                ticker = self.ticker
                
            if not ticker or not ticker.mark_price or ticker.mark_price <= 0:
                self.logger.warning("Cannot update quotes: Invalid ticker data")
                return
                
            # Rate limiting - check time since last update
            current_time = time.time()
            if hasattr(self, 'last_quote_update_time'):
                time_since_last = current_time - self.last_quote_update_time
                min_update_interval = 1.0  # Minimum 1 second between quote updates
                if time_since_last < min_update_interval:
                    self.logger.info(f"Skipping quote update - too soon since last update ({time_since_last:.1f}s < {min_update_interval:.1f}s)")
                    return
                    
            # Check if we have any active orders in the order manager
            active_orders_count = len(self.order_manager.active_bids) + len(self.order_manager.active_asks)
            
            # Reset current_quotes if no active orders exist in order manager
            if active_orders_count == 0:
                self.current_quotes = ([], [])
                    
            # Check if active quotes need updating
            if len(self.current_quotes[0]) > 0 or len(self.current_quotes[1]) > 0:
                # If we have active quotes, check if they need to be updated
                should_update = self.market_maker.should_update_quotes(self.current_quotes, ticker.mark_price)
                if not should_update:
                    self.logger.info("Skipping quote update - current quotes still valid")
                    return
                
            # Generate quotes using market maker
            self.logger.info("Generating new quotes...")
            bid_quotes, ask_quotes = self.market_maker.generate_quotes(ticker, market_conditions)
            
            if not bid_quotes and not ask_quotes:
                self.logger.warning("No valid quotes generated")
                return
            
            # Validate quotes before sending
            bid_quotes, ask_quotes = self.market_maker.validate_quotes(bid_quotes, ask_quotes)
            
            # Extra validation after market maker validation
            if self.tick_size > 0:
                # Check and align all quote prices to tick size
                aligned_bid_quotes = []
                for quote in bid_quotes:
                    aligned_price = round(quote.price / self.tick_size) * self.tick_size
                    if aligned_price != quote.price:
                        # Create a new quote with the aligned price
                        aligned_quote = Quote(
                            price=aligned_price,
                            amount=quote.amount,
                            instrument=quote.instrument,
                            side=quote.side,
                            timestamp=quote.timestamp
                        )
                        aligned_bid_quotes.append(aligned_quote)
                        self.logger.debug(f"Aligned bid price from {quote.price} to {aligned_price}")
                    else:
                        aligned_bid_quotes.append(quote)
                
                aligned_ask_quotes = []
                for quote in ask_quotes:
                    aligned_price = round(quote.price / self.tick_size) * self.tick_size
                    if aligned_price != quote.price:
                        # Create a new quote with the aligned price
                        aligned_quote = Quote(
                            price=aligned_price,
                            amount=quote.amount,
                            instrument=quote.instrument,
                            side=quote.side,
                            timestamp=quote.timestamp
                        )
                        aligned_ask_quotes.append(aligned_quote)
                        self.logger.debug(f"Aligned ask price from {quote.price} to {aligned_price}")
                    else:
                        aligned_ask_quotes.append(quote)
                
                # Replace with aligned quotes
                bid_quotes = aligned_bid_quotes
                ask_quotes = aligned_ask_quotes
            
            # Verify quote prices are valid
            valid_bid_quotes = []
            for quote in bid_quotes:
                if quote.price <= 0 or quote.amount <= 0:
                    self.logger.warning(f"Invalid bid quote: price={quote.price}, amount={quote.amount}")
                    continue
                valid_bid_quotes.append(quote)
            
            valid_ask_quotes = []
            for quote in ask_quotes:
                if quote.price <= 0 or quote.amount <= 0:
                    self.logger.warning(f"Invalid ask quote: price={quote.price}, amount={quote.amount}")
                    continue
                valid_ask_quotes.append(quote)
            
            # Check if we have any valid quotes left
            if not valid_bid_quotes and not valid_ask_quotes:
                self.logger.error("No valid quotes after price alignment, skipping update")
                return
                
            # Replace with validated quotes
            bid_quotes = valid_bid_quotes
            ask_quotes = valid_ask_quotes
            
            # Always update quotes if it's been too long since the last update
            max_update_interval = 60.0  # Force update every 60 seconds 
            if hasattr(self, 'last_quote_update_time') and current_time - self.last_quote_update_time >= max_update_interval:
                self.logger.info(f"Forcing quote update - {current_time - self.last_quote_update_time:.1f}s elapsed since last update")
                # Cancel existing quotes
                await self.cancel_quotes()
                # Brief pause to allow cancellations to process
                await asyncio.sleep(0.5)
                # Place new quotes
                await self.place_quotes(bid_quotes, ask_quotes)
                # Update last quote time
                self.last_quote_time = current_time
                self.last_quote_update_time = current_time
                return
            
            # Check if order manager has active quotes before comparing prices
            if active_orders_count == 0:
                # No active orders in the order manager, force update
                self.logger.info("No active orders in order manager - forcing quote placement")
                await self.place_quotes(bid_quotes, ask_quotes)
                self.last_quote_time = current_time
                self.last_quote_update_time = current_time
                return
                
            # Check if cancellation is really needed
            should_cancel = True
            
            # Only do price comparison check if there are enough quotes on both sides
            bid_prices = [q.price for q in bid_quotes]
            ask_prices = [q.price for q in ask_quotes]
            current_bid_prices = [q.price for q in self.current_quotes[0]]  
            current_ask_prices = [q.price for q in self.current_quotes[1]]
            
            if (len(current_bid_prices) >= 3 and len(current_ask_prices) >= 3 and 
                len(bid_prices) >= 3 and len(ask_prices) >= 3):
                # Compare all quotes (not just the first level)
                # Check first 3 levels which are most important
                price_tolerance = 1.0  # Reduced from 2.0 to 1.0 - even more sensitive to changes
                
                # Count how many levels have significant changes
                changed_levels = 0
                
                # Check the first 3 levels of bids
                for i in range(min(3, len(bid_prices))):
                    if i < len(current_bid_prices):
                        if abs(bid_prices[i] - current_bid_prices[i]) > price_tolerance:
                            changed_levels += 1
                
                # Check the first 3 levels of asks
                for i in range(min(3, len(ask_prices))):
                    if i < len(current_ask_prices):
                        if abs(ask_prices[i] - current_ask_prices[i]) > price_tolerance:
                            changed_levels += 1
                
                # Only skip update if less than 2 levels have significant changes
                if changed_levels < 2:
                    self.logger.info(f"Current quotes are close to desired prices (only {changed_levels} levels changed) - skipping update")
                    self.last_quote_update_time = current_time
                    return
                else:
                    self.logger.info(f"Updating quotes - {changed_levels} levels have significant price changes")
            
            # Cancel existing quotes if needed
            if should_cancel:
                self.logger.info("Cancelling existing quotes...")
                await self.cancel_quotes()
                
                # Brief pause to allow cancellations to process
                await asyncio.sleep(0.5)
                
                # Place new quotes
                self.logger.info("Placing new quotes...")
                await self.place_quotes(bid_quotes, ask_quotes)
            
            # Update last quote time
            self.last_quote_time = current_time
            self.last_quote_update_time = current_time
                
            # Record successful quote placement
            self.last_quote_time = time.time()
            
            # Log quote summary
            if bid_quotes or ask_quotes:
                bid_count = len(bid_quotes)
                ask_count = len(ask_quotes)
                self.logger.info(f"Sending limit orders: {bid_count} bids, {ask_count} asks")
                self.logger.info(f"Limit orders sent: {bid_count} bids, {ask_count} asks")
                
            # Verify order counts after placement to detect any potential tracking issues
            await asyncio.sleep(0.5)  # Brief pause to allow orders to be processed
            current_active_bids = len(self.order_manager.active_bids)
            current_active_asks = len(self.order_manager.active_asks)
            max_levels = TRADING_CONFIG["quoting"].get("levels", 6)
            
            if current_active_bids > max_levels or current_active_asks > max_levels:
                self.logger.error(f"Order tracking issue detected! Active orders exceed max levels: {current_active_bids} bids, {current_active_asks} asks (max: {max_levels})")
                self.logger.info(f"Initiating emergency cancellation to restore order consistency")
                await self.cancel_quotes("Order tracking issue detected")
                
        except Exception as e:
            self.logger.error(f"Error updating quotes: {str(e)}", exc_info=True)
            # On error, try to cancel all quotes for safety
            try:
                await self.cancel_quotes()
            except Exception as cancel_error:
                self.logger.error(f"Error cancelling quotes after error: {str(cancel_error)}")

    async def cancel_quotes(self, reason="Unknown"):
        """Cancel all outstanding quotes"""
        if not self.quoting_enabled:
            self.logger.debug("Quoting disabled, not cancelling quotes")
            return
            
        try:
            # Cancel all orders for our instrument
            self.logger.info(f"Cancelling all orders for {self.perp_name} - reason: {reason}")
            
            # Use the order_manager's cancel_all_orders method without parameters
            await self.order_manager.cancel_all_orders()
            
            # Clear current quotes
            self.current_quotes = ([], [])
            
        except Exception as e:
            self.logger.error(f"Error cancelling quotes: {str(e)}", exc_info=True)

    async def quote_task(self):
        """Main quoting loop - periodically checks if quotes need updating"""
        self.logger.info("Quote task started - waiting for updates")
        
        last_quote_time = 0
        min_quote_interval = 1.0  # Reduced from 3.0 to 1.0 seconds for more frequent updates
        
        while True:
            try:
                # Wait for price updates
                self.logger.debug("Quote task waiting for notification")
                async with self.quote_cv:
                    await self.quote_cv.wait()
                
                self.logger.info("Quote task received notification")
                current_time = time.time()
                
                # Enforce minimum interval between quote updates
                if current_time - last_quote_time < min_quote_interval:
                    self.logger.info(f"Quote update too frequent - waiting {min_quote_interval}s between updates")
                    continue
                    
                # Check if we're in cooldown
                if self.cooldown_active and current_time < self.cooldown_until:
                    self.logger.info(f"In cooldown period - {(self.cooldown_until - current_time):.1f}s remaining")
                    continue
                
                # Skip if no ticker data
                if not self.ticker:
                    self.logger.warning("Cannot quote - no ticker data available")
                    continue
                
                # Only proceed if we actually need to update quotes
                if not self.market_maker.should_update_quotes(self.current_quotes, self.ticker.mark_price):
                    self.logger.info("Market maker indicates no quote update needed")
                    continue
                
                # Check if we have enough price history
                min_samples = TRADING_PARAMS["volatility"]["min_samples"]
                if len(self.price_history) < min_samples:
                    self.logger.info(f"Need more price history before quoting: {len(self.price_history)}/{min_samples}")
                    continue
                
                # Get market state and conditions
                market_state = self.market_data.get_market_state()
                market_conditions = {
                    "volatility": market_state["yz_volatility"] if "yz_volatility" in market_state and not np.isnan(market_state["yz_volatility"]) else 
                                 (market_state["volatility"] if not np.isnan(market_state["volatility"]) else TRADING_CONFIG["avellaneda"]["fixed_volatility"]),
                    "market_impact": market_state.get("market_impact", 0.0)
                }
                        
                # Check rate limit before attempting quote update
                if not await self.check_rate_limit(priority="high"):
                    self.logger.warning("Rate limit reached - delaying quote update")
                    # Set a cooldown period to prevent hammering the API
                    self.cooldown_active = True
                    self.cooldown_until = current_time + 5.0  # 5 second cooldown
                    continue
                
                # Update quotes
                self.logger.info("Generating new quotes")
                await self.update_quotes(None, market_conditions)
                last_quote_time = current_time
                self.logger.info("Quote update completed")
                
                # Add a delay after successfully updating quotes to prevent excessive updates
                await asyncio.sleep(1.0)
                    
            except Exception as e:
                self.logger.error(f"Error in quote task: {str(e)}", exc_info=True)
                await asyncio.sleep(1)

    async def heartbeat_task(self):
        """Periodic heartbeat to ensure connection is alive"""
        self.logger.info("Starting heartbeat task")
        
        while True:
            try:
                await self.send_heartbeat()
                
                # Add a small random jitter (up to ±20%) to prevent heartbeats from bunching up with other operations
                jitter = random.uniform(0.8, 1.2)
                wait_time = self.heartbeat_interval * jitter
                
                await asyncio.sleep(wait_time)
            except Exception as e:
                self.logger.error(f"Error in heartbeat task: {str(e)}")
                await asyncio.sleep(5)  # Brief pause after error
                
    async def send_heartbeat(self):
        """Send a heartbeat message to keep the connection alive"""
        try:
            # Check if connection is alive before attempting heartbeat
            if hasattr(self.thalex, 'connected') and callable(self.thalex.connected) and not self.thalex.connected():
                self.logger.warning("Connection lost before heartbeat, attempting to reconnect...")
                try:
                    await self.thalex.connect()
                    self.logger.info("Successfully reconnected in heartbeat task")
                except Exception as reconnect_err:
                    self.logger.error(f"Failed to reconnect in heartbeat task: {str(reconnect_err)}")
                    return False
            
            # Try different methods based on what's available in the client
            heartbeat_sent = False
            err_msg = None
            
            # Method 1: Try to use the ping method if available
            if not heartbeat_sent and hasattr(self.thalex, 'ping') and callable(self.thalex.ping):
                try:
                    await self.thalex.ping(id=CALL_IDS.get("heartbeat", 1010))
                    heartbeat_sent = True
                except Exception as e:
                    err_msg = f"Ping method failed: {str(e)}"
            
            # Method 2: Try to use the _send method with public/ping
            if not heartbeat_sent and hasattr(self.thalex, '_send') and callable(self.thalex._send):
                try:
                    await self.thalex._send("public/ping", id=CALL_IDS.get("heartbeat", 1010))
                    heartbeat_sent = True
                except Exception as e:
                    if err_msg:
                        err_msg += f"; _send method failed: {str(e)}"
                    else:
                        err_msg = f"_send method failed: {str(e)}"
            
            # Method 3: Check if ws attribute exists and try to send a ping frame directly
            if not heartbeat_sent and hasattr(self.thalex, 'ws') and self.thalex.ws:
                try:
                    # Send a WebSocket ping frame
                    pong_waiter = await self.thalex.ws.ping()
                    await asyncio.wait_for(pong_waiter, timeout=5.0)
                    heartbeat_sent = True
                except Exception as e:
                    if err_msg:
                        err_msg += f"; WebSocket ping failed: {str(e)}"
                    else:
                        err_msg = f"WebSocket ping failed: {str(e)}"
            
            if heartbeat_sent:
                self.last_heartbeat = time.time()
                self.logger.debug("Heartbeat sent successfully")
                return True
            else:
                self.logger.warning(f"All heartbeat methods failed: {err_msg}")
                return False
                
        except Exception as e:
            self.logger.warning(f"Failed to send heartbeat: {str(e)}")
            return False

    async def place_quotes(self, bid_quotes: List[Quote], ask_quotes: List[Quote]):
        """Place quotes using individual limit orders instead of mass_quote API
        
        This method deliberately uses individual limit orders rather than mass quotes
        to avoid dependency on market maker protection configuration.
        """
        if not self.quoting_enabled:
            self.logger.debug("Quoting disabled, not placing quotes")
            return
        
        if self.cooldown_active and time.time() < self.cooldown_until:
            self.logger.debug("Cooldown active, not placing quotes")
            return
            
        try:
            # Enforce max order levels from config
            max_levels = TRADING_CONFIG["quoting"].get("levels", 6)
            if len(bid_quotes) > max_levels:
                self.logger.warning(f"Limiting bid quotes to configured max levels: {max_levels}")
                bid_quotes = bid_quotes[:max_levels]
            if len(ask_quotes) > max_levels:
                self.logger.warning(f"Limiting ask quotes to configured max levels: {max_levels}")
                ask_quotes = ask_quotes[:max_levels]
            
            # Check existing orders to avoid exceeding limits
            active_bids_count = len(self.order_manager.active_bids)
            active_asks_count = len(self.order_manager.active_asks)
            
            # If we already have orders, cancel them all first to ensure clean state
            if active_bids_count > 0 or active_asks_count > 0:
                self.logger.warning(f"Found existing orders before placing quotes: {active_bids_count} bids, {active_asks_count} asks")
                self.logger.warning(f"Cancelling all existing orders to ensure clean order book")
                await self.cancel_quotes("Refreshing order book")
                # Brief pause to allow cancellations to process
                await asyncio.sleep(0.5)
                
                # Verify orders were cancelled
                if len(self.order_manager.active_bids) > 0 or len(self.order_manager.active_asks) > 0:
                    self.logger.error(f"Failed to cancel all orders. Still have {len(self.order_manager.active_bids)} bids and {len(self.order_manager.active_asks)} asks")
                    # Force another cancellation
                    await self.cancel_quotes("Emergency cancellation")
                    await asyncio.sleep(1.0)  # Longer wait time
            
            # Store current quotes before sending to properly track what was sent
            self.current_quotes = (bid_quotes, ask_quotes)
            
            # Reset tracking variables to ensure clean state
            placed_bids = 0
            placed_asks = 0
            
            # Place bid quotes as individual limit orders
            for quote in bid_quotes:
                if quote.price <= 0 or quote.amount <= 0:
                    self.logger.warning(f"Invalid bid quote detected: {quote.price}@{quote.amount} - skipping")
                    continue
                
                # Check if we reached the max limit for this side
                if placed_bids >= max_levels:
                    self.logger.warning(f"Maximum bid levels ({max_levels}) reached - skipping remaining bids")
                    break
                    
                # Use the order manager to place a limit order
                try:
                    await self.order_manager.place_order(
                        instrument=self.perp_name,
                        direction="buy",
                        price=quote.price,
                        amount=quote.amount,
                        label="AvellanedaQuoter",
                        post_only=True
                    )
                    self.logger.info(f"Placed bid: {quote.amount:.3f}@{quote.price:.2f}")
                    placed_bids += 1
                except Exception as e:
                    self.logger.error(f"Error placing bid: {str(e)}")
            
            # Place ask quotes as individual limit orders
            for quote in ask_quotes:
                if quote.price <= 0 or quote.amount <= 0:
                    self.logger.warning(f"Invalid ask quote detected: {quote.price}@{quote.amount} - skipping")
                    continue
                
                # Check if we reached the max limit for this side
                if placed_asks >= max_levels:
                    self.logger.warning(f"Maximum ask levels ({max_levels}) reached - skipping remaining asks")
                    break
                    
                # Use the order manager to place a limit order
                try:
                    await self.order_manager.place_order(
                        instrument=self.perp_name,
                        direction="sell",
                        price=quote.price,
                        amount=quote.amount,
                        label="AvellanedaQuoter",
                        post_only=True
                    )
                    self.logger.info(f"Placed ask: {quote.amount:.3f}@{quote.price:.2f}")
                    placed_asks += 1
                except Exception as e:
                    self.logger.error(f"Error placing ask: {str(e)}")
            
            # Record successful quote placement
            self.last_quote_time = time.time()
            
            # Log quote summary
            self.logger.info(f"Sending limit orders: {placed_bids} bids, {placed_asks} asks")
            self.logger.info(f"Limit orders sent: {placed_bids} bids, {placed_asks} asks")
                
            # Verify order counts after placement to detect any potential tracking issues
            await asyncio.sleep(0.5)  # Brief pause to allow orders to be processed
            current_active_bids = len(self.order_manager.active_bids)
            current_active_asks = len(self.order_manager.active_asks)
            
            if current_active_bids > max_levels or current_active_asks > max_levels:
                self.logger.error(f"Order tracking issue detected! Active orders exceed max levels: {current_active_bids} bids, {current_active_asks} asks (max: {max_levels})")
                self.logger.info(f"Initiating emergency cancellation to restore order consistency")
                await self.cancel_quotes("Order tracking issue detected")
            
        except Exception as e:
            self.logger.error(f"Error placing quotes: {str(e)}", exc_info=True)
            
            # If rate limit exceeded, implement cooldown
            if "rate limit" in str(e).lower() or "circuit breaker" in str(e).lower():
                self.cooldown_active = True
                self.cooldown_until = time.time() + 5  # 5 second cooldown
                self.logger.warning(f"Rate limit exceeded, enabling cooldown until {self.cooldown_until}")
            
            # Clear current quotes on error to prevent stale state
            self.current_quotes = ([], [])

    def get_market_conditions(self) -> Dict:
        """Get the current market conditions from market data buffer"""
        market_state = self.market_data.get_market_state()
        
        # Get volatility - prefer Yang-Zhang volatility when available
        if "yz_volatility" in market_state and not np.isnan(market_state["yz_volatility"]):
            volatility = market_state["yz_volatility"]
        else:
            volatility = market_state["volatility"] if not np.isnan(market_state["volatility"]) else TRADING_CONFIG["avellaneda"]["fixed_volatility"]
        
        return {
            "volatility": volatility,
            "market_impact": market_state.get("market_impact", 0.0)
        }

    async def gamma_update_task(self):
        """Periodically updates the gamma value based on market conditions"""
        self.logger.info("Gamma update task started - will update gamma dynamically based on market conditions")
        
        # Wait for initial data availability
        while not self.ticker or len(self.price_history) < TRADING_PARAMS["volatility"]["min_samples"]:
            self.logger.info("Waiting for market data before starting gamma updates")
            await asyncio.sleep(10)
        
        self.logger.info("Gamma update task is now active - market data sufficient")
        
        # Main update loop
        counter = 0
        # Default update interval is 60 seconds
        update_interval = 60
        # Flags for tracking market conditions
        high_volatility = False
        high_market_impact = False
        
        while True:
            try:
                # Determine the appropriate update interval based on market conditions
                if high_volatility or high_market_impact:
                    # More frequent updates during volatile conditions (15 seconds)
                    update_interval = 15
                    self.logger.info(f"Gamma update task: using accelerated update interval of {update_interval}s due to market conditions")
                else:
                    # Standard update interval during normal conditions (60 seconds)
                    update_interval = 60
                
                # Log that we're about to sleep
                self.logger.info(f"Gamma update task: waiting for next update cycle ({counter}) - interval: {update_interval}s")
                
                # Sleep for the determined interval between updates
                await asyncio.sleep(update_interval)
                
                counter += 1
                self.logger.info(f"Gamma update task: woke up for update cycle {counter}")
                
                # Skip if no ticker data or insufficient price history
                if not self.ticker or len(self.price_history) < TRADING_PARAMS["volatility"]["min_samples"]:
                    self.logger.warning("Cannot update gamma - insufficient market data")
                    continue
                
                # Get current market conditions
                market_state = self.market_data.get_market_state()
                self.logger.info(f"Gamma update task: retrieved market state: {market_state.keys()}")
                
                # Get volatility from market data buffer
                if "yz_volatility" in market_state and not np.isnan(market_state["yz_volatility"]):
                    volatility = market_state["yz_volatility"]
                else:
                    volatility = market_state["volatility"] if not np.isnan(market_state["volatility"]) else TRADING_CONFIG["avellaneda"]["fixed_volatility"]
                
                # Get market impact
                market_impact = market_state.get("market_impact", 0.0)
                
                # Check if we're in high volatility or high market impact conditions
                high_volatility = volatility > TRADING_CONFIG["volatility"]["ceiling"] * 0.7
                high_market_impact = market_impact > TRADING_CONFIG["avellaneda"]["adverse_selection_threshold"]
                
                # Log market condition assessment
                if high_volatility or high_market_impact:
                    self.logger.warning(
                        f"Gamma update task: Detected volatile market conditions - "
                        f"volatility: {volatility:.5f} (threshold: {TRADING_CONFIG['volatility']['ceiling'] * 0.7:.5f}), "
                        f"impact: {market_impact:.5f} (threshold: {TRADING_CONFIG['avellaneda']['adverse_selection_threshold']:.5f})"
                    )
                
                self.logger.info(f"Gamma update task: using volatility={volatility:.5f}, impact={market_impact:.5f}")
                
                # Calculate new optimal gamma value
                new_gamma = self.market_maker.calculate_dynamic_gamma(volatility, market_impact)
                
                # Update gamma in market maker
                old_gamma = self.market_maker.gamma
                self.market_maker.gamma = new_gamma
                
                # Calculate percentage change in gamma
                gamma_pct_change = ((new_gamma - old_gamma) / old_gamma) * 100 if old_gamma > 0 else 0
                
                # Log the gamma update with percentage change
                self.logger.info(f"Updated gamma: {old_gamma:.3f} -> {new_gamma:.3f} ({gamma_pct_change:+.1f}%) based on volatility={volatility:.5f}, impact={market_impact:.5f}")
                
                # Trigger immediate quote update if gamma changed significantly
                if abs(gamma_pct_change) > 10:  # More than 10% change
                    self.logger.info("Significant gamma change detected - triggering immediate quote update")
                    async with self.quote_cv:
                        self.quote_cv.notify()
                else:
                    self.logger.info("Notifying quote task to generate new quotes with updated gamma")
                    async with self.quote_cv:
                        self.quote_cv.notify()
                
            except Exception as e:
                self.logger.error(f"Error updating gamma: {str(e)}", exc_info=True)
                await asyncio.sleep(5)  # Brief pause after errors

async def main():
    """Main entry point"""
    # Initialize Thalex client
    thalex = th.Thalex(network=BOT_CONFIG["market"]["network"])
    
    # Create quoter
    quoter = AvellanedaQuoter(thalex)
    
    # Set reference to quoter in thalex client for rate limit tracking
    thalex.quoter = quoter
    
    try:
        # Start the quoter
        await quoter.start()
    except KeyboardInterrupt:
        await quoter.shutdown()
    except Exception as e:
        quoter.logger.critical(f"Fatal error in main loop: {str(e)}")
        await quoter.shutdown()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass  # Handled in main()
    except Exception as e:
        print(f"Fatal error: {str(e)}")  # Fallback if logger isn't available 