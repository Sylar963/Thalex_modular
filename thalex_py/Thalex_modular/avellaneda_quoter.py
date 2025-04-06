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
from thalex_py.Thalex_modular.logging import LoggerFactory
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
        
        # Rate limiting parameters
        self.max_requests_per_minute = BOT_CONFIG["connection"]["rate_limit"]
        self.request_counter = 0
        self.request_counter_reset_time = time.time()
        self.rate_limit_window = 60
        self.rate_limit_warning_sent = False
        self.volatile_market_warning_sent = False  # Flag for tracking volatile market warnings
        
        # For coordination
        self.quote_cv = asyncio.Condition()
        self.condition_met = False
        self.setup_complete = asyncio.Event()
        
        # Connection parameters
        self.heartbeat_interval = BOT_CONFIG["connection"]["heartbeat_interval"]
        self.last_heartbeat = time.time()
        
        # Quoting management
        self.quoting_enabled = True
        self.cooldown_active = False
        self.cooldown_until = 0
        
        # Quote storage and metrics
        self.price_history = deque(maxlen=100)  # Store last 100 prices
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
        
        self.logger.info("Avellaneda quoter initialized")

    async def start(self):
        """Start the quoter and initialize logging"""
        # Initialize logging system
        await LoggerFactory.initialize()
        self.logger.info("Starting Avellaneda quoter...")
        
        try:
            # Initialize required attributes
            self.request_counter = 0
            self.request_counter_reset_time = time.time()
            self.rate_limit_warning_sent = False
            self.volatile_market_warning_sent = False
            self.cooldown_active = False
            self.cooldown_until = 0
            self.last_ticker_time = 0
            self.last_quote_time = 0
            self.max_requests_per_minute = BOT_CONFIG["connection"]["rate_limit"]
            
            # Connect to WebSocket
            self.logger.info("Connecting to WebSocket...")
            await self.thalex.connect()
            
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
            
            # Shutdown logging
            await LoggerFactory.shutdown()
            
            self.logger.info("Quoter shutdown complete")
        except Exception as e:
            self.logger.error(f"Error during shutdown: {str(e)}")

    async def await_instruments(self):
        """Initialize instrument details"""
        await self.thalex.instruments(CALL_IDS["instruments"])
        msg = await self.thalex.receive()
        msg = json.loads(msg)
        assert msg["id"] == CALL_IDS["instruments"]
        
        for i in msg["result"]:
            if i["type"] == "perpetual" and i["underlying"] == MARKET_CONFIG["underlying"]:
                tick_size = i["tick_size"]
                self.perp_name = i["instrument_name"]
                
                # Set tick size for all components
                self.order_manager.set_tick_size(tick_size)
                self.market_maker.set_tick_size(tick_size)
                self.logger.info(f"Found perpetual {self.perp_name} with tick size {tick_size}")
                return
                
        raise ValueError(f"Perpetual {MARKET_CONFIG['underlying']} not found")

    async def check_rate_limit(self, priority: str = "normal") -> bool:
        """Check if we're under the rate limit and increment counter if we are"""
        try:
            # Get current rate limit from config but reduce by 30%
            rate_limit = int(self.max_requests_per_minute * 0.7)  # Reduced by 30%
            
            # Check if we need to reset the counter (new time window)
            current_time = time.time()
            window_elapsed = current_time - self.request_counter_reset_time
            
            if window_elapsed >= self.rate_limit_window:
                # Log previous window's stats before resetting
                if self.request_counter > 0:
                    request_rate = self.request_counter / min(window_elapsed, self.rate_limit_window)
                    self.logger.debug(f"Previous window stats: {self.request_counter} requests, {request_rate:.1f} req/s")
                
                # Reset for new window
                self.request_counter = 0
                self.request_counter_reset_time = current_time
                self.rate_limit_warning_sent = False
            
            # Calculate current usage percentage
            usage_percentage = (self.request_counter / rate_limit) * 100 if rate_limit > 0 else 0
            
            # Get current market conditions to adjust rate limiting
            market_conditions = {}
            if hasattr(self, 'market_data') and self.market_data:
                market_conditions = self.market_data.get_market_state()
            
            # Detect high volatility or market impact for adaptive rate limiting
            high_volatility = False
            high_market_impact = False
            
            if market_conditions:
                # Check volatility (if available)
                volatility = market_conditions.get('volatility', 0)
                if volatility > TRADING_CONFIG["volatility"]["ceiling"] * 0.7:  # 70% of ceiling
                    high_volatility = True
                    self.logger.debug(f"High volatility detected: {volatility:.5f}")
                
                # Check market impact (if available)
                market_impact = market_conditions.get('market_impact', 0)
                if market_impact > TRADING_CONFIG["avellaneda"]["adverse_selection_threshold"]:
                    high_market_impact = True
                    self.logger.debug(f"High market impact detected: {market_impact:.5f}")
            
            # Add minimum time between requests based on priority with longer minimum times
            min_request_interval = 0.5  # Base 500ms minimum between requests (increased from 300ms)
            
            # Adjust minimum intervals based on market conditions
            if high_volatility or high_market_impact:
                # In high volatility or high market impact, prioritize critical operations
                if priority == "critical":
                    min_request_interval = 0.3  # Allow faster critical operations (300ms)
                elif priority == "high":
                    min_request_interval = 0.8  # Slightly slow down high priority (800ms)
                elif priority == "normal":
                    min_request_interval = 1.2  # Significantly slow down normal priority (1200ms)
                else:  # "low"
                    min_request_interval = 2.0  # Dramatically slow down low priority (2000ms)
            else:
                # Normal market conditions
                if priority == "normal":
                    min_request_interval = 0.8  # 800ms for normal priority (increased from 500ms)
                elif priority == "low":
                    min_request_interval = 1.5  # 1500ms for low priority (increased from 1000ms)
            
            # Check if we're requesting too quickly
            if hasattr(self, 'last_request_time'):
                time_since_last = current_time - self.last_request_time
                if time_since_last < min_request_interval:
                    return False
            
            # Stricter progressive throttling based on usage
            usage_threshold_critical = 70  # Lowered from 80% to 70%
            usage_threshold_high = 50      # Lowered from 60% to 50%
            usage_threshold_normal = 30    # Lowered from 40% to 30%
            
            # Further restrict in high volatility/market impact conditions
            if high_volatility or high_market_impact:
                usage_threshold_critical = 60  # Even lower threshold during volatile markets
                usage_threshold_high = 40
                usage_threshold_normal = 20
                
                if not self.volatile_market_warning_sent:
                    self.logger.warning(f"Volatile market detected - applying stricter rate limiting (high_vol={high_volatility}, high_impact={high_market_impact})")
                    self.volatile_market_warning_sent = True
            else:
                self.volatile_market_warning_sent = False
            
            if usage_percentage >= usage_threshold_critical:
                if not self.rate_limit_warning_sent:
                    self.logger.warning(f"Rate limit high: {usage_percentage:.1f}% used")
                    self.rate_limit_warning_sent = True
                
                # Only critical operations above threshold
                if priority != "critical":
                    return False
            elif usage_percentage >= usage_threshold_high:
                # Above high threshold, only allow critical and high priority
                if priority not in ["critical", "high"]:
                    return False
            elif usage_percentage >= usage_threshold_normal:
                # Random throttling for normal operations at normal-high usage
                if priority == "low" or (priority == "normal" and random.random() < 0.7):  # Increased throttling probability
                    return False
            
            # Update tracking
            self.request_counter += 1
            self.last_request_time = current_time
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error in rate limit check: {str(e)}")
            return False  # Fail safe - deny request on error

    async def send_heartbeat(self):
        """Send a heartbeat to keep the connection alive"""
        if not await self.check_rate_limit():
            return
            
        try:
            # Use a lightweight ping when available
            if hasattr(self.thalex, "ping") and callable(self.thalex.ping):
                is_alive = await self.thalex.ping()
                if is_alive:
                    self.logger.debug("Ping successful - connection is alive")
                    self.last_heartbeat = time.time()
                else:
                    self.logger.warning("Ping failed - connection may be dead")
                    raise ConnectionError("WebSocket ping failed")
            # Fallback to connection check
            elif self.thalex.connected():
                self.logger.debug("Connection is still alive - heartbeat check successful")
                self.last_heartbeat = time.time()
            else:
                self.logger.warning("Connection appears closed during heartbeat check")
                raise ConnectionError("WebSocket connection is closed")
        except Exception as e:
            self.logger.error(f"Error during heartbeat check: {str(e)}")
            # We don't raise here to avoid crashing the heartbeat task
            # The main listen task will detect the connection issue

    async def listen_task(self):
        """Main websocket listener task"""
        self.logger.info("Starting listen task")
        
        cooldown_active = False
        cooldown_until = 0
        consecutive_errors = 0  # Track consecutive errors for exponential backoff
        max_consecutive_errors = 3  # Allow 3 consecutive errors before longer cooldown

        while True:
            try:
                # Check if we are in cooldown mode
                if cooldown_active:
                    current_time = time.time()
                    if current_time >= cooldown_until:
                        cooldown_active = False
                        self.logger.info("Connection cooldown completed, resuming normal operation")
                    else:
                        cooldown_remaining = int(cooldown_until - current_time)
                        self.logger.warning(f"In connection cooldown, {cooldown_remaining}s remaining")
                        await asyncio.sleep(min(cooldown_remaining, 9))  # Reduced from 30s to 9s
                        continue

                # Attempt to establish connection
                self.logger.info("Connecting to Thalex websocket (attempt 1/5)...")
                retry_count = 1
                retry_delay = 1  # Initial retry delay in seconds
                
                self.logger.info(f"Debug: Network config - {BOT_CONFIG['market']['network']}")
                
                # Connection loop with retry
                while True:
                    try:
                        # Connect to WebSocket
                        await self.thalex.connect()
                        self.logger.info("Connection established")
                        
                        # Reset consecutive errors on successful connection
                        consecutive_errors = 0
                        break
                    except Exception as e:
                        if retry_count >= 5:
                            if consecutive_errors >= max_consecutive_errors:
                                # Reduce cooldown duration - high frequency trading needs shorter recovery
                                cooldown_duration = min(60, 9 * (2 ** min(consecutive_errors - max_consecutive_errors, 2)))  # Modified to use 9s as base unit instead of 60s
                                self.logger.warning(f"Too many consecutive errors ({consecutive_errors}). Entering connection cooldown for {cooldown_duration}s")
                                cooldown_active = True
                                cooldown_until = time.time() + cooldown_duration
                            
                            raise
                        retry_count += 1
                        self.logger.warning(f"Connection attempt {retry_count} failed: {str(e)}. Retrying in {retry_delay}s...")
                        await asyncio.sleep(retry_delay)
                        retry_delay = min(9, retry_delay * 2)  # Exponential backoff capped at 9 seconds instead of 5 minutes
                
                # Initialize connection
                # Wait for instruments
                await asyncio.wait_for(self.await_instruments(), timeout=30)
                
                # Login with proper network keys
                network = MARKET_CONFIG["network"]
                if network == Network.TEST:
                    key_id = key_ids[Network.TEST]
                    private_key = private_keys[Network.TEST]
                else:
                    key_id = key_ids[Network.PROD]
                    private_key = private_keys[Network.PROD]
                    
                self.logger.info(f"Logging in with network: {network}")
                await self.thalex.login(
                    key_id,
                    private_key,
                    id=CALL_IDS["login"]
                )
                
                # Set cancel on disconnect
                await self.thalex.set_cancel_on_disconnect(6, id=CALL_IDS["set_cod"])
                
                # Tell the market maker which instrument we're trading
                if self.perp_name:
                    self.market_maker.set_instrument(self.perp_name)
                    self.logger.info(f"Set instrument for market maker to {self.perp_name}")
                
                # Subscribe to channels - allow multiple attempts with backoff
                subscription_success = False
                subscription_attempts = 0
                subscription_retry_delay = 1
                
                while not subscription_success and subscription_attempts < 3:
                    try:
                        await self.thalex.public_subscribe(
                            channels=[f"ticker.{self.perp_name}.raw", f"price_index.{BOT_CONFIG['market']['underlying']}"],
                            id=CALL_IDS["subscribe"]
                        )
                        
                        # Also subscribe to index and order channels
                        await self.thalex.private_subscribe(
                            channels=["session.orders", "account.portfolio", "account.trade_history"],
                            id=CALL_IDS["subscribe"] + 1
                        )
                        
                        subscription_success = True
                        self.logger.info("Successfully subscribed to channels")
                        
                    except Exception as e:
                        subscription_attempts += 1
                        self.logger.warning(f"Failed to subscribe (attempt {subscription_attempts}/3): {str(e)}")
                        if subscription_attempts >= 3:
                            raise
                        await asyncio.sleep(subscription_retry_delay)
                        subscription_retry_delay *= 2
                
                # Main message loop
                message_timeout = 30  # Timeout for receiving messages
                while True:
                    try:
                        # Add a timeout to detect stalled connections
                        message = await asyncio.wait_for(self.thalex.receive(), timeout=message_timeout)
                        # Parse and process message
                        self.logger.info(f"Received message: {message[:200]}...")
                        data = json.loads(message)
                        
                        # Process message based on type - Fixed to better handle the current message format
                        if "channel_name" in data:
                            # New format with channel_name at the top level
                            channel_name = data["channel_name"]
                            if "notification" in data:
                                self.logger.info(f"Processing notification from {channel_name}")
                                await self.handle_notification(channel_name, data["notification"])
                        elif "notification" in data:
                            # Legacy format with notification containing channel
                            notification = data["notification"]
                            if "channel" in notification:
                                channel = notification["channel"]
                                await self.handle_notification(channel, notification["data"])
                            else:
                                self.logger.warning(f"Notification missing channel: {notification}")
                        elif "result" in data:
                            result = data["result"]
                            cid = data.get("id")
                            await self.handle_result(result, cid)
                        elif "error" in data:
                            error = data["error"]
                            cid = data.get("id")
                            await self.handle_error(error, cid)
                        else:
                            self.logger.warning(f"Unknown message format: {message[:100]}...")
                            
                    except asyncio.TimeoutError:
                        # No message received within timeout, check connection
                        self.logger.warning(f"No message received for {message_timeout} seconds, checking connection...")
                        # Send a heartbeat to check connection
                        try:
                            if hasattr(self.thalex, "ping") and callable(self.thalex.ping):
                                is_alive = await self.thalex.ping()
                                if not is_alive:
                                    self.logger.error("Connection ping failed, reconnecting...")
                                    raise ConnectionError("WebSocket ping failed")
                            elif not self.thalex.connected():
                                self.logger.error("Connection appears closed, reconnecting...")
                                raise ConnectionError("WebSocket connection is closed")
                            else:
                                self.logger.info("Connection still alive after timeout check")
                        except Exception as e:
                            self.logger.error(f"Connection check failed: {str(e)}")
                            raise ConnectionError("Connection check failed after message timeout")
                                
                    except websockets.exceptions.ConnectionClosed as e:
                        self.logger.warning(f"WebSocket connection closed: {str(e)}")
                        # Cancel all orders on disconnect
                        await self.order_manager.cancel_all_orders()
                        consecutive_errors += 1
                        raise  # Re-raise to trigger reconnection
                            
                    except Exception as e:
                        self.logger.error(f"Error in listen loop: {str(e)}")
                        if "no close frame received" in str(e).lower() or "too many request" in str(e).lower():
                            self.logger.info("WebSocket connection lost or rate limited, attempting to reconnect...")
                            # Cancel all orders before reconnecting
                            await self.order_manager.cancel_all_orders()
                            consecutive_errors += 1
                            raise  # Re-raise to trigger reconnection
                        
                        # For other errors, log but continue processing
                        self.logger.exception("Unexpected error in message processing:")
                        await asyncio.sleep(1)

            # Handle different types of exceptions at the outermost level
            except (websockets.ConnectionClosed, ConnectionRefusedError, socket.gaierror) as e:
                # Connection level errors - retry with backoff
                self.logger.error(f"Connection error: {str(e)}")
                
                # Wait with exponential backoff
                retry_delay = 1 * (2 ** min(consecutive_errors, 8))  # Cap at 2^8 = 256 seconds
                retry_delay = min(retry_delay, 300)  # But never more than 5 minutes
                
                self.logger.info(f"Reconnecting in {retry_delay} seconds (consecutive errors: {consecutive_errors})...")
                await asyncio.sleep(retry_delay)
                
            except Exception as e:
                # Other unexpected errors
                self.logger.error(f"Error in connection initialization: {str(e)}")
                consecutive_errors += 1
                # Cancel any outstanding orders
                try:
                    await self.order_manager.cancel_all_orders()
                except Exception as cancel_error:
                    self.logger.error(f"Error cancelling orders during reconnection: {str(cancel_error)}")
                
                # Log and continue with brief pause
                self.logger.exception(f"Unexpected error in listen task: {str(e)}")
                await asyncio.sleep(5)  # Brief pause before retrying

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
        """Process ticker updates and generate quotes if conditions are met"""
        try:
            current_time = time.time()
            
            # Add rate limiting check - Increase to 500ms to reduce ticker processing frequency
            min_ticker_interval = 0.5  # Minimum 500ms between ticker processing (increased from 200ms)
            if hasattr(self, 'last_ticker_time') and current_time - self.last_ticker_time < min_ticker_interval:
                self.logger.debug("Skipping ticker update - too frequent")
                return
            self.last_ticker_time = current_time
            
            # Create or update ticker object
            self.ticker = Ticker(ticker_data)
            
            # Log ticker data with improved format
            bid_display = f"{self.ticker.best_bid_price:.2f}" if self.ticker.best_bid_price else "None"
            ask_display = f"{self.ticker.best_ask_price:.2f}" if self.ticker.best_ask_price else "None"
            self.logger.info(
                f"Ticker update received: mark={self.ticker.mark_price:.2f}, "
                f"bid={bid_display}, "
                f"ask={ask_display}"
            )
            
            # Validate ticker data
            if not self.ticker.mark_price or self.ticker.mark_price <= 0:
                self.logger.warning(f"Invalid mark price: {self.ticker.mark_price}")
                return
                
            # Update the market data buffer
            start_time = time.time()
            is_buy = None
            if "direction" in ticker_data:
                is_buy = ticker_data["direction"] == "buy"
                
            self.market_data.update(
                price=self.ticker.mark_price,
                volume=ticker_data.get("volume", 0.0),
                timestamp=int(time.time() * 1000),
                is_buy=is_buy
            )
            
            # Get market state from the buffer
            market_state = self.market_data.get_market_state()
            
            # Update price history for metrics
            self.price_history.append(self.ticker.mark_price)
            self.logger.debug(f"Price history samples: {len(self.price_history)}/{TRADING_PARAMS['volatility']['min_samples']}")
            
            # Get volatility from market data buffer
            if "yz_volatility" in market_state and not np.isnan(market_state["yz_volatility"]):
                volatility = market_state["yz_volatility"]
            else:
                volatility = market_state["volatility"] if not np.isnan(market_state["volatility"]) else TRADING_CONFIG["avellaneda"]["fixed_volatility"]
            
            # Calculate market conditions using the buffer
            market_conditions = {
                "volatility": volatility,
                "market_impact": market_state.get("market_impact", 0.0)
            }
            
            # Update market maker with latest conditions
            self.market_maker.update_market_conditions(
                volatility=market_conditions["volatility"],
                market_impact=market_conditions["market_impact"]
                )
            
            # Check if we now have enough price history and signal the quote task
            min_samples = TRADING_PARAMS["volatility"]["min_samples"]
            
            # Only notify if we have enough samples and should update quotes
            if len(self.price_history) >= min_samples:
                current_quotes = self.current_quotes
                should_update = self.market_maker.should_update_quotes(current_quotes, self.ticker.mark_price)
                
                # Only notify if we need to update quotes and enough time has passed
                if should_update and (not hasattr(self, 'last_quote_time') or current_time - self.last_quote_time >= 2.0):
                    self.logger.info("Notifying quote task to update quotes")
                    async with self.quote_cv:
                        self.quote_cv.notify()
                        self.logger.debug("Quote task notified")
                else:
                    self.logger.debug(f"Waiting for more price history: {len(self.price_history)}/{min_samples}")
                
            # Calculate execution time for performance monitoring
            execution_time = time.time() - start_time
            self.logger.debug(f"Ticker processing time: {execution_time*1000:.2f}ms")
                
        except Exception as e:
            self.logger.error(f"Error handling ticker update: {str(e)}", exc_info=True)

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
                if order.amount >= significant_fill_threshold:
                    self.logger.info(f"Triggering quote update after significant fill of {order.amount:.4f}")
                    
                    # Create comprehensive market conditions
                    market_state = self.market_data.get_market_state()
                    market_conditions = {
                        "volatility": market_state["yz_volatility"] if "yz_volatility" in market_state and not np.isnan(market_state["yz_volatility"]) else 
                                     (market_state["volatility"] if not np.isnan(market_state["volatility"]) else TRADING_CONFIG["avellaneda"]["fixed_volatility"]),
                        "market_impact": market_state.get("market_impact", 0.0),
                        # Add higher impact for a recent fill
                        "fill_impact": 0.5  # Increase market impact after fills
                    }
                    
                    # Queue quote update with ticker object
                    if self.ticker:
                        # Create task for immediate update
                        asyncio.create_task(self.update_quotes(None, market_conditions))
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
        """Process portfolio updates"""
        try:
            self.portfolio = portfolio_data
            
            # Update position tracking
            position_size = 0.0
            mark_price = 0.0
            
            # Extract account balance and margin information
            account_balance = 0.0
            margin_used = 0.0
            margin_utilization = 0.0
            
            for item in portfolio_data:
                # Find position for our instrument
                if item["instrument_name"] == self.perp_name:
                    position_size = item["position"]
                    mark_price = item["mark_price"]
                
                # Extract account balance, usually in the account item
                if "account_balance" in item:
                    account_balance = float(item["account_balance"])
                
                # Extract margin information
                if "margin_used" in item:
                    margin_used = float(item["margin_used"])
                
                # Some exchanges provide utilization directly
                if "margin_utilization" in item:
                    margin_utilization = float(item["margin_utilization"])
            
            # Calculate margin utilization if not provided directly
            if margin_utilization == 0.0 and account_balance > 0:
                margin_utilization = margin_used / account_balance
            
            # Update market maker with position and margin information
            self.market_maker.update_position(position_size, mark_price)
            
            # Set margin utilization in market maker
            if hasattr(self.market_maker, 'margin_utilization'):
                self.market_maker.margin_utilization = margin_utilization
            else:
                setattr(self.market_maker, 'margin_utilization', margin_utilization)
            
            # Log margin information
            self.logger.info(
                f"Portfolio update: Position={position_size:.4f}, Mark price={mark_price:.2f}, "
                f"Margin utilization={margin_utilization:.2%}"
            )
            
        except Exception as e:
            self.logger.error(f"Error handling portfolio update: {str(e)}")

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
            elif cid >= 100:  # Order IDs
                await self.order_manager.handle_order_result(result, cid)
                
        except Exception as e:
            self.logger.error(f"Error handling result: {str(e)}")

    async def handle_error(self, error: Dict, cid: Optional[int]):
        """Handle API errors"""
        try:
            error_msg = error.get('message', '')
            error_code = error.get('code', -1)
            
            # Handle rate limit errors specifically
            if error_code == 4 or "throttle exceeded" in error_msg.lower() or "too many request" in error_msg.lower():
                self.logger.warning(f"Rate limit hit (code {error_code}), slowing down requests")
                
                # Apply stronger rate limiting measures when we hit actual rate limit errors
                current_time = time.time()
                
                # Track rate limit errors to implement exponential backoff
                if not hasattr(self, 'rate_limit_errors'):
                    self.rate_limit_errors = 0
                    self.last_rate_limit_error = 0
                
                # Reset error count if it's been a while since last error
                if current_time - self.last_rate_limit_error > 60:
                    self.rate_limit_errors = 0
                
                self.rate_limit_errors += 1
                self.last_rate_limit_error = current_time
                
                # Immediately reduce request counter to 75% of limit to slow down
                self.request_counter = int(self.max_requests_per_minute * 0.75)
                
                # Implement exponential backoff for cooldown duration based on error frequency
                if self.rate_limit_errors > 1:
                    # Calculate cooldown with shorter exponential backoff (0.5s, 1s, 2s, max 3s)
                    cooldown_time = min(2 ** (self.rate_limit_errors - 2), 3)
                    self.cooldown_active = True
                    self.cooldown_until = current_time + cooldown_time
                    self.logger.warning(f"Rate limit cooldown active for {cooldown_time}s until: {time.ctime(self.cooldown_until)}")
                    
                    # If we're getting many consecutive errors, force a shorter pause
                    if self.rate_limit_errors >= 5:
                        self.logger.warning(f"Multiple consecutive rate limit errors, enforcing stronger cooldown")
                        # Wait for a bit to let the rate limit reset, reduced from 1.5s to 0.5s
                        await asyncio.sleep(0.5)
                
                # For order-related rate limits, handle just that order error
                if cid and cid >= 100:
                    await self.order_manager.handle_order_error(error, cid)
                
                return
            
            # Handle other API errors
            self.logger.error(f"API error for message {cid}: {error}")
            
            if cid and cid >= 100:  # Order IDs
                await self.order_manager.handle_order_error(error, cid)
                
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
                ticker = Ticker(price_data)
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
            
            # Always update quotes if it's been too long since the last update
            max_update_interval = 60.0  # Force update every 60 seconds 
            if hasattr(self, 'last_quote_update_time') and current_time - self.last_quote_update_time >= max_update_interval:
                self.logger.info(f"Forcing quote update - {current_time - self.last_quote_update_time:.1f}s elapsed since last update")
                # Cancel existing quotes
                await self.cancel_all_quotes()
                # Brief pause to allow cancellations to process
                await asyncio.sleep(0.5)
                # Place new quotes
                await self.place_quotes(bid_quotes, ask_quotes)
                # Update last quote time
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
                await self.cancel_all_quotes()
                
                # Brief pause to allow cancellations to process
                await asyncio.sleep(0.5)
                
                # Place new quotes
                self.logger.info("Placing new quotes...")
                await self.place_quotes(bid_quotes, ask_quotes)
            
            # Update last quote time
            self.last_quote_time = current_time
            self.last_quote_update_time = current_time
                
        except Exception as e:
            self.logger.error(f"Error updating quotes: {str(e)}", exc_info=True)
            # On error, try to cancel all quotes for safety
            try:
                await self.cancel_all_quotes()
            except Exception as cancel_error:
                self.logger.error(f"Error cancelling quotes after update error: {str(cancel_error)}")

    async def cancel_all_quotes(self):
        """Cancel all open quotes"""
        try:
            # Only cancel if we actually have quotes
            order_count = len(self.order_manager.active_bids) + len(self.order_manager.active_asks)
            if order_count == 0:
                self.logger.info("No quotes to cancel")
                self.current_quotes = ([], [])
                return
            
            self.logger.info(f"Cancelling {order_count} quotes...")
            await self.order_manager.cancel_all_orders()
            self.current_quotes = ([], [])
            self.logger.info("All quotes cancelled")
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

    async def place_quotes(self, bid_quotes: List[Quote], ask_quotes: List[Quote]):
        """
        Place new quotes on the exchange
        
        Args:
            bid_quotes: List of bid quotes to place
            ask_quotes: List of ask quotes to place
        """
        try:
            start_time = time.time()
            
            # Enforce maximum number of quotes per side
            max_quotes_per_side = 3  # Increased from 2 to 3, but still conservative
            
            if len(bid_quotes) > max_quotes_per_side:
                self.logger.warning(f"Limiting bid quotes from {len(bid_quotes)} to {max_quotes_per_side}")
                bid_quotes = bid_quotes[:max_quotes_per_side]
                
            if len(ask_quotes) > max_quotes_per_side:
                self.logger.warning(f"Limiting ask quotes from {len(ask_quotes)} to {max_quotes_per_side}")
                ask_quotes = ask_quotes[:max_quotes_per_side]
                
            # Log quote placement plan
            self.logger.info(f"Placing {len(bid_quotes)} bids and {len(ask_quotes)} asks")
            
            # Store the new quotes for later reference
            self.current_quotes = (bid_quotes, ask_quotes)
            
            # Increased delay between quote placements (30% slower)
            quote_delay = 0.5  # Increased from 0.2 to 0.5 seconds
            
            # Place quotes in balanced pairs to maintain market presence
            new_bids = []
            new_asks = []
            
            max_pairs = max(len(bid_quotes), len(ask_quotes))
            for i in range(max_pairs):
                # Check rate limit before each pair
                if not await self.check_rate_limit(priority="high"):
                    self.logger.warning("Rate limit reached - pausing quote placement")
                    await asyncio.sleep(1.0)  # Added pause when hitting rate limit
                    if not await self.check_rate_limit(priority="high"):
                        break
                    
                # Place bid if available
                if i < len(bid_quotes):
                    try:
                        order_id = await self.order_manager.place_order(
                            instrument=self.perp_name,
                            direction="buy",
                            price=bid_quotes[i].price,
                            amount=bid_quotes[i].amount,
                            label=MARKET_CONFIG["label"]
                        )
                        if order_id:
                            new_bids.append(order_id)
                            self.logger.debug(f"Placed bid: {bid_quotes[i].amount:.4f} @ {bid_quotes[i].price:.2f}")
                    except Exception as e:
                        self.logger.error(f"Error placing bid quote: {str(e)}")
                
                await asyncio.sleep(quote_delay)  # Increased delay between quotes
                
                # Check rate limit again before ask
                if not await self.check_rate_limit(priority="high"):
                    self.logger.warning("Rate limit reached after bid - pausing")
                    await asyncio.sleep(1.0)
                    if not await self.check_rate_limit(priority="high"):
                        break
                    
                # Place ask if available
                if i < len(ask_quotes):
                    try:
                        order_id = await self.order_manager.place_order(
                            instrument=self.perp_name,
                            direction="sell",
                            price=ask_quotes[i].price,
                            amount=ask_quotes[i].amount,
                            label=MARKET_CONFIG["label"]
                        )
                        if order_id:
                            new_asks.append(order_id)
                            self.logger.debug(f"Placed ask: {ask_quotes[i].amount:.4f} @ {ask_quotes[i].price:.2f}")
                    except Exception as e:
                        self.logger.error(f"Error placing ask quote: {str(e)}")
                
                await asyncio.sleep(quote_delay)  # Increased delay between pairs
                    
            # Log placement results
            elapsed = time.time() - start_time
            self.logger.info(
                f"Quote placement completed in {elapsed:.2f}s. Placed {len(new_bids)}/{len(bid_quotes)} bids and "
                f"{len(new_asks)}/{len(ask_quotes)} asks"
            )
                
        except Exception as e:
            self.logger.error(f"Error placing quotes: {str(e)}", exc_info=True)

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