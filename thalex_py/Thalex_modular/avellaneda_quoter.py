import asyncio
import json
import logging
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
from config.market_config import (
    BOT_CONFIG,
    MARKET_CONFIG, 
    CALL_IDS, 
    RISK_LIMITS,
    TRADING_PARAMS,
    TECHNICAL_PARAMS,
    TRADING_CONFIG
)
from thalex_py.Thalex_modular.models.data_models import Ticker, Order, OrderStatus, Quote
from thalex_py.Thalex_modular.components.risk_manager import RiskManager
from thalex_py.Thalex_modular.components.order_manager import OrderManager
from thalex_py.Thalex_modular.components.avellaneda_market_maker import AvellanedaMarketMaker
from thalex_py.Thalex_modular.components.technical_analysis import TechnicalAnalysis
from thalex_py.Thalex_modular.models.keys import key_ids, private_keys
from thalex_py.Thalex_modular.performance_monitor import PerformanceMonitor

# Set up logging
logging.basicConfig(
    level=logging.INFO,  # Changed from DEBUG to INFO for less verbose output
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('avellaneda_quoter.log')
    ]
)
logger = logging.getLogger('AvellanedaQuoter')  # Give the logger a specific name

class AvellanedaQuoter:
    def __init__(self, thalex: th.Thalex):
        """Initialize the Avellaneda-Stoikov market maker"""
        self.thalex = thalex
        self.ticker: Optional[Ticker] = None
        self.index: Optional[float] = None
        self.quote_cv = asyncio.Condition()
        self.portfolio: Dict[str, float] = {}
        self.perp_name: Optional[str] = None
        
        # Initialize components
        self.risk_manager = RiskManager()
        self.order_manager = OrderManager(thalex)
        self.market_maker = AvellanedaMarketMaker()
        self.technical_analysis = TechnicalAnalysis()
        
        # Performance monitoring
        self.performance_monitor = PerformanceMonitor(output_dir="metrics")
        
        # Market data
        self.price_history = deque(maxlen=TRADING_PARAMS["volatility"]["window"])
        self.current_quotes: Tuple[List[Quote], List[Quote]] = ([], [])
        
        # Performance tracking
        self.quoting_enabled = True
        self.last_quote_time = time.time()
        self.last_position_check = time.time()
        
        # Rate limiting and connection health
        self.request_counter = 0
        self.request_reset_time = time.time()
        self.request_counter_reset_time = time.time()
        self.rate_limit_window = 60  # 1 minute window
        self.rate_limit_warning_sent = False
        self.max_requests_per_minute = BOT_CONFIG["connection"]["rate_limit"]
        self.last_heartbeat = time.time()
        self.heartbeat_interval = BOT_CONFIG["connection"]["heartbeat_interval"]
        self.cooldown_active = False
        self.cooldown_until = 0
        
        # Use the already configured logger
        self.logger = logger.getChild('quoter')  # Use the parent logger
        self.logger.info("Avellaneda quoter initialized")

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

    async def check_rate_limit(self) -> bool:
        """Check if we're under the rate limit and increment counter if we are"""
        # Get current rate limit from config
        rate_limit = BOT_CONFIG["connection"]["rate_limit"]
        
        # Check if we need to reset the counter (new time window)
        current_time = time.time()
        if current_time - self.request_counter_reset_time > self.rate_limit_window:
            # Reset counter for new window
            previous_count = self.request_counter
            self.request_counter = 0
            self.request_counter_reset_time = current_time
            self.rate_limit_warning_sent = False
            
            # Log previous window's request rate
            request_rate = previous_count / self.rate_limit_window
            self.logger.debug(f"Request rate for previous window: {request_rate:.2f} requests/second")
            
        # Calculate current usage percentage
        usage_percentage = (self.request_counter / rate_limit) * 100
        
        # Implement progressive throttling based on current usage
        if usage_percentage >= 100:
            # Hard limit exceeded
            if not self.rate_limit_warning_sent:
                self.logger.warning(f"Rate limit reached: {self.request_counter} requests in the last {self.rate_limit_window}s")
                self.rate_limit_warning_sent = True
            return False
        elif usage_percentage >= 90:
            # At 90% capacity, allow only critical operations
            # For quote updates, only let through 10% of requests
            if random.random() < 0.1:
                self.logger.debug(f"Allowing operation at high usage ({usage_percentage:.1f}%)")
                self.request_counter += 1
                return True
            return False
        elif usage_percentage >= 75:
            # At 75% capacity, start throttling
            # Let through 30% of requests
            if random.random() < 0.3:
                self.logger.debug(f"Allowing operation at moderate usage ({usage_percentage:.1f}%)")
                self.request_counter += 1
                return True
            return False
        elif usage_percentage >= 50:
            # At 50% capacity, slow down slightly
            # Let through 60% of requests
            if random.random() < 0.6:
                self.request_counter += 1
                return True
            return False
            
        # Under 50% capacity, allow operation
        self.request_counter += 1
        return True

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
                        await asyncio.sleep(min(cooldown_remaining, 30))
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
                                cooldown_duration = min(1800, 60 * (2 ** min(consecutive_errors - max_consecutive_errors, 4)))
                                self.logger.warning(f"Too many consecutive errors ({consecutive_errors}). Entering connection cooldown for {cooldown_duration}s")
                                cooldown_active = True
                                cooldown_until = time.time() + cooldown_duration
                            
                            raise
                        retry_count += 1
                        self.logger.warning(f"Connection attempt {retry_count} failed: {str(e)}. Retrying in {retry_delay}s...")
                        await asyncio.sleep(retry_delay)
                        retry_delay = min(300, retry_delay * 2)  # Exponential backoff capped at 5 minutes
                
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
            # Create or update ticker object
            self.ticker = Ticker(ticker_data)
            
            # Log ticker data with improved format - Change to INFO level for better visibility
            # Fix the formatting by moving conditional logic outside format specifiers
            bid_display = f"{self.ticker.best_bid_price:.2f}" if self.ticker.best_bid_price else "None"
            ask_display = f"{self.ticker.best_ask_price:.2f}" if self.ticker.best_ask_price else "None"
            self.logger.info(
                f"Ticker update: mark={self.ticker.mark_price:.2f}, "
                f"bid={bid_display}, "
                f"ask={ask_display}"
            )
            
            # Validate ticker data
            if not self.ticker.mark_price or self.ticker.mark_price <= 0:
                self.logger.warning(f"Invalid mark price: {self.ticker.mark_price}")
                return
                
            # 1. Update price history for technical analysis - Add more detailed logging
            prev_len = len(self.price_history)
            self.price_history.append(self.ticker.mark_price)
            self.logger.info(f"Price history updated: {len(self.price_history)}/{TRADING_PARAMS['volatility']['min_samples']} samples")
            
            # 2. Update technical analysis with new price
            self.technical_analysis.update(self.ticker.mark_price)
            volatility = self.technical_analysis.get_volatility()
            self.logger.info(f"Volatility calculated: {volatility:.6f}")
            
            # 3. Calculate comprehensive market conditions
            market_conditions = {
                "volatility": volatility,
                "is_volatile": self.technical_analysis.is_volatile(),
                "is_trending": self.technical_analysis.is_trending(),
                "trend_direction": self.technical_analysis.get_trend_direction(),
                "trend_strength": self.technical_analysis.get_trend_strength(),
                "mean_reverting_signal": self.technical_analysis.is_mean_reverting(),
                "zscore": self.technical_analysis.get_zscore(),
                "market_impact": self.technical_analysis.get_market_impact()
            }
            
            # Log market conditions summary
            self.logger.info(f"Market conditions: volatile={market_conditions['is_volatile']}, trending={market_conditions['is_trending']}, impact={market_conditions['market_impact']:.4f}")
            
            # 4. Update market maker with latest conditions
            self.market_maker.update_market_conditions(
                volatility=market_conditions["volatility"],
                market_impact=market_conditions["market_impact"]
            )
            
            # 5. Update VAMP calculation with latest price data
            if hasattr(self.market_maker, 'update_vamp'):
                self.market_maker.update_vamp(
                    price=self.ticker.mark_price,
                    volume=ticker_data.get("volume", 0.1),
                    is_buy=ticker_data.get("direction", "buy") == "buy"
                )
                self.logger.info(f"VAMP calculation updated")
            
            # Check if we now have enough price history and signal the quote task
            min_samples = TRADING_PARAMS["volatility"]["min_samples"]
            if len(self.price_history) >= min_samples:
                self.logger.info(f"We now have enough price history: {len(self.price_history)}/{min_samples} samples!")
                
                # 7. Check if quotes should be updated
                current_quotes = self.current_quotes
                should_update = self.market_maker.should_update_quotes(current_quotes, self.ticker.mark_price)
                
                if should_update:
                    self.logger.info("Updating quotes based on market conditions")
                    # Pass ticker directly to update_quotes - Use create_task for concurrent execution
                    task = asyncio.create_task(self.update_quotes(None, market_conditions))
                    # Add callback to log completion - improved error handling
                    task.add_done_callback(lambda t: self.logger.info(f"Quote update task completed: {t.exception() or 'success'}"))
                else:
                    self.logger.info("No quote update needed")
            else:
                self.logger.info(f"Still waiting for price history: {len(self.price_history)}/{min_samples} samples")
                # Check if we're making progress collecting samples
                if len(self.price_history) > prev_len:
                    self.logger.info(f"Progress: {len(self.price_history)} samples collected, need {min_samples}")
                
            # Notify quote task of the ticker update
            async with self.quote_cv:
                self.quote_cv.notify()
                self.logger.info("Quote task notified of ticker update")
                
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
            # Create order object from update data
            order = Order(
                id=order_data["client_order_id"],
                price=order_data["price"],
                amount=order_data["amount"],
                status=OrderStatus(order_data["status"]),
                direction=order_data.get("direction", "")
            )
            
            # Update order status in the order manager
            await self.order_manager.update_order(order)
            
            # Handle filled orders
            if order.status == OrderStatus.FILLED:
                is_buy = order.direction == "buy"
                
                # Log the fill with improved formatting
                self.logger.info(
                    f"Order filled: {order.id} - {order.direction.upper()} {order.amount:.4f} @ {order.price:.2f}"
                )
                
                # Update market maker with fill information
                self.market_maker.on_order_filled(
                    order_id=order.id,
                    fill_price=order.price,
                    fill_size=order.amount,
                    is_buy=is_buy
                )
                
                # Update VAMP with order fill data if supported
                if hasattr(self.market_maker, 'update_vamp'):
                    self.market_maker.update_vamp(
                        price=order.price,
                        volume=order.amount,
                        is_buy=is_buy,
                        is_aggressive=False  # Assuming most of our fills are passive
                    )
                
                # Force quote update after fills based on size threshold
                significant_fill_threshold = TRADING_CONFIG["avellaneda"].get("significant_fill_threshold", 0.1)
                if order.amount >= significant_fill_threshold:
                    self.logger.info(f"Triggering quote update after significant fill of {order.amount:.4f}")
                    
                    # Create comprehensive market conditions
                    market_conditions = {
                        "volatility": self.technical_analysis.get_volatility(),
                        "is_volatile": self.technical_analysis.is_volatile(),
                        "is_trending": self.technical_analysis.is_trending(),
                        "trend_direction": self.technical_analysis.get_trend_direction(),
                        "market_impact": self.technical_analysis.get_market_impact(),
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

    async def handle_portfolio_update(self, portfolio_data: List[Dict]):
        """Process portfolio updates"""
        try:
            self.portfolio = portfolio_data
            
            # Update position tracking
            for item in portfolio_data:
                if item["instrument_name"] == self.perp_name:
                    position_size = item["position"]
                    mark_price = item["mark_price"]
                    self.market_maker.update_position(position_size, mark_price)
                    break
                    
        except Exception as e:
            self.logger.error(f"Error handling portfolio update: {str(e)}")

    async def handle_trade_update(self, trade_data: Union[Dict, List[Dict]]):
        """Process trade updates from the market"""
        try:
            # Handle both single trade and list of trades
            trades = [trade_data] if isinstance(trade_data, dict) else trade_data
            
            for trade in trades:
                if trade.get("label") == MARKET_CONFIG["label"]:
                    # Update risk metrics
                    await self.risk_manager.update_trade_metrics(trade)
                    
                    # Record trade in performance monitor
                    self.performance_monitor.record_trade(trade)
                    
                    # Check if we need to adjust our quoting
                    if not await self.risk_manager.check_risk_limits():
                        self.quoting_enabled = False
                        await self.cancel_all_quotes()
                        self.logger.warning("Quoting disabled due to risk limits")
                else:
                    # This is a market trade (not ours) - update VAMP with aggressive trade
                    is_buy = trade.get("direction", "") == "buy"
                    price = trade.get("price", 0.0)
                    amount = trade.get("amount", 0.0)
                    
                    if price > 0 and amount > 0:
                        self.market_maker.update_vamp(
                            price=price,
                            volume=amount,
                            is_buy=is_buy,
                            is_aggressive=True  # Market trades are aggressive
                        )
                        
                        # Update technical analysis with trade impact
                        self.technical_analysis.update_market_impact(amount, price)
                    
        except Exception as e:
            self.logger.error(f"Error handling trade update: {str(e)}")

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
                    # Calculate cooldown with exponential backoff (1s, 2s, 4s, 8s, max 30s)
                    cooldown_time = min(2 ** (self.rate_limit_errors - 1), 30)
                    self.cooldown_active = True
                    self.cooldown_until = current_time + cooldown_time
                    self.logger.warning(f"Rate limit cooldown active for {cooldown_time}s until: {time.ctime(self.cooldown_until)}")
                    
                    # If we're getting many consecutive errors, force a longer pause
                    if self.rate_limit_errors >= 5:
                        self.logger.warning(f"Multiple consecutive rate limit errors, enforcing stronger cooldown")
                        # Wait for a bit to let the rate limit reset
                        await asyncio.sleep(3)
                
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
        """Generate and place new quotes with enhanced error handling and tracking"""
        try:
            self.logger.info("Beginning quote generation process")
            
            if not self.quoting_enabled:
                self.logger.warning("Quote generation skipped: quoting disabled")
                return
                
            if not self.ticker:
                self.logger.warning("Quote generation skipped: no ticker data")
                return
                
            # Skip quote generation until we have enough price history
            if len(self.price_history) < TRADING_PARAMS["volatility"]["min_samples"]:
                self.logger.info(f"Quote generation skipped: insufficient price history ({len(self.price_history)}/{TRADING_PARAMS['volatility']['min_samples']})")
                return
                
            # Check risk limits before generating quotes
            self.logger.info("Checking risk limits before generating quotes")
            if not await self.risk_manager.check_risk_limits():
                self.logger.warning("Quote generation skipped: risk limits exceeded")
                await self.cancel_all_quotes()
                return
                
            self.logger.info(f"Generating quotes with mark price: {self.ticker.mark_price}")
                
            # Generate quotes using enhanced Avellaneda-Stoikov model
            self.logger.info("Calling market_maker.generate_quotes")
            bid_quotes, ask_quotes = self.market_maker.generate_quotes(
                self.ticker,  # Pass the ticker object directly instead of price_data
                market_conditions
            )
            
            # Validate quotes
            self.logger.info(f"Raw quotes generated - Bids: {len(bid_quotes)}, Asks: {len(ask_quotes)}")
            bid_quotes, ask_quotes = self.market_maker.validate_quotes(bid_quotes, ask_quotes)
            
            if not bid_quotes and not ask_quotes:
                self.logger.warning("No valid quotes generated after validation")
                return
                
            self.logger.info(f"Validated quotes - Bids: {len(bid_quotes)}, Asks: {len(ask_quotes)}")
            
            # Cancel existing orders before placing new ones
            self.logger.info("Cancelling existing orders before placing new ones")
            await self.order_manager.cancel_all_orders()
                
            # Place new quotes with explicit result handling
            total_orders = len(bid_quotes) + len(ask_quotes)
            successful_orders = 0
            
            # Process bid quotes
            for i, quote in enumerate(bid_quotes):
                self.logger.info(f"Placing bid #{i+1}: {quote.amount:.3f} @ {quote.price:.2f}")
                order_id = await self.order_manager.place_order(
                    instrument=self.perp_name,
                    direction="buy",
                    price=quote.price,
                    amount=quote.amount,
                    label=MARKET_CONFIG["label"]
                )
                if order_id:
                    self.logger.info(f"Successfully placed bid #{i+1}: {quote.amount:.3f} @ {quote.price:.2f} (ID: {order_id})")
                    successful_orders += 1
                else:
                    self.logger.error(f"Failed to place bid #{i+1}: {quote.amount:.3f} @ {quote.price:.2f}")
                    
            # Process ask quotes
            for i, quote in enumerate(ask_quotes):
                self.logger.info(f"Placing ask #{i+1}: {quote.amount:.3f} @ {quote.price:.2f}")
                order_id = await self.order_manager.place_order(
                    instrument=self.perp_name,
                    direction="sell",
                    price=quote.price,
                    amount=quote.amount,
                    label=MARKET_CONFIG["label"]
                )
                if order_id:
                    self.logger.info(f"Successfully placed ask #{i+1}: {quote.amount:.3f} @ {quote.price:.2f} (ID: {order_id})")
                    successful_orders += 1
                else:
                    self.logger.error(f"Failed to place ask #{i+1}: {quote.amount:.3f} @ {quote.price:.2f}")
            
            # Update current quotes tracking only if we successfully placed orders
            if successful_orders > 0:
                self.current_quotes = (bid_quotes, ask_quotes)
                self.market_maker.last_quote_time = time.time()  # Update the last quote time in the market maker
                
                # Log success rate
                success_percentage = (successful_orders / total_orders) * 100 if total_orders > 0 else 0
                self.logger.info(f"Quote update completed: {successful_orders}/{total_orders} orders successful ({success_percentage:.1f}%)")
                
                # Log detailed quote information
                if bid_quotes and ask_quotes:
                    realized_spread = ask_quotes[0].price - bid_quotes[0].price
                    realized_spread_bps = (realized_spread / self.ticker.mark_price) * 10000
                    self.logger.info(
                        f"Spread: {realized_spread:.2f} ({realized_spread_bps:.1f} bps), " 
                        f"Bid: {bid_quotes[0].price:.2f} x {bid_quotes[0].amount:.3f}, "
                        f"Ask: {ask_quotes[0].price:.2f} x {ask_quotes[0].amount:.3f}"
                    )
                elif bid_quotes:
                    self.logger.info(f"Only bid: {bid_quotes[0].price:.2f} x {bid_quotes[0].amount:.3f}")
                elif ask_quotes:
                    self.logger.info(f"Only ask: {ask_quotes[0].price:.2f} x {ask_quotes[0].amount:.3f}")
            else:
                self.logger.warning("Quote update failed: No orders were placed successfully")
                
        except Exception as e:
            self.logger.error(f"Error updating quotes: {str(e)}", exc_info=True)

    async def cancel_all_quotes(self):
        """Cancel all open quotes"""
        try:
            await self.order_manager.cancel_all_orders()
            self.current_quotes = ([], [])
        except Exception as e:
            self.logger.error(f"Error cancelling quotes: {str(e)}")
            
    def reset_cooldown(self):
        """Reset cooldown state to allow order placement"""
        self.cooldown_active = False
        self.cooldown_until = 0
        self.request_counter = 0
        self.request_counter_reset_time = time.time()
        self.rate_limit_warning_sent = False
        self.logger.info("Cooldown state has been manually reset")
        # Also reset the cooldown in order manager if needed
        try:
            if hasattr(self, 'order_manager') and self.order_manager:
                self.order_manager.last_order_time = 0
                self.logger.info("Order manager timing constraints also reset")
        except Exception as e:
            self.logger.error(f"Error resetting order manager state: {str(e)}")

    async def quote_task(self):
        """Main quoting loop - periodically checks if quotes need updating"""
        self.logger.info("Starting quote task")
        
        force_update_timer = 0
        last_update_attempt = 0
        last_forced_update = 0
        
        # Track consecutive updates to prevent hammering
        consecutive_updates = 0
        
        while True:
            try:
                # Wait for price updates
                self.logger.debug("Quote task waiting for updates")
                async with self.quote_cv:
                    await self.quote_cv.wait()
                
                self.logger.debug("Quote task woken up")
                
                # First check if we're in cooldown due to rate limiting
                current_time = time.time()
                if self.cooldown_active and current_time < self.cooldown_until:
                    remaining_cooldown = int(self.cooldown_until - current_time)
                    self.logger.info(f"Skipping quote update - in cooldown for {remaining_cooldown}s")
                    continue
                elif self.cooldown_active:
                    self.cooldown_active = False
                    self.logger.info("Cooldown period ended, resuming quote updates")
                
                # Force update if we haven't updated in a long time (15 seconds is max)
                time_since_last_update = current_time - last_update_attempt
                time_since_forced_update = current_time - last_forced_update
                min_quote_interval = TRADING_CONFIG["quoting"]["min_quote_interval"]
                
                # Always respect the minimum quote interval to avoid hammering the API
                if time_since_last_update < min_quote_interval:
                    self.logger.debug(f"Skipping update - too soon since last attempt ({time_since_last_update:.1f}s < {min_quote_interval}s)")
                    continue
                
                # Check if we have enough price history
                min_samples = TRADING_PARAMS["volatility"]["min_samples"]
                enough_samples = len(self.price_history) >= min_samples
                
                if enough_samples:
                    self.logger.info(f"We have enough price history: {len(self.price_history)}/{min_samples} samples")
                else:
                    self.logger.info(f"Not enough price history: {len(self.price_history)}/{min_samples} samples")
                    # Even with insufficient samples, we might still want to quote after collecting some data
                    if len(self.price_history) >= min_samples // 2:
                        self.logger.info(f"Proceeding with quoting despite insufficient samples")
                    else:
                        continue
                
                # Check risk limits
                risk_check_result = await self.risk_manager.check_risk_limits()
                
                if not risk_check_result:
                    if self.quoting_enabled:
                        self.logger.warning("Risk limits exceeded, disabling quoting")
                        self.quoting_enabled = False
                        await self.cancel_all_quotes()
                    continue
                else:
                    if not self.quoting_enabled:
                        self.logger.info("Risk limits OK, enabling quoting")
                        self.quoting_enabled = True
                
                # Skip if no ticker data
                if not self.ticker:
                    self.logger.warning("No ticker data available")
                    continue
                
                # Update comprehensive market conditions
                market_conditions = {
                    "volatility": self.technical_analysis.get_volatility(),
                    "is_volatile": self.technical_analysis.is_volatile(),
                    "is_trending": self.technical_analysis.is_trending(),
                    "trend_direction": self.technical_analysis.get_trend_direction(),
                    "trend_strength": self.technical_analysis.get_trend_strength(),
                    "mean_reverting_signal": self.technical_analysis.is_mean_reverting(),
                    "zscore": self.technical_analysis.get_zscore(),
                    "market_impact": self.technical_analysis.get_market_impact()
                }
                
                # Calculate mid price for quote update check
                mid_price = None
                if self.ticker.best_bid_price and self.ticker.best_ask_price:
                    mid_price = (self.ticker.best_bid_price + self.ticker.best_ask_price) / 2
                else:
                    mid_price = self.ticker.mark_price
                
                # Update position's unrealized PnL with current market price
                if self.market_maker:
                    self.market_maker.position_tracker.update_unrealized_pnl(self.ticker.mark_price)
                    position_info = self.market_maker.get_position_metrics()
                    self.logger.info(
                        f"Position: {position_info['position']:.4f}, "
                        f"PnL: ${position_info['total_pnl']:.2f}, "
                        f"Unrealized: ${position_info['unrealized_pnl']:.2f}, "
                        f"Realized: ${position_info['realized_pnl']:.2f}"
                    )
                
                # Determine if we should update quotes
                should_update = False
                
                # Force quote generation if no active quotes
                if len(self.current_quotes[0]) == 0 and len(self.current_quotes[1]) == 0:
                    self.logger.info("No active quotes - forcing quote generation")
                    should_update = True
                    last_forced_update = current_time
                # Force periodic updates but with increasing intervals during high rate limit pressure
                elif time_since_forced_update > max(15, min_quote_interval * 3):
                    # Forced updates happen less frequently when we've had many consecutive updates
                    self.logger.info(f"Periodic forced quote update (last forced update {time_since_forced_update:.1f}s ago)")
                    should_update = True
                    last_forced_update = current_time
                # Update if the market maker suggests it based on conditions
                elif self.market_maker.should_update_quotes(self.current_quotes, mid_price):
                    self.logger.info("Market conditions indicate quotes should be updated")
                    should_update = True
                
                # Add rate limit awareness - if we've been updating frequently, slow down
                if should_update:
                    # Check if we're approaching rate limit
                    usage_percentage = (self.request_counter / self.max_requests_per_minute) * 100
                    
                    # If we're over 65% of rate limit and have made several consecutive updates,
                    # start throttling update frequency
                    if usage_percentage > 65 and consecutive_updates > 2:
                        # Probability of skipping decreases as time since last update increases
                        skip_probability = max(0, 0.8 - (time_since_last_update / 30))
                        if random.random() < skip_probability:
                            self.logger.info(f"Skipping update to manage rate limit load ({usage_percentage:.1f}% usage)")
                            should_update = False
                    
                    if should_update:
                        consecutive_updates += 1
                    else:
                        consecutive_updates = max(0, consecutive_updates - 1)
                        
                # Check rate limit before attempting quote update
                if should_update and not await self.check_rate_limit():
                    self.logger.warning("Rate limit would be exceeded - delaying quote update")
                    should_update = False
                    
                # Update quotes if needed
                if should_update:
                    self.logger.info("Updating quotes now")
                    await self.update_quotes(None, market_conditions)
                    last_update_attempt = current_time
                else:
                    self.logger.debug("Skipping quote update - conditions don't warrant an update")
                    
            except Exception as e:
                self.logger.error(f"Error in quote task: {str(e)}", exc_info=True)
                await asyncio.sleep(1)

    async def heartbeat_task(self):
        """Heartbeat task to keep the connection alive"""
        while True:
            try:
                # Only do active checks every 2 minutes
                # Just monitor connection state, don't send actual requests
                if time.time() - self.last_heartbeat > self.heartbeat_interval * 4:
                    await self.send_heartbeat()
                await asyncio.sleep(self.heartbeat_interval)
            except Exception as e:
                self.logger.error(f"Error in heartbeat task: {str(e)}")
                await asyncio.sleep(5)  # Regular wait before retrying

async def main():
    """Main entry point"""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('avellaneda_quoter.log')
        ]
    )
    logger = logging.getLogger(__name__)
    
    # Initialize Thalex client
    thalex = th.Thalex(network=BOT_CONFIG["market"]["network"])
    
    # Create and run the quoter
    quoter = AvellanedaQuoter(thalex)
    
    # Set reference to quoter in thalex client for rate limit tracking
    thalex.quoter = quoter
    
    # Create tasks
    tasks = [
        asyncio.create_task(quoter.listen_task()),
        asyncio.create_task(quoter.quote_task()),
        # Add heartbeat task
        asyncio.create_task(quoter.heartbeat_task()),
        # Add performance monitoring task
        asyncio.create_task(quoter.performance_monitor.start_recording(quoter))
    ]
    
    # Main loop with reconnection logic
    while True:
        try:
            logger.info(f"Starting on {BOT_CONFIG['market']['network']} with underlying {BOT_CONFIG['market']['underlying']}")
            await asyncio.gather(*tasks)
        except (websockets.ConnectionClosed, socket.gaierror) as e:
            logger.error(f"Connection error ({e}). Reconnecting...")
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.sleep(1)
        except KeyboardInterrupt:
            logger.info("Shutting down...")
            for task in tasks:
                if not task.done():
                    task.cancel()
            break
        except Exception as e:
            logger.exception("Unexpected error:")
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.sleep(1)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("Shutting down gracefully...")
    except Exception as e:
        logging.critical(f"Fatal error in main loop: {str(e)}") 