import asyncio
import json
import logging
import socket
import time
from typing import Optional, Dict, List, Tuple, Union
import websockets
from websockets.protocol import State as WsState
from collections import deque

import thalex as th
from thalex import Network
from thalex_py.Thalex_modular.config.market_config import (
    BOT_CONFIG,
    MARKET_CONFIG, 
    CALL_IDS, 
    RISK_LIMITS,
    TRADING_PARAMS,
    TECHNICAL_PARAMS
)
from thalex_py.Thalex_modular.models.data_models import Ticker, Order, OrderStatus, Quote
from thalex_py.Thalex_modular.components.risk_manager import RiskManager
from thalex_py.Thalex_modular.components.order_manager import OrderManager
from thalex_py.Thalex_modular.components.avellaneda_market_maker import AvellanedaMarketMaker
from thalex_py.Thalex_modular.components.technical_analysis import TechnicalAnalysis
from thalex_py.Thalex_modular.models.keys import key_ids, private_keys

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
        self.max_requests_per_minute = BOT_CONFIG["connection"]["rate_limit"]  # From unified config
        self.last_heartbeat = time.time()
        self.heartbeat_interval = BOT_CONFIG["connection"]["heartbeat_interval"]  # From unified config
        self.cooldown_active = False
        self.cooldown_until = 0
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
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
        """Check if we're within rate limits, implement cooldown if needed"""
        current_time = time.time()
        
        # Reset counter every minute
        if current_time - self.request_reset_time > 60:
            self.logger.debug(f"Resetting request counter from {self.request_counter} to 0")
            self.request_counter = 0
            self.request_reset_time = current_time
            
        # Check if we're in cooldown
        if self.cooldown_active:
            if current_time < self.cooldown_until:
                remaining = round(self.cooldown_until - current_time)
                if remaining % 10 == 0:  # Log only every 10 seconds to avoid spam
                    self.logger.debug(f"In cooldown period - {remaining}s remaining")
                return False
            else:
                self.logger.info("Cooldown period ended, resuming normal operation")
                self.cooldown_active = False
                # Reset counter after cooldown
                self.request_counter = 0
                self.request_reset_time = current_time
                
        # Check if we're approaching rate limit
        if self.request_counter > self.max_requests_per_minute * 0.7:
            self.logger.warning(f"Approaching rate limit: {self.request_counter}/{self.max_requests_per_minute}")
            
            # Progressive backoff based on usage percentage
            if self.request_counter > self.max_requests_per_minute * 0.9:
                cooldown_time = 180  # 3 minutes
                self.logger.warning(f"90%+ of rate limit reached, entering extended cooldown ({cooldown_time}s)")
            elif self.request_counter > self.max_requests_per_minute * 0.8:
                cooldown_time = 120  # 2 minutes
                self.logger.warning(f"80%+ of rate limit reached, entering medium cooldown ({cooldown_time}s)")
            elif self.request_counter > self.max_requests_per_minute * 0.7:
                cooldown_time = 60   # 1 minute
                self.logger.warning(f"70%+ of rate limit reached, entering short cooldown ({cooldown_time}s)")
            else:
                return True  # Continue if under 70%
                
            self.cooldown_active = True
            self.cooldown_until = current_time + cooldown_time
            return False
                
        self.request_counter += 1
        return True

    async def send_heartbeat(self):
        """Send a heartbeat to keep the connection alive"""
        if not await self.check_rate_limit():
            return
            
        try:
            # Use a lightweight call that requires no actual API request
            # Just check if the connection is still alive without making a request
            if self.thalex.connected():
                self.logger.debug("Connection is still alive - heartbeat check successful")
                self.last_heartbeat = time.time()
            else:
                self.logger.warning("Connection appears closed during heartbeat check")
                raise ConnectionError("WebSocket connection is closed")
        except Exception as e:
            self.logger.error(f"Error during heartbeat check: {str(e)}")

    async def listen_task(self):
        """Main websocket listener task"""
        self.logger.info("Starting listen task")
        
        # Track consecutive connection issues for exponential backoff
        consecutive_errors = 0
        
        while True:
            try:
                # Connection with retry logic
                max_retries = BOT_CONFIG["connection"]["max_retries"]
                retry_count = 0
                retry_delay = BOT_CONFIG["connection"]["retry_delay"] * (2 ** min(consecutive_errors, 6))  # Exponential backoff
                
                while retry_count < max_retries:
                    try:
                        self.logger.info("Connecting to Thalex websocket...")
                        await self.thalex.connect()
                        self.logger.info("Connection established")
                        consecutive_errors = 0  # Reset on successful connection
                        break
                    except Exception as e:
                        retry_count += 1
                        if retry_count >= max_retries:
                            self.logger.error(f"Failed to connect after {max_retries} attempts: {str(e)}")
                            raise
                        self.logger.warning(f"Connection attempt {retry_count} failed: {str(e)}. Retrying in {retry_delay}s...")
                        await asyncio.sleep(retry_delay)
                        retry_delay = min(300, retry_delay * 2)  # Exponential backoff capped at 5 minutes
                
                # Initialize connection
                try:
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
                    
                    # Subscribe to channels - allow multiple attempts with backoff
                    subscription_success = False
                    subscription_attempts = 0
                    while not subscription_success and subscription_attempts < 3:
                        try:
                            subscription_attempts += 1
                            await self.thalex.private_subscribe(
                                ["session.orders", "account.portfolio", "account.trade_history"], 
                                id=CALL_IDS["subscribe"]
                            )
                            await self.thalex.public_subscribe(
                                [f"ticker.{self.perp_name}.raw", f"price_index.{MARKET_CONFIG['underlying']}"], 
                                id=CALL_IDS["subscribe"]
                            )
                            subscription_success = True
                        except Exception as sub_err:
                            if subscription_attempts >= 3:
                                raise
                            self.logger.warning(f"Subscription attempt {subscription_attempts} failed: {str(sub_err)}. Retrying...")
                            await asyncio.sleep(2 * subscription_attempts)
                    
                    # Reset rate limit counters after reconnection
                    self.request_counter = 0
                    self.request_reset_time = time.time()
                    self.cooldown_active = False
                    self.last_heartbeat = time.time()  # Reset heartbeat timer after connection
                    
                    # Main listen loop
                    while True:
                        try:
                            # Check if we need to send a heartbeat
                            current_time = time.time()
                            if current_time - self.last_heartbeat > self.heartbeat_interval:
                                await self.send_heartbeat()
                            
                            # Check for rate limiting - if in cooldown, just keep listening without sending requests
                            if self.cooldown_active and current_time < self.cooldown_until:
                                # Still receive messages during cooldown, but don't send any requests
                                try:
                                    msg = await asyncio.wait_for(self.thalex.receive(), timeout=1.0)
                                    # Process the message minimally during cooldown
                                    msg_data = json.loads(msg)
                                    if "channel_name" in msg_data and msg_data.get("channel_name", "").startswith("ticker."):
                                        # Process ticker updates even during cooldown
                                        await self.handle_ticker_update(msg_data["notification"])
                                    elif "channel_name" in msg_data and msg_data.get("channel_name", "") == "account.portfolio":
                                        # Process portfolio updates during cooldown
                                        await self.handle_portfolio_update(msg_data["notification"])
                                except asyncio.TimeoutError:
                                    # Just continue the loop if no message received during timeout
                                    pass
                                continue
                                
                            msg = await self.thalex.receive()
                            msg = json.loads(msg)
                            
                            if "channel_name" in msg:
                                channel = msg["channel_name"]
                                self.logger.debug(f"Received notification from channel: {channel}")
                                await self.handle_notification(channel, msg["notification"])
                            elif "result" in msg:
                                self.logger.debug(f"Received result for message ID: {msg.get('id')}")
                                await self.handle_result(msg["result"], msg.get("id"))
                            elif "error" in msg:
                                error_msg = msg['error'].get('message', '')
                                error_code = msg['error'].get('code', -1)
                                
                                if error_code == 4 or "throttle exceeded" in error_msg.lower() or "too many request" in error_msg.lower():
                                    self.logger.warning(f"Rate limit hit (code {error_code}), activating cooldown")
                                    self.cooldown_active = True
                                    self.cooldown_until = time.time() + 300  # 5 minute cooldown
                                else:
                                    self.logger.error(f"Received error: {msg['error']} for message ID: {msg.get('id')}")
                                
                                await self.handle_error(msg["error"], msg.get("id"))
                            else:
                                self.logger.warning(f"Received unknown message type: {msg}")
                                
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
                            await asyncio.sleep(1)
                    
                except Exception as e:
                    self.logger.error(f"Error in connection initialization: {str(e)}")
                    consecutive_errors += 1
                    raise
                    
            except Exception as e:
                self.logger.error(f"Fatal error in listen task: {str(e)}")
                # Cancel all orders before retrying
                try:
                    await self.order_manager.cancel_all_orders()
                except:
                    pass
                    
                # If rate limited, add extra delay with exponential backoff
                if "throttle exceeded" in str(e).lower() or "too many request" in str(e).lower() or "code 4" in str(e).lower():
                    consecutive_errors += 1
                    cooldown = min(600, 120 * (2 ** min(consecutive_errors-1, 3)))  # Exponential backoff up to 10 minutes
                    self.logger.warning(f"Rate limit detected, cooling down for {cooldown} seconds (consecutive errors: {consecutive_errors})")
                    
                    # Set cooldown state
                    self.cooldown_active = True
                    self.cooldown_until = time.time() + cooldown
                    self.request_counter = self.max_requests_per_minute  # Mark as at limit
                    
                    await asyncio.sleep(cooldown)
                else:
                    # Regular connection error - use exponential backoff based on consecutive errors
                    backoff_time = min(300, 5 * (2 ** min(consecutive_errors, 6)))
                    self.logger.warning(f"Connection error, waiting {backoff_time}s before reconnect (consecutive errors: {consecutive_errors})")
                    await asyncio.sleep(backoff_time)
                    
                continue  # Retry the entire connection process

    async def handle_notification(self, channel: str, notification: Union[Dict, List[Dict]]):
        """Handle incoming notifications from different channels.
        
        Args:
            channel: The notification channel name
            notification: The notification payload
        """
        try:
            if not isinstance(notification, (dict, list)):
                logging.error(f"Invalid notification format: {type(notification)}")
                return
            
            if channel.startswith("ticker."):
                await self.handle_ticker_update(notification)
            elif channel.startswith("price_index."):
                await self.handle_index_update(notification)
            elif channel == "session.orders":
                if isinstance(notification, list):
                    for order_data in notification:
                        await self.handle_order_update(order_data)
                else:
                    await self.handle_order_update(notification)
            elif channel == "account.portfolio":
                await self.handle_portfolio_update(notification)
            elif channel == "account.trade_history":
                await self.handle_trade_update(notification)
            else:
                logging.error(f"Notification for unknown channel: {channel}")
        except Exception as e:
            logging.error(f"Error processing notification: {str(e)}")

    async def handle_ticker_update(self, ticker_data: Dict):
        """Process ticker updates"""
        try:
            self.ticker = Ticker(ticker_data)
            
            # Update price history
            if self.ticker.mark_price > 0:
                self.price_history.append(self.ticker.mark_price)
                
                # Update technical analysis
                self.technical_analysis.update(self.ticker.mark_price)
                
                # Calculate market conditions
                market_conditions = {
                    "volatility": self.technical_analysis.get_volatility(),
                    "is_volatile": self.technical_analysis.is_volatile(),
                    "is_trending": self.technical_analysis.is_trending(),
                    "trend_direction": self.technical_analysis.get_trend_direction(),
                    "trend_strength": self.technical_analysis.get_trend_strength(),
                    "mean_reverting_signal": self.technical_analysis.is_mean_reverting(),
                    "zscore": self.technical_analysis.get_zscore()
                }
                
                # Update market maker
                self.market_maker.update_market_conditions(
                    volatility=market_conditions["volatility"],
                    market_impact=self.technical_analysis.get_market_impact()
                )
                
                # Check if we should update quotes
                if self.market_maker.should_update_quotes(self.current_quotes, self.ticker.mark_price):
                    await self.update_quotes(market_conditions)
            
            # Notify quote task
            async with self.quote_cv:
                self.quote_cv.notify()
                
        except Exception as e:
            self.logger.error(f"Error handling ticker update: {str(e)}")

    async def handle_index_update(self, index_data: Dict):
        """Process index price updates"""
        try:
            self.index = float(index_data["price"])
            async with self.quote_cv:
                self.quote_cv.notify()
        except Exception as e:
            self.logger.error(f"Error handling index update: {str(e)}")

    async def handle_order_update(self, order_data: Dict):
        """Process order updates"""
        try:
            order = Order(
                id=order_data["client_order_id"],
                price=order_data["price"],
                amount=order_data["amount"],
                status=OrderStatus(order_data["status"]),
                direction=order_data.get("direction", "")
            )
            
            await self.order_manager.update_order(order)
            
            # Update position if order is filled
            if order.status == OrderStatus.FILLED:
                direction_sign = 1 if order.direction == "buy" else -1
                new_position = self.market_maker.position_size + (direction_sign * order.amount)
                self.market_maker.update_position(new_position, order.price)
                
        except Exception as e:
            self.logger.error(f"Error handling order update: {str(e)}")

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
        """Process trade updates"""
        try:
            # Handle both single trade and list of trades
            trades = [trade_data] if isinstance(trade_data, dict) else trade_data
            
            for trade in trades:
                if trade.get("label") == MARKET_CONFIG["label"]:
                    # Update risk metrics
                    await self.risk_manager.update_trade_metrics(trade)
                    
                    # Check if we need to adjust our quoting
                    if not await self.risk_manager.check_risk_limits():
                        self.quoting_enabled = False
                        await self.cancel_all_quotes()
                        self.logger.warning("Quoting disabled due to risk limits")
                    
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
                self.logger.warning(f"Rate limit hit (code {error_code}), activating extended cooldown")
                
                # Calculate cooldown time based on consecutive rate limit errors
                self.request_counter = self.max_requests_per_minute  # Mark as at limit
                self.cooldown_active = True
                
                # Use longer cooldown for code 4 errors
                cooldown_time = 300  # 5 minutes
                self.cooldown_until = time.time() + cooldown_time
                
                self.logger.warning(f"Rate limit cooldown active until: {time.ctime(self.cooldown_until)}")
                
                # For order-related rate limits, cancel pending orders
                if cid and cid >= 100:
                    await self.order_manager.handle_order_error(error, cid)
                    
                # Cancel all orders to reduce load
                try:
                    self.logger.info("Cancelling all orders due to rate limiting")
                    await self.order_manager.cancel_all_orders()
                except Exception as cancel_err:
                    self.logger.error(f"Error cancelling orders during rate limit: {str(cancel_err)}")
                
                return
            
            # Handle other API errors
            self.logger.error(f"API error for message {cid}: {error}")
            
            if cid and cid >= 100:  # Order IDs
                await self.order_manager.handle_order_error(error, cid)
                
        except Exception as e:
            self.logger.error(f"Error handling error message: {str(e)}")

    async def update_quotes(self, market_conditions: Dict):
        """Generate and place new quotes"""
        try:
            if not self.quoting_enabled or not self.ticker:
                return
                
            # Generate quotes using Avellaneda-Stoikov model
            bid_quotes, ask_quotes = self.market_maker.generate_quotes(
                self.ticker.mark_price,
                market_conditions
            )
            
            # Validate quotes
            bid_quotes, ask_quotes = self.market_maker.validate_quotes(
                bid_quotes, ask_quotes, market_conditions
            )
            
            # Update current quotes tracking
            self.current_quotes = (bid_quotes, ask_quotes)
            
            # Place quotes through order manager
            await self.order_manager.update_quotes(
                self.perp_name,
                bid_quotes,
                ask_quotes,
                MARKET_CONFIG["label"]
            )
            
        except Exception as e:
            self.logger.error(f"Error updating quotes: {str(e)}")

    async def cancel_all_quotes(self):
        """Cancel all open quotes"""
        try:
            await self.order_manager.cancel_all_orders()
            self.current_quotes = ([], [])
        except Exception as e:
            self.logger.error(f"Error cancelling quotes: {str(e)}")

    async def quote_task(self):
        """Main quoting loop"""
        self.logger.info("Starting quote task")
        
        while True:
            try:
                # Wait for price updates
                async with self.quote_cv:
                    await self.quote_cv.wait()
                
                # Check if we have enough price history
                if len(self.price_history) < TRADING_PARAMS["volatility"]["min_samples"]:
                    continue
                
                # Check risk limits
                if not await self.risk_manager.check_risk_limits():
                    if self.quoting_enabled:
                        self.quoting_enabled = False
                        await self.cancel_all_quotes()
                    continue
                else:
                    self.quoting_enabled = True
                
                # Update market conditions
                market_conditions = {
                    "volatility": self.technical_analysis.get_volatility(),
                    "is_volatile": self.technical_analysis.is_volatile(),
                    "is_trending": self.technical_analysis.is_trending(),
                    "trend_direction": self.technical_analysis.get_trend_direction(),
                    "trend_strength": self.technical_analysis.get_trend_strength(),
                    "mean_reverting_signal": self.technical_analysis.is_mean_reverting(),
                    "zscore": self.technical_analysis.get_zscore()
                }
                
                # Update quotes if needed
                if self.market_maker.should_update_quotes(self.current_quotes, self.ticker.mark_price):
                    await self.update_quotes(market_conditions)
                
            except Exception as e:
                self.logger.error(f"Error in quote task: {str(e)}")
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
        asyncio.create_task(quoter.heartbeat_task())
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