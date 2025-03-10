import asyncio
import json
import logging
import socket
import time
from typing import Optional, Dict, List
import websockets

import thalex as th
from .legacy_quoter_files.keys import *

from .Thalex_modular.config.market_config import (
    MARKET_CONFIG, CALL_IDS, RISK_LIMITS
)
from .Thalex_modular.models.data_models import Ticker, Order, OrderStatus
from .Thalex_modular.components.risk_manager import RiskManager
from .Thalex_modular.components.order_manager import OrderManager
from .Thalex_modular.components.quote_manager import QuoteManager

+-class PerpQuoter:
    def __init__(self, thalex: th.Thalex):
        self.thalex = thalex
        self.ticker: Optional[Ticker] = None
        self.index: Optional[float] = None
        self.quote_cv = asyncio.Condition()
        self.portfolio: Dict[str, float] = {}
        self.perp_name: Optional[str] = None
        
        # Initialize components
        self.risk_manager = RiskManager()
        self.order_manager = OrderManager()
        self.quote_manager = QuoteManager()
        
        # Performance tracking
        self.quoting_enabled = True
        self.last_quote_time = time.time()
        self.last_position_check = time.time()

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
                self.quote_manager.set_tick_size(tick_size)
                logging.info(f"Found perpetual {self.perp_name} with tick size {tick_size}")
                return
                
        raise ValueError(f"Perpetual {MARKET_CONFIG['underlying']} not found")

    async def listen_task(self):
        """Main websocket listener task"""
        logging.info("Starting listen task")
        
        # Connection with retry logic
        max_retries = 5
        retry_count = 0
        retry_delay = 1
        
        while retry_count < max_retries:
            try:
                logging.info("Connecting to Thalex websocket...")
                await self.thalex.connect()
                logging.info("Connection established")
                break
            except Exception as e:
                retry_count += 1
                if retry_count >= max_retries:
                    logging.error(f"Failed to connect after {max_retries} attempts: {str(e)}")
                    raise
                logging.warning(f"Connection attempt {retry_count} failed: {str(e)}. Retrying in {retry_delay}s...")
                await asyncio.sleep(retry_delay)
                retry_delay = min(30, retry_delay * 2)  # Exponential backoff capped at 30s
        
        # Wait for instruments with timeout
        try:
            await asyncio.wait_for(self.await_instruments(), timeout=30)
        except asyncio.TimeoutError:
            logging.error("Timeout waiting for instruments")
            raise RuntimeError("Timeout waiting for instruments")
        
        # Initialize connection with retry logic
        retry_count = 0
        retry_delay = 1
        
        while retry_count < max_retries:
            try:
                logging.info("Authenticating with Thalex API...")
                await self.thalex.login(
                    key_ids[MARKET_CONFIG["network"]], 
                    private_keys[MARKET_CONFIG["network"]], 
                    id=CALL_IDS["login"]
                )
                logging.info("Authentication successful")
                break
            except Exception as e:
                retry_count += 1
                if retry_count >= max_retries:
                    logging.error(f"Authentication failed after {max_retries} attempts: {str(e)}")
                    raise
                logging.warning(f"Authentication attempt {retry_count} failed: {str(e)}. Retrying in {retry_delay}s...")
                await asyncio.sleep(retry_delay)
                retry_delay = min(30, retry_delay * 2)
        
        # Set cancel on disconnect
        await self.thalex.set_cancel_on_disconnect(6, id=CALL_IDS["set_cod"])
        
        # Subscribe to channels with retry logic
        retry_count = 0
        retry_delay = 1
        
        while retry_count < max_retries:
            try:
                logging.info("Subscribing to private channels...")
                await self.thalex.private_subscribe(
                    ["session.orders", "account.portfolio", "account.trade_history"], 
                    id=CALL_IDS["subscribe"]
                )
                logging.info("Subscribing to public channels...")
                await self.thalex.public_subscribe(
                    [f"ticker.{self.perp_name}.raw", f"price_index.{MARKET_CONFIG['underlying']}"], 
                    id=CALL_IDS["subscribe"]
                )
                logging.info("Successfully subscribed to all channels")
                break
            except Exception as e:
                retry_count += 1
                if retry_count >= max_retries:
                    logging.error(f"Channel subscription failed after {max_retries} attempts: {str(e)}")
                    raise
                logging.warning(f"Subscription attempt {retry_count} failed: {str(e)}. Retrying in {retry_delay}s...")
                await asyncio.sleep(retry_delay)
                retry_delay = min(30, retry_delay * 2)
        
        # Main message loop with heartbeat
        last_heartbeat = time.time()
        heartbeat_interval = 30  # Send heartbeat every 30 seconds
        
        while True:
            # Check position periodically
            current_time = time.time()
            if current_time - self.last_position_check > 5:  # Every 5 seconds
                await self.manage_position()
                self.last_position_check = current_time
            
            # Keep connection alive with periodic activity instead of ping
            if current_time - last_heartbeat > heartbeat_interval:
                try:
                    logging.debug("Sending heartbeat activity")
                    # Instead of ping, we'll use a lightweight API call
                    await self.thalex.system_info(id=999)
                    last_heartbeat = current_time
                except Exception as e:
                    logging.error(f"Error sending heartbeat: {str(e)}")
                    raise  # Let the main loop handle reconnection
            
            # Set a timeout for receiving messages
            try:
                msg = await asyncio.wait_for(self.thalex.receive(), timeout=60)
                msg = json.loads(msg)
                
                if "channel_name" in msg:
                    await self.notification(msg["channel_name"], msg["notification"])
                elif "result" in msg:
                    await self.result_callback(msg["result"], msg.get("id"))
                else:
                    await self.error_callback(msg["error"], msg.get("id"))
            except asyncio.TimeoutError:
                logging.warning("No message received for 60 seconds, reconnecting...")
                raise websockets.ConnectionClosed(1001, "Receive timeout")
            except Exception as e:
                logging.error(f"Error in message loop: {str(e)}")
                raise  # Let the main loop handle reconnection

    async def quote_task(self):
        """Main quoting loop"""
        logging.info("Starting quote task")
        while True:
            try:
                async with self.quote_cv:
                    await self.quote_cv.wait()
                    
                current_time = time.time()
                if current_time - self.last_quote_time < 0.5:  # Minimum quote interval
                    continue
                    
                self.last_quote_time = current_time
                    
                if not self.quoting_enabled:
                    logging.debug("Quoting disabled, skipping quote generation")
                    continue
                    
                if not self.ticker or not self.index:
                    logging.debug("Missing ticker or index data, skipping quote generation")
                    continue
                    
                # Check risk limits
                within_limits, reason = self.risk_manager.check_position_limits(self.ticker.mark_price)
                if not within_limits:
                    logging.warning(f"Risk limit breach: {reason}")
                    await self.handle_risk_breach()
                    continue
                    
                # Get market conditions from technical analysis
                market_conditions = self.quote_manager.technical_analysis.get_market_conditions(self.ticker.mark_price)
                
                # Update market data in components
                self.quote_manager.update_market_data(self.ticker.mark_price, self.ticker.volume)
                self.quote_manager.update_position(self.risk_manager.position_size, self.risk_manager.entry_price)
                
                # Update market maker with volatility from market conditions
                self.quote_manager.market_maker.update_market_conditions(
                    volatility=market_conditions.get("volatility", 0.001),
                    market_impact=market_conditions.get("atr", 0) / self.ticker.mark_price if self.ticker.mark_price > 0 else 0
                )
                
                # Generate and validate quotes using Avellaneda-Stoikov model
                try:
                    bid_price, ask_price, bid_size, ask_size = self.quote_manager.market_maker.calculate_optimal_quotes(
                        self.ticker.mark_price
                    )
                    logging.info(f"Generated quotes: bid={bid_price:.2f}({bid_size:.4f}), ask={ask_price:.2f}({ask_size:.4f})")
                except Exception as e:
                    logging.error(f"Error calculating optimal quotes: {str(e)}")
                    continue
                
                if bid_price <= 0 or ask_price <= 0:
                    logging.warning(f"Invalid quote prices generated: bid={bid_price}, ask={ask_price}")
                    continue
                
                if bid_size <= 0 or ask_size <= 0:
                    logging.warning(f"Invalid quote sizes generated: bid_size={bid_size}, ask_size={ask_size}")
                    continue

                # Create quote lists
                quotes = [
                    [th.SideQuote(price=bid_price, amount=bid_size)],
                    [th.SideQuote(price=ask_price, amount=ask_size)]
                ]
                
                if not self.quote_manager.validate_quotes(quotes):
                    logging.warning("Quote validation failed")
                    continue
                
                # Place/amend orders
                await self.order_manager.adjust_quotes(self.thalex, quotes, self.perp_name)
                
            except Exception as e:
                logging.error(f"Unexpected error in quote task: {str(e)}")
                await asyncio.sleep(1)  # Add delay to prevent rapid retries on persistent errors

    async def notification(self, channel: str, notification: Dict):
        """Handle incoming notifications"""
        try:
            if channel.startswith("ticker."):
                self.ticker = Ticker.from_dict(notification)
                self.risk_manager.update_market_data(self.ticker.mark_price, self.ticker.volume)
                logging.debug(f"Ticker update: {self.ticker.mark_price}")
                
                # Notify quote task
                async with self.quote_cv:
                    self.quote_cv.notify()
                    
            elif channel.startswith("price_index."):
                self.index = float(notification["price"])
                logging.debug(f"Index update: {self.index}")
                
                # Notify quote task
                async with self.quote_cv:
                    self.quote_cv.notify()
                    
            elif channel == "session.orders":
                await self.handle_orders_update(notification)
                
            elif channel == "account.portfolio":
                self.handle_portfolio_update(notification)
                
            elif channel == "account.trade_history":
                await self.handle_trades_update(notification)
                
        except Exception as e:
            logging.error(f"Error processing notification: {str(e)}")

    async def handle_orders_update(self, orders: List[Dict]):
        """Process order updates"""
        self.order_manager.cleanup_expired_orders()
        
        for order_data in orders:
            order = self.order_manager.order_from_data(order_data)
            if not order:
                continue
                
            if not self.order_manager.update_order(order):
                logging.warning(f"Order not found in tracking: {order.id}")
                
            if order.status == OrderStatus.FILLED:
                logging.info(f"Order filled: {order.id} at price {order.price} amount {order.amount}")
                # Notify quote task to update quotes
                async with self.quote_cv:
                    self.quote_cv.notify()
            elif order.status in [OrderStatus.CANCELLED, OrderStatus.CANCELLED_PARTIALLY_FILLED]:
                self.order_manager.cleanup_cancelled_order(order)

    def handle_portfolio_update(self, portfolio: List[Dict]):
        """Update portfolio positions"""
        for position in portfolio:
            instrument = position.get("instrument_name")
            pos_size = position.get("position")
            
            if not instrument or pos_size is None:
                continue
                
            try:
                pos_size = float(pos_size)
                self.portfolio[instrument] = pos_size
                
                if instrument == self.perp_name:
                    old_position = self.risk_manager.position_size
                    self.risk_manager.position_size = pos_size
                    
                    if old_position != pos_size:
                        logging.info(f"Position updated: {pos_size}")
                        # Notify quote task to update quotes based on new position
                        asyncio.create_task(self.notify_quote_task())
                    
            except ValueError as e:
                logging.error(f"Error parsing position size: {str(e)}")

    async def notify_quote_task(self):
        """Helper to notify quote task"""
        async with self.quote_cv:
            self.quote_cv.notify()

    async def handle_trades_update(self, trades: List[Dict]):
        """Process trade updates"""
        for trade in trades:
            if trade.get("label") != MARKET_CONFIG["label"]:
                continue
                
            try:
                amount = float(trade["amount"])
                price = float(trade["price"])
                direction = trade["direction"]
                
                # Update position tracking
                trade_size = -amount if direction == "sell" else amount
                self.risk_manager.update_position(trade_size, price)
                
                logging.info(f"Trade executed: {direction} {amount} @ {price}")
                
                # Notify quote task
                async with self.quote_cv:
                    self.quote_cv.notify()
                
            except Exception as e:
                logging.error(f"Error processing trade: {str(e)}")

    async def handle_risk_breach(self):
        """Handle risk limit breaches"""
        logging.warning("Handling risk breach - temporarily disabling quoting")
        self.quoting_enabled = False
        
        # Calculate reduction size (50% of current position)
        reduction_size = self.risk_manager.position_size * 0.5
        if abs(reduction_size) < 0.001:  # Too small to reduce
            self.quoting_enabled = True
            return
            
        direction = th.Direction.SELL if reduction_size > 0 else th.Direction.BUY
        
        client_order_id = self.order_manager.get_next_client_order_id()
        logging.info(f"Reducing position by {abs(reduction_size)}")
        
        await self.thalex.insert(
            direction=direction,
            instrument_name=self.perp_name,
            amount=abs(reduction_size),
            price=self.ticker.mark_price,
            client_order_id=client_order_id,
            id=client_order_id
        )
        
        # Re-enable quoting after a short delay
        await asyncio.sleep(5)
        self.quoting_enabled = True

    async def manage_position(self):
        """Monitor and manage open positions"""
        if not self.ticker:
            return
            
        position_size = self.risk_manager.position_size
        if abs(position_size) < 0.001:  # Effectively no position
            return
            
        logging.debug(f"Managing position: {position_size}")
            
        # Check stop loss
        if self.risk_manager.check_stop_loss(self.ticker.mark_price):
            logging.warning("Stop loss triggered")
            await self.close_position("Stop loss triggered")
            return
            
        # Check take profit
        take_profit_triggered, reason = self.risk_manager.check_take_profit(self.ticker.mark_price)
        if take_profit_triggered:
            logging.info(f"Take profit triggered: {reason}")
            await self.close_position(f"Take profit triggered: {reason}")
            return
            
        # Check if rebalance needed
        should_rebalance, reason = self.risk_manager.should_rebalance()
        if should_rebalance:
            logging.info(f"Rebalance needed: {reason}")
            await self.rebalance_position()

    async def close_position(self, reason: str):
        """Close entire position"""
        position_size = self.risk_manager.position_size
        if abs(position_size) < 0.001:  # Effectively no position
            return
            
        direction = th.Direction.SELL if position_size > 0 else th.Direction.BUY
        client_order_id = self.order_manager.get_next_client_order_id()
        
        logging.info(f"Closing position: {reason}")
        await self.thalex.insert(
            direction=direction,
            instrument_name=self.perp_name,
            amount=abs(position_size),
            price=self.ticker.mark_price,
            client_order_id=client_order_id,
            id=client_order_id
        )

    async def rebalance_position(self):
        """Rebalance position to target size"""
        reduction_size = self.risk_manager.calculate_rebalance_size()
        if reduction_size <= 0:
            return
            
        direction = th.Direction.SELL if self.risk_manager.position_size > 0 else th.Direction.BUY
        client_order_id = self.order_manager.get_next_client_order_id()
        
        logging.info(f"Rebalancing position by {reduction_size}")
        await self.thalex.insert(
            direction=direction,
            instrument_name=self.perp_name,
            amount=reduction_size,
            price=self.ticker.mark_price,
            client_order_id=client_order_id,
            id=client_order_id
        )

    async def error_callback(self, error: str, cid: Optional[int] = None):
        """Handle API errors"""
        if cid and cid > 99:  # Order error
            logging.error(f"Order error ({cid}): {error}")
            if cid in self.order_manager.order_cache:
                order = self.order_manager.order_cache[cid]
                if order.is_open():
                    await self.thalex.cancel(client_order_id=cid, id=cid)
                self.order_manager.cleanup_cancelled_order(order)
        else:
            logging.error(f"API error ({cid}): {error}")
            if cid == CALL_IDS["login"]:
                raise RuntimeError("Login failed")

    async def result_callback(self, result: Dict, cid: Optional[int] = None):
        """Handle successful API responses"""
        if cid == CALL_IDS["login"]:
            logging.info("Successfully authenticated with Thalex")
        elif cid == CALL_IDS["instruments"]:
            logging.info(f"Initialized perpetual instrument: {self.perp_name}")
        elif cid == CALL_IDS["subscribe"]:
            logging.debug("Successfully subscribed to channels")
        elif cid and cid > 99:  # Order-related result
            logging.info(f"Order {cid} processed successfully")

async def main():
    """Main entry point"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    
    async def shutdown(thalex, tasks):
        """Graceful shutdown handler"""
        try:
            if thalex.connected():  # Call the connected method
                await thalex.cancel_session(id=CALL_IDS["cancel_session"])
                await thalex.disconnect()
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        except Exception as e:
            logging.error(f"Error during shutdown: {str(e)}")
    
    max_retries = 5
    retry_count = 0
    backoff_time = 1
    
    while True:
        try:
            logging.info("Creating Thalex client...")
            thalex = th.Thalex(network=MARKET_CONFIG["network"])
            logging.info("Connecting to Thalex API...")
            
            # Verify API keys before proceeding but don't modify them
            if not hasattr(keys, 'API_KEY') or not keys.API_KEY:
                logging.error("API_KEY is missing or empty in keys.py")
                raise ValueError("API_KEY is missing or empty")
                
            if not hasattr(keys, 'SECRET_KEY') or not keys.SECRET_KEY:
                logging.error("SECRET_KEY is missing or empty in keys.py")
                raise ValueError("SECRET_KEY is missing or empty")
            
            logging.info("Authenticating with API keys...")
            
            quoter = PerpQuoter(thalex)
            
            tasks = [
                asyncio.create_task(quoter.listen_task()),
                asyncio.create_task(quoter.quote_task()),
            ]
            
            logging.info(f"Starting on {MARKET_CONFIG['network']} {MARKET_CONFIG['underlying']}")
            # Reset retry count on successful connection
            retry_count = 0
            backoff_time = 1
            
            await asyncio.gather(*tasks)
            
        except (websockets.ConnectionClosed, socket.gaierror) as e:
            retry_count += 1
            if retry_count > max_retries:
                logging.error(f"Maximum connection retries ({max_retries}) exceeded. Exiting.")
                break
                
            logging.error(f"Connection error ({e}). Retry {retry_count}/{max_retries} in {backoff_time}s...")
            await shutdown(thalex, tasks)
            await asyncio.sleep(backoff_time)
            # Exponential backoff
            backoff_time = min(60, backoff_time * 2)  # Cap at 60 seconds
            
        except KeyboardInterrupt:
            logging.info("Shutting down...")
            await shutdown(thalex, tasks)
            break
            
        except ValueError as e:
            logging.error(f"Configuration error: {str(e)}")
            logging.info("Please fix the configuration and restart the bot.")
            break
            
        except Exception as e:
            retry_count += 1
            if retry_count > max_retries:
                logging.error(f"Maximum error retries ({max_retries}) exceeded. Exiting.")
                break
                
            logging.exception(f"Unexpected error: {str(e)}. Retry {retry_count}/{max_retries} in {backoff_time}s...")
            await shutdown(thalex, tasks)
            await asyncio.sleep(backoff_time)
            # Exponential backoff
            backoff_time = min(60, backoff_time * 2)  # Cap at 60 seconds

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("Shutting down gracefully...")
