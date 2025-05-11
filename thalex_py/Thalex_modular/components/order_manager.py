import asyncio
import time
from typing import List, Dict, Optional, Tuple
import thalex as th

from ..models.data_models import Order, OrderStatus, Quote
from ..config.market_config import (
    TRADING_CONFIG,
    MARKET_CONFIG,
    CALL_IDS
)
from ..thalex_logging import LoggerFactory

class OrderManager:
    """Order management component handling order placement and tracking"""
    
    def __init__(self, thalex: th.Thalex):
        # Thalex client
        self.thalex = thalex
        
        # Initialize logger
        self.logger = LoggerFactory.configure_component_logger(
            "order_manager",
            log_file="order_manager.log",
            high_frequency=True
        )
        
        # Order tracking
        self.orders: Dict[int, Order] = {}  # Main order tracking by ID
        self.active_bids: Dict[int, Order] = {}  # Active buy orders
        self.active_asks: Dict[int, Order] = {}  # Active sell orders
        
        # Order ID management
        self.next_order_id = 100
        self.order_id_lock = asyncio.Lock()
        
        # Operation control
        self.operation_semaphore = asyncio.Semaphore(TRADING_CONFIG["quoting"]["max_pending"])
        self.pending_operations = set()
        
        # Order metrics
        self.order_metrics = {
            "total_orders": 0,
            "filled_orders": 0,
            "cancelled_orders": 0,
            "amended_orders": 0,
            "rejected_orders": 0
        }
        
        # Tick size and instrument name
        self.tick_size = 0.0
        self.perp_name = None
        
        # Track mass quotes separately for better handling
        # Note: We're using individual limit orders now, but keeping this tracking for backward compatibility
        self.active_mass_quotes = {}
        self.last_mass_quote_id = None
        self.recent_mass_quotes = set()  # Track recently sent mass quotes to detect duplicates
        
        self.logger.info("Order manager initialized")
        
    def set_tick_size(self, tick_size: float) -> None:
        """Set tick size for order price alignment"""
        self.tick_size = tick_size
        self.logger.info(f"Tick size set to {tick_size}")

    def set_perp_name(self, perp_name: str) -> None:
        """Set perpetual instrument name"""
        self.perp_name = perp_name
        self.logger.info(f"Perpetual name set to {perp_name}")
        
    async def get_next_order_id(self) -> int:
        """Get next available order ID"""
        async with self.order_id_lock:
            order_id = self.next_order_id
            self.next_order_id += 1
            if self.next_order_id > 1000000:
                self.next_order_id = 100
            return order_id
            
    def round_to_tick(self, price: float) -> float:
        """Round price to nearest tick size"""
        if self.tick_size <= 0:
            return price
        return round(price / self.tick_size) * self.tick_size

    async def place_order(
        self,
        instrument: str,
        direction: str,
        price: float,
        amount: float,
        label: str,
        post_only: bool = True
    ) -> Optional[int]:
        """Place a new order with improved tracking and validation"""
        try:
            # Check if parent quoter is in cooldown mode
            if hasattr(self.thalex, "quoter") and getattr(self.thalex, "quoter", None) is not None:
                quoter = self.thalex.quoter
                if hasattr(quoter, "cooldown_active") and quoter.cooldown_active:
                    current_time = time.time()
                    if current_time < quoter.cooldown_until:
                        # For market making, only apply full cooldown for critical issues
                        # For normal rate limiting, we prioritize quote levels based on how close they are to market
                        # Level 1 (closest to market) quotes still go through
                        if price is None or price <= 0.0001 or self.is_level_one_quote(price, direction):
                            remaining_time = int(quoter.cooldown_until - current_time)
                            self.logger.info(f"Allowing level 1 quote despite cooldown period ({remaining_time}s remaining)")
                        else:
                            remaining_time = int(quoter.cooldown_until - current_time)
                            self.logger.warning(f"Skipping non-priority quote during cooldown period - {remaining_time}s remaining until {time.ctime(quoter.cooldown_until)}")
                            return None
                    else:
                        # Cooldown period has ended, reset the cooldown flag
                        quoter.cooldown_active = False
                        self.logger.info("Cooldown period has ended, resuming normal operation")
                
                # Increment request counter if tracking is enabled
                if hasattr(quoter, "request_counter"):
                    quoter.request_counter += 1
            
            # Get next order ID
            order_id = await self.get_next_order_id()
            
            # Validate amount is not too small
            if amount < 0.001:
                self.logger.warning(f"Order amount {amount} is below minimum. Rounding to 0.001")
                amount = 0.001
            
            # Check for market order (price is None)
            is_market_order = price is None
            
            # Validate price is above zero for limit orders
            if not is_market_order and price <= 0:
                self.logger.warning(f"Invalid price {price}. Skipping order")
                return None
            
            # For market orders, set a placeholder price for tracking
            if is_market_order:
                # Use different placeholders for tracking based on direction
                if direction == "buy":
                    price = 999999.0  # High price for buy
                else:
                    price = 0.01      # Low price for sell
                post_only = False     # Market orders can't be post-only
            
            # Align price and amount to tick size
            aligned_price = self.round_to_tick(price) if not is_market_order else price
            aligned_amount = round(amount / 0.001) * 0.001  # Align to 0.001 tick size
            
            # Create order object
            order = Order(
                id=order_id,
                price=aligned_price,
                amount=aligned_amount,
                status=OrderStatus.PENDING,
                direction=direction
            )
            
            # Add to tracking
            self.orders[order_id] = order
            if direction == "buy":
                self.active_bids[order_id] = order
            else:
                self.active_asks[order_id] = order
            
            # Update metrics
            self.order_metrics["total_orders"] += 1
            
            # Actually place the order via Thalex API
            try:
                if is_market_order:
                    self.logger.info(
                        f"Placing MARKET order: {direction} {aligned_amount:.3f} @ market "
                        f"(ID: {order_id}, label: {label})"
                    )
                    
                    await self.thalex.insert(
                        direction=th.Direction.BUY if direction == "buy" else th.Direction.SELL,
                        instrument_name=instrument,
                        amount=aligned_amount,
                        # Use default market handling
                        client_order_id=order_id,
                        id=order_id,
                        label=label,
                        post_only=False,  # Can't be post-only
                        collar="none"     # No price protection for market orders
                    )
                else:
                    self.logger.info(
                        f"Placing order: {direction} {aligned_amount:.3f} @ {aligned_price:.2f} "
                        f"(ID: {order_id}, post_only: {post_only}, label: {label})"
                    )
                    
                    await self.thalex.insert(
                        direction=th.Direction.BUY if direction == "buy" else th.Direction.SELL,
                        instrument_name=instrument,
                        amount=aligned_amount,
                        price=aligned_price,
                        client_order_id=order_id,
                        id=order_id,
                        label=label,
                        post_only=post_only,
                        collar="clamp"
                    )
                
                return order_id
            except Exception as api_error:
                # If API call fails, clean up tracking
                if direction == "buy":
                    self.active_bids.pop(order_id, None)
                else:
                    self.active_asks.pop(order_id, None)
                
                self.orders.pop(order_id, None)
                self.order_metrics["rejected_orders"] += 1
                
                self.logger.error(f"Thalex API error placing order: {str(api_error)}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error placing order: {str(e)}")
            return None

    async def cancel_order(self, order_id: int) -> bool:
        """Cancel an existing order"""
        try:
            if order_id not in self.orders:
                self.logger.warning(f"Order {order_id} not found for cancellation")
                return False
                
            order = self.orders[order_id]
            if not order.is_open():
                self.logger.warning(f"Order {order_id} already closed")
                return False
                
            try:
                # Send cancel request to Thalex
                await self.thalex.cancel(client_order_id=order_id, id=order_id)
            except Exception as e:
                if "order not found" in str(e).lower():
                    self.logger.info(f"Order {order_id} already cancelled on exchange")
                else:
                    raise
            
            # Update order status
            order.status = OrderStatus.CANCELLED
            
            # Remove from active orders
            if order.direction == "buy":
                self.active_bids.pop(order_id, None)
            else:
                self.active_asks.pop(order_id, None)
                
            # Update metrics
            self.order_metrics["cancelled_orders"] += 1
            
            self.logger.info(f"Cancelled order {order_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error cancelling order: {str(e)}")
            return False
            
    async def cancel_all_orders(self) -> None:
        """Cancel all active orders"""
        try:
            # Get all active order IDs
            active_orders = list(self.active_bids.keys()) + list(self.active_asks.keys())
            
            # Cancel each order
            for order_id in active_orders:
                await self.cancel_order(order_id)
                
            self.logger.info(f"Cancelled {len(active_orders)} orders")
            
        except Exception as e:
            self.logger.error(f"Error cancelling all orders: {str(e)}")
            
    async def update_order(self, order: Order) -> None:
        """Update order status"""
        try:
            if order.id not in self.orders:
                self.logger.warning(f"Order {order.id} not found for update")
                return
                
            old_order = self.orders[order.id]
            old_status = old_order.status
            
            # Update order
            self.orders[order.id] = order
            
            # Update active orders
            if order.status != old_status:
                if not order.is_open():
                    # Remove from active orders
                    if order.direction == "buy":
                        if order.id in self.active_bids:
                            del self.active_bids[order.id]
                    else:
                        if order.id in self.active_asks:
                            del self.active_asks[order.id]
                            
                    # Update metrics
                    if order.status == OrderStatus.FILLED:
                        self.order_metrics["filled_orders"] += 1
                    elif order.status == OrderStatus.CANCELLED:
                        self.order_metrics["cancelled_orders"] += 1
                else:
                    # Add to active orders
                    if order.direction == "buy":
                        self.active_bids[order.id] = order
                    else:
                        self.active_asks[order.id] = order
                    
            self.logger.debug(
                f"Updated order {order.id}: {old_status} -> {order.status}"
            )
            
        except Exception as e:
            self.logger.error(f"Error updating order: {str(e)}")
            
    async def handle_order_result(self, result: Dict, order_id: int) -> None:
        """Handle order operation results from API"""
        try:
            # Special handling for mass_quote response
            if order_id == CALL_IDS.get("mass_quote", 0) or order_id == 999:
                self.logger.info(f"Received mass_quote response with ID {order_id}")
                self.last_mass_quote_id = order_id
                
                # Log the full result for debugging 
                self.logger.info(f"Mass quote response type: {type(result)}, content: {result}")
                
                if isinstance(result, dict):
                    for key, value in result.items():
                        self.logger.info(f"Response key: {key}, value type: {type(value)}, value: {value}")
                
                # In some versions of the API, quotes information is in the root of the result
                quote_data_root = result.get("quotes", [])
                if quote_data_root:
                    self.logger.info(f"Found {len(quote_data_root)} quotes in the response")
                    
                    # Track quote IDs
                    quote_ids = []
                    
                    # Process each quote
                    for quote in quote_data_root:
                        if "quote_id" in quote:
                            quote_ids.append(quote["quote_id"])
                            
                        # Process bid side if present
                        if "b" in quote and quote["b"]:
                            bid_data = quote["b"]
                            bid_price = float(bid_data.get("p", 0))
                            bid_amount = float(bid_data.get("a", 0))
                            
                            if bid_price > 0 and bid_amount > 0:
                                quote_id = quote.get("quote_id", f"bid_{len(quote_ids)}")
                                bid_order = Order(
                                    id=quote_id,
                                    price=bid_price,
                                    amount=bid_amount,
                                    status=OrderStatus.OPEN,
                                    direction="buy"
                                )
                                self.orders[quote_id] = bid_order
                                self.active_bids[quote_id] = bid_order
                                
                        # Process ask side if present
                        if "a" in quote and quote["a"]:
                            ask_data = quote["a"]
                            ask_price = float(ask_data.get("p", 0))
                            ask_amount = float(ask_data.get("a", 0))
                            
                            if ask_price > 0 and ask_amount > 0:
                                quote_id = quote.get("quote_id", f"ask_{len(quote_ids)}")
                                ask_order = Order(
                                    id=quote_id,
                                    price=ask_price,
                                    amount=ask_amount,
                                    status=OrderStatus.OPEN,
                                    direction="sell"
                                )
                                self.orders[quote_id] = ask_order
                                self.active_asks[quote_id] = ask_order
                
                    # Store the quote IDs linked to this mass quote ID
                    if quote_ids:
                        self.active_mass_quotes[order_id] = quote_ids
                        self.logger.info(f"Tracked {len(quote_ids)} quotes from mass quote {order_id}")
                    
                    # Add to our set of recent mass quotes
                    self.recent_mass_quotes.add(order_id)
                    if len(self.recent_mass_quotes) > 10:  # Keep only the 10 most recent
                        oldest = min(self.recent_mass_quotes)
                        self.recent_mass_quotes.remove(oldest)
                else:
                    self.logger.warning(f"No quotes found in mass quote response. Format may be different than expected.")
                    
                    # Check if this is a simple success response (which would indicate the quotes are placed
                    # but details are not returned)
                    if "result" in result and result.get("result") == "ok":
                        self.logger.info("Mass quote was successful based on 'result': 'ok'")
                        
                        # Even without details, we know the quotes were placed
                        # Add to our set of recent mass quotes
                        self.recent_mass_quotes.add(order_id)
                        if len(self.recent_mass_quotes) > 10:
                            oldest = min(self.recent_mass_quotes)
                            self.recent_mass_quotes.remove(oldest)
                
                return
                
            # Check if order is in our tracking
            if order_id not in self.orders:
                # Special handling for mass quotes to avoid "order not found" warnings
                if order_id in self.recent_mass_quotes or order_id == CALL_IDS.get("mass_quote", 0) or order_id == 999:
                    self.logger.debug(f"Processing result for known mass quote ID {order_id}")
                    return
                
                self.logger.warning(f"Order {order_id} not found for result update")
                
                # Try to determine if this might be part of a mass quote
                if "order_id" in result and "client_order_id" not in result:
                    exchange_order_id = result.get("order_id", "unknown")
                    self.logger.info(f"Result may be for mass quote: exchange order ID {exchange_order_id}, client ID {order_id}")
                    return
                
                # If it's a regular order we should have tracked, try to recreate it
                if "status" in result and "price" in result and "amount" in result:
                    self.logger.info(f"Attempting to recover tracking for order {order_id}")
                    direction = result.get("direction", "")
                    new_order = Order(
                        id=order_id,
                        price=float(result.get("price", 0)),
                        amount=float(result.get("amount", 0)),
                        status=OrderStatus(result.get("status", "pending")),
                        direction=direction
                    )
                    
                    # Add to tracking
                    self.orders[order_id] = new_order
                    
                    # Add to side-specific tracking if order is still open
                    if new_order.is_open():
                        if direction == "buy":
                            self.active_bids[order_id] = new_order
                        elif direction == "sell":
                            self.active_asks[order_id] = new_order
                    
                    self.logger.info(f"Recovered tracking for order {order_id}")
                    
                return
                
            # Normal order handling continues here...
            # Extract order details from result
            order_data = result
            
            # Update order status if present in result
            if order_data.get("status"):
                # Create updated order
                updated_order = Order(
                    id=order_id,
                    price=float(order_data.get("price", 0)),
                    amount=float(order_data.get("amount", 0)),
                    status=OrderStatus(order_data.get("status", "pending")),
                    direction=order_data.get("direction", "")
                )
                
                # Update order in our tracking
                await self.update_order(updated_order)
                
            self.logger.debug(f"Processed result for order {order_id}")
            
        except Exception as e:
            self.logger.error(f"Error handling order result: {str(e)}")
            
    async def handle_order_error(self, error: Dict, order_id: int) -> None:
        """Handle order operation errors from API"""
        try:
            # Special handling for mass_quote errors
            if order_id == CALL_IDS.get("mass_quote", 0) or order_id == 999:  # Also handle hardcoded ID 999
                error_msg = error.get("message", "")
                error_code = error.get("code", -1)
                
                # If it's an unknown API method error, this is likely due to using a hardcoded ID
                # Just log it at debug level and don't treat it as a critical error
                if "unknown API method" in error_msg.lower():
                    self.logger.debug(f"Ignoring unknown API method error for ID {order_id} - likely using legacy ID")
                    return
                
                self.logger.error(f"Mass quote error (ID {order_id}): {error_code} - {error_msg}")
                
                # Clean up any tracking for this mass quote
                if order_id in self.active_mass_quotes:
                    quote_ids = self.active_mass_quotes.pop(order_id, [])
                    for quote_id in quote_ids:
                        # Clean up any associated quote orders
                        self.active_bids.pop(quote_id, None)
                        self.active_asks.pop(quote_id, None)
                        self.orders.pop(quote_id, None)
                    
                    self.logger.info(f"Cleaned up tracking for {len(quote_ids)} quotes associated with failed mass quote {order_id}")
                
                # If rate limit error, we should back off
                if "rate limit" in error_msg.lower() or "too many requests" in error_msg.lower():
                    # Notify quoter about rate limit if possible
                    if hasattr(self.thalex, "quoter") and getattr(self.thalex, "quoter", None) is not None:
                        quoter = self.thalex.quoter
                        if hasattr(quoter, 'cooldown_active'):
                            quoter.cooldown_active = True
                            quoter.cooldown_until = time.time() + 10  # 10-second cooldown
                            self.logger.warning("Set quoter cooldown due to rate limit error")
                
                return
                
            # Check if order is in our tracking
            if order_id not in self.orders:
                # Don't warn about mass quote IDs that we know about
                if order_id in self.recent_mass_quotes or order_id == CALL_IDS.get("mass_quote", 0) or order_id == 999:
                    return
                    
                self.logger.warning(f"Order {order_id} not found for error update")
                return
                
            error_msg = error.get("message", "")
            error_code = error.get("code", -1)
            
            # Handle specific error cases
            if "order not found" in error_msg.lower():
                # Order already canceled or never existed, remove from tracking
                if order_id in self.active_bids:
                    del self.active_bids[order_id]
                if order_id in self.active_asks:
                    del self.active_asks[order_id]
                    
                # Update order status
                order = self.orders[order_id]
                order.status = OrderStatus.CANCELLED
                
                self.logger.info(f"Order {order_id} marked as cancelled (not found on exchange)")
                
            else:
                # Generic error handling
                self.logger.error(f"API error for order {order_id}: {error_code} - {error_msg}")
                
        except Exception as e:
            self.logger.error(f"Error handling order error: {str(e)}")

    def is_level_one_quote(self, price: float, direction: str) -> bool:
        """
        Determine if this is a level 1 quote (closest to the market)
        Level 1 quotes are prioritized even during rate limiting
        
        Args:
            price: Quote price
            direction: 'buy' or 'sell'
            
        Returns:
            bool: True if this is a level 1 (highest priority) quote
        """
        try:
            # If we don't have a quote reference, allow all quotes
            if not hasattr(self.thalex, "quoter") or getattr(self.thalex, "quoter", None) is None:
                return True
                
            quoter = self.thalex.quoter
            if not hasattr(quoter, "ticker") or quoter.ticker is None:
                return True
                
            # For buy orders, check if this is close to the best bid
            if direction == "buy":
                if quoter.ticker.best_bid_price:
                    # Allow if within 2 ticks of best bid
                    return abs(price - quoter.ticker.best_bid_price) <= self.tick_size * 2
                    
            # For sell orders, check if this is close to the best ask
            elif direction == "sell":
                if quoter.ticker.best_ask_price:
                    # Allow if within 2 ticks of best ask
                    return abs(price - quoter.ticker.best_ask_price) <= self.tick_size * 2
                    
            # If we can't determine from best bid/ask, use mark price
            if quoter.ticker.mark_price:
                # For buys, allow if within 0.1% below mark
                if direction == "buy":
                    return price >= quoter.ticker.mark_price * 0.999
                # For sells, allow if within 0.1% above mark
                else:
                    return price <= quoter.ticker.mark_price * 1.001
                
            # If all else fails, allow the quote
            return True
            
        except Exception as e:
            self.logger.error(f"Error determining quote priority: {str(e)}")
            # On error, allow the quote to go through
            return True 