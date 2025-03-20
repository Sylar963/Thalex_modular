import logging
import time
from typing import List, Dict, Optional, Tuple
import thalex as th
import asyncio

from ..models.data_models import Order, OrderStatus, Quote
from ..config.market_config import (
    ORDERBOOK_CONFIG,
    MARKET_CONFIG,
    QUOTING_CONFIG
)

class OrderManager:
    """Order management component handling order placement and tracking"""
    
    def __init__(self, thalex: th.Thalex):
        # Thalex client
        self.thalex = thalex
        
        # Order tracking
        self.orders: Dict[int, Order] = {}  # Main order tracking by ID
        self.active_bids: Dict[int, Order] = {}  # Active buy orders
        self.active_asks: Dict[int, Order] = {}  # Active sell orders
        
        # Order ID management
        self.next_order_id = 100
        self.order_id_lock = asyncio.Lock()
        
        # Operation control
        self.operation_semaphore = asyncio.Semaphore(ORDERBOOK_CONFIG["max_pending_operations"])
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
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        
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
        label: str
    ) -> Optional[int]:
        """Place a new order"""
        try:
            # Check if parent quoter is in cooldown mode
            if hasattr(self.thalex, "quoter") and getattr(self.thalex, "quoter", None) is not None:
                quoter = self.thalex.quoter
                if hasattr(quoter, "cooldown_active") and quoter.cooldown_active:
                    current_time = time.time()
                    if current_time < quoter.cooldown_until:
                        self.logger.info("Skipping order placement during cooldown period")
                        return None
                        
                # Increment request counter if tracking is enabled
                if hasattr(quoter, "request_counter"):
                    quoter.request_counter += 1
                        
            # Get next order ID
            order_id = await self.get_next_order_id()
            
            # Align price and amount to tick size
            aligned_price = self.round_to_tick(price)
            aligned_amount = round(amount / 0.001) * 0.001  # Align to 0.001 tick size
            
            # Create order object
            order = Order(
                id=order_id,
                price=aligned_price,
                amount=aligned_amount,  # Use aligned amount
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
            await self.thalex.insert(
                direction=th.Direction.BUY if direction == "buy" else th.Direction.SELL,
                instrument_name=instrument,
                amount=aligned_amount,  # Use aligned amount
                price=aligned_price,
                client_order_id=order_id,
                id=order_id,
                label=label,
                post_only=True,
                collar="clamp"  # Use clamp mode to handle price limits
            )
            
            self.logger.info(
                f"Placing order: {direction} {aligned_amount:.3f} @ {aligned_price:.2f} "
                f"(ID: {order_id})"
            )
            
            return order_id
            
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
            
    async def update_quotes(
        self,
        instrument: str,
        bid_quotes: List[Quote],
        ask_quotes: List[Quote],
        label: str
    ) -> None:
        """Update quotes by cancelling old orders and placing new ones"""
        try:
            # Cancel existing orders
            await self.cancel_all_orders()
            
            # Place new bid quotes
            for quote in bid_quotes:
                await self.place_order(
                    instrument=instrument,
                    direction="buy",
                    price=quote.price,
                    amount=quote.amount,
                    label=label
                )
                
            # Place new ask quotes
            for quote in ask_quotes:
                await self.place_order(
                    instrument=instrument,
                    direction="sell",
                    price=quote.price,
                    amount=quote.amount,
                    label=label
                )
                
            self.logger.info(
                f"Updated quotes: {len(bid_quotes)} bids, {len(ask_quotes)} asks"
            )
            
        except Exception as e:
            self.logger.error(f"Error updating quotes: {str(e)}")
            
    async def handle_order_result(self, result: Dict, order_id: int) -> None:
        """Handle order placement/cancellation result"""
        try:
            if "error" in result:
                self.logger.error(f"Order {order_id} error: {result['error']}")
                self.order_metrics["rejected_orders"] += 1
                
                if order_id in self.orders:
                    order = self.orders[order_id]
                    order.status = OrderStatus.CANCELLED
                    
                    if order.direction == "buy":
                        if order_id in self.active_bids:
                            del self.active_bids[order_id]
                    else:
                        if order_id in self.active_asks:
                            del self.active_asks[order_id]
            else:
                self.logger.debug(f"Order {order_id} result: {result}")
                
        except Exception as e:
            self.logger.error(f"Error handling order result: {str(e)}")
            
    async def handle_order_error(self, error: Dict, order_id: int) -> None:
        """Handle order error"""
        try:
            error_code = error.get('code', None)
            error_message = error.get('message', '')
            
            # Handle "order not found" errors (code 1)
            if error_code == 1 and "order not found" in error_message.lower():
                self.logger.info(f"Order {order_id} not found on exchange - cleaning up local state")
                
                # Clean up local order state
                if order_id in self.orders:
                    order = self.orders[order_id]
                    order.status = OrderStatus.CANCELLED
                    
                    # Remove from active orders
                    if order.direction == "buy":
                        self.active_bids.pop(order_id, None)
                    else:
                        self.active_asks.pop(order_id, None)
                        
                    # Update metrics
                    self.order_metrics["cancelled_orders"] += 1
                return
            
            # Handle other errors
            self.logger.error(f"Order {order_id} error: {error}")
            self.order_metrics["rejected_orders"] += 1
            
            if order_id in self.orders:
                order = self.orders[order_id]
                order.status = OrderStatus.CANCELLED
                
                if order.direction == "buy":
                    self.active_bids.pop(order_id, None)
                else:
                    self.active_asks.pop(order_id, None)
                    
        except Exception as e:
            self.logger.error(f"Error handling order error: {str(e)}")
            
    def get_order_metrics(self) -> Dict:
        """Get order performance metrics"""
        try:
            total_orders = self.order_metrics["total_orders"]
            if total_orders == 0:
                return self.order_metrics
                
            metrics = self.order_metrics.copy()
            metrics.update({
                "fill_rate": self.order_metrics["filled_orders"] / total_orders,
                "cancel_rate": self.order_metrics["cancelled_orders"] / total_orders,
                "reject_rate": self.order_metrics["rejected_orders"] / total_orders
            })
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error getting order metrics: {str(e)}")
            return self.order_metrics

    async def place_new_quote(self, quote: Quote, side: th.Direction, instrument_name: str) -> Optional[int]:
        """Place new quote with proper order tracking"""
        async with self.operation_semaphore:
            order_id = await self.get_next_order_id()
            if order_id in self.pending_operations:
                return None

            self.pending_operations.add(order_id)
            try:
                # Align price and amount to tick size
                aligned_price = self.round_to_tick(quote.price)
                aligned_amount = round(quote.amount / 0.001) * 0.001  # Align to 0.001 tick size
                
                # Create order object
                new_order = Order(
                    id=order_id,
                    price=aligned_price,
                    amount=aligned_amount,
                    status=OrderStatus.PENDING,
                    direction="buy" if side == th.Direction.BUY else "sell"
                )
                
                # Add to tracking before API call
                self.orders[order_id] = new_order
                if side == th.Direction.BUY:
                    self.active_bids[order_id] = new_order
                else:
                    self.active_asks[order_id] = new_order
                
                # Place order
                await self.thalex.insert(
                    direction=side,
                    instrument_name=instrument_name,
                    amount=aligned_amount,
                    price=aligned_price,
                    post_only=True,
                    label=MARKET_CONFIG["label"],
                    client_order_id=order_id,
                    id=order_id,
                    collar="clamp"
                )
                
                self.order_metrics["total_orders"] += 1
                return order_id
                
            except Exception as e:
                # Remove from tracking if API call fails
                if order_id in self.orders:
                    del self.orders[order_id]
                if side == th.Direction.BUY:
                    if order_id in self.active_bids:
                        del self.active_bids[order_id]
                else:
                    if order_id in self.active_asks:
                        del self.active_asks[order_id]
                self.logger.error(f"Error placing quote: {str(e)}")
                return None
                
            finally:
                self.pending_operations.remove(order_id)
                await asyncio.sleep(QUOTING_CONFIG["order_operation_interval"])
