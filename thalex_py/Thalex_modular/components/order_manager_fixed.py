import logging
import time
from typing import List, Dict, Optional, Tuple
import thalex as th
import asyncio

from ..models.data_models import Order, OrderStatus, Quote
from ..config.market_config import (
    ORDERBOOK_CONFIG,
    MARKET_CONFIG,
    QUOTING_CONFIG,
    TRADING_CONFIG
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
                        if price <= 0.0001 or self.is_level_one_quote(price, direction):
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
            
            # Validate price is above zero
            if price <= 0:
                self.logger.warning(f"Invalid price {price}. Skipping order")
                return None
            
            # Align price and amount to tick size
            aligned_price = self.round_to_tick(price)
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