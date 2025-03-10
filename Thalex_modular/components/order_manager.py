import logging
import time
from typing import List, Dict, Optional, Tuple
import thalex as th
import asyncio

from ..models.data_models import Order, OrderStatus
from ..config.market_config import ORDERBOOK_CONFIG

class OrderManager:
    def __init__(self):
        self.orders: List[List[Order]] = [[], []]  # bids, asks
        self.order_cache: Dict[int, Order] = {}
        self.last_order_update: Dict[int, float] = {}
        self.client_order_id: int = 100
        self.max_orders: int = 50
        self.min_order_interval: float = 0.1
        self.last_order_time: float = 0
        self.tick: Optional[float] = None

    def set_tick_size(self, tick: float) -> None:
        """Set the tick size for price rounding"""
        self.tick = tick

    def round_to_tick(self, value: float) -> float:
        """Round price to valid tick size"""
        if not self.tick:
            raise ValueError("Tick size not set")
        return self.tick * round(value / self.tick)

    async def can_place_order(self) -> bool:
        """Check if new order can be placed based on limits"""
        current_time = time.time()
        
        # Check rate limiting
        if current_time - self.last_order_time < self.min_order_interval:
            return False
            
        # Check maximum order count
        open_orders = len([o for orders in self.orders for o in orders if o.is_open()])
        if open_orders >= self.max_orders:
            logging.warning(f"Maximum order count reached: {open_orders}")
            return False
            
        self.last_order_time = current_time
        return True

    def cleanup_cancelled_order(self, order: Order) -> None:
        """Clean up cancelled order from tracking"""
        if order.id in self.order_cache:
            del self.order_cache[order.id]
        if order.id in self.last_order_update:
            del self.last_order_update[order.id]

    def update_order(self, order: Order) -> bool:
        """Update order in tracking systems"""
        if not order:
            return False
            
        # Update cache
        self.order_cache[order.id] = order
        self.last_order_update[order.id] = time.time()
        
        # Update order lists
        for side in [0, 1]:
            for i, have in enumerate(self.orders[side]):
                if have.id == order.id:
                    self.orders[side][i] = order
                    return True
        return False

    def cleanup_expired_orders(self) -> None:
        """Remove expired orders from cache"""
        current_time = time.time()
        expired_orders = [
            oid for oid, last_time in self.last_order_update.items()
            if current_time - last_time > 300  # 5 minutes
        ]
        for oid in expired_orders:
            if oid in self.order_cache:
                self.cleanup_cancelled_order(self.order_cache[oid])

    async def adjust_quotes(self, thalex: th.Thalex, desired: List[List[th.SideQuote]], 
                          perp_name: str) -> None:
        """Adjust quotes with rate limiting and validation"""
        logging.info(f"Adjusting quotes for {perp_name} with tick size {self.tick}")
        
        if not self.tick:
            logging.error("Tick size not set! Cannot place orders.")
            return
            
        if not await self.can_place_order():
            logging.debug("Rate limit or order count exceeded")
            return
            
        for side_i, side in enumerate([th.Direction.BUY, th.Direction.SELL]):
            orders = self.orders[side_i]
            quotes = desired[side_i]
            
            logging.info(f"Processing {len(quotes)} {side.name} quotes, have {len(orders)} existing orders")
            
            # Cancel excess orders
            for i in range(len(quotes), len(orders)):
                if orders[i].is_open():
                    try:
                        logging.info(f"Cancelling excess {side.name} order {orders[i].id}")
                        await thalex.cancel(client_order_id=orders[i].id, id=orders[i].id)
                        self.cleanup_cancelled_order(orders[i])
                    except Exception as e:
                        logging.error(f"Error cancelling order {orders[i].id}: {str(e)}")
            
            # Process each quote level
            for q_lvl, q in enumerate(quotes):
                if q.a <= 0 or q.p <= 0:
                    logging.error(f"Invalid quote: price={q.p}, amount={q.a}")
                    continue
                    
                if len(orders) <= q_lvl or not orders[q_lvl].is_open():
                    if not await self.can_place_order():
                        logging.debug("Rate limit reached during quote processing")
                        return
                        
                    # Create new order
                    client_order_id = self.client_order_id
                    self.client_order_id += 1
                    
                    new_order = Order(client_order_id, q.p, q.a)
                    if len(orders) <= q_lvl:
                        orders.append(new_order)
                    else:
                        orders[q_lvl] = new_order
                        
                    # Place order with retry logic
                    max_retries = 3
                    retry_delay = 0.5
                    
                    logging.info(f"Placing new {side.name} order: {q.a} @ {q.p} (ID: {client_order_id})")
                    
                    for retry in range(max_retries):
                        try:
                            await thalex.insert(
                                direction=side,
                                instrument_name=perp_name,
                                amount=q.a,
                                price=q.p,
                                post_only=True,
                                client_order_id=client_order_id,
                                id=client_order_id
                            )
                            self.order_cache[client_order_id] = new_order
                            self.last_order_update[client_order_id] = time.time()
                            logging.info(f"Order placed: {side.name} {q.a} @ {q.p}")
                            break
                        except Exception as e:
                            if retry < max_retries - 1:
                                logging.warning(f"Order submission failed (retry {retry+1}/{max_retries}): {str(e)}")
                                await asyncio.sleep(retry_delay)
                                retry_delay *= 2  # Exponential backoff
                            else:
                                logging.error(f"Order submission failed after {max_retries} retries: {str(e)}")
                    
                elif abs(orders[q_lvl].price - q.p) > ORDERBOOK_CONFIG["amend_threshold"] * self.tick:
                    # Amend existing order with retry logic
                    max_retries = 3
                    retry_delay = 0.5
                    
                    logging.info(f"Amending {side.name} order {orders[q_lvl].id}: {q.a} @ {q.p} (old price: {orders[q_lvl].price})")
                    
                    for retry in range(max_retries):
                        try:
                            await thalex.amend(
                                amount=q.a,
                                price=q.p,
                                client_order_id=orders[q_lvl].id,
                                id=orders[q_lvl].id
                            )
                            orders[q_lvl].price = q.p
                            orders[q_lvl].amount = q.a
                            self.order_cache[orders[q_lvl].id] = orders[q_lvl]
                            self.last_order_update[orders[q_lvl].id] = time.time()
                            logging.info(f"Order amended: {side.name} {q.a} @ {q.p}")
                            break
                        except Exception as e:
                            if retry < max_retries - 1:
                                logging.warning(f"Order amendment failed (retry {retry+1}/{max_retries}): {str(e)}")
                                await asyncio.sleep(retry_delay)
                                retry_delay *= 2  # Exponential backoff
                            else:
                                logging.error(f"Order amendment failed after {max_retries} retries: {str(e)}")
                else:
                    logging.debug(f"No need to amend {side.name} order {orders[q_lvl].id}: price difference {abs(orders[q_lvl].price - q.p)} < threshold {ORDERBOOK_CONFIG['amend_threshold'] * self.tick}")

    def get_next_client_order_id(self) -> int:
        """Get and increment client order ID"""
        current_id = self.client_order_id
        self.client_order_id += 1
        return current_id

    def order_from_data(self, data: Dict) -> Optional[Order]:
        """Create Order object from API data"""
        try:
            client_order_id = data["client_order_id"]
            price = float(data["price"])
            amount = float(data["amount"])
            status = OrderStatus(data["status"])
            
            if price <= 0 or amount <= 0:
                logging.error(f"Invalid order data: price={price}, amount={amount}")
                return None
                
            order = Order(client_order_id, price, amount, status)
            
            # Update cache
            self.order_cache[client_order_id] = order
            self.last_order_update[client_order_id] = time.time()
            
            return order
        except (KeyError, ValueError) as e:
            logging.error(f"Error parsing order data: {str(e)}")
            return None
