"""
Hedge execution module for placing and managing hedge orders.
Handles the actual execution of orders on the exchange.
"""

from typing import Dict, List, Tuple, Optional, Union, Any
from enum import Enum
import time
import asyncio
from datetime import datetime
import uuid
import json

from .hedge_config import HedgeConfig
from ...thalex_logging import LoggerFactory

# Enum for order types
class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"

# Enum for order side
class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"

# Enum for order status
class OrderStatus(Enum):
    NEW = "new"
    OPEN = "open"
    FILLED = "filled"
    CANCELED = "canceled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class HedgeOrder:
    """Represents a hedge order"""
    
    def __init__(
        self,
        order_id: str,
        symbol: str,
        side: OrderSide,
        size: float,
        price: Optional[float] = None,
        order_type: OrderType = OrderType.MARKET,
        status: OrderStatus = OrderStatus.NEW,
        filled_size: float = 0.0,
        filled_price: float = 0.0,
        created_at: Optional[float] = None,
        updated_at: Optional[float] = None,
        metadata: Optional[Dict] = None
    ):
        """
        Initialize a hedge order
        
        Args:
            order_id: Unique order ID
            symbol: Asset symbol
            side: Buy or sell
            size: Order size
            price: Order price (for limit orders)
            order_type: Market or limit
            status: Order status
            filled_size: Amount filled
            filled_price: Average fill price
            created_at: Creation timestamp
            updated_at: Last update timestamp
            metadata: Additional order metadata
        """
        self.order_id = order_id
        self.symbol = symbol
        self.side = side
        self.size = size
        self.price = price
        self.order_type = order_type
        self.status = status
        self.filled_size = filled_size
        self.filled_price = filled_price
        self.created_at = created_at or time.time()
        self.updated_at = updated_at or self.created_at
        self.metadata = metadata or {}
        
    def update(self, status: OrderStatus, filled_size: float = None, filled_price: float = None):
        """Update order status and fill information"""
        self.status = status
        if filled_size is not None:
            self.filled_size = filled_size
        if filled_price is not None:
            self.filled_price = filled_price
        self.updated_at = time.time()
    
    def to_dict(self) -> Dict:
        """Convert order to dictionary"""
        return {
            "order_id": self.order_id,
            "symbol": self.symbol,
            "side": self.side.value,
            "size": self.size,
            "price": self.price,
            "order_type": self.order_type.value,
            "status": self.status.value,
            "filled_size": self.filled_size,
            "filled_price": self.filled_price,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": self.metadata
        }
    
    @staticmethod
    def from_dict(data: Dict) -> 'HedgeOrder':
        """Create order from dictionary"""
        return HedgeOrder(
            order_id=data["order_id"],
            symbol=data["symbol"],
            side=OrderSide(data["side"]),
            size=data["size"],
            price=data["price"],
            order_type=OrderType(data["order_type"]),
            status=OrderStatus(data["status"]),
            filled_size=data["filled_size"],
            filled_price=data["filled_price"],
            created_at=data["created_at"],
            updated_at=data["updated_at"],
            metadata=data["metadata"]
        )


class HedgeExecution:
    """Handles execution of hedge orders"""
    
    def __init__(self, config: HedgeConfig, exchange_client: Any = None):
        """
        Initialize hedge execution module
        
        Args:
            config: Hedge configuration
            exchange_client: Exchange API client (optional)
        """
        self.config = config
        self.logger = LoggerFactory.configure_component_logger(
            "hedge_execution",
            log_file="execution.log"  # Will be placed in logs/hedge/ directory
        )
        
        # Initialize exchange client if not provided
        if exchange_client is None and config.exchange_api_key and config.exchange_api_secret:
            try:
                from ...exchange_clients.thalex_client import ThalexClient
                self.logger.info("Creating Thalex exchange client")
                self.exchange_client = ThalexClient(
                    api_key=config.exchange_api_key,
                    api_secret=config.exchange_api_secret,
                    testnet=config.use_testnet
                )
                self.logger.info("Thalex exchange client created successfully")
            except Exception as e:
                self.logger.error(f"Failed to create exchange client: {e}")
                self.exchange_client = None
        else:
            self.exchange_client = exchange_client
            
        # Active orders and fill history
        self.active_orders: Dict[str, HedgeOrder] = {}
        self.order_history: List[HedgeOrder] = []
    
    def place_market_order(
        self, 
        symbol: str, 
        side: OrderSide, 
        size: float,
        metadata: Optional[Dict] = None
    ) -> HedgeOrder:
        """
        Place a market order for hedging
        
        Args:
            symbol: Asset symbol
            side: Buy or sell
            size: Order size
            metadata: Additional order metadata
            
        Returns:
            HedgeOrder object
        """
        order_id = f"hedge-{uuid.uuid4()}"
        
        # Create order object
        order = HedgeOrder(
            order_id=order_id,
            symbol=symbol,
            side=side,
            size=abs(size),
            order_type=OrderType.MARKET,
            status=OrderStatus.NEW,
            metadata=metadata or {}
        )
        
        try:
            # If exchange client is provided, place actual order
            if self.exchange_client is not None:
                # Implementation depends on the exchange client interface
                exchange_response = self._place_exchange_order(order)
                
                # Update order with exchange response
                if exchange_response:
                    order.update(
                        status=OrderStatus.FILLED,
                        filled_size=exchange_response.get("filled_size", order.size),
                        filled_price=exchange_response.get("filled_price", 0.0)
                    )
            else:
                # Simulate order execution (for testing/development)
                # In a real implementation, this would connect to the exchange API
                self.logger.warning(f"Exchange client not provided, simulating order execution: {order.to_dict()}")
                order.update(status=OrderStatus.FILLED, filled_size=order.size)
            
            # Track order
            self.active_orders[order_id] = order
            
            self.logger.info(
                f"Placed market {side.value} order for {size} {symbol} | "
                f"Order ID: {order_id} | Status: {order.status.value}"
            )
            
            return order
            
        except Exception as e:
            self.logger.error(f"Error placing market order: {e}")
            order.update(status=OrderStatus.REJECTED)
            self.order_history.append(order)
            return order
    
    def place_limit_order(
        self, 
        symbol: str, 
        side: OrderSide, 
        size: float,
        price: float, 
        metadata: Optional[Dict] = None
    ) -> HedgeOrder:
        """
        Place a limit order for hedging
        
        Args:
            symbol: Asset symbol
            side: Buy or sell
            size: Order size
            price: Limit price
            metadata: Additional order metadata
            
        Returns:
            HedgeOrder object
        """
        order_id = f"hedge-{uuid.uuid4()}"
        
        # Create order object
        order = HedgeOrder(
            order_id=order_id,
            symbol=symbol,
            side=side,
            size=abs(size),
            price=price,
            order_type=OrderType.LIMIT,
            status=OrderStatus.NEW,
            metadata=metadata or {}
        )
        
        try:
            # If exchange client is provided, place actual order
            if self.exchange_client is not None:
                # Implementation depends on the exchange client interface
                exchange_response = self._place_exchange_order(order)
                
                # Update order with exchange response
                if exchange_response:
                    order.update(
                        status=OrderStatus.OPEN,
                        filled_size=exchange_response.get("filled_size", 0.0),
                        filled_price=exchange_response.get("filled_price", 0.0)
                    )
            else:
                # Simulate order execution (for testing/development)
                self.logger.warning(f"Exchange client not provided, simulating order execution: {order.to_dict()}")
                order.update(status=OrderStatus.OPEN, filled_size=0.0)
            
            # Track order
            self.active_orders[order_id] = order
            
            self.logger.info(
                f"Placed limit {side.value} order for {size} {symbol} @ {price} | "
                f"Order ID: {order_id} | Status: {order.status.value}"
            )
            
            return order
            
        except Exception as e:
            self.logger.error(f"Error placing limit order: {e}")
            order.update(status=OrderStatus.REJECTED)
            self.order_history.append(order)
            return order
    
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an open order
        
        Args:
            order_id: Order ID to cancel
            
        Returns:
            True if cancellation was successful
        """
        if order_id not in self.active_orders:
            self.logger.warning(f"Order {order_id} not found in active orders")
            return False
        
        order = self.active_orders[order_id]
        
        # Can only cancel open orders
        if order.status != OrderStatus.OPEN:
            self.logger.warning(f"Cannot cancel order {order_id} with status {order.status.value}")
            return False
            
        try:
            # If exchange client is provided, cancel actual order
            if self.exchange_client is not None:
                # Implementation depends on the exchange client interface
                success = self._cancel_exchange_order(order)
                
                if success:
                    order.update(status=OrderStatus.CANCELED)
                    self.order_history.append(order)
                    del self.active_orders[order_id]
                    self.logger.info(f"Canceled order {order_id}")
                    return True
                else:
                    self.logger.error(f"Failed to cancel order {order_id}")
                    return False
            else:
                # Simulate order cancellation
                self.logger.warning(f"Exchange client not provided, simulating order cancellation: {order_id}")
                order.update(status=OrderStatus.CANCELED)
                self.order_history.append(order)
                del self.active_orders[order_id]
                return True
                
        except Exception as e:
            self.logger.error(f"Error canceling order {order_id}: {e}")
            return False
    
    def update_order_status(self, order_id: str, status: OrderStatus, filled_size: float = None, filled_price: float = None):
        """
        Update order status (typically from exchange callbacks)
        
        Args:
            order_id: Order ID to update
            status: New order status
            filled_size: Updated filled size
            filled_price: Updated filled price
        """
        if order_id in self.active_orders:
            order = self.active_orders[order_id]
            order.update(status, filled_size, filled_price)
            
            # If order is no longer active, move to history
            if status in [OrderStatus.FILLED, OrderStatus.CANCELED, OrderStatus.REJECTED, OrderStatus.EXPIRED]:
                self.order_history.append(order)
                del self.active_orders[order_id]
                
            self.logger.info(f"Updated order {order_id} status to {status.value}")
    
    def get_active_orders(self, symbol: Optional[str] = None) -> List[HedgeOrder]:
        """
        Get active orders, optionally filtered by symbol
        
        Args:
            symbol: Optional symbol to filter by
            
        Returns:
            List of active orders
        """
        if symbol:
            return [o for o in self.active_orders.values() if o.symbol == symbol]
        else:
            return list(self.active_orders.values())
    
    def get_filled_orders(self, symbol: Optional[str] = None) -> List[HedgeOrder]:
        """
        Get filled orders from history
        
        Args:
            symbol: Optional symbol to filter by
            
        Returns:
            List of filled orders
        """
        return [o for o in self.order_history if o.status == OrderStatus.FILLED and (symbol is None or o.symbol == symbol)]
    
    def _place_exchange_order(self, order: HedgeOrder) -> Optional[Dict]:
        """
        Place order on the exchange
        
        Args:
            order: HedgeOrder object
            
        Returns:
            Exchange response or None if simulation
        """
        self.logger.info(f"Attempting to place order: {order.symbol} {order.side.value} {order.size} {order.order_type.value}")
        
        if self.exchange_client is None:
            self.logger.warning(f"SIMULATION MODE: Exchange client is None, cannot execute real order for {order.symbol}")
            self.logger.warning(f"Order will be simulated but NOT sent to exchange: {order.symbol} {order.side.value} {order.size}")
            return None
        
        # Log exchange client type and details
        self.logger.info(f"Exchange client type: {type(self.exchange_client).__name__}")
        
        # Implement exchange-specific order placement
        try:
            # Check if the exchange client is a Thalex client or a wrapper
            is_thalex_client = ('thalex.thalex.Thalex' in str(type(self.exchange_client)) or 
                               hasattr(self.exchange_client, 'thalex') or
                               'ThalexClient' in str(type(self.exchange_client)))
            
            if is_thalex_client:
                self.logger.info(f"REAL ORDER: Using Thalex client API for order execution: {order.symbol} {order.side.value} {order.size}")
                
                # Get the actual Thalex client
                if hasattr(self.exchange_client, 'thalex'):
                    thalex_client = self.exchange_client 
                else:
                    thalex_client = self.exchange_client
                
                # Prepare async execution
                loop = asyncio.get_event_loop()
                
                # Use Thalex client to place order
                if order.order_type == OrderType.MARKET:
                    # Place using appropriate method based on client type
                    if hasattr(thalex_client, 'place_market_order') and callable(thalex_client.place_market_order):
                        response = loop.run_until_complete(thalex_client.place_market_order(
                            symbol=order.symbol,
                            side=order.side.value,
                            size=order.size
                        ))
                        self.logger.info(f"REAL ORDER EXECUTED: Thalex market order response: {response}")
                        return response
                    # Fallback to native Thalex client methods
                    else:
                        # Call either buy or sell based on direction
                        if order.side == OrderSide.BUY:
                            response = loop.run_until_complete(thalex_client.buy(
                                instrument_name=order.symbol,
                                amount=order.size,
                                order_type="market"
                            ))
                        else:
                            response = loop.run_until_complete(thalex_client.sell(
                                instrument_name=order.symbol,
                                amount=order.size,
                                order_type="market"
                            ))
                        
                        self.logger.info(f"REAL ORDER EXECUTED: Thalex market order response: {response}")
                        return {
                            "status": "filled",
                            "filled_size": order.size,
                            "filled_price": response.get("price", 0.0)
                        }
                    
                else:  # Limit order
                    # Place using appropriate method based on client type
                    if hasattr(thalex_client, 'place_limit_order') and callable(thalex_client.place_limit_order):
                        response = loop.run_until_complete(thalex_client.place_limit_order(
                            symbol=order.symbol,
                            side=order.side.value,
                            size=order.size,
                            price=order.price
                        ))
                        self.logger.info(f"REAL ORDER EXECUTED: Thalex limit order response: {response}")
                        return response
                    # Fallback to native Thalex client methods
                    else:
                        # Call either buy or sell based on direction
                        if order.side == OrderSide.BUY:
                            response = loop.run_until_complete(thalex_client.buy(
                                instrument_name=order.symbol,
                                amount=order.size,
                                price=order.price,
                                order_type="limit"
                            ))
                        else:
                            response = loop.run_until_complete(thalex_client.sell(
                                instrument_name=order.symbol,
                                amount=order.size,
                                price=order.price,
                                order_type="limit"
                            ))
                        
                        self.logger.info(f"REAL ORDER EXECUTED: Thalex limit order response: {response}")
                        return {
                            "status": "open",
                            "filled_size": 0.0,
                            "filled_price": 0.0,
                            "order_id": response.get("order_id", "")
                        }
            
            # Execute actual order based on order type using generic method
            elif order.order_type == OrderType.MARKET:
                # Place market order using the exchange client
                if hasattr(self.exchange_client, 'place_market_order') and callable(self.exchange_client.place_market_order):
                    self.logger.info(f"REAL ORDER: Using exchange client's place_market_order for {order.symbol} {order.side.value} {order.size}")
                    response = self.exchange_client.place_market_order(
                        symbol=order.symbol,
                        side=order.side.value,
                        size=order.size
                    )
                    self.logger.info(f"REAL ORDER EXECUTED: Market order response: {response}")
                    return response
            else:
                # Place limit order using the exchange client
                if hasattr(self.exchange_client, 'place_limit_order') and callable(self.exchange_client.place_limit_order):
                    self.logger.info(f"REAL ORDER: Using exchange client's place_limit_order for {order.symbol} {order.side.value} {order.size} @ {order.price}")
                    response = self.exchange_client.place_limit_order(
                        symbol=order.symbol,
                        side=order.side.value,
                        size=order.size,
                        price=order.price
                    )
                    self.logger.info(f"REAL ORDER EXECUTED: Limit order response: {response}")
                    return response
            
            # Fallback to simulation if exchange client doesn't have proper methods
            self.logger.warning(f"SIMULATION MODE: Exchange client doesn't have proper order execution methods, simulating order instead")
            
            simulated_response = {
                "status": "filled" if order.order_type == OrderType.MARKET else "open",
                "filled_size": order.size if order.order_type == OrderType.MARKET else 0.0,
                "filled_price": order.price or 0.0
            }
            
            self.logger.warning(f"SIMULATED ORDER (NOT SENT TO EXCHANGE): {order.symbol} {order.side.value} {order.size}, response: {simulated_response}")
            return simulated_response
            
        except Exception as e:
            self.logger.error(f"ERROR PLACING ORDER: Error placing order on exchange: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def _cancel_exchange_order(self, order: HedgeOrder) -> bool:
        """
        Cancel order on the exchange
        
        Args:
            order: HedgeOrder object
            
        Returns:
            True if cancellation was successful
        """
        if self.exchange_client is None:
            return True
            
        # Implement exchange-specific order cancellation
        try:
            # Check if exchange client has cancel_order method
            if hasattr(self.exchange_client, 'cancel_order') and callable(self.exchange_client.cancel_order):
                # For async clients
                if asyncio.iscoroutinefunction(self.exchange_client.cancel_order):
                    loop = asyncio.get_event_loop()
                    response = loop.run_until_complete(self.exchange_client.cancel_order(order_id=order.order_id))
                    return response
                # For sync clients
                else:
                    response = self.exchange_client.cancel_order(order_id=order.order_id)
                    return response
                    
            # Check if it's a Thalex client with native cancel method
            elif hasattr(self.exchange_client, 'cancel') and callable(self.exchange_client.cancel):
                loop = asyncio.get_event_loop()
                response = loop.run_until_complete(self.exchange_client.cancel(order_id=order.order_id))
                self.logger.info(f"Thalex cancel response: {response}")
                return True
                
            # For now, simulate successful cancellation if no method is found
            self.logger.warning("No cancel method found on exchange client, simulating successful cancellation")
            return True
        except Exception as e:
            self.logger.error(f"Exchange order cancellation error: {e}")
            return False 