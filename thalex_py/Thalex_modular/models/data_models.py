import asyncio
import json
import logging
import socket
import time
import traceback
from typing import Union, Dict, Optional, List, Any, Tuple
import enum
import websockets
from threading import Lock
import numpy as np
from collections import deque
import plotly.graph_objects as go
from enum import Enum
from dataclasses import dataclass, field

import thalex as th
from .keys import *  # Import from the local keys.py file

# Import configurations from market_config.py
from ..config.market_config import (
    BOT_CONFIG,
    MARKET_CONFIG,
    TRADING_CONFIG,
    RISK_CONFIG,
    PERFORMANCE_CONFIG
)

# Basic configuration - use values from MARKET_CONFIG
UNDERLYING = MARKET_CONFIG["underlying"]
LABEL = MARKET_CONFIG["label"]
NETWORK = MARKET_CONFIG["network"]

# Order book configuration - use values from TRADING_CONFIG
AMEND_THRESHOLD = TRADING_CONFIG["order"]["amend_threshold"]
SPREAD = TRADING_CONFIG["order"]["spread"]
BID_STEP = TRADING_CONFIG["order"]["bid_step"]
ASK_STEP = TRADING_CONFIG["order"]["ask_step"]
BID_SIZES = TRADING_CONFIG["order"]["bid_sizes"]
ASK_SIZES = TRADING_CONFIG["order"]["ask_sizes"]

# These configuration dictionaries have been moved to market_config.py
# Reference them from the imported configurations
# POSITION_LIMITS now comes from RISK_CONFIG["limits"]
# QUOTING_CONFIG now comes from TRADING_CONFIG["quoting"]
# INVENTORY_CONFIG now comes from RISK_CONFIG["inventory"]
# SIGNAL_CONFIG now comes from BOT_CONFIG["technical"]["signal"]
# AVELLANEDA_CONFIG now comes from TRADING_CONFIG["avellaneda"]

# Define convenience variables for imported configurations
POSITION_LIMITS = RISK_CONFIG["limits"] if "limits" in RISK_CONFIG else {}
QUOTING_CONFIG = TRADING_CONFIG["quoting"] if "quoting" in TRADING_CONFIG else {}
INVENTORY_CONFIG = RISK_CONFIG["inventory"] if "inventory" in RISK_CONFIG else {}
SIGNAL_CONFIG = BOT_CONFIG.get("technical", {}).get("signal", {})
AVELLANEDA_CONFIG = TRADING_CONFIG["avellaneda"] if "avellaneda" in TRADING_CONFIG else {}

# Performance metrics
performance_metrics = {
    "successful_trades": 0,
    "average_fill_price": 0.0,
    "total_trades": 0,
    "win_loss_ratio": 0.0,
    "expected_value": 0.0,
}

# Add metrics lock
metrics_lock = Lock()

# Call IDs for Thalex API - use from BOT_CONFIG
CALL_ID_INSTRUMENTS = BOT_CONFIG["call_ids"].get("instruments", 0)
CALL_ID_INSTRUMENT = BOT_CONFIG["call_ids"].get("ticker", 1)
CALL_ID_SUBSCRIBE = BOT_CONFIG["call_ids"].get("subscribe", 2)
CALL_ID_LOGIN = BOT_CONFIG["call_ids"].get("login", 3)
CALL_ID_CANCEL_SESSION = BOT_CONFIG["call_ids"].get("cancel_all", 4)
CALL_ID_SET_COD = BOT_CONFIG["call_ids"].get("set_cod", 5)

# Define CALL_IDS for backward compatibility
CALL_IDS = {
    "instruments": CALL_ID_INSTRUMENTS,
    "ticker": CALL_ID_INSTRUMENT,
    "subscribe": CALL_ID_SUBSCRIBE,
    "login": CALL_ID_LOGIN,
    "cancel_all": CALL_ID_CANCEL_SESSION,
    "set_cod": CALL_ID_SET_COD
}

# Order status enumeration
class OrderStatus(Enum):
    """Order status enumeration"""
    OPEN = "open"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    CANCELLED_PARTIALLY_FILLED = "cancelled_partially_filled"
    FILLED = "filled"
    PENDING = "pending"

@dataclass
class Order:
    """Order data structure"""
    id: int
    price: float
    amount: float
    status: OrderStatus = OrderStatus.PENDING
    direction: Optional[str] = None
    
    def __post_init__(self):
        """Initialize additional fields after constructor"""
        self.creation_time: float = 0.0
        self.last_update_time: float = 0.0
        self.filled_amount: float = 0.0
        self.average_fill_price: float = 0.0
        self.client_id: Optional[str] = None
        self.instrument_id: Optional[str] = None
        self.type: Optional[str] = None
    
    def is_open(self) -> bool:
        """Check if order is still open"""
        return self.status in [
            OrderStatus.OPEN,
            OrderStatus.PARTIALLY_FILLED,
            OrderStatus.PENDING
        ]

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "price": self.price,
            "amount": self.amount,
            "status": self.status.value if self.status else None,
            "creation_time": self.creation_time,
            "last_update_time": self.last_update_time,
            "filled_amount": self.filled_amount,
            "average_fill_price": self.average_fill_price
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'Order':
        order = cls(
            id=data["id"],  # Changed from oid to id
            price=data["price"],
            amount=data["amount"],
            status=OrderStatus(data["status"]) if data.get("status") else None
        )
        order.creation_time = data.get("creation_time", 0.0)
        order.last_update_time = data.get("last_update_time", 0.0)
        order.filled_amount = data.get("filled_amount", 0.0)
        order.average_fill_price = data.get("average_fill_price", 0.0)
        return order

# Ticker data structure
class Ticker:
    """Ticker data structure"""
    def __init__(self, data: Dict[str, Any]):
        self.mark_price: float = float(data["mark_price"])
        self.best_bid_price: Optional[float] = float(data["best_bid_price"]) if "best_bid_price" in data else None
        self.best_ask_price: Optional[float] = float(data["best_ask_price"]) if "best_ask_price" in data else None
        self.index: float = float(data["index"]) if "index" in data else 0.0
        self.mark_ts: float = float(data["mark_timestamp"]) if "mark_timestamp" in data else time.time()
        self.funding_rate: float = float(data["funding_rate"]) if "funding_rate" in data else 0.0
        self.volume: float = float(data.get("volume", 0.0))
        self.open_interest: float = float(data.get("open_interest", 0.0))
        # Add timestamp field with default value to avoid the error
        self.timestamp: float = float(data.get("timestamp", time.time()))

    def to_dict(self) -> Dict:
        return {
            "mark_price": self.mark_price,
            "best_bid_price": self.best_bid_price,
            "best_ask_price": self.best_ask_price,
            "index": self.index,
            "mark_timestamp": self.mark_ts,
            "funding_rate": self.funding_rate,
            "volume": self.volume,
            "open_interest": self.open_interest
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'Ticker':
        return cls(data)

@dataclass
class Quote:
    """Quote data structure representing a proposed order price and size"""
    price: float
    amount: float
    instrument: str = ""
    side: str = ""  # "buy" or "sell"
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict:
        """Convert quote to dictionary"""
        return {
            "price": self.price,
            "amount": self.amount,
            "instrument": self.instrument,
            "side": self.side,
            "timestamp": self.timestamp
        }
        
    @classmethod
    def from_dict(cls, data: Dict) -> 'Quote':
        """Create Quote from dictionary"""
        return cls(
            price=data["price"],
            amount=data["amount"],
            instrument=data.get("instrument", ""),
            side=data.get("side", ""),
            timestamp=data.get("timestamp", time.time())
        )

class PerpQuoter:
    def __init__(self, thalex: th.Thalex):
        self.thalex = thalex
        self.ticker = None
        self.instruments = []
        self.perp_instrument_name = None
        self.tick = 0
        self.position_size = 0
        self.entry_price = 0
        self.realized_pnl = 0
        self.unrealized_pnl = 0
        
        # Initialize operation semaphore with concurrency limit
        self.operation_semaphore = asyncio.Semaphore(TRADING_CONFIG["quoting"]["max_pending_operations"])
        self.quote_cv = asyncio.Condition()
        self.portfolio: Dict[str, float] = {}
        self.orders: List[List[Order]] = [[], []]  # bids, asks
        self.client_order_id: int = 100
        self.tick: Optional[float] = None
        self.perp_name: Optional[str] = None
        
        # Position management
        self.position_size = 0.0
        self.entry_price = None  # Initialize as None to distinguish between no position and zero price
        self.last_rebalance = 0
        
        # Initialize price tracking
        self.last_mark_price = None
        # Risk management
        self.alert_counts = {}
        self.alert_cooldown = 50  # 5 minutes
        self.price_history = deque(maxlen=100)
        self.last_alert_time = {}
        self.quoting_enabled = True
        # Add to existing init
        self.highest_profit = 0.0
        self.trailing_stop_active = False
        self.entry_prices = {}  # Dictionary to track entry prices for partial positions
        # Add parameters for Avellaneda-Stoikov model
        self.gamma = 0.1  # Risk aversion parameter
        self.k = 1.5     # Order flow intensity
        self.sigma = 0.0 # Market volatility (dynamic)
        self.T = 1.0     # Time horizon
        self.price_window = deque(maxlen=100)
        self.pnl_history = []  # List to store cumulative PnL over time
        self.time_history = []  # List to store timestamps for plotting
        
        # Add PnL tracking attributes
        self.realized_pnl = 0.0
        self.unrealized_pnl = 0.0
        self.cumulative_pnl = 0.0
        self.trade_history = []
        self.take_profit_orders = {}  # Track take profit orders
        self.executed_take_profits = set()  # Track which levels have been executed
        self.highest_profit_levels = {}  # Track highest profit for each trailing stop level
        self.pending_operations = set()
        self.last_operation_times = {}
        self.last_quote_task = 0
        self.last_quote_update = 0  # Also add this for quote update tracking
        self.last_position_check = 0
        self.last_rebalance_time = 0
        self.last_stop_loss = 0
        self.last_take_profit = 0
        self.last_close_attempt = 0
        self.last_emergency_close = 0
        self.orders_lock = asyncio.Lock()  # Add this line
        
        # Add these new attributes
        self.last_inventory_update = time.time()
        self.inventory_holding_cost = 0.0
        self.inventory_imbalance = 0.0

    def round_to_tick(self, value):
        return self.tick * round(value / self.tick)

    def calculate_zscore(self) -> float:
        """Calculate Z-score for current price"""
        if len(self.price_history) < 20:
            return 0
        prices = np.array(self.price_history)
        return (self.ticker.mark_price - np.mean(prices)) / np.std(prices)

    def calculate_atr(self):
        """Calculate Average True Range"""
        if len(self.price_history) < 2:
            return 0.0
        
        prices = np.array(list(self.price_history))
        high_low = np.abs(np.diff(prices))
        return np.mean(high_low)

    async def check_risk_limits(self) -> bool:
        """Check if position and notional risk limits are exceeded"""
        position = self.position_size
        
        # Check absolute position limit
        if abs(position) >= RISK_CONFIG["limits"]["max_position"]:
            logging.warning(f"Position {position} exceeds limit of {RISK_CONFIG['limits']['max_position']}")
            return False
            
        # Check notional value limit
        price = self.ticker.mark_price if self.ticker else 0
        notional = abs(position * price)
        if notional >= RISK_CONFIG["limits"]["max_notional"]:
            logging.warning(f"Notional value {notional} exceeds limit of {RISK_CONFIG['limits']['max_notional']}")
            return False
            
        # Check if position is approaching max and needs rebalancing
        if abs(position) >= RISK_CONFIG["limits"]["max_position"] * RISK_CONFIG["limits"]["position_rebalance_threshold"]:
            logging.warning(f"Position {position} approaching max {RISK_CONFIG['limits']['max_position']}, consider rebalancing")
            
        return True

    def should_alert(self, alert_key: str, current_time: float) -> bool:
        """Determine if alert should be shown based on cooldown"""
        if alert_key not in self.alert_counts:
            self.alert_counts[alert_key] = 0
            self.last_alert_time[alert_key] = 0

        if current_time - self.last_alert_time[alert_key] > self.alert_cooldown:
            self.alert_counts[alert_key] = 0

        if self.alert_counts[alert_key] < 2:
            self.alert_counts[alert_key] += 1
            self.last_alert_time[alert_key] = current_time
            return True
        return False

    async def handle_risk_breach(self):
        """Handle risk limit breach"""
        try:
            position_size = self.get_position_size()
            current_price = self.get_current_price()
            
            if current_price <= 0:
                logging.error("Cannot handle risk breach: invalid current price")
                return
                
            # Calculate reduction size and price
            reduction_size = -position_size * 0.5  # Reduce position by 50%
            
            # Round the reduction size to nearest 0.0001 for better precision
            reduction_size = round(abs(reduction_size) * 10000) / 10000  # This ensures 4 decimal places
            
            # Calculate reduction price with a small buffer for faster execution
            if position_size > 0:  # Long position
                reduction_price = current_price * 0.9995  # Slightly below market
            else:  # Short position
                reduction_price = current_price * 1.0005  # Slightly above market
                
            # Align price to tick size
            aligned_price = self.round_to_tick(reduction_price)
            
            logging.info(f"Starting risk breach reduction: {reduction_size} @ {aligned_price}")
            
            direction = th.Direction.SELL if position_size > 0 else th.Direction.BUY
            await self.thalex.insert(
                direction=direction,
                instrument_name=self.perp_name,
                amount=reduction_size,  # Now properly rounded to 0.0001
                price=aligned_price,
                client_order_id=self.client_order_id,
                id=self.client_order_id
            )
            self.client_order_id += 1
            
        except Exception as e:
            logging.error(f"Error handling risk breach: {str(e)}")
            await asyncio.sleep(5)  # Add delay on error

    async def smart_position_reduction(self, take_profit_pct: float):
        """Reduce position size based on market conditions"""
        if self.position_size == 0:
            return

        reduction_size = self.position_size * 0.25  # Reduce 25% at a time
        direction = th.Direction.SELL if self.position_size > 0 else th.Direction.BUY
        
        price_adjustment = (1 + take_profit_pct) if self.position_size > 0 else (1 - take_profit_pct)
        target_price = self.entry_price * price_adjustment
        
        logging.info(f"Starting smart position reduction: {reduction_size} @ {target_price}")
        
        await self.thalex.insert(
            direction=direction,
            instrument_name=self.perp_name,
            amount=abs(reduction_size),
            price=target_price,
            client_order_id=self.client_order_id,
            id=self.client_order_id
        )
        self.client_order_id += 1

    async def manage_position(self):
        """Enhanced position management with notional limits and inventory controls"""
        current_time = time.time()
        if current_time - self.last_position_check < 5:
            return
        
        self.last_position_check = current_time
        
        try:
            # Calculate current notional value
            if not self.ticker or self.ticker.mark_price <= 0:
                return
            
            current_notional = abs(self.position_size * self.ticker.mark_price)
            
            # Check notional limits
            if current_notional > INVENTORY_CONFIG["max_position_notional"]:
                logging.warning(f"Position notional {current_notional} exceeds limit")
                await self.manage_notional_breach()
                return
            
            # Calculate position metrics
            pnl_pct = self.calculate_position_pnl()
            inventory_age = current_time - self.last_inventory_update
            
            # Handle losses
            if pnl_pct < -INVENTORY_CONFIG["max_loss_threshold"]:
                await self.handle_position_loss()
                return
            
            # Check for profitable rebalancing opportunity
            if pnl_pct > INVENTORY_CONFIG["min_profit_rebalance"]:
                if abs(self.position_size) > POSITION_LIMITS["max_position"] * POSITION_LIMITS["rebalance_threshold"]:
                    await self.rebalance_position()
                
            # Update inventory metrics
            await self.update_inventory_metrics()
            
        except Exception as e:
            logging.error(f"Error in position management: {str(e)}")

    async def manage_notional_breach(self):
        """Handle breach of notional limits with improved execution"""
        try:
            if not self.ticker or self.ticker.mark_price <= 0:
                logging.error("Invalid market price for notional breach handling")
                return

            current_notional = abs(self.position_size * self.ticker.mark_price)
            target_notional = INVENTORY_CONFIG["max_position_notional"] * 0.8
            
            if current_notional <= target_notional:
                return

            # Calculate reduction size
            reduction_ratio = (current_notional - target_notional) / current_notional
            reduction_size = abs(self.position_size) * reduction_ratio
            
            # Align size to valid increment
            reduction_size = round(reduction_size * 1000) / 1000
            
            if reduction_size < 0.001:
                return
            
            direction = th.Direction.SELL if self.position_size > 0 else th.Direction.BUY
            
            # Split into smaller orders
            num_orders = 3
            size_per_order = round((reduction_size / num_orders) * 1000) / 1000
            
            for i in range(num_orders):
                if size_per_order >= 0.001:
                    # Use calculate_exit_price instead of dynamic spread directly
                    price = self.calculate_exit_price(direction, urgency_factor=0.2 + (i * 0.1))
                    
                    await self.submit_order(
                        direction=direction,
                        amount=size_per_order,
                        price=price
                    )
                    await asyncio.sleep(1)  # Reduced sleep time
                    
        except Exception as e:
            logging.error(f"Error managing notional breach: {str(e)}")

    async def handle_position_loss(self):
        """Enhanced loss handling with gradual exit"""
        try:
            remaining_size = abs(self.position_size)
            step_size = remaining_size / INVENTORY_CONFIG["gradual_exit_steps"]
            
            direction = th.Direction.SELL if self.position_size > 0 else th.Direction.BUY
            
            for i in range(INVENTORY_CONFIG["gradual_exit_steps"]):
                size = self.align_amount(step_size)
                if size < 0.001:
                    continue
                
                # Calculate exit price with increasing urgency
                urgency_factor = (i + 1) / INVENTORY_CONFIG["gradual_exit_steps"]
                price = self.calculate_exit_price(direction, urgency_factor)
                
                await self.submit_order(
                    direction=direction,
                    amount=size,
                    price=price
                )
                await asyncio.sleep(1)
            
        except Exception as e:
            logging.error(f"Error handling position loss: {str(e)}")

    def calculate_exit_price(self, direction: th.Direction, urgency_factor: float = 0.5) -> float:
        """Calculate exit price based on market conditions and urgency"""
        if not self.ticker:
            return 0
        
        base_price = self.ticker.mark_price
        spread = self.calculate_dynamic_spread(self.calculate_atr(), self.calculate_zscore())
        
        # Adjust spread based on urgency
        spread *= (1 + urgency_factor)
        
        if direction == th.Direction.SELL:
            price = base_price * (1 - spread)
        else:
            price = base_price * (1 + spread)
        
        return self.round_to_tick(price)

    async def update_inventory_metrics(self):
        """Update inventory metrics and costs"""
        try:
            current_time = time.time()
            
            # Calculate inventory holding cost
            time_held = current_time - self.last_inventory_update
            holding_cost = abs(self.position_size) * INVENTORY_CONFIG["inventory_cost_factor"] * time_held
            
            # Update metrics
            self.inventory_holding_cost += holding_cost
            self.last_inventory_update = current_time
            
            # Calculate inventory imbalance
            max_size = POSITION_LIMITS["max_position"]
            self.inventory_imbalance = self.position_size / max_size
            
        except Exception as e:
            logging.error(f"Error updating inventory metrics: {str(e)}")

    def calculate_dynamic_spread(self, atr: float, zscore: float) -> float:
        """Calculate dynamic spread based on market conditions"""
        try:
            # Base spread from config
            base_spread = QUOTING_CONFIG["min_spread"] * self.tick
            
            # Volatility adjustment
            volatility_factor = 1.0
            if atr > 0:
                volatility_factor = 1.0 + (atr / self.ticker.mark_price) * QUOTING_CONFIG["volatility_spread_factor"]
                
            # Trend adjustment
            trend_factor = 1.0 + abs(zscore) * 0.1
            
            # Market impact protection
            market_spread = float('inf')
            if self.ticker.best_bid and self.ticker.best_ask:
                market_spread = self.ticker.best_ask - self.ticker.best_bid
                if market_spread > 0:
                    base_spread = max(base_spread, market_spread * 0.5)
            
            # Add collar buffer adjustment
            collar_buffer = self.ticker.mark_price * 0.005  # 0.5% collar buffer
            min_spread = collar_buffer * 2  # Minimum spread must be at least the collar buffer
            
            # Calculate final spread
            spread = base_spread * volatility_factor * trend_factor
            
            # Ensure spread respects collar buffer
            spread = max(min_spread, spread)
            
            # Clamp to min/max bounds
            return min(
                QUOTING_CONFIG["max_spread"] * self.tick,
                max(QUOTING_CONFIG["min_spread"] * self.tick, spread)
            )
            
        except Exception as e:
            logging.error(f"Error calculating dynamic spread: {str(e)}")
            return QUOTING_CONFIG["min_spread"] * self.tick  # Return minimum spread on error

    def validate_and_adjust_spread(self, bid_price: float, ask_price: float) -> Tuple[float, float]:
        """Validate and adjust spread to respect price collar"""
        try:
            if not self.ticker or not self.ticker.mark_price:
                return bid_price, ask_price

            # Calculate collar bounds
            collar_buffer = self.ticker.mark_price * 0.005
            max_bid = self.ticker.mark_price - collar_buffer
            min_ask = self.ticker.mark_price + collar_buffer

            # Adjust prices to respect collar
            adjusted_bid = min(bid_price, max_bid)
            adjusted_ask = max(ask_price, min_ask)

            # Ensure minimum spread
            min_spread = collar_buffer * 2
            if adjusted_ask - adjusted_bid < min_spread:
                mid_price = (adjusted_bid + adjusted_ask) / 2
                adjusted_bid = mid_price - min_spread/2
                adjusted_ask = mid_price + min_spread/2

            return self.round_to_tick(adjusted_bid), self.round_to_tick(adjusted_ask)

        except Exception as e:
            logging.error(f"Error validating spread: {str(e)}")
            return bid_price, ask_price

    def calculate_quote_size(self, base_size: float, side: str, position_ratio: float) -> float:
        """Enhanced quote size calculation with inventory management and safety checks"""
        try:
            # Initial validation
            if not self.ticker or not self.ticker.mark_price:
                return 0.001
            
            # Calculate current notional value
            current_notional = abs(self.position_size * self.ticker.mark_price)
            remaining_notional = INVENTORY_CONFIG["max_position_notional"] - current_notional
            
            if remaining_notional <= 0:
                return 0.001

            # Base size adjustment based on inventory imbalance
            inventory_factor = 1.0
            if hasattr(self, 'inventory_imbalance'):
                if side == "bid" and self.inventory_imbalance > 0:
                    inventory_factor = max(0.2, 1 - self.inventory_imbalance * INVENTORY_CONFIG["inventory_skew_factor"])
                elif side == "ask" and self.inventory_imbalance < 0:
                    inventory_factor = max(0.2, 1 + self.inventory_imbalance * INVENTORY_CONFIG["inventory_skew_factor"])

            # Calculate size with notional limit consideration
            max_size_by_notional = remaining_notional / (self.ticker.mark_price * 2)  # Use half of remaining notional
            base_size = min(base_size, max_size_by_notional)
            
            # Apply inventory factor
            size = base_size * inventory_factor
            
            # Align to minimum size increment (0.001)
            size = round(size * 1000) / 1000
            
            # Apply final limits
            return max(0.001, min(size, POSITION_LIMITS["max_position"] * 0.2))
            
        except Exception as e:
            logging.error(f"Error calculating quote size: {str(e)}")
            return 0.001

    async def adjust_quotes(self, desired: List[List[Quote]]):
        """Optimized quote adjustment with proper order cleanup"""
        current_time = time.time()
        if current_time - self.last_quote_update < QUOTING_CONFIG["min_quote_interval"]:
            return
        
        self.last_quote_update = current_time

        try:
            # First cancel all existing orders that don't match desired quotes
            for side_i, side in enumerate([th.Direction.BUY, th.Direction.SELL]):
                orders = self.orders[side_i]
                quotes = desired[side_i]
                
                # Create set of desired prices for quick lookup
                desired_prices = {quote.price for quote in quotes}
                
                # Cancel orders not matching desired quotes
                for order in orders[:]:  # Use slice to avoid modification during iteration
                    if order.is_open():
                        if order.price not in desired_prices:
                            await self.fast_cancel_order(order)
                            # Remove from our tracking immediately
                            self.orders[side_i] = [o for o in self.orders[side_i] if o.id != order.id]

            # Wait a small interval for cancellations to process
            await asyncio.sleep(QUOTING_CONFIG["order_operation_interval"])

            # Now place new quotes
            operations_count = 0
            max_operations = 10  # Maximum operations per cycle
            
            for side_i, side in enumerate([th.Direction.BUY, th.Direction.SELL]):
                quotes = desired[side_i]
                existing_prices = {order.price for order in self.orders[side_i] if order.is_open()}
                
                for quote in quotes:
                    if operations_count >= max_operations:
                        logging.info("Reached maximum operations per cycle")
                        return
                        
                    # Only place quote if we don't have an open order at this price
                    if quote.price not in existing_prices:
                        await self.place_new_quote(quote, side)
                        operations_count += 1

        except Exception as e:
            logging.error(f"Error in quote adjustment: {str(e)}")
            await asyncio.sleep(QUOTING_CONFIG["error_retry_interval"])

    async def fast_cancel_order(self, order: Order):
        """Expedited order cancellation with confirmation"""
        async with self.operation_semaphore:
            if order.id in self.pending_operations:
                return

            self.pending_operations.add(order.id)
            try:
                # Send cancel request
                await self.thalex.cancel(
                    client_order_id=order.id,
                    id=order.id
                )
                
                # Mark order as cancelled in our tracking
                order.status = OrderStatus.CANCELLED
                
                # Log cancellation
                logging.info(f"Cancelled order {order.id} at price {order.price}")
                
                await asyncio.sleep(QUOTING_CONFIG["order_operation_interval"])
            except Exception as e:
                logging.error(f"Error cancelling order {order.id}: {str(e)}")
            finally:
                self.pending_operations.remove(order.id)

    def should_place_new_quote(self, quote: Quote, side: th.Direction) -> bool:
        """Enhanced check for new quote placement"""
        # Position limit check
        if side == th.Direction.BUY and self.position_size >= POSITION_LIMITS["max_position"]:
            return False
        if side == th.Direction.SELL and self.position_size <= -POSITION_LIMITS["max_position"]:
            return False
        
        # Check notional value
        if self.ticker and self.ticker.mark_price > 0:
            potential_notional = quote.amount * quote.price
            current_notional = abs(self.position_size * self.ticker.mark_price)
            if current_notional + potential_notional > POSITION_LIMITS["max_notional"]:
                return False
        
        # Check for existing orders at similar price
        side_idx = 0 if side == th.Direction.BUY else 1
        threshold = QUOTING_CONFIG["amend_threshold"] * self.tick
        
        for order in self.orders[side_idx]:
            if order.is_open():
                price_diff = abs(order.price - quote.price)
                size_diff = abs(order.amount - quote.amount)
                
                if price_diff < threshold and size_diff < 0.01:
                    return False  # Already have a similar order
                    
        return True

    async def cleanup_stale_orders(self):
        """Periodically cleanup stale orders"""
        while True:
            try:
                current_time = time.time()
                
                async with self.orders_lock:  # Add lock
                    for side in [0, 1]:
                        for order in self.orders[side][:]:
                            try:
                                if order.is_open():
                                    if hasattr(order, 'timestamp') and \
                                       current_time - order.timestamp > QUOTING_CONFIG["quote_lifetime"]:
                                        await self.fast_cancel_order(order)
                                        
                                    elif self.ticker and self.should_fast_cancel(order):
                                        await self.fast_cancel_order(order)
                            except Exception as e:
                                logging.error(f"Error processing order {order.id} in cleanup: {str(e)}")
                                continue
                                
                await self.cleanup_completed_orders()  # Add periodic cleanup
                await asyncio.sleep(5)
                
            except Exception as e:
                logging.error(f"Error in cleanup_stale_orders: {str(e)}")
                await asyncio.sleep(5)

    async def quote_task(self):
        """Enhanced quoting loop with state validation"""
        while True:
            try:
                async with self.quote_cv:
                    await self.quote_cv.wait()

                current_time = time.time()
                if current_time - self.last_quote_task < QUOTING_CONFIG["min_quote_interval"]:
                    continue

                self.last_quote_task = current_time

                # Validate state before quoting
                if not self.ticker or not self.index:
                    logging.warning("Missing ticker or index data, skipping quote cycle")
                    continue
                    
                if not await self.validate_position_state():
                    logging.warning("Invalid position state, skipping quote cycle")
                    continue

                quotes = await self.make_quotes()
                await self.adjust_quotes(quotes)
                
                # Non-blocking tasks
                asyncio.create_task(self.manage_take_profit())
                asyncio.create_task(self.cleanup_completed_orders())

            except Exception as e:
                logging.error(f"Error in quote task: {str(e)}")
                await asyncio.sleep(QUOTING_CONFIG["error_retry_interval"])

    async def await_instruments(self):
        await self.thalex.instruments(CALL_ID_INSTRUMENTS)
        msg = await self.thalex.receive()
        msg = json.loads(msg)
        assert msg["id"] == CALL_ID_INSTRUMENTS
        for i in msg["result"]:
            if i["type"] == "perpetual" and i["underlying"] == UNDERLYING:
                self.tick = i["tick_size"]
                self.perp_name = i["instrument_name"]
                return
        assert False  # Perp not found

    async def listen_task(self):
        await self.thalex.connect()
        await self.await_instruments()
        await self.thalex.login(keys.key_ids[NETWORK], keys.private_keys[NETWORK], id=CALL_ID_LOGIN)
        await self.thalex.set_cancel_on_disconnect(6, id=CALL_ID_SET_COD)
        await self.thalex.private_subscribe(["session.orders", "account.portfolio", "account.trade_history"], id=CALL_ID_SUBSCRIBE)
        await self.thalex.public_subscribe([f"ticker.{self.perp_name}.raw", f"price_index.{UNDERLYING}"], id=CALL_ID_SUBSCRIBE)
        
        while True:
            await self.manage_position()
            msg = await self.thalex.receive()
            msg = json.loads(msg)
            if "channel_name" in msg:
                await self.notification(msg["channel_name"], msg["notification"])
            elif "result" in msg:
                await self.result_callback(msg["result"], msg.get("id"))  # Use class method
            else:
                await self.error_callback(msg["error"], msg.get("id"))

    async def notification(self, channel: str, notification: Union[Dict, List[Dict]]):
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
                await self.ticker_callback(notification)
            elif channel.startswith("price_index."):
                await self.index_callback(notification["price"])
            elif channel == "session.orders":
                self.orders_callback(notification)
            elif channel == "account.portfolio":
                self.portfolio_callback(notification)
            elif channel == "account.trade_history":
                await self.trades_callback(notification)
            else:
                logging.error(f"Notification for unknown channel: {channel}")
        except Exception as e:
            logging.error(f"Error processing notification: {str(e)}")

    async def ticker_callback(self, ticker: Dict[str, Any]):
        """Handle ticker updates."""
        try:
            self.ticker = Ticker(ticker)
            self.price_history.append(self.ticker.mark_price)
            async with self.quote_cv:
                self.quote_cv.notify()
        except Exception as e:
            logging.error(f"Error in ticker callback: {str(e)}")

    async def index_callback(self, index: float):
        """Handle index price updates."""
        try:
            self.index = float(index)  # Validate type
            async with self.quote_cv:
                self.quote_cv.notify()
        except (ValueError, TypeError) as e:
            logging.error(f"Invalid index value: {str(e)}")

    async def trades_callback(self, trades: List[Dict]):
        """Process trades with PnL tracking"""
        for t in trades:
            if t.get("label") == LABEL:
                try:
                    amount = float(t.get("amount", 0))
                    price = float(t.get("price", 0))
                    direction = t.get("direction")
                    
                    if amount <= 0 or price <= 0 or not direction:
                        logging.warning(f"Invalid trade data: {t}")
                        continue
                        
                    # Update PnL
                    is_buy = direction == "buy"
                    self.update_realized_pnl(price, amount, is_buy)
                    
                    # Update performance metrics
                    await self.update_performance_metrics(t)
                    
                except Exception as e:
                    logging.error(f"Error processing trade: {str(e)}")

    async def order_error(self, error, oid):
        logging.error(f"Error with order({oid}): {error}")
        for side in [0, 1]:
            for i, o in enumerate(self.orders[side]):
                if o.id == oid:
                    if o.is_open():
                        side = "buy" if i == 0 else "sell"
                        await self.thalex.cancel(client_order_id=oid, id=oid)
                    return

    async def error_callback(self, error, cid=None):
        if cid > 99:
            await self.order_error(error, cid)
        else:
            logging.error(f"{cid=}: error: {error}")

    async def update_order(self, order: Order) -> bool:
        """Thread-safe order update"""
        try:
            async with self.orders_lock:
                for side in [0, 1]:
                    for i, existing in enumerate(self.orders[side]):
                        if existing.id == order.id:
                            self.orders[side][i] = order
                            return True
                return False
        except Exception as e:
            logging.error(f"Error updating order: {str(e)}")
            return False

    def orders_callback(self, orders: List[Dict]):
        """Process order updates with enhanced validation and tracking"""
        try:
            for o in orders:
                order = self.order_from_data(o)
                if not self.update_order(order):
                    logging.warning(
                        f"Order not found in tracking:\n"
                        f"  ID: {order.id}\n"
                        f"  Status: {order.status}\n"
                        f"  Price: {order.price}\n"
                        f"  Amount: {order.amount}"
                    )
                    # Add the order to the appropriate list if it's not found
                    # This fixes the "Order not found in tracking" warnings
                    if order.is_open():
                        direction = o.get("direction", "")
                        if direction == "buy":
                            self.orders[0].append(order)
                            logging.info(f"Added missing buy order {order.id} to tracking")
                        elif direction == "sell":
                            self.orders[1].append(order)
                            logging.info(f"Added missing sell order {order.id} to tracking")
                        else:
                            logging.error(f"Unknown direction for order {order.id}: {direction}")
                            
                    # Attempt recovery for filled orders
                    if order.status == OrderStatus.FILLED:
                        logging.info(f"Attempting to recover filled order state: {order.id}")
                        asyncio.create_task(self.validate_position_state())
        except Exception as e:
            logging.error(f"Error in orders_callback: {str(e)}\nTrace: {traceback.format_exc()}")

    def portfolio_callback(self, portfolio: List[Dict]):
        """Handle portfolio updates with improved drift handling"""
        for position in portfolio:
            instrument = position["instrument_name"]
            new_position = float(position["position"])  # Ensure float conversion
            old_position = self.portfolio.get(instrument, 0)
            self.portfolio[instrument] = new_position
            
            if instrument == self.perp_name:
                # Use a smaller threshold for drift detection
                drift = abs(self.position_size - new_position)
                if drift > 0.001:  # Reduced from 1e-6 to 0.001 for more practical threshold
                    logging.info(f"Position adjustment: internal={self.position_size}, portfolio={new_position}")
                    self.position_size = new_position
                    
                    # Update entry price if needed
                    if new_position != 0 and (self.entry_price is None or self.entry_price <= 0):
                        if self.ticker and self.ticker.mark_price > 0:
                            self.entry_price = self.round_to_tick(self.ticker.mark_price)
                            logging.info(f"Updated entry price to {self.entry_price}")
            
            # Monitor significant position changes
            if abs(new_position - old_position) > 0.001:
                logging.info(f"Position changed: {old_position:.3f} -> {new_position:.3f}")
                asyncio.create_task(self.check_risk_limits())

    def order_from_data(self, data: Dict) -> Order:
        return Order(
            id=data.get("client_order_id", data.get("order_id", 0)),
            price=float(data.get("price", 0)),
            amount=float(data.get("amount", 0)),
            status=OrderStatus(data.get("status", "pending")),
            direction=data.get("direction", "")
        )

    # Add to existing class
    async def result_callback(self, result, cid=None):
        """Handle API call results within class context"""
        try:
            if cid == CALL_ID_INSTRUMENT:
                logging.debug(f"Instrument result: {result}")
            elif cid == CALL_ID_SUBSCRIBE:
                logging.info(f"Subscription confirmed: {result}")
            elif cid == CALL_ID_LOGIN:
                logging.info("Login successful")
            elif cid == CALL_ID_SET_COD:
                logging.debug("Cancel on disconnect set")
            elif cid > 99:
                # Handle order results
                if "error" in result:
                    await self.order_error(result["error"], cid)
                else:
                    logging.debug(f"Order {cid} result: {result}")
            else:
                logging.debug(f"Result {cid}: {result}")
        except Exception as e:
            logging.error(f"Error in result_callback: {str(e)}")

    # When receiving new prices
    def update_price(self, new_price):
        self.price_history.append(float(new_price))

    def get_current_price(self) -> float:
        """Get current market price with validation"""
        if not self.ticker:
            logging.warning("No ticker available for current price")
            return 0.0
            
        if self.ticker.mark_price <= 0:
            logging.warning(f"Invalid mark price: {self.ticker.mark_price}")
            return 0.0
            
        return self.ticker.mark_price

    async def check_position_limits(self) -> bool:
        """Check if position needs rebalancing"""
        abs_position = abs(self.position_size)
        rebalance_size = POSITION_LIMITS["max_position"] * POSITION_LIMITS["rebalance_threshold"]
        
        if abs_position >= rebalance_size:
            zscore = self.calculate_zscore()
            atr = self.calculate_atr()
            
            # Adjust rebalancing based on market conditions
            if abs(zscore) > 2.0:  # High volatility
                await self.aggressive_rebalance()
            else:
                await self.gradual_rebalance()
            return False
        return True

    async def gradual_rebalance(self):
        """Gradually reduce position size"""
        target_size = POSITION_LIMITS["max_position"] * 0.5
        current_size = abs(self.position_size)
        reduction_size = (current_size - target_size) * 0.25  # 25% reduction steps
        
        # Align reduction size to proper precision
        reduction_size = self.align_amount(reduction_size)
        
        direction = th.Direction.SELL if self.position_size > 0 else th.Direction.BUY
        price = self.get_rebalance_price()
        aligned_price = self.round_to_tick(price)
        
        logging.info(f"Gradual rebalance: {reduction_size} @ {aligned_price}")
        
        await asyncio.sleep(1)  # Rate limiting delay
        
        try:
            await self.thalex.insert(
                direction=direction,
                instrument_name=self.perp_name,
                amount=reduction_size,  # Already aligned, no need for abs()
                price=aligned_price,
                client_order_id=self.client_order_id,
                id=self.client_order_id
            )
            self.client_order_id += 1
        except Exception as e:
            logging.error(f"Error in gradual rebalance: {str(e)}")
            await asyncio.sleep(5)

    async def aggressive_rebalance(self):
        """Quickly reduce position in volatile conditions"""
        target_size = POSITION_LIMITS["max_position"] * 0.3
        current_size = abs(self.position_size)
        reduction_size = (current_size - target_size) * 0.5  # 50% reduction steps
        
        # Align reduction size to proper precision
        reduction_size = self.align_amount(reduction_size)
        
        direction = th.Direction.SELL if self.position_size > 0 else th.Direction.BUY
        price = self.get_rebalance_price()
        aligned_price = self.round_to_tick(price)
        
        logging.info(f"Aggressive rebalance: {reduction_size} @ {aligned_price}")
        
        await asyncio.sleep(1)
        
        try:
            await self.thalex.insert(
                direction=direction,
                instrument_name=self.perp_name,
                amount=reduction_size,  # Already aligned, no need for abs()
                price=aligned_price,
                client_order_id=self.client_order_id,
                id=self.client_order_id
            )
            self.client_order_id += 1
        except Exception as e:
            logging.error(f"Error in aggressive rebalance: {str(e)}")
            await asyncio.sleep(5)

    def get_rebalance_price(self) -> float:
        """Calculate rebalance price based on market conditions with tick alignment"""
        if not self.ticker or not self.tick:
            logging.error("Missing ticker or tick size for rebalance price calculation")
            return 0
            
        base_price = self.ticker.mark_price
        zscore = self.calculate_zscore()
        atr = self.calculate_atr()
        
        # Adjust price based on market conditions
        if self.position_size > 0:
            price_adjustment = max(0.1, min(0.5, abs(zscore) * 0.1)) * atr
            price = base_price - price_adjustment
        else:
            price_adjustment = max(0.1, min(0.5, abs(zscore) * 0.1)) * atr
            price = base_price + price_adjustment
            
        # Ensure price is aligned with tick size
        return self.round_to_tick(price)

    def calculate_dynamic_take_profit(self) -> float:
        """Calculate take profit based on market conditions"""
        base_tp = POSITION_LIMITS["base_take_profit_pct"]
        
        # Adjust based on volatility (ATR)
        atr = self.calculate_atr()
        volatility_scalar = min(2.0, max(0.5, atr / self.ticker.mark_price))
        
        # Adjust based on trend strength (Z-score)
        zscore = abs(self.calculate_zscore())
        trend_scalar = min(1.5, max(0.7, zscore / 2))
        
        # Calculate final take profit
        dynamic_tp = base_tp * volatility_scalar * trend_scalar
        
        # Clamp to min/max bounds
        return min(
            POSITION_LIMITS["max_take_profit_pct"],
            max(POSITION_LIMITS["min_take_profit_pct"], dynamic_tp)
        )

    async def manage_take_profit(self):
        """Enhanced take profit management with multiple levels"""
        if self.position_size == 0:
            self.trailing_stop_active = False
            self.highest_profit = 0.0
            self.executed_take_profits.clear()
            self.highest_profit_levels.clear()
            return

        try:
            current_pnl_pct = self.calculate_position_pnl()
            
            # Cancel any existing take profit orders if price moved significantly
            await self.cleanup_take_profit_orders()
            
            # Handle layered take profits
            await self.handle_layered_take_profits(current_pnl_pct)
            
            # Handle multiple trailing stops
            await self.handle_trailing_stops(current_pnl_pct)
            
        except Exception as e:
            logging.error(f"Error in take profit management: {str(e)}")

    async def handle_layered_take_profits(self, current_pnl_pct: float):
        """Handle multiple take profit levels"""
        if not self.ticker or not self.position_size:
            return

        for level in POSITION_LIMITS["take_profit_levels"]:
            level_pct = level["percentage"]
            level_size = level["size"]
            
            if level_pct not in self.executed_take_profits and current_pnl_pct >= level_pct:
                # Calculate the exact amount for this level
                amount = abs(self.position_size * level_size)
                # Round to nearest valid amount
                amount = round(amount * 1000) / 1000  # Round to 0.001
                
                if amount >= 0.001:  # Minimum order size
                    try:
                        direction = th.Direction.SELL if self.position_size > 0 else th.Direction.BUY
                        price = self.calculate_take_profit_price(level_pct)
                        
                        order_id = self.client_order_id
                        await self.thalex.insert(
                            direction=direction,
                            instrument_name=self.perp_name,
                            amount=amount,
                            price=price,
                            client_order_id=order_id,
                            id=order_id
                        )
                        self.client_order_id += 1
                        
                        self.take_profit_orders[order_id] = {
                            "level": level_pct,
                            "amount": amount,
                            "price": price
                        }
                        self.executed_take_profits.add(level_pct)
                        
                        logging.info(f"Take profit order placed: {amount} @ {price} (Level: {level_pct*100}%)")
                        
                    except Exception as e:
                        logging.error(f"Error placing take profit order: {str(e)}")

    async def handle_trailing_stops(self, current_pnl_pct: float):
        """Handle multiple trailing stop levels"""
        for level in POSITION_LIMITS["trailing_stop_levels"]:
            activation = level["activation"]
            distance = level["distance"]
            
            if current_pnl_pct >= activation:
                # Update highest profit for this level
                if activation not in self.highest_profit_levels:
                    self.highest_profit_levels[activation] = current_pnl_pct
                elif current_pnl_pct > self.highest_profit_levels[activation]:
                    self.highest_profit_levels[activation] = current_pnl_pct
                    
                # Check if trailing stop is hit
                trailing_stop_level = self.highest_profit_levels[activation] - distance
                if current_pnl_pct < trailing_stop_level:
                    await self.execute_trailing_stop(activation, current_pnl_pct)
                    break  # Exit after first trailing stop hit

    async def execute_trailing_stop(self, activation: float, current_pnl_pct: float):
        """Execute trailing stop with proper amount alignment"""
        try:
            # Calculate remaining position size
            remaining_size = abs(self.position_size)
            if remaining_size < 0.001:  # Check minimum order size
                return
            
            # Round to valid amount
            amount = round(remaining_size * 1000) / 1000
            
            direction = th.Direction.SELL if self.position_size > 0 else th.Direction.BUY
            price = self.calculate_trailing_stop_price(direction)
            
            await self.thalex.insert(
                direction=direction,
                instrument_name=self.perp_name,
                amount=amount,
                price=price,
                client_order_id=self.client_order_id,
                id=self.client_order_id
            )
            self.client_order_id += 1
            
            logging.info(f"Trailing stop executed at {activation*100}% level: {amount} @ {price}")
            
        except Exception as e:
            logging.error(f"Error executing trailing stop: {str(e)}")

    def calculate_take_profit_price(self, level_pct: float) -> float:
        """Calculate properly aligned take profit price"""
        if self.position_size > 0:
            price = self.entry_price * (1 + level_pct)
        else:
            price = self.entry_price * (1 - level_pct)
        return self.round_to_tick(price)

    def calculate_trailing_stop_price(self, direction: th.Direction) -> float:
        """Calculate properly aligned trailing stop price"""
        current_price = self.ticker.mark_price
        buffer = 0.0005  # 0.05% buffer for faster execution
        
        if direction == th.Direction.SELL:
            price = current_price * (1 - buffer)
        else:
            price = current_price * (1 + buffer)
        
        return self.round_to_tick(price)

    async def cleanup_take_profit_orders(self):
        """Clean up existing take profit orders if needed"""
        for order_id, order_info in list(self.take_profit_orders.items()):
            try:
                # Cancel order if price has moved significantly
                current_price = self.ticker.mark_price
                price_diff = abs(current_price - order_info["price"]) / order_info["price"]
                
                if price_diff > 0.01:  # 1% price movement
                    await self.thalex.cancel(
                        client_order_id=order_id,
                        id=order_id
                    )
                    del self.take_profit_orders[order_id]
                    self.executed_take_profits.remove(order_info["level"])
                
            except Exception as e:
                logging.error(f"Error cleaning up take profit order {order_id}: {str(e)}")

    async def validate_position_state(self) -> bool:
        """Validate position state consistency"""
        try:
            portfolio_position = self.portfolio.get(self.perp_name, 0)
            
            # Check position consistency
            if abs(self.position_size - portfolio_position) > 1e-6:
                logging.error(f"Position state invalid: internal={self.position_size}, portfolio={portfolio_position}")
                self.position_size = portfolio_position
                return False
                
            # Validate entry price
            if self.position_size != 0:
                if self.entry_price is None or self.entry_price <= 0:
                    if self.ticker and self.ticker.mark_price > 0:
                        self.entry_price = self.ticker.mark_price  # Use current price as fallback
                        logging.warning(f"Fixed invalid entry price, set to mark price: {self.entry_price}")
                    else:
                        logging.error("Cannot fix invalid entry price - no valid mark price")
                        return False
            elif self.entry_price is not None:
                self.entry_price = None
                logging.info("Reset entry price for zero position")
                
            return True
            
        except Exception as e:
            logging.error(f"Position validation error: {str(e)}\nTrace: {traceback.format_exc()}")
            return False

    async def recover_position_state(self):
        """Recover position state from portfolio"""
        try:
            # Get position from portfolio
            portfolio_position = self.portfolio.get(self.perp_name, 0)
            self.position_size = portfolio_position
            
            # Reset or recover entry price
            if portfolio_position == 0:
                self.entry_price = None
                logging.info("Position recovery: Reset entry price for zero position")
            elif self.ticker and self.ticker.mark_price > 0:
                self.entry_price = self.round_to_tick(self.ticker.mark_price)
                logging.warning(f"Position recovery: Set entry price to mark price {self.entry_price}")
            
            logging.info(f"Position state recovered: size={self.position_size}, entry_price={self.entry_price}")
            
        except Exception as e:
            logging.error(f"Position recovery failed: {str(e)}\nTrace: {traceback.format_exc()}")

    def calculate_position_pnl(self) -> float:
        """Calculate current position PnL percentage with enhanced safety checks"""
        if self.position_size == 0 or not self.ticker:
            return 0.0
            
        # Validate entry price
        if self.entry_price is None:
            # Try to recover entry price from portfolio if available
            if self.perp_name in self.portfolio and self.portfolio[self.perp_name] != 0:
                logging.warning("Attempting to recover entry price from portfolio")
                if self.ticker and self.ticker.mark_price > 0:
                    self.entry_price = self.ticker.mark_price  # Use current price as fallback
                    logging.info(f"Recovered entry price set to mark price: {self.entry_price}")
                else:
                    logging.warning("Invalid mark price, cannot recover entry price")
                    return 0.0
            else:
                logging.warning("No entry price available, cannot calculate PnL")
                return 0.0
            
        try:
            if self.entry_price <= 0:
                logging.error(f"Invalid entry price: {self.entry_price}")
                return 0.0
                
            if self.ticker.mark_price <= 0:
                logging.error(f"Invalid mark price: {self.ticker.mark_price}")
                return 0.0
                
            direction = 1 if self.position_size > 0 else -1
            pnl = direction * (self.ticker.mark_price - self.entry_price) / self.entry_price
            
            # Sanity check on PnL value
            if abs(pnl) > 1.0:  # Over 100% PnL is suspicious
                logging.warning(f"Unusually large PnL detected: {pnl}")
                
            return pnl
            
        except ZeroDivisionError:
            logging.error(f"Division by zero in PnL calculation: entry_price={self.entry_price}")
            return 0.0
        except Exception as e:
            logging.error(f"Error calculating PnL: {str(e)}")
            return 0.0

    async def handle_order(self, order_id: int):
        try:
            # Your order handling code
            order = await self.get_order(order_id)
            if order is None:
                logging.warning(f"Order {order_id} not found")
                return
            # Process order
        except Exception as e:
            logging.error(f"Error handling order {order_id}: {str(e)}")

    def calculate_optimal_spread(self) -> float:
        """Calculate optimal spread using Avellaneda-Stoikov formula"""
        if len(self.price_window) < 2:
            return SPREAD * self.tick
            
        # Calculate volatility
        self.sigma = np.std(np.diff(np.array(self.price_window)))
        
        # Optimal spread = γσ²(T-t) + 2/γ log(1 + γ/k)
        spread = (self.gamma * self.sigma**2 * self.T + 
                 2/self.gamma * np.log(1 + self.gamma/self.k))
        return max(SPREAD * self.tick, spread)
        
    def get_position_size(self) -> float:
        """Get current position size with portfolio reconciliation"""
        portfolio_position = self.portfolio.get(self.perp_name, 0)
        
        # Check for position mismatch
        if abs(self.position_size - portfolio_position) > 1e-6:
            logging.warning(f"Position mismatch: internal={self.position_size}, portfolio={portfolio_position}")
            self.position_size = portfolio_position
            
        return self.position_size
        
    def calculate_inventory_skew(self) -> float:
        """Calculate inventory-based price skew using reconciled position"""
        position = self.get_position_size()
        q = position / POSITION_LIMITS["max_position"]
        return self.gamma * self.sigma**2 * self.T * q

    def calculate_level_size(self, base_size: float, side: str, level: int) -> float:
        """Calculate size for each quote level based on reconciled inventory with safety checks"""
        try:
            # Get reconciled position
            position = self.get_position_size()
            inventory_ratio = abs(position) / POSITION_LIMITS["max_position"]
            
            # Validate inputs
            if base_size <= 0:
                logging.warning(f"Invalid base size: {base_size}, using minimum size")
                base_size = 0.001
            if level < 0:
                logging.warning(f"Invalid level: {level}, using level 0")
                level = 0
                
            # Reduce size when inventory grows in that direction
            if (side == "bid" and position > 0) or \
               (side == "ask" and position < 0):
                size_multiplier = max(0.1, 1 - inventory_ratio)  # Ensure minimum multiplier
            else:
                size_multiplier = min(2.0, 1 + inventory_ratio * 0.5)  # Cap maximum multiplier
                
            # Calculate size with level reduction and safety bounds
            level_reduction = max(0, min(0.9, level * 0.1))  # Cap level reduction at 90%
            raw_size = base_size * size_multiplier * (1 - level_reduction)
            
            # Ensure minimum size and round to nearest 0.001
            min_size = 0.001
            max_size = POSITION_LIMITS["max_position"] * 0.5  # No single order > 50% of max position
            
            raw_size = max(min_size, min(max_size, raw_size))
            return round(raw_size * 1000) / 1000
            
        except Exception as e:
            logging.error(f"Error calculating level size: {str(e)}")
            return 0.001  # Return minimum size on error

    async def update_model_parameters(self):
        """Update model parameters based on market conditions"""
        # Update volatility estimate
        if self.ticker and self.ticker.mark_price:
            self.price_window.append(self.ticker.mark_price)
            
        # Adjust risk aversion based on PnL
        pnl_pct = self.calculate_position_pnl()
        self.gamma = max(0.05, min(0.3, 0.1 + abs(pnl_pct) * 0.5))
        
        # Update order flow intensity based on recent trades
        # Implementation depends on available market data

    async def sanity_check(self):
        """Run sanity checks with improved validation"""
        try:
            # Configuration validation
            if not ConfigValidator.validate_config():
                raise ValueError("Invalid configuration parameters")

            # Position consistency check
            portfolio_position = self.portfolio.get(self.perp_name, 0)
            if abs(self.position_size - portfolio_position) > 0.001:  # Updated threshold
                logging.info(f"Position adjustment needed: internal={self.position_size}, portfolio={portfolio_position}")
                self.position_size = portfolio_position

            # Order book consistency
            for side in [0, 1]:
                for order in self.orders[side]:
                    if order.is_open() and order.price <= 0:
                        logging.error(f"Invalid order price: {order.price}")
                        await self.thalex.cancel(client_order_id=order.id, id=order.id)

            # Performance metrics validation
            with metrics_lock:
                if performance_metrics["total_trades"] > 0:
                    avg_price = performance_metrics["average_fill_price"]
                    if avg_price <= 0:
                        logging.warning("Resetting invalid average fill price")
                        # Use current mark price as fallback
                        if self.ticker and self.ticker.mark_price > 0:
                            performance_metrics["average_fill_price"] = self.ticker.mark_price
                        else:
                            performance_metrics["average_fill_price"] = 0
                            performance_metrics["total_trades"] = 0  # Reset counter

            # Risk limits validation
            if self.ticker:
                notional = abs(self.position_size * self.ticker.mark_price)
                if notional > POSITION_LIMITS["max_notional"] * 1.1:  # 10% buffer
                    logging.error(f"Critical notional value exceeded: {notional}")
                    await self.emergency_close()

        except Exception as e:
            logging.error(f"Sanity check failed: {str(e)}")
            return False
        return True

    async def emergency_close(self):
        """Emergency position closure with tick-aligned pricing"""
        try:
            if self.position_size == 0:
                return
                
            if not self.ticker or not self.tick:
                logging.error("Missing ticker or tick size for emergency close")
                return
                
            # Align position size to proper precision
            close_amount = self.align_amount(self.position_size)
            
            direction = th.Direction.SELL if self.position_size > 0 else th.Direction.BUY
            base_price = self.ticker.mark_price
            
            # Add a small buffer for faster execution
            if direction == th.Direction.SELL:
                price = self.round_to_tick(base_price * 0.9995)
            else:
                price = self.round_to_tick(base_price * 1.0005)
                
            await self.thalex.insert(
                direction=direction,
                instrument_name=self.perp_name,
                amount=close_amount,
                price=price,
                client_order_id=self.client_order_id,
                id=self.client_order_id
            )
            self.client_order_id += 1
            logging.warning(f"Emergency position closure initiated: {close_amount} @ {price}")
            
            self.last_emergency_close = time.time()
            
        except Exception as e:
            logging.error(f"Emergency close failed: {str(e)}")

    async def update_performance_metrics(self, trade: Dict):
        """Update performance metrics with proper validation"""
        try:
            with metrics_lock:
                fill_price = float(trade.get("price", 0))
                fill_amount = float(trade.get("amount", 0))
                
                if fill_price <= 0 or fill_amount <= 0:
                    logging.warning(f"Invalid fill price ({fill_price}) or amount ({fill_amount})")
                    return
                    
                # Update metrics
                performance_metrics["total_trades"] += 1
                
                # Update average fill price using weighted average
                old_avg = performance_metrics["average_fill_price"]
                old_total = performance_metrics["total_trades"] - 1
                
                if old_total > 0:
                    performance_metrics["average_fill_price"] = (
                        (old_avg * old_total + fill_price) / performance_metrics["total_trades"]
                    )
                else:
                    performance_metrics["average_fill_price"] = fill_price
                    
                # Determine if trade was successful
                if self.entry_price and self.position_size != 0:
                    if (self.position_size > 0 and fill_price > self.entry_price) or \
                       (self.position_size < 0 and fill_price < self.entry_price):
                        performance_metrics["successful_trades"] += 1
                        
                # Update win/loss ratio
                if performance_metrics["total_trades"] > 0:
                    performance_metrics["win_loss_ratio"] = (
                        performance_metrics["successful_trades"] / 
                        performance_metrics["total_trades"]
                    )
                    
        except Exception as e:
            logging.error(f"Error updating performance metrics: {str(e)}")

    async def log_pnl(self):
        """Log PnL metrics periodically"""
        while True:
            try:
                self.calculate_unrealized_pnl()
                self.pnl_history.append(self.cumulative_pnl)
                self.time_history.append(time.time())
                
                logging.info(
                    f"PnL Update:\n"
                    f"  Realized: {self.realized_pnl:.2f}\n"
                    f"  Unrealized: {self.unrealized_pnl:.2f}\n"
                    f"  Cumulative: {self.cumulative_pnl:.2f}"
                )
                
                await asyncio.sleep(60)  # Log every 60 seconds
                
            except Exception as e:
                logging.error(f"Error in PnL logging: {str(e)}")
                await asyncio.sleep(60)  # Continue logging even after error

    def calculate_unrealized_pnl(self) -> float:
        """Calculate unrealized PnL based on current position"""
        try:
            if self.position_size == 0 or not self.ticker:
                self.unrealized_pnl = 0.0
                return 0.0

            if not self.entry_price or self.entry_price <= 0:
                logging.warning("Invalid entry price for PnL calculation")
                return 0.0

            # Calculate unrealized PnL
            current_price = self.ticker.mark_price
            if current_price <= 0:
                logging.warning("Invalid mark price for PnL calculation")
                return 0.0

            # PnL = (current_price - entry_price) * position_size
            # For short positions, the formula is reversed
            pnl = (current_price - self.entry_price) * self.position_size
            
            self.unrealized_pnl = pnl
            self.cumulative_pnl = self.realized_pnl + self.unrealized_pnl
            
            return pnl

        except Exception as e:
            logging.error(f"Error calculating unrealized PnL: {str(e)}")
            return 0.0

    def update_realized_pnl(self, trade_price: float, trade_amount: float, is_buy: bool):
        """Update realized PnL when a trade occurs"""
        try:
            if not self.entry_price or self.entry_price <= 0:
                return

            # Calculate PnL for this trade
            if is_buy:
                pnl = (self.entry_price - trade_price) * trade_amount
            else:
                pnl = (trade_price - self.entry_price) * trade_amount

            self.realized_pnl += pnl
            self.cumulative_pnl = self.realized_pnl + self.unrealized_pnl
            
            # Store trade information
            self.trade_history.append({
                'timestamp': time.time(),
                'price': trade_price,
                'amount': trade_amount,
                'is_buy': is_buy,
                'pnl': pnl
            })

        except Exception as e:
            logging.error(f"Error updating realized PnL: {str(e)}")

    def align_amount(self, amount: float) -> float:
        """Align amount to 0.0001 precision"""
        return round(abs(amount) * 10000) / 10000

    def validate_order_amount(self, amount: float) -> bool:
        """Validate if amount is properly aligned with tick size"""
        aligned = self.align_amount(amount)
        if abs(aligned - amount) > 1e-8:  # Use small epsilon for float comparison
            logging.warning(f"Amount {amount} not aligned with tick size, should be {aligned}")
            return False
        return True

    async def submit_order(self, direction: th.Direction, amount: float, price: float) -> bool:
        """Submit order with validated parameters"""
        amount, price = self.validate_order_params(amount, price)
        
        if amount <= 0 or price <= 0:
            return False
        
        try:
            await self.thalex.insert(
                direction=direction,
                instrument_name=self.perp_name,
                amount=amount,
                price=price,
                client_order_id=self.client_order_id,
                id=self.client_order_id,
                collar="clamp"  # Add collar parameter to handle price limits
            )
            self.client_order_id += 1
            return True
        except Exception as e:
            logging.error(f"Order submission failed: {str(e)}")
            return False

    async def place_new_quote(self, quote: Quote, side: th.Direction):
        """Place new quote with proper order tracking"""
        async with self.operation_semaphore:
            order_id = self.client_order_id
            if order_id in self.pending_operations:
                return

            self.pending_operations.add(order_id)
            try:
                # Validate quote against price collar
                if not self.ticker or not self.ticker.mark_price:
                    logging.warning("Missing ticker data for quote validation")
                    return

                # Calculate collar bounds (0.5% from mark price)
                collar_buffer = self.ticker.mark_price * 0.005
                max_bid = self.ticker.mark_price - collar_buffer
                min_ask = self.ticker.mark_price + collar_buffer

                # Validate price against collar
                if (side == th.Direction.BUY and quote.price > max_bid) or \
                   (side == th.Direction.SELL and quote.price < min_ask):
                    logging.warning(f"Quote price {quote.price} outside collar bounds for {side}")
                    return

                # Create order object with timestamp
                new_order = Order(
                    id=order_id,
                    price=quote.price,
                    amount=quote.amount,
                    status=OrderStatus.OPEN
                )
                new_order.timestamp = time.time()
                
                # Add to tracking before API call
                side_idx = 0 if side == th.Direction.BUY else 1
                self.orders[side_idx].append(new_order)
                
                # Place order with collar parameter
                await self.thalex.insert(
                    direction=side,
                    instrument_name=self.perp_name,
                    amount=quote.amount,
                    price=quote.price,
                    post_only=True,
                    label=LABEL,
                    client_order_id=order_id,
                    id=order_id,
                    collar="clamp"  # Use clamp mode to handle price limits
                )
                self.client_order_id += 1
                
            except Exception as e:
                # Remove from tracking if API call fails
                side_idx = 0 if side == th.Direction.BUY else 1
                self.orders[side_idx] = [o for o in self.orders[side_idx] if o.id != order_id]
                logging.error(f"Error placing quote: {str(e)}")
                
            finally:
                self.pending_operations.remove(order_id)
                await asyncio.sleep(QUOTING_CONFIG["order_operation_interval"])

    def should_fast_cancel(self, order: Order) -> bool:
        """Check if order should be cancelled due to price movement"""
        if not order.is_open() or not self.ticker:
            return False

        price_diff = abs(order.price - self.ticker.mark_price) / self.ticker.mark_price
        return price_diff > QUOTING_CONFIG["fast_cancel_threshold"]

    async def cleanup_completed_orders(self):
        """Remove completed orders from tracking"""
        async with self.orders_lock:
            for side in [0, 1]:
                self.orders[side] = [
                    order for order in self.orders[side]
                    if order.status in [OrderStatus.OPEN, OrderStatus.PARTIALLY_FILLED]
                    or time.time() - getattr(order, 'timestamp', 0) < 3600  # Keep completed orders for 1 hour
                ]

    async def make_quotes(self) -> List[List[Quote]]:
        """Generate quotes using Avellaneda-Stoikov model"""
        if not await self.check_risk_limits():
            return [[], []]

        try:
            if not self.ticker or not self.tick:
                return [[], []]

            # Get optimal quotes from Avellaneda-Stoikov model
            bid_price, ask_price, bid_size, ask_size = self.calculate_optimal_quotes()
            
            if bid_price <= 0 or ask_price <= 0:
                return [[], []]
                
            # Create quote lists
            bids = []
            asks = []
            
            # Add base quotes
            if bid_size >= 0.001:
                bids.append(Quote(price=bid_price, amount=bid_size))
            if ask_size >= 0.001:
                asks.append(Quote(price=ask_price, amount=ask_size))
                
            # Add additional levels with wider spreads
            for i, level in enumerate(QUOTING_CONFIG["base_levels"][1:], 1):
                spread_multiplier = 1 + (i * 0.5)  # Increase spread by 50% each level
                
                level_bid = self.round_to_tick(bid_price * (1 - 0.0001 * spread_multiplier))
                level_ask = self.round_to_tick(ask_price * (1 + 0.0001 * spread_multiplier))
                
                level_size = bid_size * level["size"] / QUOTING_CONFIG["base_levels"][0]["size"]
                
                if level_size >= 0.001:
                    bids.append(Quote(price=level_bid, amount=level_size))
                    asks.append(Quote(price=level_ask, amount=level_size))

            return [bids, asks]

        except Exception as e:
            logging.error(f"Error generating quotes: {str(e)}")
            return [[], []]

    def calculate_optimal_quotes(self) -> Tuple[float, float, float, float]:
        """Calculate optimal quotes using Avellaneda-Stoikov model"""
        try:
            if not self.ticker or not self.price_history:
                logging.warning("Missing ticker or price history for optimal quotes calculation")
                return 0, 0, 0, 0

            if len(self.price_history) < AVELLANEDA_CONFIG["vol_window"]:
                logging.warning(f"Insufficient price history: {len(self.price_history)} < {AVELLANEDA_CONFIG['vol_window']}")
                return 0, 0, 0, 0

            # Calculate mid price and volatility
            mid_price = self.ticker.mark_price
            if mid_price <= 0:
                logging.warning(f"Invalid mid price: {mid_price}")
                return 0, 0, 0, 0

            # Add price collar buffer (0.5% from mid price)
            collar_buffer = mid_price * 0.005
            max_bid = mid_price - collar_buffer
            min_ask = mid_price + collar_buffer

            volatility = self.calculate_volatility()
            if volatility <= 0:
                logging.warning(f"Invalid volatility: {volatility}")
                return 0, 0, 0, 0

            # Calculate reservation price
            q = self.position_size / AVELLANEDA_CONFIG["position_limit"]
            r = mid_price - q * AVELLANEDA_CONFIG["gamma"] * volatility**2 * AVELLANEDA_CONFIG["inventory_weight"]
            
            # Calculate optimal spread
            spread = AVELLANEDA_CONFIG["gamma"] * volatility**2 + (2/AVELLANEDA_CONFIG["gamma"]) * \
                    np.log(1 + AVELLANEDA_CONFIG["gamma"]/AVELLANEDA_CONFIG["k"])
                    
            # Apply spread limits with collar consideration
            spread = min(max(spread, AVELLANEDA_CONFIG["min_spread"]), AVELLANEDA_CONFIG["max_spread"])
            
            # Calculate initial bid and ask prices
            bid_price = self.round_to_tick(r - spread/2)
            ask_price = self.round_to_tick(r + spread/2)
            
            # Ensure prices respect collar buffer
            bid_price = min(bid_price, max_bid)
            ask_price = max(ask_price, min_ask)
            
            # Calculate optimal sizes with volatility adjustment
            bid_size = self.calculate_optimal_size("bid", q, volatility)
            ask_size = self.calculate_optimal_size("ask", q, volatility)
            
            # Log the calculations for debugging
            logging.debug(f"Optimal quotes: bid={bid_price}, ask={ask_price}, bid_size={bid_size}, ask_size={ask_size}")
            
            return bid_price, ask_price, bid_size, ask_size
            
        except Exception as e:
            logging.error(f"Error calculating optimal quotes: {str(e)}")
            return 0, 0, 0, 0

    def calculate_volatility(self) -> float:
        """Calculate rolling volatility with improved error handling"""
        try:
            if len(self.price_history) < AVELLANEDA_CONFIG["vol_window"]:
                return 0.0
                
            prices = np.array(list(self.price_history))
            if np.any(prices <= 0):
                logging.warning("Invalid prices in history")
                return 0.0
                
            returns = np.diff(np.log(prices[-AVELLANEDA_CONFIG["vol_window"]:]))
            vol = np.std(returns) * np.sqrt(252 * 24 * 60 * 60)  # Annualized volatility
            
            if not np.isfinite(vol):
                logging.warning(f"Invalid volatility calculated: {vol}")
                return 0.0
                
            return vol
            
        except Exception as e:
            logging.error(f"Error calculating volatility: {str(e)}")
            return 0.0

    def calculate_optimal_size(self, side: str, q: float, volatility: float) -> float:
        """Calculate optimal order size based on inventory and volatility"""
        try:
            # Base size from position limit
            base_size = AVELLANEDA_CONFIG["position_limit"] * 0.1  # 10% of position limit
            
            # Inventory adjustment factor
            inventory_factor = np.exp(-AVELLANEDA_CONFIG["gamma"] * abs(q))
            
            # Volatility adjustment
            vol_factor = 1 / (1 + volatility)
            
            # Calculate size with inventory skew
            if side == "bid":
                size = base_size * (2 - inventory_factor) * vol_factor
            else:
                size = base_size * inventory_factor * vol_factor
                
            # Apply notional limits
            size = self.apply_notional_limits(size)
            
            return round(size * 1000) / 1000  # Round to 0.001
            
        except Exception as e:
            logging.error(f"Error calculating optimal size: {str(e)}")
            return 0.001

    def apply_notional_limits(self, size: float) -> float:
        """Apply notional value limits to order size"""
        try:
            if not self.ticker or self.ticker.mark_price <= 0:
                return 0.001
                
            notional = size * self.ticker.mark_price
            max_notional = INVENTORY_CONFIG["max_position_notional"] * 0.2  # Use 20% per order
            
            if notional > max_notional:
                size = max_notional / self.ticker.mark_price
                
            return max(0.001, min(size, POSITION_LIMITS["max_position"] * 0.2))
            
        except Exception as e:
            logging.error(f"Error applying notional limits: {str(e)}")
            return 0.001

    def validate_order_params(self, amount: float, price: float) -> tuple[float, float]:
        """Validate and align order parameters"""
        try:
            # Align amount to 0.001 precision
            aligned_amount = round(amount * 1000) / 1000
            
            # Ensure minimum size
            if aligned_amount < 0.001:
                return 0, 0
            
            # Align price to tick size
            aligned_price = self.round_to_tick(price)
            
            # Validate notional value
            notional = aligned_amount * aligned_price
            if notional > INVENTORY_CONFIG["max_position_notional"] * 0.2:
                aligned_amount = (INVENTORY_CONFIG["max_position_notional"] * 0.2) / aligned_price
                aligned_amount = round(aligned_amount * 1000) / 1000
                
            return aligned_amount, aligned_price
            
        except Exception as e:
            logging.error(f"Error validating order parameters: {str(e)}")
            return 0, 0

    def calculate_signals(self) -> Dict[str, float]:
        """Calculate trading signals with notional awareness"""
        try:
            if len(self.price_history) < SIGNAL_CONFIG["bbands_period"]:
                return {}

            prices = np.array(list(self.price_history))
            
            # Calculate available notional first
            available_notional = self.calculate_available_notional()
            if available_notional <= 0:
                return {"composite": 0}  # No trading if no notional available
            
            signals = {
                "bbands": self.calculate_bbands(prices),
                "momentum": self.calculate_momentum(prices),
                "volume_profile": self.calculate_volume_profile(),
                "trend_strength": self.calculate_trend_strength(prices),
                "volatility_signal": self.calculate_volatility_signal(),
                "available_notional": available_notional
            }
            
            # Calculate composite signal with notional consideration
            signals["composite"] = self.calculate_composite_signal(signals)
            
            return signals
            
        except Exception as e:
            logging.error(f"Error calculating signals: {str(e)}")
            return {"composite": 0}

    def calculate_composite_signal(self, signals: Dict[str, Any]) -> float:
        """Calculate composite signal with position and notional limits"""
        try:
            weights = {
                "bbands": 0.25,
                "momentum": 0.20,
                "volume_profile": 0.20,
                "trend_strength": 0.20,
                "volatility_signal": 0.15
            }
            
            composite = 0
            
            # Bollinger Bands signal
            bb = signals.get("bbands", {}).get("bb_position", 0.5)
            bb_signal = 2 * (bb - 0.5)  # Normalize to [-1, 1]
            composite += bb_signal * weights["bbands"]
            
            # Other signals
            composite += signals.get("momentum", 0) * weights["momentum"]
            composite += signals.get("volume_profile", 0) * weights["volume_profile"]
            composite += signals.get("trend_strength", 0) * weights["trend_strength"]
            composite += signals.get("volatility_signal", 0) * weights["volatility_signal"]
            
            # Normalize composite signal
            composite = np.tanh(composite)
            
            # Apply notional-based dampening
            available_notional = signals.get("available_notional", 0)
            max_notional = INVENTORY_CONFIG["max_position_notional"]
            notional_factor = min(1.0, available_notional / max_notional)
            
            return composite * notional_factor * SIGNAL_CONFIG["signal_size_dampening"]
            
        except Exception as e:
            logging.error(f"Error calculating composite signal: {str(e)}")
            return 0

    def calculate_available_notional(self) -> float:
        """Calculate remaining notional capacity"""
        try:
            if not self.ticker or self.ticker.mark_price <= 0:
                return 0
                
            current_notional = abs(self.position_size * self.ticker.mark_price)
            max_notional = INVENTORY_CONFIG["max_position_notional"]
            
            available = max(0, max_notional - current_notional)
            # Add buffer to prevent exceeding limits
            return available * SIGNAL_CONFIG["notional_utilization_threshold"]
            
        except Exception as e:
            logging.error(f"Error calculating available notional: {str(e)}")
            return 0

    def calculate_bbands(self, prices: np.ndarray) -> Dict[str, float]:
        """Calculate Bollinger Bands"""
        try:
            ma = np.mean(prices[-SIGNAL_CONFIG["bbands_period"]:])
            std = np.std(prices[-SIGNAL_CONFIG["bbands_period"]:])
            
            upper = ma + (SIGNAL_CONFIG["bbands_std"] * std)
            lower = ma - (SIGNAL_CONFIG["bbands_std"] * std)
            
            current_price = prices[-1]
            bb_position = (current_price - lower) / (upper - lower)
            
            return {
                "bb_position": bb_position,
                "upper": upper,
                "lower": lower
            }
            
        except Exception as e:
            logging.error(f"Error calculating Bollinger Bands: {str(e)}")
            return {"bb_position": 0.5, "upper": 0, "lower": 0}

    def calculate_momentum(self, prices: np.ndarray) -> float:
        """Calculate price momentum"""
        try:
            momentum = (prices[-1] / prices[-SIGNAL_CONFIG["momentum_period"]] - 1)
            return momentum
        except Exception as e:
            logging.error(f"Error calculating momentum: {str(e)}")
            return 0

    def calculate_volume_profile(self) -> float:
        """Calculate volume profile signal"""
        try:
            if not hasattr(self, 'trade_history') or len(self.trade_history) < SIGNAL_CONFIG["volume_ma_period"]:
                return 0
                
            recent_trades = self.trade_history[-SIGNAL_CONFIG["volume_ma_period"]:]
            buy_volume = sum(t['amount'] for t in recent_trades if t['is_buy'])
            sell_volume = sum(t['amount'] for t in recent_trades if not t['is_buy'])
            
            if buy_volume + sell_volume == 0:
                return 0
                
            volume_imbalance = (buy_volume - sell_volume) / (buy_volume + sell_volume)
            return volume_imbalance
            
        except Exception as e:
            logging.error(f"Error calculating volume profile: {str(e)}")
            return 0

    def calculate_trend_strength(self, prices: np.ndarray) -> float:
        """Calculate trend strength"""
        try:
            if len(prices) < 20:
                return 0
                
            # Linear regression
            x = np.arange(len(prices))
            slope, _ = np.polyfit(x, prices, 1)
            
            # R-squared calculation
            y_pred = slope * x + _
            r_squared = 1 - (np.sum((prices - y_pred) ** 2) / np.sum((prices - np.mean(prices)) ** 2))
            
            return r_squared * np.sign(slope)
            
        except Exception as e:
            logging.error(f"Error calculating trend strength: {str(e)}")
            return 0

    def calculate_volatility_signal(self) -> float:
        """Calculate volatility-based signal"""
        try:
            if not self.ticker:
                return 0
                
            current_atr = self.calculate_atr()
            if current_atr == 0:
                return 0
                
            volatility_ratio = current_atr / (self.ticker.mark_price * 0.01)  # Compare to 1% of price
            return min(max(-1, volatility_ratio - 1), 1)  # Normalize to [-1, 1]
            
        except Exception as e:
            logging.error(f"Error calculating volatility signal: {str(e)}")
            return 0

class ConfigValidator:
    @staticmethod
    def validate_config():
        """Validate configuration parameters"""
        checks = [
            SPREAD > 0,
            len(BID_SIZES) == len(ASK_SIZES),
            POSITION_LIMITS["max_position"] > 0,
            POSITION_LIMITS["max_notional"] > 0,
            POSITION_LIMITS["stop_loss_pct"] > 0,
            POSITION_LIMITS["base_take_profit_pct"] > POSITION_LIMITS["min_take_profit_pct"],
            POSITION_LIMITS["max_take_profit_pct"] > POSITION_LIMITS["base_take_profit_pct"],
            POSITION_LIMITS["rebalance_threshold"] > 0 and POSITION_LIMITS["rebalance_threshold"] < 1
        ]
        return all(checks)

def plot_pnl(time_history, pnl_history):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time_history, y=pnl_history, mode='lines', name='Cumulative PnL'))
    
    fig.update_layout(
        title='Execution PnL Over Time',
        xaxis_title='Time',
        yaxis_title='Cumulative PnL',
        xaxis=dict(tickmode='linear'),
        yaxis=dict(title='PnL'),
        showlegend=True
    )
    
    fig.show()

async def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    
    import tracemalloc
    tracemalloc.start()
    
    run = True
    while run:
        try:
            thalex = th.Thalex(network=NETWORK)
            perp_quoter = PerpQuoter(thalex)
            tasks = [
                asyncio.create_task(perp_quoter.listen_task()),
                asyncio.create_task(perp_quoter.quote_task()),
                asyncio.create_task(perp_quoter.log_pnl()),  # Add PnL logging task
            ]
            
            logging.info(f"Starting on {NETWORK} {UNDERLYING=}")
            await asyncio.gather(*tasks)
            
        except (websockets.ConnectionClosed, socket.gaierror) as e:
            logging.error(f"Connection error ({e}). Reconnecting...")
            await asyncio.sleep(1)
        except KeyboardInterrupt:
            run = False
            logging.info("Shutting down...")
        except Exception as e:
            logging.exception("Unexpected error:")
            await asyncio.sleep(1)

    # Example condition to plot PnL
    await asyncio.sleep(3600)  # Run for an hour
    plot_pnl(perp_quoter.time_history, perp_quoter.pnl_history)  # Plot PnL after an hour

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    
    async def shutdown(thalex, tasks):
        """Graceful shutdown handler"""
        if thalex.connected():  # Remove await since connected() returns bool
            await thalex.cancel_session(id=CALL_ID_CANCEL_SESSION)
            await thalex.disconnect()
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    async def run_quoter():
        while True:
            thalex = th.Thalex(network=NETWORK)
            perp_quoter = PerpQuoter(thalex)
            tasks = [
                asyncio.create_task(perp_quoter.listen_task()),
                asyncio.create_task(perp_quoter.quote_task()),
                asyncio.create_task(perp_quoter.log_pnl()),  # Add PnL logging task
            ]
            
            try:
                logging.info(f"Starting on {NETWORK} {UNDERLYING=}")
                await asyncio.gather(*tasks)
            except (websockets.ConnectionClosed, socket.gaierror) as e:
                logging.error(f"Connection error ({e}). Reconnecting...")
                await shutdown(thalex, tasks)
                await asyncio.sleep(1)
            except KeyboardInterrupt:
                logging.info("Shutting down...")
                await shutdown(thalex, tasks)
                break
            except Exception as e:
                logging.exception("Unexpected error:")
                await shutdown(thalex, tasks)
                await asyncio.sleep(1)

    try:
        asyncio.run(run_quoter())
    except KeyboardInterrupt:
        logging.info("Shutting down gracefully...")
