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
from enum import Enum
from dataclasses import dataclass, field
import itertools

import thalex as th
from .keys import *  # Import from the local keys.py file

# Import configurations from market_config.py
from ..config.market_config import (
    BOT_CONFIG,
    MARKET_CONFIG,
    TRADING_CONFIG,
    RISK_LIMITS, # Changed from RISK_CONFIG
    ORDERBOOK_CONFIG # Added import for ORDERBOOK_CONFIG
)
from ..thalex_logging import LoggerFactory # Added import

# Basic configuration - use values from MARKET_CONFIG
UNDERLYING = MARKET_CONFIG["underlying"]
LABEL = MARKET_CONFIG["label"]
NETWORK = MARKET_CONFIG["network"]

# Order book configuration - use values from TRADING_CONFIG
# REMOVED global constants: AMEND_THRESHOLD, SPREAD, BID_STEP, ASK_STEP, BID_SIZES, ASK_SIZES
# These were problematic as TRADING_CONFIG does not have an "order" subkey as previously assumed.
# Functionality should rely on QUOTING_CONFIG or AVELLANEDA_CONFIG.

# These configuration dictionaries have been moved to market_config.py
# Reference them from the imported configurations
# POSITION_LIMITS now comes from RISK_LIMITS
# QUOTING_CONFIG now comes from TRADING_CONFIG["quoting"]
# INVENTORY_CONFIG now comes from RISK_CONFIG["inventory"]
# SIGNAL_CONFIG now comes from BOT_CONFIG["technical"]["signal"]
# AVELLANEDA_CONFIG now comes from TRADING_CONFIG["avellaneda"]

# Define convenience variables for imported configurations
POSITION_LIMITS = RISK_LIMITS # Changed from RISK_CONFIG["limits"]

# Construct QUOTING_CONFIG from modern config sources
_quoting_config_source = {
    # From TRADING_CONFIG["avellaneda"]
    "min_spread": TRADING_CONFIG.get("avellaneda", {}).get("min_spread"),
    "max_spread": TRADING_CONFIG.get("avellaneda", {}).get("max_spread"),
    "adverse_selection_threshold": TRADING_CONFIG.get("avellaneda", {}).get("adverse_selection_threshold"), # For amend_threshold mapping
    "size_multipliers": TRADING_CONFIG.get("avellaneda", {}).get("size_multipliers"), # For base_levels logic
    "max_levels": TRADING_CONFIG.get("avellaneda", {}).get("max_levels"), # For base_levels logic
    "base_size": TRADING_CONFIG.get("avellaneda", {}).get("base_size"), # For base_levels logic

    # From TRADING_CONFIG["quote_timing"]
    "min_quote_interval": TRADING_CONFIG.get("quote_timing", {}).get("min_interval"),
    "quote_lifetime": TRADING_CONFIG.get("quote_timing", {}).get("max_lifetime"),
    "order_operation_interval": TRADING_CONFIG.get("quote_timing", {}).get("operation_interval"),
    "max_pending_operations": TRADING_CONFIG.get("quote_timing", {}).get("max_pending"),

    # From TRADING_CONFIG["market_impact"]
    "fast_cancel_threshold": TRADING_CONFIG.get("market_impact", {}).get("fast_cancel_threshold"),
    "market_impact_threshold": TRADING_CONFIG.get("market_impact", {}).get("threshold"), # Could be an alternative for amend_threshold

    # From BOT_CONFIG["connection"]
    "error_retry_interval": BOT_CONFIG.get("connection", {}).get("retry_delay"),

    # From BOT_CONFIG["trading_strategy"]["execution"]
    "post_only": BOT_CONFIG.get("trading_strategy", {}).get("execution", {}).get("post_only"),

    # Custom/Legacy - to be reviewed if still needed or can be mapped
    # "volatility_spread_factor": ???, # Needs a source or default
    # "base_levels": ???, # Needs to be constructed or logic adapted
}
QUOTING_CONFIG = {k: v for k, v in _quoting_config_source.items() if v is not None}

# Fallback for amend_threshold if not found directly, map from adverse_selection or market_impact
if "amend_threshold" not in QUOTING_CONFIG:
    QUOTING_CONFIG["amend_threshold"] = QUOTING_CONFIG.get("adverse_selection_threshold") or QUOTING_CONFIG.get("market_impact_threshold")

# Reconstruct INVENTORY_CONFIG directly from BOT_CONFIG, TRADING_CONFIG, and RISK_LIMITS
# Using original key names for compatibility with existing code in this file.
_inventory_config_source = {
    "max_inventory_imbalance": BOT_CONFIG.get("risk", {}).get("inventory_imbalance_limit"),
    "target_inventory": BOT_CONFIG.get("risk", {}).get("inventory_target"),
    "inventory_fade_time": TRADING_CONFIG.get("avellaneda", {}).get("position_fade_time"),
    "adverse_selection_threshold": TRADING_CONFIG.get("avellaneda", {}).get("adverse_selection_threshold"),
    "inventory_skew_factor": TRADING_CONFIG.get("avellaneda", {}).get("inventory_weight"),
    "max_position_notional": RISK_LIMITS.get("max_position_notional"),
    "min_profit_rebalance": TRADING_CONFIG.get("avellaneda", {}).get("min_profit_rebalance"),
    "gradual_exit_steps": TRADING_CONFIG.get("avellaneda", {}).get("max_loss_threshold"), # Note: source key name is max_loss_threshold
    "inventory_cost_factor": TRADING_CONFIG.get("avellaneda", {}).get("inventory_cost_factor"),
}
INVENTORY_CONFIG = {k: v for k, v in _inventory_config_source.items() if v is not None}

# Ensure keys used in data_models.py are present with defaults if necessary
# e.g. the key "max_loss_threshold" is used in handle_position_loss, 
# and its value was sourced from "gradual_exit_steps" in the legacy config.
if "max_loss_threshold" not in INVENTORY_CONFIG and INVENTORY_CONFIG.get("gradual_exit_steps") is not None:
    INVENTORY_CONFIG["max_loss_threshold"] = INVENTORY_CONFIG["gradual_exit_steps"]

# SIGNAL_CONFIG = BOT_CONFIG.get("technical", {}).get("signal", {}) # Removed SIGNAL_CONFIG
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
        self.tick: Optional[float] = None # This is critical, should be set (e.g. in await_instruments)
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
        self.logger = LoggerFactory.configure_component_logger(
            "perp_quoter", 
            log_file="perp_quoter.log",
            high_frequency=False # Assuming PerpQuoter doesn't need HFT logging by default
        )

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
        """Check position and notional value against limits."""
        # Ensure self.position_size is a float
        position = float(self.position_size) if self.position_size is not None else 0.0
        entry_price = float(self.entry_price) if self.entry_price is not None else 0.0

        if abs(position) >= POSITION_LIMITS.get("max_position", float('inf')):
            logging.warning(f"Position {position} exceeds limit of {POSITION_LIMITS.get('max_position')}")
            await self.handle_risk_breach() # Added await
            return False
        
        # Calculate notional value
        notional = abs(position * entry_price)
        if notional >= POSITION_LIMITS.get("max_notional", float('inf')):
            logging.warning(f"Notional value {notional} exceeds limit of {POSITION_LIMITS.get('max_notional')}")
            await self.handle_risk_breach() # Added await
            return False
            
        # Check rebalance threshold from BOT_CONFIG directly as it's not in RISK_LIMITS
        rebalance_threshold = BOT_CONFIG["risk"].get("position_rebalance_threshold")
        if rebalance_threshold is not None and abs(position) >= POSITION_LIMITS.get("max_position", float('inf')) * rebalance_threshold:
            logging.warning(f"Position {position} approaching max {POSITION_LIMITS.get('max_position')}, consider rebalancing")
            # Potentially trigger rebalance logic here
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
        """Calculate dynamic spread based on ATR and Z-score (absolute monetary value)"""
        if not self.ticker or self.ticker.mark_price == 0:
            self.logger.warning("Ticker or mark_price not available for dynamic spread calculation.")
            # Fallback: use configured min_spread_ticks * self.tick if possible, or a small default if not.
            if self.tick and self.tick > 0:
                min_spread_ticks = QUOTING_CONFIG.get("min_spread", 3.0) # Default to 3.0 ticks
                return min_spread_ticks * self.tick
            return 0.01 # Small absolute fallback if no tick size

        if not self.tick or self.tick <= 0:
            self.logger.warning("Tick size not available or invalid. Cannot calculate dynamic spread accurately using ticks.")
            # Fallback to a small percentage of mark price if ticks are not usable
            min_spread_fallback_pct = 0.001 # 0.1%
            return min_spread_fallback_pct * self.ticker.mark_price

        # min_spread from config is in ticks (e.g., 3.0 ticks)
        min_spread_ticks = QUOTING_CONFIG.get("min_spread", 3.0)
        base_spread_abs = min_spread_ticks * self.tick
        
        # Volatility component (ATR based)
        # atr is assumed to be an absolute price value from self.calculate_atr()
        volatility_spread_factor = QUOTING_CONFIG.get("volatility_spread_factor", 0.5) 
        volatility_adjustment_abs = atr * volatility_spread_factor
        
        calculated_spread_abs = base_spread_abs + volatility_adjustment_abs
        
        # Ensure spread is within min/max limits (absolute monetary values, derived from ticks)
        min_allowable_spread_abs = min_spread_ticks * self.tick
        
        max_spread_ticks = QUOTING_CONFIG.get("max_spread", 25.0) # Default to 25.0 ticks
        max_allowable_spread_abs = max_spread_ticks * self.tick

        # Clamp spread
        final_spread_abs = max(min_allowable_spread_abs, calculated_spread_abs)
        if max_allowable_spread_abs > 0 and max_allowable_spread_abs > min_allowable_spread_abs: # Ensure max is valid and greater than min
            final_spread_abs = min(final_spread_abs, max_allowable_spread_abs)
        
        self.logger.debug(f"Dynamic spread: final_abs={final_spread_abs:.4f} (base_abs={base_spread_abs:.4f}, vol_adj_abs={volatility_adjustment_abs:.4f})")
        return final_spread_abs

    def validate_and_adjust_spread(self, bid_price: float, ask_price: float) -> Tuple[float, float]:
        """Validate and adjust spread to meet minimum and maximum configured requirements (tick-based)."""
        if not self.tick or self.tick <= 0:
            self.logger.warning("Tick size not available or invalid. Cannot validate spread accurately using ticks.")
            return bid_price, ask_price
        if not self.ticker or self.ticker.mark_price == 0: # mark_price might not be strictly needed if all calcs are tick based
            self.logger.warning("Ticker or mark_price not available. Spread validation might be less accurate if fallbacks occur.")
            # Proceeding, but min/max calculations will rely on self.tick only

        current_spread_abs = ask_price - bid_price
        
        # Min spread check (tick-based)
        min_spread_ticks = QUOTING_CONFIG.get("min_spread", 3.0) # Default to 3.0 ticks
        min_required_spread_abs = min_spread_ticks * self.tick

        if current_spread_abs < min_required_spread_abs:
            self.logger.warning(f"Spread {current_spread_abs:.4f} is less than minimum required {min_required_spread_abs:.4f} (ticks: {min_spread_ticks}). Adjusting.")
            adjustment = (min_required_spread_abs - current_spread_abs) / 2
            # Adjust and round to tick. Ensure bid < ask.
            bid_price = self.round_to_tick(bid_price - adjustment)
            ask_price = self.round_to_tick(ask_price + adjustment)
            # Re-check spread after rounding, adjust ask if necessary to meet minimum
            if ask_price - bid_price < min_required_spread_abs:
                ask_price = self.round_to_tick(bid_price + min_required_spread_abs)

        # Max spread check (tick-based)
        max_spread_ticks = QUOTING_CONFIG.get("max_spread", 25.0) # Default to 25.0 ticks
        if max_spread_ticks is not None and max_spread_ticks > 0: # Check if max_spread is configured and positive
            max_allowable_spread_abs = max_spread_ticks * self.tick
            current_spread_after_min_adj = ask_price - bid_price # re-calculate current spread
            
            if max_allowable_spread_abs < min_required_spread_abs: # Sanity check: max spread shouldn't be less than min
                 self.logger.warning(f"Configured max_spread_ticks ({max_spread_ticks}) results in a smaller absolute spread ({max_allowable_spread_abs:.4f}) than min_required_spread_abs ({min_required_spread_abs:.4f}). Ignoring max_spread clamping.")
            elif current_spread_after_min_adj > max_allowable_spread_abs:
                 self.logger.warning(f"Spread {current_spread_after_min_adj:.4f} exceeds max allowable {max_allowable_spread_abs:.4f} (ticks: {max_spread_ticks}). Clamping.")
                 excess_spread = current_spread_after_min_adj - max_allowable_spread_abs
                 # Adjust and round to tick
                 bid_price = self.round_to_tick(bid_price + excess_spread / 2)
                 ask_price = self.round_to_tick(ask_price - excess_spread / 2)
                 # Ensure ask is still above bid after clamping and rounding
                 if ask_price <= bid_price:
                     ask_price = self.round_to_tick(bid_price + self.tick) # Ensure at least one tick spread
                 # Ensure it still meets min spread
                 if ask_price - bid_price < min_required_spread_abs:
                     ask_price = self.round_to_tick(bid_price + min_required_spread_abs)


        self.logger.debug(f"Validated spread: BidP={bid_price:.4f}, AskP={ask_price:.4f}")
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
        """Cancel orders that are stale based on quote_lifetime."""
        if not self.orders: # self.orders is List[List[Order]]
            return

        current_time = time.time()
        orders_to_cancel_candidates = []
        
        # quote_lifetime_val should be present due to QUOTING_CONFIG reconstruction
        quote_lifetime_val = QUOTING_CONFIG.get("quote_lifetime") 
        if quote_lifetime_val is None:
            self.logger.warning("'quote_lifetime' not found in QUOTING_CONFIG. Stale order cleanup might not work as expected.")
            return # Or use a default like 60s

        # Collect all open orders that are stale
        async with self.orders_lock: # Acquire lock for reading self.orders state
            for side_order_list in self.orders:
                for order in side_order_list:
                    if order.is_open(): # Check if order is effectively open
                        order_timestamp = getattr(order, 'timestamp', 0.0)
                        if not isinstance(order_timestamp, (float, int)):
                            order_timestamp = 0.0 # Default if timestamp is invalid
                        
                        if current_time - order_timestamp > quote_lifetime_val:
                            self.logger.info(f"Order {order.id} (timestamp: {order_timestamp:.2f}) identified as stale (age: {current_time - order_timestamp:.2f}s > {quote_lifetime_val}s). Queuing for cancellation.")
                            orders_to_cancel_candidates.append(order)
        
        # Cancel identified stale orders
        if orders_to_cancel_candidates:
            self.logger.debug(f"Found {len(orders_to_cancel_candidates)} stale orders to cancel.")
            for order_to_cancel in orders_to_cancel_candidates:
                try:
                    # fast_cancel_order should ideally handle removal from self.orders upon successful cancellation
                    # or update its status so it's no longer considered active.
                    await self.fast_cancel_order(order_to_cancel)
                except Exception as e:
                    self.logger.error(f"Error during fast_cancel_order for stale order {order_to_cancel.id}: {e}")
                    # Decide if retry or other error handling is needed here for individual cancel failures
        else:
            self.logger.debug("No stale orders found to cancel.")

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
        await self.thalex.login(key_ids[NETWORK], private_keys[NETWORK], id=CALL_ID_LOGIN)
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
                # logging.debug(f"Instrument result: {result}")
                pass # Debug log removed
            elif cid == CALL_ID_SUBSCRIBE:
                logging.info(f"Subscription confirmed: {result}")
            elif cid == CALL_ID_LOGIN:
                logging.info("Login successful")
            elif cid == CALL_ID_SET_COD:
                # logging.debug("Cancel on disconnect set")
                pass # Debug log removed
            elif cid > 99:
                # Handle order results
                if "error" in result:
                    await self.order_error(result["error"], cid)
                else:
                    # logging.debug(f"Order {cid} result: {result}")
                    pass # Debug log removed
            else:
                # logging.debug(f"Result {cid}: {result}")
                pass # Debug log removed
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
            if not self.tick:
                self.logger.warning("Tick size not available for default spread. Returning 0.0")
                return 0.0
            # Using min_spread from config as the default if price window is too short
            min_spread_ticks_from_config = AVELLANEDA_CONFIG.get("min_spread", 3.0) # Assuming 3.0 ticks as default
            self.logger.debug(f"Price window too short, returning default min_spread: {min_spread_ticks_from_config * self.tick}")
            return min_spread_ticks_from_config * self.tick
            
        # Calculate volatility
        # self.sigma is set by update_model_parameters or needs to be calculated here if not.
        # Assuming self.sigma is already up-to-date via another mechanism (e.g., update_model_parameters).
        # If not, it should be: self.sigma = self.calculate_volatility() or similar
        # For this function, we rely on self.sigma being current.
        if self.sigma is None or self.sigma <= 0:
            self.logger.warning(f"Invalid market volatility (sigma: {self.sigma}), cannot calculate optimal spread accurately.")
            if not self.tick:
                self.logger.warning("Tick size not available for fallback spread. Returning 0.0")
                return 0.0
            min_spread_ticks_from_config = AVELLANEDA_CONFIG.get("min_spread", 3.0)
            return min_spread_ticks_from_config * self.tick

        # Optimal spread = γσ²(T-t) + 2/γ log(1 + γ/k)
        # Note: self.T is time horizon. (T-t) implies remaining time. If self.T is total horizon, this might be just self.T.
        # Assuming self.T represents the relevant time factor for the formula.
        # Ensure self.gamma and self.k are valid.
        if self.gamma <= 0 or self.k <= 0:
            self.logger.warning(f"Invalid Avellaneda params: gamma={self.gamma}, k={self.k}. Using default min_spread.")
            if not self.tick:
                self.logger.warning("Tick size not available for fallback spread. Returning 0.0")
                return 0.0
            min_spread_ticks_from_config = AVELLANEDA_CONFIG.get("min_spread", 3.0)
            return min_spread_ticks_from_config * self.tick
            
        calculated_model_spread_abs = (self.gamma * self.sigma**2 * self.T + 
                 2/self.gamma * np.log(1 + self.gamma/self.k))
        
        if not (np.isfinite(calculated_model_spread_abs) and calculated_model_spread_abs > 0):
            self.logger.warning(f"Calculated model spread is not valid: {calculated_model_spread_abs}. Using configured min_spread.")
            if not self.tick:
                self.logger.warning("Tick size not available for fallback spread. Returning 0.0")
                return 0.0
            min_spread_ticks_from_config = AVELLANEDA_CONFIG.get("min_spread", 3.0)
            return min_spread_ticks_from_config * self.tick

        if not self.tick:
            self.logger.warning("Tick size not available, cannot enforce tick-based min spread. Returning model spread.")
            return calculated_model_spread_abs

        min_spread_ticks_from_config = AVELLANEDA_CONFIG.get("min_spread", 3.0) # Defaulting to 3.0 ticks
        min_abs_spread_monetary = min_spread_ticks_from_config * self.tick
        
        final_spread = max(min_abs_spread_monetary, calculated_model_spread_abs)
        self.logger.debug(f"Calculated optimal spread: {final_spread} (model: {calculated_model_spread_abs}, floor: {min_abs_spread_monetary})")
        return final_spread
        
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
        fast_cancel_thresh = QUOTING_CONFIG.get("fast_cancel_threshold", 0.005) # Added .get
        return price_diff > fast_cancel_thresh

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
        """Generate quotes using Avellaneda-Stoikov model and configured levels."""
        if not await self.check_risk_limits():
            self.logger.warning("Risk limits check failed, not generating quotes.")
            return [[], []]

        try:
            if not self.ticker or not self.tick or self.ticker.mark_price <= 0:
                self.logger.warning("Ticker, tick size, or valid mark price not available. Cannot generate quotes.")
                return [[], []]

            l0_bid_price, l0_ask_price, l0_bid_size, l0_ask_size = self.calculate_optimal_quotes()
            
            if not (l0_bid_price > 0 and l0_ask_price > 0 and l0_ask_price > l0_bid_price):
                self.logger.warning(f"Invalid L0 prices from calculate_optimal_quotes: Bid={l0_bid_price}, Ask={l0_ask_price}. Not generating quotes.")
                return [[], []]
                
            bids = []
            asks = []
            
            min_trade_size = TRADING_CONFIG.get("execution", {}).get("min_size", 0.001)
            price_decimals = TRADING_CONFIG.get("execution", {}).get("price_decimals", 2) # Assuming default from BOT_CONFIG
            size_decimals = TRADING_CONFIG.get("execution", {}).get("size_decimals", 3)   # Assuming default from BOT_CONFIG

            # Add base quotes (Level 0)
            if l0_bid_size >= min_trade_size:
                bids.append(Quote(price=round(l0_bid_price, price_decimals), 
                                  amount=round(l0_bid_size, size_decimals), 
                                  instrument=self.perp_name, side="BUY"))
            if l0_ask_size >= min_trade_size:
                asks.append(Quote(price=round(l0_ask_price, price_decimals), 
                                   amount=round(l0_ask_size, size_decimals), 
                                   instrument=self.perp_name, side="SELL"))
                
            # Config for additional levels from TRADING_CONFIG["avellaneda"] (via QUOTING_CONFIG reconstruction)
            max_levels = QUOTING_CONFIG.get("max_levels", 1) 
            level_spacing_pct = QUOTING_CONFIG.get("level_spacing", 0.001) 
            size_multipliers = QUOTING_CONFIG.get("size_multipliers", [1.0]) 

            # Add additional levels (Level 1 onwards)
            # max_levels includes L0. If max_levels is 5, we want L0, L1, L2, L3, L4.
            # Loop for i from 1 to max_levels-1.
            for i in range(1, max_levels):
                # Price adjustment: spread further from L0 prices by i * level_spacing_pct * L0_mid_price
                # L0 mid price can be approximated as (l0_bid_price + l0_ask_price) / 2 or based on self.ticker.mark_price if L0 is symmetric
                l0_mid_for_spacing = (l0_bid_price + l0_ask_price) / 2
                spread_increment_abs = i * level_spacing_pct * l0_mid_for_spacing

                level_bid_p = self.round_to_tick(l0_bid_price - spread_increment_abs)
                level_ask_p = self.round_to_tick(l0_ask_price + spread_increment_abs)

                if not (level_bid_p > 0 and level_ask_p > 0 and level_ask_p > level_bid_p):
                    self.logger.debug(f"Skipping level {i} due to invalid/crossed prices: Bid={level_bid_p}, Ask={level_ask_p}")
                    continue

                current_level_size_multiplier = 1.0 # Default multiplier
                if i < len(size_multipliers): # size_multipliers[0] is for L0, [1] for L1, etc.
                    current_level_size_multiplier = size_multipliers[i]
                elif size_multipliers: # Fallback to last available multiplier if list is too short
                    current_level_size_multiplier = size_multipliers[-1]
                
                level_bid_s = round(l0_bid_size * current_level_size_multiplier, size_decimals)
                level_ask_s = round(l0_ask_size * current_level_size_multiplier, size_decimals)
                
                if level_bid_s >= min_trade_size:
                    bids.append(Quote(price=round(level_bid_p, price_decimals), 
                                      amount=level_bid_s, 
                                      instrument=self.perp_name, side="BUY"))
                if level_ask_s >= min_trade_size:
                    asks.append(Quote(price=round(level_ask_p, price_decimals), 
                                       amount=level_ask_s, 
                                       instrument=self.perp_name, side="SELL"))
            
            self.logger.debug(f"Generated quotes: Bids({len(bids)}), Asks({len(asks)})")
            return [bids, asks]

        except Exception as e:
            self.logger.error(f"Error in make_quotes: {str(e)}, Trace: {traceback.format_exc()}")
            return [[], []]

    def calculate_optimal_quotes(self) -> Tuple[float, float, float, float]:
        """Calculate optimal quotes using Avellaneda-Stoikov model"""
        try:
            if not self.ticker or not self.price_history or self.ticker.mark_price <= 0:
                self.logger.warning("Missing ticker, price history, or valid mark price for optimal quotes calculation")
                return 0, 0, 0, 0

            # Correctly source vol_window from TRADING_CONFIG["volatility"]
            vol_config = TRADING_CONFIG.get("volatility", {})
            vol_window = vol_config.get("window")

            if vol_window is None or len(self.price_history) < vol_window:
                self.logger.warning(f"Insufficient price history: {len(self.price_history)} < {vol_window if vol_window is not None else 'N/A'} for volatility calculation.")
                return 0,0,0,0

            mid_price = self.ticker.mark_price

            # Avellaneda parameters from AVELLANEDA_CONFIG (TRADING_CONFIG["avellaneda"])
            gamma = AVELLANEDA_CONFIG.get("gamma", 0.1) # Risk aversion
            inventory_weight_param = AVELLANEDA_CONFIG.get("inventory_weight", 1.0) # Inventory sensitivity
            k_param = AVELLANEDA_CONFIG.get("k", 1.5) # Order book liquidity parameter
            time_horizon_t = AVELLANEDA_CONFIG.get("time_horizon", 1.0) # Trader's time horizon
            
            # Configured spread limits in ticks
            min_spread_ticks_cfg = AVELLANEDA_CONFIG.get("min_spread", 3.0) # Default 3.0 ticks
            max_spread_ticks_cfg = AVELLANEDA_CONFIG.get("max_spread", 25.0) # Default 25.0 ticks
            
            # Normalized inventory q
            max_pos_for_q_calc = RISK_LIMITS.get("max_position")
            if max_pos_for_q_calc is None or max_pos_for_q_calc == 0:
                self.logger.warning("max_position for q-calculation is not set or zero in RISK_LIMITS.")
                return 0,0,0,0
            q = self.position_size / max_pos_for_q_calc

            volatility_sigma = self.calculate_volatility() 
            if not (volatility_sigma > 0 and np.isfinite(volatility_sigma)):
                self.logger.warning(f"Invalid volatility calculated: {volatility_sigma}. Using fixed fallback if configured.")
                fixed_vol = AVELLANEDA_CONFIG.get("fixed_volatility")
                if fixed_vol is not None and fixed_vol > 0:
                    volatility_sigma = fixed_vol
                    self.logger.info(f"Using fixed_volatility: {volatility_sigma}")
                else:
                    self.logger.error("No valid dynamic volatility and no valid fixed_volatility fallback.")
                    return 0,0,0,0
            
            reservation_price_offset = q * gamma * (volatility_sigma**2) * time_horizon_t * inventory_weight_param
            reservation_price = mid_price - reservation_price_offset
            
            spread_component_variance = gamma * (volatility_sigma**2) * time_horizon_t
            spread_component_liquidity = (2 / (gamma * time_horizon_t) if time_horizon_t > 0 else float('inf')) * np.log(1 + (gamma * time_horizon_t / k_param if k_param > 0 else float('inf'))) 
            if k_param <=0 : spread_component_liquidity = float('inf') 
            
            model_calculated_spread_abs = spread_component_variance + spread_component_liquidity
            
            if not self.tick or self.tick <= 0:
                self.logger.error("Tick size not available or invalid. Cannot apply tick-based spread limits or round prices.")
                # Fallback to a simple percentage of mid_price if tick is not available for safety
                if not (np.isfinite(model_calculated_spread_abs) and model_calculated_spread_abs > 0):
                     model_calculated_spread_abs = mid_price * 0.001 # 0.1% of mid as a desperate fallback
                # Cannot round or apply tick-based limits.
                final_bid_price = reservation_price - model_calculated_spread_abs / 2
                final_ask_price = reservation_price + model_calculated_spread_abs / 2
                # Early exit if no tick size
                bid_size_no_tick = self.calculate_optimal_size("bid", q, volatility_sigma)
                ask_size_no_tick = self.calculate_optimal_size("ask", q, volatility_sigma)
                return final_bid_price, final_ask_price, bid_size_no_tick, ask_size_no_tick

            # Apply tick-based min/max spread limits to the model-calculated absolute spread
            min_abs_spread_from_config = min_spread_ticks_cfg * self.tick
            max_abs_spread_from_config = max_spread_ticks_cfg * self.tick

            if not (np.isfinite(model_calculated_spread_abs) and model_calculated_spread_abs > 0):
                self.logger.warning(f"Model-calculated spread is not valid: {model_calculated_spread_abs}. Falling back to min_spread_ticks * self.tick ({min_abs_spread_from_config}).")
                final_spread_abs = min_abs_spread_from_config
            else:
                # Clamp the model's absolute spread directly
                clamped_model_spread_abs = model_calculated_spread_abs
                if clamped_model_spread_abs < min_abs_spread_from_config:
                    self.logger.debug(f"Model spread {clamped_model_spread_abs:.4f} < min_config_abs {min_abs_spread_from_config:.4f}. Using min_config_abs.")
                    clamped_model_spread_abs = min_abs_spread_from_config
                if max_abs_spread_from_config > 0 and max_abs_spread_from_config > min_abs_spread_from_config and clamped_model_spread_abs > max_abs_spread_from_config: # Ensure max is valid
                    self.logger.debug(f"Model spread {clamped_model_spread_abs:.4f} > max_config_abs {max_abs_spread_from_config:.4f}. Using max_config_abs.")
                    clamped_model_spread_abs = max_abs_spread_from_config
                final_spread_abs = clamped_model_spread_abs
            
            model_bid_price = reservation_price - final_spread_abs / 2
            model_ask_price = reservation_price + final_spread_abs / 2

            exchange_fee_rate = AVELLANEDA_CONFIG.get("exchange_fee_rate", 0.0001) 
            desired_margin_rate = AVELLANEDA_CONFIG.get("desired_margin_rate_above_fee", 0.00025)
            fee_margin_factor = exchange_fee_rate + desired_margin_rate
            adjusted_bid_price = model_bid_price * (1 - fee_margin_factor)
            adjusted_ask_price = model_ask_price * (1 + fee_margin_factor)
            
            final_bid_price = self.round_to_tick(adjusted_bid_price)
            final_ask_price = self.round_to_tick(adjusted_ask_price)
            
            collar_buffer_pct = AVELLANEDA_CONFIG.get("collar_buffer_pct", 0.005) 
            max_allowable_bid = mid_price * (1 - collar_buffer_pct) # Note: mid_price is used here for collar
            min_allowable_ask = mid_price * (1 + collar_buffer_pct) # Note: mid_price is used here for collar

            final_bid_price = min(final_bid_price, self.round_to_tick(max_allowable_bid)) # Round collar limits too
            final_ask_price = max(final_ask_price, self.round_to_tick(min_allowable_ask)) # Round collar limits too

            if final_bid_price >= final_ask_price:
                self.logger.warning(f"Crossed book after all adjustments: Bid={final_bid_price}, Ask={final_ask_price}. Widening around reservation price using min_spread_ticks_cfg.")
                # Fallback using min_spread_ticks_cfg around reservation_price or mid_price
                fallback_spread_abs = min_spread_ticks_cfg * self.tick
                
                temp_bid = self.round_to_tick(reservation_price - fallback_spread_abs / 2)
                temp_ask = self.round_to_tick(reservation_price + fallback_spread_abs / 2)
                if temp_bid < temp_ask:
                    final_bid_price, final_ask_price = temp_bid, temp_ask
                else: 
                    self.logger.warning(f"Still crossed with reservation price. Using mid_price ({mid_price}) for fallback spread centering.")
                    final_bid_price = self.round_to_tick(mid_price - fallback_spread_abs / 2)
                    final_ask_price = self.round_to_tick(mid_price + fallback_spread_abs / 2)
                    if final_bid_price >= final_ask_price: # Final desperate measure
                         final_ask_price = self.round_to_tick(final_bid_price + self.tick)

            bid_size = self.calculate_optimal_size("bid", q, volatility_sigma)
            ask_size = self.calculate_optimal_size("ask", q, volatility_sigma)
            
            # Ensure price_decimals and size_decimals are available, e.g. from self.price_decimals / self.size_decimals if set on init
            # Using hardcoded or BOT_CONFIG lookup for now if not class members
            price_decimals = BOT_CONFIG.get("trading_strategy", {}).get("execution", {}).get("price_decimals", 2)
            size_decimals = BOT_CONFIG.get("trading_strategy", {}).get("execution", {}).get("size_decimals", 3)

            self.logger.debug(f"Optimal L0 Quotes: BidP={final_bid_price:.{price_decimals}f}, AskP={final_ask_price:.{price_decimals}f}, BidS={bid_size:.{size_decimals}f}, AskS={ask_size:.{size_decimals}f} | r={reservation_price:.2f}, final_spread_abs={final_spread_abs:.2f}, vol={volatility_sigma:.4f}, q={q:.4f}")
            return final_bid_price, final_ask_price, bid_size, ask_size
            
        except Exception as e:
            self.logger.error(f"Error calculating optimal quotes: {str(e)}, Trace: {traceback.format_exc()}")
            return 0, 0, 0, 0

    def calculate_volatility(self) -> float:
        """Calculate rolling volatility of log returns for the configured window."""
        try:
            vol_config = TRADING_CONFIG.get("volatility", {})
            vol_window = vol_config.get("window") # Correctly sourced
            vol_floor = vol_config.get("floor", 0.0001) # Min volatility (e.g. 0.01% if returns are daily)
            vol_ceiling = vol_config.get("ceiling", 1.0)  # Max volatility (e.g. 100%)
            # The annualization factor depends on how frequently price_history is updated and what T represents.
            # If T (time_horizon in A-S) is 1 (representing 1 day), then volatility should be daily.
            # If price_history has per-second data, need to sample or adjust. Assuming it aligns with T or is raw.
            # For now, this returns standard deviation of log returns for the window. A-S will use time_horizon_t.

            if vol_window is None or len(self.price_history) < vol_window:
                return 0.0 
                
            # Use the most recent `vol_window` elements from the deque
            prices_in_window = list(itertools.islice(self.price_history, max(0, len(self.price_history) - vol_window), len(self.price_history)))
            if len(prices_in_window) < vol_window:
                 self.logger.debug(f"Not enough price points in deque slice ({len(prices_in_window)}) for volatility window {vol_window}.")
                 return 0.0

            prices = np.array(prices_in_window)
            if np.any(prices <= 0):
                self.logger.warning("Invalid (zero/negative) prices in history window for volatility.")
                return 0.0
                
            log_returns = np.diff(np.log(prices))
            if log_returns.size == 0:
                 self.logger.debug("Not enough price points (after diff) to calculate log returns for volatility.")
                 return 0.0

            calculated_vol_std = np.std(log_returns)
            
            if not (np.isfinite(calculated_vol_std) and calculated_vol_std >= 0):
                self.logger.warning(f"Volatility calculation resulted in non-finite/negative value: {calculated_vol_std}")
                return 0.0
            
            # Apply floor and ceiling
            final_vol = max(vol_floor, min(calculated_vol_std, vol_ceiling))
            self.logger.debug(f"Calculated volatility: {final_vol:.6f} (std of {log_returns.size} log returns, window: {vol_window})")
            return final_vol
            
        except Exception as e:
            self.logger.error(f"Error calculating volatility: {str(e)}, Trace: {traceback.format_exc()}")
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

class ConfigValidator:
    @staticmethod
    def validate_config():
        """Validate configuration parameters"""
        checks = [
            TRADING_CONFIG["avellaneda"]["min_spread"] > 0, # Use TRADING_CONFIG for min_spread
            len(ORDERBOOK_CONFIG["bid_sizes"]) == len(ORDERBOOK_CONFIG["ask_sizes"]), # Use ORDERBOOK_CONFIG
            RISK_LIMITS["max_position"] > 0, # Use RISK_LIMITS directly
            RISK_LIMITS["max_notional"] > 0, # Use RISK_LIMITS directly
            RISK_LIMITS["stop_loss_pct"] > 0, # Use RISK_LIMITS directly
            RISK_LIMITS.get("base_take_profit_pct", 0) > RISK_LIMITS.get("min_take_profit_pct", -1),
            RISK_LIMITS.get("max_take_profit_pct", 0) > RISK_LIMITS.get("base_take_profit_pct", -1),
            RISK_LIMITS.get("rebalance_threshold", 0) > 0 and RISK_LIMITS.get("rebalance_threshold", 0) < 1
        ]
        return all(checks)

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
