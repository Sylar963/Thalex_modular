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
import os
import sys
from datetime import datetime

import thalex as th
import keys  # Rename _keys.py to keys.py and add your keys. There are instructions how to create keys in that file.

# Configuration parameters
UNDERLYING = "BTCUSD"
LABEL = "P"
AMEND_THRESHOLD = 25  # ticks
NETWORK = th.Network.TEST
DEBUG_MODE = False  # Set to True to enable debug monitoring

# Add to configuration section
POSITION_LIMITS = {
    "max_position": 1.0,
    "max_notional": 50000,  # Maximum notional value in USD
    "stop_loss_pct": 0.06,  # 6% stop loss
    "take_profit_pct": 0.01,  # 3% take profit
    "base_take_profit_pct": 0.0033,  # Base take profit
    "max_take_profit_pct": 0.05,   # Maximum take profit
    "min_take_profit_pct": 0.01,   # Minimum take profit
    "trailing_stop_activation": 0.015,  # Activate trailing at 1.5% profit
    "trailing_stop_distance": 0.01,     # 1% trailing distance
    "rebalance_threshold": 0.8,  # Rebalance when position reaches 80% of max
    "take_profit_levels": [
        {"percentage": 0.01, "size": 0.2},  # Take 20% profit at 1%
        {"percentage": 0.02, "size": 0.3},  # Take 30% profit at 2%
        {"percentage": 0.03, "size": 0.3},  # Take 30% profit at 3%
        {"percentage": 0.05, "size": 0.2},  # Take remaining 20% at 5%
    ],
    "trailing_stop_levels": [
        {"activation": 0.015, "distance": 0.01},  # First trailing stop
        {"activation": 0.03, "distance": 0.015},  # Second trailing stop
        {"activation": 0.05, "distance": 0.02},   # Final trailing stop
    ],
}

# Add new risk management configuration
RISK_MANAGEMENT_CONFIG = {
    # Adverse Selection Detection
    "adverse_selection": {
        "loss_speed_threshold": 0.002,  # 0.2% loss per minute threshold
        "measurement_interval": 60,      # Measure loss speed over 60 seconds
        "consecutive_losses": 3,         # Number of consecutive loss intervals before action
    },
    
    # Position Reduction Triggers
    "position_reduction": {
        "notional_warning_threshold": 0.85,    # Start reducing at 85% of max notional
        "notional_critical_threshold": 0.95,   # Aggressive reduction at 95% of max notional
        "reduction_steps": [                   # Multi-step position reduction
            {"threshold": 0.85, "reduce_pct": 0.20},  # Reduce 20% at 85% utilization
            {"threshold": 0.90, "reduce_pct": 0.30},  # Reduce 30% at 90% utilization
            {"threshold": 0.95, "reduce_pct": 0.50},  # Reduce 50% at 95% utilization
        ],
        "min_reduction_interval": 30,    # Minimum seconds between reductions
    },
    
    # Inventory Management
    "inventory_management": {
        "max_holding_time": 3600,        # Maximum position holding time (1 hour)
        "time_decay_factor": 0.1,        # Increase urgency by 10% per hour held
        "max_inventory_imbalance": 0.7,  # Maximum allowed inventory imbalance
        "gradual_reduction": {
            "interval": 300,             # Check every 5 minutes
            "reduction_size": 0.1,       # Reduce 10% if holding too long
        }
    },
    
    # Market Impact
    "market_impact": {
        "price_impact_threshold": 0.003,  # 0.3% price impact threshold
        "volume_impact_threshold": 0.2,   # 20% of average volume
        "impact_measurement_period": 300,  # Measure impact over 5 minutes
    },
    
    # Emergency Procedures
    "emergency": {
        "max_drawdown": 0.10,            # 10% maximum drawdown
        "max_daily_loss": 0.05,          # 5% maximum daily loss
        "panic_close_spread": 0.002,     # 0.2% spread for emergency closure
        "circuit_breaker": {
            "loss_threshold": 0.03,      # 3% loss in
            "time_window": 300,          # 5 minutes
            "cooldown_period": 900       # 15 minutes cooldown
        }
    }
}

# Add to configuration section
INVENTORY_CONFIG = {
    "max_inventory_imbalance": 0.7,  # Maximum allowed inventory imbalance (70%)
    "target_inventory": 0.0,  # Target inventory level
    "inventory_fade_time": 300,  # Time to fade inventory position (seconds)
    "adverse_selection_threshold": 0.002,  # 0.2% price move threshold
    "inventory_skew_factor": 0.3,  # How much to skew quotes based on inventory
    "max_position_notional": 40000,  # Maximum notional position (80% of max_notional)
    "min_profit_rebalance": 0.01,  # Minimum profit to trigger rebalance (1%)
    "gradual_exit_steps": 4,  # Number of steps for gradual position exit
    "max_loss_threshold": 0.03,  # Maximum loss before gradual exit (3%)
    "inventory_cost_factor": 0.0001,  # Cost of holding inventory
}



# Avellaneda-Stoikov configuration
AVELLANEDA_CONFIG = {
    "gamma": 0.1,            # Risk aversion parameter
    "k": 1.5,                # Order book liquidity
    "sigma_window": 100,     # Volatility lookback (precios)
    "T": 4.0,                # Time horizon (horas)
    "q_max": 0.5,            # Max inventory position
    "phi": 0.001,            # Inventory urgency factor
    "eta": 0.0001,           # Order book density
    "min_spread": 0.0001,    # 0.01% spread mínimo
    "max_spread": 0.01,      # 1% spread máximo
    "base_size": 0.1,        # Base order size
    "size_adj_factor": 0.3,  # Inventory size adjustment factor
    "volatility_floor": 0.15, # Minimum annualized volatility
    "position_limit": 50000,  # Position limit in USD
    "inventory_weight": 0.1,  # Inventory impact weight
    "vol_window": 100,       # Window for volatility calculation
    "min_price_history": 20,  # Minimum required price points
    "quote_interval": 1.0,   # Quote update interval in seconds
    "notional_utilization_threshold": 0.8,  # 80% notional utilization threshold
    "reservation_spread": 0.0002,  # Base reservation spread
    "fast_cancel_threshold": 0.002  # 0.2% price move for fast cancel
}

# Add after AVELLANEDA_CONFIG
QUOTING_CONFIG = {
    "base_levels": [
        {"size": 0.1, "distance": 0},
        {"size": 0.2, "distance": 5},
        {"size": 0.3, "distance": 10}
    ],
    "spread_multiplier": 1.5,
    "size_multiplier": 0.8,
    "min_quote_size": 0.001,
    "max_quote_size": 1.0,
    "quote_interval": 1.0,
    "fast_quote_threshold": 0.002,
    "bid_step": 5,
    "ask_step": 5,
    "amend_threshold": 1.0,  # Threshold for amending orders instead of placing new ones
    "min_price_history": 20,  # Minimum price history entries required
    "order_operation_interval": 0.1,  # Interval between order operations
    "error_retry_interval": 1.0,  # Interval to wait after errors
    "min_quote_interval": 0.5,  # Minimum interval between quote updates
    "quote_lifetime": 60.0  # Maximum lifetime of a quote in seconds
}

# Define MAX_HOLDING_TIME constant
MAX_HOLDING_TIME = 300  # Maximum time to hold a position in seconds (1 hour)

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

# Call IDs for Thalex API
CALL_ID_INSTRUMENTS = 0
CALL_ID_INSTRUMENT = 1
CALL_ID_SUBSCRIBE = 2
CALL_ID_LOGIN = 3
CALL_ID_CANCEL_SESSION = 4
CALL_ID_SET_COD = 5

# Order status enumeration
class OrderStatus(enum.Enum):
    OPEN = "open"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    CANCELLED_PARTIALLY_FILLED = "cancelled_partially_filled"
    FILLED = "filled"
    PENDING = "pending"

# Order data structure
class Order:
    def __init__(self, oid: int, price: float, amount: float, status: Optional[OrderStatus] = None, direction: Optional[th.Direction] = None):
        self.id = oid
        self.price = price
        self.amount = amount
        self.status = status or OrderStatus.OPEN
        self.timestamp = time.time()
        self.direction = direction
        
    def is_open(self):
        return self.status in [OrderStatus.OPEN, OrderStatus.PARTIALLY_FILLED, OrderStatus.PENDING]

# Ticker data structure
class Ticker:
    def __init__(self, data: Dict):
        self.mark_price: float = data["mark_price"]
        self.best_bid: Optional[float] = data.get("best_bid_price")
        self.best_ask: Optional[float] = data.get("best_ask_price")
        self.index: float = data["index"]
        self.mark_ts: float = data["mark_timestamp"]
        self.funding_rate: float = data["funding_rate"]

class Quote:
    def __init__(self, price: float, amount: float):
        self.price = price
        self.amount = amount
        
    def to_dict(self):
        return {
            "price": self.price,
            "amount": self.amount
        }
        
    def to_side_quote(self):
        return th.SideQuote(price=self.price, amount=self.amount)

# Add after RISK_MANAGEMENT_CONFIG

class PositionTracker:
    def __init__(self):
        self.position_size = 0.0
        self.entry_price = None
        self.weighted_entries = {}  # Dictionary to track entry prices and sizes
        self.executed_levels = {}   # Track executed amounts per take profit level
        self.highest_profit_per_entry = {}  # Track highest profit per entry point
        self.last_validation_time = 0
        self.validation_interval = 60  # Validate every 60 seconds
        
    def update_position(self, trade_size: float, trade_price: float):
        """Update position with new trade, maintaining weighted average entry"""
        try:
            if self.position_size == 0:
                # New position
                self.position_size = trade_size
                self.entry_price = trade_price
                self.weighted_entries = {trade_price: trade_size}
                self.highest_profit_per_entry = {trade_price: 0.0}
            else:
                # Update existing position
                new_size = self.position_size + trade_size
                
                if abs(new_size) < 0.001:  # Position effectively closed
                    self.reset()
                else:
                    if (trade_size > 0) == (self.position_size > 0):  # Adding to position
                        self.weighted_entries[trade_price] = self.weighted_entries.get(trade_price, 0) + trade_size
                    else:  # Reducing position
                        self._reduce_position(abs(trade_size), trade_price)
                    
                    # Recalculate weighted average entry
                    self._recalculate_weighted_average()
                    self.position_size = new_size

            # Validate after update
            if not self.validate_weighted_average():
                logging.error("Position update validation failed")
                
        except Exception as e:
            logging.error(f"Error updating position: {str(e)}")

    def _recalculate_weighted_average(self):
        """Recalculate weighted average entry price"""
        try:
            total_value = sum(price * size for price, size in self.weighted_entries.items())
            total_size = sum(self.weighted_entries.values())
            
            if total_size > 0:
                self.entry_price = total_value / total_size
            else:
                self.entry_price = None
                
        except Exception as e:
            logging.error(f"Error recalculating weighted average: {str(e)}")

    def validate_weighted_average(self) -> bool:
        """Validate weighted average calculation"""
        try:
            current_time = time.time()
            if current_time - self.last_validation_time < self.validation_interval:
                return True  # Skip validation if too recent
                
            self.last_validation_time = current_time
            
            total_size = sum(self.weighted_entries.values())
            if abs(total_size - self.position_size) > 0.001:
                logging.error(f"Position size mismatch: tracked={self.position_size}, calculated={total_size}")
                return False
                
            if total_size > 0:
                total_value = sum(price * size for price, size in self.weighted_entries.items())
                calculated_avg = total_value / total_size
                if abs(calculated_avg - self.entry_price) > 0.0001:
                    logging.error(f"Entry price mismatch: tracked={self.entry_price}, calculated={calculated_avg}")
                    return False
                    
            return True
            
        except Exception as e:
            logging.error(f"Error validating weighted average: {str(e)}")
            return False

    def get_weighted_entry_price(self) -> Optional[float]:
        """Get current weighted average entry price with validation"""
        try:
            if not self.validate_weighted_average():
                self._recalculate_weighted_average()
            return self.entry_price
        except Exception as e:
            logging.error(f"Error getting weighted entry price: {str(e)}")
            return None

    def _reduce_position(self, reduction_size: float, exit_price: float):
        """Handle position reduction, maintaining FIFO order"""
        remaining_reduction = reduction_size
        entries_to_remove = []
        
        # Sort entries by time (FIFO)
        for entry_price, entry_size in sorted(self.weighted_entries.items()):
            if remaining_reduction <= 0:
                break
                
            if entry_size <= remaining_reduction:
                # Remove entire entry
                remaining_reduction -= entry_size
                entries_to_remove.append(entry_price)
            else:
                # Partial reduction
                self.weighted_entries[entry_price] -= remaining_reduction
                remaining_reduction = 0
                
        # Clean up fully closed entries
        for price in entries_to_remove:
            del self.weighted_entries[price]
            if price in self.highest_profit_per_entry:
                del self.highest_profit_per_entry[price]
    
    def reset(self):
        """Reset all tracking data"""
        self.position_size = 0.0
        self.entry_price = None
        self.weighted_entries.clear()
        self.executed_levels.clear()
        self.highest_profit_per_entry.clear()
    
    def update_take_profit_execution(self, level: float, amount: float):
        """Track executed amount for each take profit level"""
        self.executed_levels[level] = self.executed_levels.get(level, 0) + amount
    
    def get_remaining_size_at_level(self, level: float, target_size: float) -> float:
        """Calculate remaining size to execute at a take profit level"""
        executed = self.executed_levels.get(level, 0)
        total_target = abs(self.position_size) * target_size
        return max(0, total_target - executed)
    
    def update_profit_tracking(self, current_price: float):
        """Update highest profit tracking per entry point"""
        if not self.position_size or not self.entry_price:
            return
            
        for entry_price in self.weighted_entries:
            profit_pct = (current_price - entry_price) / entry_price
            if self.position_size < 0:
                profit_pct = -profit_pct
                
            current_highest = self.highest_profit_per_entry.get(entry_price, 0.0)
            if profit_pct > current_highest:
                self.highest_profit_per_entry[entry_price] = profit_pct

# Perpetual Quoter class
class PerpQuoter:
    def __init__(self, thalex: th.Thalex):
        """Initialize the perpetual quoter with enhanced state management"""
        self.thalex = thalex
        self.logger = logging.getLogger(__name__)
        
        # Exchange and instrument data
        self.perp_name = f"BTC-PERPETUAL"
        self.tick = 1
        self.index = None
        self.ticker = None
        self.instrument = None
        
        # Order tracking
        self.orders = [[], []]  # [buy_orders, sell_orders]
        self.orders_lock = asyncio.Lock()
        self.client_order_id = 100
        self.operation_semaphore = asyncio.Semaphore(5)  # Limit concurrent operations
        self.pending_operations = set()  # Track pending operations
        
        # Position tracking
        self.position_size = 0
        self.position_tracker = PositionTracker()
        self.position_lock = asyncio.Lock()
        self.portfolio = []  # Initialize portfolio as empty list
        
        # Market data
        self.price_history = deque(maxlen=AVELLANEDA_CONFIG["vol_window"] * 2)
        self.trade_history = deque(maxlen=1000)
        self.volatility = AVELLANEDA_CONFIG["volatility_floor"]
        self.sigma = AVELLANEDA_CONFIG["volatility_floor"]
        self.T = AVELLANEDA_CONFIG["T"]
        
        # Performance metrics
        self.realized_pnl = 0
        self.unrealized_pnl = 0
        self.cumulative_pnl = 0
        self.entry_price = 0
        self.last_mark_price = 0
        
        # Quoting state
        self.last_quote_update = 0
        self.quote_update_count = 0
        self.quoting_enabled = True
        self.quote_cv = asyncio.Condition()  # Condition variable for quote signaling
        self.last_quote_task = 0
        
        # Risk management
        self.risk_breaches = {}
        self.alert_timestamps = {}
        self.take_profit_orders = {}
        self.trailing_stops = {}
        
        # Debug and monitoring
        self.debug_info = {}
        self.last_log_time = time.time()
        self.log_interval = 60  # Log PnL every minute
        
        # Initialize performance tracking
        self.performance = {
            "trades": 0,
            "profitable_trades": 0,
            "losing_trades": 0,
            "total_volume": 0,
            "avg_trade_duration": 0,
            "max_drawdown": 0,
            "peak_pnl": 0,
            "trade_history": []
        }
        
        # Initialize quote tracking
        self.quote_performance = {
            "total_quotes": 0,
            "filled_quotes": 0,
            "cancelled_quotes": 0,
            "avg_quote_lifetime": 0,
            "quote_fill_ratio": 0
        }
        
        # Initialize risk metrics
        self.risk_metrics = {
            "max_position": 0,
            "max_notional": 0,
            "daily_loss": 0,
            "max_drawdown": 0,
            "position_duration": 0
        }
        
        # Initialize PnL tracking
        self.pnl_history = []
        self.pnl_timestamps = []
        
        # Set up logger
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            stream=sys.stdout
        )
        self.logger = logging.getLogger(__name__)
        
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
        self.operation_semaphore = asyncio.Semaphore(5)  # Limit concurrent operations
        
        # Add these lines
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
        
        # Add position tracker
        self.position_tracker = PositionTracker()
        
        # Add health monitoring attributes
        self.last_health_check = time.time()
        self.health_check_interval = 60  # seconds
        self.quote_update_count = 0
        self.order_update_count = 0
        self.error_count = 0
        self.last_successful_quote = 0
        self.health_status = "initializing"
        
        # Add deadlock detection
        self.operation_timestamps = {}
        self.operation_timeouts = {
            "quote_task": 120,  # 2 minutes
            "listen_task": 120,  # 2 minutes
            "order_update": 60,  # 1 minute
            "position_update": 60,  # 1 minute
        }

        # Add client order ID lock
        self.client_order_id_lock = asyncio.Lock()
        self.order_tracking_lock = asyncio.Lock()
        self.order_tracking = {}  # Dictionary to track orders by ID
        self.trade_volumes = deque(maxlen=100)  # Añadir para trackear volumen
        self.price_impact = 0.0  # Nuevo campo

    def round_to_tick(self, value):
        return self.tick * round(value / self.tick)

    def align_with_tick(self, value, min_value=0.001):
        """
        Align a value with the instrument's tick size
        
        Args:
            value: The value to align
            min_value: The minimum allowed value (default: 0.001)
            
        Returns:
            The aligned value
        """
        if not hasattr(self, 'tick') or not self.tick:
            logging.warning("Tick size not available, returning unaligned value")
            return value
            
        # Round to the nearest multiple of the tick size
        aligned_value = round(value / self.tick) * self.tick
        
        # Ensure we have at least the minimum value
        if aligned_value < min_value:
            aligned_value = min_value
            
        return aligned_value

    def calculate_zscore(self) -> float:
        """Calculate Z-score for current price"""
        # Lower the threshold to 90% of required entries (90 entries)
        min_required_entries = max(20, int(AVELLANEDA_CONFIG["vol_window"] * 0.9))  # 90% of required entries
        
        if len(self.price_history) < min_required_entries:
            return 0
            
        # If we have enough entries but not the full window, use what we have
        prices = np.array(self.price_history)
        
        # Avoid division by zero
        std_dev = np.std(prices)
        if std_dev == 0:
            return 0
            
        return (self.ticker.mark_price - np.mean(prices)) / std_dev

    def calculate_atr(self):
        """Calculate Average True Range"""
        # Need at least 2 prices to calculate differences
        if len(self.price_history) < 2:
            return 0.0
        
        prices = np.array(list(self.price_history))
        # Use all available price history
        high_low = np.abs(np.diff(prices))
        return np.mean(high_low)

    async def check_risk_limits(self) -> bool:
        """Check if current position exceeds risk limits"""
        try:
            # Get current position and mark price
            position = self.get_position_size()  # Use reconciled position
            
            if not self.ticker or not hasattr(self.ticker, 'mark_price') or self.ticker.mark_price <= 0:
                logging.warning("Missing or invalid mark price for risk check")
                return True  # Allow quoting if we can't check risk
                
            mark_price = self.ticker.mark_price
            
            # Calculate notional value
            notional = abs(position * mark_price)
            
            # Check against position limits
            if abs(position) > POSITION_LIMITS["max_position"]:
                logging.warning(f"Position size limit exceeded: {abs(position)} > {POSITION_LIMITS['max_position']}")
                
                # Record breach for alerting
                self.risk_breaches["position_size"] = {
                    "timestamp": time.time(),
                    "value": abs(position),
                    "limit": POSITION_LIMITS["max_position"]
                }
                
                # Only alert if we haven't recently
                if self.should_alert("position_breach", time.time()):
                    logging.error(f"RISK ALERT: Position size {abs(position)} exceeds limit {POSITION_LIMITS['max_position']}")
                    
                return False
                
            # Check against notional limits
            if notional > POSITION_LIMITS["max_notional"]:
                logging.warning(f"Notional limit exceeded: ${notional:.2f} > ${POSITION_LIMITS['max_notional']}")
                
                # Record breach for alerting
                self.risk_breaches["notional"] = {
                    "timestamp": time.time(),
                    "value": notional,
                    "limit": POSITION_LIMITS["max_notional"]
                }
                
                # Only alert if we haven't recently
                if self.should_alert("notional_breach", time.time()):
                    logging.error(f"RISK ALERT: Notional value ${notional:.2f} exceeds limit ${POSITION_LIMITS['max_notional']}")
                    
                return False
                
            # All checks passed
            return True
            
        except Exception as e:
            logging.error(f"Error checking risk limits: {str(e)}")
            if hasattr(e, "__traceback__"):
                logging.error(f"Traceback: {traceback.format_exc()}")
            return True  # Allow quoting if check fails

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
        """Manage position based on time and risk parameters"""
        try:
            if self.position_size == 0:
                return
            
            # Check position holding time
            current_time = time.time()
            position_duration = current_time - self.position_entry_time
            
            # Log position information
            logging.debug(f"Position management: size={self.position_size}, duration={position_duration:.1f}s")
            
            # Check if position has been held too long
            if position_duration > MAX_HOLDING_TIME:
                logging.info(f"Position held for {position_duration:.1f}s > {MAX_HOLDING_TIME}s, initiating gradual reduction")
                await self.gradual_rebalance()
                
            # Check for adverse movement
            if await self.check_adverse_selection():
                logging.warning("Adverse selection detected, initiating position reduction")
                await self.execute_position_reduction(0.5)  # Reduce by 50%
                
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
            
            # Align size to valid increment using the utility method
            reduction_size = self.align_with_tick(reduction_size)
            logging.info(f"Aligned notional breach reduction size to {reduction_size}")
            
            if reduction_size < 0.001:
                return
            
            direction = th.Direction.SELL if self.position_size > 0 else th.Direction.BUY
            
            # Split into smaller orders
            num_orders = 3
            size_per_order = self.align_with_tick(reduction_size / num_orders)
            logging.info(f"Aligned size per order to {size_per_order}")
            
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
        """Spread dinámico usando modelo Avellaneda-Stoikov"""
        # Base spread desde configuración
        base_spread = AVELLANEDA_CONFIG["reservation_spread"] * self.tick
        
        # Componente de inventario
        inventory_impact = AVELLANEDA_CONFIG["inventory_weight"] * abs(self.position_size)
        
        # Componente de volatilidad
        volatility_component = AVELLANEDA_CONFIG["gamma"] * (self.sigma**2) * self.T
        
        spread = base_spread + inventory_impact + volatility_component
        
        return min(
            AVELLANEDA_CONFIG["max_spread"] * self.tick,
            max(AVELLANEDA_CONFIG["min_spread"] * self.tick, spread)
        )

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
                async with self.orders_lock:
                    orders = self.orders[side_i]
                    quotes = desired[side_i]
                    
                    # Create set of desired prices for quick lookup
                    desired_prices = {quote.price for quote in quotes}
                    
                    # Cancel orders not matching desired quotes
                    for order in orders[:]:  # Use slice to avoid modification during iteration
                        if order.is_open():
                            if order.price not in desired_prices:
                                logging.info(f"Cancelling undesired {side.name.lower()} order: {order.id} at price {order.price}")
                                await self.fast_cancel_order(order)

            # Wait a small interval for cancellations to process
            await asyncio.sleep(QUOTING_CONFIG["order_operation_interval"])

            # Now place new quotes
            operations_count = 0
            max_operations = 10  # Maximum operations per cycle
            
            for side_i, side in enumerate([th.Direction.BUY, th.Direction.SELL]):
                quotes = desired[side_i]
                logging.info(f"Processing {side.name} quotes: {len(quotes)} quotes")
                
                async with self.orders_lock:
                    existing_prices = {order.price for order in self.orders[side_i] if order.is_open()}
                
                for quote in quotes:
                    if operations_count >= max_operations:
                        logging.info("Reached maximum operations per cycle")
                        return
                        
                    # Check if we should place this quote
                    logging.info(f"Checking if we should place new quote: {side.name} {quote.amount} @ {quote.price}")
                    if quote.price not in existing_prices and self.should_place_new_quote(quote, side):
                        await self.place_new_quote(quote, side)
                        operations_count += 1
                    else:
                        logging.info(f"Not placing {side.name} quote: Similar order exists (price diff: 0, size diff: 0.0)")

        except Exception as e:
            logging.error(f"Error in quote adjustment: {str(e)}")
            if hasattr(e, "__traceback__"):
                logging.error(f"Traceback: {traceback.format_exc()}")
            await asyncio.sleep(QUOTING_CONFIG["error_retry_interval"])

    async def place_new_quote(self, quote: Quote, side: th.Direction):
        """Place new quote with proper order tracking"""
        async with self.operation_semaphore:
            order_id = await self.get_next_client_order_id()
            if order_id in self.pending_operations:
                return

            self.pending_operations.add(order_id)
            try:
                # Validate quote parameters
                if quote.price <= 0 or quote.amount <= 0:
                    logging.warning(f"Invalid quote parameters: price={quote.price}, amount={quote.amount}")
                    return
                
                # Create order object with timestamp
                new_order = Order(
                    oid=order_id,
                    price=quote.price,
                    amount=quote.amount,
                    status=OrderStatus.PENDING,
                    direction=side
                )
                new_order.timestamp = time.time()  # Add timestamp for cleanup
                
                # Add to tracking before API call
                side_idx = 0 if side == th.Direction.BUY else 1
                await self.add_order_to_tracking(new_order)
                
                # Submit the order
                logging.info(f"Placing {side.name} quote: {quote.amount} @ {quote.price}")
                
                # Place order - note that insert doesn't return a result directly
                # Results are handled through callbacks
                await self.thalex.insert(
                    direction=side,
                    instrument_name=self.perp_name,
                    amount=quote.amount,
                    price=quote.price,
                    client_order_id=order_id,
                    post_only=True
                )
                
            except Exception as e:
                # Remove from tracking if API call fails
                await self.remove_order_from_tracking(order_id)
                logging.error(f"Error placing quote: {str(e)}")
                if hasattr(e, "__traceback__"):
                    logging.error(f"Traceback: {traceback.format_exc()}")
                
            finally:
                self.pending_operations.remove(order_id)
                await asyncio.sleep(QUOTING_CONFIG["order_operation_interval"])

    def should_place_new_quote(self, quote: Quote, side: th.Direction) -> bool:
        """Check if we should place a new quote"""
        try:
            # Check if we already have a similar quote
            for order in self.orders[0 if side == th.Direction.BUY else 1]:
                if order.is_open():
                    # Check if price is close enough
                    price_diff = abs(order.price - quote.price)
                    if price_diff < AMEND_THRESHOLD * self.tick:
                        return False
                        
            return True
        except Exception as e:
            logging.error(f"Error in should_place_new_quote: {str(e)}")
            return False

    async def cleanup_stale_orders(self):
        """Remove stale orders from tracking and cancel stale orders on the exchange"""
        try:
            current_time = time.time()
            stale_threshold = 60  # 1 minute - reduced from 5 minutes for more active management
            
            async with self.orders_lock:
                for side in [0, 1]:
                    stale_orders = []
                    for order in self.orders[side]:
                        # Check if order is stale but still open
                        if (order.is_open() and 
                            hasattr(order, 'timestamp') and 
                            current_time - order.timestamp > stale_threshold):
                            stale_orders.append(order)
                            logging.info(f"Found stale order to cancel: {order.id} at price {order.price}")
                    
                    # Cancel stale orders
                    for order in stale_orders:
                        await self.fast_cancel_order(order)
                    
                    # Keep only non-stale orders or open orders that weren't stale
                    self.orders[side] = [
                        order for order in self.orders[side]
                        if (
                            (order.is_open() and 
                             (not hasattr(order, 'timestamp') or 
                              current_time - order.timestamp <= stale_threshold)) or  # Keep non-stale open orders
                            (not order.is_open() and 
                             hasattr(order, 'timestamp') and 
                             current_time - order.timestamp < stale_threshold)  # Keep recent completed orders
                        )
                    ]
                    
            # Validate position after cleanup
            await self.validate_position_state()
            
        except Exception as e:
            logging.error(f"Error cleaning up stale orders: {str(e)}")
            if hasattr(e, "__traceback__"):
                logging.error(f"Traceback: {traceback.format_exc()}")

    async def quote_task(self):
        """Enhanced quoting loop with state validation and periodic cleanup"""
        last_cancel_all_time = time.time()
        cancel_all_interval = 300  # Cancel all orders every 5 minutes
        last_duplicate_check_time = time.time()
        duplicate_check_interval = 60  # Check for duplicates every minute
        
        while True:
            try:
                # Validate state before quoting
                if not self.ticker or not self.index:
                    logging.warning("Missing ticker or index data, skipping quote cycle")
                    await asyncio.sleep(1)
                    continue
                    
                if not await self.validate_position_state():
                    logging.warning("Invalid position state, skipping quote cycle")
                    await asyncio.sleep(1)
                    continue
                
                # Cleanup stale orders periodically
                await self.cleanup_stale_orders()
                
                # Periodically cancel all orders to prevent accumulation
                current_time = time.time()
                if current_time - last_cancel_all_time > cancel_all_interval:
                    await self.cancel_all_orders()
                    last_cancel_all_time = current_time
                
                # Periodically check for and clean up duplicate orders
                if current_time - last_duplicate_check_time > duplicate_check_interval:
                    await self.cleanup_duplicate_orders()
                    last_duplicate_check_time = current_time
                
                # Wait for price history
                if len(self.price_history) < QUOTING_CONFIG["min_price_history"]:
                    logging.info(f"Price history size: {len(self.price_history)}/{QUOTING_CONFIG['min_price_history']}")
                    await asyncio.sleep(0.1)
                    continue
                    
                logging.info(f"Price history size: {len(self.price_history)}/{QUOTING_CONFIG['min_price_history']}")
                logging.info(f"Price history threshold reached: {len(self.price_history)} >= {QUOTING_CONFIG['min_price_history']}")
                
                # Generate and place quotes
                await self.update_model_parameters()
                quotes = await self.make_quotes()
                if quotes:
                    await self.adjust_quotes(quotes)
                
                # Non-blocking tasks
                asyncio.create_task(self.manage_take_profit())
                
                # Sleep between quote updates
                await asyncio.sleep(QUOTING_CONFIG["quote_interval"])
                
            except Exception as e:
                logging.error(f"Error in quote_task: {str(e)}")
                if hasattr(e, "__traceback__"):
                    logging.error(f"Traceback: {traceback.format_exc()}")
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
        logging.info("Starting listen task")
        
        try:
            logging.info("Connecting to exchange")
            await self.thalex.connect()
            logging.info("Connected to exchange")
            
            logging.info("Fetching instruments")
            await self.await_instruments()
            logging.info(f"Found instrument: {self.perp_name}, tick size: {self.tick}")
            
            logging.info("Logging in")
            await self.thalex.login(keys.key_ids[NETWORK], keys.private_keys[NETWORK], id=CALL_ID_LOGIN)
            logging.info("Setting cancel on disconnect")
            await self.thalex.set_cancel_on_disconnect(6, id=CALL_ID_SET_COD)
            
            logging.info("Subscribing to private channels")
            await self.thalex.private_subscribe(["session.orders", "account.portfolio", "account.trade_history"], id=CALL_ID_SUBSCRIBE)
            logging.info("Subscribing to public channels")
            await self.thalex.public_subscribe([f"ticker.{self.perp_name}.raw", f"price_index.{UNDERLYING}"], id=CALL_ID_SUBSCRIBE)
            
            logging.info("Starting main listen loop")
            while True:
                try:
                    await self.manage_position()
                    msg = await self.thalex.receive()
                    msg = json.loads(msg)
                    
                    if "channel_name" in msg:
                        channel = msg["channel_name"]
                        logging.debug(f"Received notification from channel: {channel}")
                        await self.notification(channel, msg["notification"])
                    elif "result" in msg:
                        logging.debug(f"Received result for message ID: {msg.get('id')}")
                        await self.result_callback(msg["result"], msg.get("id"))
                    elif "error" in msg:
                        logging.error(f"Received error: {msg['error']} for message ID: {msg.get('id')}")
                        await self.error_callback(msg["error"], msg.get("id"))
                    else:
                        logging.warning(f"Received unknown message type: {msg}")
                except Exception as e:
                    logging.error(f"Error in listen loop: {str(e)}\nTraceback: {traceback.format_exc()}")
                    await asyncio.sleep(1)  # Brief pause before continuing
        except Exception as e:
            logging.error(f"Fatal error in listen task: {str(e)}\nTraceback: {traceback.format_exc()}")
            # Re-raise to allow the main loop to handle reconnection
            raise

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
                
            # Update last quote update time for health monitoring
            self.last_quote_update = time.time()
            
            if channel.startswith("ticker."):
                self.quote_update_count += 1
                await self.ticker_callback(notification)
            elif channel.startswith("price_index."):
                self.quote_update_count += 1
                await self.index_callback(notification["price"])
            elif channel == "session.orders":
                self.order_update_count += 1
                self.orders_callback(notification)
            elif channel == "account.portfolio":
                self.portfolio_callback(notification)
            elif channel == "session.trades":
                await self.trades_callback(notification)
            else:
                logging.debug(f"Unhandled notification channel: {channel}")
        except Exception as e:
            self.error_count += 1
            logging.error(f"Error processing notification: {str(e)}")

    async def ticker_callback(self, ticker: Dict[str, Any]):
        """Handle ticker updates."""
        try:
            self.ticker = Ticker(ticker)
            
            # Add the new price to the price history
            self.price_history.append(self.ticker.mark_price)
            
            # Log the current price history size every 10 updates
            if len(self.price_history) % 10 == 0 or len(self.price_history) >= 90:
                logging.info(f"Price history size: {len(self.price_history)}/{AVELLANEDA_CONFIG['vol_window']}")
            
            # Check if we've reached the threshold for quoting
            min_required_entries = max(20, int(AVELLANEDA_CONFIG["vol_window"] * 0.9))
            if len(self.price_history) >= min_required_entries:
                logging.info(f"Price history threshold reached: {len(self.price_history)} >= {min_required_entries}")
            
            # Notify the quote task that we have a new price
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
        """Process trades with PnL tracking and position updates"""
        for t in trades:
            if t.get("label") == LABEL:
                try:
                    amount = float(t.get("amount", 0))
                    price = float(t.get("price", 0))
                    direction = t.get("direction")
                    
                    if amount <= 0 or price <= 0 or not direction:
                        logging.warning(f"Invalid trade data: {t}")
                        continue
                        
                    # Update position tracker first
                    trade_size = amount if direction == "buy" else -amount
                    self.position_tracker.update_position(trade_size, price)
                    
                    # Validate position tracking after update
                    if not self.position_tracker.validate_weighted_average():
                        logging.error("Position tracking validation failed after trade")
                        self.position_tracker._recalculate_weighted_average()
                    
                    # Update internal tracking
                    self.position_size = self.position_tracker.position_size
                    self.entry_price = self.position_tracker.entry_price
                    
                    # Log position update
                    logging.info(f"Position Update:")
                    logging.info(f"  Trade: {amount} @ {price} ({direction})")
                    logging.info(f"  New Position: {self.position_size}")
                    logging.info(f"  Weighted Entry: {self.entry_price}")
                    
                    # Update PnL tracking
                    self.update_realized_pnl(price, amount, direction == "buy")
                    
                    # Update performance metrics
                    await self.update_performance_metrics(t)
                    
                except Exception as e:
                    logging.error(f"Error processing trade: {str(e)}")
                    logging.error(f"Trade data: {t}")

        # Añadir al final
        # Trackear impacto de mercado
        total_volume = sum(float(t['amount']) for t in trades)
        self.trade_volumes.append(total_volume)
        
        # Calcular impacto de precio
        if self.ticker and len(trades) > 0:
            price_change = abs(trades[-1]['price'] - self.ticker.mark_price) / self.ticker.mark_price
            self.price_impact = max(self.price_impact, price_change)

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
        """Handle error messages from the exchange with improved error handling"""
        self.logger.error(f"Received error: {error} for message ID: {cid}")
        
        if cid is None:
            return
            
        try:
            cid_int = int(cid)
            if cid_int >= 100:  # All order IDs are >= 100
                await self.order_error(error, cid_int)
        except (TypeError, ValueError):
            self.logger.warning(f"Invalid message ID format: {cid}")

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
                # Create a task for the async update_order call
                asyncio.create_task(self.handle_order_update(order, o))
        except Exception as e:
            logging.error(f"Error in orders_callback: {str(e)}\nTrace: {traceback.format_exc()}")

    async def handle_order_update(self, order: Order, original_data: Dict):
        """Handle order updates asynchronously"""
        try:
            # First check if order exists in any side's tracking
            found = False
            for side in [0, 1]:
                for i, o in enumerate(self.orders[side]):
                    if o.id == order.id:
                        found = True
                        # Update order status
                        self.orders[side][i] = order
                        
                        # Handle filled orders
                        if order.status == OrderStatus.FILLED:
                            logging.info(f"Order {order.id} filled at price {order.price} amount {order.amount}")
                            # Update position tracking
                            direction = original_data.get("direction", "")
                            side_sign = 1 if direction == "buy" else -1
                            self.position_size += side_sign * order.amount
                            
                        # Handle cancelled orders
                        elif order.status in [OrderStatus.CANCELLED, OrderStatus.CANCELLED_PARTIALLY_FILLED]:
                            logging.info(f"Order {order.id} cancelled/partially filled")
                            # Remove from tracking
                            self.orders[side] = [o for o in self.orders[side] if o.id != order.id]
                        break
                if found:
                    break
                    
            if not found:
                logging.warning(
                    f"Order not found in tracking:\n"
                    f"  ID: {order.id}\n"
                    f"  Status: {order.status}\n"
                    f"  Price: {order.price}\n"
                    f"  Amount: {order.amount}"
                )
                # Add the order to the appropriate list if it's not found and still open
                if order.is_open():
                    direction = original_data.get("direction", "")
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
                    await self.validate_position_state()
                    
        except Exception as e:
            logging.error(f"Error handling order update: {str(e)}\nTrace: {traceback.format_exc()}")

    def portfolio_callback(self, portfolio: List[Dict]):
        """Process portfolio updates with improved position tracking"""
        try:
            # Initialize portfolio attribute if not already done
            if not hasattr(self, 'portfolio'):
                self.portfolio = []
                
            # Update portfolio
            self.portfolio = portfolio
            
            # Extract position for our instrument
            for item in portfolio:
                if item.get('instrument_name') == self.perp_name:
                    new_position = item.get('size', 0)
                    mark_price = item.get('mark_price', 0)
                    
                    # Update position tracking
                    if new_position != self.position_size:
                        logging.info(f"Position changed: {self.position_size:.3f} -> {new_position:.3f}")
                        
                        # Update position tracker
                        if mark_price > 0:
                            self.position_tracker.update_position(new_position, mark_price)
                            logging.info(f"Updated entry price to {self.position_tracker.entry_price}")
                        
                        # Update internal tracking
                        self.position_size = new_position
                        self.entry_price = self.position_tracker.entry_price
                        
                    # Update mark price for PnL calculations
                    if mark_price > 0:
                        self.last_mark_price = mark_price
                        
                    # Check if position needs rebalancing
                    if abs(new_position) > 0:
                        asyncio.create_task(self.gradual_rebalance())
                        
                    break
                    
        except Exception as e:
            logging.error(f"Error processing portfolio update: {str(e)}")
            if hasattr(e, "__traceback__"):
                logging.error(f"Traceback: {traceback.format_exc()}")

    def order_from_data(self, data: Dict) -> Order:
        return Order(
            oid=data["client_order_id"],
            price=data["price"],
            amount=data["amount"],
            status=OrderStatus(data["status"]),
            direction=th.Direction(data["direction"])
        )

    # Add to existing class
    async def result_callback(self, result, cid=None):
        """Handle API call results within class context"""
        try:
            logging.debug(f"Received result callback: cid={cid}, result={result}")
            
            if cid is None:
                logging.warning("Received result with None cid")
                return
                
            if cid == CALL_ID_INSTRUMENT:
                logging.debug(f"Instrument result: {result}")
            elif cid == CALL_ID_SUBSCRIBE:
                logging.info(f"Subscription confirmed: {result}")
            elif cid == CALL_ID_LOGIN:
                logging.info("Login successful")
            elif cid == CALL_ID_SET_COD:
                logging.debug("Cancel on disconnect set")
            elif isinstance(cid, int) and cid > 99:  # Ensure cid is an integer and > 99
                # Handle order results
                if isinstance(result, dict) and "error" in result:
                    await self.order_error(result["error"], cid)
                else:
                    logging.info(f"Order {cid} result: {result}")
                    
                    # Check if this is a successful order placement
                    if isinstance(result, dict) and result.get("order_state") in ["open", "filled", "partially_filled"]:
                        logging.info(f"Order {cid} successfully placed: {result.get('order_state')}")
            else:
                logging.debug(f"Result {cid}: {result}")
        except Exception as e:
            logging.error(f"Error in result_callback: {str(e)}\nTraceback: {traceback.format_exc()}")

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
        """Handle multiple take profit levels with partial fill tracking"""
        if not self.ticker or not self.position_size:
            return

        for level in POSITION_LIMITS["take_profit_levels"]:
            level_pct = level["percentage"]
            target_size = level["size"]
            
            # Calculate remaining size for this level
            remaining_size = self.position_tracker.get_remaining_size_at_level(level_pct, target_size)
            
            if remaining_size >= 0.001 and current_pnl_pct >= level_pct:
                try:
                    direction = th.Direction.SELL if self.position_size > 0 else th.Direction.BUY
                    price = self.calculate_take_profit_price(level_pct)
                    
                    order_id = self.client_order_id
                    await self.thalex.insert(
                        direction=direction,
                        instrument_name=self.perp_name,
                        amount=remaining_size,
                        price=price,
                        client_order_id=order_id,
                        id=order_id
                    )
                    self.client_order_id += 1
                    
                    self.take_profit_orders[order_id] = {
                        "level": level_pct,
                        "amount": remaining_size,
                        "price": price
                    }
                    
                    # Update executed amount tracking
                    self.position_tracker.update_take_profit_execution(level_pct, remaining_size)
                    
                    logging.info(f"Take profit order placed: {remaining_size} @ {price} (Level: {level_pct*100}%)")
                    
                except Exception as e:
                    logging.error(f"Error placing take profit order: {str(e)}")

    async def handle_trailing_stops(self, current_pnl_pct: float):
        """Handle multiple trailing stop levels with per-entry tracking"""
        # Update profit tracking for all entry points
        self.position_tracker.update_profit_tracking(self.ticker.mark_price)
        
        for level in POSITION_LIMITS["trailing_stop_levels"]:
            activation = level["activation"]
            distance = level["distance"]
            
            # Check each entry point
            for entry_price, highest_profit in self.position_tracker.highest_profit_per_entry.items():
                if highest_profit >= activation:
                    # Calculate trailing stop level for this entry
                    trailing_stop_level = highest_profit - distance
                    
                    # Get current profit for this entry
                    current_profit = (self.ticker.mark_price - entry_price) / entry_price
                    if self.position_size < 0:
                        current_profit = -current_profit
                    
                    if current_profit < trailing_stop_level:
                        # Calculate size to exit for this entry point
                        exit_size = self.position_tracker.weighted_entries.get(entry_price, 0)
                        if exit_size >= 0.001:
                            await self.execute_trailing_stop(activation, current_profit, exit_size)
                        break  # Exit after first trailing stop hit

    async def execute_trailing_stop(self, activation: float, current_pnl_pct: float, exit_size: float):
        """Execute trailing stop with specific size"""
        try:
            if exit_size < 0.001:  # Check minimum order size
                return
            
            direction = th.Direction.SELL if self.position_size > 0 else th.Direction.BUY
            price = self.calculate_trailing_stop_price(direction)
            
            await self.thalex.insert(
                direction=direction,
                instrument_name=self.perp_name,
                amount=exit_size,
                price=price,
                client_order_id=self.client_order_id,
                id=self.client_order_id
            )
            self.client_order_id += 1
            
            logging.info(f"Trailing stop executed at {activation*100}% level: {exit_size} @ {price}")
            
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
        """Clean up existing take profit orders with proper state management"""
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
                    # Revert the executed amount tracking
                    self.position_tracker.executed_levels[order_info["level"]] -= order_info["amount"]
                    if self.position_tracker.executed_levels[order_info["level"]] <= 0:
                        del self.position_tracker.executed_levels[order_info["level"]]
                    
                    del self.take_profit_orders[order_id]
                    
            except Exception as e:
                logging.error(f"Error cleaning up take profit order {order_id}: {str(e)}")

    async def validate_position_state(self) -> bool:
        """Validate position state and reconcile with exchange if needed"""
        try:
            # Check if we have portfolio data
            if not hasattr(self, 'portfolio') or not self.portfolio:
                logging.warning("No portfolio data available for validation")
                return False
                
            # Get position from portfolio
            exchange_position = 0
            for item in self.portfolio:
                if item.get('instrument_name') == self.perp_name:
                    exchange_position = item.get('size', 0)
                    break
                    
            # Compare with local position
            if abs(self.position_size - exchange_position) > 0.0001:  # Allow small rounding differences
                logging.warning(f"Position mismatch detected: local={self.position_size}, exchange={exchange_position}")
                
                # Update local position to match exchange
                logging.info(f"Updating position tracker with exchange position: {exchange_position}")
                self.position_size = exchange_position
                
                # Update position tracker state
                if self.ticker and self.ticker.mark_price:
                    logging.info(f"Position state updated with current mark price: {self.ticker.mark_price}")
                    self.position_tracker.update_position(exchange_position, self.ticker.mark_price)
                    
            return True
            
        except Exception as e:
            logging.error(f"Error validating position state: {str(e)}")
            if hasattr(e, "__traceback__"):
                logging.error(f"Traceback: {traceback.format_exc()}")
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
        """Calculate current position PnL percentage using weighted average entry price"""
        if self.position_size == 0 or not self.ticker:
            return 0.0
        
        try:
            # Use position tracker's weighted average entry price
            weighted_entry = self.position_tracker.get_weighted_entry_price()
            if weighted_entry is None or weighted_entry <= 0:
                logging.error(f"Invalid weighted entry price: {weighted_entry}")
                return 0.0
                
            if self.ticker.mark_price <= 0:
                logging.error(f"Invalid mark price: {self.ticker.mark_price}")
                return 0.0
                
            direction = 1 if self.position_size > 0 else -1
            pnl = direction * (self.ticker.mark_price - weighted_entry) / weighted_entry
            
            # Add detailed logging for significant PnL changes
            if abs(pnl) > 0.01:  # Log when PnL exceeds 1%
                logging.info(f"Significant PnL: {pnl*100:.2f}%")
                logging.info(f"  Entry: {weighted_entry}")
                logging.info(f"  Current: {self.ticker.mark_price}")
                logging.info(f"  Position: {self.position_size}")
                
            return pnl
            
        except Exception as e:
            logging.error(f"Error calculating PnL: {str(e)}")
            return 0.0

    async def check_stop_loss(self) -> bool:
        """Check if stop loss has been hit using weighted average entry"""
        try:
            current_pnl = self.calculate_position_pnl()  # Now uses weighted average
            
            if current_pnl < -POSITION_LIMITS["stop_loss_pct"]:
                logging.warning(f"Stop loss hit at {current_pnl*100:.2f}% loss")
                logging.warning("Position Details:")
                logging.warning(f"  Size: {self.position_size}")
                logging.warning(f"  Weighted Entry: {self.position_tracker.get_weighted_entry_price()}")
                logging.warning(f"  Current Price: {self.ticker.mark_price}")
                return True
                
            return False
            
        except Exception as e:
            logging.error(f"Error checking stop loss: {str(e)}")
            return False

    async def handle_stop_loss(self):
        """Handle stop loss event with proper position closing"""
        try:
            if self.position_size == 0:
                return

            # Cancel all existing orders first
            for side in [0, 1]:
                for order in self.orders[side]:
                    if order.is_open():
                        await self.fast_cancel_order(order)

            # Calculate closing price with buffer for quick execution
            direction = th.Direction.SELL if self.position_size > 0 else th.Direction.BUY
            base_price = self.ticker.mark_price
            
            # More aggressive pricing for stop loss
            price_adjustment = 0.002  # 0.2% buffer for faster execution
            
            if direction == th.Direction.SELL:
                price = self.round_to_tick(base_price * (1 - price_adjustment))
            else:
                price = self.round_to_tick(base_price * (1 + price_adjustment))

            # Place the stop loss order
            await self.thalex.insert(
                direction=direction,
                instrument_name=self.perp_name,
                amount=abs(self.position_size),
                price=price,
                client_order_id=self.client_order_id,
                id=self.client_order_id
            )
            self.client_order_id += 1

            logging.warning(f"Stop loss order placed: {abs(self.position_size)} @ {price}")
            
            # Track the stop loss execution
            self.last_stop_loss = time.time()

        except Exception as e:
            logging.error(f"Error handling stop loss: {str(e)}")

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
        """Get current position size with improved reconciliation"""
        try:
            # First check if we have a position from portfolio
            if hasattr(self, 'portfolio') and isinstance(self.portfolio, list):
                for item in self.portfolio:
                    if isinstance(item, dict) and item.get('instrument_name') == self.perp_name:
                        return item.get('size', 0)
            
            # If no portfolio data or instrument not found, return tracked position
            return self.position_size
            
        except Exception as e:
            logging.error(f"Error getting position size: {str(e)}")
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
        """
        Align amount with the instrument's tick size
        
        Args:
            amount: The amount to align
            
        Returns:
            The aligned amount
        """
        return self.align_with_tick(amount)

    def validate_order_amount(self, amount: float) -> bool:
        """Validate if amount is properly aligned with tick size"""
        aligned = self.align_amount(amount)
        if abs(aligned - amount) > 1e-8:  # Use small epsilon for float comparison
            logging.warning(f"Amount {amount} not aligned with tick size, should be {aligned}")
            return False
        return True

    async def submit_order(self, direction: th.Direction, amount: float, price: float) -> bool:
        """Submit order with validated parameters"""
        self.logger.info(f"Submitting order to exchange - Direction: {direction}, Amount: {amount}, Price: {price}")
        try:
            amount, price = self.validate_order_params(amount, price)
            if not self.validate_order_amount(amount):
                self.logger.warning(f"Invalid order amount: {amount}")
                return False
                
            async with self.operation_semaphore:
                self.logger.info("Acquired operation semaphore, sending order to exchange")
                order_result = await self.thalex.insert(
                    direction=direction,
                    instrument_name=self.perp_name,
                    amount=amount,
                    price=price,
                    order_type=th.OrderType.LIMIT,
                    time_in_force=th.TimeInForce.GTC,
                    post_only=True,
                    label=LABEL
                )
                
                if order_result and 'id' in order_result:
                    self.logger.info(f"Order placed successfully with ID: {order_result['id']}")
                    return True
                else:
                    self.logger.error(f"Failed to place order, response: {order_result}")
                    return False
                    
        except Exception as e:
            self.logger.error(f"Error in submit_order: {str(e)}")
            return False

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
        self.logger.info(f"Checking if we should place new quote: {side.name} {quote.amount} @ {quote.price}")
        
        # Position limit check
        if side == th.Direction.BUY and self.position_size >= POSITION_LIMITS["max_position"]:
            self.logger.warning(f"Not placing {side.name} quote: Position size {self.position_size} >= max position {POSITION_LIMITS['max_position']}")
            return False
        if side == th.Direction.SELL and self.position_size <= -POSITION_LIMITS["max_position"]:
            self.logger.warning(f"Not placing {side.name} quote: Position size {self.position_size} <= -max position {-POSITION_LIMITS['max_position']}")
            return False
        
        # Check notional value
        if self.ticker and self.ticker.mark_price > 0:
            potential_notional = quote.amount * quote.price
            current_notional = abs(self.position_size * self.ticker.mark_price)
            total_notional = current_notional + potential_notional
            
            self.logger.debug(f"Notional check: current={current_notional:.2f}, potential={potential_notional:.2f}, total={total_notional:.2f}, limit={POSITION_LIMITS['max_notional']:.2f}")
            
            if total_notional > POSITION_LIMITS["max_notional"]:
                self.logger.warning(f"Not placing {side.name} quote: Total notional {total_notional:.2f} > max notional {POSITION_LIMITS['max_notional']:.2f}")
                return False
        else:
            self.logger.warning(f"Not placing {side.name} quote: No valid ticker data available")
            return False
        
        # Check for existing orders at similar price
        side_idx = 0 if side == th.Direction.BUY else 1
        threshold = QUOTING_CONFIG["amend_threshold"] * self.tick
        
        self.logger.debug(f"Checking for similar orders with threshold {threshold}")
        for order in self.orders[side_idx]:
            if order.is_open():
                price_diff = abs(order.price - quote.price)
                size_diff = abs(order.amount - quote.amount)
                
                self.logger.debug(f"Comparing with existing order: {order.amount} @ {order.price} (price diff: {price_diff}, size diff: {size_diff})")
                if price_diff < threshold and size_diff < 0.01:
                    self.logger.info(f"Not placing {side.name} quote: Similar order exists (price diff: {price_diff}, size diff: {size_diff})")
                    return False  # Already have a similar order
        
        self.logger.info(f"Quote placement check passed for {side.name} quote: {quote.amount} @ {quote.price}")
        return True

    async def cleanup_stale_orders(self):
        """Remove stale orders from tracking and cancel stale orders on the exchange"""
        try:
            current_time = time.time()
            stale_threshold = 60  # 1 minute - reduced from 5 minutes for more active management
            
            async with self.orders_lock:
                for side in [0, 1]:
                    stale_orders = []
                    for order in self.orders[side]:
                        # Check if order is stale but still open
                        if (order.is_open() and 
                            hasattr(order, 'timestamp') and 
                            current_time - order.timestamp > stale_threshold):
                            stale_orders.append(order)
                            logging.info(f"Found stale order to cancel: {order.id} at price {order.price}")
                    
                    # Cancel stale orders
                    for order in stale_orders:
                        await self.fast_cancel_order(order)
                    
                    # Keep only non-stale orders or open orders that weren't stale
                    self.orders[side] = [
                        order for order in self.orders[side]
                        if (
                            (order.is_open() and 
                             (not hasattr(order, 'timestamp') or 
                              current_time - order.timestamp <= stale_threshold)) or  # Keep non-stale open orders
                            (not order.is_open() and 
                             hasattr(order, 'timestamp') and 
                             current_time - order.timestamp < stale_threshold)  # Keep recent completed orders
                        )
                    ]
                    
            # Validate position after cleanup
            await self.validate_position_state()
            
        except Exception as e:
            logging.error(f"Error cleaning up stale orders: {str(e)}")
            if hasattr(e, "__traceback__"):
                logging.error(f"Traceback: {traceback.format_exc()}")

    async def cancel_all_orders(self):
        """Cancel all open orders - used periodically to prevent order accumulation"""
        try:
            logging.info("Performing periodic cancellation of all orders")
            cancel_count = 0
            
            async with self.orders_lock:
                for side in [0, 1]:
                    for order in self.orders[side]:
                        if order.is_open():
                            await self.fast_cancel_order(order)
                            cancel_count += 1
            
            if cancel_count > 0:
                logging.info(f"Cancelled {cancel_count} orders during periodic cleanup")
                
            # Wait for cancellations to process
            await asyncio.sleep(1.0)
            
        except Exception as e:
            logging.error(f"Error in cancel_all_orders: {str(e)}")
            if hasattr(e, "__traceback__"):
                logging.error(f"Traceback: {traceback.format_exc()}")
                
    async def cleanup_duplicate_orders(self):
        """Detect and cancel duplicate orders at the same price level"""
        try:
            logging.info("Checking for duplicate orders")
            
            async with self.orders_lock:
                for side in [0, 1]:
                    # Group orders by price
                    price_groups = {}
                    for order in self.orders[side]:
                        if order.is_open():
                            price = order.price
                            if price not in price_groups:
                                price_groups[price] = []
                            price_groups[price].append(order)
                    
                    # Cancel duplicate orders (keep the newest one)
                    for price, orders in price_groups.items():
                        if len(orders) > 1:
                            # Sort by timestamp (newest first)
                            sorted_orders = sorted(orders, key=lambda o: getattr(o, 'timestamp', 0), reverse=True)
                            
                            # Keep the newest order, cancel the rest
                            for order in sorted_orders[1:]:
                                logging.info(f"Cancelling duplicate order: {order.id} at price {order.price}")
                                await self.fast_cancel_order(order)
            
        except Exception as e:
            logging.error(f"Error in cleanup_duplicate_orders: {str(e)}")
            if hasattr(e, "__traceback__"):
                logging.error(f"Traceback: {traceback.format_exc()}")

    async def make_quotes(self) -> List[List[Quote]]:
        """Generate quotes using Avellaneda-Stoikov model"""
        self.logger.info("Starting make_quotes() execution")
        if not await self.check_risk_limits():
            logging.info("Risk limits check failed, not generating quotes")
            return [[], []]

        try:
            if not self.ticker or not self.tick:
                logging.warning("Missing ticker or tick size, not generating quotes")
                return [[], []]

            # Calculate current notional utilization
            current_notional = abs(self.position_size * self.ticker.mark_price) if self.position_size else 0
            notional_utilization = current_notional / POSITION_LIMITS["max_notional"] if current_notional > 0 else 0
            
            # Get optimal quotes from Avellaneda-Stoikov model
            logging.debug("Calculating optimal quotes...")
            bid_price, ask_price, bid_size, ask_size = self.calculate_optimal_quotes()
            
            if bid_price <= 0 or ask_price <= 0:
                logging.warning(f"Invalid quote prices: bid={bid_price}, ask={ask_price}")
                return [[], []]
                
            # Log successful quote generation
            logging.info(f"Generated quotes: bid={bid_price}, ask={ask_price}, bid_size={bid_size}, ask_size={ask_size}")
                
            # Adjust quote sizes based on notional utilization
            # Reduce quote sizes as we approach position limits
            size_adjustment_factor = 1.0
            if notional_utilization > 0.7:  # Start reducing at 70% utilization
                size_adjustment_factor = max(0.1, 1.0 - (notional_utilization - 0.7) * 3.0)  # Linear reduction
                logging.info(f"Reducing quote sizes due to high utilization ({notional_utilization:.2f}): factor={size_adjustment_factor:.2f}")
                bid_size *= size_adjustment_factor
                ask_size *= size_adjustment_factor
            
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
            logging.error(f"Error generating quotes: {str(e)}\nTrace: {traceback.format_exc()}")
            return [[], []]

    def calculate_optimal_quotes(self) -> Tuple[float, float, float, float]:
        """Calculate optimal quotes using Avellaneda-Stoikov model"""
        try:
            if not self.ticker or not self.price_history:
                logging.warning("Missing ticker or price history for optimal quotes calculation")
                return 0, 0, 0, 0

            # Check minimum price history requirement
            min_required_entries = AVELLANEDA_CONFIG["min_price_history"]
            if len(self.price_history) < min_required_entries:
                logging.warning(f"Insufficient price history: {len(self.price_history)} < {min_required_entries}")
                return 0, 0, 0, 0

            # Calculate mid price and volatility
            mid_price = self.ticker.mark_price
            if mid_price <= 0:
                logging.warning(f"Invalid mid price: {mid_price}")
                return 0, 0, 0, 0

            # Calculate volatility with floor
            volatility = max(self.calculate_volatility(), AVELLANEDA_CONFIG["volatility_floor"])
            logging.debug(f"Calculated volatility: {volatility:.6f}")

            # Calculate inventory ratio
            q = self.position_size / AVELLANEDA_CONFIG["position_limit"]
            
            # Calculate reservation price with inventory skew
            r = mid_price - q * AVELLANEDA_CONFIG["gamma"] * volatility**2 * AVELLANEDA_CONFIG["inventory_weight"]
            logging.debug(f"Reservation price: {r:.2f}")
            
            # Calculate optimal spread
            base_spread = (
                AVELLANEDA_CONFIG["gamma"] * volatility**2 * AVELLANEDA_CONFIG["T"] +
                2/AVELLANEDA_CONFIG["gamma"] * np.log(1 + AVELLANEDA_CONFIG["gamma"]/AVELLANEDA_CONFIG["k"])
            )
            
            # Add inventory component to spread
            inventory_spread = AVELLANEDA_CONFIG["phi"] * abs(q) * AVELLANEDA_CONFIG["T"]
            total_spread = base_spread + inventory_spread
            
            # Apply spread limits
            total_spread = min(
                max(total_spread, AVELLANEDA_CONFIG["min_spread"]),
                AVELLANEDA_CONFIG["max_spread"]
            )
            
            # Calculate bid and ask prices
            bid_price = self.round_to_tick(r - total_spread/2)
            ask_price = self.round_to_tick(r + total_spread/2)
            
            # Calculate optimal sizes
            bid_size = self.calculate_optimal_size("bid", q, volatility)
            ask_size = self.calculate_optimal_size("ask", q, volatility)
            
            # Log the calculations
            logging.debug(f"Optimal quotes: bid={bid_price}, ask={ask_price}, bid_size={bid_size}, ask_size={ask_size}")
            logging.debug(f"Market conditions: pos={self.position_size}, vol={volatility:.4f}, q={q:.4f}")
            
            return bid_price, ask_price, bid_size, ask_size
            
        except Exception as e:
            logging.error(f"Error calculating optimal quotes: {str(e)}")
            return 0, 0, 0, 0

    def calculate_volatility(self) -> float:
        """Calculate rolling volatility with improved estimation"""
        try:
            if len(self.price_history) < AVELLANEDA_CONFIG["min_price_history"]:
                logging.info(f"Insufficient price history for volatility: {len(self.price_history)} < {AVELLANEDA_CONFIG['min_price_history']}")
                return AVELLANEDA_CONFIG["volatility_floor"]
                
            # Use the most recent prices up to vol_window
            window_size = min(len(self.price_history), AVELLANEDA_CONFIG["vol_window"])
            prices = np.array(list(self.price_history)[-window_size:])
            
            # Validate prices
            if np.any(prices <= 0):
                logging.warning("Invalid prices in history")
                return AVELLANEDA_CONFIG["volatility_floor"]
                
            # Calculate log returns
            log_returns = np.diff(np.log(prices))
            
            # Calculate annualized volatility (assuming 1-minute data)
            # √(252 * 24 * 60) for annualization from minute data
            vol = np.std(log_returns) * np.sqrt(252 * 24 * 60)
            
            # Apply floor and ceiling
            vol = max(AVELLANEDA_CONFIG["volatility_floor"], vol)
            vol = min(vol, AVELLANEDA_CONFIG["max_spread"])  # Use max_spread as volatility ceiling
            
            logging.debug(f"Volatility calculation: std={np.std(log_returns):.6f}, annualized={vol:.6f}")
            return vol
            
        except Exception as e:
            logging.error(f"Error calculating volatility: {str(e)}")
            return AVELLANEDA_CONFIG["volatility_floor"]

    def calculate_optimal_size(self, side: str, q: float, volatility: float) -> float:
        """Calculate optimal order size based on inventory and volatility"""
        try:
            # Get base size from config
            base_size = AVELLANEDA_CONFIG["base_size"]
            
            # Inventory adjustment factor
            inventory_factor = np.exp(-AVELLANEDA_CONFIG["gamma"] * abs(q))
            
            # Volatility adjustment - reduce size in high volatility
            vol_factor = 1.0 / (1.0 + volatility)
            
            # Calculate size with inventory skew
            if side == "bid":
                # Reduce bid size when long, increase when short
                size = base_size * (2.0 - inventory_factor if q < 0 else inventory_factor) * vol_factor
            else:
                # Reduce ask size when short, increase when long
                size = base_size * (2.0 - inventory_factor if q > 0 else inventory_factor) * vol_factor
            
            # Apply size adjustment factor
            size *= AVELLANEDA_CONFIG["size_adj_factor"]
            
            # Apply notional limits
            size = self.apply_notional_limits(size)
            
            # Ensure minimum size and round to 0.001
            size = max(0.001, min(size, AVELLANEDA_CONFIG["q_max"]))
            size = round(size * 1000) / 1000
            
            logging.debug(f"Calculated {side} size: {size:.3f} (inv_factor={inventory_factor:.2f}, vol_factor={vol_factor:.2f})")
            return size
            
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
            if amount <= 0 or price <= 0:
                logging.warning(f"Invalid order parameters: amount={amount}, price={price}")
                return 0, 0

            # Align amount to 0.001 precision
            aligned_amount = self.align_amount(amount)
            if aligned_amount < 0.001:
                logging.warning(f"Amount {amount} too small after alignment")
                return 0, 0

            # Align price to tick size
            aligned_price = self.round_to_tick(price)
            if aligned_price <= 0:
                logging.warning(f"Invalid price after alignment: {aligned_price}")
                return 0, 0

            # Check notional value
            notional = aligned_amount * aligned_price
            max_notional = INVENTORY_CONFIG["max_position_notional"] * 0.2
            if notional > max_notional:
                aligned_amount = (max_notional / aligned_price)
                aligned_amount = self.align_amount(aligned_amount)
                logging.info(f"Adjusted amount to {aligned_amount} due to notional limit")

            return aligned_amount, aligned_price

        except Exception as e:
            logging.error(f"Error validating order parameters: {str(e)}")
            return 0, 0

   
    def calculate_available_notional(self) -> float:
        """Calculate remaining notional capacity"""
        try:
            if not self.ticker or self.ticker.mark_price <= 0:
                return 0
                
            current_notional = abs(self.position_size * self.ticker.mark_price)
            max_notional = INVENTORY_CONFIG["max_position_notional"]
            
            available = max(0, max_notional - current_notional)
            # Add buffer to prevent exceeding limits
            return available * AVELLANEDA_CONFIG["notional_utilization_threshold"]
            
        except Exception as e:
            logging.error(f"Error calculating available notional: {str(e)}")
            return 0

       
            

    

    def calculate_volume_profile(self) -> float:
        """Calculate volume profile signal"""
        try:
            if not hasattr(self, 'trade_history') or len(self.trade_history) < AVELLANEDA_CONFIG["volume_ma_period"]:
                return 0
                
            recent_trades = self.trade_history[-AVELLANEDA_CONFIG["volume_ma_period"]:]
            buy_volume = sum(t['amount'] for t in recent_trades if t['is_buy'])
            sell_volume = sum(t['amount'] for t in recent_trades if not t['is_buy'])
            
            if buy_volume + sell_volume == 0:
                return 0
                
            volume_imbalance = (buy_volume - sell_volume) / (buy_volume + sell_volume)
            return volume_imbalance
            
        except Exception as e:
            logging.error(f"Error calculating volume profile: {str(e)}")
            return 0


    # Add new risk management methods
    async def check_adverse_selection(self) -> bool:
        """
        Monitor for adverse selection by measuring loss speed
        Returns True if adverse selection is detected
        """
        try:
            if not hasattr(self, 'loss_measurements'):
                self.loss_measurements = deque(maxlen=RISK_MANAGEMENT_CONFIG["adverse_selection"]["consecutive_losses"])
                self.last_pnl_check = time.time()
                self.last_pnl = self.calculate_position_pnl()
                return False

            current_time = time.time()
            interval = RISK_MANAGEMENT_CONFIG["adverse_selection"]["measurement_interval"]
            
            if current_time - self.last_pnl_check < interval:
                return False

            current_pnl = self.calculate_position_pnl()
            pnl_change = current_pnl - self.last_pnl
            time_diff = current_time - self.last_pnl_check

            # Calculate loss speed (per minute)
            loss_speed = (pnl_change / time_diff) * 60
            
            # Record if it's a loss
            if loss_speed < -RISK_MANAGEMENT_CONFIG["adverse_selection"]["loss_speed_threshold"]:
                self.loss_measurements.append(1)
            else:
                self.loss_measurements.append(0)

            # Update last values
            self.last_pnl = current_pnl
            self.last_pnl_check = current_time

            # Check if we have consecutive losses
            if len(self.loss_measurements) >= RISK_MANAGEMENT_CONFIG["adverse_selection"]["consecutive_losses"]:
                if sum(self.loss_measurements) >= RISK_MANAGEMENT_CONFIG["adverse_selection"]["consecutive_losses"]:
                    logging.warning(f"Adverse selection detected! Loss speed: {loss_speed:.4f} per minute")
                    return True

            return False

        except Exception as e:
            logging.error(f"Error checking adverse selection: {str(e)}")
            return False

    async def manage_position_reduction(self):
        """
        Manage position reduction based on notional value and risk thresholds
        """
        try:
            if not self.ticker or self.position_size == 0:
                return

            current_notional = abs(self.position_size * self.ticker.mark_price)
            notional_utilization = current_notional / POSITION_LIMITS["max_notional"]

            # Check each reduction step
            for step in RISK_MANAGEMENT_CONFIG["position_reduction"]["reduction_steps"]:
                if notional_utilization >= step["threshold"]:
                    await self.execute_position_reduction(step["reduce_pct"])
                    break  # Only execute one reduction step at a time

        except Exception as e:
            logging.error(f"Error in position reduction management: {str(e)}")

    async def execute_position_reduction(self, reduction_percentage: float):
        """
        Execute position reduction with smart order placement
        """
        try:
            if not hasattr(self, 'last_reduction_time'):
                self.last_reduction_time = 0

            current_time = time.time()
            if current_time - self.last_reduction_time < RISK_MANAGEMENT_CONFIG["position_reduction"]["min_reduction_interval"]:
                return

            reduction_size = abs(self.position_size * reduction_percentage)
            if reduction_size < 0.001:  # Minimum order size
                return
            
            # Align the reduction size with the instrument's tick size
            reduction_size = self.align_with_tick(reduction_size)
            logging.info(f"Aligned reduction size to {reduction_size}")

            # Calculate aggressive price for faster execution
            # Calculate a slightly aggressive price for faster execution
            # But not too aggressive since we're already at a loss
            direction = th.Direction.SELL if self.position_size > 0 else th.Direction.BUY
            base_price = self.ticker.mark_price
            
            # Use a smaller price adjustment for unprofitable positions
            # to minimize further losses
            price_adjustment = 0.0005  # 0.05% price adjustment
            
            if direction == th.Direction.SELL:
                price = self.round_to_tick(base_price * (1 - price_adjustment))
            else:
                price = self.round_to_tick(base_price * (1 + price_adjustment))

            # Place the reduction order
            logging.info(f"Placing reduction order: direction={direction}, amount={reduction_size}, price={price}")
            await self.thalex.insert(
                direction=direction,
                instrument_name=self.perp_name,
                amount=reduction_size,
                price=price,
                client_order_id=self.client_order_id,
                id=self.client_order_id
            )
            self.client_order_id += 1
            
            # Store the last reduction time to prevent too frequent reductions
            if not hasattr(self, 'last_reduction_time'):
                self.last_reduction_time = 0
            self.last_reduction_time = time.time()

            logging.warning(f"Unprofitable excess position reduction: {reduction_size} @ {price} ({reduction_percentage*100}% reduction)")
            
            # Disable quoting temporarily to prevent building up more position
            self.quoting_enabled = False
            
            # Schedule re-enabling quoting after a delay
            async def reenable_quoting():
                await asyncio.sleep(60)  # 1 minute delay
                self.quoting_enabled = True
                logging.info("Re-enabling quoting after unprofitable position reduction")
                
            asyncio.create_task(reenable_quoting())

        except Exception as e:
            logging.error(f"Error handling unprofitable excess: {str(e)}")

    async def emergency_profitable_exit(self):
        """
        Execute emergency exit when position is profitable and exceeds limits
        """
        try:
            if not self.ticker or not self.position_size:
                return

            # Cancel all existing orders first
            for side in [0, 1]:
                for order in self.orders[side]:
                    if order.is_open():
                        await self.fast_cancel_order(order)

            # Calculate aggressive exit price (cross spread if needed)
            direction = th.Direction.SELL if self.position_size > 0 else th.Direction.BUY
            base_price = self.ticker.mark_price
            
            # More aggressive pricing to ensure execution
            price_adjustment = RISK_MANAGEMENT_CONFIG["emergency"]["panic_close_spread"]
            
            if direction == th.Direction.SELL:
                price = self.round_to_tick(base_price * (1 - price_adjustment))
            else:
                price = self.round_to_tick(base_price * (1 + price_adjustment))

            # Get the position size and align it with the tick size
            exit_size = abs(self.position_size)
            
            # Align the exit size with the instrument's tick size
            exit_size = self.align_with_tick(exit_size)
            logging.info(f"Aligned emergency exit size to {exit_size}")

            # Place the emergency exit order
            await self.thalex.insert(
                direction=direction,
                instrument_name=self.perp_name,
                amount=exit_size,
                price=price,
                client_order_id=self.client_order_id,
                id=self.client_order_id
            )
            self.client_order_id += 1

            logging.warning(f"Emergency profitable exit initiated: {exit_size} @ {price}")
            
            # Track the emergency exit execution
            self.last_emergency_close = time.time()

        except Exception as e:
            logging.error(f"Emergency profitable exit failed: {str(e)}")

    def validate_price(self, price, direction):
        """Validate if price is within acceptable collar range"""
        try:
            # Get current mark price
            mark_price = self.get_mark_price()
            if mark_price is None:
                self.logger.warning("Cannot validate price - mark price unavailable")
                return False
                
            # Define collar bounds (example: 10% from mark price)
            collar_pct = 0.10
            upper_bound = mark_price * (1 + collar_pct)
            lower_bound = mark_price * (1 - collar_pct)
            
            # Check if price is within bounds
            if price < lower_bound or price > upper_bound:
                self.logger.warning(f"Price {price} outside collar range [{lower_bound:.2f}, {upper_bound:.2f}]")
                return False
                
            return True
        except Exception as e:
            self.logger.error(f"Error validating price: {str(e)}")
            return False

    async def submit_order(self, direction, amount, price, reduce_only=False):
        """Submit order with price validation and improved error handling"""
        try:
            # Validate price before submission
            if not self.validate_price(price, direction):
                return None
                
            # Get next client order ID
            client_order_id = self.get_next_client_order_id()
            
            # Create order object for tracking
            order = self.order_from_data({
                'client_order_id': client_order_id,
                'direction': direction,
                'price': price,
                'amount': amount,
                'status': OrderStatus.PENDING,
                'timestamp': time.time()
            })
            
            # Add to tracking before submission
            await self.add_order_to_tracking(order)
            
            try:
                # Submit order
                response = await self.exchange.submit_order(
                    direction=direction,
                    amount=amount,
                    price=price,
                    reduce_only=reduce_only,
                    client_order_id=client_order_id
                )
                
                if response is None:
                    self.logger.error(f"Order submission failed - removing from tracking: {client_order_id}")
                    await self.remove_order_from_tracking(client_order_id)
                    return None
                    
                return response
                
            except Exception as e:
                self.logger.error(f"Error submitting order: {str(e)}")
                await self.remove_order_from_tracking(client_order_id)
                return None
                
        except Exception as e:
            self.logger.error(f"Error in submit_order: {str(e)}")
            return None

    def get_mark_price(self) -> Optional[float]:
        """Get current mark price with validation"""
        try:
            if not self.ticker:
                logging.warning("No ticker available for mark price")
                return None
                
            if self.ticker.mark_price <= 0:
                logging.warning(f"Invalid mark price: {self.ticker.mark_price}")
                return None
                
            return self.ticker.mark_price
        except Exception as e:
            logging.error(f"Error getting mark price: {str(e)}")
            return None

    async def get_next_client_order_id(self) -> int:
        """Get the next client order ID"""
        order_id = self.client_order_id
        self.client_order_id += 1
        if self.client_order_id > 1000000:  # Reset to avoid overflow
            self.client_order_id = 100
        return order_id

    async def add_order_to_tracking(self, order: Order):
        """Add an order to tracking"""
        if not order.direction:
            logging.warning(f"Order {order.id} has no direction, cannot add to tracking")
            return False
        
        side_idx = 0 if order.direction == th.Direction.BUY else 1
        self.orders[side_idx].append(order)
        logging.info(f"Added order to tracking: {order.id}, side: {order.direction.name}, price: {order.price}, amount: {order.amount}")
        return True

    async def remove_order_from_tracking(self, order_id: int):
        """Remove order from tracking with proper synchronization"""
        async with self.order_tracking_lock:
            if order_id in self.order_tracking:
                del self.order_tracking[order_id]
                # Remove from side-specific tracking
                for side in [0, 1]:
                    self.orders[side] = [o for o in self.orders[side] if o.id != order_id]

    def calculate_loss_speed(self) -> float:
        """Calcula velocidad de pérdidas en % por minuto"""
        now = time.time()
        time_window = RISK_MANAGEMENT_CONFIG["adverse_selection"]["measurement_interval"]
        recent_losses = [p for t,p in self.pnl_history if (now - t) < time_window and p < 0]
        return abs(sum(recent_losses)) / (time_window / 60)  # % por minuto

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
    """Main entry point for the market maker"""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()]
    )
    
    # Initialize Thalex client
    thalex = th.Thalex(network=NETWORK)
    
    # Create and run the quoter
    quoter = PerpQuoter(thalex)
    
    # Create tasks
    tasks = [
        asyncio.create_task(quoter.listen_task()),
        asyncio.create_task(quoter.quote_task()),
        asyncio.create_task(quoter.log_pnl())
    ]
    
    # Add debug tasks if in debug mode
    if DEBUG_MODE:
        tasks.append(asyncio.create_task(monitor_health(quoter)))
        tasks.append(asyncio.create_task(dump_state(quoter)))
    
    async def shutdown(tasks):
        """Gracefully shutdown all tasks"""
        for task in tasks:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
    
    # Main loop with reconnection logic
    while True:
        try:
            logging.info(f"Starting on {NETWORK} UNDERLYING='{UNDERLYING}'")
            await asyncio.gather(*tasks)
        except (websockets.ConnectionClosed, socket.gaierror) as e:
            logging.error(f"Connection error ({e}). Reconnecting...")
            await shutdown(tasks)
            await asyncio.sleep(1)
        except KeyboardInterrupt:
            logging.info("Shutting down...")
            await shutdown(tasks)
            break
        except Exception as e:
            logging.exception("Unexpected error:")
            await shutdown(tasks)
            await asyncio.sleep(1)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("Shutting down gracefully...")
    except Exception as e:
        logging.critical(f"Fatal error in main loop: {str(e)}")
        traceback.print_exc()

# Add to configuration section
MAX_HOLDING_TIME = 3600  # 1 hour in seconds


async def monitor_health(quoter: PerpQuoter):
    """Monitor bot health metrics"""
    logger = logging.getLogger("HealthMonitor")
    
    while True:
        try:
            # Check quote updates
            time_since_quote = time.time() - quoter.last_quote_update
            if time_since_quote > 60:  # Check every minute
                logger.warning(f"No quote updates for {time_since_quote:.1f} seconds")
            
            # Check position tracking
            portfolio_position = quoter.portfolio.get(quoter.perp_name, 0)
            if abs(quoter.position_size - portfolio_position) > 0.001:
                logger.warning(f"Position mismatch: internal={quoter.position_size}, portfolio={portfolio_position}")
            
            # Check price history
            if len(quoter.price_history) < AVELLANEDA_CONFIG["min_price_history"]:
                logger.warning(f"Insufficient price history: {len(quoter.price_history)}/{AVELLANEDA_CONFIG['min_price_history']}")
            
            # Log current state
            mark_price = quoter.ticker.mark_price if quoter.ticker else 0
            active_orders = len(quoter.orders[0]) + len(quoter.orders[1]) if quoter.orders else 0
            
            logger.info(
                f"Health check:\n"
                f"  Position: {quoter.position_size:.3f}\n"
                f"  Entry price: {quoter.entry_price:.2f}\n"
                f"  Mark price: {mark_price:.2f}\n"
                f"  Active orders: {active_orders}\n"
                f"  Price history: {len(quoter.price_history)} points\n"
                f"  Quote updates: {quoter.quote_update_count}\n"
                f"  Error count: {quoter.error_count}"
            )
            
            await asyncio.sleep(60)  # Check every minute
            
        except Exception as e:
            logger.error(f"Error in health monitor: {str(e)}")
            await asyncio.sleep(5)

async def dump_state(quoter: PerpQuoter):
    """Periodically dump full state for debugging"""
    logger = logging.getLogger("StateMonitor")
    
    while True:
        try:
            state = {
                "timestamp": datetime.now().isoformat(),
                "position": {
                    "size": quoter.position_size,
                    "entry_price": quoter.entry_price,
                    "unrealized_pnl": quoter.calculate_unrealized_pnl(),
                    "realized_pnl": quoter.realized_pnl
                },
                "market": {
                    "mark_price": quoter.ticker.mark_price if quoter.ticker else None,
                    "volatility": quoter.calculate_volatility()
                },
                "orders": {
                    "bids": len(quoter.orders[0]),
                    "asks": len(quoter.orders[1])
                },
                "metrics": {
                    "quote_updates": quoter.quote_update_count,
                    "error_count": quoter.error_count
                }
            }
            
            # Log state
            logger.info(f"State dump: {json.dumps(state)}")
            
            await asyncio.sleep(300)  # Dump every 5 minutes
            
        except Exception as e:
            logger.error(f"Error dumping state: {str(e)}")
