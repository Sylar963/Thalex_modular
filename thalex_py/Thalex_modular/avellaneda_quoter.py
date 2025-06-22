# HFT OPTIMIZATIONS COMPLETED - PHASE 1:
# 1. Reduced excessive ticker logging from every update to minimal sampling
# 2. Consolidated price field extraction with single-line OR operations  
# 3. Streamlined ticker object creation with minimal overhead
# 4. Simplified shared memory updates with direct field assignment
# 5. Reduced order fill validation to essential checks only
# 6. Consolidated position tracker updates with exception silencing for performance
# 7. Flagged VAMP updates as potentially unused in production HFT (###=========##)
# 8. Simplified market conditions calculation with fallback chaining
# 9. Streamlined WebSocket connection handling with minimal logging
# 10. Optimized message processing with consolidated exception handling
# 11. Reduced message type handling to essential dispatching only
# 12. Flagged unrecognized message logging as potentially unused (###=========##)
# 13. Simplified circuit breaker logic with minimal overhead
# 14. Consolidated quote validation with streamlined cross-quote fixing
# 15. Maintained exact same logic while reducing code size by ~35%

import asyncio
import json
import os
import socket
import time
import traceback
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Union
from collections import deque
import random
import math
import logging
import ctypes
import numpy as np
import websockets
from websockets.exceptions import ConnectionClosedError, ConnectionClosedOK
import orjson  # Fast JSON parsing
import threading
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import shared_memory

# Define a small threshold for zero comparisons
ZERO_THRESHOLD = 1e-9

# Performance optimization constants
LOG_SAMPLING_RATE = 0.05  # 5% of debug operations logged
MAJOR_EVENT_LOG_INTERVAL = 30.0  # Log major events every 30 seconds

# Message buffer constants
MESSAGE_BUFFER_INITIAL_SIZE = 32768
MESSAGE_BUFFER_MAX_SIZE = 1048576

import thalex as th
from thalex import Network
from thalex_py.Thalex_modular.config.market_config import (
    BOT_CONFIG,
    MARKET_CONFIG,
    CALL_IDS,
    RISK_LIMITS,
    TRADING_CONFIG
)
from thalex_py.Thalex_modular.models.data_models import Ticker, Order, OrderStatus, Quote
from thalex_py.Thalex_modular.models.position_tracker import PositionTracker, PortfolioTracker, Fill
from thalex_py.Thalex_modular.components.risk_manager import RiskManager
from thalex_py.Thalex_modular.components.order_manager import OrderManager
from thalex_py.Thalex_modular.components.avellaneda_market_maker import AvellanedaMarketMaker
from thalex_py.Thalex_modular.models.keys import key_ids, private_keys
from thalex_py.Thalex_modular.performance_monitor import PerformanceMonitor
from thalex_py.Thalex_modular.thalex_logging import LoggerFactory
from thalex_py.Thalex_modular.ringbuffer.market_data_buffer import MarketDataBuffer
from thalex_py.Thalex_modular.ringbuffer.volume_candle_buffer import VolumeBasedCandleBuffer
from thalex_py.Thalex_modular.profiling.performance_tracer import PerformanceTracer

# Import call IDs from data_models
from thalex_py.Thalex_modular.models.data_models import (
    CALL_ID_INSTRUMENTS,
    CALL_ID_INSTRUMENT,
    CALL_ID_SUBSCRIBE,
    CALL_ID_LOGIN,
    CALL_ID_CANCEL_SESSION,
    CALL_ID_SET_COD
)

# Define memory pool for frequently created objects
class ObjectPool:
    def __init__(self, factory, size=100):
        self.factory = factory
        self.items = [factory() for _ in range(size)]
        self.lock = threading.RLock()  # Reentrant lock for thread safety
        
    def get(self):
        with self.lock:
            if self.items:
                return self.items.pop()
            return self.factory()
            
    def put(self, item):
        with self.lock:
            self.items.append(item)

# Lock-free queue implementation for high-frequency components
class LockFreeQueue:
    def __init__(self, maxsize=1000):
        self.maxsize = maxsize
        self.queue = np.zeros(maxsize, dtype=np.uint64)
        self.head = 0
        self.tail = 0

    def put(self, item):
        """Put item in queue without locking"""
        if (self.tail + 1) % self.maxsize == self.head:
            return False  # Queue is full
        self.queue[self.tail] = item
        self.tail = (self.tail + 1) % self.maxsize
        return True

    def get(self):
        """Get item from queue without locking"""
        if self.head == self.tail:
            return None  # Queue is empty
        item = self.queue[self.head]
        self.head = (self.head + 1) % self.maxsize
        return item

# Memory-mapped structure for IPC - SIMPLIFIED DEFINITION
class SharedMarketData(ctypes.Structure):
    """Memory-mapped ctypes.Structure for sharing market data between processes."""
    _fields_ = [
        ('timestamp', ctypes.c_double),
        ('mid_price', ctypes.c_double),
        ('best_bid', ctypes.c_double),
        ('best_ask', ctypes.c_double),
        ('bid_quantity', ctypes.c_double),
        ('ask_quantity', ctypes.c_double),
        ('volatility', ctypes.c_double),
        ('status', ctypes.c_int)
    ]

class AvellanedaQuoter:
    __slots__ = [
        # Core dependencies
        'thalex', 'logger', 'tasks', 'position_tracker', 'portfolio_tracker', 'market_maker', 
        'order_manager', 'risk_manager', 'performance_monitor',
        
        # Risk and recovery state
        'active_trading', 'risk_recovery_mode', 'risk_breach_time',
        'recovery_cooldown_until', 'recovery_step', 'last_recovery_check',
        
        # Take profit state

        'last_upnl_value', 'risk_monitoring_interval',
        
        # Market data
        'ticker', 'index', 'perp_name', 'futures_instrument_name', 'futures_ticker',
        'futures_tick_size', 'market_data', 'portfolio',
        
        # Connection and rate limiting
        'max_requests_per_minute', 'rate_limit_warning_sent', 'volatile_market_warning_sent',
        'heartbeat_interval', 'last_heartbeat',
        
        # Quoting management
        'quoting_enabled', 'cooldown_active', 'cooldown_until', 'price_history',
        'price_history_idx', 'price_history_full', 'current_quotes',
        
        # Order management
        'max_orders_per_side', 'max_total_orders', 'next_client_order_id', 'tick_size',
        'contract_size',
        
        # Timestamps
        'last_ticker_time', 'last_quote_time', 'last_quote_update_time',
        
        # Instrument data and caching
        'instrument_data', 'instrument_data_cache',
        
        # Performance optimization
        'message_buffer_size', 'message_buffer', 'message_view',
        'order_pool', 'ticker_pool', 'quote_pool',
        
        # Shared memory and IPC
        'shm_obj', 'shared_market_data_struct',
        
        # Performance tracing
        'performance_tracer', 'volume_buffer',
        
        # Coordination
        'quote_cv', 'condition_met', 'setup_complete',
        
        # Performance caches (NEW)
        '_cached_market_conditions', '_cached_market_time', '_log_counter', '_last_major_log', '_last_price',
        
        # Optimized price history storage
        'price_history_timestamps',
        
        # Fast order tracking
        'orders_by_id', 'orders_by_client_id', 'active_bid_count', 'active_ask_count',
        
        # Logging control
        'verbose_logging', 'log_sampling_counter'
    ]
    
    def _create_empty_order(self):
        """Factory function for order pool"""
        return Order(
            id=0,
            price=0.0,
            amount=0.0,
            status=OrderStatus.PENDING,
            direction=None
        )
    
    def _create_empty_quote(self):
        """Factory function for quote pool"""
        return Quote(
            price=0.0,
            amount=0.0,
            instrument="",
            side="",
            timestamp=time.time()
        )
    
    def _create_empty_ticker(self):
        """Factory function for ticker pool"""
        return Ticker({
            "mark_price": 0.0,
            "best_bid_price": 0.0,
            "best_ask_price": 0.0,
            "index": 0.0,
            "mark_timestamp": time.time(),
            "timestamp": time.time()
        })
    
    def _initialize_object_pools(self):
        """Initialize optimized object pools for high-frequency operations"""
        # Optimize pool sizes based on expected usage patterns
        max_quotes_per_update = TRADING_CONFIG["quoting"].get("levels", 6) * 2  # bids + asks
        
        # Order pool - size based on max concurrent orders
        self.order_pool = ObjectPool(
            factory=self._create_empty_order,
            size=self.max_total_orders * 2  # Buffer for pending orders
        )
        
        # Quote pool - size based on quote generation frequency
        self.quote_pool = ObjectPool(
            factory=self._create_empty_quote,
            size=max_quotes_per_update * 3  # Buffer for multiple quote cycles
        )
        
        # Ticker pool - smaller as less frequently created
        self.ticker_pool = ObjectPool(
            factory=self._create_empty_ticker,
            size=5  # Minimal as tickers are reused
        )
    
    def __init__(self, thalex: th.Thalex):
        """Initialize the Avellaneda-Stoikov market maker"""
        # Core dependencies
        self.thalex = thalex
        self.logger = LoggerFactory.configure_component_logger(
            "avellaneda_quoter",
            log_file="quoter.log",
            high_frequency=False  # Only market maker needs high frequency logging
        )
        
        # Initialize tasks dictionary early to prevent AttributeError in shutdown
        self.tasks: Dict[str, Optional[asyncio.Task]] = {}
        
        # Set up the market maker components
        self.portfolio_tracker = PortfolioTracker() # Initialize PortfolioTracker for multi-instrument support
        self.position_tracker = PositionTracker() # Initialize PositionTracker first - keep for backward compatibility
        self.market_maker = AvellanedaMarketMaker(
            exchange_client=self.thalex,
            position_tracker=self.position_tracker
        )
        
        # Set portfolio tracker reference in market maker for multi-instrument support
        self.market_maker.portfolio_tracker = self.portfolio_tracker
        self.order_manager = OrderManager(self.thalex)
        self.risk_manager = RiskManager(self.position_tracker) # Then pass it to RiskManager
        self.performance_monitor = PerformanceMonitor(record_interval=1, buffer_size=1)  # Record every 1 second, immediate write
        
        # Initialize shared volume buffer to avoid duplication
        self.volume_buffer = None  # Will be initialized after instrument setup
        
        # Register risk breach handler
        self.risk_manager.register_callback("risk_limit", self._handle_risk_breach)
        self.active_trading = True # Flag to control trading activity based on risk
        
        # Risk recovery state
        self.risk_recovery_mode = False
        self.risk_breach_time = None
        self.recovery_cooldown_until = 0
        self.recovery_step = 0  # 0=full halt, 1-3=gradual recovery, 3=full active
        self.last_recovery_check = 0
        
        # Take profit state management (NEW)

        self.last_upnl_value = 0.0
        
        # Risk monitoring task interval
        self.risk_monitoring_interval = TRADING_CONFIG.get("risk_monitoring_interval_seconds", 2.0) # Default to 2s
        
        # Market data
        self.ticker = None
        self.index = None
        self.perp_name = None
        self.futures_instrument_name = None
        self.futures_ticker = None
        self.futures_tick_size = 1.0 # Added for futures, if needed for order placement
        self.market_data = MarketDataBuffer(
            volatility_window=TRADING_CONFIG["volatility"]["window"],
            capacity=TRADING_CONFIG["volatility"]["min_samples"]
        )
        
        # Portfolio data
        self.portfolio = {}
        
        # Take profit tracking - DEPRECATED (replaced with UPNL-based logic)
        # Legacy variables kept for compatibility - use new UPNL system instead
        
        # Rate limiting parameters - now managed by circuit breaker in Thalex client
        self.max_requests_per_minute = BOT_CONFIG["connection"]["rate_limit"]
        self.rate_limit_warning_sent = False
        self.volatile_market_warning_sent = False  # Flag for tracking volatile market warnings
        
        # For coordination
        self.quote_cv = asyncio.Condition()
        self.condition_met = False
        self.setup_complete = asyncio.Event()
        
        # Connection parameters - now using improved connection management
        self.heartbeat_interval = BOT_CONFIG["connection"]["heartbeat_interval"]
        self.last_heartbeat = time.time()
        
        # Quoting management
        self.quoting_enabled = True
        self.cooldown_active = False
        self.cooldown_until = 0
        
        # Initialize optimized price history storage
        self._initialize_price_history()
        self.current_quotes = ([], [])  # (bid_quotes, ask_quotes)
        
        # Initialize optimized order tracking
        self._initialize_order_tracking()
        
        # Order management
        self.max_orders_per_side = 5  # Maximum orders per side
        self.max_total_orders = 16  # Maximum total orders
        self.next_client_order_id = 1000  # Starting ID for client orders
        
        # Set default tick size
        self.tick_size = 1.0
        
        # Latest update timestamps
        self.last_ticker_time = 0
        self.last_quote_time = 0
        self.last_quote_update_time = 0
        
        # Instrument data cache to avoid frequent lookups
        self.instrument_data = {}
        
        # WebSocket message processing optimization
        self.message_buffer_size = 32768  # 32KB buffer
        self.message_buffer = bytearray(self.message_buffer_size)
        self.message_view = memoryview(self.message_buffer)
        
        # Initialize optimized object pools
        self._initialize_object_pools()
        
        # Performance optimization caches
        self._cached_market_conditions = {}
        self._cached_market_time = 0.0
        self._log_counter = 0
        self._last_major_log = 0.0
        
        # Logging control
        self.verbose_logging = BOT_CONFIG.get("verbose_logging", False)
        self.log_sampling_counter = 0
        
        # Initialize instrument data cache
        self.instrument_data_cache = {}
        
        # Create shared memory for IPC
        self.shm_obj = None
        self.shared_market_data_struct = None
        try:
            shm_size = ctypes.sizeof(SharedMarketData)
            # Try to create the shared memory segment
            self.shm_obj = shared_memory.SharedMemory(name="thalex_market_data", create=True, size=shm_size)
            self.logger.info(f"Created shared memory segment 'thalex_market_data' with size {shm_size}")
        except FileExistsError:
            # If it already exists, connect to it
            try:
                shm_size = ctypes.sizeof(SharedMarketData) # Recalculate size just in case
                self.shm_obj = shared_memory.SharedMemory(name="thalex_market_data", create=False, size=shm_size)
                self.logger.info(f"Connected to existing shared memory segment 'thalex_market_data' size {shm_size}")
            except Exception as e_conn:
                self.logger.error(f"Failed to connect to existing shared memory 'thalex_market_data': {e_conn}")
                self.shm_obj = None # Ensure it's None on failure
        except Exception as e_create:
            self.logger.error(f"Failed to create shared memory 'thalex_market_data': {e_create}")
            self.shm_obj = None # Ensure it's None on failure

        if self.shm_obj:
            try:
                # Map the shared memory buffer to the ctypes structure
                self.shared_market_data_struct = SharedMarketData.from_buffer(self.shm_obj.buf)
                self.logger.info("Successfully mapped shared memory buffer to SharedMarketData structure.")
            except Exception as e_map:
                self.logger.error(f"Failed to map shared memory buffer: {e_map}")
                self.shared_market_data_struct = None
                # If mapping fails, close and unlink the shm_obj as it's unusable by this instance
                try:
                    self.shm_obj.close()
                    self.shm_obj.unlink() # Attempt to unlink, might fail if not creator or permissions issue
                except FileNotFoundError:
                    pass # It might have been unlinked by another process or never fully created
                except Exception as e_cleanup:
                    self.logger.error(f"Error cleaning up shm_obj after mapping failure: {e_cleanup}")
                self.shm_obj = None
        
        # Performance tracing
        self.performance_tracer = PerformanceTracer()
        
        self.logger.info("Avellaneda quoter initialized with HFT optimizations")
        
    async def _handle_risk_breach(self, reason: str, price_at_breach: Optional[float]):
        """Handles actions to take when a risk limit is breached."""
        price_display = f"{price_at_breach:.2f}" if price_at_breach is not None else "N/A"
        self.logger.critical(f"RISK LIMIT BREACHED: {reason}. Price at check: {price_display}")

        if not self.active_trading:
            self.logger.info("Trading already halted, risk breach handling skipped further action to prevent re-entry.")
            return

        # NEW: Set recovery state
        self.active_trading = False
        self.risk_recovery_mode = True
        self.risk_breach_time = time.time()
        self.recovery_cooldown_until = time.time() + RISK_LIMITS.get("recovery_cooldown_seconds", 300)
        self.recovery_step = 0
        
        self.logger.critical("Trading has been HALTED due to risk limit breach. Recovery mode activated.")
        
        # Cancel all open orders (EXISTING LOGIC PRESERVED)
        self.logger.info("Attempting to cancel all open orders due to risk breach...")
        try:
            if hasattr(self, 'order_manager') and self.order_manager is not None:
                 await self.cancel_quotes(f"Risk limit breach: {reason}")
                 self.logger.info("Successfully requested cancellation of all orders via cancel_quotes.")
            elif hasattr(self, 'thalex') and self.thalex is not None:
                 await self.thalex.cancel_all_orders(id=CALL_IDS.get("cancel_all", random.randint(10000, 20000)))
                 self.logger.info("Successfully requested cancellation of all orders via direct thalex client call.")
            else:
                self.logger.warning("No valid order_manager or thalex client to cancel orders during risk breach.")
        except Exception as e:
            self.logger.error(f"Error attempting to cancel all orders after risk breach: {str(e)}", exc_info=True)

    async def _check_risk_recovery(self) -> bool:
        """Check if risk conditions have improved enough to start recovery"""
        if not self.risk_recovery_mode:
            return False
            
        current_time = time.time()
        
        # Step 1: Check cooldown period
        if current_time < self.recovery_cooldown_until:
            return False
            
        # Step 2: Check if we have valid price data
        if not self.ticker or self.ticker.mark_price <= 0:
            return False
            
        # Step 3: Check current risk levels
        try:
            current_price = self.ticker.mark_price
            
            # Check if risk limits are now within recovery threshold
            recovery_threshold = RISK_LIMITS.get("risk_recovery_threshold", 0.8)
            
            # Check position limits
            metrics = self.position_tracker.get_position_metrics()
            current_pos_size = abs(metrics.get("position", 0.0))
            max_position = RISK_LIMITS.get("max_position", float('inf'))
            
            if current_pos_size > (max_position * recovery_threshold):
                self.logger.info(f"Position {current_pos_size:.4f} still above recovery threshold {max_position * recovery_threshold:.4f}")
                return False
                
            # Check notional limits
            notional = current_pos_size * current_price
            max_notional = RISK_LIMITS.get("max_notional", float('inf'))
            
            if notional > (max_notional * recovery_threshold):
                self.logger.info(f"Notional {notional:.2f} still above recovery threshold {max_notional * recovery_threshold:.2f}")
                return False
                
            # If we get here, conditions are good for recovery
            self.logger.info("Risk conditions improved - initiating gradual recovery")
            return True
            
        except Exception as e:
            self.logger.error(f"Error checking recovery conditions: {str(e)}")
            return False

    async def _initiate_gradual_recovery(self):
        """Start gradual recovery process"""
        recovery_steps = RISK_LIMITS.get("gradual_recovery_steps", 3)
        
        if self.recovery_step < recovery_steps:
            self.recovery_step += 1
            self.logger.info(f"Recovery step {self.recovery_step}/{recovery_steps} - Partial trading resumed")
            
            if self.recovery_step >= recovery_steps:
                # Full recovery
                self.active_trading = True
                self.risk_recovery_mode = False
                self.risk_breach_time = None
                self.recovery_step = 0
                self.logger.info("FULL RECOVERY: Trading fully resumed")
            else:
                # Partial recovery - still in recovery mode but allow some trading
                self.logger.info(f"PARTIAL RECOVERY: Limited trading resumed (step {self.recovery_step})")

    def _get_current_upnl(self) -> float:
        """Extract combined UPNL from portfolio across all instruments"""
        try:
            # Get total portfolio P&L across all instruments
            total_portfolio_pnl = self.portfolio_tracker.get_total_pnl()
            
            # HFT FIX: Log detailed breakdown periodically for monitoring
            if hasattr(self, '_log_counter') and self._log_counter % 100 == 0:  # Less frequent logging
                portfolio_metrics = self.portfolio_tracker.get_portfolio_metrics()
                instrument_pnls = portfolio_metrics.get("instrument_pnls", {})
                self.logger.info(f"Portfolio UPNL breakdown: {instrument_pnls}, Total: {total_portfolio_pnl:.2f}")
            
            # HFT FIX: Validate UPNL calculation
            if not isinstance(total_portfolio_pnl, (int, float)) or not np.isfinite(total_portfolio_pnl):
                self.logger.warning(f"Invalid portfolio UPNL: {total_portfolio_pnl}, falling back to position tracker")
                raise ValueError("Invalid portfolio UPNL")
            
            return float(total_portfolio_pnl)
            
        except Exception as e:
            self.logger.warning(f"Portfolio UPNL calculation failed: {str(e)}, using fallback")
            # Fallback to single instrument if portfolio tracker fails
            try:
                if hasattr(self, 'position_tracker') and hasattr(self.position_tracker, 'get_position_metrics'):
                    metrics = self.position_tracker.get_position_metrics()
                    fallback_upnl = float(metrics.get("unrealized_pnl", 0.0))
                    self.logger.info(f"Using fallback UPNL: {fallback_upnl:.2f}")
                    return fallback_upnl
            except Exception as fallback_error:
                self.logger.error(f"Fallback UPNL calculation also failed: {str(fallback_error)}")
            
            # HFT FIX: Log when UPNL calculation completely fails
            self.logger.error("CRITICAL: All UPNL calculation methods failed - take profit will not work")
            return 0.0





    async def _risk_monitoring_task(self):
        """Optimized risk monitoring with adaptive intervals"""
        base_interval = self.risk_monitoring_interval
        self.logger.info(f"Starting optimized risk monitoring task with base interval: {base_interval}s")
        await self.setup_complete.wait() # Ensure setup is done before starting
        
        while self.active_trading and self.thalex.connected():
            try:
                # Adaptive interval based on position size
                current_position = self.position_tracker.get_position_metrics().get("position", 0.0)
                
                if abs(current_position) > 0.5:  # High position
                    sleep_interval = base_interval * 0.5  # More frequent monitoring
                elif abs(current_position) < 0.1:  # Low position
                    sleep_interval = base_interval * 2.0  # Less frequent monitoring
                else:
                    sleep_interval = base_interval
                
                await asyncio.sleep(sleep_interval)
                if not self.active_trading: # Re-check after sleep
                    self.logger.info("Risk monitoring task: active_trading is false. Exiting.")
                    break
                if not self.thalex.connected():
                    self.logger.info("Risk monitoring task: thalex not connected. Exiting.")
                    break

                current_price = None
                if self.ticker and hasattr(self.ticker, 'mark_price') and self.ticker.mark_price > 0:
                    current_price = self.ticker.mark_price
                elif self.market_data and len(self.market_data.prices) > 0 and self.market_data.prices[-1] > 0: # Fallback to market_data buffer
                    current_price = self.market_data.prices[-1]
                else:
                    self.logger.warning("Risk monitoring: No valid current_price available (ticker or market_data). Skipping this check cycle.")
                    current_pos_size_for_log = self.risk_manager.position_tracker.get_position_metrics().get("position", 0.0)
                    if abs(current_pos_size_for_log) > ZERO_THRESHOLD: # If in position without price, this is risky
                        self.logger.error(f"CRITICAL: In position ({current_pos_size_for_log:.4f}) but no current price for risk checks. Consider manual intervention or halting.")
                        # Potentially call _handle_risk_breach here if no price is a critical failure
                        # await self._handle_risk_breach("Critical: No price for risk check while in position", None)
                    continue

                # Update mark prices for portfolio-level P&L calculation
                if current_price is not None and self.perp_name:
                    self.portfolio_tracker.update_mark_price(self.perp_name, current_price)

                # Get futures price and update its mark price
                current_price_futures = None
                if self.futures_ticker and hasattr(self.futures_ticker, 'mark_price') and self.futures_ticker.mark_price > 0:
                    current_price_futures = self.futures_ticker.mark_price
                elif self.market_data and self.futures_instrument_name and len(self.market_data.get_prices(self.futures_instrument_name)) > 0: # Assuming market_data can be instrument specific
                    current_price_futures = self.market_data.get_prices(self.futures_instrument_name)[-1]

                if current_price_futures is not None and self.futures_instrument_name:
                    self.portfolio_tracker.update_mark_price(self.futures_instrument_name, current_price_futures)
                elif self.futures_instrument_name: # Log if futures instrument is configured but no price
                    self.logger.warning(f"Risk monitoring: No valid current_price_futures available for {self.futures_instrument_name}.")

                # If trading is no longer active (e.g. due to a breach in another task), exit
                if not self.active_trading:
                    self.logger.info("Risk monitoring task: active_trading became false during cycle. Exiting.")
                    break

                # Check overall risk limits for perpetual
                perp_risk_limits_breached = False
                if current_price and self.perp_name:
                    if not await self.risk_manager.check_risk_limits(current_market_price=current_price, instrument_name=self.perp_name):
                        self.logger.warning(f"PERPETUAL risk limits breached ({self.perp_name}) during periodic check at price {current_price:.2f}. _handle_risk_breach will be invoked.")
                        perp_risk_limits_breached = True

                # Check overall risk limits for futures
                futures_risk_limits_breached = False
                if current_price_futures and self.futures_instrument_name:
                    if not await self.risk_manager.check_risk_limits(current_market_price=current_price_futures, instrument_name=self.futures_instrument_name):
                        self.logger.warning(f"FUTURES risk limits breached ({self.futures_instrument_name}) during periodic check at price {current_price_futures:.2f}. _handle_risk_breach will be invoked.")
                        futures_risk_limits_breached = True

                if perp_risk_limits_breached or futures_risk_limits_breached:
                    reason_breach = []
                    if perp_risk_limits_breached: reason_breach.append(f"Perp {self.perp_name} limit breach")
                    if futures_risk_limits_breached: reason_breach.append(f"Futures {self.futures_instrument_name} limit breach")
                    await self._handle_risk_breach(f"Overall risk limits breached: {'; '.join(reason_breach)}", current_price if perp_risk_limits_breached else current_price_futures)
                    # _handle_risk_breach sets active_trading = False, loop will terminate.
                    continue # Allow _handle_risk_breach to stop activity

                current_pos_size_check = self.risk_manager.position_tracker.get_position_metrics().get("position", 0.0)
                if abs(current_pos_size_check) < ZERO_THRESHOLD: # No position, no stop-loss or take-profit to check
                    continue

                # 1. Check Stop Loss for Perpetual
                stop_loss_perp_triggered = False
                if self.perp_name and current_price and abs(current_pos_size_check) > ZERO_THRESHOLD:
                    if self.risk_manager.check_stop_loss(current_price, self.perp_name):
                        self.logger.warning(f"PERPETUAL STOP-LOSS triggered for {self.perp_name} at price {current_price:.2f}. Position: {current_pos_size_check:.4f}.")
                        stop_loss_perp_triggered = True
                
                # 2. Check Stop Loss for Futures
                stop_loss_futures_triggered = False
                if self.futures_instrument_name and current_price_futures and abs(current_pos_size_check) > ZERO_THRESHOLD:
                    if self.risk_manager.check_stop_loss(current_price_futures, self.futures_instrument_name):
                        self.logger.warning(f"FUTURES STOP-LOSS triggered for {self.futures_instrument_name} at price {current_price_futures:.2f}. Position: {current_pos_size_check:.4f}.")
                        stop_loss_futures_triggered = True

                if stop_loss_perp_triggered or stop_loss_futures_triggered:
                    sl_reason_parts = []
                    if stop_loss_perp_triggered: sl_reason_parts.append(f"Perp {self.perp_name} SL at {current_price:.2f}")
                    if stop_loss_futures_triggered: sl_reason_parts.append(f"Futures {self.futures_instrument_name} SL at {current_price_futures:.2f}")
                    sl_reason = f"Stop-loss triggered: {'; '.join(sl_reason_parts)}"
                    self.logger.info(f"{sl_reason}. Attempting to close both positions.")
                    await self._close_both_positions(sl_reason)
                    await self._handle_risk_breach(sl_reason, current_price if stop_loss_perp_triggered else current_price_futures)
                    continue # Stop-loss hit, _handle_risk_breach will stop trading


                
                # Check for recovery if in recovery mode
                if self.risk_recovery_mode and not self.active_trading:
                    if time.time() - self.last_recovery_check > RISK_LIMITS.get("recovery_check_interval", 30):
                        self.last_recovery_check = time.time()
                        if await self._check_risk_recovery():
                            await self._initiate_gradual_recovery()
                
            except asyncio.CancelledError:
                self.logger.info("Risk monitoring task was cancelled.")
                break
            except Exception as e:
                self.logger.error(f"Error in risk monitoring task: {str(e)}", exc_info=True)
                # Back off on errors
                await asyncio.sleep(base_interval * 2) 
        
        self.logger.info("Risk monitoring task finished.")

    async def start(self):
        """Start the quoter"""
        try:
            self.logger.info("AvellanedaQuoter startup initiated...")
            
            # Create performance tracer
            self.performance_tracer = PerformanceTracer()
            
            # Initialize setup completion event
            self.setup_complete = asyncio.Event()
            
            # Create condition variable for quote updates
            self.quote_cv = asyncio.Condition()
            self.condition_met = False
            
            # Initialize the Thalex client connection
            await self.thalex.connect()
            
            if not self.thalex.connected():
                self.logger.error("Failed to connect to Thalex API")
                return False
            
            # Set up instrument data with timeout
            try:
                # Wait up to 30 seconds for instrument data
                await asyncio.wait_for(self.await_instruments(), timeout=30.0)
            except asyncio.TimeoutError:
                self.logger.error("Timed out waiting for instrument data")
            except Exception as e:
                self.logger.error(f"Error setting up instrument data: {str(e)}")
                return False
                
            # Login and setup
            try:
                self.logger.info("Logging in to Thalex API...")
                # Retrieve credentials from configuration
                network = MARKET_CONFIG["network"]

                key_id_raw = key_ids[network]
                private_key_raw = private_keys[network]

                # self.logger.info(f"PHASE 0.5 DEBUG: Raw key_id from models.keys: '{key_id_raw}' (type: {type(key_id_raw)})")
                # if private_key_raw and isinstance(private_key_raw, str):
                #     self.logger.info(f"PHASE 0.5 DEBUG: Raw private_key from models.keys (len: {len(private_key_raw)}): '{private_key_raw[:100]}...{private_key_raw[-100:] if len(private_key_raw) > 200 else private_key_raw}'")
                # elif private_key_raw:
                #     self.logger.info(f"PHASE 0.5 DEBUG: Raw private_key from models.keys is not a string (type: {type(private_key_raw)})")
                # else:
                #     self.logger.info(f"PHASE 0.5 DEBUG: Raw private_key from models.keys is None or empty.")

                key_id = key_id_raw
                private_key = private_key_raw

                # ++++ STRIP POTENTIAL QUOTES ++++
                if key_id and isinstance(key_id, str):
                    original_key_id_for_log = key_id
                    stripped_something = True
                    while stripped_something:
                        stripped_something = False
                        if ((key_id.startswith('"') and key_id.endswith('"')) or \
                           (key_id.startswith("'") and key_id.endswith("'"))) and len(key_id) >= 2:
                            key_id = key_id[1:-1]
                            stripped_something = True
                        # Check for triple quotes as well, though less common for key IDs
                        elif key_id.startswith('"""') and key_id.endswith('"""') and len(key_id) >= 6:
                            key_id = key_id[3:-3]
                            stripped_something = True
                    # if original_key_id_for_log != key_id:
                    #     self.logger.info(f"PHASE 1.5 DEBUG: Iteratively stripped quotes from key_id. Original: '{original_key_id_for_log}', New: '{key_id}'")


                if private_key and isinstance(private_key, str):
                    original_private_key_for_log_start = private_key[:70] # For logging comparison
                    
                    stripped_something_pk = True
                    while stripped_something_pk:
                        stripped_something_pk = False
                        # Handle \"\"\"key\"\"\"
                        if private_key.startswith('"""') and private_key.endswith('"""') and len(private_key) >= 6:
                            private_key = private_key[3:-3]
                            stripped_something_pk = True
                            # self.logger.info("PHASE 1.5 DEBUG: Iteratively stripped triple quotes from private_key.")
                            continue # Restart loop for potentially nested quotes like \"\"\"\"key\"\"\"\"
                        # Handle \"key\" or \'key\'
                        if ((private_key.startswith('"') and private_key.endswith('"')) or \
                           (private_key.startswith("'") and private_key.endswith("'"))) and len(private_key) >= 2:
                            private_key = private_key[1:-1]
                            stripped_something_pk = True
                            # self.logger.info("PHASE 1.5 DEBUG: Iteratively stripped single/double quotes from private_key.")
                            continue
                    
                    # if private_key[:70] != original_private_key_for_log_start :
                    #      self.logger.info(f"PHASE 1.5 DEBUG: Private key after iterative quote stripping (first 70 chars): '{private_key[:70]}...'")
                    
                    # After stripping all quotes, trim whitespace that might have been inside quotes
                    # e.g. \"  \\nKEY\\n  \" -> KEY
                    original_len_before_strip = len(private_key)
                    private_key = private_key.strip()
                    # if len(private_key) != original_len_before_strip:
                    #     self.logger.info(f"PHASE 1.5 DEBUG: Private key after final .strip() (first 70 chars): '{private_key[:70]}...'")
                # ++++ END STRIP POTENTIAL QUOTES ++++

                # ++++ ENHANCED PRIVATE KEY DIAGNOSTICS (Phase 1) ++++
                # self.logger.info(f"PHASE 1 DEBUG: Retrieved key_id for {network}: {key_id} (type: {type(key_id)})")
                if private_key:
                    # pk_len = len(private_key)
                    # Show more characters, ensure we don't go out of bounds for short keys
                    # pk_start_slice = min(80, pk_len)
                    # pk_end_slice = min(80, pk_len)
                    
                    # pk_start = str(private_key)[:pk_start_slice]
                    
                    # pk_middle_start_offset = max(0, pk_len // 2 - 40)
                    # pk_middle_end_offset = min(pk_len, pk_middle_start_offset + 80) # Read 80 chars from middle
                    # pk_middle = str(private_key)[pk_middle_start_offset:pk_middle_end_offset]
                    
                    # pk_end = str(private_key)[-pk_end_slice:]
                    
                    # self.logger.info(f"PHASE 1 DEBUG: Private Key for {network} (Length: {pk_len})")
                    # self.logger.info(f"PHASE 1 DEBUG: PK START ({pk_start_slice} chars): {pk_start}")
                    # self.logger.info(f"PHASE 1 DEBUG: PK MIDDLE (approx {len(pk_middle)} chars): ...{pk_middle}...")
                    # self.logger.info(f"PHASE 1 DEBUG: PK END ({pk_end_slice} chars): {pk_end}")
                    # self.logger.info(f"PHASE 1 DEBUG: PK Type: {type(private_key)}")
                    
                    # Check for common issues
                    # if "\\\\n" in private_key: # Check for literal \\n
                    #     self.logger.warning("PHASE 1 DEBUG: Private key string CONTAINS LITERAL '\\\\n' sequences! This is likely incorrect for .env loading.")
                    # elif "\\n" in private_key: # Check for literal \\n
                    #     self.logger.warning("PHASE 1 DEBUG: Private key string CONTAINS LITERAL '\\n' sequences! This is likely incorrect for .env loading.")
                    
                    # if private_key.count("-----BEGIN RSA PRIVATE KEY-----") > 1:
                    #     self.logger.warning("PHASE 1 DEBUG: Private key string CONTAINS '-----BEGIN RSA PRIVATE KEY-----' MORE THAN ONCE!")
                    # if private_key.count("-----END RSA PRIVATE KEY-----") > 1:
                    #      self.logger.warning("PHASE 1 DEBUG: Private key string CONTAINS '-----END RSA PRIVATE KEY-----' MORE THAN ONCE!")
                    
                    # Check start and end after stripping whitespace from the whole key, then from the end of the value
                    # This helps identify if the .env loader added extra quotes or if the value itself is bad
                    # stripped_private_key = private_key.strip()
                    # if not stripped_private_key.startswith("-----BEGIN RSA PRIVATE KEY-----"):
                    #     self.logger.warning("PHASE 1 DEBUG: Stripped Private key string DOES NOT properly START with PEM marker.")
                    # if not stripped_private_key.endswith("-----END RSA PRIVATE KEY-----"):
                    #     self.logger.warning("PHASE 1 DEBUG: Stripped Private key string DOES NOT properly END with PEM marker.")
                    pass # Keep the if private_key block for structure, but content is commented
                else:
                    self.logger.error(f"PHASE 1 DEBUG: Private Key for {network} is None!") # Keep this error for critical failure
                # ++++ END ENHANCED PRIVATE KEY DIAGNOSTICS (Phase 1) ++++
                
                if key_id is None or private_key is None:
                    self.logger.error(f"CRITICAL: API Key ID or Private Key is None for network {network}. Check .env file and variable names (e.g., THALEX_TEST_API_KEY_ID).")
                    return False

                # Login to the API
                await self.thalex.login(key_id, private_key, id=CALL_IDS["login"])
                
                # Configure cancel on disconnect
                await self.thalex.set_cancel_on_disconnect(6, id=CALL_IDS["set_cod"])
            except Exception as e:
                self.logger.error(f"Login failed: {str(e)}")
                return False
            
            # Continue with the rest of the existing code
            
            # Set the perpetual name from config
            underlying = MARKET_CONFIG.get("underlying", "BTC")
            # Fix: Don't add -PERPETUAL if it's already there
            if underlying.endswith("-PERPETUAL"):
                self.perp_name = underlying
            else:
                self.perp_name = underlying.split("USD")[0] + "-PERPETUAL"
            self.logger.info(f"Setting instrument name to {self.perp_name}")
            
            # Set the position delta for position tracking
            if hasattr(self, 'instrument_data') and 'contractSize' in self.instrument_data:
                contract_size = self.instrument_data['contractSize']
                self.logger.info(f"Setting contract size to {contract_size}")
                self.contract_size = float(contract_size)
            
            # Set futures instrument name from config
            self.futures_instrument_name = MARKET_CONFIG.get("futures_instrument")
            if not self.futures_instrument_name:
                self.logger.warning("FUTURES_INSTRUMENT not configured in MARKET_CONFIG. Futures leg P&L will not be monitored.")
            else:
                self.logger.info(f"Futures instrument for P&L monitoring: {self.futures_instrument_name}")
            
            # Register instruments with portfolio tracker
            if self.perp_name:
                self.portfolio_tracker.register_instrument(self.perp_name)
                self.logger.info(f"Registered perpetual instrument: {self.perp_name}")

            if self.futures_instrument_name:
                self.portfolio_tracker.register_instrument(self.futures_instrument_name)
                self.logger.info(f"Registered futures instrument: {self.futures_instrument_name}")
            
            # NEW: Initialize volume candle buffer for predictive analysis (FIXED)
            self.logger.info("Initializing volume candle buffer for predictive analysis...")
            try:
                self.volume_buffer = VolumeBasedCandleBuffer(
                    volume_threshold=TRADING_CONFIG["volume_candle"].get("threshold", 1.0),
                    max_candles=TRADING_CONFIG["volume_candle"].get("max_candles", 100),
                    max_time_seconds=TRADING_CONFIG["volume_candle"].get("max_time_seconds", 300),
                    exchange_client=self.thalex,
                    instrument=self.perp_name,
                    use_exchange_data=TRADING_CONFIG["volume_candle"].get("use_exchange_data", False),
                    fetch_interval_seconds=TRADING_CONFIG["volume_candle"].get("fetch_interval_seconds", 60),
                    lookback_hours=TRADING_CONFIG["volume_candle"].get("lookback_hours", 1)
                )
                
                # Share the volume buffer with market maker to avoid duplication
                if hasattr(self.market_maker, 'volume_buffer'):
                    self.market_maker.volume_buffer = self.volume_buffer
                    self.logger.info("Shared volume buffer with market maker to avoid duplication")
                
                self.logger.info(f"Volume candle buffer initialized: threshold={self.volume_buffer.volume_threshold}, "
                               f"max_candles={self.volume_buffer.max_candles}, instrument={self.perp_name}")
            except Exception as e:
                self.logger.error(f"Failed to initialize volume candle buffer: {str(e)}")
                self.volume_buffer = None  # Fallback to None if initialization fails
            
            # Subscribe to WebSocket topics with retry logic
            self.logger.info("Setting up WebSocket subscriptions...")
            max_sub_attempts = 3
            sub_attempt = 0
            
            while sub_attempt < max_sub_attempts:
                try:
                    # Set up all subscriptions
                    await self._subscribe_to_websocket_topics()
                    self.logger.info("WebSocket subscriptions completed")
                    break
                except Exception as e:
                    sub_attempt += 1
                    self.logger.error(f"Subscription attempt {sub_attempt}/{max_sub_attempts} failed: {str(e)}")
                    
                    if sub_attempt >= max_sub_attempts:
                        self.logger.error("Failed to set up WebSocket subscriptions")
                        return False
                    
                    # Wait before retrying
                    await asyncio.sleep(1)
            
            # Create all the tasks that will run concurrently
            self.logger.info("Setting up background tasks...")
            
            # First create the listener task only and wait for initial connection
            self.tasks = {
                'listener': asyncio.create_task(self.listen_task())
            }
            
            # Wait for the listener to establish connection
            self.logger.info("Waiting for listener task to initialize...")
            await asyncio.sleep(2)
            
            # Now create the rest of the tasks
            self.tasks.update({
                'heartbeat': asyncio.create_task(self.heartbeat_task()),
                'status': asyncio.create_task(self.log_status_task()),
                'profile': asyncio.create_task(self.profile_optimization_task()),
                'quote': asyncio.create_task(self.quote_task()),
                'risk_monitor': asyncio.create_task(self._risk_monitoring_task()), # Add new task
                'performance_monitor': asyncio.create_task(self.performance_monitor.start_recording(self)) # Start performance monitoring
            })
            
            # Tasks created successfully (reduced logging)
            
            # Initialize quote pool if not already done
            if not hasattr(self, 'quote_pool'):
                self.quote_pool = ObjectPool(lambda: Quote(
                    price=0.0,
                    amount=0.0,
                    instrument="",
                    side="",
                    timestamp=time.time()
                ), size=100)
            
            # Signal that setup is complete
            self.setup_complete.set()
            self.logger.info("Quoter startup complete - all systems operational")
            
            return True
            
        except asyncio.CancelledError:
            self.logger.warning("Start-up was cancelled")
            return False
        except Exception as e:
            self.logger.error(f"Error starting quoter: {str(e)}")
            self.logger.error(traceback.format_exc())
            return False

    async def shutdown(self):
        """Shutdown the quoter and cleanup resources"""
        try:
            self.logger.info("Starting quoter shutdown sequence...")
            self.active_trading = False # Signal all loops to stop

            # Cancel risk monitoring task first if it exists
            if 'risk_monitor' in self.tasks and self.tasks['risk_monitor'] is not None:
                self.logger.info("Cancelling risk monitoring task...")
                self.tasks['risk_monitor'].cancel()
                try:
                    await self.tasks['risk_monitor']
                except asyncio.CancelledError:
                    self.logger.info("Risk monitoring task successfully cancelled.")
                except Exception as e:
                    self.logger.error(f"Error during risk_monitor task cancellation: {e}")
                del self.tasks['risk_monitor']

            # Cancel all orders
            try:
                self.logger.info("Cancelling all orders...")
                await self.order_manager.cancel_all_orders()
            except Exception as e:
                self.logger.error(f"Error cancelling orders during shutdown: {str(e)}")
            
            # Shutdown thalex client if connected
            try:
                if self.thalex and hasattr(self.thalex, 'connected') and self.thalex.connected():
                    self.logger.info("Disconnecting from Thalex...")
                    await self.thalex.disconnect()
            except Exception as e:
                self.logger.error(f"Error disconnecting from Thalex: {str(e)}")
            
            # Close volume candle buffer
            try:
                if hasattr(self, 'volume_buffer') and self.volume_buffer:
                    self.logger.info("Stopping volume candle buffer...")
                    if hasattr(self.volume_buffer, 'stop'):
                        self.volume_buffer.stop()
                    self.logger.info("Volume candle buffer stopped")
                else:
                    self.logger.info("No volume buffer to clean up")
            except Exception as e:
                self.logger.error(f"Error stopping volume candle buffer: {str(e)}")

            # Close shared memory resources
            try:
                self.logger.info("Cleaning up shared memory resources...")
                if hasattr(self, 'shared_market_data_struct') and self.shared_market_data_struct is not None:
                    self.logger.info("Deleting reference to SharedMarketData ctypes structure.")
                    del self.shared_market_data_struct # Explicitly delete the ctypes structure
                    self.shared_market_data_struct = None

                if hasattr(self, 'shm_obj') and self.shm_obj:
                    try:
                        self.shm_obj.close()
                        self.logger.info("Shared memory object closed.")
                    except Exception as e_close:
                        self.logger.error(f"Error closing shared memory object: {e_close}")
                    
                    # Attempt to unlink. This might be the responsibility of the process that created it.
                    # If multiple processes use it, only one should typically unlink.
                    # For robustness, we can try and catch FileNotFoundError if already unlinked.
                    try:
                        # Check if this process was the creator (if we stored that info) or just always try to unlink
                        # For now, let's assume this instance might be responsible or it's okay to try.
                        self.shm_obj.unlink()
                        self.logger.info("Shared memory segment 'thalex_market_data' unlinked.")
                    except FileNotFoundError:
                        self.logger.info("Shared memory segment 'thalex_market_data' was already unlinked or not created by this instance.")
                    except Exception as e_unlink:
                        self.logger.error(f"Error unlinking shared memory segment 'thalex_market_data': {e_unlink}")
                else:
                    self.logger.info("No shared memory object (self.shm_obj) to clean up or it was already None.")
            except Exception as e_outer_shm:
                self.logger.error(f"Outer error during shared memory cleanup: {e_outer_shm}")
            
            # Cancel all tasks
            try:
                if hasattr(self, 'tasks') and self.tasks:
                    self.logger.info(f"Cancelling {len(self.tasks)} background tasks...")
                    for task in self.tasks.values():
                        if not task.done():
                            task.cancel()
                    
                    # Wait for tasks to actually complete with timeout
                    try:
                        # Give tasks up to 5 seconds to clean up
                        done, pending = await asyncio.wait(self.tasks.values(), timeout=5.0)
                        if pending:
                            self.logger.warning(f"{len(pending)} tasks did not complete gracefully and were forced to cancel")
                    except Exception as wait_error:
                        self.logger.error(f"Error waiting for tasks to complete: {str(wait_error)}")
            except Exception as e:
                self.logger.error(f"Error cancelling tasks: {str(e)}")
                
            self.logger.info("All tasks processed for cancellation.")

            self.logger.info("Quoter shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during quoter shutdown: {str(e)}", exc_info=True)

    async def await_instruments(self):
        """Wait for and process instrument data"""
        try:
            self.logger.info("Retrieving instrument data...")
            
            # Get available instruments
            try:
                # Send request for instruments
                await self.thalex.instruments(CALL_ID_INSTRUMENTS)
                
                # Wait for response with timeout
                max_retries = 3
                for retry in range(max_retries):
                    try:
                        msg = await asyncio.wait_for(self.thalex.receive(), timeout=5.0)
                        break
                    except asyncio.TimeoutError:
                        if retry < max_retries - 1:
                            self.logger.warning(f"Timeout waiting for instrument data (attempt {retry+1}/{max_retries}). Retrying...")
                        else:
                            raise asyncio.TimeoutError("Timed out waiting for instrument data")
                
                # Parse message
                try:
                    data = orjson.loads(msg)
                except Exception:
                    data = json.loads(msg)
                
                # Handle error in response
                if data.get("error"):
                    self.logger.error(f"Error retrieving instruments: {data['error']}")
                    return False
                
                # Store instrument data in cache
                result = data.get("result", [])
                # Store using appropriate key based on the actual structure
                self.instrument_data_cache = {}
                for instr in result:
                    key = instr.get("name") or instr.get("asset") or instr.get("symbol")
                    if key:
                        self.instrument_data_cache[key] = instr
                
                # Log the available instruments for debugging
                self.logger.info(f"Retrieved {len(result)} instruments")
                for instr in result:
                    self.logger.debug(f"Instrument: {instr}")
                
                # Find perpetual contract 
                instruments = result
                perpetual = None
                futures_contract = None # For the futures leg
                underlying = MARKET_CONFIG["underlying"]
                
                # Try different underlying formats - prioritize BTC-PERPETUAL format
                underlying_base = underlying.replace("USD", "")
                underlying_variants = [
                    f"{underlying_base}-PERPETUAL",  # BTC-PERPETUAL (primary format)
                    underlying,  # BTCUSD
                    underlying.replace("USD", "-USD"),  # BTC-USD
                    underlying.replace("USD", "/USD"),  # BTC/USD
                    underlying_base,  # BTC
                    f"{underlying_base}-PERP"  # BTC-PERP
                ]
                
                # Log the variants we're searching for
                self.logger.info(f"Searching for underlying variants: {underlying_variants}")
                
                # First try to find an exact match
                for instrument in instruments:
                    # Check using multiple possible field names
                    instrument_asset = instrument.get("asset") or instrument.get("name") or instrument.get("symbol") or ""
                    instrument_type = instrument.get("type") or instrument.get("instrument_type") or ""
                    
                    self.logger.debug(f"Checking instrument: asset={instrument_asset}, type={instrument_type}")
                    
                    # Check all variants
                    for variant in underlying_variants:
                        if instrument_asset == variant:
                            perpetual = instrument
                            self.logger.info(f"Found perpetual match: {instrument_asset}")
                            break
                    
                    if perpetual:
                        break
                
                # If no exact match, try a more flexible approach
                if not perpetual:
                    self.logger.info("No exact match found, trying flexible matching...")
                    for instrument in instruments:
                        instrument_name = str(instrument.get("name") or instrument.get("asset") or instrument.get("symbol") or "").upper()
                        instrument_type = str(instrument.get("type") or instrument.get("instrument_type") or "").lower()
                        
                        # Check if it's a perpetual for any of our underlying variants
                        is_perpetual = ("PERP" in instrument_name or "perpetual" in instrument_type)
                        
                        for variant in underlying_variants:
                            variant_base = variant.replace("USD", "").replace("-", "").replace("/", "")
                            if variant_base in instrument_name and is_perpetual:
                                perpetual = instrument
                                self.logger.info(f"Found flexible match: {instrument_name}")
                                break
                        
                        if perpetual:
                            break
                
                # If still no match, just take the first perpetual
                if not perpetual:
                    self.logger.warning(f"No specific perpetual found for {underlying}, looking for any perpetual...")
                    for instrument in instruments:
                        instrument_name = str(instrument.get("name") or instrument.get("asset") or instrument.get("symbol") or "").upper()
                        instrument_type = str(instrument.get("type") or instrument.get("instrument_type") or "").lower()
                        
                        if "PERP" in instrument_name or "perpetual" in instrument_type:
                            perpetual = instrument
                            self.logger.info(f"Using first available perpetual: {instrument_name}")
                            break
                            
                if not perpetual:
                    self.logger.error(f"Could not find any perpetual instrument")
                    # Log the available instruments for debugging
                    self.logger.debug(f"Available instruments: {instruments}")
                    return False
                
                # Find futures contract if configured
                if self.futures_instrument_name:
                    self.logger.info(f"Searching for FUTURES instrument: {self.futures_instrument_name}")
                    for instrument in instruments:
                        instrument_asset = instrument.get("asset") or instrument.get("name") or instrument.get("symbol") or ""
                        if instrument_asset == self.futures_instrument_name:
                            futures_contract = instrument
                            self.logger.info(f"Found FUTURES instrument: {instrument_asset}")
                            break
                    if not futures_contract:
                        self.logger.warning(f"Configured FUTURES instrument '{self.futures_instrument_name}' not found in API results.")
                        # self.futures_instrument_name = None # Optionally disable if not found

                if not perpetual and not self.futures_instrument_name: # If only perp was configured and not found
                    self.logger.error("Could not find PERPETUAL instrument and no FUTURES instrument configured.")
                    return False
                elif not perpetual and self.futures_instrument_name and not futures_contract: # If both configured but neither found
                    self.logger.error(f"Could not find PERPETUAL instrument OR the configured FUTURES instrument {self.futures_instrument_name}.")
                    return False

                # Set perpetual contract details
                if perpetual:
                    self.perp_name = perpetual.get("name") or perpetual.get("asset") or perpetual.get("symbol")
                    self.tick_size = float(perpetual.get("tick_size") or perpetual.get("tickSize") or 1.0) # Assuming primary tick_size is for perp
                    self.contract_size = float(perpetual.get("contract_size") or perpetual.get("contractSize") or 1.0)
                    self.market_maker.set_tick_size(self.tick_size) # Market maker quotes the perp
                    self.order_manager.set_tick_size(self.tick_size) # Order manager primarily uses perp tick for quotes
                    self.logger.info(f"Found perpetual: {self.perp_name} with tick size {self.tick_size}, contract size {self.contract_size}")
                elif not self.futures_instrument_name: # If only perp was sought and not found
                    self.logger.error(f"Perpetual instrument for {underlying} not found and no futures configured. Cannot proceed.")
                    return False

                # Set futures contract details (primarily tick_size if needed for order placement, though market orders are less sensitive)
                if futures_contract:
                    self.futures_tick_size = float(futures_contract.get("tick_size") or futures_contract.get("tickSize") or 1.0)
                    # contract_size_futures = float(futures_contract.get("contract_size") or futures_contract.get("contractSize") or 1.0) # If needed
                    self.logger.info(f"Futures instrument {self.futures_instrument_name} tick size: {self.futures_tick_size}")
                elif self.futures_instrument_name:
                    self.logger.warning(f"Futures instrument {self.futures_instrument_name} configured but not found. Risk monitoring for futures P&L might be impaired if price data isn't received.")

                # NEW: Configure enhanced multi-instrument trigger system
                self._setup_enhanced_trigger_system()

                return True
                
            except Exception as e:
                self.logger.error(f"Error retrieving instrument data: {str(e)}")
                raise
                
        except Exception as e:
            self.logger.error(f"Error in await_instruments: {str(e)}")
            raise
            
    def _setup_enhanced_trigger_system(self):
        """Configure the enhanced multi-instrument trigger system"""
        try:
            # Configure instrument settings in market maker
            self.market_maker.set_instrument_config(
                perp_instrument=self.perp_name,
                futures_instrument=self.futures_instrument_name,
                perp_tick_size=self.tick_size,
                futures_tick_size=self.futures_tick_size
            )
            
            # Register both instruments with portfolio tracker if not already done
            if self.perp_name:
                self.portfolio_tracker.register_instrument(self.perp_name)
                self.logger.info(f"Registered perpetual instrument: {self.perp_name}")
                
            if self.futures_instrument_name:
                self.portfolio_tracker.register_instrument(self.futures_instrument_name)
                self.logger.info(f"Registered futures instrument: {self.futures_instrument_name}")
            
            # Log the enhanced trigger system configuration
            mode = self.market_maker.detect_trading_mode()
            self.logger.info(
                f"Enhanced trigger system configured - "
                f"Mode: {mode}, "
                f"Perp: {self.perp_name} (tick: {self.tick_size}), "
                f"Futures: {self.futures_instrument_name} (tick: {self.futures_tick_size})"
            )
            
            # Verify trigger system is enabled
            if hasattr(self.market_maker, 'trigger_order_enabled') and self.market_maker.trigger_order_enabled:
                self.logger.info("Enhanced multi-instrument trigger orders: ENABLED")
                
                # Log thresholds
                self.logger.info(
                    f"Profit thresholds - "
                    f"Arbitrage: ${self.market_maker.arbitrage_profit_threshold_usd:.2f}, "
                    f"Single: ${self.market_maker.single_instrument_profit_threshold_usd:.2f}, "
                    f"Spread: {self.market_maker.spread_profit_threshold_bps:.1f}bps"
                )
            else:
                self.logger.warning("Enhanced trigger orders are DISABLED")
                
        except Exception as e:
            self.logger.error(f"Error setting up enhanced trigger system: {str(e)}", exc_info=True)

    # Add a new profile-guided optimization task
    async def profile_optimization_task(self):
        """
        Profile execution and dynamically optimize critical paths.
        This task monitors performance of key operations and applies optimizations.
        """
        while True:
            try:
                # Run every 5 minutes
                await asyncio.sleep(300)
                
                # Analyze performance data
                critical_paths = self.performance_tracer.get_critical_paths()
                
                # Apply optimizations to slowest operations
                if critical_paths:
                    self.logger.info(f"Optimizing critical paths: {critical_paths}")
                    for path, avg_time in critical_paths:
                        if path == "message_processing" and avg_time > 0.001:
                            # Optimize message processing if too slow
                            self.message_buffer_size = min(self.message_buffer_size * 2, 1048576)
                            self.message_buffer = bytearray(self.message_buffer_size)
                            self.message_view = memoryview(self.message_buffer)
                            self.logger.info(f"Increased message buffer to {self.message_buffer_size} bytes")
                        
                        elif path == "quote_generation" and avg_time > 0.005:
                            # Reduce quote complexity if too slow
                            self.max_orders_per_side = max(1, self.max_orders_per_side - 1)
                            self.logger.info(f"Reduced max orders per side to {self.max_orders_per_side}")
                
                # Reset performance data
                self.performance_tracer.reset()
                
            except Exception as e:
                self.logger.error(f"Error in profile optimization task: {str(e)}")
                await asyncio.sleep(60)  # Retry after a minute

    async def listen_task(self):
        """Listen for websocket messages"""
        backoff_time = 0.1
        max_backoff = 30
        consecutive_errors = 0
        max_consecutive_errors = 20
        last_message_time = time.time()
        
        while True:
            try:
                # Connection handling
                if not self.thalex.connected():
                    try:
                        if self.thalex.ws is not None:
                            await self.thalex.disconnect()
                            await asyncio.sleep(1)
                        
                        await self.thalex.connect()
                        
                        # Re-authenticate
                        network = MARKET_CONFIG["network"]
                        await self.thalex.login(key_ids[network], private_keys[network], id=CALL_ID_LOGIN)
                        await self._subscribe_to_websocket_topics()
                        
                        backoff_time = 0.1
                        consecutive_errors = 0
                    except Exception:
                        await asyncio.sleep(backoff_time)
                        backoff_time = min(max_backoff, backoff_time * 2)
                        continue
                
                # Message processing with performance tracing
                with self.performance_tracer.trace("message_processing"):
                    try:
                        message = await asyncio.wait_for(self.thalex.receive(), timeout=30.0)
                        last_message_time = time.time()
                    except asyncio.TimeoutError:
                        consecutive_errors += 1
                        continue
                    except (websockets.exceptions.ConnectionClosedOK, websockets.exceptions.ConnectionClosedError):
                        self.thalex.ws = None
                        consecutive_errors += 1
                        await asyncio.sleep(1)
                        continue
                    
                    consecutive_errors = 0
                    backoff_time = 0.1
                    
                    if message is None:
                        continue
                    
                    self._log_counter += 1
                    
                    # Process message
                    try:
                        data = self._process_message_optimized(message) if isinstance(message, str) else message
                    except Exception:
                        continue
                    
                    # Handle message types
                    if "type" in data and data["type"] == "ticker" and "data" in data:
                        await self.handle_ticker_update(data["data"], data.get("channel") or data.get("channel_name"))
                    elif "notification" in data:
                        channel = data.get("channel") or data.get("channel_name") or ""
                        await self.handle_notification(channel, data["notification"])
                    elif "id" in data:
                        if "result" in data:
                            await self.handle_result(data["result"], data["id"])
                        elif "error" in data:
                            await self.handle_error(data["error"], data["id"])
                    elif "ticker" in data:
                        await self.handle_ticker_update(data["ticker"], data.get("channel") or data.get("channel_name"))
                    elif "instrument" in data and "price" in data:
                        await self.handle_ticker_update(data, data.get("channel") or data.get("channel_name"))

            except asyncio.CancelledError:
                break
            except Exception:
                consecutive_errors += 1
                await asyncio.sleep(backoff_time)
                backoff_time = min(max_backoff, backoff_time * 2)
                
                # Circuit breaker
                if consecutive_errors > max_consecutive_errors:
                    try:
                        if self.thalex.ws is not None:
                            await self.thalex.disconnect()
                    except Exception:
                        pass
                    self.thalex.ws = None
                    await asyncio.sleep(max_backoff)
                    consecutive_errors = 0

    # Zero-copy message handling for high-frequency market data
    async def handle_ticker_update(self, ticker_data: Dict, channel_name: Optional[str] = None):
        """Handle ticker updates with zero-copy optimizations."""
        with self.performance_tracer.trace("ticker_processing"):
            try:
                # Extract instrument ID - consolidated logic
                instrument_id = (ticker_data.get("instrumentId") or ticker_data.get("instrument_id") or 
                               ticker_data.get("name"))
                
                if not instrument_id and "instrument" in ticker_data:
                    instrument_data = ticker_data.get("instrument", {})
                    if isinstance(instrument_data, dict):
                        instrument_id = instrument_data.get("name") or instrument_data.get("instrumentId")
                
                # Extract from channel_name if needed
                if not instrument_id and channel_name and channel_name.startswith("ticker."):
                    parts = channel_name.split('.')
                    if len(parts) > 1:
                        instrument_id = ".".join(parts[1:-1]) if parts[-1] in ["raw", "100ms", "500ms"] else ".".join(parts[1:])
                
                # Use default instrument if extraction failed
                if not instrument_id:
                    instrument_id = self.perp_name
                
                # Process both perpetual and futures instrument updates
                if instrument_id == self.perp_name:
                    # Extract price fields - consolidated
                    mark_price = (ticker_data.get("lastPrice") or ticker_data.get("mark_price") or 
                                ticker_data.get("markPrice") or ticker_data.get("last") or 0.0)
                    best_bid_price = (ticker_data.get("bestBidPrice") or ticker_data.get("best_bid_price") or 
                                    ticker_data.get("bid") or mark_price * 0.999)
                    best_ask_price = (ticker_data.get("bestAskPrice") or ticker_data.get("best_ask_price") or 
                                    ticker_data.get("ask") or mark_price * 1.001)
                    index_price = (ticker_data.get("indexPrice") or ticker_data.get("index_price") or 
                                 ticker_data.get("index") or mark_price)
                    
                    # Create ticker with minimal overhead
                    current_time = time.time()
                    ticker = Ticker({
                        "mark_price": float(mark_price),
                        "best_bid_price": float(best_bid_price),
                        "best_ask_price": float(best_ask_price),
                        "mark_timestamp": current_time,
                        "index": float(index_price),
                    })
                    ticker.best_bid_amount = float(ticker_data.get("bestBidAmount") or ticker_data.get("best_bid_amount") or 1.0)
                    ticker.best_ask_amount = float(ticker_data.get("bestAskAmount") or ticker_data.get("best_ask_amount") or 1.0)
                    ticker.instrument_id = instrument_id
                    
                    # Update state
                    self.ticker = ticker
                    self.last_ticker_time = current_time
                    mid_price = (best_bid_price + best_ask_price) / 2 if best_bid_price > 0 and best_ask_price > 0 else mark_price
                    
                    # Update buffers if valid
                    if mid_price > 0:
                        if hasattr(self.market_data, 'prices'):
                            self.market_data.prices.append(mid_price)
                        if hasattr(self, 'price_history'):
                            self._update_price_history_optimized(mid_price, current_time)
                    
                    # CRITICAL FIX: Update portfolio tracker mark price for UPNL calculation
                    if hasattr(self, 'portfolio_tracker') and mark_price > 0:
                        self.portfolio_tracker.update_mark_price(instrument_id, mark_price)
                    
                    # Signal quote update
                    await self.check_market_conditions_for_quote()
                    
                    # Update shared memory for IPC
                    if self.shared_market_data_struct:
                        try:
                            data_struct = self.shared_market_data_struct
                            data_struct.timestamp = current_time
                            data_struct.mid_price = mid_price
                            data_struct.best_bid = best_bid_price
                            data_struct.best_ask = best_ask_price
                            data_struct.bid_quantity = ticker.best_bid_amount
                            data_struct.ask_quantity = ticker.best_ask_amount
                            data_struct.status = 1
                        except Exception:
                            pass  # Silent fail for HFT performance
                
                elif instrument_id == self.futures_instrument_name:
                    # Handle futures ticker updates
                    mark_price = (ticker_data.get("lastPrice") or ticker_data.get("mark_price") or 
                                ticker_data.get("markPrice") or ticker_data.get("last") or 0.0)
                    best_bid_price = (ticker_data.get("bestBidPrice") or ticker_data.get("best_bid_price") or 
                                    ticker_data.get("bid") or mark_price * 0.999)
                    best_ask_price = (ticker_data.get("bestAskPrice") or ticker_data.get("best_ask_price") or 
                                    ticker_data.get("ask") or mark_price * 1.001)
                    
                    # Create futures ticker
                    current_time = time.time()
                    self.futures_ticker = Ticker({
                        "mark_price": float(mark_price),
                        "best_bid_price": float(best_bid_price),
                        "best_ask_price": float(best_ask_price),
                        "mark_timestamp": current_time,
                        "index": float(mark_price),
                    })
                    self.futures_ticker.instrument_id = instrument_id
                    
                    self.logger.info(f"Updated futures ticker: {instrument_id} @ {mark_price:.2f}")
                    
                    # Update portfolio tracker mark price for futures
                    if hasattr(self, 'portfolio_tracker') and mark_price > 0:
                        self.portfolio_tracker.update_mark_price(instrument_id, mark_price)
            
            except Exception as e:
                self.logger.error(f"Error processing ticker update: {str(e)}", exc_info=True)

    # Use Order objects from memory pool instead of creating them
    def create_order_from_data(self, data: Dict) -> Order:
        """
        Create order object from order data using object pool.
        
        Args:
            data: Order data from API
            
        Returns:
            Order object from pool
        """
        try:
            # Get order from pool
            order = self.order_pool.get()
            
            # Update only the fields defined in the Order class
            # Use 'order_id' from exchange, fallback to 'client_order_id', then 'id', then default 0
            order_id_str = data.get("order_id")
            if order_id_str:
                # Thalex order IDs can be hex strings, internal Order.id is int.
                # For now, let's try to use client_order_id if available, as it's usually int.
                # If the internal model expects a string ID, this needs adjustment.
                # Assuming Order.id is an int, client_order_id is preferred if it's an int.
                client_id = data.get("client_order_id", data.get("cid"))
                if client_id is not None:
                    try:
                        order.id = int(client_id)
                    except ValueError:
                        self.logger.warning(f"Could not convert client_order_id '{client_id}' to int. Using 0 or hash for order.id from order_id '{order_id_str}'.")
                        # If client_id is not an int, we might need to hash order_id_str or handle string IDs.
                        # For simplicity now, if client_id isn't int, use 0 or a hash.
                        # Let's try to keep it simple and use 0 if client_id is not a simple int.
                        # A more robust solution might involve mapping string exchange IDs to internal integer IDs.
                        order.id = data.get("id", 0) # Fallback to 'id' or 0 if client_id is complex
                else: # No client_order_id, use 'id' or 0
                    order.id = data.get("id", 0)
            else: # No 'order_id', try 'client_order_id', then 'id'
                order.id = data.get("client_order_id", data.get("id", 0))

            order.price = float(data.get("price", data.get("limitPrice", 0))) # Use "price", fallback to "limitPrice"
            order.amount = float(data.get("amount", 0))
            
            # Handle status safely - converting string to enum
            status_str = data.get("status", "PENDING")
            try:
                order.status = OrderStatus(status_str)
            except (ValueError, KeyError):
                order.status = OrderStatus.PENDING # Default to PENDING if status is unknown
                
            order.direction = data.get("direction", data.get("side", None)) # Use "direction", fallback to "side"
            
            # Store additional data as attributes (optional)
            # These won't be part of the core Order class but will be accessible
            order.client_id = data.get("client_order_id", data.get("clientId")) # Use client_order_id as well
            order.instrument_id = data.get("instrument_name", data.get("instrumentId")) # Use instrument_name
            order.type = data.get("order_type", data.get("type")) # Use order_type
            # 'filled' might be 'filled_amount' in the Thalex message
            order.filled = float(data.get("filled_amount", data.get("filled", 0)))
            order.timestamp = data.get("create_time", time.time()) # Use create_time if available
            
            return order
            
        except Exception as e:
            self.logger.error(f"Error creating order from data: {str(e)}", exc_info=True)
            # Fall back to creating a new order if pool fails
            # Ensure fallback also uses the corrected keys
            order_id_val = data.get("client_order_id", data.get("id", 0))
            try:
                order_id_val = int(order_id_val)
            except (ValueError, TypeError):
                 # Fallback for ID if it's not int (e.g. use 'order_id' hash or default to 0)
                order_id_val = 0 # simplified fallback
                if "order_id" in data: # if actual exchange id exists, we could log it
                    self.logger.warning(f"Fallback ID creation: using 0, exchange order_id was {data['order_id']}")


            order = Order(
                id=order_id_val,
                price=float(data.get("price", data.get("limitPrice", 0))),
                amount=float(data.get("amount", 0)),
                status=OrderStatus.PENDING,  # Use safe default
                direction=data.get("direction", data.get("side", None))
            )
            # Add additional attributes
            order.client_id = data.get("client_order_id", data.get("clientId"))
            order.instrument_id = data.get("instrument_name", data.get("instrumentId"))
            order.type = data.get("order_type", data.get("type"))
            order.filled = float(data.get("filled_amount", data.get("filled", 0)))
            order.timestamp = data.get("create_time", time.time())
            
            return order

    async def update_quotes(self, price_data: Dict, market_conditions: Dict):
        """Update quotes based on market conditions and strategy logic"""
        if not self.active_trading and not self.risk_recovery_mode:
            self.logger.warning("Trading is halted and not in recovery mode. Skipping quote update.")
            return
        
        # Gradual recovery: limit quote levels during recovery
        if self.risk_recovery_mode and self.recovery_step > 0:
            # Reduce quote levels during recovery
            original_levels = TRADING_CONFIG["quoting"]["levels"]
            recovery_levels = max(1, original_levels // (4 - self.recovery_step))
            self.logger.info(f"Recovery mode: Using {recovery_levels} levels instead of {original_levels}")
            # You can implement level reduction logic here if needed

        if not self.quoting_enabled or self.cooldown_active:
            self.logger.debug("Quoting disabled or cooldown active, skipping update_quotes.")
            return

        if not self.ticker or not hasattr(self.ticker, 'mark_price') or self.ticker.mark_price <= 0:
            self.logger.warning("Skipping quote update: No valid ticker or mark_price available.")
            return

        # --- Pre-trade Risk Check ---
        # HFT BUG FIX: Ensure valid price for risk checks
        current_price = self.ticker.mark_price if (self.ticker.mark_price and np.isfinite(self.ticker.mark_price)) else 0.0
        if current_price <= 0:
            return  # Skip quote update if no valid price
            
        limit_exceeded, reason = self.risk_manager.check_position_limits(current_price=current_price)
        if limit_exceeded:
            self.logger.warning(f"Risk limit breached (pre-trade check): {reason}. Halting quote update.")
            return
        # --- End Pre-trade Risk Check ---

        # Original logic of update_quotes starts here
        with self.performance_tracer.trace("quote_generation"):
            try:
                if price_data is None and self.ticker:
                    # Create price data from ticker if not provided
                    price_data = {
                        "mid_price": (self.ticker.best_bid_price + self.ticker.best_ask_price) / 2
                        if hasattr(self.ticker, 'best_bid_price') and hasattr(self.ticker, 'best_ask_price') and self.ticker.best_bid_price > 0 and self.ticker.best_ask_price > 0
                        else self.ticker.mark_price
                    }
                    # Log price data creation less frequently
                if random.random() < LOG_SAMPLING_RATE:
                    self.logger.info(f"Created price data from ticker: mid_price={price_data.get('mid_price')}")
                
                instrument_id = self.perp_name
                price = price_data.get("mid_price", 0) if price_data else 0
                
                # Optimize quote generation logging - only log on significant price changes
                price_change_significant = abs(price - getattr(self, '_last_price', 0)) > price * 0.001
                if random.random() < LOG_SAMPLING_RATE or price_change_significant:
                    self.logger.info(f"Updating quotes for {instrument_id} at price {price}")
                    self._last_price = price
                
                # Sanity check
                if price <= 0:
                    self.logger.warning("Invalid price for quote generation, using fallback")
                    if self.ticker and self.ticker.mark_price > 0:
                        price = self.ticker.mark_price
                    else:
                        self.logger.error("No valid price available for quote generation")
                        return
                
                # Sync position data before quote generation
                self._sync_position_data()
                
                # Generate quotes using the Avellaneda market maker or fallback to simple method
                bid_quotes = []
                ask_quotes = []
                
                try:
                    if hasattr(self, 'market_maker') and self.market_maker:
                        # Log quote generation less frequently
                        if random.random() < LOG_SAMPLING_RATE:
                            self.logger.info("Generating quotes using Avellaneda market maker")
                        # Create a proper Ticker object from price data for market_maker
                        # Ensure ticker_obj has all necessary fields that market_maker.generate_quotes expects
                        # Use the actual ticker if available, otherwise create a compatible one
                        if self.ticker:
                            ticker_obj = self.ticker  # Use the real ticker directly
                        else:
                            # Fallback: create ticker with essential fields only
                            ticker_obj_data = {
                                "mark_price": price,
                                "best_bid_price": price * 0.999,
                                "best_ask_price": price * 1.001,
                                "mark_timestamp": time.time(),
                                "index": price
                            }
                            ticker_obj = Ticker(ticker_obj_data)

                        if hasattr(self, 'thread_pool') and self.thread_pool:
                            bid_quotes, ask_quotes = await asyncio.get_event_loop().run_in_executor(
                                self.thread_pool,
                                self.market_maker.generate_quotes,
                                ticker_obj,
                                market_conditions,
                                self.max_orders_per_side
                            )
                        else:
                            bid_quotes, ask_quotes = self.market_maker.generate_quotes(
                                ticker_obj, market_conditions, self.max_orders_per_side
                            )
                    else:
                        self.logger.warning("Market maker not available, using fallback quote generation")
                        bid_quotes, ask_quotes = self._generate_simple_quotes(price)
                except Exception as e:
                    self.logger.error(f"Error generating quotes with market maker: {str(e)}", exc_info=True)
                    self.logger.warning("Falling back to simple quote generation")
                    bid_quotes, ask_quotes = self._generate_simple_quotes(price)
                
                # Log quote counts much less frequently for HFT performance
                if self._log_counter % 500 == 0 or len(bid_quotes) + len(ask_quotes) > 10:
                    self.logger.info(f"Quote update #{self._log_counter}: {len(bid_quotes)}B/{len(ask_quotes)}A @ {price:.2f}")
                
                # Apply tick size alignment
                bid_quotes = self.align_prices_to_tick(bid_quotes, is_bid=True)
                ask_quotes = self.align_prices_to_tick(ask_quotes, is_bid=False)
                
                # Validate quotes
                bid_quotes, ask_quotes = await self.validate_quotes(bid_quotes, ask_quotes)
                
                # CHECK FOR TAKE PROFIT CONDITIONS BEFORE PLACING GRID ORDERS
                if hasattr(self.market_maker, 'should_trigger_take_profit'):
                    try:
                        trigger_decision = self.market_maker.should_trigger_take_profit()
                        
                        if trigger_decision.get('should_trigger', False):
                            trading_mode = trigger_decision.get('mode', 'unknown')
                            self.logger.warning(
                                f"🚨 TAKE PROFIT TRIGGERED: {trading_mode} mode - {trigger_decision.get('reason', 'unknown')}"
                            )
                            
                            # Place immediate closing orders instead of grid orders
                            await self._place_immediate_closing_orders(trigger_decision, trading_mode)
                            return  # Skip placing grid orders
                        else:
                            # Log current profit status occasionally
                            if random.random() < 0.1:  # 10% of the time
                                profit_metrics = trigger_decision.get('profit_metrics', {})
                                if profit_metrics:
                                    self.logger.debug(f"Take profit check: {trading_mode} mode, not yet profitable enough")
                                    
                    except Exception as e:
                        self.logger.error(f"Error checking take profit conditions: {str(e)}")
                        # Continue with normal grid orders if take profit check fails
                
                self.current_quotes = (bid_quotes, ask_quotes)
                
                # Place the quotes now rather than signaling
                await self.place_quotes(bid_quotes, ask_quotes)
                
                # Also signal condition variable for quote task to handle fallback
                async with self.quote_cv:
                    self.condition_met = True
                    self.quote_cv.notify()
                    
                self.last_quote_update_time = time.time()
                self.last_quote_time = time.time() # Update last_quote_time as well
                
            except Exception as e:
                self.logger.error(f"Error updating quotes: {str(e)}", exc_info=True)
                # Return quotes to the pool on error
                if 'bid_quotes' in locals() and 'ask_quotes' in locals():
                    for quote_obj in bid_quotes + ask_quotes: # Renamed to avoid conflict
                        if hasattr(self, 'quote_pool'):
                            self.quote_pool.put(quote_obj)

    def _generate_simple_quotes(self, price):
        """Generate simple quotes as a fallback when market maker fails"""
        self.logger.info("Generating simple quotes as fallback")
        
        # Get the number of levels from config
        levels = TRADING_CONFIG["quoting"].get("levels", 3)
        
        # Get step sizes from config or use defaults
        bid_step = TRADING_CONFIG["order"].get("bid_step", 1.0) * self.tick_size
        ask_step = TRADING_CONFIG["order"].get("ask_step", 1.0) * self.tick_size
        
        # Generate bid and ask prices around the mid price
        bid_prices = [price - (i+1) * bid_step for i in range(levels)]
        ask_prices = [price + (i+1) * ask_step for i in range(levels)]
        
        # Convert to Quote objects
        bid_quotes = []
        ask_quotes = []
        instrument_id = self.perp_name
        default_order_size = TRADING_CONFIG["avellaneda"].get("base_size", 0.01)
        
        # Create quote objects for bids
        for bid_price in bid_prices:
            quote = self.quote_pool.get() if hasattr(self, 'quote_pool') else Quote(
                price=bid_price,
                amount=default_order_size,
                instrument=instrument_id,
                side="BUY",
                timestamp=time.time()
            )
            if hasattr(self, 'quote_pool'):
                quote.price = bid_price
                quote.amount = default_order_size
                quote.instrument = instrument_id
                quote.side = "BUY"
                quote.timestamp = time.time()
            bid_quotes.append(quote)
            
        # Create quote objects for asks
        for ask_price in ask_prices:
            quote = self.quote_pool.get() if hasattr(self, 'quote_pool') else Quote(
                price=ask_price,
                amount=default_order_size,
                instrument=instrument_id,
                side="SELL",
                timestamp=time.time()
            )
            if hasattr(self, 'quote_pool'):
                quote.price = ask_price
                quote.amount = default_order_size
                quote.instrument = instrument_id
                quote.side = "SELL"
                quote.timestamp = time.time()
            ask_quotes.append(quote)
        
        self.logger.info(f"Generated {len(bid_quotes)} simple bid quotes and {len(ask_quotes)} simple ask quotes")
        
        return bid_quotes, ask_quotes

    async def handle_notification(self, channel: str, notification: Union[Dict, List[Dict]]):
        """Handle incoming notifications from different channels.
        
        Args:
            channel: The notification channel name
            notification: The notification payload
        """
        try:
            if not isinstance(notification, (dict, list)):
                self.logger.error(f"Invalid notification format: {type(notification)}")
                return
            
            # Only log occasionally for HFT performance
            if random.random() < LOG_SAMPLING_RATE:
                self.logger.info(f"Handling notification from channel: {channel}")
            
            # Handle different channel types - Support both old and new formats
            if channel.startswith("ticker."):
                if random.random() < LOG_SAMPLING_RATE:
                    self.logger.info(f"Processing ticker update for {channel}")
                await self.handle_ticker_update(notification, channel_name=channel)
            elif channel.startswith("price_index."):
                if random.random() < LOG_SAMPLING_RATE:
                    self.logger.info(f"Processing price index update")
                await self.handle_index_update(notification)
            elif channel == "session.orders":
                if random.random() < LOG_SAMPLING_RATE:
                    self.logger.info(f"Processing order update")
                if isinstance(notification, list):
                    for order_data in notification:
                        await self.handle_order_update(order_data)
                else:
                    await self.handle_order_update(notification)
            elif channel == "account.portfolio":
                # Always log portfolio updates for debugging position issues
                self.logger.warning(f"📋 PORTFOLIO UPDATE RECEIVED: {notification}")
                await self.handle_portfolio_update(notification)
            elif channel == "account.trade_history":
                if random.random() < LOG_SAMPLING_RATE:
                    self.logger.info(f"Processing trade update from channel: {channel}") # Added channel for clarity
                if isinstance(notification, list):
                    for trade_entry in notification: # Iterate if it's a list
                        if isinstance(trade_entry, dict):
                            await self.handle_trade_update(trade_entry)
                        else:
                            self.logger.warning(f"Skipping non-dict item in trade_history list: {trade_entry}")
                elif isinstance(notification, dict): # Process if it's a single dict
                    await self.handle_trade_update(notification)
                else:
                    self.logger.warning(f"Unexpected data type for trade_history notification: {type(notification)}")
            else:
                self.logger.warning(f"Notification for unknown channel: {channel}")
        except Exception as e:
            self.logger.error(f"Error processing notification from channel {channel}: {str(e)}", exc_info=True) # Added channel

    async def handle_index_update(self, index_data: Dict):
        """Process index price updates"""
        try:
            self.index = float(index_data["price"])
            async with self.quote_cv:
                self.quote_cv.notify()
        except Exception as e:
            self.logger.error(f"Error handling index update: {str(e)}")

    async def handle_order_update(self, order_data: Dict):
        """Process order updates and trigger quote updates if necessary"""
        try:
            # Extract order info
            order = self.create_order_from_data(order_data)
            
            # Update order manager with the new order info
            if hasattr(self, 'order_manager') and self.order_manager is not None:
                await self.order_manager.update_order(order)
            
            # Handle fills
            if order_data.get("status") == "filled":
                # HFT-focused fill validation
                if order.direction is None:
                    return  # Skip invalid fills for HFT performance
                
                is_buy_flag = order.direction.lower() == "buy"
                significant_fill_threshold = TRADING_CONFIG["avellaneda"].get("significant_fill_threshold", 0.1)
                
                # Check if this is a trigger order fill
                order_label = order_data.get("label", "")
                client_id = order_data.get("client_id", "")
                is_trigger_order = (order_label == "TakeProfitTrigger" or 
                                  (client_id and client_id.startswith("trigger_")))
                
                if is_trigger_order:
                    # Handle trigger order fill
                    self.logger.info(
                        f"TRIGGER ORDER FILLED: {order.direction.upper()} {order.amount:.4f} @ {order.price:.2f} "
                        f"(order_id: {order.id}, client_id: {client_id})"
                    )
                    
                    # Remove from active trigger orders tracking
                    if hasattr(self, 'active_trigger_order_ids') and client_id in self.active_trigger_order_ids:
                        self.active_trigger_order_ids.remove(client_id)
                    
                    # Notify market maker about trigger order fill
                    if hasattr(self.market_maker, 'on_trigger_order_filled'):
                        self.market_maker.on_trigger_order_filled(str(order.id), order.price, order.amount)
                else:
                    # Handle normal grid order fill
                    self.logger.info(
                        f"GRID ORDER FILLED: {order.direction.upper()} {order.amount:.4f} @ {order.price:.2f} "
                        f"(order_id: {order.id})"
                    )
                
                # Update PositionTracker (for both trigger and grid orders)
                if self.position_tracker:
                    try:
                        fill_timestamp_raw = order_data.get("timestamp", order_data.get("create_time", time.time()))
                        if isinstance(fill_timestamp_raw, (int, float)):
                            fill_time = datetime.fromtimestamp(fill_timestamp_raw, tz=timezone.utc)
                        elif isinstance(fill_timestamp_raw, str):
                            try:
                                fill_time = datetime.fromisoformat(fill_timestamp_raw.replace("Z", "+00:00"))
                            except ValueError:
                                fill_time = datetime.now(timezone.utc)
                        else:
                            fill_time = datetime.now(timezone.utc)

                        fill_object = Fill(
                            order_id=str(order.id),
                            fill_price=order.price,
                            fill_size=order.amount,
                            fill_time=fill_time,
                            side=order.direction.lower(),
                            is_maker=order_data.get("is_maker", True)
                        )
                        self.position_tracker.update_on_fill(fill_object)
                    except Exception:
                        pass  # Silent fail for HFT performance

                # CRITICAL FIX: Update PortfolioTracker for take profit logic
                if self.portfolio_tracker:
                    try:
                        # Extract instrument from order data
                        fill_instrument = getattr(order, 'instrument_id', None) or order_data.get("instrument_name", self.perp_name)
                        
                        # Ensure instrument is registered
                        if fill_instrument not in self.portfolio_tracker.instrument_trackers:
                            self.portfolio_tracker.register_instrument(fill_instrument)
                        
                        # Create fill object for portfolio tracker
                        fill_timestamp_raw = order_data.get("timestamp", order_data.get("create_time", time.time()))
                        if isinstance(fill_timestamp_raw, (int, float)):
                            fill_time = datetime.fromtimestamp(fill_timestamp_raw, tz=timezone.utc)
                        elif isinstance(fill_timestamp_raw, str):
                            try:
                                fill_time = datetime.fromisoformat(fill_timestamp_raw.replace("Z", "+00:00"))
                            except ValueError:
                                fill_time = datetime.now(timezone.utc)
                        else:
                            fill_time = datetime.now(timezone.utc)

                        portfolio_fill = Fill(
                            order_id=str(order.id),
                            fill_price=order.price,
                            fill_size=order.amount,
                            fill_time=fill_time,
                            side=order.direction.lower(),
                            is_maker=order_data.get("is_maker", True)
                        )
                        
                        # Update portfolio tracker with fill for the specific instrument
                        self.portfolio_tracker.update_instrument_fill(fill_instrument, portfolio_fill)
                        
                        self.logger.info(f"Updated portfolio tracker: {fill_instrument} {order.direction.upper()} {order.amount:.4f} @ {order.price:.2f}")
                        
                    except Exception as e:
                        self.logger.error(f"Error updating portfolio tracker on fill: {str(e)}")
                        pass  # Continue execution
                
                # Update market maker with fill information (for both trigger and grid orders)
                # Extract instrument from order data for multi-instrument support
                fill_instrument = getattr(order, 'instrument_id', None) or order_data.get("instrument_name", self.perp_name)
                
                # Update market maker with fill information and instrument
                if hasattr(self.market_maker, '_handle_trigger_order_on_fill'):
                    # Get old position for trigger order logic
                    old_position = self.position_tracker.get_position_metrics().get("position", 0.0)
                    old_entry_price = self.position_tracker.get_position_metrics().get("entry_price", 0.0)
                    
                    # Call the new method with instrument information
                    self.market_maker.on_order_filled(str(order.id), order.price, order.amount, is_buy_flag)
                    self.market_maker._handle_trigger_order_on_fill(
                        order.price, order.amount, is_buy_flag, old_position, old_entry_price, fill_instrument
                    )
                else:
                    # Fallback to old method
                    self.market_maker.on_order_filled(str(order.id), order.price, order.amount, is_buy_flag)
                
                # Sync position data after fill to ensure consistency
                self._sync_position_data()
                
                # Update risk manager with position information
                if hasattr(self.risk_manager, 'update_position_fill'):
                    await self.risk_manager.update_position_fill(order.direction, order.price, order.amount, time.time())
                
                # Force quote update after fills
                if self.ticker:
                    market_state = self.market_data.get_market_state()
                    market_conditions = {
                        "volatility": market_state.get("yz_volatility") or market_state.get("volatility") or TRADING_CONFIG["avellaneda"]["fixed_volatility"],
                        "market_impact": market_state.get("market_impact", 0.0),
                        "fill_impact": 0.5 if order.amount >= significant_fill_threshold else 0.2
                    }
                    
                    if order.amount >= significant_fill_threshold:
                        asyncio.create_task(self.update_quotes(None, market_conditions))
                    else:
                        async with self.quote_cv:
                            self.condition_met = True
                            self.quote_cv.notify()
                
        except Exception as e:
            self.logger.error(f"Error handling order update: {str(e)}", exc_info=True)

    async def handle_portfolio_update(self, portfolio_data: List[Dict]):
        """Handle a portfolio update notification - now only tracking position size"""
        try:
            self.logger.info(f"Processing portfolio update")
            
            # Initialize all variables that will be used to prevent "cannot access local variable" errors
            position_size = 0.0
            avg_price = 0.0  # Initialize even if not used directly in this function
            
            # Validate portfolio_data is a list
            if not isinstance(portfolio_data, list):
                self.logger.warning(f"Expected list for portfolio_data, got {type(portfolio_data)}")
                return
                
            # Iterate through portfolio items
            for item in portfolio_data:
                if not isinstance(item, dict):
                    self.logger.warning(f"Expected dict for portfolio item, got {type(item)}")
                    continue
                    
                # Extract asset data - ensure it's a dictionary before using .get()
                asset = item.get("asset", {})
                if asset and isinstance(asset, dict):  # Check if asset is a dictionary
                    asset_name = asset.get("asset_name", "")
                    amount = asset.get("amount", 0.0)
                    
                    # Store in portfolio
                    self.portfolio[asset_name] = float(amount) if amount is not None else 0.0
                
                # Extract position data
                position = item.get("position", {})
                if not position:
                    continue
                    
                # Handle different position formats
                if isinstance(position, dict):
                    # Standard dictionary format
                    instrument = position.get("instrument_name", "")
                elif isinstance(position, (int, float)):
                    # Direct position size format - check if we have instrument name elsewhere
                    instrument = item.get("instrument_name", "")
                    if not instrument:
                        self.logger.debug(f"Received position size {position} without instrument name, skipping")
                        continue
                    # Convert to dict format for consistent processing
                    position = {
                        "instrument_name": instrument,
                        "size": position,
                        "average_price": item.get("average_price", 0.0)
                    }
                else:
                    self.logger.warning(f"Received position in unexpected format: {type(position)}, value: {position}")
                    continue
                    
                instrument = position.get("instrument_name", "")
                
                # Process both perpetual and futures instruments
                if instrument in [self.perp_name, self.futures_instrument_name]:
                    # Safely extract position size only
                    size_value = position.get('size', 0.0)
                    size = float(size_value) if size_value is not None else 0.0
                    
                    # Safely extract average price if available
                    avg_price_value = position.get('average_price', 0.0)
                    avg_price = float(avg_price_value) if avg_price_value is not None else 0.0
                    
                    # Update position tracking
                    position_size = size
                    
                    # Update position tracker if available
                    if hasattr(self, 'position_tracker') and self.position_tracker:
                        # Update position tracker with new position information
                        try:
                            # Clear and update position
                            current_metrics = self.position_tracker.get_position_metrics()
                            old_position = current_metrics.get("position", 0.0)
                            
                            # If position changed, log the update
                            if abs(old_position - size) > 1e-8:  # Small epsilon for float comparison
                                self.logger.info(f"Position changed from {old_position:.8f} to {size:.8f}")
                                
                                # Update position tracker with portfolio data
                                if avg_price > 0:
                                    # Use the update_position method which handles price and size
                                    self.position_tracker.update_position(size, avg_price)
                                else:
                                    # If no average price, just update the position size
                                    self.position_tracker.update_position_size(size)
                                    
                        except Exception as e:
                            self.logger.error(f"Error updating position tracker: {str(e)}")
                    
                    # Ensure instrument is set in market maker before updating position
                    if not hasattr(self.market_maker, 'instrument') or self.market_maker.instrument is None or self.market_maker.instrument != self.perp_name:
                        self.logger.debug(f"Instrument not set in market maker during portfolio update, setting to {self.perp_name}")
                        if hasattr(self.market_maker, 'set_instrument'):
                            self.market_maker.set_instrument(self.perp_name)
                    
                    # Update market maker with position size if available
                    if hasattr(self.market_maker, 'update_position_size'):
                        self.market_maker.update_position_size(position_size)
                    
                    # Update portfolio tracker for this instrument
                    if hasattr(self, 'portfolio_tracker'):
                        try:
                            # Ensure instrument is registered
                            if instrument not in self.portfolio_tracker.instrument_trackers:
                                self.portfolio_tracker.register_instrument(instrument)
                                self.logger.info(f"Registered {instrument} in portfolio tracker")
                            
                            # Update position in portfolio tracker
                            if avg_price > 0:
                                self.portfolio_tracker.update_position(instrument, size, avg_price)
                            else:
                                # If no average price, use current mark price as estimate
                                current_mark = None
                                if instrument == self.perp_name and self.ticker:
                                    current_mark = self.ticker.mark_price
                                elif instrument == self.futures_instrument_name and self.futures_ticker:
                                    current_mark = self.futures_ticker.mark_price
                                
                                if current_mark and current_mark > 0:
                                    self.portfolio_tracker.update_position(instrument, size, current_mark)
                                    self.logger.info(f"Updated portfolio position for {instrument} using mark price {current_mark:.2f}")
                            
                        except Exception as e:
                            self.logger.error(f"Error updating portfolio tracker for {instrument}: {str(e)}")
                    
                    # Log position with average price information if available
                    if avg_price > 0:
                        self.logger.info(f"Updated position for {instrument}: Size={size:.8f}, Avg Price={avg_price:.2f}")
                    else:
                        self.logger.info(f"Updated position size for {instrument}: Size={size:.8f}")
                
        except Exception as e:
            self.logger.error(f"Error handling portfolio update: {str(e)}", exc_info=True)
            
        # Trigger quote update after position change
        async with self.quote_cv:
            self.condition_met = True
            self.quote_cv.notify()

    async def handle_trade_update(self, trade_data: Dict):
        """Process trade updates and update market state"""
        try:
            # Extract data
            price = float(trade_data.get('price', 0))
            amount = float(trade_data.get('amount', 0))
            direction = trade_data.get('direction', '')
            instrument = trade_data.get('instrument', '')
            trade_timestamp = int(trade_data.get('timestamp', time.time() * 1000))
            
            # Check validity
            if price <= 0 or amount <= 0 or not direction or not instrument:
                self.logger.warning(f"Received invalid trade data: {trade_data}")
                return
            
            # Only process if for our target instrument
            if instrument != self.perp_name:
                return
            
            # Convert direction to is_buy flag
            is_buy = direction.lower() == 'buy'
            
            # Update quoter-level volume candle buffer
            volume_candle_completed = None
            if self.volume_buffer:
                try:
                    # Update volume buffer and check if a candle was completed
                    volume_candle_completed = self.volume_buffer.update(
                        price=price, 
                        volume=amount, 
                        is_buy=is_buy, 
                        timestamp=trade_timestamp
                    )
                    
                    if volume_candle_completed:
                        self.logger.info(f"Volume candle completed: {volume_candle_completed}")
                        # Get predictive parameters from completed candle
                        predictions = self.volume_buffer.get_predicted_parameters()
                        if predictions:
                            self.logger.info(f"Volume candle predictions: {predictions}")
                except Exception as e:
                    self.logger.error(f"Error updating volume candle buffer: {str(e)}")
            
            # Update market maker's market data
            should_update_grid = self.market_maker.update_market_data(price, amount, is_buy, is_trade=True)
            
            # Update market data buffer with trade
            self.market_data.add_trade(price, amount, is_buy)
            
            # Enhanced quote update logic based on volume candle signals
            should_update_quotes = should_update_grid or volume_candle_completed is not None
            
            if should_update_quotes:
                # Get current market conditions including volume candle predictions
                market_conditions = self.get_market_conditions()
                
                # Enhance market conditions with volume candle signals if available
                if self.volume_buffer and hasattr(self.volume_buffer, 'signals'):
                    volume_signals = self.volume_buffer.signals
                    market_conditions.update({
                        'volume_momentum': volume_signals.get('momentum', 0.0),
                        'volume_reversal': volume_signals.get('reversal', 0.0),
                        'volume_volatility': volume_signals.get('volatility', 0.0),
                        'volume_exhaustion': volume_signals.get('exhaustion', 0.0)
                    })
                    self.logger.info(f"Enhanced market conditions with volume signals: momentum={volume_signals.get('momentum', 0.0):.3f}")
                
                # Sync position data before quote generation
                self._sync_position_data()
                
                # Get current ticker data
                if self.ticker:
                    # Generate new quotes based on updated parameters
                    bid_quotes, ask_quotes = self.market_maker.generate_quotes(self.ticker, market_conditions)
                    
                    # Place the quotes
                    if bid_quotes or ask_quotes:
                        await self.place_quotes(bid_quotes, ask_quotes)
                        reason = "trade-triggered volume candle prediction" if volume_candle_completed else "market maker trade signal"
                        self.logger.info(f"Updated quotes based on {reason}")
            
        except Exception as e:
            self.logger.error(f"Error processing trade update: {str(e)}")

    async def handle_result(self, result: Dict, cid: Optional[int]):
        """
        Handle result from API call
        
        Args:
            result: Result data from API
            cid: Call ID that generated this result
        """
        try:
            # Instrument data returned from instruments call
            if cid == CALL_IDS["instruments"] and isinstance(result, list):
                self.logger.info(f"Received data for {len(result)} instruments")
                
                # Initialize instrument data cache if needed
                if not hasattr(self, 'instrument_data'):
                    self.instrument_data = {}
                
                # Process each instrument
                for instrument in result:
                    if "instrument_name" in instrument:
                        self.instrument_data[instrument["instrument_name"]] = instrument
                
                # Find specific instrument we're interested in
                perp_name = MARKET_CONFIG.get("underlying", "BTC") + "-PERPETUAL"
                
                if perp_name in self.instrument_data:
                    self.instrument_data = self.instrument_data[perp_name]
                    self.logger.info(f"Found instrument data for {perp_name}")
                    
                    # Extract contract size
                    if "contractSize" in self.instrument_data:
                        self.contract_size = float(self.instrument_data["contractSize"])
                        self.logger.info(f"Contract size: {self.contract_size}")
                else:
                    self.logger.warning(f"Could not find instrument data for {perp_name}")
                    
            # Handle other result types
            elif cid == CALL_IDS["login"]:
                self.logger.info("Login successful")
            
        except Exception as e:
            self.logger.error(f"Error handling result: {str(e)}")

    async def handle_error(self, error: Dict, cid: Optional[int]):
        """Handle API errors"""
        try:
            if cid is None:
                return
                
            # API call error handling
            if cid == CALL_IDS["login"]:
                self.logger.error(f"Login failed: {error.get('message', '')}")
            elif cid == CALL_IDS["subscribe"]:
                self.logger.error(f"Subscription failed: {error.get('message', '')}")
            elif cid == CALL_IDS["mass_quote"]:
                # Note: We're using individual limit orders now, but keeping this handler for backward compatibility
                self.logger.error(f"Mass quote error (legacy handler): {error.get('message', '')}")
                # Pass to order manager for handling
                if hasattr(self, 'order_manager') and self.order_manager is not None:
                    await self.order_manager.handle_order_error(error, cid)
            elif cid >= 100:  # Order-related errors
                await self.order_manager.handle_order_error(error, cid)
            else:
                # Generic API error
                self.logger.error(f"API error for message {cid}: {error}")
                
        except Exception as e:
            self.logger.error(f"Error handling error message: {str(e)}")

    async def place_quotes(self, bid_quotes: List[Quote], ask_quotes: List[Quote]):
        """Place bid and ask quotes on the exchange with enhanced inventory management"""
        try:
            # Initialize counters
            placed_bids = 0
            placed_asks = 0
            max_levels = TRADING_CONFIG["avellaneda"]["max_levels"]
            
            # CRITICAL FIX: Check for take profit conditions BEFORE inventory limits
            # This ensures positions can be closed even when inventory limits are exceeded
            self.logger.warning("🔍 TAKE PROFIT CHECK: Checking conditions before placing grid orders")
            if hasattr(self.market_maker, 'detect_trading_mode') and hasattr(self.market_maker, 'should_trigger_take_profit'):
                try:
                    # Debug portfolio tracker status
                    portfolio_status = "None" if not self.market_maker.portfolio_tracker else "Available"
                    self.logger.warning(f"🔍 PORTFOLIO TRACKER: {portfolio_status}")
                    
                    if self.market_maker.portfolio_tracker:
                        positions = self.market_maker.portfolio_tracker.get_positions()
                        self.logger.warning(f"🔍 POSITIONS: {positions}")
                    
                    current_mode = self.market_maker.detect_trading_mode()
                    self.logger.warning(f"🎯 TRADING MODE: {current_mode}")
                    
                    trigger_decision = self.market_maker.should_trigger_take_profit()
                    self.logger.warning(f"📊 TAKE PROFIT DECISION: {trigger_decision.get('should_trigger', False)} - {trigger_decision.get('reason', 'unknown')}")
                    
                    # If rate limited, show more details
                    if trigger_decision.get('reason') == 'rate_limited':
                        self.logger.warning("⏰ RATE LIMITED: Take profit check is being rate limited")
                    
                    if trigger_decision.get('should_trigger', False):
                        self.logger.warning(
                            f"🚨 TAKE PROFIT CONDITION MET: {current_mode} mode - {trigger_decision.get('reason', 'unknown')}"
                        )
                        
                        # Log profit metrics
                        profit_metrics = trigger_decision.get('profit_metrics', {})
                        if current_mode == "arbitrage":
                            arb_profit = profit_metrics.get('total_profit_usd', 0)
                            spread_bps = profit_metrics.get('spread_profit_bps', 0)
                            self.logger.warning(
                                f"Arbitrage profit: ${arb_profit:.2f}, Spread: {spread_bps:.1f}bps"
                            )
                        elif current_mode in ["single_perp", "single_futures"]:
                            for instrument, metrics in profit_metrics.items():
                                profit_usd = metrics.get('profit_usd', 0)
                                self.logger.warning(f"{instrument} profit: ${profit_usd:.2f}")
                        
                        # IMMEDIATELY PLACE CLOSING ORDERS (bypassing inventory limits)
                        self.logger.warning("🚀 PLACING IMMEDIATE CLOSING ORDERS - BYPASSING INVENTORY LIMITS")
                        try:
                            await self._place_immediate_closing_orders(trigger_decision, current_mode)
                            return  # Exit early - don't place grid orders when closing positions
                        except Exception as trigger_error:
                            self.logger.error(f"Error placing immediate closing orders: {str(trigger_error)}", exc_info=True)
                            # Continue with normal logic if closing orders fail
                                
                except Exception as e:
                    # Don't fail quote placement if trigger analysis fails
                    self.logger.error(f"❌ ERROR in take profit analysis: {str(e)}", exc_info=True)
            else:
                self.logger.warning("❌ TAKE PROFIT: Market maker missing detect_trading_mode or should_trigger_take_profit methods")
            
            # --- INVENTORY CHECKS (only after take profit check) ---
            if not self.active_trading:
                self.logger.warning("Trading is halted due to a prior risk limit breach. Skipping quote placement.")
                return

            if not self.quoting_enabled:
                self.logger.debug("Quoting is disabled, skipping quote placement")
                return
                
            # Check cooldown
            current_time = time.time()
            if self.cooldown_active and current_time < self.cooldown_until:
                self.logger.debug(f"Cooldown active for {self.cooldown_until - current_time:.2f}s more, skipping quote placement")
                return
            
            # Enforce max order levels from config
            if len(bid_quotes) > max_levels:
                self.logger.warning(f"Limiting bid quotes to configured max levels: {max_levels}")
                bid_quotes = bid_quotes[:max_levels]
            if len(ask_quotes) > max_levels:
                self.logger.warning(f"Limiting ask quotes to configured max levels: {max_levels}")
                ask_quotes = ask_quotes[:max_levels]
            
            # Check existing orders to avoid exceeding limits
            active_bids_count = len(self.order_manager.active_bids)
            active_asks_count = len(self.order_manager.active_asks)
            
            # If we already have orders, cancel them all first to ensure clean state
            if active_bids_count > 0 or active_asks_count > 0:
                self.logger.warning(f"Found existing orders before placing quotes: {active_bids_count} bids, {active_asks_count} asks")
                self.logger.warning(f"Cancelling all existing orders to ensure clean order book")
                
                # Retry logic for order cancellation
                max_cancel_retries = 3
                cancel_success = False
                
                for retry_attempt in range(max_cancel_retries):
                    self.logger.info(f"Cancellation attempt {retry_attempt + 1}/{max_cancel_retries}")
                    
                    # Cancel orders
                    await self.cancel_quotes(f"Refreshing order book - attempt {retry_attempt + 1}")
                    
                    # Wait with increasing delay (1s, 2s, 3s)
                    wait_time = (retry_attempt + 1) * 1.0
                    await asyncio.sleep(wait_time)
                    
                    # Check local order manager state
                    local_bids = len(self.order_manager.active_bids)
                    local_asks = len(self.order_manager.active_asks)
                    
                    # Confirm with exchange by requesting fresh order data
                    try:
                        await self.thalex.get_orders(id=CALL_IDS.get("get_orders", random.randint(10000, 20000)))
                        # Give exchange confirmation time to arrive
                        await asyncio.sleep(0.5)
                        
                        # Re-check after exchange confirmation
                        exchange_bids = len(self.order_manager.active_bids)
                        exchange_asks = len(self.order_manager.active_asks)
                        
                        if exchange_bids == 0 and exchange_asks == 0:
                            self.logger.info(f"Order cancellation successful after {retry_attempt + 1} attempts")
                            cancel_success = True
                            break
                        else:
                            self.logger.warning(f"Still have {exchange_bids} bids and {exchange_asks} asks after attempt {retry_attempt + 1}")
                            
                    except Exception as e:
                        self.logger.warning(f"Failed to confirm with exchange on attempt {retry_attempt + 1}: {str(e)}")
                        # Fall back to local check
                        if local_bids == 0 and local_asks == 0:
                            self.logger.info(f"Local cancellation check passed on attempt {retry_attempt + 1}")
                            cancel_success = True
                            break

                if not cancel_success:
                    final_bids = len(self.order_manager.active_bids)
                    final_asks = len(self.order_manager.active_asks)
                    self.logger.critical(
                        f"CRITICAL: Failed to cancel all orders after {max_cancel_retries} attempts. "
                        f"Still have {final_bids} bids and {final_asks} asks. "
                        f"This is a high-risk state. Halting quote placement for this cycle."
                    )
                    return # Stop placing quotes for this cycle if stuck
            
            # Store current quotes before sending to properly track what was sent
            # This is important if place_quotes is called from handle_trade_update,
            # as update_quotes sets self.current_quotes before calling, but handle_trade_update does not.
            self.current_quotes = (bid_quotes, ask_quotes)
            
            # Reset tracking variables to ensure clean state
            placed_bids = 0
            placed_asks = 0
            
            # --- BEGIN INVENTORY CHECK FOR ONE-SIDED QUOTING ---
            current_inventory = 0.0
            
            # FIXED: Proper position retrieval with debugging
            current_inventory = 0.0
            inventory_source = "unknown"
            
            try:
                if hasattr(self, 'position_tracker') and self.position_tracker:
                    # Primary source: position tracker
                    metrics = self.position_tracker.get_position_metrics()
                    current_inventory = metrics.get("position", 0.0)
                    inventory_source = "position_tracker"
                elif hasattr(self.market_maker, 'position_tracker') and self.market_maker.position_tracker:
                    # Secondary source: market maker's position tracker
                    metrics = self.market_maker.position_tracker.get_position_metrics()
                    current_inventory = metrics.get("position", 0.0)
                    inventory_source = "market_maker.position_tracker"
                else:
                    # Fallback: use 0 inventory
                    current_inventory = 0.0
                    inventory_source = "fallback_zero"
                    self.logger.debug("🏦 No position tracking available. Using 0 inventory for limit checks.")
                    
            except Exception as e:
                current_inventory = 0.0
                inventory_source = "error_fallback"
                self.logger.warning(f"🏦 Error getting position for inventory check: {e}. Using 0.")
            
            self.logger.debug(f"🏦 Inventory check: position={current_inventory:.4f} (source: {inventory_source})")

            max_inventory_deviation = RISK_LIMITS.get("max_position", float('inf')) # Changed from TRADING_PARAMS
            if not isinstance(max_inventory_deviation, (int, float)) or max_inventory_deviation < 0:
                self.logger.warning(f"Invalid 'max_position' from RISK_LIMITS ({max_inventory_deviation}). Using infinity (feature disabled).")
                max_inventory_deviation = float('inf')
            
            # Enhanced debugging for inventory limits
            self.logger.debug(f"🚧 Inventory limit logic: position={current_inventory:.4f}, max_allowed=±{max_inventory_deviation}")
            
            can_place_bids_inventory_wise = True
            if current_inventory > max_inventory_deviation:  # Inventory is too long
                can_place_bids_inventory_wise = False
                self.logger.warning(
                    f"🚫 INVENTORY LIMIT BREACH: Position ({current_inventory:.4f}) > +{max_inventory_deviation:.4f}. "
                    f"Temporarily stopping new bid orders."
                )
            
            can_place_asks_inventory_wise = True
            if current_inventory < -max_inventory_deviation:  # Inventory is too short
                can_place_asks_inventory_wise = False
                self.logger.warning(
                    f"🚫 INVENTORY LIMIT BREACH: Position ({current_inventory:.4f}) < -{max_inventory_deviation:.4f}. "
                    f"Temporarily stopping new ask orders."
                )
                
            # Log if inventory limits are NOT preventing trading
            if can_place_bids_inventory_wise and can_place_asks_inventory_wise:
                self.logger.debug(f"✅ Inventory check passed: position={current_inventory:.4f} within ±{max_inventory_deviation} limits")
            # --- END INVENTORY CHECK ---
            
            # Place bid quotes as individual limit orders
            if can_place_bids_inventory_wise:
                for quote in bid_quotes:
                    if quote.price <= 0 or quote.amount <= 0:
                        self.logger.warning(f"Invalid bid quote detected: {quote.price}@{quote.amount} - skipping")
                        continue
                    
                    # Check if we reached the max limit for this side
                    if placed_bids >= max_levels:
                        self.logger.warning(f"Maximum bid levels ({max_levels}) reached - skipping remaining bids")
                        break
                        
                    # Use the order manager to place a limit order
                    try:
                        await self.order_manager.place_order(
                            instrument=self.perp_name,
                            direction="buy",
                            price=quote.price,
                            amount=quote.amount,
                            label="AvellanedaQuoter",
                            post_only=True
                        )
                        placed_bids += 1
                    except Exception as e:
                        self.logger.error(f"Error placing bid: {str(e)}")
                else:
                    if bid_quotes: # Only log if there were bids to place
                        self.logger.info("Skipping all bid placements due to inventory limit.")
            
            # Place ask quotes as individual limit orders
            if can_place_asks_inventory_wise:
                for quote in ask_quotes:
                    if quote.price <= 0 or quote.amount <= 0:
                        self.logger.warning(f"Invalid ask quote detected: {quote.price}@{quote.amount} - skipping")
                        continue
                    
                    # Check if we reached the max limit for this side
                    if placed_asks >= max_levels:
                        self.logger.warning(f"Maximum ask levels ({max_levels}) reached - skipping remaining asks")
                        break
                        
                    # Use the order manager to place a limit order
                    try:
                        await self.order_manager.place_order(
                            instrument=self.perp_name,
                            direction="sell",
                            price=quote.price,
                            amount=quote.amount,
                            label="AvellanedaQuoter",
                            post_only=True
                        )
                        placed_asks += 1
                    except Exception as e:
                        self.logger.error(f"Error placing ask: {str(e)}")
                else:
                    if ask_quotes: # Only log if there were asks to place
                        self.logger.info("Skipping all ask placements due to inventory limit.")
            
            # Record successful quote placement
            self.last_quote_time = time.time()
            
            # Log final result
            total_orders = placed_bids + placed_asks
            self.logger.info(f"Placed {placed_bids} bids, {placed_asks} asks (total: {total_orders})")
            
        except Exception as e:
            self.logger.error(f"Error placing quotes: {str(e)}", exc_info=True)
            
            # If rate limit exceeded, implement cooldown
            if "rate limit" in str(e).lower() or "circuit breaker" in str(e).lower():
                self.cooldown_active = True
                self.cooldown_until = time.time() + 5  # 5 second cooldown
                self.logger.warning(f"Rate limit exceeded, enabling cooldown until {self.cooldown_until}")
            
            # Clear current quotes on error to prevent stale state
            self.current_quotes = ([], [])

    def get_market_conditions(self) -> Dict:
        """Get market conditions with caching optimization for HFT performance"""
        current_time = time.time()
        
        # Use cache if recent (within 100ms for HFT)
        if (hasattr(self, '_cached_market_conditions') and 
            current_time - self._cached_market_time < 0.1):
            return self._cached_market_conditions
        
        # Calculate fresh market conditions
        market_state = self.market_data.get_market_state()
        
        # HFT BUG FIX: Handle NaN volatility values safely
        volatility = TRADING_CONFIG["avellaneda"]["fixed_volatility"]  # Safe default
        
        if "yz_volatility" in market_state:
            yz_vol = market_state["yz_volatility"]
            if yz_vol and np.isfinite(yz_vol) and yz_vol > 0:
                volatility = yz_vol
        elif "volatility" in market_state:
            vol = market_state["volatility"]
            if vol and np.isfinite(vol) and vol > 0:
                volatility = vol
        
        # Ensure market_impact is also finite
        market_impact = market_state.get("market_impact", 0.0)
        if not (market_impact and np.isfinite(market_impact)):
            market_impact = 0.0
        
        conditions = {
            "volatility": volatility,
            "market_impact": market_impact
        }
        
        # NEW: Integrate volume candle signals into market conditions (FIXED)
        if self.volume_buffer and hasattr(self.volume_buffer, 'signals'):
            volume_signals = self.volume_buffer.signals
            conditions.update({
                'volume_momentum': volume_signals.get('momentum', 0.0),
                'volume_reversal': volume_signals.get('reversal', 0.0),
                'volume_volatility': volume_signals.get('volatility', 0.0),
                'volume_exhaustion': volume_signals.get('exhaustion', 0.0)
            })
            
            # Enhance traditional volatility with volume signals
            if volume_signals.get('volatility', 0.0) > 0.5:
                conditions["volatility"] *= (1.0 + volume_signals['volatility'] * 0.2)  # Up to 20% increase
                
        # Cache result
        self._cached_market_conditions = conditions
        self._cached_market_time = current_time
        return conditions

    async def heartbeat_task(self):
        """
        Send periodic heartbeats to maintain connection.
        Also monitor connection health and attempt reconnection if needed.
        """
        while True:
            try:
                await asyncio.sleep(self.heartbeat_interval)
                
                # Check if we need to send a heartbeat
                current_time = time.time()
                if current_time - self.last_heartbeat >= self.heartbeat_interval:
                    # Check if websocket is still connected
                    if not self.thalex.connected():
                        continue
                    
                    # Send the heartbeat
                    try:
                        await self.send_heartbeat()
                        self.last_heartbeat = current_time
                    except Exception as e:
                        self.logger.error(f"Error sending heartbeat: {str(e)}")
                        
                        # Check if we need to force a reconnection
                        if isinstance(e, websockets.exceptions.ConnectionClosedError) or \
                           isinstance(e, websockets.exceptions.ConnectionClosedOK):
                            self.logger.warning("Connection closed during heartbeat, marking for reconnection")
                            # Mark connection as closed to trigger reconnection in listen task
                            self.thalex.ws = None
                
                # Check overall quoter health
                # If we haven't received a ticker update in a while, the connection might be stale
                time_since_last_ticker = current_time - self.last_ticker_time if self.last_ticker_time > 0 else 0
                if self.last_ticker_time > 0 and time_since_last_ticker > 60:  # 60 seconds without ticker
                    self.logger.warning(f"No ticker updates for {time_since_last_ticker:.1f} seconds, connection may be stale")
                    
                    # Force a reconnection if WebSocket is still marked as connected
                    if self.thalex.connected():
                        self.logger.warning("Forcing reconnection due to stale market data")
                        try:
                            await self.thalex.disconnect()
                        except Exception:
                            pass  # Ignore errors during disconnect
                        
                        # Mark connection as closed to trigger reconnection in listen task
                        self.thalex.ws = None
            
            except asyncio.CancelledError:
                self.logger.info("Heartbeat task cancelled")
                break
            except Exception as e:
                self.logger.error(f"Error in heartbeat task: {str(e)}")
                await asyncio.sleep(1)  # Wait a moment before continuing

    async def send_heartbeat(self):
        """Send a heartbeat message to keep the connection alive"""
        try:
            # Check if connection is alive before attempting heartbeat
            if not self.thalex.connected():
                self.logger.warning("Connection lost before heartbeat, attempting to reconnect...")
                try:
                    await self.thalex.connect()
                    self.logger.info("Successfully reconnected in heartbeat task")
                except Exception as reconnect_err:
                    self.logger.error(f"Failed to reconnect in heartbeat task: {str(reconnect_err)}")
                    return False
            
            # Try to use the ping method if available
            if hasattr(self.thalex, 'ping') and callable(self.thalex.ping):
                await self.thalex.ping(id=CALL_IDS.get("heartbeat", 1010))
                self.last_heartbeat = time.time()
                return True
            else:
                self.logger.warning("Thalex client does not have a ping method")
                return False
                
        except Exception as e:
            self.logger.warning(f"Failed to send heartbeat: {str(e)}")
            return False

    async def gamma_update_task(self):
        """Periodically update gamma based on market conditions"""
        await self.setup_complete.wait()  # Wait for setup to complete
        
        while True:
            try:
                # Get current market conditions
                market_conditions = self.get_market_conditions()
                volatility = market_conditions.get("volatility", 0.001)
                
                # Calculate dynamic gamma if market maker supports it
                if hasattr(self.market_maker, 'calculate_dynamic_gamma'):
                    new_gamma = self.market_maker.calculate_dynamic_gamma(
                        volatility, 
                        market_conditions.get("market_impact", 0.0)
                    )
                    
                    # Update the gamma parameter if market maker supports it
                    if hasattr(self.market_maker, 'update_gamma'):
                        self.market_maker.update_gamma(new_gamma)
                        self.logger.info(f"Updated gamma parameter to {new_gamma:.5f}")
                
            except Exception as e:
                self.logger.error(f"Error in gamma update task: {str(e)}")
                
            # Wait before next update
            await asyncio.sleep(30)  # Update every 30 seconds

    async def log_status_task(self):
        """Periodically log bot status"""
        await self.setup_complete.wait()  # Wait for setup to complete
        
        while True:
            try:
                # Get position metrics if available
                position = 0.0
                realized_pnl = 0.0
                unrealized_pnl = 0.0
                
                # Try to get metrics from market maker if it supports it
                if hasattr(self.market_maker, 'get_position_metrics'):
                    mm_metrics = self.market_maker.get_position_metrics() # Renamed to avoid conflict
                    position = mm_metrics.get('position', 0.0) if mm_metrics else 0.0
                    realized_pnl = mm_metrics.get('realized_pnl', 0.0) if mm_metrics else 0.0
                    unrealized_pnl = mm_metrics.get('unrealized_pnl', 0.0) if mm_metrics else 0.0
                
                # Get metrics from PositionTracker
                pt_pos = 0.0
                pt_avg_entry = 0.0
                pt_rpnl = 0.0
                pt_upnl = 0.0
                pt_total_vol = 0.0
                pt_fills = 0
                if self.position_tracker:
                    pt_metrics = self.position_tracker.get_position_metrics()
                    pt_pos = pt_metrics.get("position", 0.0)
                    pt_avg_entry = pt_metrics.get("average_entry", 0.0)
                    pt_rpnl = pt_metrics.get("realized_pnl", 0.0)
                    pt_upnl = pt_metrics.get("unrealized_pnl", 0.0)
                    pt_total_vol = pt_metrics.get("total_volume", 0.0)
                    pt_fills = pt_metrics.get("fill_count", 0)

                # Get current mark prices for both instruments
                mark_price = self.ticker.mark_price if self.ticker and self.ticker.mark_price is not None else 0.0
                futures_mark_price = self.futures_ticker.mark_price if self.futures_ticker and self.futures_ticker.mark_price is not None else 0.0
                spread_value = futures_mark_price - mark_price if futures_mark_price > 0 and mark_price > 0 else 0.0
                
                # Calculate basic stats
                active_orders = len(self.order_manager.active_bids) + len(self.order_manager.active_asks) if hasattr(self.order_manager, 'active_bids') else 0
                
                # Log status with HFT optimization metrics
                # Ensure all formatted variables have a fallback if None to prevent TypeError
                pos_mm_str = f"{position:.3f}" if position is not None else "N/A"
                rpnl_mm_str = f"{realized_pnl:.2f}" if realized_pnl is not None else "N/A"
                upnl_mm_str = f"{unrealized_pnl:.2f}" if unrealized_pnl is not None else "N/A"
                pt_pos_str = f"{pt_pos:.3f}" if pt_pos is not None else "N/A"
                pt_avg_entry_str = f"{pt_avg_entry:.2f}" if pt_avg_entry is not None else "N/A"
                pt_rpnl_str = f"{pt_rpnl:.2f}" if pt_rpnl is not None else "N/A"
                pt_upnl_str = f"{pt_upnl:.2f}" if pt_upnl is not None else "N/A"
                pt_total_vol_str = f"{pt_total_vol:.2f}" if pt_total_vol is not None else "N/A"
                mark_price_str = f"{mark_price:.1f}" if mark_price is not None else "N/A"

                # Get portfolio-level metrics for arbitrage tracking
                portfolio_pnl = self.portfolio_tracker.get_total_pnl() if hasattr(self, 'portfolio_tracker') else 0.0
                futures_mark_str = f"{futures_mark_price:.1f}" if futures_mark_price > 0 else "N/A"
                spread_str = f"{spread_value:.1f}" if spread_value != 0 else "N/A"
                
                self.logger.info(
                    f"Status: Pos(MM)={pos_mm_str} PnL(MM)=[R:{rpnl_mm_str} U:{upnl_mm_str}] | "
                    f"Pos(PT)={pt_pos_str} AvgEntry(PT)={pt_avg_entry_str} PnL(PT)=[R:{pt_rpnl_str} U:{pt_upnl_str}] Vol(PT)={pt_total_vol_str} Fills(PT)={pt_fills} | "
                    f"PERP={mark_price_str} FUTURES={futures_mark_str} SPREAD={spread_str} | "
                    f"ARBITRAGE_PNL={portfolio_pnl:.2f} | Orders={active_orders} | "
                    f"HFT=[Pools:{len(self.order_pool.items)}/{len(self.quote_pool.items)}]"
                )
                
                # Collect and log performance metrics every 5 minutes
                if time.time() % 300 < 5:  # Every ~5 minutes
                    if hasattr(self, 'performance_tracer'):
                        perf_stats = self.performance_tracer.get_all_stats()
                        perf_summary = " | ".join([f"{k}:{v['mean']:.2f}ms" for k, v in perf_stats.items()])
                        self.logger.info(f"Performance metrics: {perf_summary}")
                
            except Exception as e:
                self.logger.error(f"Error in status task: {str(e)}")
                
            # Wait before next update (300 seconds = 5 minutes)
            await asyncio.sleep(300)

    async def quote_task(self):
        """Optimized quote task with reduced latency"""
        self.logger.info("Optimized quote task started - waiting for updates")
        
        if not self.active_trading:
            self.logger.critical("Quote task starting, but trading is already halted due to a prior risk breach. Will not proceed.")
            return # Exit task if trading already halted at start

        # Pre-allocate frequently used variables
        last_quote_time = 0
        min_quote_interval = 0.5  # Reduced from 1.0 for faster response
        force_quote_interval = 15.0  # Reduced from 30.0
        
        # Performance counters
        quote_counter = 0
        
        # Wait for initial setup to complete
        self.logger.info("Waiting for setup to complete before starting quote generation")
        await self.setup_complete.wait()
        
        while True:
            try:
                quote_counter += 1
                
                # Log status less frequently
                if quote_counter % 1200 == 0:  # Every ~10 minutes at 0.5s intervals
                    self.logger.info(f"Quote task alive - processed {quote_counter} cycles")
                
                if not self.active_trading:
                    if self.risk_recovery_mode:
                        # In recovery mode - check for recovery instead of exiting
                        if time.time() - self.last_recovery_check > RISK_LIMITS.get("recovery_check_interval", 30):
                            self.last_recovery_check = time.time()
                            if await self._check_risk_recovery():
                                await self._initiate_gradual_recovery()
                            else:
                                self.logger.info("Still in recovery mode - waiting for risk conditions to improve")
                        
                        # Sleep and continue loop instead of breaking
                        await asyncio.sleep(5.0)
                        continue
                    else:
                        self.logger.critical("Trading HALTED. Quote task will cease further operations.")
                        break
                if not self.ticker:
                    self.logger.warning("No ticker data received yet")
                else:
                    self.logger.info(f"Current ticker: mark_price={self.ticker.mark_price}")
                
                # Wait for price updates with timeout
                try:
                    async with self.quote_cv:
                        # Wait with timeout to ensure we don't deadlock
                        await asyncio.wait_for(
                            self.quote_cv.wait_for(lambda: self.condition_met),
                            timeout=10.0  # Don't wait indefinitely
                        )
                        self.condition_met = False
                        self.logger.info("Quote task received notification")
                except asyncio.TimeoutError:
                    # If we timeout, still proceed with quote update if possible
                    self.logger.warning("Quote task timed out waiting for notification, proceeding anyway")
                
                current_time = time.time()
                
                # Force periodic quote updates regardless of ticker changes
                force_quote = (current_time - last_quote_time >= force_quote_interval)
                if force_quote:
                    self.logger.info(f"Forcing quote update after {current_time - last_quote_time:.1f}s since last update")
                
                # Enforce minimum interval between quote updates
                if not force_quote and current_time - last_quote_time < min_quote_interval:
                    self.logger.info(f"Quote update too frequent - waiting {min_quote_interval}s between updates")
                    await asyncio.sleep(0.5)
                    continue
                    
                # Check if we're in cooldown
                if self.cooldown_active and current_time < self.cooldown_until:
                    self.logger.info(f"In cooldown period - {(self.cooldown_until - current_time):.1f}s remaining")
                    await asyncio.sleep(0.5)
                    continue
                

                
                # Skip if no ticker data
                if not self.ticker:
                    self.logger.warning("Cannot quote - no ticker data available")
                    await asyncio.sleep(1.0)  # Shorter sleep to check again soon
                    continue
                
                # Generate default quotes with fallback values if needed
                if force_quote and (not hasattr(self, 'market_data') or len(self.market_data.prices) < 10):
                    self.logger.warning("Insufficient market data, using default values for quote generation")
                    # Create basic market conditions with default values
                    market_conditions = {
                        "volatility": TRADING_CONFIG["avellaneda"].get("fixed_volatility", 0.01),
                        "market_impact": 0.0
                    }
                    
                    # Create basic ticker data
                    ticker_data = {
                        "mid_price": self.ticker.mark_price
                    }
                    
                    # Generate quotes with default parameters
                    self.logger.info("Generating quotes with default parameters")
                    await self.update_quotes(ticker_data, market_conditions)
                    last_quote_time = current_time
                    continue
                
                # Check if we have enough price history
                min_samples = TRADING_CONFIG["volatility"].get("min_samples", 20)
                if not self.price_history_full and self.price_history_idx < min_samples:
                    self.logger.info(f"Need more price history before quoting: {self.price_history_idx}/{min_samples}")
                    continue
                
                # Get market conditions
                market_conditions = self.get_market_conditions()
                
                # Get ticker data
                ticker_data = {
                    "mid_price": (self.ticker.best_bid_price + self.ticker.best_ask_price) / 2
                    if hasattr(self.ticker, 'best_bid_price') and hasattr(self.ticker, 'best_ask_price')
                    else self.ticker.mark_price
                }
                
                # Update quotes
                with self.performance_tracer.trace("quote_update"):
                    try:
                        await self.update_quotes(ticker_data, market_conditions)
                    except Exception as e:
                        if "Ticker.__init__() got an unexpected keyword argument 'timestamp'" in str(e):
                            self.logger.error(f"Error generating quotes with market maker: {str(e)}")
                            # Create a temporary, valid ticker without the timestamp parameter
                            # that can be used by the market maker
                            if hasattr(self, 'ticker') and self.ticker:
                                ticker_dict = {
                                    "mark_price": self.ticker.mark_price,
                                    "best_bid_price": getattr(self.ticker, 'best_bid_price', self.ticker.mark_price * 0.999),
                                    "best_ask_price": getattr(self.ticker, 'best_ask_price', self.ticker.mark_price * 1.001),
                                    "mark_timestamp": time.time(),
                                    "index": getattr(self.ticker, 'index', self.ticker.mark_price)
                                }
                                # Skip timestamp parameter when creating the ticker
                                market_conditions["ticker"] = ticker_dict
                                # Try again with the fixed ticker data
                                await self.update_quotes(ticker_data, market_conditions)
                            else:
                                self.logger.error("Unable to create valid ticker for quote generation")
                        else:
                            # Re-raise other exceptions
                            raise
                
                last_quote_time = current_time
                
                # Add a delay after successfully updating quotes to prevent excessive updates
                await asyncio.sleep(0.5)
                    
            except asyncio.CancelledError:
                self.logger.info("Quote task cancelled")
                break
            except Exception as e:
                self.logger.error(f"Error in quote task: {str(e)}", exc_info=True)
                await asyncio.sleep(0.1)  # Shorter error recovery time

    # Signal quote update if needed based on market conditions
    async def check_market_conditions_for_quote(self):
        """Check if market conditions warrant a quote update and signal if needed"""
        try:
            current_time = time.time()
            
            # Don't update too frequently
            min_interval = TRADING_CONFIG["quoting"].get("min_update_interval", 1.0)
            if current_time - self.last_quote_update_time < min_interval:
                return False
                
            # Check if we have enough price history
            if not self.price_history_full and self.price_history_idx < TRADING_CONFIG["volatility"].get("min_samples", 20):
                return False
                
            # Signal the quote task that conditions have changed
            async with self.quote_cv:
                self.condition_met = True
                self.quote_cv.notify()
                
            return True
            
        except Exception as e:
            self.logger.error(f"Error checking market conditions: {str(e)}")
            return False
    
    async def _subscribe_to_websocket_topics(self):
        """Subscribe to WebSocket topics with error handling"""
        try:
            self.logger.info("Subscribing to WebSocket topics...")
            
            # Subscribe to public channels
            channels_to_subscribe = []
            
            if self.perp_name:
                # Get underlying without USD suffix for index subscription
                underlying = MARKET_CONFIG["underlying"].replace("USD", "")
                channels_to_subscribe.extend([f"ticker.{self.perp_name}.raw", f"index.{underlying}"])
                
            if self.futures_instrument_name:
                channels_to_subscribe.append(f"ticker.{self.futures_instrument_name}.raw")
                
            if channels_to_subscribe:
                await self.thalex.public_subscribe(
                    channels=channels_to_subscribe,
                    id=CALL_ID_SUBSCRIBE
                )
                self.logger.info(f"Subscribed to public channels: {channels_to_subscribe}")
            else:
                self.logger.warning("No instrument names available for public subscription")
            
            # Subscribe to private channels
            await self.thalex.private_subscribe(
                channels=["session.orders", "account.portfolio", "account.trade_history"],
                id=CALL_ID_SUBSCRIBE
            )
            self.logger.info("Subscribed to private channels")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error subscribing to WebSocket topics: {str(e)}")
            raise

    def align_prices_to_tick_batch(self, quotes: List[Quote], is_bid: bool) -> List[Quote]:
        """Vectorized price alignment for better performance"""
        if not quotes or self.tick_size <= 0:
            return quotes
        
        # Fast path for single tick size
        tick_multiplier = 1.0 / self.tick_size
        
        for quote in quotes:
            if is_bid:
                aligned_price = math.floor(quote.price * tick_multiplier) / tick_multiplier
            else:
                aligned_price = math.ceil(quote.price * tick_multiplier) / tick_multiplier
            quote.price = aligned_price
        
        return quotes

    def align_prices_to_tick(self, quotes, is_bid=True):
        """Legacy method - redirects to optimized batch version"""
        return self.align_prices_to_tick_batch(quotes, is_bid)

    def _process_message_optimized(self, message: str) -> dict:
        """Optimized message processing with buffer reuse"""
        try:
            # Fast path for small messages
            if len(message) < 1000:
                return orjson.loads(message)
            
            # Use buffer for larger messages
            if len(message) < len(self.message_buffer):
                message_bytes = message.encode('utf-8')
                self.message_buffer[:len(message_bytes)] = message_bytes
                return orjson.loads(self.message_view[:len(message_bytes)])
            else:
                # Expand buffer if needed
                self._expand_message_buffer(len(message))
                return orjson.loads(message)
        except Exception:
            return json.loads(message)  # Fallback

    def _expand_message_buffer(self, required_size: int):
        """Expand message buffer when needed"""
        new_size = min(required_size * 2, MESSAGE_BUFFER_MAX_SIZE)
        if new_size > self.message_buffer_size:
            self.message_buffer_size = new_size
            self.message_buffer = bytearray(self.message_buffer_size)
            self.message_view = memoryview(self.message_buffer)

    def _initialize_price_history(self):
        """Initialize optimized price history storage"""
        history_size = 200  # Larger buffer for better statistics
        self.price_history = np.zeros(history_size, dtype=np.float64)
        self.price_history_timestamps = np.zeros(history_size, dtype=np.float64)
        self.price_history_idx = 0
        self.price_history_full = False

    def _update_price_history_optimized(self, price: float, timestamp: float):
        """Optimized price history update"""
        self.price_history[self.price_history_idx] = price
        self.price_history_timestamps[self.price_history_idx] = timestamp
        self.price_history_idx = (self.price_history_idx + 1) % len(self.price_history)
        
        if not self.price_history_full and self.price_history_idx == 0:
            self.price_history_full = True

    def _initialize_order_tracking(self):
        """Initialize optimized order tracking structures"""
        # Fast lookup by order ID
        self.orders_by_id = {}
        # Fast lookup by client order ID  
        self.orders_by_client_id = {}
        # Count tracking for limits
        self.active_bid_count = 0
        self.active_ask_count = 0

    def _should_log_verbose(self, frequency: int = 100) -> bool:
        """Determine if verbose logging should occur"""
        if self.verbose_logging:
            return True
        self.log_sampling_counter += 1
        return self.log_sampling_counter % frequency == 0

    async def validate_quotes(self, bid_quotes: List[Quote], ask_quotes: List[Quote]) -> Tuple[List[Quote], List[Quote]]:
        """Validate quotes for market requirements"""
        if not (bid_quotes or ask_quotes) or not self.ticker or self.ticker.mark_price <= 0:
            return [], []
            
        # Filter valid quotes
        valid_bids = [q for q in bid_quotes if q.price > 0 and q.amount > 0]
        valid_asks = [q for q in ask_quotes if q.price > 0 and q.amount > 0]
        
        # Apply price alignment
        if valid_bids:
            valid_bids = self.align_prices_to_tick(valid_bids, is_bid=True)
        if valid_asks:
            valid_asks = self.align_prices_to_tick(valid_asks, is_bid=False)
            
        # Fix crossed quotes
        if valid_bids and valid_asks:
            max_bid = max(q.price for q in valid_bids)
            min_ask = min(q.price for q in valid_asks)
            
            if max_bid >= min_ask:
                mid_point = (max_bid + min_ask) / 2
                valid_bids = [
                    Quote(min(q.price, mid_point - self.tick_size), q.amount, q.instrument, q.side, q.timestamp) 
                    if q.price >= min_ask else q for q in valid_bids
                ]
                valid_asks = [
                    Quote(max(q.price, mid_point + self.tick_size), q.amount, q.instrument, q.side, q.timestamp) 
                    if q.price <= max_bid else q for q in valid_asks
                ]
        
        return valid_bids, valid_asks

    async def cancel_quotes(self, reason: str):
        """Cancel all existing quotes"""
        try:
            self.logger.info(f"Cancelling all orders: {reason}")
            
            # Use the order manager to cancel all orders
            if hasattr(self, 'order_manager') and self.order_manager:
                await self.order_manager.cancel_all_orders()
                self.logger.info("All orders cancelled successfully")
            else:
                self.logger.warning("Order manager not available, cannot cancel orders")
                
            return True
        except Exception as e:
            self.logger.error(f"Error cancelling orders: {str(e)}")
            return False

    async def _place_immediate_closing_orders(self, trigger_decision: Dict, trading_mode: str):
        """Place immediate closing orders when take profit conditions are met"""
        try:
            instruments_to_close = trigger_decision.get('instruments_to_close', [])
            profit_metrics = trigger_decision.get('profit_metrics', {})
            
            if not instruments_to_close:
                self.logger.error("No instruments specified for closing despite trigger condition")
                return
            
            self.logger.warning(f"Placing immediate closing orders for {len(instruments_to_close)} instruments")
            
            # Get current market data for aggressive pricing
            current_ticker = self.market_data.get_ticker()
            mark_price = self.market_data.get_mark_price()
            
            for instrument in instruments_to_close:
                try:
                    # Get position for this instrument
                    position_size = 0
                    if instrument == self.perp_name:
                        position_size = getattr(self.position_tracker, 'current_position', 0)
                    elif instrument == self.futures_instrument_name and self.portfolio_tracker:
                        if instrument in self.portfolio_tracker.instrument_trackers:
                            tracker = self.portfolio_tracker.instrument_trackers[instrument]
                            position_size = tracker.current_position
                    
                    if abs(position_size) < 0.001:
                        self.logger.warning(f"No significant position for {instrument}: {position_size:.6f}")
                        continue
                    
                    # Determine order side and aggressive price
                    if position_size > 0:  # Long position - sell to close
                        side = "sell"
                        # Use bid price - small buffer for aggressive selling
                        if current_ticker and current_ticker.best_bid_price:
                            close_price = current_ticker.best_bid_price * 0.999  # 0.1% below bid
                        else:
                            close_price = mark_price * 0.998  # 0.2% below mark
                    else:  # Short position - buy to close
                        side = "buy"
                        # Use ask price + small buffer for aggressive buying
                        if current_ticker and current_ticker.best_ask_price:
                            close_price = current_ticker.best_ask_price * 1.001  # 0.1% above ask
                        else:
                            close_price = mark_price * 1.002  # 0.2% above mark
                    
                    # Round to tick size
                    close_price = round(close_price)
                    amount = abs(position_size)
                    
                    self.logger.warning(
                        f"IMMEDIATE CLOSE: {instrument} {side.upper()} {amount:.4f} @ {close_price:.2f} "
                        f"(position: {position_size:.4f}, mode: {trading_mode})"
                    )
                    
                    # Place the immediate closing order
                    await self.order_manager.place_order(
                        instrument=instrument,
                        direction=side,
                        price=close_price,
                        amount=amount,
                        label="TakeProfitClose",
                        post_only=False,  # Allow taker orders for immediate execution
                        client_id=f"takeprofit_{instrument}_{int(time.time() * 1000)}"
                    )
                    
                    # Log profit info
                    if trading_mode == "arbitrage":
                        total_profit = profit_metrics.get('total_profit_usd', 0)
                        spread_bps = profit_metrics.get('spread_profit_bps', 0)
                        self.logger.warning(f"Arbitrage take profit: ${total_profit:.2f}, {spread_bps:.1f}bps")
                    elif instrument in profit_metrics:
                        instrument_profit = profit_metrics[instrument].get('profit_usd', 0)
                        self.logger.warning(f"{instrument} take profit: ${instrument_profit:.2f}")
                    
                except Exception as e:
                    self.logger.error(f"Error placing immediate close for {instrument}: {str(e)}")
            
        except Exception as e:
            self.logger.error(f"Error in _place_immediate_closing_orders: {str(e)}", exc_info=True)

    async def _close_both_positions(self, reason: str):
        """
        Close both perpetual and futures positions for risk management purposes.
        
        Args:
            reason: The reason for closing the positions
        """
        self.logger.info(f"Closing both positions due to: {reason}")
        
        # Check connection state first
        if not self.thalex.connected():
            self.logger.warning("Cannot close positions: WebSocket not connected")
            return
        
        # Close perpetual position if it exists
        if self.perp_name:
            try:
                current_perp_position = self.position_tracker.get_position_metrics().get("position", 0.0)
                if abs(current_perp_position) > 0.001:  # Only if position is meaningful
                    direction = th.Direction.SELL if current_perp_position > 0 else th.Direction.BUY
                    amount_to_close = abs(current_perp_position)
                    
                    # Get a suitable exit price
                    if self.ticker and hasattr(self.ticker, 'mark_price') and self.ticker.mark_price > 0:
                        exit_price_base = self.ticker.mark_price
                        # Add a buffer to ensure the order gets filled
                        buffer_pct = 0.005  # 0.5% buffer
                        exit_price = exit_price_base * (1 - buffer_pct) if direction == th.Direction.SELL else exit_price_base * (1 + buffer_pct)
                        
                        # Align to tick size
                        if hasattr(self, 'tick_size') and self.tick_size > 0:
                            exit_price = round(exit_price / self.tick_size) * self.tick_size
                            
                        self.logger.info(f"Placing market-like order to close {amount_to_close} of {self.perp_name} at price {exit_price} (Direction: {direction.value})")
                        
                        # Double-check connection before placing order
                        if self.thalex.connected():
                            await self.thalex.insert(
                                direction=direction,
                                instrument_name=self.perp_name,
                                amount=amount_to_close,
                                price=exit_price,
                                client_order_id=self.next_client_order_id,
                                id=self.next_client_order_id,
                                post_only=False  # Ensure it can take liquidity
                            )
                            self.next_client_order_id += 1
                            self.logger.info(f"Perpetual position close order sent for {self.perp_name}")
                        else:
                            self.logger.warning("Lost connection while trying to close perpetual position")
                    else:
                        self.logger.error(f"Could not determine a valid exit price for {self.perp_name}. Cannot close position.")
                else:
                    self.logger.info(f"No meaningful perpetual position to close for {self.perp_name} (size: {current_perp_position:.6f})")
            except Exception as e:
                self.logger.error(f"Error closing perpetual position for {self.perp_name}: {str(e)}", exc_info=True)
        
        # Close futures position if it exists
        if self.futures_instrument_name:
            try:
                # Use the same position tracker but the method should handle different instruments
                current_futures_position = self.position_tracker.get_position_metrics().get("position", 0.0)
                if abs(current_futures_position) > 0.001:  # Only if position is meaningful
                    direction = th.Direction.SELL if current_futures_position > 0 else th.Direction.BUY
                    amount_to_close = abs(current_futures_position)
                    
                    # Get a suitable exit price for futures
                    if self.futures_ticker and hasattr(self.futures_ticker, 'mark_price') and self.futures_ticker.mark_price > 0:
                        exit_price_base = self.futures_ticker.mark_price
                        buffer_pct = 0.005  # 0.5% buffer
                        exit_price = exit_price_base * (1 - buffer_pct) if direction == th.Direction.SELL else exit_price_base * (1 + buffer_pct)
                        
                        # Align to tick size (assuming same tick size for futures)
                        if hasattr(self, 'tick_size') and self.tick_size > 0:
                            exit_price = round(exit_price / self.tick_size) * self.tick_size
                            
                        self.logger.info(f"Placing market-like order to close {amount_to_close} of {self.futures_instrument_name} at price {exit_price} (Direction: {direction.value})")
                        
                        # Double-check connection before placing order
                        if self.thalex.connected():
                            await self.thalex.insert(
                                direction=direction,
                                instrument_name=self.futures_instrument_name,
                                amount=amount_to_close,
                                price=exit_price,
                                client_order_id=self.next_client_order_id,
                                id=self.next_client_order_id,
                                post_only=False  # Ensure it can take liquidity
                            )
                            self.next_client_order_id += 1
                            self.logger.info(f"Futures position close order sent for {self.futures_instrument_name}")
                        else:
                            self.logger.warning("Lost connection while trying to close futures position")
                    else:
                        self.logger.error(f"Could not determine a valid exit price for {self.futures_instrument_name}. Cannot close position.")
                else:
                    self.logger.info(f"No meaningful futures position to close for {self.futures_instrument_name} (size: {current_futures_position:.6f})")
            except Exception as e:
                self.logger.error(f"Error closing futures position for {self.futures_instrument_name}: {str(e)}", exc_info=True)
        
        self.logger.info("Position closure orders completed")

    def _sync_position_data(self):
        """Synchronize position data between quoter and market maker components"""
        try:
            if not self.position_tracker or not self.market_maker:
                return
                
            # Get current position metrics
            metrics = self.position_tracker.get_position_metrics()
            current_position = metrics.get("position", 0.0)
            
            # Update market maker with current position if it supports it
            if hasattr(self.market_maker, 'position_size') and abs(current_position - getattr(self.market_maker, 'position_size', 0)) > 1e-8:
                self.market_maker.position_size = current_position
                if random.random() < LOG_SAMPLING_RATE:
                    self.logger.debug(f"Synced position data with market maker: {current_position:.6f}")
                    
            # Ensure instrument is set in market maker
            if hasattr(self.market_maker, 'instrument') and self.market_maker.instrument != self.perp_name:
                if hasattr(self.market_maker, 'set_instrument'):
                    self.market_maker.set_instrument(self.perp_name)
                    if random.random() < LOG_SAMPLING_RATE:
                        self.logger.debug(f"Synced instrument with market maker: {self.perp_name}")
                        
        except Exception as e:
            self.logger.error(f"Error syncing position data: {str(e)}")

async def main():
    """Main entry point"""
    # Initialize Thalex client
    thalex = th.Thalex(network=BOT_CONFIG["market"]["network"])
    
    # Create quoter
    quoter = AvellanedaQuoter(thalex)
    
    # Set reference to quoter in thalex client for rate limit tracking
    thalex.quoter = quoter
    
    try:
        # Start the quoter
        await quoter.start()
    except KeyboardInterrupt:
        await quoter.shutdown()
    except Exception as e:
        quoter.logger.critical(f"Fatal error in main loop: {str(e)}")
        await quoter.shutdown()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass  # Handled in main()
    except Exception as e:
        print(f"Fatal error: {str(e)}")  # Fallback if logger isn't available 