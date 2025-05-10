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
import mmap
import ctypes
import numpy as np
import websockets
from websockets.exceptions import ConnectionClosedError, ConnectionClosedOK
import orjson  # Fast JSON parsing
import threading
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import shared_memory

import thalex as th
from thalex import Network
from thalex_py.Thalex_modular.config.market_config import (
    BOT_CONFIG,
    MARKET_CONFIG, 
    CALL_IDS, 
    RISK_LIMITS,
    TRADING_PARAMS,
    TRADING_CONFIG
)
from thalex_py.Thalex_modular.models.data_models import Ticker, Order, OrderStatus, Quote
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

# Memory-mapped structure for IPC
class SharedMarketData:
    """Memory-mapped structure for sharing market data between processes"""
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
    
    def __init__(self, name="thalex_market_data", size=1024):
        try:
            self.shm = shared_memory.SharedMemory(name=name, create=True, size=size)
        except FileExistsError:
            # Connect to existing shared memory
            self.shm = shared_memory.SharedMemory(name=name, create=False)
        
        # Create numpy array backed by shared memory
        self.np_array = np.ndarray((1,), dtype=np.float64, buffer=self.shm.buf)
        
    def update(self, data_dict):
        """Update shared memory with market data"""
        offset = 0
        for key, value in data_dict.items():
            if key in ['timestamp', 'mid_price', 'best_bid', 'best_ask', 'bid_quantity', 
                      'ask_quantity', 'volatility']:
                self.np_array[offset] = float(value)
                offset += 1
                
    def close(self):
        """Close shared memory"""
        self.shm.close()
        
    def unlink(self):
        """Unlink shared memory"""
        self.shm.unlink()

class AvellanedaQuoter:
    def __init__(self, thalex: th.Thalex):
        """Initialize the Avellaneda-Stoikov market maker"""
        # Core dependencies
        self.thalex = thalex
        self.logger = LoggerFactory.configure_component_logger(
            "avellaneda_quoter",
            log_file="quoter.log",
            high_frequency=False  # Only market maker needs high frequency logging
        )
        
        # Set up the market maker components
        self.market_maker = AvellanedaMarketMaker(exchange_client=self.thalex)
        self.order_manager = OrderManager(self.thalex)
        self.risk_manager = RiskManager()
        self.performance_monitor = PerformanceMonitor()
        
        # Market data
        self.ticker = None
        self.index = None
        self.perp_name = None
        self.market_data = MarketDataBuffer(
            volatility_window=TRADING_CONFIG["volatility"]["window"],
            capacity=TRADING_CONFIG["volatility"]["min_samples"]
        )
        
        # Portfolio data
        self.portfolio = {}
        
        # Take profit tracking - advanced implementation
        self.take_profit_levels_executed = set()
        self.highest_profit_levels = {}
        self.last_take_profit_check = 0
        self.take_profit_cooldown = 60  # 60 seconds between take profit checks
        self.take_profit_orders = {}  # Track take profit orders
        self.take_profit_order_id = None  # Track take profit order ID
        
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
        
        # Quote storage and metrics - use NumPy arrays for better performance
        self.price_history = np.zeros(100)  # Store last 100 prices
        self.price_history_idx = 0
        self.price_history_full = False
        self.current_quotes = ([], [])  # (bid_quotes, ask_quotes)
        
        # Order management
        self.max_orders_per_side = 5  # Maximum orders per side
        self.max_total_orders = 16  # Maximum total orders
        
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
        
        # Create object pools for frequently created objects
        # Create a factory function to ensure proper Order initialization 
        def _create_empty_order():
            return Order(
                id=0,
                price=0.0,
                amount=0.0,
                status=OrderStatus.PENDING,
                direction=None
            )

        self.order_pool = ObjectPool(lambda: Order(
            id=0,
            price=0.0,
            amount=0.0,
            status=OrderStatus.PENDING,
            direction=None
        ), size=50)
        self.ticker_pool = ObjectPool(lambda: Ticker({
            "mark_price": 0.0,
            "best_bid_price": 0.0,
            "best_ask_price": 0.0,
            "index": 0.0,
            "mark_timestamp": time.time(),
            "timestamp": time.time()  # Add timestamp parameter
        }), size=10)
        
        # Initialize quote pool
        self.quote_pool = ObjectPool(lambda: Quote(
            price=0.0,
            amount=0.0,
            instrument="",
            side="",
            timestamp=time.time()
        ), size=100)
        
        # Initialize instrument data cache
        self.instrument_data_cache = {}
        
        # Create shared memory for IPC
        try:
            self.shared_market_data = SharedMarketData(name="thalex_market_data", size=1024)
        except Exception as e:
            self.logger.warning(f"Failed to create shared memory: {str(e)}")
            self.shared_market_data = None
        
        # Performance tracing
        self.performance_tracer = PerformanceTracer()
        
        self.logger.info("Avellaneda quoter initialized with HFT optimizations")
        
    async def start(self):
        """Start the quoter"""
        try:
            self.logger.info("Starting Avellaneda quoter...")
            
            # Create performance tracer
            self.performance_tracer = PerformanceTracer()
            
            # Initialize setup completion event
            self.setup_complete = asyncio.Event()
            
            # Create condition variable for quote updates
            self.quote_cv = asyncio.Condition()
            self.condition_met = False
            
            # Initialize the Thalex client connection
            self.logger.info("Connecting to Thalex API...")
            await self.thalex.connect()
            
            if not self.thalex.connected():
                self.logger.error("Failed to connect to Thalex API")
                return False
                
            self.logger.info("Successfully connected to Thalex API")
            
            # Set up instrument data with timeout
            self.logger.info("Setting up instrument data...")
            try:
                # Wait up to 30 seconds for instrument data
                await asyncio.wait_for(self.await_instruments(), timeout=30.0)
                self.logger.info("Instrument data retrieved successfully")
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
                key_id = key_ids[network]
                private_key = private_keys[network]
                
                # Login to the API
                await self.thalex.login(key_id, private_key, id=CALL_IDS["login"])
                self.logger.info("Login successful")
                
                # Configure cancel on disconnect
                await self.thalex.set_cancel_on_disconnect(6, id=CALL_IDS["set_cod"])
                self.logger.info("Cancel on disconnect configured")
            except Exception as e:
                self.logger.error(f"Login failed: {str(e)}")
                return False
            
            # Continue with the rest of the existing code
            
            # Set the perpetual name from config
            underlying = MARKET_CONFIG.get("underlying", "BTC")
            self.perp_name = underlying.split("USD")[0] + "-PERPETUAL"
            self.logger.info(f"Setting instrument name to {self.perp_name}")
            
            # Set the position delta for position tracking
            if hasattr(self, 'instrument_data') and 'contractSize' in self.instrument_data:
                contract_size = self.instrument_data['contractSize']
                self.logger.info(f"Setting contract size to {contract_size}")
                self.contract_size = float(contract_size)
            
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
                'quote': asyncio.create_task(self.quote_task())
            })
            
            # Extra debug logging to track initialization
            self.logger.info(f"All {len(self.tasks)} tasks created successfully")
            self.logger.info(f"Tasks: {', '.join(self.tasks.keys())}")
            
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
            
            # Close shared memory resources with timeout protection
            try:
                self.logger.info("Cleaning up shared memory resources...")
                if hasattr(self, 'shared_market_data') and self.shared_market_data is not None:
                    try:
                        self.shared_market_data.close()
                        self.logger.info("Shared memory closed")
                    except Exception as se:
                        self.logger.error(f"Error closing shared memory: {str(se)}")
                    
                    try:
                        self.shared_market_data.unlink()
                        self.logger.info("Shared memory unlinked")
                    except FileNotFoundError:
                        self.logger.error(f"Error unlinking shared memory: File not found")
                    except Exception as se:
                        self.logger.error(f"Error unlinking shared memory: {str(se)}")
            except Exception as e:
                self.logger.error(f"Error cleaning up shared memory: {str(e)}")
            
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
                
                # Set perpetual contract details
                self.perp_name = perpetual.get("name") or perpetual.get("asset") or perpetual.get("symbol")
                self.tick_size = float(perpetual.get("tick_size") or perpetual.get("tickSize") or 1.0)
                self.contract_size = float(perpetual.get("contract_size") or perpetual.get("contractSize") or 1.0)
                
                # Pass tick size to market maker
                self.market_maker.set_tick_size(self.tick_size)
                self.order_manager.set_tick_size(self.tick_size)
                
                self.logger.info(f"Found perpetual: {self.perp_name} with tick size {self.tick_size}")
                
                return True
                
            except Exception as e:
                self.logger.error(f"Error retrieving instrument data: {str(e)}")
                raise
                
        except Exception as e:
            self.logger.error(f"Error in await_instruments: {str(e)}")
            raise

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
        backoff_time = 0.1  # Initial backoff time
        max_backoff = 30  # Maximum backoff time in seconds
        consecutive_errors = 0
        max_consecutive_errors = 20  # Maximum consecutive errors before forcing reconnection
        
        # Initialize timestamp for the last received message
        last_message_time = time.time()
        
        self.logger.info("Listen task started - ready to receive messages")
        
        while True:
            try:
                # Wait for connection to be established
                if not self.thalex.connected():
                    self.logger.warning("WebSocket not connected. Attempting to reconnect...")
                    try:
                        # First try to disconnect if there's a stale connection
                        try:
                            if self.thalex.ws is not None:
                                await self.thalex.disconnect()
                                self.logger.info("Closed stale WebSocket connection")
                                await asyncio.sleep(1)  # Brief pause before reconnecting
                        except Exception as disconnect_error:
                            self.logger.error(f"Error closing stale connection: {str(disconnect_error)}")
                        
                        # Now try to connect
                        await self.thalex.connect()
                        
                        # If we got here, connection was successful
                        self.logger.info("Successfully reconnected to WebSocket")
                        
                        # Re-authenticate after reconnection
                        network = MARKET_CONFIG["network"]
                        key_id = key_ids[network]
                        private_key = private_keys[network]
                        
                        try:
                            self.logger.info(f"Re-authenticating with key ID: {key_id}")
                            await self.thalex.login(key_id, private_key, id=CALL_ID_LOGIN)
                            self.logger.info("Re-authentication successful")
                            
                            # Resubscribe to channels
                            await self._subscribe_to_websocket_topics()
                            self.logger.info("Successfully resubscribed to WebSocket topics")
                            
                            # Reset backoff time after successful reconnection and resubscription
                            backoff_time = 0.1
                            consecutive_errors = 0
                        except Exception as auth_error:
                            self.logger.error(f"Failed to re-authenticate: {str(auth_error)}")
                            raise  # Re-raise to trigger the connection retry logic
                    except Exception as reconnect_error:
                        self.logger.error(f"Failed to reconnect: {str(reconnect_error)}")
                        # Use exponential backoff for retries
                        await asyncio.sleep(backoff_time)
                        backoff_time = min(max_backoff, backoff_time * 2)
                        continue
                
                # Check for message timeout
                current_time = time.time()
                if current_time - last_message_time > 60:
                    # If no messages for 60 seconds, log a warning
                    self.logger.warning(f"No messages received for {current_time - last_message_time:.1f} seconds")
                
                # Use the tracer to monitor message processing performance
                with self.performance_tracer.trace("message_processing"):
                    # Get message with timeout protection
                    try:
                        # Set a timeout for the receive operation
                        message = await asyncio.wait_for(self.thalex.receive(), timeout=30.0)
                        # Update the last received message time
                        last_message_time = time.time()
                    except asyncio.TimeoutError:
                        self.logger.warning("WebSocket receive timed out after 30 seconds")
                        consecutive_errors += 1
                        continue
                    except websockets.exceptions.ConnectionClosedOK:
                        self.logger.info("WebSocket connection closed normally")
                        # Mark the connection as closed to trigger reconnection
                        self.thalex.ws = None
                        consecutive_errors += 1
                        await asyncio.sleep(1)  # Brief pause before attempting to reconnect
                        continue
                    except websockets.exceptions.ConnectionClosedError as ws_error:
                        self.logger.error(f"WebSocket connection closed with error: {str(ws_error)}")
                        # Mark the connection as closed to trigger reconnection
                        self.thalex.ws = None
                        consecutive_errors += 1
                        await asyncio.sleep(1)  # Brief pause before attempting to reconnect
                        continue
                    
                    # Reset error counter on successful message
                    consecutive_errors = 0
                    backoff_time = 0.1  # Reset backoff time on success
                    
                    # Process message
                    if message is None:
                        self.logger.warning("Received None message from WebSocket")
                        continue
                    
                    # Add detailed logging for debugging
                    if isinstance(message, str):
                        # Log a sample of the message for debugging
                        msg_sample = message[:100] + "..." if len(message) > 100 else message
                        self.logger.info(f"Received message: {msg_sample}")
                    
                    # Use faster JSON parsing if message is still a string with pre-allocated buffer
                        try:
                            # Check if message buffer is large enough
                            if len(message) < self.message_buffer_size:
                                # Copy message bytes directly into pre-allocated buffer
                                message_bytes = message.encode('utf-8')
                                self.message_buffer[:len(message_bytes)] = message_bytes
                                try:
                                    data = orjson.loads(self.message_view[:len(message_bytes)])
                                except Exception as json_error:
                                    self.logger.error(f"Error parsing JSON with orjson: {str(json_error)}")
                                    # Fall back to standard JSON parsing
                                    data = json.loads(message)
                            else:
                                # Increase buffer size if message is too large
                                self.message_buffer_size = min(self.message_buffer_size * 2, 1048576)
                                self.message_buffer = bytearray(self.message_buffer_size)
                                self.message_view = memoryview(self.message_buffer)
                                self.logger.info(f"Increased message buffer to {self.message_buffer_size} bytes")
                                # Fall back to standard parsing for now
                                data = json.loads(message)
                        except Exception as e:
                            self.logger.error(f"Error parsing message: {str(e)}")
                            # Fall back to standard JSON parsing
                            try:
                                data = json.loads(message)
                            except Exception as json_error:
                                self.logger.error(f"Failed to parse message as JSON: {str(json_error)}")
                                continue  # Skip this message
                    else:
                        data = message
                    
                    # Log summary of received data
                    self.logger.info(f"Received data with keys: {list(data.keys())}")
                    
                    # Handle the new ticker format directly 
                    # Some exchanges send ticker updates in a different format
                    if "type" in data and data["type"] == "ticker" and "data" in data:
                        self.logger.info(f"Detected direct ticker update format")
                        ticker_data = data["data"]
                        await self.handle_ticker_update(ticker_data)
                        continue
                    
                    # Process notification - handle both old and new API formats
                    # New format uses 'channel_name' instead of 'channel'
                    if "notification" in data:
                        # Get the channel name from whichever field is present
                        channel = data.get("channel") or data.get("channel_name") or ""
                        notification = data["notification"]
                        
                        self.logger.info(f"Processing notification from channel: {channel}")
                        await self.handle_notification(channel, notification)
                    # Process response to API call
                    elif "id" in data:
                        cid = data["id"]
                        if "result" in data:
                            await self.handle_result(data["result"], cid)
                        elif "error" in data:
                            await self.handle_error(data["error"], cid)
                    # Direct ticker data pattern
                    elif "ticker" in data:
                        self.logger.info("Detected ticker data pattern")
                        await self.handle_ticker_update(data["ticker"])
                    # Other direct data structures
                    elif "instrument" in data and "price" in data:
                        self.logger.info("Detected direct market data format")
                        await self.handle_ticker_update(data)
                    else:
                        self.logger.warning(f"Received unrecognized message format with keys: {data.keys()}")
                        self.logger.warning(f"Received unrecognized message format: {data.keys()}")
                
            except asyncio.CancelledError:
                self.logger.info("Listen task cancelled")
                break
            except Exception as e:
                consecutive_errors += 1
                self.logger.error(f"Error in listen task: {str(e)}", exc_info=True)
                
                # Use exponential backoff strategy
                await asyncio.sleep(backoff_time)
                backoff_time = min(max_backoff, backoff_time * 2)
                
                # Circuit breaker - if too many errors, force reconnection
                if consecutive_errors > max_consecutive_errors:
                    self.logger.warning(f"Circuit breaker activated after {consecutive_errors} errors. Forcing reconnection.")
                    try:
                        # Close the connection if it's still open
                        if self.thalex.ws is not None:
                            await self.thalex.disconnect()
                    except Exception:
                        pass  # Ignore errors during disconnect
                    
                    # Mark the connection as closed to trigger reconnection on the next iteration
                    self.thalex.ws = None
                    
                    # Take a longer break before reconnecting
                    await asyncio.sleep(max_backoff)
                    
                    # Reset error counter
                    consecutive_errors = 0

    # Zero-copy message handling for high-frequency market data
    async def handle_ticker_update(self, ticker_data: Dict):
        """
        Handle ticker updates with zero-copy optimizations.
        
        Args:
            ticker_data: Dictionary containing ticker update data
        """
        with self.performance_tracer.trace("ticker_processing"):
            try:
                # Add debug logging
                self.logger.info(f"Received ticker update: {ticker_data.keys()}")
                
                # Extract instrument ID - handle different possible data formats
                instrument_id = (
                    ticker_data.get("instrumentId") or 
                    ticker_data.get("instrument_id") or 
                    ticker_data.get("name")
                )
                
                # Handle case where ticker might be nested
                if not instrument_id and "instrument" in ticker_data:
                    instrument_data = ticker_data.get("instrument", {})
                    if isinstance(instrument_data, dict):
                        instrument_id = instrument_data.get("name") or instrument_data.get("instrumentId")
                
                # Debug log
                if instrument_id:
                    self.logger.info(f"Processing ticker for instrument: {instrument_id}")
                else:
                    self.logger.warning(f"Could not extract instrument ID from ticker data: {ticker_data}")
                    # Try to infer from our expected instrument
                    instrument_id = self.perp_name
                    self.logger.info(f"Using default instrument ID: {instrument_id}")
                
                # Only process relevant instrument updates
                if instrument_id and self.perp_name and instrument_id == self.perp_name:
                    # Extract price fields with multiple possible keys
                    mark_price = (
                        ticker_data.get("lastPrice") or
                        ticker_data.get("mark_price") or
                        ticker_data.get("markPrice") or
                        ticker_data.get("last") or
                        0.0
                    )
                    
                    best_bid_price = (
                        ticker_data.get("bestBidPrice") or
                        ticker_data.get("best_bid_price") or
                        ticker_data.get("bid") or
                        mark_price * 0.999  # fallback - slightly below mark
                    )
                    
                    best_ask_price = (
                        ticker_data.get("bestAskPrice") or
                        ticker_data.get("best_ask_price") or
                        ticker_data.get("ask") or
                        mark_price * 1.001  # fallback - slightly above mark
                    )
                    
                    index_price = (
                        ticker_data.get("indexPrice") or
                        ticker_data.get("index_price") or
                        ticker_data.get("index") or
                        mark_price  # fallback to mark price
                    )
                    
                    # Debug pricing info
                    self.logger.info(f"Ticker prices: mark={mark_price}, bid={best_bid_price}, ask={best_ask_price}")
                    
                    # Create a proper Ticker object
                    ticker_dict = {
                        "mark_price": float(mark_price),
                        "best_bid_price": float(best_bid_price),
                        "best_ask_price": float(best_ask_price),
                        "mark_timestamp": time.time(),
                        "index": float(index_price),
                    }
                    
                    ticker = Ticker(ticker_dict)
                    
                    # Add computed fields
                    ticker.best_bid_amount = float(ticker_data.get("bestBidAmount") or ticker_data.get("best_bid_amount") or 1.0)
                    ticker.best_ask_amount = float(ticker_data.get("bestAskAmount") or ticker_data.get("best_ask_amount") or 1.0)
                    ticker.instrument_id = instrument_id
                    
                    # Update the ticker
                    self.ticker = ticker
                    self.last_ticker_time = time.time()
                    
                    # Calculate mid price for convenience
                    mid_price = (ticker.best_bid_price + ticker.best_ask_price) / 2 if ticker.best_bid_price > 0 and ticker.best_ask_price > 0 else ticker.mark_price
                    
                    # Update market data buffer
                    if mid_price > 0:
                        if hasattr(self, 'market_data') and hasattr(self.market_data, 'prices'):
                            self.market_data.prices.append(mid_price)
                        else:
                            self.logger.warning("Market data buffer not properly initialized")
                    
                    # Update ring buffer for market impact metrics
                    if hasattr(self, 'price_history'):
                        self.price_history[self.price_history_idx] = mid_price
                        self.price_history_idx = (self.price_history_idx + 1) % len(self.price_history)
                        if not self.price_history_full and self.price_history_idx == 0:
                            self.price_history_full = True
                    
                    # Log ticker update and signal for quote update
                    self.logger.info(f"Updated ticker: {instrument_id}, mark: {ticker.mark_price}, bid: {ticker.best_bid_price}, ask: {ticker.best_ask_price}")
                    
                    # Signal quote update if needed
                    await self.check_market_conditions_for_quote()
                    
                    # Update shared memory for IPC with other processes
                    if hasattr(self, 'shared_market_data') and self.shared_market_data is not None:
                        try:
                            # Create a dictionary with data matching SharedMarketData fields
                            market_data_dict = {
                                'timestamp': time.time(),
                                'mid_price': mid_price,
                                'best_bid': ticker.best_bid_price,
                                'best_ask': ticker.best_ask_price,
                                'bid_quantity': ticker.best_bid_amount,
                                'ask_quantity': ticker.best_ask_amount,
                                'volatility': 0.0,  # Calculate volatility if needed
                                'status': 1 # 1 for active/updated
                            }
                            
                            # Write directly to the shared memory buffer using ctypes structure
                            # Ensure the target buffer exists and is accessible
                            if self.shared_market_data.shm.buf:
                                # Create a ctypes structure instance from the shared memory buffer
                                data_struct = SharedMarketData.from_buffer(self.shared_market_data.shm.buf)
                                
                                # Update fields - handle potential type errors
                                for field_name, field_type in SharedMarketData._fields_:
                                    value = market_data_dict.get(field_name)
                                    if value is not None:
                                        try:
                                            # Cast value to the expected ctypes type
                                            setattr(data_struct, field_name, field_type(value))
                                        except (TypeError, ValueError) as type_err:
                                            self.logger.error(f"Type error setting shared memory field '{field_name}': {type_err}. Value: {value}")
                                    else:
                                         self.logger.warning(f"Missing value for shared memory field: {field_name}")

                            # # Old incorrect logic - removed:
                            # # Update only the first array element to avoid index errors
                            # self.shared_market_data.np_array[0] = market_data_dict.get('timestamp', 0.0)
                        except AttributeError as ae:
                             # Handle cases where from_buffer might not be available directly on the class
                            if "'type' object has no attribute 'from_buffer'" in str(ae):
                                 self.logger.error("Cannot use from_buffer directly on SharedMarketData class. Need instance or different approach.")
                            else:
                                 self.logger.error(f"Attribute error updating shared memory: {str(ae)}")
                        except Exception as shared_memory_error:
                            # More specific error handling to identify the issue
                            self.logger.error(f"Error updating shared memory: {str(shared_memory_error)}", exc_info=True)
                else:
                    if instrument_id:
                        self.logger.info(f"Skipping ticker for irrelevant instrument: {instrument_id}, expected: {self.perp_name}")
                    else:
                        self.logger.warning("Received ticker without instrument ID")
            
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
            order.id = data.get("id", 0)
            order.price = float(data.get("limitPrice", 0))
            order.amount = float(data.get("amount", 0))
            
            # Handle status safely - converting string to enum
            status_str = data.get("status", "PENDING")
            try:
                order.status = OrderStatus(status_str)
            except (ValueError, KeyError):
                order.status = OrderStatus.PENDING
                
            order.direction = data.get("side", None)
            
            # Store additional data as attributes (optional)
            # These won't be part of the core Order class but will be accessible
            order.client_id = data.get("clientId")
            order.instrument_id = data.get("instrumentId")
            order.type = data.get("type")
            order.filled = float(data.get("filled", 0))
            order.timestamp = time.time()
            
            return order
            
        except Exception as e:
            self.logger.error(f"Error creating order from data: {str(e)}")
            # Fall back to creating a new order if pool fails
            order = Order(
                id=data.get("id", 0),
                price=float(data.get("limitPrice", 0)),
                amount=float(data.get("amount", 0)),
                status=OrderStatus.PENDING,  # Use safe default
                direction=data.get("side", None)
            )
            # Add additional attributes
            order.client_id = data.get("clientId")
            order.instrument_id = data.get("instrumentId")
            order.type = data.get("type")
            order.filled = float(data.get("filled", 0))
            order.timestamp = time.time()
            
            return order

    async def update_quotes(self, price_data: Dict, market_conditions: Dict):
        """Generate new quotes based on current market data"""
        with self.performance_tracer.trace("quote_generation"):
            try:
                if price_data is None and self.ticker:
                    # Create price data from ticker if not provided
                    price_data = {
                        "mid_price": (self.ticker.best_bid_price + self.ticker.best_ask_price) / 2
                        if hasattr(self.ticker, 'best_bid_price') and hasattr(self.ticker, 'best_ask_price')
                        else self.ticker.mark_price
                    }
                    self.logger.info(f"Created price data from ticker: mid_price={price_data['mid_price']}")
                
                instrument_id = self.perp_name
                price = price_data.get("mid_price", 0) if price_data else 0
                
                self.logger.info(f"Updating quotes for {instrument_id} at price {price}")
                
                # Sanity check
                if price <= 0:
                    self.logger.warning("Invalid price for quote generation, using fallback")
                    if self.ticker and self.ticker.mark_price > 0:
                        price = self.ticker.mark_price
                    else:
                        self.logger.error("No valid price available for quote generation")
                        return
                
                if not self.quoting_enabled:
                    self.logger.info("Quoting is disabled, skipping quote update")
                    return
                
                if self.cooldown_active and time.time() < self.cooldown_until:
                    self.logger.info(f"Cooldown active until {self.cooldown_until}, skipping quote update")
                    return
                
                # Generate quotes using the Avellaneda market maker or fallback to simple method
                bid_quotes = []
                ask_quotes = []
                
                try:
                    if hasattr(self, 'market_maker') and self.market_maker:
                        self.logger.info("Generating quotes using Avellaneda market maker")
                        # Use thread pool for CPU-intensive calculations if available
                        if hasattr(self, 'thread_pool') and self.thread_pool:
                            # Create a Ticker object from price data
                            ticker_obj = Ticker({
                                "mark_price": price,
                                "best_bid_price": price * 0.999,  # Approximate bid as slightly below price
                                "best_ask_price": price * 1.001,   # Approximate ask as slightly above price
                                "mark_timestamp": time.time(),
                                "index": price
                            })
                            
                            # Call generate_quotes with proper parameters
                            bid_quotes, ask_quotes = await asyncio.get_event_loop().run_in_executor(
                                self.thread_pool,
                                self.market_maker.generate_quotes,
                                ticker_obj,
                                market_conditions,
                                self.max_orders_per_side
                            )
                        else:
                            # Fallback if thread pool not available
                            # Create a Ticker object from price data
                            ticker_obj = Ticker({
                                "mark_price": price,
                                "best_bid_price": price * 0.999,  # Approximate bid as slightly below price
                                "best_ask_price": price * 1.001,   # Approximate ask as slightly above price
                                "mark_timestamp": time.time(),
                                "index": price
                            })
                            
                            # Call generate_quotes with proper parameters
                            bid_quotes, ask_quotes = self.market_maker.generate_quotes(
                                ticker_obj, market_conditions, self.max_orders_per_side
                            )
                    else:
                        # Fallback to simple quote generation
                        self.logger.warning("Market maker not available, using fallback quote generation")
                        bid_quotes, ask_quotes = self._generate_simple_quotes(price)
                except Exception as e:
                    self.logger.error(f"Error generating quotes with market maker: {str(e)}")
                    self.logger.warning("Falling back to simple quote generation")
                    bid_quotes, ask_quotes = self._generate_simple_quotes(price)
                
                self.logger.info(f"Generated {len(bid_quotes)} bid quotes and {len(ask_quotes)} ask quotes")
                
                # Apply tick size alignment
                bid_quotes = self.align_prices_to_tick(bid_quotes, is_bid=True)
                ask_quotes = self.align_prices_to_tick(ask_quotes, is_bid=False)
                
                # Validate quotes
                bid_quotes, ask_quotes = await self.validate_quotes(bid_quotes, ask_quotes)
                
                self.current_quotes = (bid_quotes, ask_quotes)
                
                # Place the quotes now rather than signaling
                self.logger.info("Directly placing quotes instead of signaling")
                await self.place_quotes(bid_quotes, ask_quotes)
                
                # Also signal condition variable for quote task to handle fallback
                async with self.quote_cv:
                    self.condition_met = True
                    self.quote_cv.notify()
                    
                self.last_quote_update_time = time.time()
                
            except Exception as e:
                self.logger.error(f"Error updating quotes: {str(e)}")
                # Return quotes to the pool on error
                if 'bid_quotes' in locals() and 'ask_quotes' in locals():
                    for quote in bid_quotes + ask_quotes:
                        if hasattr(self, 'quote_pool'):
                            self.quote_pool.put(quote)
    
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
        
        # Create quote objects for bids
        for bid_price in bid_prices:
            quote = self.quote_pool.get() if hasattr(self, 'quote_pool') else Quote(
                price=bid_price,
                amount=TRADING_PARAMS.get("order_size", 0.01),
                instrument=instrument_id,
                side="BUY",
                timestamp=time.time()
            )
            if hasattr(self, 'quote_pool'):
                quote.price = bid_price
                quote.amount = TRADING_PARAMS.get("order_size", 0.01)
                quote.instrument = instrument_id
                quote.side = "BUY"
                quote.timestamp = time.time()
            bid_quotes.append(quote)
            
        # Create quote objects for asks
        for ask_price in ask_prices:
            quote = self.quote_pool.get() if hasattr(self, 'quote_pool') else Quote(
                price=ask_price,
                amount=TRADING_PARAMS.get("order_size", 0.01),
                instrument=instrument_id,
                side="SELL",
                timestamp=time.time()
            )
            if hasattr(self, 'quote_pool'):
                quote.price = ask_price
                quote.amount = TRADING_PARAMS.get("order_size", 0.01)
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
            
            self.logger.info(f"Handling notification from channel: {channel}")
            
            # Handle different channel types - Support both old and new formats
            if channel.startswith("ticker."):
                self.logger.info(f"Processing ticker update for {channel}")
                await self.handle_ticker_update(notification)
            elif channel.startswith("price_index."):
                self.logger.info(f"Processing price index update")
                await self.handle_index_update(notification)
            elif channel == "session.orders":
                self.logger.info(f"Processing order update")
                if isinstance(notification, list):
                    for order_data in notification:
                        await self.handle_order_update(order_data)
                else:
                    await self.handle_order_update(notification)
            elif channel == "account.portfolio":
                self.logger.info(f"Processing portfolio update")
                await self.handle_portfolio_update(notification)
            elif channel == "account.trade_history":
                self.logger.info(f"Processing trade update")
                await self.handle_trade_update(notification)
            else:
                self.logger.warning(f"Notification for unknown channel: {channel}")
        except Exception as e:
            self.logger.error(f"Error processing notification: {str(e)}", exc_info=True)

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
                # Log the fill with improved formatting
                self.logger.info(
                    f"Order filled: {order.id} - {order.direction.upper()} {order.amount:.4f} @ {order.price:.2f}"
                )
                
                # Update market maker with fill information
                self.market_maker.on_order_filled(
                    order_id=order.id,
                    fill_price=order.price,
                    fill_size=order.amount,
                    is_buy=order.direction == "buy"
                )
                
                # Update risk manager with position information directly from fill
                if hasattr(self.risk_manager, 'update_position_fill'):
                    await self.risk_manager.update_position_fill(
                        direction=order.direction,
                        price=order.price,
                        size=order.amount,
                        timestamp=time.time()
                    )
                
                # Update VAMP with order fill data if supported
                if hasattr(self.market_maker, 'update_vamp'):
                    self.market_maker.update_vamp(
                        price=order.price,
                        volume=order.amount,
                        is_buy=order.direction == "buy",
                        is_aggressive=False  # Assuming most of our fills are passive
                    )
                
                # Force quote update after fills based on size threshold
                significant_fill_threshold = TRADING_CONFIG["avellaneda"].get("significant_fill_threshold", 0.1)
                
                # Create comprehensive market conditions
                market_state = self.market_data.get_market_state()
                market_conditions = {
                    "volatility": market_state["yz_volatility"] if "yz_volatility" in market_state and not np.isnan(market_state["yz_volatility"]) else 
                                 (market_state["volatility"] if not np.isnan(market_state["volatility"]) else TRADING_CONFIG["avellaneda"]["fixed_volatility"]),
                    "market_impact": market_state.get("market_impact", 0.0),
                    # Add higher impact for a recent fill
                    "fill_impact": 0.5 if order.amount >= significant_fill_threshold else 0.2  # Higher impact for significant fills
                }
                
                # Try to use calculated volatility if available and valid
                if "yz_volatility" in market_state and market_state["yz_volatility"] is not None and not np.isnan(market_state["yz_volatility"]):
                    market_conditions["volatility"] = market_state["yz_volatility"]
                elif "volatility" in market_state and market_state["volatility"] is not None and not np.isnan(market_state["volatility"]):
                    market_conditions["volatility"] = market_state["volatility"]
                
                # Try to use calculated market impact if available
                if "market_impact" in market_state and market_state["market_impact"] is not None and not np.isnan(market_state["market_impact"]):
                    market_conditions["market_impact"] = market_state["market_impact"]
                
                if self.ticker:
                    if order.amount >= significant_fill_threshold:
                        # Create task with high priority for immediate update for significant fills
                        self.logger.info(f"Scheduling immediate quote update after significant fill of {order.amount:.4f}")
                        asyncio.create_task(self.update_quotes(None, market_conditions))
                    else:
                        # Still update quotes for smaller fills, but with less urgency
                        self.logger.info(f"Scheduling quote update after fill of {order.amount:.4f}")
                        # Notify quote task instead of creating a separate task
                        async with self.quote_cv:
                            self.condition_met = True
                            self.quote_cv.notify()
                else:
                    self.logger.warning("Cannot update quotes after fill: No ticker data available")
                
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
                    
                # Ensure position is a dictionary before using .get()
                if not isinstance(position, dict):
                    self.logger.warning(f"Received position in unexpected format: {type(position)}, value: {position}")
                    continue
                    
                instrument = position.get("instrument_name", "")
                
                # Only process our target instrument
                if self.perp_name and instrument == self.perp_name:
                    # Safely extract position size only
                    size_value = position.get('size', 0.0)
                    size = float(size_value) if size_value is not None else 0.0
                    
                    # Safely extract average price if available
                    avg_price_value = position.get('average_price', 0.0)
                    avg_price = float(avg_price_value) if avg_price_value is not None else 0.0
                    
                    # Update position tracking
                    position_size = size
                    
                    # Ensure instrument is set in market maker before updating position
                    if not hasattr(self.market_maker, 'instrument') or self.market_maker.instrument is None or self.market_maker.instrument != self.perp_name:
                        self.logger.warning(f"Instrument not set in market maker during portfolio update, setting to {self.perp_name}")
                        if hasattr(self.market_maker, 'set_instrument'):
                            self.market_maker.set_instrument(self.perp_name)
                    
                    # Log position with average price information if available
                    if avg_price > 0:
                        self.logger.info(f"Updated position for {instrument}: Size={size:.8f}, Avg Price={avg_price:.2f}")
                    else:
                        self.logger.info(f"Updated position size for {instrument}: Size={size:.8f}")
                    
            # Update market maker with position size only
            if hasattr(self.market_maker, 'update_position_size'):
                self.market_maker.update_position_size(position_size)
                
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
            
            # Check validity
            if price <= 0 or amount <= 0 or not direction or not instrument:
                self.logger.warning(f"Received invalid trade data: {trade_data}")
                return
            
            # Only process if for our target instrument
            if instrument != self.perp_name:
                return
            
            # Convert direction to is_buy flag
            is_buy = direction.lower() == 'buy'
            
            # Update market maker's market data
            should_update_grid = self.market_maker.update_market_data(price, amount, is_buy, is_trade=True)
            
            # Update market data buffer with trade
            self.market_data.add_trade(price, amount, is_buy)
            
            # If the volume candle buffer indicates we should update quotes, do so
            if should_update_grid:
                # Get current market conditions
                market_conditions = self.get_market_conditions()
                
                # Get current ticker data
                ticker = self.get_current_ticker()
                
                if ticker:
                    # Generate new quotes based on updated parameters
                    bid_quotes, ask_quotes = self.market_maker.generate_quotes(ticker, market_conditions)
                    
                    # Place the quotes
                    if bid_quotes or ask_quotes:
                        await self.place_quotes(bid_quotes, ask_quotes)
                        self.logger.info("Updated quotes based on trade-triggered volume candle prediction")
            
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
        """Place quotes using limit orders
        
        Args:
            bid_quotes: List of bid quotes to place
            ask_quotes: List of ask quotes to place
        """
        try:
            if not self.quoting_enabled:
                self.logger.debug("Quoting is disabled, skipping quote placement")
                return
                
            # Check cooldown
            current_time = time.time()
            if self.cooldown_active and current_time < self.cooldown_until:
                self.logger.debug(f"Cooldown active for {self.cooldown_until - current_time:.2f}s more, skipping quote placement")
                return
            
            # Enforce max order levels from config
            max_levels = TRADING_CONFIG["quoting"].get("levels", 6)
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
                await self.cancel_quotes("Refreshing order book")
                # Brief pause to allow cancellations to process
                await asyncio.sleep(0.5)
                
                # Verify orders were cancelled
                if len(self.order_manager.active_bids) > 0 or len(self.order_manager.active_asks) > 0:
                    self.logger.error(f"Failed to cancel all orders. Still have {len(self.order_manager.active_bids)} bids and {len(self.order_manager.active_asks)} asks")
                    # Force another cancellation
                    await self.cancel_quotes("Emergency cancellation")
                    await asyncio.sleep(1.0)  # Longer wait time
            
            # Store current quotes before sending to properly track what was sent
            self.current_quotes = (bid_quotes, ask_quotes)
            
            # Reset tracking variables to ensure clean state
            placed_bids = 0
            placed_asks = 0
            
            # --- BEGIN INVENTORY CHECK FOR ONE-SIDED QUOTING ---
            current_inventory = 0.0
            if hasattr(self.market_maker, 'position_size'):
                current_inventory = self.market_maker.position_size
            else:
                self.logger.warning("Market maker object or 'position_size' attribute not found. Inventory check will use 0 inventory.")

            max_inventory_deviation = TRADING_PARAMS.get("max_allowed_inventory_deviation", float('inf'))
            if not isinstance(max_inventory_deviation, (int, float)) or max_inventory_deviation < 0:
                self.logger.warning(f"Invalid 'max_allowed_inventory_deviation' ({max_inventory_deviation}). Using infinity (feature disabled).")
                max_inventory_deviation = float('inf')
            
            can_place_bids_inventory_wise = True
            if current_inventory > max_inventory_deviation:  # Inventory is too long
                can_place_bids_inventory_wise = False
                self.logger.info(
                    f"Inventory ({current_inventory:.4f}) exceeds +max_allowed_inventory_deviation ({max_inventory_deviation:.4f}). "
                    f"Temporarily stopping new bid orders."
                )
            
            can_place_asks_inventory_wise = True
            if current_inventory < -max_inventory_deviation:  # Inventory is too short
                can_place_asks_inventory_wise = False
                self.logger.info(
                    f"Inventory ({current_inventory:.4f}) is less than -max_allowed_inventory_deviation ({-max_inventory_deviation:.4f}). "
                    f"Temporarily stopping new ask orders."
                )
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
                        self.logger.info(f"Placed bid: {quote.amount:.3f}@{quote.price:.2f}")
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
                        self.logger.info(f"Placed ask: {quote.amount:.3f}@{quote.price:.2f}")
                        placed_asks += 1
                    except Exception as e:
                        self.logger.error(f"Error placing ask: {str(e)}")
                else:
                    if ask_quotes: # Only log if there were asks to place
                        self.logger.info("Skipping all ask placements due to inventory limit.")
            
            # Record successful quote placement
            self.last_quote_time = time.time()
            
            # Log quote summary
            self.logger.info(f"Sending limit orders: {placed_bids} bids, {placed_asks} asks")
            self.logger.info(f"Limit orders sent: {placed_bids} bids, {placed_asks} asks")
                
            # Verify order counts after placement to detect any potential tracking issues
            await asyncio.sleep(0.5)  # Brief pause to allow orders to be processed
            current_active_bids = len(self.order_manager.active_bids)
            current_active_asks = len(self.order_manager.active_asks)
            
            if current_active_bids > max_levels or current_active_asks > max_levels:
                self.logger.error(f"Order tracking issue detected! Active orders exceed max levels: {current_active_bids} bids, {current_active_asks} asks (max: {max_levels})")
                self.logger.info(f"Initiating emergency cancellation to restore order consistency")
                await self.cancel_quotes("Order tracking issue detected")
            
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
        """Get the current market conditions from market data buffer"""
        # Check if cached conditions exist and are recent (within last 500ms)
        current_time = time.time()
        if hasattr(self, '_cached_market_conditions') and current_time - getattr(self, '_cached_market_time', 0) < 0.5:
            return self._cached_market_conditions
            
        market_state = self.market_data.get_market_state()
        
        # Get volatility - prefer Yang-Zhang volatility when available
        if "yz_volatility" in market_state and not np.isnan(market_state["yz_volatility"]):
            volatility = market_state["yz_volatility"]
        else:
            volatility = market_state["volatility"] if not np.isnan(market_state["volatility"]) else TRADING_CONFIG["avellaneda"]["fixed_volatility"]
        
        conditions = {
            "volatility": volatility,
            "market_impact": market_state.get("market_impact", 0.0)
        }
        
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
                        self.logger.warning("WebSocket disconnected, skipping heartbeat and waiting for reconnect")
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
                    metrics = self.market_maker.get_position_metrics()
                    position = metrics.get('position', 0.0) if metrics else 0.0
                    realized_pnl = metrics.get('realized_pnl', 0.0) if metrics else 0.0
                    unrealized_pnl = metrics.get('unrealized_pnl', 0.0) if metrics else 0.0
                
                # Get current mark price
                mark_price = self.ticker.mark_price if self.ticker else 0.0
                
                # Calculate basic stats
                active_orders = len(self.order_manager.active_bids) + len(self.order_manager.active_asks) if hasattr(self.order_manager, 'active_bids') else 0
                
                # Log status with HFT optimization metrics
                self.logger.info(
                    f"Status: Pos={position:.3f} | PnL=[R:{realized_pnl:.2f} U:{unrealized_pnl:.2f}] | "
                    f"Price={mark_price:.1f} | Orders={active_orders} | "
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
                
            # Wait before next update (60 seconds)
            await asyncio.sleep(60)

    async def quote_task(self):
        """Main quoting loop - periodically checks if quotes need updating"""
        self.logger.info("Quote task started - waiting for updates")
        
        last_quote_time = 0
        min_quote_interval = 1.0  # Reduced from 3.0 to 1.0 seconds for more frequent updates
        force_quote_interval = 30.0  # Force new quotes every 30 seconds even without ticker updates
        
        # Add a diagnostic heartbeat to ensure task is running
        heartbeat_counter = 0
        
        # Wait for initial setup to complete
        self.logger.info("Waiting for setup to complete before starting quote generation")
        await self.setup_complete.wait()
        
        while True:
            try:
                # Add a diagnostic log every minute
                heartbeat_counter += 1
                if heartbeat_counter % 60 == 0:
                    self.logger.info(f"Quote task heartbeat #{heartbeat_counter} - task is alive")
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
                self.logger.info("Generating new quotes")
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
                self.logger.info("Quote update completed")
                
                # Add a delay after successfully updating quotes to prevent excessive updates
                await asyncio.sleep(0.5)
                    
            except asyncio.CancelledError:
                self.logger.info("Quote task cancelled")
                break
            except Exception as e:
                self.logger.error(f"Error in quote task: {str(e)}", exc_info=True)
                await asyncio.sleep(1.0)

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
            if self.perp_name:
                # Get underlying without USD suffix for index subscription
                underlying = MARKET_CONFIG["underlying"].replace("USD", "")
                
                await self.thalex.public_subscribe(
                    channels=[f"ticker.{self.perp_name}.raw", f"index.{underlying}"],
                    id=CALL_ID_SUBSCRIBE
                )
                self.logger.info(f"Subscribed to public channels for {self.perp_name}")
            else:
                self.logger.warning("No instrument name available for public subscription")
            
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

    def align_prices_to_tick(self, quotes, is_bid=True):
        """Align multiple quote prices to tick size in one vectorized operation
        
        This is a batch optimization that aligns multiple prices at once instead
        of calling individual alignment functions repeatedly.
        
        Args:
            quotes: List of Quote objects to align
            is_bid: Boolean indicating if these are bid quotes (True) or ask quotes (False)
            
        Returns:
            List of aligned Quote objects
        """
        if not quotes or self.tick_size <= 0:
            return quotes
            
        aligned_quotes = []
        direction = 'down' if is_bid else 'up'  # Bids round down, asks round up
        
        for quote in quotes:
            # Round price to nearest tick
            if direction == 'down':
                # For bids, round down to avoid overpaying
                aligned_price = math.floor(quote.price / self.tick_size) * self.tick_size
            else:
                # For asks, round up to avoid selling too cheaply
                aligned_price = math.ceil(quote.price / self.tick_size) * self.tick_size
            
            # Create new quote with updated price (only if needed)
            if abs(aligned_price - quote.price) > 1e-8:  # Use epsilon for float comparison
                aligned_quote = Quote(
                    price=aligned_price,
                    amount=quote.amount,
                    instrument=quote.instrument,
                    side=quote.side,
                    timestamp=quote.timestamp
                )
                aligned_quotes.append(aligned_quote)
            else:
                aligned_quotes.append(quote)  # Already aligned
                
        return aligned_quotes

    async def validate_quotes(self, bid_quotes: List[Quote], ask_quotes: List[Quote]) -> Tuple[List[Quote], List[Quote]]:
        """Validate quotes to ensure they meet market requirements"""
        # Early return if no quotes
        if not bid_quotes and not ask_quotes:
            return [], []
            
        # Check if we have ticker data
        if not self.ticker:
            return [], []
            
        # Get current mark price
        mid_price = self.ticker.mark_price
        if mid_price <= 0:
            return [], []
            
        # Get current best bid/ask from ticker if available
        best_bid = getattr(self.ticker, 'best_bid_price', 0)
        best_ask = getattr(self.ticker, 'best_ask_price', 0)
        
        # Validate bids - use batch alignment for better performance
        valid_bids = [q for q in bid_quotes if q.price > 0 and q.amount > 0]
        
        # Validate asks - use batch alignment for better performance
        valid_asks = [q for q in ask_quotes if q.price > 0 and q.amount > 0]
        
        # Apply price alignment in batches (faster)
        if valid_bids:
            valid_bids = self.align_prices_to_tick(valid_bids, is_bid=True)
            
        if valid_asks:
            valid_asks = self.align_prices_to_tick(valid_asks, is_bid=False)
            
        # Ensure bid/ask spread is maintained
        if valid_bids and valid_asks:
            max_bid = max(q.price for q in valid_bids)
            min_ask = min(q.price for q in valid_asks)
            
            # Check for crossed quotes and fix if needed
            if max_bid >= min_ask:
                # Use a simple middle point approach for speed
                mid_point = (max_bid + min_ask) / 2
                
                # Adjust bids and asks to not cross
                valid_bids = [
                    Quote(
                        price=min(q.price, mid_point - self.tick_size),
                        amount=q.amount,
                        instrument=q.instrument,
                        side=q.side,
                        timestamp=q.timestamp
                    ) if q.price >= min_ask else q
                    for q in valid_bids
                ]
                
                valid_asks = [
                    Quote(
                        price=max(q.price, mid_point + self.tick_size),
                        amount=q.amount,
                        instrument=q.instrument,
                        side=q.side,
                        timestamp=q.timestamp
                    ) if q.price <= max_bid else q
                    for q in valid_asks
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