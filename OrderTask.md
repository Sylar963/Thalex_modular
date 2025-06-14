# AvellanedaQuoter Performance Optimization Tasks

## Overview
This document provides granular steps to optimize `avellaneda_quoter.py` for high-frequency trading performance while maintaining 100% of existing functionality. Each task should be completed independently and tested before proceeding to the next.

## Critical Fix Tasks (Execute First)

### Task 1: Fix Syntax Error
**Priority: CRITICAL** 
**File: `avellaneda_quoter.py`**
**Line: 139**

```python
# BROKEN (line 139):
self.recovery_review _until = 0

# FIX TO:
self.recovery_cooldown_until = 0
```

**Verification**: File should compile without syntax errors.

---

## Memory Optimization Tasks

### Task 2: Add __slots__ to AvellanedaQuoter Class
**Priority: HIGH**
**Estimated Impact: 40-60% memory reduction**

Add `__slots__` to the main AvellanedaQuoter class:

```python
class AvellanedaQuoter:
    __slots__ = [
        # Core dependencies
        'thalex', 'logger', 'tasks', 'position_tracker', 'market_maker', 
        'order_manager', 'risk_manager', 'performance_monitor',
        
        # Risk and recovery state
        'active_trading', 'risk_recovery_mode', 'risk_breach_time',
        'recovery_cooldown_until', 'recovery_step', 'last_recovery_check',
        
        # Take profit state
        'take_profit_active', 'last_take_profit_check', 'take_profit_cooldown_until',
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
        '_cached_market_conditions', '_cached_market_time', '_log_counter'
    ]
```

**Verification**: Check memory usage before/after with `tracemalloc`.

### Task 3: Optimize Object Pools
**Priority: HIGH**
**Current Issue**: Object pools are created but may not be efficiently sized

Replace existing object pool initialization with optimized version:

```python
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
```

### Task 4: Add Performance Caching
**Priority: MEDIUM**

Add caching fields to `__init__` method:

```python
# Performance optimization caches
self._cached_market_conditions = {}
self._cached_market_time = 0.0
self._log_counter = 0
self._last_major_log = 0.0
```

---

## Logging Optimization Tasks

### Task 5: Implement Statistical Logging
**Priority: HIGH**
**Estimated Impact: 60-80% reduction in I/O overhead**

Add logging constants at the top of the file:

```python
# Performance optimization constants
LOG_SAMPLING_RATE = 0.05  # 5% of debug operations logged
MAJOR_EVENT_LOG_INTERVAL = 30.0  # Log major events every 30 seconds
```

### Task 6: Optimize Hot Path Logging - Message Processing
**Priority: HIGH**

In `listen_task()` method, replace frequent logging:

```python
# BEFORE (around line 580):
self.logger.info(f"Received message: {msg_sample}")

# AFTER:
self._log_counter += 1
if self._log_counter % 20 == 0:  # Log every 20th message
    self.logger.info(f"Processed {self._log_counter} messages. Sample: {msg_sample}")
```

### Task 7: Optimize Quote Generation Logging
**Priority: HIGH**

In `update_quotes()` method:

```python
# BEFORE:
self.logger.info(f"Updating quotes for {instrument_id} at price {price}")

# AFTER:
if random.random() < LOG_SAMPLING_RATE or price_change_significant:
    self.logger.info(f"Updating quotes for {instrument_id} at price {price}")
```

### Task 8: Optimize Ticker Update Logging
**Priority: MEDIUM**

In `handle_ticker_update()` method:

```python
# BEFORE:
self.logger.info(f"Processing ticker for instrument: {instrument_id}")

# AFTER:
if self._log_counter % 100 == 0:  # Log every 100th ticker update
    self.logger.info(f"Processed {self._log_counter} ticker updates for: {instrument_id}")
```

---

## Computational Optimization Tasks

### Task 9: Cache Market Conditions
**Priority: HIGH**

Optimize `get_market_conditions()` method:

```python
def get_market_conditions(self) -> Dict:
    """Get market conditions with caching optimization"""
    current_time = time.time()
    
    # Use cache if recent (within 100ms for HFT)
    if (hasattr(self, '_cached_market_conditions') and 
        current_time - self._cached_market_time < 0.1):
        return self._cached_market_conditions
    
    # Existing calculation logic...
    market_state = self.market_data.get_market_state()
    
    # Cache result
    self._cached_market_conditions = conditions
    self._cached_market_time = current_time
    return conditions
```

### Task 10: Optimize Price Alignment
**Priority: MEDIUM**

Replace `align_prices_to_tick()` with vectorized version:

```python
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
```

### Task 11: Optimize Message Buffer Management
**Priority: MEDIUM**

In `listen_task()`, optimize buffer operations:

```python
# Add at class level
MESSAGE_BUFFER_INITIAL_SIZE = 32768
MESSAGE_BUFFER_MAX_SIZE = 1048576

# In message processing
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
```

---

## Task Execution and Risk Management Optimization

### Task 12: Optimize Risk Monitoring Task
**Priority: HIGH**

Optimize `_risk_monitoring_task()` frequency:

```python
async def _risk_monitoring_task(self):
    """Optimized risk monitoring with adaptive intervals"""
    base_interval = self.risk_monitoring_interval
    
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
            
            # Existing risk checking logic...
            
        except Exception as e:
            self.logger.error(f"Error in risk monitoring: {str(e)}")
            await asyncio.sleep(base_interval * 2)  # Back off on errors
```

### Task 13: Optimize Quote Task Performance
**Priority: HIGH**

In `quote_task()`, add performance optimizations:

```python
async def quote_task(self):
    """Optimized quote task with reduced latency"""
    # Pre-allocate frequently used variables
    last_quote_time = 0
    min_quote_interval = 0.5  # Reduced from 1.0 for faster response
    force_quote_interval = 15.0  # Reduced from 30.0
    
    # Performance counters
    quote_counter = 0
    
    while True:
        try:
            quote_counter += 1
            
            # Log status less frequently
            if quote_counter % 1200 == 0:  # Every ~10 minutes at 0.5s intervals
                self.logger.info(f"Quote task alive - processed {quote_counter} cycles")
            
            # Existing logic with optimized intervals...
            
        except Exception as e:
            self.logger.error(f"Error in quote task: {str(e)}")
            await asyncio.sleep(0.1)  # Shorter error recovery time
```

---

## Data Structure Optimization Tasks

### Task 14: Optimize Price History Storage
**Priority: MEDIUM**

Replace list-based price history with numpy array:

```python
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
```

### Task 15: Optimize Order Tracking
**Priority: MEDIUM**

Add fast lookup structures:

```python
def _initialize_order_tracking(self):
    """Initialize optimized order tracking structures"""
    # Fast lookup by order ID
    self.orders_by_id = {}
    # Fast lookup by client order ID  
    self.orders_by_client_id = {}
    # Count tracking for limits
    self.active_bid_count = 0
    self.active_ask_count = 0
```

---

## Integration and Testing Tasks

### Task 16: Add Performance Monitoring
**Priority: LOW**

Add performance counters to track optimization effectiveness:

```python
class QuoterPerformanceCounters:
    def __init__(self):
        self.message_processing_times = deque(maxlen=1000)
        self.quote_generation_times = deque(maxlen=1000)
        self.cache_hits = 0
        self.cache_misses = 0
        self.total_quotes_generated = 0
        
    def record_message_time(self, duration_ns: int):
        self.message_processing_times.append(duration_ns)
    
    def get_average_message_time_us(self) -> float:
        if not self.message_processing_times:
            return 0.0
        return sum(self.message_processing_times) / len(self.message_processing_times) / 1000
```

### Task 17: Add Memory Usage Monitoring
**Priority: LOW**

```python
def log_memory_usage(self):
    """Log current memory usage for optimization tracking"""
    if hasattr(self, '_last_memory_log') and time.time() - self._last_memory_log < 300:
        return  # Log only every 5 minutes
    
    try:
        import psutil
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        self.logger.info(f"Memory usage: {memory_mb:.2f} MB")
        self._last_memory_log = time.time()
    except ImportError:
        pass  # psutil not available
```

---

## Final Cleanup Task

### Task 18: Remove Excessive Logging
**Priority: HIGH**
**Estimated Impact: 70-90% reduction in log noise**

Systematically remove or reduce excessive logging throughout the file:

#### 18.1: Remove Debug Logging in Hot Paths

```python
# LOCATIONS TO OPTIMIZE:

# In handle_ticker_update() (around lines 650-750):
# REMOVE these excessive info logs:
self.logger.info(f"Received ticker update. Keys: {list(ticker_data.keys())}. Channel: {channel_name}")
self.logger.info(f"Extracted instrument ID '{instrument_id}' from channel_name '{channel_name}'")
self.logger.info(f"Processing ticker for instrument: {instrument_id}")
self.logger.info(f"Ticker prices: mark={mark_price}, bid={best_bid_price}, ask={best_ask_price}")
self.logger.info(f"Updated ticker: {instrument_id}, mark: {ticker.mark_price}, bid: {ticker.best_bid_price}, ask: {ticker.best_ask_price}")

# REPLACE WITH (log only every 1000th update):
if self._log_counter % 1000 == 0:
    self.logger.info(f"Ticker update #{self._log_counter}: {instrument_id} mark={ticker.mark_price:.2f}")
```

#### 18.2: Reduce Order Processing Logging

```python
# In handle_order_update() (around lines 850-950):
# REDUCE frequency of these logs:
self.logger.info(f"Order filled: {order.id} - {order.direction.upper()} {order.amount:.4f} @ {order.price:.2f}")
self.logger.info(f"PositionTracker updated with fill: {fill_object.order_id}")

# REPLACE WITH (log only significant fills):
if order.amount >= TRADING_CONFIG["avellaneda"].get("significant_fill_threshold", 0.1):
    self.logger.info(f"SIGNIFICANT FILL: {order.id} - {order.direction.upper()} {order.amount:.4f} @ {order.price:.2f}")
elif self._log_counter % 10 == 0:
    self.logger.debug(f"Fill #{self._log_counter}: {order.amount:.4f} @ {order.price:.2f}")
```

#### 18.3: Reduce Setup and Connection Logging

```python
# In start() method (around lines 180-350):
# REMOVE excessive setup logging:
# self.logger.info("Starting Avellaneda quoter...")
# self.logger.info("Successfully connected to Thalex API")
# self.logger.info("Setting up instrument data...")
# self.logger.info("Instrument data retrieved successfully")

# REPLACE WITH consolidated startup log:
self.logger.info("AvellanedaQuoter startup initiated...")
# ... (keep only critical setup logs)
self.logger.info("AvellanedaQuoter startup complete - all systems operational")
```

#### 18.4: Optimize WebSocket Message Logging

```python
# In listen_task() (around lines 450-600):
# REMOVE these frequent logs:
self.logger.info(f"Received message: {msg_sample}")
self.logger.info(f"Received data with keys: {list(data.keys())}")
self.logger.info(f"Processing notification from channel: {channel}")

# REPLACE WITH batch logging:
self._message_count = getattr(self, '_message_count', 0) + 1
if self._message_count % 100 == 0:
    self.logger.info(f"Processed {self._message_count} WebSocket messages")
```

#### 18.5: Reduce Quote Generation Logging

```python
# In update_quotes() (around lines 1200-1400):
# REMOVE excessive quote logging:
self.logger.info(f"Created price data from ticker: mid_price={price_data.get('mid_price')}")
self.logger.info(f"Updating quotes for {instrument_id} at price {price}")
self.logger.info(f"Generating quotes using Avellaneda market maker")
self.logger.info(f"Generated {len(bid_quotes)} bid quotes and {len(ask_quotes)} ask quotes")
self.logger.info("Directly placing quotes instead of signaling")

# REPLACE WITH summary logging:
if self._log_counter % 50 == 0 or quote_generation_significant:
    self.logger.info(f"Quote update #{self._log_counter}: {len(bid_quotes)}B/{len(ask_quotes)}A @ {price:.2f}")
```

#### 18.6: Clean Up Place Quotes Logging

```python
# In place_quotes() (around lines 1800-2000):
# REMOVE excessive order placement logs:
self.logger.info(f"Placed bid: {quote.amount:.3f}@{quote.price:.2f}")
self.logger.info(f"Placed ask: {quote.amount:.3f}@{quote.price:.2f}")
self.logger.info(f"Sending limit orders: {placed_bids} bids, {placed_asks} asks")
self.logger.info(f"Limit orders sent: {placed_bids} bids, {placed_asks} asks")

# REPLACE WITH single summary log:
self.logger.info(f"Orders placed: {placed_bids}B/{placed_asks}A")
```

#### 18.7: Optimize Status and Heartbeat Logging

```python
# In log_status_task() (around lines 2200-2300):
# REDUCE status logging frequency from every 60s to every 300s (5 minutes):
await asyncio.sleep(300)  # Changed from 60

# In heartbeat_task() - remove verbose connection status logs:
# Remove: self.logger.warning("WebSocket disconnected, skipping heartbeat...")
# Keep only: Critical connection failures
```

#### 18.8: Remove Development/Debug Logging

Search and remove or comment out all development logging patterns:

```python
# REMOVE all instances of:
# self.logger.debug(f"...")  # Remove debug logs in production
# self.logger.info(f"PHASE X DEBUG: ...")  # Remove phase debug logs
# Excessive instrument data logging
# Verbose error context logging (keep errors, remove verbose context)
```

#### 18.9: Add Logging Level Control

Add a logging control mechanism:

```python
# Add to __init__:
self.verbose_logging = BOT_CONFIG.get("verbose_logging", False)
self.log_sampling_counter = 0

def _should_log_verbose(self, frequency: int = 100) -> bool:
    """Determine if verbose logging should occur"""
    if not self.verbose_logging:
        self.log_sampling_counter += 1
        return self.log_sampling_counter % frequency == 0
    return True

# Usage example:
if self._should_log_verbose(50):  # Log every 50th occurrence
    self.logger.info(f"Verbose log message")
```

**Expected Results**:
- 70-90% reduction in log volume
- Significant I/O performance improvement
- Cleaner, more focused log output
- Maintain all critical error and event logging
- Preserve debugging capability through verbose mode

**Verification**:
1. Compare log file sizes before/after
2. Measure I/O wait time reduction
3. Verify all critical events still logged
4. Test verbose mode still provides debugging info

---

## Execution Order and Dependencies

### Phase 1: Critical Fixes (Execute First)
- Task 1: Fix Syntax Error

### Phase 2: Memory Optimizations  
- Task 2: Add __slots__
- Task 3: Optimize Object Pools
- Task 4: Add Performance Caching

### Phase 3: Logging Optimizations
- Task 5: Implement Statistical Logging
- Task 6-8: Optimize Hot Path Logging

### Phase 4: Computational Optimizations
- Task 9: Cache Market Conditions
- Task 10-11: Optimize Processing Functions

### Phase 5: Task Optimizations
- Task 12-13: Optimize Background Tasks

### Phase 6: Data Structure Optimizations
- Task 14-15: Optimize Storage Structures

### Phase 7: Monitoring and Validation
- Task 16-17: Add Performance Monitoring

### Phase 8: Final Cleanup
- Task 18: Remove Excessive Logging

## Success Metrics

After completing all tasks, measure:
- **Latency**: Quote generation time < 100 microseconds
- **Memory**: 40-60% reduction in memory usage
- **CPU**: 50-70% reduction in logging overhead
- **I/O**: 70-90% reduction in log volume
- **Throughput**: Handle 10,000+ messages/second without degradation

## Testing Requirements

For each task:
1. **Unit Test**: Verify functionality unchanged
2. **Performance Test**: Measure improvement
3. **Integration Test**: Verify with full system
4. **Regression Test**: Ensure no functionality loss

## Rollback Plan

Each task should be committed separately to git with clear commit messages, allowing easy rollback if issues are discovered. 