# MFT Implementation Guide: Minimal-Impact Migration to Multi-Process Architecture

## Executive Summary

This document provides a granular, step-by-step migration plan from the current single-process Thalex SimpleQuoter to the MFT (Mid-Frequency Trading) multi-process architecture. The migration follows a **"Strangler Fig Pattern"** - gradually replacing components while maintaining system functionality, minimizing risk, and enabling rollback at each stage.

## Architectural Comparison

### Current State (Architecture.md)
```
Single Process (Python GIL Bottleneck)
├── AvellanedaQuoter (Main orchestrator, 2845 lines)
├── AvellanedaMarketMaker (A-S model, 1786 lines)
├── OrderManager (Order lifecycle, 599 lines)
├── RiskManager (Risk controls, 333 lines)
└── PositionTracker (Position state, 398 lines)

Issues:
- GIL contention on CPU-bound math
- GC pauses freeze entire system
- I/O blocking affects all components
- Single point of failure
```

### Target State (MTFArquitecture.md)
```
Multi-Process (GIL-Free)
├── MarketDataProcess (Core 1) → WebSocket + parsing
├── StrategyProcess (Core 2) → A-S model + decisions
└── ExecutionProcess (Core 3) → Order management

Benefits:
- True parallelism across CPU cores
- Isolated GC domains
- Non-blocking I/O separation
- Process-level fault isolation
```

## Migration Philosophy

### Core Principles
1. **Incremental Replacement**: Replace components one at a time
2. **Backward Compatibility**: Maintain existing interfaces during transition
3. **Feature Parity**: Each migration step preserves all functionality
4. **Risk Mitigation**: Enable rollback at every stage
5. **Performance Validation**: Measure improvements at each step

### Migration Phases
1. **Phase 1**: Preparation and Infrastructure (2-3 days)
2. **Phase 2**: Market Data Process Extraction (3-4 days)
3. **Phase 3**: Strategy Process Isolation (4-5 days)
4. **Phase 4**: Execution Process Separation (3-4 days)
5. **Phase 5**: Optimization and Polish (2-3 days)

---

## Phase 1: Preparation and Infrastructure (Days 1-3)

### Step 1.1: Create MFT Directory Structure
**Objective**: Establish new structure alongside existing code
**Risk Level**: ⚠️ LOW - No existing code modified

**Implementation**:
```bash
mkdir -p thalex_py/Thalex_modular/mft/
mkdir -p thalex_py/Thalex_modular/mft/processes/
mkdir -p thalex_py/Thalex_modular/mft/shared/
mkdir -p thalex_py/Thalex_modular/mft/ipc/
mkdir -p thalex_py/Thalex_modular/mft/models_optimized/
mkdir -p thalex_py/Thalex_modular/mft/utils/
```

**Files to Create**:
- `mft/__init__.py` - MFT package initialization
- `mft/shared/__init__.py` - Shared memory interfaces
- `mft/ipc/__init__.py` - Inter-process communication
- `mft/utils/__init__.py` - Performance utilities

### Step 1.2: Implement Shared Memory Infrastructure
**Objective**: Create the backbone for inter-process data sharing
**Risk Level**: ⚠️ LOW - New isolated components

**Create**: `thalex_py/Thalex_modular/mft/shared/shared_book.py`
```python
import multiprocessing as mp
import numpy as np
from typing import Optional
import struct
import time

class SharedOrderBook:
    """Shared memory order book for zero-copy access across processes"""
    
    # Memory layout constants
    LEVELS = 50  # Max order book levels per side
    HEADER_SIZE = 64  # Metadata section
    LEVEL_SIZE = 24   # price(8) + size(8) + count(4) + padding(4)
    
    def __init__(self, name: str = "thalex_book", create: bool = True):
        self.name = name
        self.total_size = self.HEADER_SIZE + (self.LEVELS * 2 * self.LEVEL_SIZE)
        
        if create:
            try:
                # Try to unlink existing
                mp.shared_memory.SharedMemory(name, create=False).unlink()
            except FileNotFoundError:
                pass
            
            self.shm = mp.shared_memory.SharedMemory(
                name=name, create=True, size=self.total_size
            )
            self._initialize_memory()
        else:
            self.shm = mp.shared_memory.SharedMemory(name=name, create=False)
        
        self._setup_views()
    
    def _initialize_memory(self):
        """Initialize shared memory structure"""
        # Zero out all memory
        self.shm.buf[:] = b'\x00' * self.total_size
        
        # Write header
        header = struct.pack('QQdd', 0, 0, 0.0, 0.0)  # sequence, timestamp, mid, spread
        self.shm.buf[0:len(header)] = header
    
    def _setup_views(self):
        """Create numpy views for efficient access"""
        # Header view
        self.header = np.frombuffer(
            self.shm.buf, dtype=[
                ('sequence', 'u8'),
                ('timestamp', 'u8'), 
                ('mid_price', 'f8'),
                ('spread', 'f8')
            ], count=1, offset=0
        )
        
        # Bids view (descending price order)
        bids_offset = self.HEADER_SIZE
        self.bids = np.frombuffer(
            self.shm.buf, 
            dtype=[('price', 'f8'), ('size', 'f8'), ('count', 'u4'), ('_pad', 'u4')],
            count=self.LEVELS,
            offset=bids_offset
        )
        
        # Asks view (ascending price order)  
        asks_offset = bids_offset + (self.LEVELS * self.LEVEL_SIZE)
        self.asks = np.frombuffer(
            self.shm.buf,
            dtype=[('price', 'f8'), ('size', 'f8'), ('count', 'u4'), ('_pad', 'u4')],
            count=self.LEVELS,
            offset=asks_offset
        )
```

### Step 1.3: Create IPC Queue Infrastructure
**Objective**: Fast messaging between processes
**Risk Level**: ⚠️ LOW - New components only

**Create**: `thalex_py/Thalex_modular/mft/ipc/message_types.py`
```python
from dataclasses import dataclass
from typing import Optional, Dict, Any
from enum import Enum
import time

class MessageType(Enum):
    TRADE_EVENT = "trade_event"
    FILL_EVENT = "fill_event" 
    ORDER_REQUEST = "order_request"
    CANCEL_REQUEST = "cancel_request"
    HEARTBEAT = "heartbeat"
    SHUTDOWN = "shutdown"

@dataclass
class TradeEvent:
    """Trade event from market data"""
    instrument: str
    trade_id: str
    price: float
    size: float
    side: str  # "buy" or "sell"
    timestamp: float
    sequence: int

@dataclass  
class FillEvent:
    """Order fill notification"""
    order_id: str
    instrument: str
    fill_price: float
    fill_size: float
    remaining_size: float
    side: str
    timestamp: float
    trade_id: str

@dataclass
class OrderRequest:
    """New order request"""
    instrument: str
    side: str  # "buy" or "sell"
    price: float
    size: float
    order_type: str = "limit"
    post_only: bool = True
    label: str = ""
    request_id: str = ""
```

### Step 1.4: Performance Utilities Foundation
**Objective**: CPU pinning, GC control, and monitoring tools
**Risk Level**: ⚠️ LOW - Utility functions only

**Create**: `thalex_py/Thalex_modular/mft/utils/perf_utils.py`
```python
import os
import gc
import psutil
import threading
import time
from contextlib import contextmanager
from typing import Set, Optional

class CPUAffinity:
    """CPU core pinning utilities"""
    
    @staticmethod
    def pin_to_core(core_id: int, pid: Optional[int] = None) -> bool:
        """Pin process/thread to specific CPU core"""
        try:
            if pid is None:
                pid = os.getpid()
            
            # Use psutil for cross-platform compatibility
            process = psutil.Process(pid)
            process.cpu_affinity([core_id])
            return True
        except Exception as e:
            print(f"Failed to pin to core {core_id}: {e}")
            return False
    
    @staticmethod
    def get_available_cores() -> Set[int]:
        """Get available CPU cores"""
        return set(range(psutil.cpu_count()))

class GCController:
    """Garbage collection management"""
    
    def __init__(self):
        self.gc_enabled = True
        self.collect_count = 0
        
    @contextmanager
    def disabled(self):
        """Context manager to disable GC temporarily"""
        was_enabled = gc.isenabled()
        if was_enabled:
            gc.disable()
        try:
            yield
        finally:
            if was_enabled:
                gc.enable()
    
    def controlled_collect(self, iterations: int = 1000):
        """Perform GC collection every N iterations"""
        self.collect_count += 1
        if self.collect_count >= iterations:
            gc.collect()
            self.collect_count = 0

# Global instances
cpu_affinity = CPUAffinity()
gc_controller = GCController()
```

---

## Phase 2: Market Data Process Extraction (Days 4-7)

### Step 2.1: Create Market Data Process Stub
**Objective**: Build the process shell with existing interface
**Risk Level**: ⚠️ MEDIUM - Creates new process but doesn't change existing flow

**Create**: `thalex_py/Thalex_modular/mft/processes/market_data_proc.py`
```python
import asyncio
import uvloop
import multiprocessing as mp
import signal
import sys
from typing import Optional
import json

from ..shared.shared_book import SharedOrderBook
from ..ipc.message_types import TradeEvent, MessageType
from ..utils.perf_utils import cpu_affinity, gc_controller

class MarketDataProcess:
    """Dedicated process for WebSocket market data ingestion"""
    
    def __init__(self, 
                 shared_book_name: str,
                 trade_queue: mp.Queue,
                 cpu_core: int = 1):
        self.shared_book_name = shared_book_name
        self.trade_queue = trade_queue
        self.cpu_core = cpu_core
        self.should_stop = False
        
        # Will be initialized in run()
        self.shared_book: Optional[SharedOrderBook] = None
        self.websocket = None
        
    def run(self):
        """Main process entry point"""
        # Pin to CPU core
        cpu_affinity.pin_to_core(self.cpu_core)
        
        # Set signal handlers
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
        
        # Use uvloop for performance
        asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
        
        # Initialize shared memory
        self.shared_book = SharedOrderBook(self.shared_book_name, create=False)
        
        # Run async event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            loop.run_until_complete(self._async_main())
        finally:
            loop.close()
    
    async def _async_main(self):
        """Main async event loop"""
        # For now, just simulate - will integrate with existing WebSocket later
        print(f"MarketDataProcess started on core {self.cpu_core}")
        
        # Simulation loop
        sequence = 0
        while not self.should_stop:
            # Update shared book header with heartbeat
            self.shared_book.header['sequence'] = sequence
            self.shared_book.header['timestamp'] = int(time.time() * 1000000)
            
            sequence += 1
            await asyncio.sleep(1.0)
            
        print("MarketDataProcess shutting down")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        print(f"MarketDataProcess received signal {signum}")
        self.should_stop = True

def start_market_data_process(shared_book_name: str, 
                             trade_queue: mp.Queue,
                             cpu_core: int = 1) -> mp.Process:
    """Factory function to start market data process"""
    
    def target():
        proc = MarketDataProcess(shared_book_name, trade_queue, cpu_core)
        proc.run()
    
    process = mp.Process(target=target, name="MarketDataProcess")
    process.start()
    return process
```

### Step 2.2: Create Adapter Layer
**Objective**: Bridge between existing code and new market data process
**Risk Level**: ⚠️ MEDIUM - Modifies existing data flow

**Create**: `thalex_py/Thalex_modular/mft/adapters/market_data_adapter.py`
```python
import multiprocessing as mp
import threading
import time
from typing import Optional, Callable

from ..shared.shared_book import SharedOrderBook
from ..processes.market_data_proc import start_market_data_process
from ..ipc.message_types import TradeEvent

class MarketDataAdapter:
    """Adapter to integrate MFT market data process with existing AvellanedaQuoter"""
    
    def __init__(self, 
                 book_update_callback: Optional[Callable] = None,
                 trade_callback: Optional[Callable] = None):
        self.book_update_callback = book_update_callback
        self.trade_callback = trade_callback
        
        # IPC components
        self.shared_book_name = "thalex_book_main"
        self.trade_queue = mp.Queue(maxsize=1000)
        
        # Process and monitoring
        self.market_data_process: Optional[mp.Process] = None
        self.monitor_thread: Optional[threading.Thread] = None
        self.should_stop = False
        
        # Shared memory access
        self.shared_book: Optional[SharedOrderBook] = None
        
    def start(self):
        """Start the market data process and monitoring"""
        if self.market_data_process is not None:
            return  # Already started
            
        # Create shared memory
        self.shared_book = SharedOrderBook(self.shared_book_name, create=True)
        
        # Start market data process
        self.market_data_process = start_market_data_process(
            self.shared_book_name, 
            self.trade_queue,
            cpu_core=1
        )
        
        # Start monitoring thread
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        
        print("MarketDataAdapter started")
    
    def stop(self):
        """Stop all processes and cleanup"""
        self.should_stop = True
        
        if self.market_data_process:
            self.market_data_process.terminate()
            self.market_data_process.join(timeout=5)
            if self.market_data_process.is_alive():
                self.market_data_process.kill()
        
        if self.shared_book:
            self.shared_book.shm.unlink()
            
        print("MarketDataAdapter stopped")
    
    def _monitor_loop(self):
        """Monitor shared memory and trade queue"""
        last_sequence = 0
        
        while not self.should_stop:
            try:
                # Check for book updates
                current_sequence = int(self.shared_book.header['sequence'])
                if current_sequence != last_sequence:
                    if self.book_update_callback:
                        self.book_update_callback(self.shared_book)
                    last_sequence = current_sequence
                
                # Check for trade events
                try:
                    trade_event = self.trade_queue.get_nowait()
                    if self.trade_callback:
                        self.trade_callback(trade_event)
                except:
                    pass  # Queue empty
                
                time.sleep(0.001)  # 1ms polling
                
            except Exception as e:
                print(f"Monitor loop error: {e}")
                time.sleep(0.1)
    
    def get_book_data(self) -> dict:
        """Get current book data in existing format for compatibility"""
        if not self.shared_book:
            return {}
            
        # Convert shared memory format to existing format
        bids = []
        asks = []
        
        for i in range(10):  # Top 10 levels
            if self.shared_book.bids[i]['price'] > 0:
                bids.append({
                    'price': float(self.shared_book.bids[i]['price']),
                    'size': float(self.shared_book.bids[i]['size'])
                })
            
            if self.shared_book.asks[i]['price'] > 0:
                asks.append({
                    'price': float(self.shared_book.asks[i]['price']),
                    'size': float(self.shared_book.asks[i]['size'])
                })
        
        mid_price = float(self.shared_book.header['mid_price'])
        spread = float(self.shared_book.header['spread'])
        
        return {
            'bids': bids,
            'asks': asks,
            'mid_price': mid_price,
            'spread': spread,
            'timestamp': int(self.shared_book.header['timestamp'])
        }
```

**Validation Step**: At this point, we can test the infrastructure without modifying existing code:
```python
# Test script: test_mft_phase1.py
from thalex_py.Thalex_modular.mft.adapters.market_data_adapter import MarketDataAdapter
import time

def test_adapter():
    adapter = MarketDataAdapter()
    adapter.start()
    
    time.sleep(5)  # Let it run
    
    book_data = adapter.get_book_data()
    print(f"Book data: {book_data}")
    
    adapter.stop()

if __name__ == "__main__":
    test_adapter()
``` 

### Step 2.3: Integration with Existing AvellanedaQuoter
**Objective**: Connect new market data process with existing system
**Risk Level**: ⚠️ HIGH - Modifies core system flow
**Rollback Strategy**: Feature flag to switch between old and new data sources

**Modify**: `thalex_py/Thalex_modular/avellaneda_quoter.py`

**Add at the top of the file**:
```python
# Add after existing imports
from .mft.adapters.market_data_adapter import MarketDataAdapter

# Add configuration flag
USE_MFT_MARKET_DATA = False  # Feature flag for gradual rollout
```

**Add to AvellanedaQuoter.__init__() method**:
```python
# Add after existing initialization
if USE_MFT_MARKET_DATA:
    self.mft_market_adapter = MarketDataAdapter(
        book_update_callback=self._handle_mft_book_update,
        trade_callback=self._handle_mft_trade_event
    )
else:
    self.mft_market_adapter = None
```

**Add new methods to AvellanedaQuoter class**:
```python
def _handle_mft_book_update(self, shared_book):
    """Handle book updates from MFT market data process"""
    try:
        # Convert to existing format and feed to existing handlers
        book_data = self.mft_market_adapter.get_book_data()
        
        # Feed to existing book processing logic
        if hasattr(self, 'on_book_update'):
            self.on_book_update(book_data)
            
    except Exception as e:
        self.logger.error(f"MFT book update error: {e}")

def _handle_mft_trade_event(self, trade_event):
    """Handle trade events from MFT market data process"""
    try:
        # Convert to existing trade format
        trade_data = {
            'trade_id': trade_event.trade_id,
            'price': trade_event.price,
            'size': trade_event.size,
            'side': trade_event.side,
            'timestamp': trade_event.timestamp,
            'instrument': trade_event.instrument
        }
        
        # Feed to existing trade processing logic
        if hasattr(self, 'on_trade'):
            self.on_trade(trade_data)
            
    except Exception as e:
        self.logger.error(f"MFT trade event error: {e}")

async def start_mft_components(self):
    """Start MFT components if enabled"""
    if self.mft_market_adapter:
        self.mft_market_adapter.start()
        self.logger.info("MFT market data adapter started")

async def stop_mft_components(self):
    """Stop MFT components"""
    if self.mft_market_adapter:
        self.mft_market_adapter.stop()
        self.logger.info("MFT market data adapter stopped")
```

**Modify the main() method**:
```python
# Add to the startup sequence
await self.start_mft_components()

# Add to shutdown sequence  
await self.stop_mft_components()
```

**Testing and Validation**:
1. Deploy with `USE_MFT_MARKET_DATA = False` (no changes to behavior)
2. Set `USE_MFT_MARKET_DATA = True` and monitor for identical behavior
3. Performance comparison between old and new data paths
4. Validate all existing market data callbacks still work

---

## Phase 3: Strategy Process Isolation (Days 8-12)

### Step 3.1: Create Numba-Accelerated A-S Model
**Objective**: Extract mathematical computations with JIT compilation
**Risk Level**: ⚠️ MEDIUM - New implementation must match existing behavior exactly

**Create**: `thalex_py/Thalex_modular/mft/models_optimized/avellaneda_numba.py`
```python
import numba as nb
import numpy as np
import math

@nb.njit(fastmath=True, cache=True)
def calculate_optimal_spread(gamma: float, 
                           sigma: float, 
                           time_horizon: float,
                           kappa: float,
                           base_spread: float,
                           market_impact_factor: float) -> float:
    """Calculate optimal bid-ask spread using A-S model"""
    
    # Core A-S formula: δ = γσ²(T-t) + (2/γ)ln(1 + γ/κ)
    volatility_component = gamma * sigma * sigma * time_horizon
    order_flow_component = (2.0 / gamma) * math.log(1.0 + gamma / kappa)
    
    optimal_spread = volatility_component + order_flow_component
    
    # Apply market impact and base spread factors
    adjusted_spread = optimal_spread * market_impact_factor + base_spread
    
    return adjusted_spread

@nb.njit(fastmath=True, cache=True) 
def calculate_reservation_price(mid_price: float,
                              inventory: float,
                              gamma: float,
                              sigma: float,
                              time_horizon: float,
                              inventory_weight: float) -> float:
    """Calculate reservation price with inventory adjustment"""
    
    # Reservation price: r = S - q·γ·σ²·(T-t)
    inventory_adjustment = inventory * gamma * sigma * sigma * time_horizon * inventory_weight
    reservation_price = mid_price - inventory_adjustment
    
    return reservation_price

@nb.njit(fastmath=True, cache=True)
def calculate_quote_prices(reservation_price: float,
                         spread: float,
                         inventory: float,
                         skew_factor: float,
                         max_spread: float) -> tuple:
    """Calculate bid and ask prices with skewing"""
    
    # Apply spread limits
    effective_spread = min(spread, max_spread)
    half_spread = effective_spread / 2.0
    
    # Apply inventory skewing
    skew_adjustment = inventory * skew_factor
    
    bid_price = reservation_price - half_spread - skew_adjustment
    ask_price = reservation_price + half_spread + skew_adjustment
    
    return bid_price, ask_price

@nb.njit(fastmath=True, cache=True)
def calculate_volatility_ewm(prices: np.ndarray, 
                           span: float) -> float:
    """Calculate exponentially weighted moving average volatility"""
    if len(prices) < 2:
        return 0.01  # Default volatility
    
    # Calculate log returns
    returns = np.empty(len(prices) - 1)
    for i in range(1, len(prices)):
        returns[i-1] = math.log(prices[i] / prices[i-1])
    
    if len(returns) == 0:
        return 0.01
    
    # EWM variance calculation
    alpha = 2.0 / (span + 1.0)
    var_ewm = returns[0] * returns[0]  # Initialize with first squared return
    
    for i in range(1, len(returns)):
        var_ewm = alpha * returns[i] * returns[i] + (1 - alpha) * var_ewm
    
    volatility = math.sqrt(var_ewm)
    return volatility

@nb.njit(fastmath=True, cache=True)
def calculate_position_sizes(base_size: float,
                           size_multipliers: np.ndarray,
                           inventory_ratio: float,
                           max_position: float,
                           current_position: float) -> tuple:
    """Calculate quote sizes for bid and ask sides"""
    
    # Inventory-based size adjustment
    position_utilization = abs(current_position) / max_position
    size_factor = max(0.1, 1.0 - position_utilization)  # Reduce size as position grows
    
    # Calculate base sizes for each side
    inventory_adjustment = inventory_ratio * 0.5  # Scale down adjustment
    
    bid_size_factor = size_factor * (1.0 + max(0, -inventory_adjustment))
    ask_size_factor = size_factor * (1.0 + max(0, inventory_adjustment))
    
    # Apply multipliers to get level sizes
    bid_sizes = np.empty(len(size_multipliers))
    ask_sizes = np.empty(len(size_multipliers))
    
    for i in range(len(size_multipliers)):
        bid_sizes[i] = base_size * size_multipliers[i] * bid_size_factor
        ask_sizes[i] = base_size * size_multipliers[i] * ask_size_factor
    
    return bid_sizes, ask_sizes

class AvellanedaModelNumba:
    """Numba-accelerated Avellaneda-Stoikov model"""
    
    def __init__(self, config: dict):
        self.config = config
        
        # Pre-allocate arrays for Numba functions
        self.price_history = np.zeros(config.get('volatility_window', 50))
        self.price_index = 0
        self.price_count = 0
        
        # Extract size multipliers as numpy array
        self.size_multipliers = np.array(config.get('size_multipliers', [1.0, 2.0, 3.0]))
        
    def update_price(self, price: float):
        """Update price history for volatility calculation"""
        self.price_history[self.price_index] = price
        self.price_index = (self.price_index + 1) % len(self.price_history)
        self.price_count = min(self.price_count + 1, len(self.price_history))
    
    def calculate_quotes(self, 
                        mid_price: float,
                        inventory: float,
                        current_position: float) -> dict:
        """Calculate optimal quotes using Numba-accelerated functions"""
        
        # Calculate volatility from price history
        if self.price_count >= 2:
            active_prices = self.price_history[:self.price_count]
            volatility = calculate_volatility_ewm(active_prices, self.config['ewm_span'])
        else:
            volatility = self.config['default_volatility']
        
        # Apply volatility bounds
        volatility = max(self.config['volatility_floor'], 
                        min(volatility, self.config['volatility_ceiling']))
        
        # Calculate optimal spread
        spread = calculate_optimal_spread(
            gamma=self.config['gamma'],
            sigma=volatility,
            time_horizon=self.config['time_horizon'],
            kappa=self.config['kappa'],
            base_spread=self.config['base_spread'],
            market_impact_factor=self.config['market_impact_factor']
        )
        
        # Calculate reservation price
        reservation_price = calculate_reservation_price(
            mid_price=mid_price,
            inventory=inventory,
            gamma=self.config['gamma'],
            sigma=volatility,
            time_horizon=self.config['time_horizon'],
            inventory_weight=self.config['inventory_weight']
        )
        
        # Calculate quote prices
        bid_price, ask_price = calculate_quote_prices(
            reservation_price=reservation_price,
            spread=spread,
            inventory=inventory,
            skew_factor=self.config['inventory_weight'],
            max_spread=self.config['max_spread']
        )
        
        # Calculate position sizes
        inventory_ratio = inventory / self.config['max_position'] if self.config['max_position'] > 0 else 0
        bid_sizes, ask_sizes = calculate_position_sizes(
            base_size=self.config['base_size'],
            size_multipliers=self.size_multipliers,
            inventory_ratio=inventory_ratio,
            max_position=self.config['max_position'],
            current_position=current_position
        )
        
        return {
            'bid_price': bid_price,
            'ask_price': ask_price,
            'bid_sizes': bid_sizes.tolist(),
            'ask_sizes': ask_sizes.tolist(),
            'spread': spread,
            'volatility': volatility,
            'reservation_price': reservation_price,
            'inventory_ratio': inventory_ratio
        }
```

### Step 3.2: Create Strategy Process Core
**Objective**: Isolated process for trading decisions
**Risk Level**: ⚠️ HIGH - Core trading logic separation

**Create**: `thalex_py/Thalex_modular/mft/processes/strategy_proc.py`
```python
import multiprocessing as mp
import signal
import time
import gc
from typing import Optional, Dict, Any
import json

from ..shared.shared_book import SharedOrderBook
from ..models_optimized.avellaneda_numba import AvellanedaModelNumba
from ..ipc.message_types import TradeEvent, OrderRequest, FillEvent, MessageType
from ..utils.perf_utils import cpu_affinity, gc_controller

class StrategyProcess:
    """Dedicated process for trading strategy and decision making"""
    
    def __init__(self,
                 shared_book_name: str,
                 trade_queue: mp.Queue,
                 order_queue: mp.Queue,
                 fill_queue: mp.Queue,
                 config: Dict[str, Any],
                 cpu_core: int = 2):
        
        self.shared_book_name = shared_book_name
        self.trade_queue = trade_queue
        self.order_queue = order_queue
        self.fill_queue = fill_queue
        self.config = config
        self.cpu_core = cpu_core
        self.should_stop = False
        
        # Strategy components
        self.model: Optional[AvellanedaModelNumba] = None
        self.shared_book: Optional[SharedOrderBook] = None
        
        # State tracking
        self.current_position = 0.0
        self.inventory = 0.0
        self.last_mid_price = 0.0
        self.last_sequence = 0
        
        # Performance tracking
        self.iteration_count = 0
        self.last_quote_time = 0
        
    def run(self):
        """Main process entry point"""
        # Pin to CPU core
        cpu_affinity.pin_to_core(self.cpu_core)
        
        # Set signal handlers
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
        
        # Initialize components
        self.shared_book = SharedOrderBook(self.shared_book_name, create=False)
        self.model = AvellanedaModelNumba(self.config['avellaneda'])
        
        print(f"StrategyProcess started on core {self.cpu_core}")
        
        # Main processing loop
        self._main_loop()
        
        print("StrategyProcess shutting down")
    
    def _main_loop(self):
        """High-frequency main processing loop"""
        gc_iterations = 0
        
        while not self.should_stop:
            # Disable GC for performance-critical section
            with gc_controller.disabled():
                # Process fill events first (highest priority)
                self._process_fill_events()
                
                # Process trade events
                self._process_trade_events()
                
                # Check for book updates and generate quotes
                self._check_book_updates()
                
                # Increment iteration counter
                self.iteration_count += 1
                gc_iterations += 1
            
            # Controlled GC every 10000 iterations
            if gc_iterations >= 10000:
                gc.collect()
                gc_iterations = 0
            
            # Minimal sleep to prevent 100% CPU usage
            time.sleep(0.0001)  # 0.1ms
    
    def _process_fill_events(self):
        """Process order fill notifications"""
        try:
            while True:
                fill_event = self.fill_queue.get_nowait()
                self._handle_fill(fill_event)
        except:
            pass  # Queue empty
    
    def _process_trade_events(self):
        """Process market trade events"""
        try:
            while True:
                trade_event = self.trade_queue.get_nowait()
                self._handle_trade(trade_event)
        except:
            pass  # Queue empty
    
    def _check_book_updates(self):
        """Check for order book updates and generate quotes if needed"""
        current_sequence = int(self.shared_book.header['sequence'])
        
        if current_sequence != self.last_sequence:
            self.last_sequence = current_sequence
            
            # Extract mid price
            mid_price = float(self.shared_book.header['mid_price'])
            
            if mid_price > 0 and mid_price != self.last_mid_price:
                self.last_mid_price = mid_price
                self.model.update_price(mid_price)
                
                # Check if we should generate new quotes
                current_time = time.time()
                time_since_last_quote = current_time - self.last_quote_time
                
                if time_since_last_quote >= self.config['quote_timing']['min_interval']:
                    self._generate_quotes(mid_price)
                    self.last_quote_time = current_time
    
    def _handle_fill(self, fill_event: FillEvent):
        """Handle order fill event"""
        fill_size = fill_event.fill_size
        if fill_event.side == "buy":
            self.current_position += fill_size
            self.inventory += fill_size
        else:
            self.current_position -= fill_size  
            self.inventory -= fill_size
            
        print(f"Fill processed: {fill_event.side} {fill_size} @ {fill_event.fill_price}, "
              f"position: {self.current_position}")
    
    def _handle_trade(self, trade_event: TradeEvent):
        """Handle market trade event"""
        # Update model with trade information
        self.model.update_price(trade_event.price)
        
        # Could add additional trade analysis here
        print(f"Trade: {trade_event.side} {trade_event.size} @ {trade_event.price}")
    
    def _generate_quotes(self, mid_price: float):
        """Generate new quotes using the A-S model"""
        try:
            # Calculate optimal quotes
            quotes = self.model.calculate_quotes(
                mid_price=mid_price,
                inventory=self.inventory,
                current_position=self.current_position
            )
            
            # Risk checks
            if abs(self.current_position) >= self.config['risk']['max_position']:
                print(f"Position limit reached: {self.current_position}")
                return
            
            # Generate order requests
            self._send_quote_orders(quotes)
            
        except Exception as e:
            print(f"Quote generation error: {e}")
    
    def _send_quote_orders(self, quotes: Dict[str, Any]):
        """Send quote orders to execution process"""
        try:
            # Create bid order
            bid_order = OrderRequest(
                instrument=self.config['market']['underlying'],
                side="buy",
                price=quotes['bid_price'],
                size=quotes['bid_sizes'][0],  # Use first level size
                order_type="limit",
                post_only=True,
                label=self.config['market']['label'],
                request_id=f"bid_{int(time.time() * 1000000)}"
            )
            
            # Create ask order  
            ask_order = OrderRequest(
                instrument=self.config['market']['underlying'],
                side="sell", 
                price=quotes['ask_price'],
                size=quotes['ask_sizes'][0],  # Use first level size
                order_type="limit",
                post_only=True,
                label=self.config['market']['label'],
                request_id=f"ask_{int(time.time() * 1000000)}"
            )
            
            # Send to execution process
            self.order_queue.put(bid_order)
            self.order_queue.put(ask_order)
            
            print(f"Quotes sent: bid {quotes['bid_price']:.2f} @ {quotes['bid_sizes'][0]:.3f}, "
                  f"ask {quotes['ask_price']:.2f} @ {quotes['ask_sizes'][0]:.3f}")
            
        except Exception as e:
            print(f"Order sending error: {e}")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        print(f"StrategyProcess received signal {signum}")
        self.should_stop = True

def start_strategy_process(shared_book_name: str,
                         trade_queue: mp.Queue,
                         order_queue: mp.Queue,
                         fill_queue: mp.Queue,
                         config: Dict[str, Any],
                         cpu_core: int = 2) -> mp.Process:
    """Factory function to start strategy process"""
    
    def target():
        proc = StrategyProcess(
            shared_book_name, trade_queue, order_queue, 
            fill_queue, config, cpu_core
        )
        proc.run()
    
    process = mp.Process(target=target, name="StrategyProcess")
    process.start()
    return process
```

---

## Phase 4: Execution Process Separation (Days 13-16)

### Step 4.1: Create Cython-Optimized Order Manager
**Objective**: High-performance order state management
**Risk Level**: ⚠️ MEDIUM - New order management implementation

**Create**: `thalex_py/Thalex_modular/mft/models_optimized/order_manager_cython.pyx`
```cython
# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False

import time
from typing import Dict, Optional, Set
from enum import Enum

cdef enum OrderStatus:
    PENDING = 0
    OPEN = 1
    FILLED = 2
    CANCELLED = 3
    REJECTED = 4

cdef struct OrderInfo:
    char order_id[64]
    char instrument[32]
    char side[8]
    double price
    double size
    double filled_size
    double remaining_size
    int status
    double timestamp
    char request_id[64]

cdef class FastOrderManager:
    """Cython-optimized order state management"""
    
    cdef dict orders  # order_id -> OrderInfo
    cdef dict request_map  # request_id -> order_id
    cdef set active_orders
    cdef int max_orders
    cdef double last_cleanup
    
    def __init__(self, max_orders: int = 1000):
        self.orders = {}
        self.request_map = {}
        self.active_orders = set()
        self.max_orders = max_orders
        self.last_cleanup = time.time()
    
    cdef OrderInfo* _get_order(self, str order_id):
        """Get order by ID (fast C-level access)"""
        if order_id in self.orders:
            return &(<OrderInfo>self.orders[order_id])
        return NULL
    
    def add_order(self, str order_id, str instrument, str side, 
                  double price, double size, str request_id = ""):
        """Add new order to tracking"""
        cdef OrderInfo order
        
        # Convert strings to bytes for struct
        order_id_bytes = order_id.encode('utf-8')[:63]
        instrument_bytes = instrument.encode('utf-8')[:31]
        side_bytes = side.encode('utf-8')[:7]
        request_id_bytes = request_id.encode('utf-8')[:63]
        
        # Initialize order struct
        strncpy(order.order_id, order_id_bytes, 63)
        strncpy(order.instrument, instrument_bytes, 31)
        strncpy(order.side, side_bytes, 7)
        strncpy(order.request_id, request_id_bytes, 63)
        
        order.price = price
        order.size = size
        order.filled_size = 0.0
        order.remaining_size = size
        order.status = OrderStatus.PENDING
        order.timestamp = time.time()
        
        # Store in dictionaries
        self.orders[order_id] = order
        if request_id:
            self.request_map[request_id] = order_id
        self.active_orders.add(order_id)
        
        # Cleanup if needed
        if len(self.orders) > self.max_orders:
            self._cleanup_old_orders()
    
    def update_order_status(self, str order_id, int status):
        """Update order status"""
        cdef OrderInfo* order = self._get_order(order_id)
        if order != NULL:
            order.status = status
            if status in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED]:
                self.active_orders.discard(order_id)
    
    def add_fill(self, str order_id, double fill_size, double fill_price):
        """Process order fill"""
        cdef OrderInfo* order = self._get_order(order_id)
        if order == NULL:
            return False
        
        order.filled_size += fill_size
        order.remaining_size = order.size - order.filled_size
        
        if order.remaining_size <= 0.001:  # Fully filled
            order.status = OrderStatus.FILLED
            order.remaining_size = 0.0
            self.active_orders.discard(order_id)
        
        return True
    
    def get_active_orders(self):
        """Get all active orders"""
        active = []
        for order_id in self.active_orders:
            if order_id in self.orders:
                order = self.orders[order_id]
                active.append({
                    'order_id': order_id,
                    'instrument': order.instrument.decode('utf-8'),
                    'side': order.side.decode('utf-8'),
                    'price': order.price,
                    'size': order.size,
                    'filled_size': order.filled_size,
                    'remaining_size': order.remaining_size,
                    'status': order.status,
                    'timestamp': order.timestamp
                })
        return active
    
    def get_order_by_request_id(self, str request_id):
        """Get order by request ID"""
        if request_id in self.request_map:
            order_id = self.request_map[request_id]
            return self.get_order_info(order_id)
        return None
    
    def get_order_info(self, str order_id):
        """Get order information"""
        if order_id in self.orders:
            order = self.orders[order_id]
            return {
                'order_id': order_id,
                'instrument': order.instrument.decode('utf-8'),
                'side': order.side.decode('utf-8'),
                'price': order.price,
                'size': order.size,
                'filled_size': order.filled_size,
                'remaining_size': order.remaining_size,
                'status': order.status,
                'timestamp': order.timestamp,
                'request_id': order.request_id.decode('utf-8')
            }
        return None
    
    cdef void _cleanup_old_orders(self):
        """Clean up old completed orders"""
        cdef double current_time = time.time()
        cdef double cleanup_threshold = 300.0  # 5 minutes
        
        if current_time - self.last_cleanup < 60.0:  # Only cleanup every minute
            return
        
        orders_to_remove = []
        for order_id, order in self.orders.items():
            if (order.status in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED] and
                current_time - order.timestamp > cleanup_threshold):
                orders_to_remove.append(order_id)
        
        for order_id in orders_to_remove:
            del self.orders[order_id]
            # Clean up request map
            for req_id, oid in list(self.request_map.items()):
                if oid == order_id:
                    del self.request_map[req_id]
        
        self.last_cleanup = current_time
```

**Create Cython setup file**: `setup_cython.py`
```python
from setuptools import setup, Extension
from Cython.Build import cythonize

extensions = [
    Extension(
        "thalex_py.Thalex_modular.mft.models_optimized.order_manager_cython",
        ["thalex_py/Thalex_modular/mft/models_optimized/order_manager_cython.pyx"],
        extra_compile_args=["-O3", "-ffast-math"]
    )
]

setup(
    ext_modules=cythonize(extensions, compiler_directives={
        'language_level': 3,
        'boundscheck': False,
        'wraparound': False
    })
)
```

### Step 4.2: Create Execution Process
**Objective**: Dedicated process for order execution and management
**Risk Level**: ⚠️ HIGH - Critical order handling logic

**Create**: `thalex_py/Thalex_modular/mft/processes/execution_proc.py`
```python
import asyncio
import uvloop
import multiprocessing as mp
import signal
import time
import websockets
import json
from typing import Optional, Dict, Any

from ..models_optimized.order_manager_cython import FastOrderManager
from ..ipc.message_types import OrderRequest, FillEvent, MessageType
from ..utils.perf_utils import cpu_affinity
from ...models.keys import THALEX_KEY_ID, THALEX_PRIVATE_KEY

class ExecutionProcess:
    """Dedicated process for order execution and management"""
    
    def __init__(self,
                 order_queue: mp.Queue,
                 fill_queue: mp.Queue,
                 config: Dict[str, Any],
                 cpu_core: int = 3):
        
        self.order_queue = order_queue
        self.fill_queue = fill_queue
        self.config = config
        self.cpu_core = cpu_core
        self.should_stop = False
        
        # Order management
        self.order_manager: Optional[FastOrderManager] = None
        self.websocket: Optional[websockets.WebSocketServerProtocol] = None
        
        # Connection state
        self.is_connected = False
        self.last_heartbeat = 0
        
        # Rate limiting
        self.last_request_time = 0
        self.request_count = 0
        self.request_window_start = time.time()
        
    def run(self):
        """Main process entry point"""
        # Pin to CPU core
        cpu_affinity.pin_to_core(self.cpu_core)
        
        # Set signal handlers
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
        
        # Use uvloop for performance
        asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
        
        # Initialize order manager
        self.order_manager = FastOrderManager(max_orders=2000)
        
        print(f"ExecutionProcess started on core {self.cpu_core}")
        
        # Run async event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            loop.run_until_complete(self._async_main())
        finally:
            loop.close()
        
        print("ExecutionProcess shutting down")
    
    async def _async_main(self):
        """Main async event loop"""
        # Start WebSocket connection
        await self._connect_websocket()
        
        # Create concurrent tasks
        tasks = [
            asyncio.create_task(self._order_processing_loop()),
            asyncio.create_task(self._websocket_message_handler()),
            asyncio.create_task(self._heartbeat_loop())
        ]
        
        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            print(f"ExecutionProcess error: {e}")
        finally:
            if self.websocket:
                await self.websocket.close()
    
    async def _connect_websocket(self):
        """Connect to Thalex WebSocket"""
        uri = "wss://api.thalex.com/ws" if self.config['market']['network'].name == "PROD" else "wss://api.testnet.thalex.com/ws"
        
        try:
            self.websocket = await websockets.connect(uri)
            
            # Authenticate
            auth_msg = {
                "jsonrpc": "2.0",
                "method": "public/auth",
                "params": {
                    "grant_type": "client_credentials",
                    "client_id": THALEX_KEY_ID,
                    "client_secret": THALEX_PRIVATE_KEY
                },
                "id": self.config['call_ids']['login']
            }
            
            await self.websocket.send(json.dumps(auth_msg))
            response = await self.websocket.recv()
            
            if "access_token" in response:
                self.is_connected = True
                print("ExecutionProcess WebSocket connected and authenticated")
            else:
                print(f"Authentication failed: {response}")
                
        except Exception as e:
            print(f"WebSocket connection error: {e}")
            self.is_connected = False
    
    async def _order_processing_loop(self):
        """Process incoming order requests"""
        while not self.should_stop:
            try:
                # Check for new order requests (non-blocking)
                try:
                    order_request = self.order_queue.get_nowait()
                    await self._handle_order_request(order_request)
                except:
                    pass  # Queue empty
                
                await asyncio.sleep(0.001)  # 1ms sleep
                
            except Exception as e:
                print(f"Order processing error: {e}")
                await asyncio.sleep(0.1)
    
    async def _handle_order_request(self, order_request: OrderRequest):
        """Handle new order request"""
        if not self.is_connected:
            print("Cannot place order: not connected")
            return
        
        # Rate limiting check
        if not self._check_rate_limit():
            print("Rate limit exceeded, dropping order")
            return
        
        try:
            # Generate order ID
            order_id = f"{order_request.side}_{int(time.time() * 1000000)}"
            
            # Add to order manager
            self.order_manager.add_order(
                order_id=order_id,
                instrument=order_request.instrument,
                side=order_request.side,
                price=order_request.price,
                size=order_request.size,
                request_id=order_request.request_id
            )
            
            # Create order message
            order_msg = {
                "jsonrpc": "2.0",
                "method": "private/buy" if order_request.side == "buy" else "private/sell",
                "params": {
                    "instrument_name": order_request.instrument,
                    "amount": order_request.size,
                    "price": order_request.price,
                    "type": order_request.order_type,
                    "label": order_request.label,
                    "post_only": order_request.post_only
                },
                "id": int(time.time() * 1000000)
            }
            
            # Send order
            await self.websocket.send(json.dumps(order_msg))
            print(f"Order sent: {order_request.side} {order_request.size} @ {order_request.price}")
            
        except Exception as e:
            print(f"Order placement error: {e}")
    
    async def _websocket_message_handler(self):
        """Handle incoming WebSocket messages"""
        while not self.should_stop and self.websocket:
            try:
                message = await asyncio.wait_for(self.websocket.recv(), timeout=1.0)
                await self._process_websocket_message(message)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                print(f"WebSocket message error: {e}")
                await asyncio.sleep(1.0)
    
    async def _process_websocket_message(self, message: str):
        """Process incoming WebSocket message"""
        try:
            data = json.loads(message)
            
            # Handle different message types
            if "method" in data:
                method = data["method"]
                
                if method == "subscription":
                    params = data.get("params", {})
                    channel = params.get("channel", "")
                    
                    if "user.orders" in channel:
                        await self._handle_order_update(params.get("data", {}))
                    elif "user.trades" in channel:
                        await self._handle_fill_update(params.get("data", {}))
            
            elif "result" in data:
                # Handle order placement responses
                result = data["result"]
                if "order" in result:
                    await self._handle_order_response(result["order"])
                    
        except Exception as e:
            print(f"Message processing error: {e}")
    
    async def _handle_order_update(self, order_data: dict):
        """Handle order status updates"""
        try:
            order_id = order_data.get("order_id", "")
            order_state = order_data.get("order_state", "")
            
            if order_id and order_state:
                # Map Thalex states to internal states
                status_map = {
                    "open": 1,      # OPEN
                    "filled": 2,    # FILLED  
                    "cancelled": 3, # CANCELLED
                    "rejected": 4   # REJECTED
                }
                
                status = status_map.get(order_state, 0)
                self.order_manager.update_order_status(order_id, status)
                
                print(f"Order {order_id} status: {order_state}")
                
        except Exception as e:
            print(f"Order update error: {e}")
    
    async def _handle_fill_update(self, trade_data: dict):
        """Handle trade/fill updates"""
        try:
            order_id = trade_data.get("order_id", "")
            trade_size = float(trade_data.get("amount", 0))
            trade_price = float(trade_data.get("price", 0))
            side = trade_data.get("direction", "")
            trade_id = trade_data.get("trade_id", "")
            
            if order_id and trade_size > 0:
                # Update order manager
                self.order_manager.add_fill(order_id, trade_size, trade_price)
                
                # Create fill event
                fill_event = FillEvent(
                    order_id=order_id,
                    instrument=trade_data.get("instrument_name", ""),
                    fill_price=trade_price,
                    fill_size=trade_size,
                    remaining_size=0.0,  # Will be updated by order manager
                    side=side,
                    timestamp=time.time(),
                    trade_id=trade_id
                )
                
                # Send to strategy process
                self.fill_queue.put(fill_event)
                
                print(f"Fill processed: {side} {trade_size} @ {trade_price}")
                
        except Exception as e:
            print(f"Fill update error: {e}")
    
    async def _handle_order_response(self, order_data: dict):
        """Handle order placement response"""
        try:
            order_id = order_data.get("order_id", "")
            if order_id:
                self.order_manager.update_order_status(order_id, 1)  # OPEN
                print(f"Order confirmed: {order_id}")
                
        except Exception as e:
            print(f"Order response error: {e}")
    
    async def _heartbeat_loop(self):
        """Send periodic heartbeats"""
        while not self.should_stop:
            try:
                if self.is_connected and self.websocket:
                    current_time = time.time()
                    if current_time - self.last_heartbeat > 30:  # 30 second heartbeat
                        heartbeat_msg = {
                            "jsonrpc": "2.0",
                            "method": "public/test",
                            "params": {},
                            "id": self.config['call_ids']['heartbeat']
                        }
                        
                        await self.websocket.send(json.dumps(heartbeat_msg))
                        self.last_heartbeat = current_time
                
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                print(f"Heartbeat error: {e}")
                await asyncio.sleep(10)
    
    def _check_rate_limit(self) -> bool:
        """Check if we're within rate limits"""
        current_time = time.time()
        
        # Reset window if needed (1 minute windows)
        if current_time - self.request_window_start > 60:
            self.request_count = 0
            self.request_window_start = current_time
        
        # Check limit (180 requests per minute from config)
        if self.request_count >= self.config['connection']['rate_limit']:
            return False
        
        self.request_count += 1
        self.last_request_time = current_time
        return True
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        print(f"ExecutionProcess received signal {signum}")
        self.should_stop = True

def start_execution_process(order_queue: mp.Queue,
                          fill_queue: mp.Queue,
                          config: Dict[str, Any],
                          cpu_core: int = 3) -> mp.Process:
    """Factory function to start execution process"""
    
    def target():
        proc = ExecutionProcess(order_queue, fill_queue, config, cpu_core)
        proc.run()
    
    process = mp.Process(target=target, name="ExecutionProcess")
    process.start()
    return process
```

---

## Phase 5: Integration and Optimization (Days 17-19)

### Step 5.1: Create MFT Orchestrator
**Objective**: Coordinate all MFT processes with existing system integration
**Risk Level**: ⚠️ HIGH - Final integration point

**Create**: `thalex_py/Thalex_modular/mft/mft_orchestrator.py`
```python
import multiprocessing as mp
import signal
import time
import atexit
from typing import Optional, Dict, Any, List

from .processes.market_data_proc import start_market_data_process
from .processes.strategy_proc import start_strategy_process  
from .processes.execution_proc import start_execution_process
from .shared.shared_book import SharedOrderBook
from .ipc.message_types import MessageType

class MFTOrchestrator:
    """Orchestrates all MFT processes"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.processes: List[mp.Process] = []
        self.shared_book: Optional[SharedOrderBook] = None
        
        # IPC queues
        self.trade_queue = mp.Queue(maxsize=10000)
        self.order_queue = mp.Queue(maxsize=1000) 
        self.fill_queue = mp.Queue(maxsize=1000)
        
        # Shared memory name
        self.shared_book_name = "thalex_mft_book"
        
        # Process management
        self.is_running = False
        
        # Register cleanup
        atexit.register(self.stop)
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
    
    def start(self):
        """Start all MFT processes"""
        if self.is_running:
            return
        
        print("Starting MFT system...")
        
        try:
            # Create shared memory
            self.shared_book = SharedOrderBook(self.shared_book_name, create=True)
            
            # Start market data process (Core 1)
            market_proc = start_market_data_process(
                shared_book_name=self.shared_book_name,
                trade_queue=self.trade_queue,
                cpu_core=1
            )
            self.processes.append(market_proc)
            
            # Start strategy process (Core 2)
            strategy_proc = start_strategy_process(
                shared_book_name=self.shared_book_name,
                trade_queue=self.trade_queue,
                order_queue=self.order_queue,
                fill_queue=self.fill_queue,
                config=self.config,
                cpu_core=2
            )
            self.processes.append(strategy_proc)
            
            # Start execution process (Core 3)
            execution_proc = start_execution_process(
                order_queue=self.order_queue,
                fill_queue=self.fill_queue,
                config=self.config,
                cpu_core=3
            )
            self.processes.append(execution_proc)
            
            self.is_running = True
            print(f"MFT system started with {len(self.processes)} processes")
            
            # Monitor process health
            self._monitor_processes()
            
        except Exception as e:
            print(f"Failed to start MFT system: {e}")
            self.stop()
            raise
    
    def stop(self):
        """Stop all MFT processes"""
        if not self.is_running:
            return
        
        print("Stopping MFT system...")
        
        # Signal all processes to stop gracefully
        for process in self.processes:
            if process.is_alive():
                process.terminate()
        
        # Wait for graceful shutdown
        for process in self.processes:
            process.join(timeout=5)
            if process.is_alive():
                print(f"Force killing process {process.name}")
                process.kill()
        
        # Cleanup shared memory
        if self.shared_book:
            try:
                self.shared_book.shm.unlink()
            except:
                pass
        
        self.processes.clear()
        self.is_running = False
        print("MFT system stopped")
    
    def _monitor_processes(self):
        """Monitor process health"""
        while self.is_running:
            dead_processes = []
            
            for i, process in enumerate(self.processes):
                if not process.is_alive():
                    dead_processes.append((i, process))
            
            if dead_processes:
                print(f"Detected {len(dead_processes)} dead processes")
                for i, process in dead_processes:
                    print(f"Process {process.name} (PID {process.pid}) died")
                
                # For now, stop the entire system if any process dies
                # In production, could implement restart logic
                self.stop()
                break
            
            time.sleep(1)  # Check every second
    
    def get_status(self) -> Dict[str, Any]:
        """Get system status"""
        status = {
            'is_running': self.is_running,
            'processes': [],
            'queues': {
                'trade_queue_size': self.trade_queue.qsize(),
                'order_queue_size': self.order_queue.qsize(),
                'fill_queue_size': self.fill_queue.qsize()
            }
        }
        
        for process in self.processes:
            status['processes'].append({
                'name': process.name,
                'pid': process.pid,
                'is_alive': process.is_alive()
            })
        
        return status
    
    def get_shared_book_data(self) -> Dict[str, Any]:
        """Get current shared book data"""
        if not self.shared_book:
            return {}
        
        return {
            'sequence': int(self.shared_book.header['sequence']),
            'timestamp': int(self.shared_book.header['timestamp']),
            'mid_price': float(self.shared_book.header['mid_price']),
            'spread': float(self.shared_book.header['spread'])
        }
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        print(f"MFTOrchestrator received signal {signum}")
        self.stop()

# Global orchestrator instance for integration
_orchestrator: Optional[MFTOrchestrator] = None

def get_mft_orchestrator(config: Dict[str, Any]) -> MFTOrchestrator:
    """Get or create global MFT orchestrator"""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = MFTOrchestrator(config)
    return _orchestrator
```

### Step 5.2: Final Integration with AvellanedaQuoter
**Objective**: Complete integration with existing system
**Risk Level**: ⚠️ HIGH - Full system integration

**Modify**: `thalex_py/Thalex_modular/avellaneda_quoter.py`

**Add imports at top**:
```python
from .mft.mft_orchestrator import get_mft_orchestrator
```

**Add to AvellanedaQuoter.__init__() method**:
```python
# MFT system integration
USE_MFT_SYSTEM = True  # Master enable/disable flag

if USE_MFT_SYSTEM:
    self.mft_orchestrator = get_mft_orchestrator(self.config)
else:
    self.mft_orchestrator = None
```

**Add new methods**:
```python
async def start_mft_system(self):
    """Start MFT multi-process system"""
    if self.mft_orchestrator:
        try:
            # Start in a separate thread to avoid blocking
            import threading
            def start_mft():
                self.mft_orchestrator.start()
            
            mft_thread = threading.Thread(target=start_mft, daemon=True)
            mft_thread.start()
            
            # Wait a moment for startup
            await asyncio.sleep(2)
            
            self.logger.info("MFT system started")
            
        except Exception as e:
            self.logger.error(f"Failed to start MFT system: {e}")

async def stop_mft_system(self):
    """Stop MFT multi-process system"""
    if self.mft_orchestrator:
        self.mft_orchestrator.stop()
        self.logger.info("MFT system stopped")

def get_mft_status(self):
    """Get MFT system status"""
    if self.mft_orchestrator:
        return self.mft_orchestrator.get_status()
    return {'is_running': False}
```

**Modify main() method startup**:
```python
# Add after existing initialization
if USE_MFT_SYSTEM:
    await self.start_mft_system()
```

**Modify shutdown sequence**:
```python
# Add to cleanup
if USE_MFT_SYSTEM:
    await self.stop_mft_system()
```

---

## Validation and Testing Strategy

### Performance Benchmarking
```python
# Create: benchmark_mft.py
import time
import asyncio
from thalex_py.Thalex_modular.config.market_config import BOT_CONFIG
from thalex_py.Thalex_modular.mft.mft_orchestrator import MFTOrchestrator

async def benchmark_system():
    """Benchmark MFT system performance"""
    
    orchestrator = MFTOrchestrator(BOT_CONFIG)
    
    # Measure startup time
    start_time = time.time()
    orchestrator.start()
    startup_time = time.time() - start_time
    
    print(f"MFT startup time: {startup_time:.3f}s")
    
    # Monitor for 30 seconds
    for i in range(30):
        status = orchestrator.get_status()
        book_data = orchestrator.get_shared_book_data()
        
        print(f"Iteration {i}: Sequence {book_data.get('sequence', 0)}")
        await asyncio.sleep(1)
    
    orchestrator.stop()

if __name__ == "__main__":
    asyncio.run(benchmark_system())
```

### Rollback Strategy
1. **Feature Flags**: All MFT integration controlled by `USE_MFT_SYSTEM` flag
2. **Parallel Running**: Both old and new systems can run simultaneously for comparison
3. **Gradual Migration**: Each process can be enabled/disabled independently
4. **Performance Monitoring**: Built-in metrics to compare old vs new performance
5. **Emergency Shutdown**: Immediate fallback to original system if issues detected

### Success Metrics
- **Latency Improvement**: 50-70% reduction in tick-to-trade latency
- **Throughput Increase**: 2-3x improvement in quote generation rate
- **CPU Utilization**: Better distribution across cores
- **Memory Efficiency**: Reduced GC pauses and memory fragmentation
- **System Stability**: No degradation in reliability

---

## Conclusion

This implementation plan provides a methodical, low-risk approach to migrating from the current single-process architecture to a high-performance multi-process MFT system. Each phase builds incrementally on the previous one, with comprehensive testing and rollback capabilities at every step.

The key innovations include:

1. **Process Isolation**: True parallelism without GIL limitations
2. **Shared Memory**: Zero-copy data sharing for order book
3. **JIT Compilation**: C-level performance for mathematical computations
4. **Cython Optimization**: High-speed order management
5. **CPU Pinning**: Predictable performance characteristics

The result will be a Python-based trading system that achieves near-C++ performance levels while maintaining the development velocity and maintainability advantages of Python.