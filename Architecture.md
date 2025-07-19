# Thalex SimpleQuoter Architecture

## Overview

The Thalex SimpleQuoter is a production-ready, high-performance cryptocurrency market making system implementing the **Avellaneda-Stoikov optimal market making model** for Bitcoin perpetuals and futures trading on the Thalex exchange. The system features async/await architecture, comprehensive risk management, volume-based predictive signals, and performance optimizations designed for algorithmic trading.

## Core Philosophy

- **Mathematical Foundation**: Complete Avellaneda-Stoikov implementation with VAMP (Volume Adjusted Market Pressure) extensions
- **Performance Optimization**: Object pooling, shared memory structures, lock-free queues, and sampling-based logging
- **Risk Management**: Multi-layered position limits, recovery systems, and emergency procedures with take profit triggers
- **Modular Design**: Clean component separation with event-driven communication and pluggable architecture
- **Production Monitoring**: Structured logging, performance metrics, and comprehensive observability

---

## Current Project Structure

```
Thalex_SimpleQuoter/
├── start_quoter.py                    # Main entry point (278 lines)
├── thalex_py/
│   └── Thalex_modular/
│       ├── avellaneda_quoter.py       # Core orchestrator (3,901 lines)
│       ├── components/                # Trading components
│       │   ├── avellaneda_market_maker.py  # A-S model with VAMP
│       │   ├── order_manager.py       # Individual order management
│       │   ├── risk_manager.py        # Position limits & recovery
│       │   ├── event_bus.py          # Inter-component communication
│       │   └── health_monitor.py     # System health monitoring
│       ├── models/                   # Data models & state
│       │   ├── data_models.py        # Core data structures
│       │   ├── position_tracker.py   # Dual position/portfolio tracking
│       │   └── keys.py              # API credentials management
│       ├── config/                   # Configuration system
│       │   ├── market_config.py      # Single source of truth (274 lines)
│       │   └── hedge/hedge_config.json # Hedging parameters
│       ├── ringbuffer/              # High-performance data structures
│       │   ├── market_data_buffer.py # Market data buffering
│       │   ├── volume_candle_buffer.py # Volume-based candles
│       │   └── fast_ringbuffer.py   # Lock-free ring buffer
│       ├── thalex_logging/          # Structured logging
│       │   ├── logger_factory.py    # Component-specific loggers
│       │   └── async_logger.py      # Non-blocking logging
│       ├── profiling/               # Performance monitoring
│       │   └── performance_tracer.py # Method-level profiling
│       ├── orderbook/              # Order book management
│       └── exchange_clients/       # Exchange client adapters
├── analysis/data/performance_monitor.py # Performance analytics
├── dashboard/monitor.py             # Real-time monitoring UI
├── metrics/                        # CSV performance data
└── logs/                          # Organized runtime logs
    ├── market/                    # Market maker decisions
    ├── orders/                    # Order execution
    ├── risk/                      # Risk events
    ├── hedge/                     # Hedging operations
    ├── performance/               # Performance data
    ├── exchange/                  # Exchange communications
    └── positions/                 # Position tracking
```

---

## Core Components

### 1. **AvellanedaQuoter** (`avellaneda_quoter.py`)
**Role**: Central orchestrator managing all trading system components (3,901 lines)
**Architecture**: High-performance async/await with HFT optimizations and `__slots__` memory management

**Core Responsibilities**:
- **WebSocket Management**: Thalex connection with 60s startup timeout, auto-reconnection, and graceful shutdown
- **Task Orchestration**: 7 concurrent async tasks (quote generation, risk monitoring, heartbeat, logging, profiling)
- **Market Data Processing**: Real-time ticker updates with 100ms caching and shared memory structures
- **Performance Optimization**: Object pools, message buffers (32KB-1MB), and 5% debug log sampling
- **Signal Handling**: Proper SIGINT/SIGTERM with 10s shutdown timeout and complete resource cleanup

**Performance Features**:
- **Object Pools**: Pre-allocated Order, Quote, and Ticker objects with configurable pool sizes
- **Shared Memory**: `SharedMarketData` ctypes structure for inter-process communication
- **ThreadSafeQueue**: Lock-free operations for high-frequency market data processing
- **Message Buffers**: Expandable byte arrays (32KB-1MB) with memory views for zero-copy processing
- **Log Sampling**: 5% sampling rate for debug operations, 30s intervals for major events

**Current Configuration**:
- **Network**: TEST (configurable via market_config.py)
- **Instrument**: BTC-PERPETUAL primary, BTC-25JUL25 futures
- **Quote Levels**: 3 levels with 150-tick spacing
- **Heartbeat**: 60-second intervals with WebSocket health monitoring

### 2. **AvellanedaMarketMaker** (`components/avellaneda_market_maker.py`)
**Role**: Complete Avellaneda-Stoikov implementation with VAMP extensions and take profit triggers
**Integration**: Deep integration with PositionTracker and VolumeCandle system for enhanced decision making

**Current Implementation**:
- **Gamma (Risk Aversion)**: 0.1 (configurable via market_config.py)
- **Kappa (Inventory Risk)**: 1.5 (market depth parameter)
- **Base Spread**: 14 ticks, Maximum: 75 ticks
- **Quote Levels**: 3 levels with 150-tick spacing
- **Size Multipliers**: [1.0, 1.5, 2.0, 2.5, 3.0] for progressive sizing

**Mathematical Framework**:

#### **Enhanced Optimal Spread Calculation**:
```python
# Avellaneda-Stoikov with market impact and volume adjustments
spread = base_spread * spread_multiplier * (1 + market_impact_factor + volume_adjustment)
delta_bid = delta_ask = spread / 2
```

#### **VAMP (Volume Adjusted Market Pressure)**:
```python
# Real implementation tracking aggressive volume
aggressive_buy_volume = sum(buy_trades_above_mid)
aggressive_sell_volume = sum(sell_trades_below_mid)
vamp = (aggressive_buy_volume - aggressive_sell_volume) / total_volume
reservation_price_adjustment = vamp * inventory_factor
```

#### **Take Profit Trigger System**:
- **Basic Triggers**: 7 bps spread profit threshold
- **Arbitrage Triggers**: $10 USD profit threshold for cross-instrument
- **Single Instrument**: $5 USD profit threshold
- **Check Interval**: 2-second monitoring for trigger conditions

#### **Volume Candle Integration**:
- **Threshold**: 1.0 BTC volume per candle
- **Predictive Signals**: Momentum, reversal, volatility, exhaustion (0-1 scale)
- **Parameter Adjustment**: Dynamic gamma/kappa based on volume signals
- **Max Age**: 300 seconds for prediction validity

### 3. **Risk Management System** (`components/risk_manager.py`)
**Production Risk Controls**: Real-time position monitoring with automated recovery system

**Current Risk Limits** (from market_config.py):
- **Max Position**: 20 BTC per instrument
- **Max Notional**: 100,000,000 USD total exposure
- **Stop Loss**: 6% from entry price
- **Max Drawdown**: 10% portfolio level
- **Take Profit**: Enabled with UPNL-based triggers

**Risk Recovery System**:
- **Breach Detection**: Real-time monitoring with immediate trading halt
- **Cooldown Period**: 9 seconds after risk breach before recovery assessment
- **Recovery Threshold**: 80% of risk limits for gradual re-entry
- **Recovery Steps**: 1-step process for quick resumption
- **Check Interval**: 30-second assessment of recovery conditions

**Advanced Features**:
- **Multi-Instrument Tracking**: Separate limits for perpetuals and futures
- **Event-Driven Callbacks**: Risk breach notifications to AvellanedaQuoter
- **Position-Based Quoting**: Inventory management with one-sided quotes
- **Emergency Procedures**: Automatic position flattening on severe breaches

### 4. **OrderManager** (`components/order_manager.py`)
**Role**: Individual limit order management with collision detection and semaphore control
**Architecture**: Post-only orders with tick-aligned pricing and comprehensive validation

**Current Implementation**:
- **Order Type**: Individual limit orders (not mass quotes)
- **Post-Only**: All orders use post-only execution to ensure maker rebates
- **Tick Alignment**: Automatic price alignment to exchange tick size (1 USD for BTC)
- **Size Validation**: 0.1 BTC minimum, 0.001 BTC increments
- **Collision Detection**: UUID-based order ID collision prevention

**Operation Control**:
- **Semaphore Limits**: Maximum 6 pending operations (configurable)
- **Operation Interval**: 0.5-second minimum between order operations
- **Retry Logic**: 2 maximum retry attempts for failed operations
- **Emergency Cancellation**: Immediate order cancellation for risk events

### 5. **Position Tracking System** (`models/position_tracker.py`)
**Dual Architecture**: PositionTracker for single instruments + PortfolioTracker for multi-instrument

**PositionTracker Features**:
- **Weighted Average Entry**: Automatic calculation on each fill
- **Thread-Safe Updates**: Concurrent-safe position modifications
- **Realized/Unrealized PnL**: Real-time mark-to-market calculations
- **Exit Price Tracking**: Complete trade lifecycle management

**PortfolioTracker Features**:
- **Multi-Instrument**: Separate tracking for BTC-PERPETUAL and futures
- **Cross-Asset Risk**: Combined notional exposure calculations
- **Portfolio Metrics**: Aggregate P&L, correlation tracking
- **Fill Integration**: Processing of Fill objects from exchange

### 6. **Volume Candle System** (`ringbuffer/volume_candle_buffer.py`)
**Role**: Real-time volume-based candle formation with predictive signal generation

**Current Configuration**:
- **Volume Threshold**: 1.0 BTC per candle completion
- **Maximum Candles**: 100 stored candles in ring buffer
- **Time Limit**: 300 seconds maximum before forced candle completion
- **Prediction Minimum**: 5+ candles required before generating signals

**Predictive Signal Generation**:
- **Momentum Signal**: Direction and strength (-1 to +1 scale)
- **Reversal Probability**: Market reversal likelihood (0 to 1 scale)
- **Volatility Prediction**: Expected volatility increase (0 to 1 scale)
- **Exhaustion Detection**: Market exhaustion signals (0 to 1 scale)

### 7. **Rescue Trading System** (Integrated in Market Maker)
**Role**: Automated averaging-down strategy for adverse position management

**Current Configuration**:
- **Trigger Threshold**: 0.3% price drop from average entry price
- **Profit Target**: 7 bps (0.07%) profit on averaged position
- **Maximum Steps**: 3 averaging-down steps before halt
- **Size Multiplier**: 1.0x base size for each rescue order
- **Minimum Interval**: 5 seconds between rescue orders

**Execution Strategy**:
- **Entry Orders**: Limit orders with "RescueTrade" label
- **Exit Orders**: Market orders with "RescueExit" label for guaranteed fills
- **Position Limits**: Maximum 2.0x base position limit for rescue positions
- **Cooldown**: 30 seconds after rescue exit before system re-entry

### 8. **Hedging System** (`components/hedge/`)
**Role**: Cross-asset hedging between BTC perpetuals and futures

**Current Configuration** (from hedge_config.json):
- **Hedge Pairs**: BTC-PERPETUAL ↔ BTC futures correlation tracking
- **Correlation Factors**: 0.85 for perpetual, 1.18 for futures
- **Rebalance Frequency**: 300 seconds (5 minutes)
- **Deviation Threshold**: 5% before rebalancing triggers
- **Execution**: Market orders with 30-second timeout

---

## Current Configuration Architecture

### **Single Source of Truth** (`market_config.py`):
```python
BOT_CONFIG = {
    "market": {
        "underlying": "BTC-PERPETUAL",
        "futures_instrument": "BTC-25JUL25", 
        "network": Network.TEST,  # TEST/LIVE configurable
    },
    "trading_strategy": {
        "avellaneda": {
            "gamma": 0.1,                    # Risk aversion parameter
            "kappa": 1.5,                    # Inventory risk factor
            "base_spread": 14.0,             # Base spread in ticks
            "max_spread": 75.0,              # Maximum spread limit
            "max_levels": 3,                 # Quote levels
            "level_spacing": 150,            # Tick spacing between levels
        }
    },
    "risk": {
        "max_position": 20,                  # BTC per instrument
        "max_notional": 100000000,           # USD total exposure
        "stop_loss_pct": 0.06,               # 6% stop loss
        "recovery_cooldown_seconds": 9,      # Risk recovery cooldown
        "risk_recovery_threshold": 0.8,      # 80% for recovery
    }
}
```

### **Command-Line Configuration Overrides**:
- `--gamma X`: Override risk aversion parameter
- `--kappa X`: Override inventory risk factor  
- `--levels X`: Override number of quote levels
- `--spacing X`: Override grid spacing in ticks
- `--vol-threshold X`: Override volume candle threshold

---

## Data Flow Architecture

### **Market Data Processing Pipeline**:
```
Thalex WebSocket → AvellanedaQuoter.handle_ticker_update() → SharedMarketData (ctypes)
       ↓                           ↓                              ↓
Object Pool Allocation → Message Buffer Processing → VolumeCandle.update_vamp()
       ↓                           ↓                              ↓
PerformanceTracer → Event Bus Notification → MarketMaker.update_quotes()
```

### **Quote Generation Flow**:
```
Market Data → VAMP Calculation → A-S Model + Volume Signals → Quote Generation → Order Placement
     ↓              ↓                    ↓                         ↓                ↓
Price Cache → Market Conditions → Predictive Adjustments → Risk Validation → Individual Limit Orders
     ↓              ↓                    ↓                         ↓                ↓
100ms Cache → Gamma/Kappa Updates → Position-Based Skew → Tick Alignment → Post-Only Orders
```

### **Risk Monitoring Flow**:
```
Fill Processing → PositionTracker.update_on_fill() → RiskManager.check_limits() → Recovery Assessment
       ↓                    ↓                              ↓                        ↓
Portfolio Update → Multi-Instrument P&L → Risk Breach Detection → Trading Halt/Resume
       ↓                    ↓                              ↓                        ↓
Event Bus → Take Profit Monitoring → Emergency Procedures → Gradual Recovery (80% threshold)
```

---

## Performance Optimizations

### **Memory Management**:
- **Object Pools**: Pre-allocated Order, Quote, and Ticker objects with configurable pool sizes
- **Shared Memory**: `SharedMarketData` ctypes structure for inter-process communication
- **Message Buffers**: Expandable 32KB-1MB byte arrays with memory views
- **Ring Buffers**: Fixed-size circular buffers for market data (100 candles)

### **High-Frequency Optimizations**:
- **Zero-Copy Processing**: Direct buffer manipulation without string copies
- **Log Sampling**: 5% sampling rate for debug operations, 30s for major events
- **Price Caching**: 100ms cache for market conditions to reduce computation
- **Lock-Free Queues**: ThreadSafeQueue for high-frequency market data processing

### **Operational Optimizations**:
- **Semaphore Control**: Maximum 6 pending order operations
- **Tick Alignment**: Automatic price alignment to exchange tick size (1 USD)
- **Post-Only Orders**: All orders use post-only to ensure maker rebates
- **Emergency Cancellation**: Immediate order cancellation for risk events

---

This architecture represents the current state of a production-ready, high-performance cryptocurrency market making system with comprehensive risk management, volume-based predictive signals, and performance optimizations designed for professional algorithmic trading.