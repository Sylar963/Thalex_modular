# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a high-performance cryptocurrency market making bot implementing the Avellaneda-Stoikov optimal market making strategy for the Thalex derivatives exchange. The system trades Bitcoin perpetuals and futures using advanced mathematical models with real-time risk management and predictive volume candle analysis.

## Quick Start Commands

### Environment Setup
```bash
# Interactive setup (recommended for new users)
python setup_env.py

# Manual setup
cp .example.env .env
# Edit .env with your API credentials
```

### Running the Bot
```bash
# Start main trading bot
python start_quoter.py

# Run with custom parameters
python start_quoter.py --test --gamma 0.15 --levels 5 --spacing 20

# Run test for 120 seconds
python thalex_py/Thalex_modular/run_test.py
```

### Development Commands
```bash
# Install dependencies
pip install -r requirements.txt

# Run tests (via pytest)
pytest thalex_py/Thalex_modular/

# Run with specific configurations
python start_quoter.py --vol-threshold 0.5 --kappa 1.2
```

### Analysis and Monitoring
```bash
# Launch monitoring dashboard
python dashboard/monitor.py

# View performance analysis
jupyter notebook analysis/notebooks/trading_performance_analysis.ipynb

# Check system performance
python analysis/data/performance_monitor.py
```

## Architecture Overview

### Core Components
- **AvellanedaQuoter**: Main orchestrator managing all trading components (~3500 lines)
- **AvellanedaMarketMaker**: Implements the mathematical A-S model with volume candle predictions
- **OrderManager**: Handles individual order lifecycle and execution
- **RiskManager**: Multi-layered risk controls with recovery mechanisms
- **PositionTracker**: Centralized position and P&L management
- **VolumeBasedCandleBuffer**: Predictive market analysis using volume-based candle formation

### Key Directories
- `thalex_py/Thalex_modular/`: Core trading system
- `thalex_py/Thalex_modular/components/`: Trading strategy components
- `thalex_py/Thalex_modular/config/`: Configuration management
- `thalex_py/Thalex_modular/models/`: Data models and state management
- `thalex_py/Thalex_modular/ringbuffer/`: High-performance data structures
- `logs/`: Runtime logs organized by component (market/, orders/, risk/, hedge/, etc.)
- `metrics/`: Performance and trading metrics CSV files

## Configuration

Primary configuration is in `thalex_py/Thalex_modular/config/market_config.py`:

### Key Configuration Sections
- **BOT_CONFIG**: Main configuration containing market, trading strategy, and risk parameters
- **TRADING_CONFIG**: Derived config for Avellaneda parameters, quoting, and volume candles
- **RISK_LIMITS**: Position limits, stop loss, recovery settings
- **MARKET_CONFIG**: Instrument selection and network settings

### Important Parameters
- `gamma`: Risk aversion parameter (0.1 - 0.5 typical range)
- `kappa`: Market depth/liquidity parameter (1.0 - 2.0 typical range)
- `max_position`: Maximum position size
- `base_spread`: Base spread in ticks
- `volume_candle.threshold`: Volume required to complete predictive candles

## Environment Variables

Essential `.env` variables:
```bash
# For testnet (recommended for development)
THALEX_TEST_API_KEY_ID="your_testnet_key"
THALEX_TEST_PRIVATE_KEY="your_testnet_private_key"
TRADING_MODE="testnet"
NETWORK="test"

# Trading parameters
MAX_POSITION_SIZE="0.01"
BASE_QUOTE_SIZE="0.001"
PRIMARY_INSTRUMENT="BTC-PERPETUAL"
```

## Advanced Features

### Volume Candle Prediction System
The bot includes a sophisticated volume-based candle system that generates predictive signals:
- Momentum prediction (-1 to 1)
- Reversal probability (0 to 1)
- Volatility prediction (0 to 1)
- Exhaustion detection (0 to 1)

These signals dynamically adjust the Avellaneda-Stoikov parameters for enhanced performance.

### High-Performance Optimizations
- Object pooling for minimal GC pressure
- Lock-free data structures for HFT performance
- Shared memory IPC for inter-process communication
- Zero-copy message processing
- Vectorized calculations using NumPy

### Risk Management
- Multi-instrument position tracking (perpetuals + futures)
- Real-time P&L monitoring with UPNL-based take profit
- Recovery mode with gradual trading resumption
- Emergency position closure mechanisms
- Circuit breaker patterns for API rate limiting

## Common Issues and Solutions

### Network Configuration
- Always test on testnet first: set `NETWORK="test"` in config
- For production: set `NETWORK="live"` and use production API keys

### Performance Tuning
- Adjust `LOG_SAMPLING_RATE` in avellaneda_quoter.py for production (default 5%)
- Monitor memory usage via PerformanceMonitor
- Tune volume candle thresholds based on market conditions

### Risk Management
- Start with small position limits and gradually increase
- Monitor logs/risk/ directory for risk events
- Use recovery mode settings to handle temporary risk breaches

## Logging

Comprehensive logging system with component-specific logs:
- `logs/quoter.log`: Main trading activity
- `logs/market/market_maker.log`: Strategy decisions
- `logs/orders/order_manager.log`: Order execution
- `logs/risk/risk_manager.log`: Risk events
- `
- `logs/market/volume_candles.log`: Predictive signals

## Testing

Run comprehensive tests:
```bash
# Quick functional test (120 seconds)
python thalex_py/Thalex_modular/run_test.py

# Full test suite
pytest thalex_py/Thalex_modular/ -v

# Performance testing
python start_quoter.py --test --levels 1 --spacing 50
```