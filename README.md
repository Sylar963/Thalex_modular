# Thalex SimpleQuoter: Avellaneda Market Maker

A high-performance, production-ready cryptocurrency market making bot implementing the Avellaneda-Stoikov strategy optimized for [Thalex](https://www.thalex.com) derivatives exchange.

![Thalex](https://thalex.com/images/thalex-logo-white.svg)

## 📚 For General Users

### What is this?
This is an automated trading bot that provides continuous buy and sell quotes on cryptocurrency markets. It uses a mathematical model called "Avellaneda-Stoikov" which is widely used by professional market makers to determine optimal prices and sizes.

### Why use it?
- **Earn Trading Fees**: Market makers often receive fee rebates for providing liquidity
- **Capture the Spread**: Profit from the difference between buy and sell prices
- **Automated Trading**: Runs 24/7 without constant supervision
- **Risk Management**: Built-in features to control position size and risk

### Getting Started

#### Quick Setup (Recommended)
1. Clone this repository: `git clone <repository-url>`
2. Install dependencies: `pip install -r requirements.txt`
3. Create a Thalex account and generate API keys (testnet recommended for testing)
4. Run the interactive setup: `python setup_env.py`
5. Start the bot: `python start_quoter.py`

#### Manual Setup
1. Copy the environment template: `cp .example.env .env`
2. Edit `.env` with your API credentials and configuration
3. Configure your settings according to the comments in the file
4. Create necessary directories: `mkdir -p logs metrics backups profiles`
5. Start the bot: `python start_quoter.py`

#### Environment Variables
Your `.env` file should contain at minimum:
```bash
# Required for testnet
THALEX_TEST_API_KEY_ID="your_testnet_api_key_id"
THALEX_TEST_PRIVATE_KEY="your_testnet_private_key"
TRADING_MODE="testnet"
NETWORK="test"

# Trading parameters
MAX_POSITION_SIZE="0.01"
BASE_QUOTE_SIZE="0.001"
PRIMARY_INSTRUMENT="BTC-PERPETUAL"
```

⚠️ **Important**: 
- Never commit your `.env` file to version control
- Always test on testnet before using production
- Keep your API keys secure and never share them

### Simple Configuration
Basic settings can be adjusted in `thalex_py/Thalex_modular/config/market_config.py`:
- Trading pairs (BTC, ETH)
- Quote size and spread
- Risk limits
- Network (testnet/mainnet)

## 💻 For Developers & Quants

### Architecture
The system employs a modular, event-driven architecture with these components:
- `AvellanedaQuoter`: Central coordinator managing all components
- `AvellanedaMarketMaker`: Implements the quoting strategy with customizable parameters
- `OrderManager`: Handles order lifecycle and execution
- `RiskManager`: Enforces position limits and monitors risk metrics
- `MarketDataBuffer`: Stores and analyzes market data with efficient data structures

### Performance Features
- **Optimized WebSocket Management**: Reliable connection handling with auto-reconnect
- **Memory-Efficient Data Structures**: Ring buffers and object pools minimize allocations
- **Vectorized Calculations**: NumPy for high-performance numerical operations
- **Asynchronous Design**: Non-blocking I/O with asyncio for high throughput
- **Circuit Breaker Pattern**: Prevents API rate limit violations

### Avellaneda-Stoikov Implementation
The implementation uses the following parameters which can be tuned:
- `γ (gamma)`: Risk aversion parameter - higher values create wider spreads
- `σ (sigma)`: Volatility - dynamically estimated from market data
- `k`: Market depth/liquidity parameter
- `T`: Time horizon for the trading period

### Model Calibration
The system features dynamic parameter adjustment:
```python
# Snippet from gamma_update_task
new_gamma = self.market_maker.calculate_dynamic_gamma(volatility, market_impact)
self.market_maker.gamma = new_gamma
```

### Predictive Volume Candle System (DEVELOPERS ONLY)
The system includes a volume-based predictive component to enhance the Avellaneda-Stoikov model:

```python
# Volume-based candle completion triggers predictive signals
def _calculate_signals(self):
    # Generate momentum, reversal, volatility, and exhaustion signals
    self.signals["momentum"] = price_momentum * delta_momentum_factor
    # These signals drive parameter adjustments
```

#### Implementation Details
- **Volume Candle Buffer**: Creates candles based on accumulated volume rather than time
- **Delta Ratio Analysis**: Tracks buy/sell volume imbalance to detect market pressure
- **Predictive Signals**: Generates four key signals:
  - Momentum (direction and strength, -1 to 1)
  - Reversal probability (0 to 1)
  - Volatility prediction (0 to 1)
  - Exhaustion detection (0 to 1)
- **Parameter Adaptation**: Dynamically adjusts Avellaneda parameters:
  - γ (gamma) adjustment based on predicted volatility
  - Reservation price offsets based on momentum signals
  - κ (kappa) adjustment for market depth
  
#### Configuration
Volume candle system is configured in `TRADING_CONFIG["volume_candle"]`:
```python
"volume_candle": {
    "threshold": 0.1,               # Volume required to complete a candle (BTC)
    "max_candles": 200,             # Maximum candles to store
    "max_time_seconds": 180,        # Max time before forcing completion
    "enable_predictions": True,     # Enable predictive features
    "sensitivity": {                # Signal sensitivity parameters
        "momentum": 1.5,            # Momentum signal multiplier
        "reversal": 1.2,            # Reversal signal sensitivity
        "volatility": 1.3,          # Volatility prediction sensitivity
        "reservation_price": 1.5    # Reservation price adjustment factor
    }
}
```

#### Logging and Monitoring
- Volume candle completions and predictions are logged to `volume_candles.log`
- Parameter adjustments are logged to `market_maker.log`
- Debug logging is enabled for both components by default

#### Integration Points
- `AvellanedaMarketMaker._update_predictive_parameters()`: Applies prediction adjustments
- Volume candle buffer is updated on each trade via `update_vamp()` method
- Predictions are generated after accumulating 5+ candles

### Extending the System
You can customize the system in several ways:
1. **Alternative Quote Strategies**: Implement different pricing models by modifying `AvellanedaMarketMaker`
2. **Enhanced Risk Controls**: Add features to `RiskManager` like portfolio VaR
3. **Custom Analytics**: Extend `MarketDataBuffer` with additional metrics
4. **Predictive Signals**: Enhance `VolumeBasedCandleBuffer` with custom signals

### Optimizing for Production
For live trading environments:
- Monitor memory usage with `PerformanceMonitor`
- Adjust log levels for production in `LoggerFactory`
- Fine-tune rate limiting parameters in `BOT_CONFIG`
- Consider deploying on low-latency infrastructure close to Thalex servers
- Monitor `volume_candles.log` for prediction efficacy

## ⚠️ Risk Warning

Trading cryptocurrencies involves substantial risk of loss. This software is provided as-is with no guarantees or warranties. Always test thoroughly on testnet before running with real funds.

## 📄 License

This project is licensed under the MIT License. 