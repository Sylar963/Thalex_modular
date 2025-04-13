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
1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Create a Thalex account and generate API keys
4. Configure your API keys in `thalex_py/Thalex_modular/models/keys.py`
5. Start the bot: `python start_quoter.py`

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

### Extending the System
You can customize the system in several ways:
1. **Alternative Quote Strategies**: Implement different pricing models by modifying `AvellanedaMarketMaker`
2. **Enhanced Risk Controls**: Add features to `RiskManager` like portfolio VaR
3. **Custom Analytics**: Extend `MarketDataBuffer` with additional metrics

### Optimizing for Production
For live trading environments:
- Monitor memory usage with `PerformanceMonitor`
- Adjust log levels for production in `LoggerFactory`
- Fine-tune rate limiting parameters in `BOT_CONFIG`
- Consider deploying on low-latency infrastructure close to Thalex servers

## ⚠️ Risk Warning

Trading cryptocurrencies involves substantial risk of loss. This software is provided as-is with no guarantees or warranties. Always test thoroughly on testnet before running with real funds.

## 📄 License

This project is licensed under the MIT License. 