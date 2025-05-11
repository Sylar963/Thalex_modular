# Thalex Avellaneda-Stoikov Market Maker

This is an implementation of the Avellaneda-Stoikov market making model for the Thalex cryptocurrency derivatives exchange. The market maker uses a sophisticated mathematical model to optimize quotes based on inventory risk and market conditions.

## Overview

The Avellaneda-Stoikov model is based on the paper "High-frequency Trading in a Limit Order Book" by Marco Avellaneda and Sasha Stoikov. The model optimizes market making by:

1. Calculating optimal bid-ask spreads based on volatility and inventory risk
2. Adjusting quote prices using a reservation price that accounts for inventory position
3. Dynamically sizing orders based on market conditions and risk limits
4. Incorporating technical analysis signals for enhanced quote placement

## Project Structure

The market maker is built using a modular architecture with the following components:

```
thalex_py/Thalex_modular/
├── config/
│   └── market_config.py        # Unified configuration for all parameters
├── components/
│   ├── avellaneda_market_maker.py  # Core implementation of the A-S model
│   ├── order_manager.py        # Manages order placement and tracking
│   ├── risk_manager.py         # Handles position limits and risk metrics
│   └── technical_analysis.py   # Provides market condition signals
├── models/
│   ├── data_models.py          # Common data structures (Ticker, Quote, Order)
│   ├── position_tracker.py     # Tracks position size, entry price, and P&L
│   └── keys.py                 # API keys for authentication
├── avellaneda_quoter.py        # Main entry point and orchestrator
├── performance_monitor.py      # Records performance metrics
└── data_loader.py              # Utility for loading metrics for analysis
```

## Configuration Structure

The market maker uses a layered configuration approach defined in `thalex_py/Thalex_modular/config/market_config.py`:

1.  **Primary Configuration (`BOT_CONFIG`)**:
    *   This is the single source of truth for all settings, defined within `market_config.py`.
    *   All custom strategy and risk parameters should be configured here.

2.  **Modern Configuration Structures (`TRADING_CONFIG`, `RISK_LIMITS`)**:
    *   `TRADING_CONFIG`: Contains parameters related to trading strategy execution, Avellaneda model specifics, quote timing, market impact, and volatility settings. It is derived directly from `BOT_CONFIG`.
    *   `RISK_LIMITS`: Contains core risk parameters like maximum position, notional limits, stop-loss, and take-profit percentages. It is also derived directly from `BOT_CONFIG`.
    *   Other direct aliases like `MARKET_CONFIG` (for underlying, label, network) and `CONNECTION_CONFIG` (for connection retry delays) also exist, sourcing from `BOT_CONFIG`.

3.  **Legacy Configurations (Removed)**:
    *   Previously, several intermediate and proxy configurations (e.g., `ORDERBOOK_CONFIG`, `TRADING_PARAMS`, `RISK_CONFIG`, `QUOTING_CONFIG`, `PERFORMANCE_CONFIG`) existed for backward compatibility.
    *   These have been **removed**. All modules should now access configuration parameters through `BOT_CONFIG` (for fundamental settings) or the modern derived structures `TRADING_CONFIG` and `RISK_LIMITS`.

Modules should now directly import and use these modern structures:
```python
# Example of accessing modern configurations:
from thalex_py.Thalex_modular.config.market_config import TRADING_CONFIG, RISK_LIMITS, BOT_CONFIG

# Access Avellaneda model parameters from TRADING_CONFIG
gamma = TRADING_CONFIG["avellaneda"]["gamma"]
vol_floor = TRADING_CONFIG["volatility"]["floor"] # Volatility settings are nested under TRADING_CONFIG

# Access risk parameters from RISK_LIMITS
max_position = RISK_LIMITS["max_position"]

# Access fundamental settings directly from BOT_CONFIG if needed
# (though most operational parameters are in TRADING_CONFIG or RISK_LIMITS)
# e.g., if a specific bot behavior flag was in BOT_CONFIG["custom_flags"]
# custom_flag_value = BOT_CONFIG["custom_flags"]["my_flag"] 
```

## Execution Flow

1. **Initialization**:
   - The `main()` function in `avellaneda_quoter.py` initializes the Thalex client
   - Creates an `AvellanedaQuoter` instance which initializes all components
   - Sets up asyncio tasks for different operations

2. **Websocket Connection**:
   - The `listen_task()` establishes a websocket connection to Thalex
   - Authenticates and subscribes to relevant channels (orders, portfolio, trades, ticker)
   - Implements robust reconnection logic with exponential backoff

3. **Market Data Processing**:
   - The bot processes real-time updates from the websocket
   - Updates are routed to specific handlers (e.g., `handle_ticker_update()`)
   - The `TechnicalAnalysis` component calculates volatility and trend indicators

4. **Quote Generation**:
   - The `quote_task()` method runs the main quoting loop
   - Waits for market data updates via an asyncio condition variable
   - Uses `AvellanedaMarketMaker` to generate optimal quotes based on market conditions
   - Submits quotes through the `OrderManager`

5. **Risk Management**:
   - The `RiskManager` continuously evaluates position against risk limits
   - If limits are exceeded, the bot automatically disables quoting
   - Implements dynamic quoter behavior based on market conditions

6. **Performance Monitoring**:
   - The `PerformanceMonitor` tracks quotes, trades, and other metrics
   - Data is saved to CSV files for later analysis
   - The `performance_analysis.ipynb` notebook provides visualization and analysis

## Components

- **AvellanedaMarketMaker**: Core implementation of the Avellaneda-Stoikov model
- **RiskManager**: Handles position limits and risk metrics
- **OrderManager**: Manages order placement and tracking
- **TechnicalAnalysis**: Provides market condition signals
- **AvellanedaQuoter**: Main class that coordinates all components
- **PositionTracker**: Manages position tracking with FIFO accounting for accurate P&L

## Features

- **Optimal Spread Calculation**: Uses volatility and order flow to determine spreads
- **Inventory Management**: Adjusts quotes to manage position risk
- **Dynamic Sizing**: Adapts order sizes based on market conditions
- **Risk Controls**: Enforces position and notional limits
- **Technical Analysis**: Incorporates trend and volatility signals
- **Robust Error Handling**: Comprehensive error handling and logging
- **Auto-Recovery**: Automatic reconnection and state recovery
- **Rate Limiting**: Intelligent handling of exchange rate limits with cooldown periods

## Volume-Based Candle Buffer

The market maker includes an enhanced volume-based candle buffer system that builds predictive signals based on trade volume and price action:

- **Volume-Based Candles**: Creates candles based on traded volume thresholds rather than fixed time intervals
- **Exchange API Integration**: Can fetch market data directly from Thalex Exchange API for complete data
- **Hybrid Processing**: Processes both direct trades from the bot and exchange data with deduplication
- **Predictive Signals**: Generates signals for momentum, reversals, volatility, and exhaustion
- **Dynamic Parameter Adjustment**: Automatically adjusts Avellaneda model parameters based on signals
- **Thread Safety**: Implements proper locking and resource management for multi-threaded operation

To enable the exchange data integration feature, adjust the following settings in `market_config.py`:

```python
TRADING_CONFIG = {
    # ... other settings ...
    "volume_candle": {
        "threshold": 1.0,             # Volume required to complete a candle (BTC)
        "max_candles": 100,           # Maximum candles to store
        "max_time_seconds": 300,      # Maximum time before closing a candle
        "use_exchange_data": True,    # Enable exchange API integration
        "fetch_interval_seconds": 60, # How often to fetch exchange data
        "lookback_hours": 1,          # How far back to look for historical data
        "prediction_update_interval": 10.0  # How often to update predictions
    }
}
```

See the example in `examples/enhanced_candle_buffer_example.py` for a standalone demonstration of the enhanced volume candle buffer.

## Configuration

The market maker behavior is customized through the `BOT_CONFIG` dictionary within `thalex_py/Thalex_modular/config/market_config.py`. This primary configuration object is then used to populate the `TRADING_CONFIG` and `RISK_LIMITS` structures used by the bot's components.

Refer to `thalex_py/Thalex_modular/config/market_config.py` for the detailed structure of `BOT_CONFIG` and how `TRADING_CONFIG` and `RISK_LIMITS` are derived. Key parameters within `BOT_CONFIG` that influence trading and risk include (but are not limited to):

*   `BOT_CONFIG["trading_strategy"]["avellaneda"]`: For Avellaneda-Stoikov model parameters like `gamma`, `kappa`, `time_horizon`, `inventory_weight`, `min_spread`, `max_spread`, `base_size`, `max_levels`, `level_spacing`, etc.
*   `BOT_CONFIG["risk"]`: For risk parameters like `max_position`, `max_notional`, `stop_loss_pct`, `take_profit_pct`, `max_drawdown`, etc.
*   `BOT_CONFIG["trading_strategy"]["execution"]`: For execution details like `min_size`, `max_size`, `price_decimals`, `size_decimals`.
*   `BOT_CONFIG["trading_strategy"]["quote_timing"]`: For quote behavior like `min_interval`, `max_lifetime`.
*   `BOT_CONFIG["volatility_config"]` (if defined at `BOT_CONFIG` root, or typically `DEFAULT_VOLATILITY_CONFIG` is used within `TRADING_CONFIG`): For volatility calculation parameters like `window`, `floor`, `ceiling`.

Please see `market_config.py` for the exact paths and default values.

## Usage

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure API keys in `keys.py`:
```python
key_ids = {
    Network.TEST: "your_test_key_id",
    Network.PROD: "your_prod_key_id"
}
```

3. Run the market maker:
```bash
python avellaneda_quoter.py
```

## Monitoring

The market maker provides detailed logging of its operations:

- Quote updates and executions
- Risk metrics and position changes
- Technical analysis signals
- Error conditions and recovery attempts

Logs are written to both console and `avellaneda_quoter.log`. Additionally, metrics are recorded for analysis:

- Performance metrics saved to CSV files in the `metrics` directory
- Analysis provided through the `performance_analysis.ipynb` notebook
- Data loader utility for accessing and analyzing metrics

## Risk Management

The system implements multiple layers of risk management:

1. Position Limits
   - Maximum absolute position
   - Maximum notional exposure
   - Dynamic take-profit and stop-loss levels

2. Quote Controls
   - Minimum/maximum spread limits
   - Quote size adjustments based on inventory
   - Fast cancellation in volatile markets

3. Market Impact
   - Monitors price impact of orders
   - Adjusts quotes to minimize market impact
   - Implements quote fading for large positions

## Performance Optimization

The market maker includes several optimizations:

- Efficient quote updates using price movement thresholds
- Batched order operations to reduce API calls
- Asynchronous processing using asyncio
- Caching of market conditions and calculations
- Rate limit awareness with adaptive backoff

## Performance Optimizations

The system has been optimized for high-frequency trading with the following enhancements:

### Network Layer Optimizations

#### Message Processing Efficiency
- **Message Batching**: Implemented request batching to reduce the number of API calls, especially for quote updates
- **Streaming JSON Parsing**: Using `orjson` for faster parsing of incoming market data
- **Fast Serialization**: Using `ujson` for more efficient message creation

#### Connection Management
- **Automatic Reconnection**: Added exponential backoff logic for reliable reconnection
- **Connection Monitoring**: Improved heartbeat system to detect and recover from stalled connections
- **Circuit Breaker Pattern**: Implemented circuit breaker to handle exchange rate limits gracefully

#### Memory Optimizations
- **Object Pooling**: Pre-allocation of frequently used objects like Quote and SideQuote
- **Efficient Data Structures**: Using NumPy arrays for price history and metrics instead of lists
- **Memory Footprint Reduction**: Added `__slots__` to data classes to reduce memory overhead

#### Latency Reduction
- **Request Prioritization**: Order operations are given higher priority than market data operations
- **Separate Processing Paths**: Critical order operations bypass batching for lower latency
- **Speculative Prefetching**: Added caching for instrument data to reduce lookup times

### Usage

The optimized network layer is enabled by default. To take full advantage of these optimizations:

1. Initialize the client with:
```python
await thalex_client.initialize()
```

2. For high-throughput quote management, use the mass_quote API:
```python
await thalex.mass_quote(quotes=quotes, label="MyQuoter", post_only=True)
```

3. To manage rate limits, configure the client:
```python
thalex.rate_limit = 60  # Requests per minute
```

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 