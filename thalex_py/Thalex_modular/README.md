# Thalex Avellaneda-Stoikov Market Maker

This is an implementation of the Avellaneda-Stoikov market making model for the Thalex cryptocurrency derivatives exchange. The market maker uses a sophisticated mathematical model to optimize quotes based on inventory risk and market conditions.

## Overview

The Avellaneda-Stoikov model is based on the paper "High-frequency Trading in a Limit Order Book" by Marco Avellaneda and Sasha Stoikov. The model optimizes market making by:

1. Calculating optimal bid-ask spreads based on volatility and inventory risk
2. Adjusting quote prices using a reservation price that accounts for inventory position
3. Dynamically sizing orders based on market conditions and risk limits
4. Incorporating technical analysis signals for enhanced quote placement

## Components

The market maker is built using a modular architecture with the following components:

- `AvellanedaMarketMaker`: Core implementation of the Avellaneda-Stoikov model
- `RiskManager`: Handles position limits and risk metrics
- `OrderManager`: Manages order placement and tracking
- `TechnicalAnalysis`: Provides market condition signals
- `AvellanedaQuoter`: Main class that coordinates all components

## Features

- **Optimal Spread Calculation**: Uses volatility and order flow to determine spreads
- **Inventory Management**: Adjusts quotes to manage position risk
- **Dynamic Sizing**: Adapts order sizes based on market conditions
- **Risk Controls**: Enforces position and notional limits
- **Technical Analysis**: Incorporates trend and volatility signals
- **Robust Error Handling**: Comprehensive error handling and logging
- **Auto-Recovery**: Automatic reconnection and state recovery

## Configuration

The market maker behavior can be customized through the following configuration files:

- `market_config.py`: Market-specific settings
- `trading_params.py`: Trading parameters
- `risk_limits.py`: Risk management settings
- `technical_params.py`: Technical analysis parameters

Key parameters include:

```python
TRADING_PARAMS = {
    "position_management": {
        "gamma": 0.1,               # Risk aversion
        "inventory_weight": 0.5,    # Inventory impact
        "position_fade_time": 300,  # Position mean reversion time
        "order_flow_intensity": 1.5 # Order arrival rate
    },
    "volatility": {
        "window": 100,        # Volatility calculation window
        "min_samples": 20,    # Minimum samples required
        "vol_floor": 0.001,   # Minimum volatility
        "vol_ceiling": 5.0    # Maximum volatility
    }
}
```

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

Logs are written to both console and `avellaneda_quoter.log`.

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

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 