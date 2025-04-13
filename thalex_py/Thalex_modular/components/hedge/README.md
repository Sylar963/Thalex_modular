# Cross-Asset Hedging System

This module provides a robust system for cross-asset delta-neutral hedging to maintain hedged positions across multiple assets.

## Features

- Cross-asset hedging (e.g. BTC hedged with ETH and vice versa)
- Multiple hedging strategies (notional value or delta-neutral)
- Position tracking and fill processing
- Automatic hedge rebalancing
- PnL calculation for hedged positions
- State persistence for recovery

## Components

- **HedgeConfig**: Configuration for hedge pairs, correlation factors, and execution parameters
- **HedgeStrategy**: Strategies to calculate appropriate hedge sizes based on different methodologies
- **HedgeExecution**: Order execution and management
- **HedgeManager**: Main orchestration class that coordinates the entire system

## Quick Start

```python
from thalex_py.Thalex_modular.components.hedge import create_hedge_manager

# Create a hedge manager with the default configuration
hedge_manager = create_hedge_manager(strategy_type="notional")

# Start the manager (initiates background rebalancing)
hedge_manager.start()

# Set market prices
hedge_manager.update_market_price("BTC-PERP", 85500.0)
hedge_manager.update_market_price("ETH-PERP", 3200.0)

# When a position changes, update the hedge
result = hedge_manager.update_position("BTC-PERP", 1.0, 85500.0)  # 1 BTC long @ 85500

# Process fills from the main trading system
hedge_manager.on_fill(fill)  # Process a fill object

# Calculate PnL across all hedged positions
pnl_result = hedge_manager.calculate_portfolio_pnl()
print(f"Portfolio PnL: {pnl_result['total_pnl']}")

# Get information about current hedged positions
hedge_position = hedge_manager.get_hedged_position("BTC-PERP", "ETH-PERP")
print(f"Current hedge: {hedge_position.primary_position} BTC, {hedge_position.hedge_position} ETH")

# Stop the manager when done
hedge_manager.stop()
```

## Configuration

The default configuration is defined in `hedge_config.py`. You can customize it by:

1. Modifying the `DEFAULT_HEDGE_CONFIG` dictionary directly
2. Creating a JSON configuration file and passing its path to `create_hedge_manager()`
3. Creating a custom `HedgeConfig` instance and passing it to the manager

The configuration includes:

- **Hedge pairs**: Which assets can be hedged with which other assets
- **Correlation factors**: Ratios to calculate appropriate hedge sizes
- **Execution parameters**: Market vs. limit orders, timeouts, etc.
- **Rebalancing settings**: How often to check and rebalance hedges

## Extending the System

### Adding New Hedge Strategies

To add a new hedge strategy:

1. Create a new class that extends `HedgeStrategy`
2. Implement the required methods: `calculate_hedge_position()` and `should_rebalance()`
3. Add the strategy to the factory function in `hedge_strategy.py`

### Connecting to an Exchange

The `HedgeExecution` class is designed to work with any exchange client. To connect to a real exchange:

1. Create a client that interacts with your exchange's API
2. Pass this client to `create_hedge_manager()` as the `exchange_client` parameter

## Testing

Run the included test script to verify the system is working correctly:

```
python test_hedge_system.py
``` 