# Configuration Guide for Thalex SimpleQuoter

This guide provides detailed information on configuring the Thalex SimpleQuoter bot for optimal market making performance. Each configuration parameter is explained in detail to help you tailor the bot to your trading strategy.

## Configuration Overview

The SimpleQuoter can be configured through a JSON configuration file, environment variables, or command-line arguments. The configuration is divided into several key sections:

1. General Settings
2. Exchange Connection
3. Instrument Configuration
4. Quoter Strategy Settings
5. Trading Parameters
6. Risk Management
7. Advanced Features

## Configuration Format

### Sample Full Configuration

```json
{
  "general": {
    "log_level": "INFO",
    "update_interval_ms": 1000,
    "health_check_port": 8080
  },
  "exchange": {
    "api_key": "your_api_key",
    "api_secret": "your_api_secret",
    "testnet": false,
    "retry_attempts": 3,
    "retry_delay_ms": 1000
  },
  "instruments": [
    {
      "symbol": "BTC-PERP",
      "enabled": true,
      "min_order_size": 0.001,
      "price_precision": 1,
      "size_precision": 3
    },
    {
      "symbol": "ETH-PERP",
      "enabled": true,
      "min_order_size": 0.01,
      "price_precision": 2,
      "size_precision": 2
    }
  ],
  "quoter": {
    "strategy": "volatility_responsive",
    "min_spread_bps": 20,
    "max_spread_bps": 200,
    "order_levels": 3,
    "level_spacing_bps": 20,
    "size_multiplier_per_level": 1.5
  },
  "trading": {
    "order_size": {
      "BTC-PERP": 0.01,
      "ETH-PERP": 0.1
    },
    "skew_percentage": {
      "BTC-PERP": 0,
      "ETH-PERP": 0
    },
    "order_type": "limit",
    "post_only": true,
    "cancel_threshold_bps": 10
  },
  "risk": {
    "max_position": {
      "BTC-PERP": 0.05,
      "ETH-PERP": 0.5
    },
    "max_notional_position_usd": 10000,
    "max_daily_loss_usd": 1000,
    "max_open_orders": 20,
    "position_based_skew": true
  },
  "volatility": {
    "lookback_periods": 20,
    "use_historical_vol": true,
    "vol_multiplier": 1.2,
    "min_vol_bps": 30,
    "max_vol_bps": 1000
  },
  "advanced": {
    "inventory_management": true,
    "inventory_target_percentage": 0,
    "inventory_skew_coefficient": 0.5,
    "dynamic_order_size": false,
    "market_impact_model": "linear",
    "market_impact_coefficient": 0.1,
    "aggressive_reposting": false
  }
}
```

## General Settings

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `log_level` | string | "INFO" | Logging level (DEBUG, INFO, WARNING, ERROR) |
| `update_interval_ms` | integer | 1000 | Frequency of order book updates in milliseconds |
| `health_check_port` | integer | 8080 | Port for health check HTTP server |

## Exchange Connection Settings

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `api_key` | string | - | Your Thalex API key |
| `api_secret` | string | - | Your Thalex API secret |
| `testnet` | boolean | false | Whether to connect to testnet or production |
| `retry_attempts` | integer | 3 | Number of retry attempts for API calls |
| `retry_delay_ms` | integer | 1000 | Delay between retry attempts in milliseconds |

## Instrument Configuration

Each instrument is configured separately within the `instruments` array:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `symbol` | string | - | Trading symbol (e.g., "BTC-PERP") |
| `enabled` | boolean | true | Whether this instrument is enabled for trading |
| `min_order_size` | float | - | Minimum order size for the instrument |
| `price_precision` | integer | - | Decimal precision for prices |
| `size_precision` | integer | - | Decimal precision for order sizes |

## Quoter Strategy Settings

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `strategy` | string | "basic" | Quoter strategy ("basic", "volatility_responsive", "inventory_management") |
| `min_spread_bps` | integer | 20 | Minimum bid-ask spread in basis points |
| `max_spread_bps` | integer | 200 | Maximum bid-ask spread in basis points |
| `order_levels` | integer | 1 | Number of order levels on each side |
| `level_spacing_bps` | integer | 20 | Spacing between order levels in basis points |
| `size_multiplier_per_level` | float | 1.0 | Size multiplier for each level (e.g., 1.5 means each level is 50% larger) |

## Trading Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `order_size` | object | - | Map of instrument symbols to order sizes |
| `skew_percentage` | object | - | Map of instrument symbols to skew percentages (-100 to 100, 0 = neutral) |
| `order_type` | string | "limit" | Order type ("limit", "post_only") |
| `post_only` | boolean | true | Whether to use post-only orders |
| `cancel_threshold_bps` | integer | 10 | Threshold to cancel and replace orders (basis points) |

## Risk Management

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_position` | object | - | Maximum position size per instrument |
| `max_notional_position_usd` | integer | 10000 | Maximum notional position in USD across all instruments |
| `max_daily_loss_usd` | integer | 1000 | Maximum daily loss in USD before stopping |
| `max_open_orders` | integer | 20 | Maximum number of open orders |
| `position_based_skew` | boolean | true | Whether to adjust quote skew based on current position |

## Volatility Settings

For volatility-responsive strategy:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `lookback_periods` | integer | 20 | Number of periods to look back for volatility calculation |
| `use_historical_vol` | boolean | true | Whether to use historical volatility or realized volatility |
| `vol_multiplier` | float | 1.2 | Multiplier applied to calculated volatility |
| `min_vol_bps` | integer | 30 | Minimum volatility in basis points |
| `max_vol_bps` | integer | 1000 | Maximum volatility in basis points |

## Advanced Settings

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `inventory_management` | boolean | true | Whether to use inventory management |
| `inventory_target_percentage` | integer | 0 | Target inventory as percentage of max position (-100 to 100, 0 = neutral) |
| `inventory_skew_coefficient` | float | 0.5 | How aggressively to skew based on inventory (0.0-1.0) |
| `dynamic_order_size` | boolean | false | Whether to dynamically adjust order size based on market conditions |
| `market_impact_model` | string | "linear" | Market impact model ("linear", "square_root") |
| `market_impact_coefficient` | float | 0.1 | Coefficient for market impact model |
| `aggressive_reposting` | boolean | false | Whether to aggressively repost orders that were filled |

## Environment Variables

All configuration parameters can be set via environment variables. The format is to use uppercase and underscores:

- General: `LOG_LEVEL`, `UPDATE_INTERVAL_MS`, `HEALTH_CHECK_PORT`
- Exchange: `API_KEY`, `API_SECRET`, `TESTNET`
- Quoter: `STRATEGY`, `MIN_SPREAD_BPS`, `MAX_SPREAD_BPS`
- Trading: `ORDER_SIZE_BTC_PERP`, `SKEW_PERCENTAGE_BTC_PERP`
- Risk: `MAX_POSITION_BTC_PERP`, `MAX_DAILY_LOSS_USD`

## Command Line Arguments

Key configuration parameters can be set via command line:

```
--config TEXT          Path to configuration file
--log-level TEXT       Logging level (DEBUG, INFO, WARNING, ERROR)
--testnet BOOLEAN      Use testnet instead of production
--symbol TEXT          Trading symbol (e.g., BTC-PERP)
--api-key TEXT         API key for the exchange
--api-secret TEXT      API secret for the exchange
--strategy TEXT        Quoter strategy (basic, volatility_responsive)
--min-spread-bps TEXT  Minimum spread in basis points
--max-spread-bps TEXT  Maximum spread in basis points
```

## Strategy-Specific Configurations

### Basic Strategy

The basic strategy uses fixed spread parameters without considering market volatility:

```json
"quoter": {
  "strategy": "basic",
  "min_spread_bps": 20,
  "max_spread_bps": 100,
  "order_levels": 1
}
```

### Volatility-Responsive Strategy

This strategy adjusts spreads based on market volatility:

```json
"quoter": {
  "strategy": "volatility_responsive",
  "min_spread_bps": 20,
  "max_spread_bps": 300,
  "order_levels": 3,
  "level_spacing_bps": 20
},
"volatility": {
  "lookback_periods": 20,
  "use_historical_vol": true,
  "vol_multiplier": 1.2,
  "min_vol_bps": 30,
  "max_vol_bps": 1000
}
```

### Inventory Management Strategy

This strategy adjusts quotes based on current inventory:

```json
"quoter": {
  "strategy": "inventory_management",
  "min_spread_bps": 20,
  "max_spread_bps": 200,
  "order_levels": 2
},
"advanced": {
  "inventory_management": true,
  "inventory_target_percentage": 0,
  "inventory_skew_coefficient": 0.5
}
```

## Configuration Tips

1. **Start Conservative**: Begin with wider spreads and smaller position sizes
2. **Testnet First**: Test your configuration on testnet before moving to production
3. **Incremental Changes**: Make one change at a time to understand its impact
4. **Monitor Logs**: Use DEBUG logging initially to understand bot behavior
5. **Backtest When Possible**: Use historical data to validate your strategy
6. **Update Regularly**: Markets change, so review your configuration regularly

## Configuration for Different Market Conditions

### High Volatility Markets

```json
"quoter": {
  "strategy": "volatility_responsive",
  "min_spread_bps": 50,
  "max_spread_bps": 500,
  "order_levels": 2
},
"trading": {
  "order_size": {
    "BTC-PERP": 0.005
  }
},
"risk": {
  "max_position": {
    "BTC-PERP": 0.02
  }
}
```

### Low Volatility Markets

```json
"quoter": {
  "strategy": "basic",
  "min_spread_bps": 10,
  "max_spread_bps": 100,
  "order_levels": 3,
  "level_spacing_bps": 10
},
"trading": {
  "order_size": {
    "BTC-PERP": 0.02
  }
}
```

### Illiquid Markets

```json
"quoter": {
  "min_spread_bps": 40,
  "max_spread_bps": 300,
  "order_levels": 1
},
"trading": {
  "cancel_threshold_bps": 20,
  "post_only": true
}
``` 