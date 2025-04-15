# Configuration Guide

This guide explains how to configure the Thalex SimpleQuoter trading system for optimal performance.

## Configuration Files

The system uses several configuration files:

- `config.json` - Main configuration file
- `.env` - Environment variables for API keys and secrets

## Main Configuration (config.json)

Example `config.json`:

```json
{
  "general": {
    "log_level": "INFO",
    "max_open_orders": 100,
    "reconnect_interval": 5
  },
  "exchange": {
    "name": "thalex",
    "testnet": true
  },
  "trading": {
    "instruments": ["BTC-PERP", "ETH-PERP"],
    "max_position_size": {
      "BTC-PERP": 1.0,
      "ETH-PERP": 10.0
    },
    "risk_limit_percentage": 0.8,
    "order_size_base": {
      "BTC-PERP": 0.01,
      "ETH-PERP": 0.1
    }
  },
  "quoter": {
    "levels": 10,
    "min_spread_bps": 10,
    "max_spread_bps": 100,
    "vol_threshold": 0.1,
    "inventory_risk_aversion": 0.9,
    "order_size_factor": 1.0,
    "order_size_limit": 1.0
  }
}
```

### Configuration Sections

#### General Settings

```json
"general": {
  "log_level": "INFO",
  "max_open_orders": 100,
  "reconnect_interval": 5
}
```

- `log_level`: Sets the logging verbosity (`DEBUG`, `INFO`, `WARNING`, `ERROR`)
- `max_open_orders`: Maximum number of open orders allowed
- `reconnect_interval`: Time in seconds to wait before reconnecting after a disconnect

#### Exchange Settings

```json
"exchange": {
  "name": "thalex",
  "testnet": true
}
```

- `name`: Exchange name (currently only 'thalex' is supported)
- `testnet`: Whether to use testnet (`true`) or production (`false`)

#### Trading Settings

```json
"trading": {
  "instruments": ["BTC-PERP", "ETH-PERP"],
  "max_position_size": {
    "BTC-PERP": 1.0,
    "ETH-PERP": 10.0
  },
  "risk_limit_percentage": 0.8,
  "order_size_base": {
    "BTC-PERP": 0.01,
    "ETH-PERP": 0.1
  }
}
```

- `instruments`: List of instruments to trade
- `max_position_size`: Maximum position size per instrument
- `risk_limit_percentage`: Percentage of exchange risk limit to use
- `order_size_base`: Base order size per instrument

#### Quoter Settings

```json
"quoter": {
  "levels": 10,
  "min_spread_bps": 10,
  "max_spread_bps": 100,
  "vol_threshold": 0.1,
  "inventory_risk_aversion": 0.9,
  "order_size_factor": 1.0,
  "order_size_limit": 1.0
}
```

- `levels`: Number of price levels
- `min_spread_bps`: Minimum spread in basis points
- `max_spread_bps`: Maximum spread in basis points
- `vol_threshold`: Volatility threshold for adapting spreads
- `inventory_risk_aversion`: Risk aversion parameter (0-1)
- `order_size_factor`: Multiplier for base order size
- `order_size_limit`: Maximum order size as a multiple of base size

## Environment Variables (.env)

Create a `.env` file in the root directory with the following variables:

```
THALEX_API_KEY=your_api_key
THALEX_API_SECRET=your_api_secret
THALEX_API_PASSPHRASE=your_api_passphrase
```

## Command-Line Override

Many configuration parameters can be overridden via command-line arguments:

```bash
python start_quoter.py --test --levels 12 --vol-threshold 0.05
```

See the [Quick Start Guide](quickstart.md) for a list of command-line arguments.

## Configuration Precedence

The system uses the following order of precedence:

1. Command-line arguments (highest priority)
2. Environment variables
3. Configuration file values
4. Default values (lowest priority)

## Example Configurations

### Conservative Market Making

```json
"quoter": {
  "levels": 8,
  "min_spread_bps": 20,
  "max_spread_bps": 150,
  "vol_threshold": 0.03,
  "inventory_risk_aversion": 0.95,
  "order_size_factor": 0.8,
  "order_size_limit": 0.5
}
```

### Aggressive Market Making

```json
"quoter": {
  "levels": 15,
  "min_spread_bps": 5,
  "max_spread_bps": 80,
  "vol_threshold": 0.08,
  "inventory_risk_aversion": 0.7,
  "order_size_factor": 1.2,
  "order_size_limit": 1.5
}
``` 