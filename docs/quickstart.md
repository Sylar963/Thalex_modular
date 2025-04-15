# Quick Start Guide

This guide will help you quickly start using the Thalex SimpleQuoter trading system.

## Prerequisites

Ensure you have:
- Completed the [installation process](installation.md)
- Set up your API keys
- Configured your basic trading parameters

## Running the Bot

### Testnet Trading

To run the bot on the testnet (recommended for initial testing):

```bash
python start_quoter.py --test --levels 12 --vol-threshold 0.05
```

Parameters:
- `--test`: Run in test mode (using testnet)
- `--levels`: Number of price levels for the Avellaneda quoter (default: 10)
- `--vol-threshold`: Volatility threshold for adaptive parameters (default: 0.1)

### Production Trading

To run the bot in production mode:

```bash
python start_quoter.py --levels 12 --vol-threshold 0.05
```

**IMPORTANT**: Before running in production, thoroughly test your strategy on the testnet and ensure all risk management settings are properly configured.

## Key Command-line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--test` | Run on testnet | False |
| `--levels` | Number of price levels | 10 |
| `--vol-threshold` | Volatility threshold | 0.1 |
| `--order-size-factor` | Order size factor | 1.0 |
| `--order-size-limit` | Maximum order size | 1.0 |
| `--min-spread` | Minimum spread in basis points | 10 |
| `--max-spread` | Maximum spread in basis points | 100 |

## Monitoring the Bot

While running, the bot will output logs to both the console and log files:

- Order logs: `logs/orders/*.log`
- Market data logs: `logs/market/*.log`
- Trade execution logs: `logs/trades/*.log`

You can monitor these logs in real-time using:

```bash
tail -f logs/orders/latest.log
```

## Stopping the Bot

To stop the bot, press `Ctrl+C` in the terminal where it's running. The bot will attempt to cancel all open orders before shutting down.

## Common Use Cases

### Conservative Market Making

```bash
python start_quoter.py --test --levels 8 --vol-threshold 0.03 --min-spread 20 --max-spread 150
```

### Aggressive Market Making

```bash
python start_quoter.py --test --levels 15 --vol-threshold 0.08 --min-spread 5 --max-spread 80
```

## Next Steps

- For detailed configuration options, see the [Configuration Guide](configuration.md)
- To optimize your trading parameters, see the [Tuning Guide](tuning.md)
- To understand system components, see [Components Documentation](components/README.md) 