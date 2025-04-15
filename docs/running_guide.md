# Running Guide for Thalex SimpleQuoter

This guide explains how to install, configure, and run the Thalex SimpleQuoter bot for automated market making on Thalex exchange.

## Prerequisites

Before running the SimpleQuoter, ensure you have:

1. A Thalex exchange account with API access
2. API credentials (API key and secret)
3. Sufficient funds in your trading account
4. Python 3.8+ installed on your system
5. Git installed on your system

## Installation

### Method 1: From Source

```bash
# Clone the repository
git clone https://github.com/thalex/SimpleQuoter.git
cd SimpleQuoter

# Install dependencies
pip install -r requirements.txt
```

### Method 2: Using Docker

```bash
# Pull the Docker image
docker pull thalex/simplequoter:latest

# Or build locally
docker build -t thalex/simplequoter:latest .
```

## Basic Usage

### Running from Command Line

The SimpleQuoter can be started with a simple command:

```bash
# Using a configuration file
python simplequoter.py --config config.json

# Using environment variables
API_KEY=your_api_key API_SECRET=your_api_secret python simplequoter.py
```

### Running with Docker

```bash
# Using a configuration file
docker run -v $(pwd)/config.json:/app/config.json thalex/simplequoter:latest --config config.json

# Using environment variables
docker run -e API_KEY=your_api_key -e API_SECRET=your_api_secret thalex/simplequoter:latest
```

## Configuration Options

You can configure the SimpleQuoter in three ways:

1. **Configuration File**: Provide a JSON configuration file
2. **Environment Variables**: Set parameters as environment variables
3. **Command Line Arguments**: Pass parameters directly as command line arguments

See the [Configuration Guide](configuration_guide.md) for detailed parameter explanations.

### Minimal Configuration Example

```json
{
  "general": {
    "log_level": "INFO"
  },
  "exchange": {
    "api_key": "your_api_key",
    "api_secret": "your_api_secret",
    "testnet": false
  },
  "instruments": [
    {
      "symbol": "BTC-PERP",
      "enabled": true
    }
  ],
  "quoter": {
    "strategy": "basic",
    "min_spread_bps": 20,
    "max_spread_bps": 200,
    "order_levels": 1
  },
  "trading": {
    "order_size": {
      "BTC-PERP": 0.01
    }
  },
  "risk": {
    "max_position": {
      "BTC-PERP": 0.05
    },
    "max_daily_loss_usd": 1000
  }
}
```

## Running in Production

### Running as a Service

For long-term production deployment, it's recommended to run the SimpleQuoter as a system service:

#### Using Systemd (Linux)

Create a service file at `/etc/systemd/system/simplequoter.service`:

```
[Unit]
Description=Thalex SimpleQuoter
After=network.target

[Service]
User=trading
WorkingDirectory=/path/to/SimpleQuoter
ExecStart=/usr/bin/python3 simplequoter.py --config config.json
Restart=always
RestartSec=10
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
```

Enable and start the service:

```bash
sudo systemctl enable simplequoter
sudo systemctl start simplequoter
```

### Docker Compose for Production

Create a `docker-compose.yml` file:

```yaml
version: '3'
services:
  simplequoter:
    image: thalex/simplequoter:latest
    restart: always
    volumes:
      - ./config.json:/app/config.json
      - ./logs:/app/logs
    environment:
      - TZ=UTC
```

Run with Docker Compose:

```bash
docker-compose up -d
```

## Monitoring and Maintenance

### Viewing Logs

Logs are written to the console and to the `logs` directory:

```bash
# View recent logs
tail -f logs/simplequoter.log

# View Docker logs
docker logs -f simplequoter
```

### Health Checks

The SimpleQuoter includes an internal health check system:

```bash
# Check bot status
curl http://localhost:8080/health

# Get current positions
curl http://localhost:8080/positions
```

### Backing Up Data

It's recommended to regularly back up your configuration and log files:

```bash
# Create a backup
tar -czf simplequoter_backup_$(date +%Y%m%d).tar.gz config.json logs/

# Store in a secure location
cp simplequoter_backup_*.tar.gz /path/to/backup/location/
```

## Troubleshooting

### Common Issues

1. **Authentication Errors**:
   - Verify API credentials are correct
   - Check if API key has necessary permissions
   - Ensure system time is synchronized

2. **Connection Issues**:
   - Check network connectivity to Thalex exchange
   - Verify firewall allows outgoing connections
   - Try using testnet first to verify connectivity

3. **Order Placement Failures**:
   - Check account balance
   - Verify order size meets minimum requirements
   - Check for rate limiting (reduce update frequency)

### Debugging

Enable DEBUG logging for more detailed information:

```bash
# In config.json
{
  "general": {
    "log_level": "DEBUG"
  }
}

# Or via environment variable
LOG_LEVEL=DEBUG python simplequoter.py
```

## Upgrading

To upgrade the SimpleQuoter:

```bash
# From source
git pull
pip install -r requirements.txt

# Docker
docker pull thalex/simplequoter:latest
```

## Security Best Practices

1. Use API keys with minimal permissions required
2. Store API credentials securely (environment variables or secure vault)
3. Run the bot with least privilege user accounts
4. Regularly rotate API credentials
5. Set appropriate risk limits
6. Enable IP restrictions on API keys if possible

## Performance Tuning

See the [Tuning Guide](tuning_guide.md) for detailed performance optimization guidance.

## Getting Support

If you encounter issues:

1. Check the [Troubleshooting](#troubleshooting) section
2. Review logs for error messages
3. Check the GitHub repository issues section
4. Contact Thalex support with detailed information about your issue

## Appendix

### Command Line Arguments

```
Usage: python simplequoter.py [OPTIONS]

Options:
  --config TEXT          Path to configuration file
  --log-level TEXT       Logging level (DEBUG, INFO, WARNING, ERROR)
  --testnet BOOLEAN      Use testnet instead of production
  --symbol TEXT          Trading symbol (e.g., BTC-PERP)
  --api-key TEXT         API key for the exchange
  --api-secret TEXT      API secret for the exchange
  --strategy TEXT        Quoter strategy (basic, volatility_responsive)
  --min-spread-bps TEXT  Minimum spread in basis points
  --max-spread-bps TEXT  Maximum spread in basis points
  --help                 Show this message and exit
```

### Environment Variables

```
API_KEY                  API key for the exchange
API_SECRET               API secret for the exchange
LOG_LEVEL                Logging level (DEBUG, INFO, WARNING, ERROR)
TESTNET                  Use testnet if set to "true"
SYMBOL                   Trading symbol (e.g., BTC-PERP)
STRATEGY                 Quoter strategy (basic, volatility_responsive)
MIN_SPREAD_BPS           Minimum spread in basis points
MAX_SPREAD_BPS           Maximum spread in basis points
ORDER_SIZE               Order size for trading
MAX_POSITION             Maximum position size
MAX_DAILY_LOSS_USD       Maximum daily loss in USD
``` 