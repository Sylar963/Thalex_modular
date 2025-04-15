# Installation Guide

This guide covers the installation and setup process for the Thalex SimpleQuoter trading system.

## Prerequisites

- Python 3.8 or higher
- Pip package manager
- Git (for cloning the repository)
- A Thalex API key (for live trading)

## Installation Steps

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/Thalex_SimpleQuouter.git
   cd Thalex_SimpleQuouter
   ```

2. **Create and activate a virtual environment (recommended):**

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   pip install -e .  # Install the package in development mode
   ```

4. **Set up your API keys:**

   Update the `thalex_py/Thalex_modular/models/keys.py` file with your Thalex API credentials:

   ```python
   key_ids = {
       Network.TEST: "your_test_key_id",
       Network.PROD: "your_production_key_id"  # Only add when ready for production
   }

   private_keys = {
       Network.TEST: """-----BEGIN RSA PRIVATE KEY-----
   Your test private key here
   -----END RSA PRIVATE KEY-----""",
       Network.PROD: """-----BEGIN RSA PRIVATE KEY-----
   Your production private key here (when ready)
   -----END RSA PRIVATE KEY-----"""
   }
   ```

5. **Configure the trading parameters:**

   Review and update the configuration in `thalex_py/Thalex_modular/config/market_config.py` to match your trading strategy. See the [Configuration Guide](configuration.md) for details.

6. **Create log directories:**

   ```bash
   mkdir -p logs/orders logs/market logs/trades
   ```

## Testing Your Installation

Run a simple test to confirm everything is set up correctly:

```bash
python start_quoter.py --test
```

This should connect to the Thalex testnet and initialize the trading bot without placing any orders.

## Next Steps

- Review the [Quick Start Guide](quickstart.md) for basic operation
- Learn how to configure the bot in the [Configuration Guide](configuration.md)
- For advanced parameter tuning, see the [Tuning Guide](tuning.md) 