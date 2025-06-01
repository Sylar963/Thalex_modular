# Thalex SimpleQuoter

A high-performance algorithmic trading system for the [Thalex](https://www.thalex.com) derivatives exchange, implementing the Avellaneda-Stoikov optimal market making model.

## What is This?

The Thalex SimpleQuoter is an automated trading system (a "bot") that places buy and sell orders on the Thalex cryptocurrency derivatives exchange. It uses advanced mathematical models to determine optimal pricing and manage risk.

**For Non-Programmers**: This is specialized software that helps traders automatically provide liquidity (place buy and sell orders) on the Thalex exchange. It handles the technical aspects of trading so you can focus on strategy and risk parameters.

## Installation Guide

### Prerequisites

- Linux operating system (Ubuntu recommended)
- Python 3.8 or higher
- Thalex exchange account with API keys

### Simple Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/Thalex_SimpleQuouter.git
   cd Thalex_SimpleQuouter
   ```

2. **Create and set up environment file**:
   Create a file named `.env` in the main directory with your Thalex API credentials:
   ```
   THALEX_TEST_API_KEY_ID="your_key_id"
   THALEX_TEST_PRIVATE_KEY="your_private_key"
   ```
   For production use, use `THALEX_PROD_API_KEY_ID` and `THALEX_PROD_PRIVATE_KEY`.

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the quoter**:
   ```bash
   python start_quoter.py
   ```

### Configuration

Configuration parameters are stored in `thalex_py/Thalex_modular/config/market_config.py`. Key settings include:

- Risk limits (position size, stop-loss)
- Trading parameters (spread, order size)
- Market configuration (instrument, network)

For non-programmers: You may want to ask a developer to help set up the initial configuration.

## Usage

### Starting the System

```bash
python start_quoter.py
```

Add optional parameters to customize behavior:
```bash
python start_quoter.py --levels 3 --gamma 0.1
```

### Command Line Options

- `--test`: Run on test network instead of production
- `--gamma`: Set risk aversion parameter
- `--levels`: Number of price levels to quote
- `--spacing`: Grid spacing in ticks
- `--vol-threshold`: Volume candle threshold

### Monitoring

The system creates detailed logs in the `logs/` directory. You can monitor these logs to track performance and system status.

## Technical Documentation

The SimpleQuoter implements the Avellaneda-Stoikov market making model with several extensions:

- Dynamic volatility calculation
- Position-based quote skewing
- Advanced risk management
- High-frequency optimizations

For developers: The system uses asynchronous programming (asyncio) for high performance and includes sophisticated mathematical models for optimal quote placement.

## Disclaimer

This software is provided as-is with no warranty. Trading cryptocurrency derivatives involves significant risk of loss. This software is not financial advice.

## Support

For questions or issues:
- Open an issue on GitHub
- Contact: support@example.com (replace with actual support email)

## License

MIT License - See LICENSE file for details.
