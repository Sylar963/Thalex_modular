# SuitMM: Visual Market Making Analysis Suite

SuitMM is a comprehensive tool for analyzing cryptocurrency markets and evaluating the performance of trading bots. It provides interactive HTML reports with rich visualizations to help market makers optimize their strategies.

## Features

### 1. Market Feasibility Analysis (`--mode analyze`)
- **Profitability Heatmap**: Visualizes estimated profit per round trip across different spread and order size combinations.
- **Volatility Cone**: Analyzing historical volatility over various time windows (1d, 7d, 14d, 30d).
- **Market Depth Profile**: Cumulative orderbook visualization to assess liquidity.
- **Parameter Recommendations**: Suggests optimal spread, order size, and position limits based on market conditions.

### 2. Bot Performance Analysis (`--mode performance`)
- **Realized PnL Curve**: Tracks Mark-to-Market profitability over time from actual trade fills.
- **Trade Execution Map**: Overlays Buy/Sell executions on the price chart to visualize entry/exit timing.
- **Indicator Correlation**: Aligns market indicators (Liquidity Score, Trend Strength, Toxicity) with price action.
- **Execution Quality (Markout)**: Analyzes price movement 1m, 5m, and 60m after each trade to measure adverse selection.

## Installation

Ensure you have the required Python packages installed:

```bash
pip install pandas plotly aiohttp asyncpg
```

## Usage

Run the tool from the project root directory using the module flag `-m`.

### Analyze Market Feasibility

To analyze a coin's potential for market making:

```bash
# Basic usage
python -m scripts.suitmm.run HYPEUSDT

# Compare multiple coins
python -m scripts.suitmm.run HYPEUSDT SUIUSDT BTCUSDT

# Override minimum order size (useful for small cap coins)
python -m scripts.suitmm.run HYPEUSDT --size 0.1

# Specify custom capital (default: 233 USDT)
python -m scripts.suitmm.run HYPEUSDT --capital 1000
```

**Output:** Generates `analysis_SYMBOL.html` files.

### Analyze Bot Performance

To analyze the historical performance of your running bot:

```bash
python -m scripts.suitmm.run HYPEUSDT --mode performance
```

**Output:** Generates `performance_SYMBOL.html` files.

## Configuration

The tool connects to a local TimescaleDB instance to fetch historical data and trade fills.
- **Default DB URL**: `postgresql://postgres:password@localhost:5432/thalex_trading`
- **Fallback URL**: Port `5433` if 5432 fails.

To change database settings, modify `scripts/suitmm/data_fetcher.py`.

## File Structure

- `run.py`: CLI entry point and orchestration.
- `data_fetcher.py`: Handles API requests (Bybit) and Database queries (TimescaleDB).
- `analyzer.py`: Core logic for market analysis and parameter recommendation.
- `performance_analyzer.py`: Logic for PnL calculation and trade statistics.
- `visualizer.py`: Plotly chart generation.
- `report_generator.py`: HTML report compilation.
