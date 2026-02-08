# Thalex Modular Trading Ecosystem - QWEN.md

## Project Overview

Thalex Modular is a high-performance, multi-exchange trading and alpha simulation framework designed for professional market making and algorithmic trading. The system supports live trading, shadow (dry-run) simulation, and historical backtesting across multiple venues including Thalex, Bybit, Binance, and Hyperliquid.

The project follows Clean Architecture principles with a modular design separating core business logic from external infrastructure. It features advanced analytics, real-time dashboard visualization, and sophisticated risk management systems.

### Key Features
- **Multi-Venue Execution**: Native support for Thalex, Bybit, Binance Futures, and Hyperliquid
- **High-Fidelity Alpha Simulation**: Ticker-by-ticker backtesting with realistic LOB (Limit Order Book) spread simulation and latency modeling
- **Advanced Alpha Analytics**: Edge tracking, adverse selection analysis, and correlation metrics
- **TimescaleDB Persistence**: Hypertables for high-performance storage of market data and trading signals
- **Visual Excellence Dashboard**: Modern SvelteKit dashboard served via FastAPI with real-time PnL and analytics
- **Modular Clean Architecture**: Decoupled domain logic from exchange-specific adapters

## Project Structure

```
Thalex_modular/
├── src/                        # Modular Core (Clean Architecture)
│   ├── adapters/               # External interfaces (Exchange Adapters, Storage)
│   ├── domain/                 # Core entities, signals, and matching logic
│   ├── use_cases/              # Orchestration (Quoting, SimulationEngine)
│   ├── api/                    # FastAPI Backend for SvelteKit Dashboard
│   └── main.py                 # Multi-Exchange Trading Bot Entry Point
├── thalex_py/                  # Thalex Core SDK
├── thalex_modular_dashboard/   # SvelteKit Frontend
├── scripts/                    # Utility scripts & API tests
├── GEMINI.md                   # Project Directives
├── TASKS.md                    # Roadmap & Progress
└── Architecture.md             # Detailed Technical Design
```

## Building and Running

### Prerequisites
- Python 3.10+
- TimescaleDB (PostgreSQL extension)
- Node.js (for SvelteKit dashboard)
- API Keys for target exchanges stored in `.env` file

### Installation
1. Clone and setup environment:
```bash
git clone <repository-url>
cd Thalex_modular
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. Configure environment variables:
```bash
cp .env.example .env
```
Initialize your API keys and DB credentials.

3. Database migration:
The system automatically initializes TimescaleDB hypertables on startup.

### Running the System
Always ensure your virtual environment is active:
```bash
source venv/bin/activate
```

#### Multi-Exchange Trading Bot
```bash
# Live Mode (Real orders)
python src/main.py --multi-venue --mode live

# Shadow Mode (Simulated fills - RECOMMENDED FOR TESTING)
python src/main.py --multi-venue --mode shadow
```

#### High-Fidelity Simulation / Backtesting
```bash
# Via API/Dashboard or CLI test scripts
python scripts/test_sim_endpoints.py

# Validate .env configuration
python scripts/validate_env.py
```

#### Start the Backend API
Required for the SvelteKit dashboard:
```bash
python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --reload
```

### Production Deployment (Screen)
For long-running sessions, use `screen` to keep the bot and API active:

#### Bot Session
```bash
screen -S thalex-bot
source venv/bin/activate
python src/main.py --multi-venue --mode live
```

#### API Session
```bash
screen -S thalex-api
source venv/bin/activate
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

## Key Components

### Core Architecture
- **QuotingService**: Central orchestrator that manages market data, strategy, and risk management
- **StateTracker**: Single source of truth for internal state, tracking orders and positions
- **MultiWindowRegimeAnalyzer**: Adjusts strategy parameters based on real-time market conditions
- **SimMatchEngine**: Pessimistic fill model for simulation with configurable latency
- **MultiExchangeStrategyManager**: Orchestrates connections and state across multiple venues

### Safety & Risk Management
- **LatencyMonitor**: Tracks RTT and enters safety mode if thresholds exceeded
- **CircuitBreaker**: Monitors PnL drawdowns and execution anomalies
- **BasicRiskManager**: Enforces position limits and order size constraints

### Data Infrastructure
- **TimescaleDB**: High-performance storage with hypertables for market data
- **TokenBucket**: Centralized rate limiting across all venues
- **OptionsVolatilityService**: Expected move and IV calculations

## Configuration

The system uses a `config.json` file for configuration with sections for:
- Primary instrument and venue settings
- Strategy parameters (gamma, volatility, position limits)
- Risk management parameters
- Signal configurations
- Data ingestion settings

## Development Conventions

- Follow Clean Architecture principles with clear separation of concerns
- Use environment variables for configuration (never commit .env files)
- Maintain self-documenting code with minimal inline comments
- Use standardized naming for environment variables across all components
- Implement proper error handling and logging
- Write unit and integration tests for all critical components

## Important Notes

- Trading derivatives involves substantial risk. This software is for educational and research purposes.
- Always test in shadow mode before deploying live strategies.
- Monitor account balances and margin requirements to avoid order rejections.
- Use screen sessions for production deployments to maintain persistent connections.