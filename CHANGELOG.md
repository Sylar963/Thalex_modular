# Changelog

All notable changes to this project will be documented in this file.

## [Current State] - 2026-02-08

### Safety & Reliability
- **Safety Plugin Architecture**: Implemented `SafetyComponent` interface with concrete plugins:
    - `LatencyMonitor`: Monitors WebSocket heartbeat and execution latency, triggering safety pauses.
    - `CircuitBreaker`: Automatically halts trading if loss thresholds or abnormal execution patterns are detected.
- **Robust Order Reconciliation**:
    - Enhanced `MultiExchangeStrategyManager` to fetch and cancel stale orders from all enabled venues upon bot restart.
    - Implemented missing `get_open_orders` methods across `BybitAdapter`, `BinanceAdapter`, and `HyperliquidAdapter`.

### Standardization & Infrastructure
- **Standard Python Packaging**: Migrated the project to use `setup.py` and `pip install -e .`, enabling clean imports across the codebase.
- **Import Cleanup**: Eliminated ad-hoc `sys.path` modifications in all scripts.
- **ORB Session Alignment**: Standardized Open Range calculations to align with 8:00 PM PST session starts across backend and frontend.

### Diagnostics & Benchmarking
- **Order Latency Suite**: Developed a dedicated benchmark tool to measure round-trip latency for order placement and cancellation on Thalex and Bybit.
- **Chart Protection**: Implemented a "band collar" for Open Range indicators to ignore illiquidity-driven price spikes on specific exchanges.

## [2026-02-06] - Previous State
### Features
- **Live Simulation Framework (Forward Testing)**:
    - Implemented `LiveSimulationRunner` within `SimStateManager` to conduct "paper trading" simulations using live market data.
    - Integrated `TimescaleHistoryProvider` as a fallback for historical data but prioritized live ticker feeds for forward testing.
    - Added `bot_executions` persistence with `exchange='sim_shadow'` to strictly separate mock fills from real trading activity.
    - Exposed API endpoints (`/api/v1/simulation/live/start`, `/status`, `/stop`) to control the simulation lifecycle remotely.

- **Decoupled Data Ingestion**: Separated the "Charting/Market Data" feed from the "Trading Engine" (Quoting).
    - Introduced `DataIngestionService` (`src/services/data_ingestor.py`) to handle public data streams independently of the bot's trading status.
    - Added `data_ingestion` block to `config.json` for managing a "Watchlist" of symbols to ingest (e.g., `ETHUSDT` for charting only).
    - Updated `dependencies.py` to initialize `DataIngestionService` instead of the legacy `MarketFeedService`.
    - ** Benefit**: Users can now chart any symbol without the bot having to trade it, and the system is more modular.

### Technical Debt Remediation
- **Security**: Removed hardcoded database credentials and blocking `psycopg2` driver from `historical_options_loader.py`. It now uses `TimescaleDBAdapter` (asyncpg) and environment variables.
- **Performance**: Added `save_tickers_bulk` and `save_options_metrics_bulk` to `TimescaleDBAdapter` for efficient batch ingestion.
- **Configuration**: Fixed incorrect Bybit symbol mapping (`HYPEUSDT` -> `BTCUSDT`) in `config.json`.
- **Reliability**: Extended `MarketFeedService` warmup period to 48 hours to ensure `OpenRangeSignalEngine` has sufficient data even with weekend gaps.

### Refactored (Technical Debt)
- **Centralized Rate Limiting**: Moved `TokenBucket` to `base_adapter.py` to provide a shared utility for all exchange adapters, reducing code duplication.
- **Environment Variable Standardization**: Standardized database connection parameters across the entire system (`DATABASE_HOST`, `DATABASE_PORT`, `DATABASE_USER`, `DATABASE_NAME`, `DATABASE_PASSWORD`).
- **Adapter Cleanup**: Fixed redundant task and variable initializations in `BybitAdapter`, `BinanceAdapter`, and `HyperliquidAdapter`.
- **Logic Consolidation**: Removed redundant `RegimeDetector` class in favor of the more robust `MultiWindowRegimeAnalyzer`.
- **Robustness**: Fixed `NameError` in Bybit adapter and restored missing imports/logger in Thalex adapter.

## [Previous State] - 2026-02-04

### Added
- **High-Fidelity Alpha Simulation**: 
    - `LOBMatchEngine`: Implemented ticker-level precision matching with simulated orderbook spreads and latency.
    - `StatsEngine`: Introduced Alpha Analytics tracking **Edge** (fill vs mid) and **Adverse Selection** (MtM decay at 5s/30s/60s).
    - `BybitHistoryAdapter`: Modular history provider that fetches 1m klines/trades from Bybit and persists them to TimescaleDB for reusable backtesting.
    - `SimulationEngine`: Unified orchestration logic for high-fidelity backtests.
- **Multi-Exchange Scaling**:
    - **Bybit Integration**: Full historical data ingestion and simulation support for Bybit.
    - **API Venue Support**: Updated `/simulation/start` to support exchange-specific backtesting via the `venue` parameter.
- **Developer Tools**:
    - `test_sim_endpoints.py`: Utility script for CLI verification of simulation API endpoints.
    - Enhanced `FRONTEND_HANDOFF.md` with specific simulation dashboard requirements.

### Enhanced
- **Documentation Overhaul**: Rewrote `README.md` to reflect the transition from a single-exchange quoter to a multi-venue HFT and alpha analysis framework.
- **Repository Layer**: Refactored `SimulationRepository` to inject modular history providers and the new simulation engine.
- **TimescaleDB Integration**: Improved ticker saving performance and added support for larger history loops in Bybit ingestion.

## [Previous State] - 2026-02-01

### Added
- **1-Minute Data Scaling**: Updated `historical_options_loader.py` to support high-resolution (1m) data fetching from Thalex.
- **Active Instrument Discovery**: The data loader now fetches active instruments from `public/instruments` to ensure valid ticker selection instead of guessing expiration dates.
- **FastAPI Backend**: Expanded Clean Architecture API in `src/api`.
    - `MetricsRepository`: Handles TimescaleDB access on port 5433.
    - `endpoints`: Provided `/api/v1/market/metrics`, `/api/v1/portfolio`, `/api/v1/simulation`, and `/api/v1/config`.
- **Project Memory (GEMINI.md)**: Updated core directives with the new 1m resolution strategy and persistence layer rules.

### Fixed
- **API Reachability**: Resolved relative import issues in `dependencies.py` and `main.py` causing `ModuleNotFoundError`.
- **Environment Configuration**: Fixed `.env` loading priority in `main.py` to ensure database credentials are available to all adapters before initialization.
- **Empty Backfills**: Resolved issue where historical data loader returned zero records due to incorrect instrument name generation and missing `BTCUSD` underlying mappings.

### Enhanced
- **Hybrid Chart History**: Implemented automatic fallback to `market_tickers` when `market_trades` are empty, ensuring charts always display data even in low-liquidity conditions.
- **Execution Persistence**: Created `bot_executions` table and API endpoint (`/portfolio/executions`) for dedicated tracking and plotting of bot fills.
- **Position Reliability**: Added robust persistence callback in `QuotingService` to ensure live positions are always saved to `portfolio_positions`, fixing empty dashboard widgets.
- **Database Schema**: Added `exchange` column migration to core tables to support multi-exchange architecture in the future.

---

## [Previous Milestones]

### Infrastructure Evolution
- **Supabase to Convex Migration**: Successfully moved the global persistence layer to Convex for real-time dashboard state.
- **Clean Architecture Transition**: Restructured core trading logic into domain, infrastructure, and application layers.

### Trading Logic
- **Hedge Logic Removal**: Excised legacy hedging components after verifying that the core market-making engine handles portfolio risk via dynamic skews.
- **Regime Detection**: Implemented the first version of the Regime Detector using ATM Straddle Expected Move (EM) as a primary signal.
- **Avellaneda-Stoikov Integration**: Integrated regime signals into the core quoting engine to adjust `gamma` and `kappa` based on market volatility.
