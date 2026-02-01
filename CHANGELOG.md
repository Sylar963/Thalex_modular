# Changelog

All notable changes to this project will be documented in this file.

## [Current State] - 2026-02-01

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

---

## [Previous Milestones]

### Infrastructure Evolution
- **Supabase to Convex Migration**: Successfully moved the global persistence layer to Convex for real-time dashboard state.
- **Clean Architecture Transition**: Restructured core trading logic into domain, infrastructure, and application layers.

### Trading Logic
- **Hedge Logic Removal**: Excised legacy hedging components after verifying that the core market-making engine handles portfolio risk via dynamic skews.
- **Regime Detection**: Implemented the first version of the Regime Detector using ATM Straddle Expected Move (EM) as a primary signal.
- **Avellaneda-Stoikov Integration**: Integrated regime signals into the core quoting engine to adjust `gamma` and `kappa` based on market volatility.
