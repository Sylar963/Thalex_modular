# Changelog

All notable changes to this project will be documented in this file.

## [Current State] - 2026-02-01

### Added
- **1-Minute Data Scaling**: Updated `historical_options_loader.py` to support high-resolution (1m) data fetching from Thalex.
- **Active Instrument Discovery**: The data loader now fetches active instruments from `public/instruments` to ensure valid ticker selection instead of guessing expiration dates.
- **FastAPI Backend**: Initialized a Clean Architecture API in `src/api`.
    - `MetricsRepository`: Handles TimescaleDB access on port 5433.
    - `endpoints`: Provides `/api/v1/market/metrics` for simulation analysis.
- **Project Memory (GEMINI.md)**: Updated core directives with the new 1m resolution strategy and persistence layer rules.

### Changed
- **Database Port**: Migrated TimescaleDB host mapping to port `5433` to resolve local conflicts.
- **Data Ingestion Filter**: Fixed loader to use `underlying="BTCUSD"` (conformed to real Thalex API responses).

### Fixed
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
