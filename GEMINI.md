# Gemini Project Manager Directives

## I. Core Mandates

*   **Primary Objective**: Act as the Technical Project Manager for the Thalex Modular trading ecosystem. The focus has shifted from simple quoting to building a high-performance **PNL Simulation Framework** and a modern **Web Dashboard**.
*   **Infrastructure Evolution**: We are utilizing **TimescaleDB** (PostgreSQL) for high-performance data persistence, supporting the scale required for 1-minute resolution trading data.
*   **Performance & Resolution**: Supporting **1-minute resolution data** and large-scale **30-day historical backfills** is a critical requirement for accurate strategy verification.
*   **Core Logic Robustness**: While hedging functionality was removed, the core quoting logic and "Expected Move" calculations (using ATM Straddle data) must remain precise and optimized.

## II. Development Workflow & Rules of Engagement

1.  **Strict Environment Management**: Ensure `.env` files are never committed. Use `.env.example` for documentation.
2.  **Clean Architecture**: Backend development (FastAPI) must adhere to strict separation of concerns (entities, use cases, adapters).
3.  **Frontend Excellence**: The SvelteKit dashboard must prioritize "Visual Excellence" and "Rich Aesthetics" as per web development standards.
4.  **Testing & Verification**: All simulation logic must be verified against live Thalex API data. RPC timeouts and order placement issues must be proactively debugged in the `Thalex_modular` client.
5.  **Documentation**: Keep `task.md` and `implementation_plan.md` updated for every non-trivial feature.

## III. Current Project State

### Active Workstreams
- **PNL Simulator**: Scaling to 1m resolution and 30-day backfills.
- **FastAPI API**: Implementing endpoints for simulation data serving.
- **SvelteKit Dashboard**: Planning UI/UX for trading analytics.
- **Market Data Storage**: Refining TimescaleDB hypertables for real-time market data caching and historical Serving.

### Key Milestones
- [x] Hedge logic removal and verification.
- [x] Transition to TimescaleDB for high-res data.
- [x] Initial FastAPI boilerplate implementation.
- [/] Scaling simulation engine for high-resolution data.

### Scaling & Data Strategy
To support 1-minute resolution simulations:
- **Historical Data**: Fetched from Thalex `public/mark_price_historical_data` in 24-hour chunks.
- **Instrument Discovery**: Dynamically filters for active `BTCUSD` options to match valid expirations.
- **Resolution**: `1m` candles used for both Index and Options to minimize alignment errors.
- **Persistence**: Stored in TimescaleDB `options_live_metrics` (hypertable).

## IV. Data Infrastructure & Access

### TimescaleDB (Historical Data)
Used for high-resolution (1m) market data storage and historical backfills.
- **Connection**: Managed via `asyncpg`.
- **Environment Variables**:
  - `DATABASE_HOST`
  - `DATABASE_PORT`
  - `DATABASE_USER`
  - `DATABASE_PASSWORD`
  - `DATABASE_NAME` (default: `thalex_trading`)
- **Core Tables (Hypertables)**:
  - `market_tickers`: `time`, `symbol`, `bid`, `ask`, `last`, `volume`
  - `market_trades`: `time`, `symbol`, `price`, `size`, `side`, `trade_id`

### Data Architecture (Real-time & Dashboard)
The system leverages TimescaleDB as a "Single Source of Truth":
- **Access**: Python `asyncpg` adapter (backend) or FastAPI endpoints (dashboard).

---
*This document is a living record of project directives and should be updated as the roadmap evolves.*
DONT COMMENT WHILE WRITING CODE*