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
- [x] Portfolio data persistence & API integration.
- [x] API Rate Limiting (Token Bucket + Cancel On Disconnect).
- [x] Technical Debt Refactoring (Standardized Env Vars + Shared Utils).
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
  - `DATABASE_HOST` (e.g., localhost)
  - `DATABASE_PORT` (e.g., 5432)
  - `DATABASE_USER` (e.g., postgres)
  - `DATABASE_PASSWORD` (e.g., password)
  - `DATABASE_NAME` (default: `thalex_trading`)
- **Core Tables (Hypertables)**:
  - `market_tickers`: `time`, `symbol`, `bid`, `ask`, `last`, `volume`
  - `market_trades`: `time`, `symbol`, `price`, `size`, `side`, `trade_id`

### Data Architecture (Real-time & Dashboard)
The system leverages TimescaleDB as a "Single Source of Truth":
- **Access**: Python `asyncpg` adapter (backend) or FastAPI endpoints (dashboard).
- **Portfolio Snapshots**: Current positions are snapshotted to `portfolio_positions` table on every ticker update to ensure low-latency API serving.

## V. Critical Technical Knowledge & Lessons Learned

### Curated Bugs & Architectural Pitfalls
- **Import Shadowing & NameErrors**: Never assume a service or entity is available in a modular file without explicit import.
  - *Case Study (Bybit)*: Missing `InstrumentService` import led to `NameError` during live trading startup.
  - *Case Study (Thalex)*: Truncated import blocks during refactoring caused immediate `SyntaxError`.
- **Environment Variable Divergence**: Standardize env var names system-wide early.
  - *Fix*: Migrated all components (`src/api`, `data_ingestion`, `main.py`) to `DATABASE_HOST`, `DATABASE_PORT` etc., eliminating connection failures in dashboard and ingestion pipelines.
- **Thalex Tick Size Alignment**: Thalex is extremely strict about price alignment with `tick_size`.
  - *Pitfall*: `ThalexAdapter` method name mismatch (`_fetch_instrument_details` vs expected `fetch_instrument_info`) caused the bot to use a default `0.5` tick size when Thalex required `1.0`. ALWAYS ensure adapter method names match the interface expected by `StrategyManager`.
- **Margin Requirements vs Order Size**: Small account balances can lead to immediate order rejections with code `24` (insufficient margin) if `order_size` is too high.
  - *Tip*: BTC-PERPETUAL on Thalex can require high margin (~$124 for 0.01 BTC). If orders aren't placing, check for code `24` in logs and reduce `order_size` (e.g., to `0.001`).
- **Token Bucket algorithm**: Centralized in `base_adapter.py` to ensure consistent rate-limiting across all venues without repeating logic.
- **Port Consistency**: Standardized on port `5432` for TimescaleDB across all services, removing the legacy `5433` fallback which caused confusion between local and containerized setups.
- **FastAPI Dependency Injection**: Always call `load_dotenv()` in `main.py` before any repository initializations to ensure DB parameters are hydrated.
- **Screen Session Strategy**: Use `screen -S <name>` for long-running bot instances to ensure persistence through SSH disconnects.
- **Process Isolation (API vs Bot)**: The API server (`src/api/main.py`) and the Trading Bot (`src/main.py`) are separate processes. Shared services like `MarketFeed` must be initialized in *both* if they need to operate in both contexts (or preferably in the API for data serving).
    - *Pitfall*: Running `src/main.py` when intending to start the API caused a confusion in service wiring.
- **Service Injection**: `SimStateManager` needs access to the *active* data stream. Wiring it into `market_feed.py` is useless if the API uses `data_ingestor.py`. Check `dependencies.py` to confirm which service is actually being instantiated.

---
*This document is a living record of project directives and should be updated as the roadmap evolves.*

DONT COMMENT WHILE WRITING CODE*
ALWAYS USE VENV FIRST TO RUN COMMANDS. AS A POLICY RULE
ALWAYS RUN COMMANDS IN VENV, AND REVIEW IMPORTS ARE CORRECT BEFORE RUNNIGN TEST SCRIPTS