# Thalex Modular Trading Ecosystem Architecture

## Overview
The Thalex Modular system is a high-performance, asynchronous trading framework built for professional market making on the Thalex exchange. It follows **Clean Architecture** principles to separate core business logic (optimal quoting, risk management) from external infrastructure (exchange adapters, database storage).

The system supports three major operational modes:
- **Live**: REAL trading on Thalex (Testnet/Mainnet).
- **Shadow (Dry-Run)**: Real-time market data with simulated execution via an internal matching engine.
- **Backtest**: Historical simulation using high-resolution (1m) data from TimescaleDB.

---

## Core Philosophy & Design Patterns
- **Clean Architecture**: Separation into Entities, Use Cases, Adapters, and Services.
- **Reactive State Management**: Event-driven tracking of orders and positions to minimize latency and synchronization gaps.
- **Simulation First**: Production-ready matching engine integrated directly into the core service logic for high-fidelity "shadow" trading.
- **Observability**: REST and SSE (Server-Sent Events) API for real-time dashboard monitoring.

---

## Project Structure (The modular way)

```text
Thalex_modular/
├── src/
│   ├── main.py                  # CLI Entry point (Live/Shadow modes)
│   ├── api/                     # FastAPI Layer
│   │   ├── endpoints/           # REST & SSE routes (Market, Simulation, Portfolio)
│   │   └── repositories/        # Backend-frontend data adapters
│   ├── use_cases/               # Core Business Logic (Orchestrators)
│   │   ├── quoting_service.py   # Main Market-Maker Orchestrator
│   │   ├── simulation_engine.py # Backtest runner
│   │   └── sim_state_manager.py # Shared state for real-time simulation
│   ├── domain/                  # Domain Models & Interfaces
│   │   ├── entities/            # Pydantic/Dataclasses (Order, Position, Ticker)
│   │   ├── interfaces.py        # Abstract contracts
│   │   ├── tracking/            # State & Position tracking (StateTracker)
│   │   ├── market/              # Logic-heavy market analysis (RegimeAnalyzer)
│   │   ├── strategies/          # Trading algorithms (Avellaneda-Stoikov)
│   │   └── sim_match_engine.py  # Internal pessimistic match engine
│   ├── adapters/                # Infrastructure (External connections)
│   │   ├── exchanges/           # Thalex API adapters
│   │   └── storage/             # TimescaleDB / high-performance persistence
│   └── services/                # Specialized domain-specific helpers
│       └── options_volatility_service.py # Expected move & IV calculator
├── tests/                       # Unit and integration test suite
├── Architecture.md              # This document
└── GEMINI.md                    # Project Management Directives
```

---

## Technical Components

### 1. **QuotingService** (`quoting_service.py`)
The central orchestrator that wires together market data, strategy, and risk management.
- **Mode Aware**: Seamlessly switches between `Live` and `Dry-Run` (Shadow) modes.
- **Concurrency**: Uses `asyncio.Lock` for atomic order reconciliation on fast price updates.
- **Toxic Flow Defense**: Automatically pulls quotes when extreme one-sided volume is detected.
- **Fast Reconcile**: Triggered by both price updates (tickers) and aggressive flow (trades).

### 2. **StateTracker** (`state_tracker.py`)
Provides a "Single Source of Truth" for the bot's internal state.
- **Order Lifecycle**: Tracks orders through `PENDING` -> `CONFIRMED` -> `FILLED/CANCELLED`.
- **Sequence Monitoring**: Detects gaps in exchange message sequences to trigger emergency re-syncs.
- **Reactive Fills**: Emits events the moment a fill is confirmed via the private trade stream.
- **LRU Cache**: Manages historical order memory while keeping the hot path optimized.

### 3. **MultiWindowRegimeAnalyzer** (`regime_analyzer.py`)
Adjusts strategy parameters based on real-time market conditions.
- **Triple Window RV**: Calculates Volatility across Fast (20), Mid (100), and Slow (500) intervals.
- **Options Convergence**: Integrates data from the `OptionsVolatilityService` to detect if realized vol is over/under-priced vs market expectations.
- **Dynamic Risk**: Informs the `AvellanedaStoikovStrategy` to adjust $\gamma$ (risk aversion) during trending or volatile regimes.

### 4. **Simulation Architecture**
- **SimMatchEngine**: Implements a pessimistic fill model. Orders are only filled if the public trade stream crosses the limit price + simulated slippage.
- **Latency Modeling**: Configurable "time-to-market" delay to simulate realistic network conditions.
- **Partial Fills**: Highly accurate tracking of simulated fill sizes.
- **SSE Streams**: Live equity curves and fills are broadcasted via SSE to the dashboard.

### 5. **Data Infrastructure**
- **ThalexAdapter**: High-performance JSON-RPC client with Token Bucket rate limiting (targeting 90% of exchange limits).
- **TimescaleDB**: High-resolution persistence. Uses hypertables for efficient storage of 1-minute OHLCV data and tick-by-tick trade history.

---

## Data Flows

### **Standard Tick Processing**:
`WebSocket` -> `Adapter` -> `StateTracker` (Latency check) -> `RegimeAnalyzer` -> `QuotingService` -> `Strategy` -> `RiskCheck` -> `OrderReconciler` -> `Adapter/SimMatchEngine`.

### **Reactive Trace**:
`Private Fill Msg` -> `StateTracker` -> `QuotingService.on_fill_event()` -> `Immediate Alpha Calculation`.

---

## Operational Modes

| Mode | Execution | Market Data | Data Persistence |
|---|---|---|---|
| **Live** | Thalex Exchange | Real-time WS | TimescaleDB |
| **Shadow** | SimMatchEngine | Real-time WS | TimescaleDB |
| **Backtest** | SimMatchEngine | Historical (DB) | Memory/CSV |

---

> [!IMPORTANT]
> **No-Comments Policy**: In accordance with project rules, code should be self-documenting. Documentation of implementation details should reside here in `Architecture.md` or in `GEMINI.md`.

> [!TIP]
> **Performance Tip**: When scaling the bot, prioritize optimizing the `StateTracker` and `RegimeAnalyzer` hot paths as they are touched on every ticker update.