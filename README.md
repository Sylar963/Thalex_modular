# Thalex Modular: Multi-Exchange Trading & Alpha Simulation Framework

A high-performance **Multi-Exchange Trading Bot** and **High-Fidelity Alpha Simulation Framework**. This project has evolved into a robust ecosystem for executing HFT strategies across multiple venues (Thalex, Bybit, Binance, Hyperliquid) and simulating trading alpha with **ticker-level precision** and persistent analysis using **TimescaleDB**.

![Thalex](https://thalex.com/images/thalex-logo-white.svg)

## 🚀 Key Features & Updates

- **Multi-Venue Execution**: Native support for **Thalex**, **Bybit**, **Binance Futures**, and **Hyperliquid**.
- **High-Fidelity Alpha Simulation**: 
  - Ticker-by-ticker backtesting with realistic **LOB (Limit Order Book)** spread simulation.
  - **Latency Modeling**: Simulated time-to-market and order activation delays.
- **Advanced Alpha Analytics**: 
  - **Edge Tracking**: Measuring profit vs. mid-price at the moment of fill.
  - **Adverse Selection (MtM Decay)**: Analyzing mark-to-market decay at 5s, 30s, and 60s post-fill.
- **TimescaleDB Persistence**: Hypertables for high-performance storage of market data, trading signals (VAMP, Open Range), and portfolio snapshots.
- **Visual Excellence Dashboard**: A modern **SvelteKit Dashboard** (served via FastAPI) providing real-time PnL, signal overlays, and simulation analytics.
- **Modular Clean Architecture**: Decoupled domain logic from exchange-specific adapters for seamless venue integration.

---

## 📂 Project Structure

```text
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

---

## 🛠️ Getting Started

### Prerequisites
- **Python 3.10+**
- **TimescaleDB** (PostgreSQL extension)
- **Node.js** (for SvelteKit dashboard)
- **API Keys**: Stored in a secure `.env` file

### Installation

1.  **Clone & Setup Environment**:
    ```bash
    git clone <repository-url>
    cd Thalex_modular
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

2.  **Configure Environment Variables**:
    ```bash
    cp .env.example .env
    ```
    Initialize your API keys (Bybit, Binance, Hyperliquid, Thalex) and DB credentials.

3.  **Database Migration**:
    The system automatically initializes TimescaleDB hypertables on startup.

---

## 🏃 Running the System

Always ensure your virtual environment is active before running commands:
```bash
source venv/bin/activate
```

### 1. Multi-Exchange Trading Bot
To run live or shadow quoting on multiple venues concurrently:

```bash
# Live Mode (Real orders)
python src/main.py --multi-venue --mode live

# Shadow Mode (Simulated fills - RECOMMENDED FOR TESTING)
python src/main.py --multi-venue --mode shadow
```

### 2. High-Fidelity Simulation / Backtesting
Simulate alpha metrics on historical Bybit/Thalex data:

```bash
# Handled via API/Dashboard or CLI test scripts
python scripts/test_sim_endpoints.py

# Validate .env configuration
python scripts/validate_env.py
```

### 3. Start the Backend API
Required for the SvelteKit dashboard:

```bash
python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --reload
```

---

## � Production Deployment (Screen)

For long-running sessions, use `screen` to keep the bot and API active:

### 1. Bot Session
```bash
screen -S thalex-bot
source venv/bin/activate
python src/main.py --multi-venue --mode live
```

### 2. API Session
```bash
screen -S thalex-api
source venv/bin/activate
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

### 3. Screen Management
| Action | Command |
|--------|---------|
| **Detach** from session | `Ctrl + A`, then `D` |
| **List** active sessions | `screen -ls` |
| **Reattach** to session | `screen -r <session-name>` |
| **Kill** a session | `screen -X -S <session-name> quit` |

---

## �📊 Analytics & Alpha Metrics

The framework introduces a dedicated `StatsEngine` to quantify trading edge:

| Metric | Description |
|--------|-------------|
| **Edge** | $\text{Fill Price} - \text{Mid Price at Fill}$ (for Maker) |
| **Adverse Selection** | Change in mid-price $N$ seconds after fill |
| **VAMP Alignment** | Correlation between VAMP signals and positive trade outcomes |

---

## ⚠️ Risk Warning

**Trading derivatives involves substantial risk.** This software is for educational and research purposes. The "Alpha Simulation Framework" allows for risk-free strategy testing, but live trading should always be approached with caution and proper risk management.

## 📄 License

MIT License.