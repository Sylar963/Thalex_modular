# Thalex Modular: PNL Simulation Framework & Trading Bot

A high-performance **PNL Simulation Framework** and **Trading Bot** for the [Thalex](https://www.thalex.com) derivatives exchange. This project has evolved from a simple quoter into a robust system for simulating trading strategies with **1-minute resolution** precision and persistent data analysis using **TimescaleDB**.

![Thalex](https://thalex.com/images/thalex-logo-white.svg)

## 🚀 Key Features & Updates

- **High-Resolution Simulation**: Simulates PnL and strategy performance using **1-minute resolution data**.
- **Historical Backfills**: Capable of fetching and processing **30-day historical data chunks** for extensive strategy verification.
- **TimescaleDB Persistence**: Native integration with **TimescaleDB** for high-performance storage of market data (ticks, trades) and portfolio snapshots.
- **Visual Analytics**: Includes a modern **SvelteKit Dashboard** (served via FastAPI) for visualizing trading performance and market dynamics.
- **Modular Architecture**: Built with **Clean Architecture** principles (adapters, domain, use cases) for testability and scalability.
- **Latency Optimized**: Retains the legacy low-latency core for live trading while adding robust simulation capabilities.

---

## 📂 Project Structure

```text
Thalex_modular/
├── src/                        # Modular Core (Clean Architecture)
│   ├── adapters/               # External interfaces (Thalex API, TimescaleDB)
│   ├── domain/                 # Core logic (Strategies, Signals, Risk)
│   ├── use_cases/              # Orchestration (Quoting, Simulation)
│   ├── api/                    # FastAPI Backend for Dashboard
│   └── main.py                 # Trading Bot Entry Point
├── thalex_py/                  # Legacy Core & SDK
├── GEMINI.md                   # Project Management & Directives
├── TASKS.md                    # Roadmap & Todo
└── Architecture.md             # Technical Documentation
```

---

## 🛠️ Getting Started

### Prerequisites
- **Python 3.10+**
- **TimescaleDB** (PostgreSQL extension)
- **Thalex API Keys** (Testnet recommended for development)

### Installation

1.  **Clone & Setup Environment**:
    ```bash
    git clone <repository-url>
    cd Thalex_modular
    python -m venv venv
    source venv/bin/activate  # Linux/macOS
    pip install -r requirements.txt
    ```

2.  **Configure Environment Variables**:
    ```bash
    cp .example.env .env
    ```
    Edit `.env` with your API keys and database credentials.

3.  **Start Infrastructure**:
    Use Docker Compose to start TimescaleDB:
    ```bash
    docker-compose up -d
    ```

---

## 🏃 Running the System

 The system consists of two main components: the **Trading/Simulation Engine** and the **Dashboard API**.

### 1. Start the Simulation Engine / Trading Bot
This runs the core logic, fetches market data, executes the strategy (or simulation), and persists data.

```bash
# Run the modular core
python src/main.py
```

### 2. Start the Reporting API
The FastAPI backend serves data to the frontend dashboard.

```bash
# Serve API on localhost:8000
python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --reload
```

### 3. Production Deployment (Screen)
For long-running sessions, use `screen`:

```bash
# Bot Session
screen -S thalex-bot
source venv/bin/activate
python src/main.py

# API Session
screen -S thalex-api
source venv/bin/activate
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

---

## 🏗️ Architecture & Data Flow

### Simulation to Dashboard Pipeline
1.  **Ingestion**: `ThalexAdapter` fetches real-time or historical data.
2.  **Processing**: `VolumeCandleSignalEngine` and `AvellanedaStoikovStrategy` calculate metrics (VAMP, spreads).
3.  **Persistence**: `TimescaleDBAdapter` stores 1m candles, trades, and portfolio snapshots into **Hypertables**.
4.  **Serving**: `src/api` queries TimescaleDB to serve aggregated JSON data to the Frontend.

---

## ⚠️ Risk Warning

**Trading derivatives involves substantial risk.** This software is for educational and research purposes. The "Simulation Framework" allows for risk-free strategy testing, but live trading should always be approached with caution.

## 📄 License

MIT License.