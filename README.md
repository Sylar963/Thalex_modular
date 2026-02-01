# Thalex SimpleQuoter: Modular Avellaneda Market Maker

A high-performance, production-ready cryptocurrency market making bot implementing the Avellaneda-Stoikov strategy, now featuring a modular "Clean Architecture" design and optimized for the [Thalex](https://www.thalex.com) derivatives exchange.

![Thalex](https://thalex.com/images/thalex-logo-white.svg)

## 🚀 Recent Updates & Current State

The project has transitioned to a modular **Ports and Adapters (Clean Architecture)** structure while maintaining the robust performance of the legacy system.

- **Modular Architecture**: New Core logic in `src/` for better testability and scalability.
- **Hedging Removed**: Hedging functionality has been completely removed to prioritize core quoting efficiency and reduce complexity.
- **Database Integration**: Added native support for **TimescaleDB** for high-performance trade and market data persistence.
- **Enhanced Signal Engine**: Refined Volume Adjusted Market Pressure (VAMP) calculations.

---

## 📂 Project Structure

```text
Thalex_modular/
├── src/                        # NEW: Modular Architecture
│   ├── adapters/               # External interfaces (Exchange, Storage)
│   ├── domain/                 # Core logic (Strategies, Signals, Risk)
│   ├── use_cases/              # Orchestration (Quoting Service)
│   └── main.py                 # Primary entry point
├── thalex_py/                  # Legacy Core & SDK
│   └── Thalex_modular/         # Legacy high-performance implementation
├── start_quoter.py             # Legacy entry point
├── TASKS.md                    # Development roadmap
├── GEMINI.md                   # Project management directives
└── Architecture.md             # Detailed technical documentation
```

---

## 🛠️ Getting Started

### Prerequisites
- Python 3.10+
- Thalex API Keys (Testnet recommended initially)
- (Optional) TimescaleDB/PostgreSQL instance

### Installation

1. **Clone & Setup Environment**:
   ```bash
   git clone <repository-url>
   cd Thalex_modular
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   pip install -r requirements.txt
   ```

2. **Configure Environment Variables**:
   ```bash
   cp .example.env .env
   ```
   Edit `.env` and fill in your Thalex API keys and database credentials.

3. **Start the Database (Required for API & Storage)**:
   The project uses TimescaleDB for data persistence. Start it using Docker Compose:
   ```bash
   docker-compose up -d
   ```
   *Note: This will start TimescaleDB on port 5433 (mapping 5432 internal).*

---

### Running the System

To have a fully functional frontend experience, you need both the Trading Bot and the Backend API running.

#### 1. Start the Trading Bot
The bot fetches market data, executes strategy logic, and persists data to the database.
```bash
# Recommended: Modular Architecture
python src/main.py
```

#### 2. Start the Backend API (For Frontend)
The API serves simulation and live data to the Svelte dashboard.
```bash
# The API runs on http://localhost:8000
python -m uvicorn src.api.main:app --reload
```

#### 3. Running in Background (Recommended)
Use `screen` to keep sessions active without an open terminal.

**Start Trading Bot:**
```bash
screen -S thalex-bot  # Create session
# Inside screen:
source venv/bin/activate
python src/main.py
# Press Ctrl+A, then D to detach
```

**Start Backend API:**
```bash
screen -S thalex-api  # Create session
# Inside screen:
source venv/bin/activate
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000
# Press Ctrl+A, then D to detach
```

**Manage Sessions**:
- List sessions: `screen -ls`
- Reattach to bot: `screen -r thalex-bot`
- Reattach to api: `screen -r thalex-api`

#### 4. Frontend Development
Ensure your `vite.config.ts` in the frontend project is routing `/api` traffic to `localhost:8000`.

---

---

## 🏗️ Architecture Deep Dive

### Modular System (`src/`)
Our new architecture separates business logic from infrastructure:

- **Domain Layer**: Contains the "brain" of the bot.
  - `AvellanedaStoikovStrategy`: The core mathematical pricing model.
  - `VolumeCandleSignalEngine`: Generates VAMP signals from volume flow.
  - `BasicRiskManager`: Enforces safety limits.
- **Adapters Layer**: Handles communication with the outside world.
  - `ThalexAdapter`: WebSocket/REST integration with Thalex.
  - `TimescaleDBAdapter`: High-throughput time-series storage.
- **Use Case Layer**:
  - `QuotingService`: Orchestrates data flow between the exchange and strategy.

### Legacy Core (`thalex_py/`)
The legacy system remains available for established workflows, featuring:
- **Object Pooling**: Extremely low latency by reusing message structures.
- **Shared Memory**: Optimized data access for high-frequency updates.
- **Direct Event Bus**: Low-overhead internal communication.

---

## 📈 Technical Features

### Avellaneda-Stoikov Model
The bot calculates optimal bid/ask spreads based on:
- **Gamma (γ)**: Risk aversion parameter.
- **Kappa (κ)**: Market depth/liquidity factor.
- **Inventory Risk**: Dynamic skewing of quotes based on current position.

### VAMP (Volume Adjusted Market Pressure)
Integrated into the `VolumeCandleSignalEngine`, VAMP analyzes aggressive order flow to detect:
- Momentum and reversal likelihood.
- Potential exhaustion points.
- Volatility predictions for spread adjustment.

### Risk Management
- **Position Limits**: Max BTC exposure per instrument.
- **Drawdown Protection**: Automatic halting on significant losses.
- **Take Profit**: Built-in triggers for unrealized PnL.

---

## ⚠️ Risk Warning

Trading cryptocurrencies involves substantial risk of loss. This software is provided "as-is" with no guarantees. Always test thoroughly on testnet before deploying real capital.

## 📄 License

This project is licensed under the MIT License.