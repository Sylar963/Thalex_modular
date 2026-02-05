# SvelteKit Dashboard: Frontend Handoff

## Overview

This document outlines the frontend features needed for the Thalex Modular Trading Dashboard. The backend API is already implemented in FastAPI and serves data from TimescaleDB.

---

## Required Features

### 1. Venue Selector Component
**Purpose**: Allow users to filter data by exchange

**Requirements**:
- Dropdown with options: `All`, `Thalex`, `Bybit`, `Binance`, `Hyperliquid`
- Store selection in a Svelte store for global access
- Apply filter to positions table, orders table, and charts

**API Endpoint**: `GET /api/v1/portfolio/positions?exchange=bybit`

---

### 2. Aggregate Portfolio View
**Purpose**: Show combined metrics across all connected venues

**Display Fields**:
- Total Unrealized PnL
- Total Realized PnL
- Aggregate Position Size (per symbol)
- Account Equity (if available per-venue)

**API Endpoint**: `GET /api/v1/portfolio/summary`

---

### 3. Per-Venue Strategy Display
**Purpose**: Show which strategy parameters each venue is using

**Display**:
- Card per venue showing: `gamma`, `volatility`, `position_limit`, `order_size`
- Highlight if venue uses custom params vs defaults

**Data Source**: Parse from `/api/v1/config` or embed in portfolio response

---

### 4. Symbol Sync (Window Linking)
**Purpose**: When user selects a symbol in one panel, update all linked panels

**Implementation**:
- Global Svelte store: `linkedSymbol`
- Panels subscribe to store and update their data queries
- Toggle button to enable/disable linking per panel

---

### 5. Signal Overlays on Chart
**Purpose**: Display VAMP and Open Range signals on price chart

**API Endpoint**: `GET /api/v1/signals/history?symbol=HYPEUSDT&type=vamp&limit=100`

**Display**:
- VAMP signals as vertical lines at signal timestamp
- Open Range as horizontal bands showing range high/low

---

## Available API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/portfolio/positions` | GET | Current positions (optionally filtered by exchange) |
| `/api/v1/portfolio/summary` | GET | Account summary |
| `/api/v1/market/history` | GET | OHLCV candles for charting |
| `/api/v1/market/ticker` | GET | Current ticker data |
| `/api/v1/simulation/results` | GET | Backtest results |
| `/api/v1/signals/history` | GET | Historical signal data |

---

## Tech Stack

- **Framework**: SvelteKit
- **Charting**: Lightweight Charts (TradingView) or similar
- **State**: Svelte stores
- **Styling**: TailwindCSS or custom CSS

---

## Notes

- All API responses are JSON
- WebSocket/SSE endpoints available for real-time updates (TBD)
- Dashboard should work in both light and dark mode
- Mobile responsiveness is secondary priority

---

*Contact backend team for API schema details or additional endpoints.*
