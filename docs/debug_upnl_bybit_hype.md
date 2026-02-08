# Debugging Guide: UPNL Calculation for Bybit Futures Positions

## Executive Summary

This guide provides a systematic approach to debugging why Unrealized Profit and Loss (UPNL) calculations are not displaying correctly for futures positions on Bybit, specifically for the HYPE (Hyperliquid) trading pair. The investigation framework covers WebSocket data feed integrity, price data pipeline, calculation engine, data source configuration, error handling, and logging analysis.

## Affected Files

The following files are involved in the UPNL calculation and display pipeline for Bybit futures:

| File | Purpose |
|------|---------|
| [`src/adapters/exchanges/bybit_adapter.py`](src/adapters/exchanges/bybit_adapter.py) | Bybit WebSocket connection, position updates, ticker handling |
| [`src/domain/entities/__init__.py`](src/domain/entities/__init__.py) | Position entity with mark_price, unrealized_pnl, delta fields |
| [`src/services/instrument_service.py`](src/services/instrument_service.py) | Symbol mapping between internal and exchange formats |
| [`src/services/market_feed.py`](src/services/market_feed.py) | Market data feed orchestration |
| [`src/use_cases/sim_state_manager.py`](src/use_cases/sim_state_manager.py) | UPNL calculation in `get_status()` method |
| [`src/domain/tracking/state_tracker.py`](src/domain/tracking/state_tracker.py) | Position state management |
| [`config.json`](config.json) | Configuration for Bybit symbols |

---

## 1. Real-Time Data Feed Integrity

### Symptom
UPNL showing zero or missing values because mark price data is not being received.

### Investigation Steps

#### 1.1 Verify WebSocket Connection Status
The Bybit adapter connects to `wss://stream.bybit.com/v5/public/linear` for linear futures. Check if the connection is active:

```python
# Location: src/adapters/exchanges/bybit_adapter.py:546-584

# Check in _public_msg_loop()
async def _public_msg_loop(self):
    while self.connected:
        try:
            async with self.session.ws_connect(self.ws_public_url) as ws:
                self.ws_public = ws
                logger.info("Connected to Bybit Public WS")
                
                # Verify connection success
                if not ws.closed:
                    logger.info(f"WebSocket state: {ws.closed}, Bybit public stream active")
```

**Debugging Checkpoint**: Add logging to verify:
- Connection establishment at line 561
- Message loop iteration at line 565
- Any reconnection attempts at line 582

#### 1.2 Verify Mark Price Subscription Format
Bybit requires specific topic format for mark prices. The current implementation only subscribes to `orderbook` and `publicTrade` topics but NOT mark prices.

**Critical Issue Identified**: In [`bybit_adapter.py:528-544`](src/adapters/exchanges/bybit_adapter.py:528-544), the `subscribe_ticker()` method only subscribes to:

```python
topic = f"orderbook.1.{mapped_symbol}"  # Line 532
```

This does NOT include mark price subscription. Bybit mark price topic format is:
- `ticker.{symbol}` for 1-second ticker updates
- For linear futures: `ticker.BTCUSDT`, `ticker.HYPEUSDT`

**Verification Steps**:
1. Check if the subscription message includes mark price topic
2. Verify the topic format matches Bybit documentation for linear futures
3. Confirm HYPEUSDT is a valid Bybit linear futures symbol

#### 1.3 Verify WebSocket Authentication and Heartbeat
The private WebSocket connection requires authentication:

```python
# Location: src/adapters/exchanges/bybit_adapter.py:115-141

async def connect(self):
    # Authentication at lines 124-134
    signature = hmac.new(
        self.api_secret.encode("utf-8"),
        f"GET/realtime{expires}".encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    
    auth_msg = {"op": "auth", "args": [self.api_key, expires, signature]}
    await self.ws_private.send_json(auth_msg)
```

**Debugging Checkpoint**: Add logging to verify:
- Authentication response from Bybit
- Ping/pong messages at [`_ping_loop()`](src/adapters/exchanges/bybit_adapter.py:246-256)
- Connection status indicators

---

## 2. Price Data Pipeline

### Symptom
Mark price data is received but not properly parsed, stored, or propagated to UPNL calculation.

### Investigation Steps

#### 2.1 Verify Mark Price Data Reception
Check if mark price messages are being received and handled in the public message loop:

```python
# Location: src/adapters/exchanges/bybit_adapter.py:556-584

async def _public_msg_loop(self):
    while self.connected:
        try:
            msg = await asyncio.wait_for(ws.receive_json(), timeout=5.0)
            topic = msg.get("topic", "")
            
            # Current handlers - MISSING mark price handler!
            if topic.startswith("orderbook"):
                await self._handle_orderbook_msg(msg)
            elif topic.startswith("publicTrade"):
                await self._handle_public_trade_msg(msg)
            # NO handler for ticker/markPrice!
```

**Critical Issue**: No handler exists for `ticker` topic which contains mark price data.

#### 2.2 Verify Ticker Entity Structure
The [`Ticker`](src/domain/entities/__init__.py:37-54) entity does not include a `mark_price` field:

```python
@dataclass(slots=True)
class Ticker:
    symbol: str
    bid: float
    ask: float
    bid_size: float
    ask_size: float
    last: float      # Last trade price - NOT mark price
    volume: float
    exchange: str = "thalex"
    timestamp: float = field(default_factory=time.time)
```

**Issue**: The `Ticker` entity uses `last` price instead of `mark_price`. Bybit provides mark price separately from last price.

#### 2.3 Verify Position Entity Mark Price Field
The [`Position`](src/domain/entities/__init__.py:87-98) entity has a `mark_price` field that is never populated:

```python
@dataclass(slots=True)
class Position:
    symbol: str
    size: float
    entry_price: float
    exchange: str = "thalex"
    mark_price: float = 0.0           # Default 0.0 - NEVER UPDATED
    unrealized_pnl: float = 0.0        # Default 0.0 - NEVER CALCULATED
    realized_pnl: float = 0.0
    delta: float = 0.0
    gamma: float = 0.0                 # Options only - correct to be 0 for futures
    theta: float = 0.0                  # Options only - correct to be 0 for futures
    timestamp: float = field(default_factory=time.time)
```

#### 2.4 Verify Position Update Handler
The [`_handle_position_update()`](src/adapters/exchanges/bybit_adapter.py:312-322) method only updates size and entry_price:

```python
async def _handle_position_update(self, data: Dict):
    symbol = data.get("symbol")
    if symbol:
        amount = self._safe_float(data.get("size"))
        entry_price = self._safe_float(data.get("entryPrice"))
        self.positions[symbol] = Position(
            symbol, amount, entry_price, exchange=self.name
        )
        # mark_price is NEVER set from position update data!
```

**Debugging Checkpoint**:
1. Log incoming position data to verify structure
2. Check if Bybit position WebSocket includes mark price
3. Verify mark price field is populated after position update

---

## 3. UPNL Calculation Engine

### Symptom
Required variables for UPNL calculation are missing or undefined, breaking the computation chain.

### Investigation Steps

#### 3.1 Verify UPNL Formula Implementation
The standard futures UPNL formula is:
```
UPNL = position_size × (mark_price - entry_price) × position_direction
```

Where:
- `position_size`: Absolute position size
- `mark_price`: Current mark price from exchange
- `entry_price`: Average entry price
- `position_direction`: +1 for long, -1 for short

#### 3.2 Check UPNL Calculation in StateTracker
The [`StateTracker.get_position()`](src/domain/tracking/state_tracker.py:233-234) returns Position without UPNL calculation:

```python
def get_position(self, symbol: str) -> Position:
    return self.positions.get(symbol, Position(symbol, 0.0, 0.0))
```

**Issue**: Returns Position with default mark_price=0.0 and unrealized_pnl=0.0.

#### 3.3 Check UPNL Calculation in SimStateManager
The [`get_status()`](src/use_cases/sim_state_manager.py:211-231) method calculates UPNL:

```python
def get_status(self) -> Dict:
    unrealized = 0.0
    if s.position_size != 0 and s.last_price > 0:
        unrealized = (s.last_price - s.position_entry_price) * s.position_size
```

**Issue**: Uses `last_price` instead of `mark_price`. This is incorrect for futures UPNL calculation because:
- Last price is the last executed trade price
- Mark price is the settlement price used for margin and PnL calculations

---

## 4. Data Source Configuration

### Symptom
HYPE trading pair is not correctly mapped to Bybit symbol identifier.

### Investigation Steps

#### 4.1 Verify Symbol Mapping
Check [`src/services/instrument_service.py`](src/services/instrument_service.py:16-25):

```python
_MAPPINGS = {
    "bybit": {
        "BTC-PERPETUAL": "BTCUSDT",
        "ETH-PERPETUAL": "ETHUSDT"
    },
    # NO HYPE MAPPING!
}
```

**Critical Issue**: HYPE is not in the symbol mapping. The configuration in [`config.json`](config.json:11) shows:

```json
"bybit": {
    "symbols": ["HYPEUSDT"],
    ...
}
```

When `InstrumentService.get_exchange_symbol("HYPEUSDT", "bybit")` is called, it returns `"HYPEUSDT"` directly because there's no mapping entry.

**Debugging Checkpoint**: Verify:
1. HYPEUSDT is a valid Bybit linear futures symbol
2. The mapping exists or passthrough works correctly
3. WebSocket subscription uses the correct symbol format

#### 4.2 Verify Product Type Configuration
Bybit uses different endpoints for:
- `linear`: USDC/USDT settled (e.g., BTCUSDT, HYPEUSDT)
- `inverse`: Coin-settled (e.g., BTCUSD)

The [`fetch_instrument_info()`](src/adapters/exchanges/bybit_adapter.py:143-179) method defaults to `category="linear"`:

```python
async def fetch_instrument_info(self, symbol: str, category: str = "linear") -> Dict:
```

**Debugging Checkpoint**: Verify HYPEUSDT is a linear futures contract on Bybit.

---

## 5. Error Handling and Edge Cases

### Symptom
Exception handlers silently catching and suppressing errors.

### Investigation Steps

#### 5.1 Check Exception Handling in Message Loop
The [`_msg_loop()`](src/adapters/exchanges/bybit_adapter.py:258-274) has broad exception handling:

```python
async def _msg_loop(self):
    while self.connected and self.ws_private:
        try:
            msg = await asyncio.wait_for(self.ws_private.receive_json(), timeout=5.0)
            await self._handle_message(msg)
        except asyncio.TimeoutError:
            continue
        except asyncio.CancelledError:
            break
        except Exception as e:
            import traceback
            logger.error(f"Error in Bybit msg loop: {e}\n{traceback.format_exc()}")
            # Error is logged but connection continues!
```

**Issue**: Errors are logged but may not surface to the UI. Add more granular error handling.

#### 5.2 Check Position State Transitions
Verify position is not in transitional state:

```python
# Location: src/adapters/exchanges/bybit_adapter.py:312-322

async def _handle_position_update(self, data: Dict):
    symbol = data.get("symbol")
    if symbol:
        amount = self._safe_float(data.get("size"))
        entry_price = self._safe_float(data.get("entryPrice"))
        # Position may be 0 during transitions
        self.positions[symbol] = Position(
            symbol, amount, entry_price, exchange=self.name
        )
```

**Debugging Checkpoint**:
1. Log position size after update
2. Verify position is not zero during the issue
3. Check for position flip scenarios

---

## 6. Controlled Testing Scenarios

### Symptom
Need to isolate whether issue is HYPE-specific or systemic.

### Test Procedure

#### 6.1 Create Test Position with Known Parameters
1. Open a small test position on HYPEUSDT
2. Record: entry_price, position_size, position_direction (long/short)
3. Get current mark price from Bybit API:
```python
# REST API call to verify mark price
import aiohttp

async def get_mark_price(symbol: str):
    url = "https://api.bybit.com/v5/market/tickers"
    params = {"category": "linear", "symbol": symbol}
    async with aiohttp.ClientSession() as session:
        async with session.get(url, params=params) as resp:
            data = await resp.json()
            if data.get("retCode") == 0:
                return data["result"]["list"][0]["markPrice"]
```

#### 6.2 Manual UPNL Calculation
Calculate expected UPNL:
```
Long Position: UPNL = size × (mark_price - entry_price)
Short Position: UPNL = size × (entry_price - mark_price)
```

#### 6.3 Compare with Displayed Values
1. Check UI/API for displayed UPNL
2. Verify against manual calculation
3. If UPNL=0, check if mark_price is being used vs last_price

#### 6.4 Test with Another Futures Pair
Repeat test with a known working pair (e.g., BTCUSDT or ETHUSDT):
1. Open small position
2. Verify UPNL calculation works
3. If BTCUSDT works but HYPEUSDT doesn't → HYPE-specific issue
4. If both fail → Systemic issue with UPNL calculation

---

## 7. Logging and Monitoring

### Symptom
Need to identify warnings, errors, or exceptions related to UPNL calculations.

### Investigation Steps

#### 7.1 Enable Debug Logging
Set environment variable:
```
LOG_LEVEL=DEBUG
VERBOSE_LOGGING=true
```

#### 7.2 Key Log Locations to Monitor

| Component | Log Level | What to Look For |
|-----------|-----------|------------------|
| BybitAdapter | DEBUG | WebSocket connection, subscription confirmations |
| Position Updates | INFO | Size, entry_price, mark_price updates |
| Ticker Processing | DEBUG | Mark price values in ticker messages |
| UPNL Calculation | DEBUG | Formula inputs and results |
| Symbol Mapping | DEBUG | Mapped symbol values |

#### 7.3 Search for Specific Log Patterns
```bash
# Search for position-related logs
grep -r "position" logs/
grep -r "HYPE" logs/
grep -r "markPrice\|mark_price" logs/
grep -r "unrealized_pnl\|UPNL" logs/
grep -r "ticker" logs/
```

#### 7.4 Verify Log Files Exist
```bash
# Check log directory
ls -la logs/
# Check for Bybit-specific logs
grep -l "bybit" logs/*
```

---

## 8. Step-by-Step Fix Implementation

### 8.1 Add Mark Price Subscription
In [`src/adapters/exchanges/bybit_adapter.py`](src/adapters/exchanges/bybit_adapter.py), add mark price subscription:

```python
async def subscribe_mark_price(self, symbol: str):
    """Subscribe to mark price stream for futures."""
    await self._ensure_public_conn()
    mapped_symbol = InstrumentService.get_exchange_symbol(symbol, self.name)
    topic = f"ticker.{mapped_symbol}"  # Bybit mark price topic
    if self.ws_public and not self.ws_public.closed:
        await self.ws_public.send_json({"op": "subscribe", "args": [topic]})
        logger.info(f"Subscribed to mark price: {topic}")
```

### 8.2 Add Mark Price Message Handler
In [`src/adapters/exchanges/bybit_adapter.py`](src/adapters/exchanges/bybit_adapter.py:556-584), add handler:

```python
async def _handle_ticker_msg(self, msg: Dict):
    """Handle ticker/mark price updates."""
    topic = msg.get("topic", "")  # ticker.BTCUSDT
    data = msg.get("data", {})
    
    symbol = topic.replace("ticker.", "")
    mark_price = self._safe_float(data.get("markPrice"))
    
    logger.debug(f"Ticker update - Symbol: {symbol}, Mark Price: {mark_price}")
    
    # Update position mark price if exists
    if symbol in self.positions:
        self.positions[symbol].mark_price = mark_price
```

### 8.3 Update Message Loop to Handle Ticker
In [`_public_msg_loop()`](src/adapters/exchanges/bybit_adapter.py:570-574):

```python
topic = msg.get("topic", "")
if topic.startswith("orderbook"):
    await self._handle_orderbook_msg(msg)
elif topic.startswith("publicTrade"):
    await self._handle_public_trade_msg(msg)
elif topic.startswith("ticker"):
    await self._handle_ticker_msg(msg)  # ADD THIS LINE
```

### 8.4 Update Market Feed Service to Subscribe
In [`src/services/market_feed.py`](src/services/market_feed.py:94-99):

```python
elif adapter.name == "bybit":
    if hasattr(adapter, "subscribe_ticker"):
        await adapter.subscribe_ticker(sym)
    if hasattr(adapter, "subscribe_trades"):
        await adapter.subscribe_trades(sym)
    if hasattr(adapter, "subscribe_mark_price"):  # ADD THIS
        await adapter.subscribe_mark_price(sym)
```

### 8.5 Update UPNL Calculation to Use Mark Price
In [`src/use_cases/sim_state_manager.py`](src/use_cases/sim_state_manager.py:213-215):

```python
def get_status(self) -> Dict:
    unrealized = 0.0
    # Use mark_price if available, fallback to last_price
    price_for_upnl = getattr(self, 'mark_price', None) or s.last_price
    if s.position_size != 0 and price_for_upnl > 0:
        unrealized = (price_for_upnl - s.position_entry_price) * s.position_size
```

---

## 9. Verification Checklist

After implementing fixes, verify the following:

- [ ] WebSocket connection to Bybit public stream is established
- [ ] Subscription confirmation for `ticker.HYPEUSDT` is logged
- [ ] Mark price messages are received and parsed correctly
- [ ] Position entity `mark_price` field is updated
- [ ] UPNL calculation uses mark price (not last price)
- [ ] Display shows non-zero UPNL for open positions
- [ ] Gamma and theta remain 0 for futures (options-only metrics)
- [ ] UPNL calculation works for other futures pairs (BTCUSDT, ETHUSDT)
- [ ] No errors in logs related to HYPE symbol handling
- [ ] Position size and direction are correctly factored into UPNL

---

## 10. Quick Reference: Bybit Mark Price Data

### WebSocket Topic Format
| Topic | Description | Message Frequency |
|-------|-------------|-------------------|
| `ticker.{symbol}` | 1-second ticker with mark price | 1Hz |
| `publicTrade.{symbol}` | Public trades | On trade |
| `orderbook.1.{symbol}` | Order book (1=1-depth) | On update |

### Ticker Message Structure
```json
{
    "topic": "ticker.BTCUSDT",
    "type": "snapshot",
    "data": {
        "symbol": "BTCUSDT",
        "lastPrice": "65000.00",
        "markPrice": "65005.50",
        "indexPrice": "64998.00",
        ...
    }
}
```

### Key Fields for UPNL
- `markPrice`: Used for PnL calculation and margin
- `indexPrice`: Reference index price
- `lastPrice`: Last executed trade price

---

## Conclusion

The UPNL display issue for HYPE on Bybit is likely caused by a combination of:

1. **Missing mark price subscription**: The system only subscribes to orderbook/trades, not ticker/markPrice
2. **Incorrect price source**: UPNL calculation uses `last_price` instead of `mark_price`
3. **Symbol mapping gap**: HYPE may not have explicit mapping (though passthrough should work)
4. **Missing handler**: No ticker message handler exists in the public message loop

Follow the debugging steps in order, implement the suggested fixes, and use the verification checklist to confirm resolution.
