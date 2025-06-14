# Implementation Tasks: Trading System Enhancements

## ✅ COMPLETED: Simple UPNL Take Profit Implementation

### Summary of Completed Work
**Status**: Phases 1-4 completed successfully ✅

A basic UPNL-based take profit system has been implemented with the following components:

#### **Phase 1: Configuration ✅**
- Added 5 take profit parameters to `market_config.py` risk section
- Added 4 state variables to `avellaneda_quoter.py`

#### **Phase 2: UPNL Monitoring ✅** 
- `_get_current_upnl()` method extracts UPNL from position tracker
- `_check_take_profit_conditions()` method checks thresholds with cooldowns

#### **Phase 3: Position Flattening ✅**
- `_execute_take_profit_flatten()` method places market orders to close positions
- Integrated with order manager for reduce-only market orders

#### **Phase 4: Integration ✅**
- Replaced legacy take profit variables with simplified system
- Integrated UPNL checks into main quote task loop

### ⚠️ **LIMITATION IDENTIFIED**
**The current implementation is NOT suitable for futures spread trading** because:
- It only monitors total position UPNL, not spread-specific PnL
- It doesn't understand futures-perpetual hedging relationships
- It can't calculate spread convergence profits correctly

---

## 🎯 NEW TASK: Spread-Aware Take Profit for Futures Trading

### Overview
The bot trades futures spreads where taking a position in futures automatically hedges with opposite position in perpetuals:
- **Long BTC-25JUL25** → **Short BTC-PERPETUAL** (same size)  
- **Short BTC-25JUL25** → **Long BTC-PERPETUAL** (same size)

**Profit comes from spread convergence, not absolute position PnL.**

### Current Challenge
Replace the simple UPNL-based take profit with spread-aware logic that:
- Tracks both futures and perpetual positions separately
- Calculates spread PnL correctly
- Takes profit when the spread is profitable (not individual legs)
- Maintains hedged position integrity

## Phase 1: Spread Position Tracking

### Step 1.1: Add Dual Take Profit Configuration
**File**: `thalex_py/Thalex_modular/config/market_config.py`
**Location**: Add to existing `"risk"` section (keep existing simple UPNL config)

```python
# Keep existing simple UPNL take profit:
"take_profit_enabled": True,              # Enable simple UPNL-based take profit
"take_profit_threshold": 100.0,           # Take profit at $100 total UPNL
"take_profit_check_interval": 5,          # Check every 5 seconds
"flatten_position_enabled": True,         # Allow position flattening
"take_profit_cooldown": 30,               # 30 second cooldown after take profit

# ADD new spread-aware take profit:
"spread_take_profit_enabled": True,       # Enable spread-based take profit
"spread_profit_threshold_usd": 50.0,      # Take profit at $50 spread profit  
"spread_check_interval": 3,               # Check every 3 seconds
"spread_flatten_both_legs": True,         # Close both futures and perpetual
"spread_profit_cooldown": 60,             # 60 second cooldown after take profit
"min_spread_position_size": 0.01,         # Minimum position size to monitor
"spread_priority_over_simple": True,      # If both trigger, spread takes priority
```

### Step 1.2: Add Spread State Variables  
**File**: `thalex_py/Thalex_modular/avellaneda_quoter.py`
**Location**: Add after existing take profit state variables (keep existing ones)

```python
# Keep existing simple take profit state:
# self.take_profit_active = False
# self.last_take_profit_check = 0  
# self.take_profit_cooldown_until = 0
# self.last_upnl_value = 0.0

# ADD new spread take profit state management:
self.spread_take_profit_active = False
self.last_spread_check = 0
self.spread_cooldown_until = 0
self.last_spread_pnl = 0.0
self.futures_position_size = 0.0
self.perpetual_position_size = 0.0
self.current_spread_value = 0.0
```

## Phase 2: Spread PnL Calculation

### Step 2.1: Add Spread PnL Extraction Method
**File**: `thalex_py/Thalex_modular/avellaneda_quoter.py`  
**Location**: Add after existing `_get_current_upnl` method (keep existing one)

```python
def _get_spread_pnl(self) -> float:
    """Calculate spread PnL from both legs of the trade"""
    try:
        # Get position metrics for both instruments
        futures_metrics = self.position_tracker.get_position_metrics(self.futures_instrument_name) 
        perpetual_metrics = self.position_tracker.get_position_metrics(self.perp_name)
        
        # Extract individual leg PnLs
        futures_pnl = futures_metrics.get("unrealized_pnl", 0.0)
        perpetual_pnl = perpetual_metrics.get("unrealized_pnl", 0.0)
            
        # Calculate spread PnL (sum of both legs)
        spread_pnl = futures_pnl + perpetual_pnl
        
        # Update tracking variables
        self.futures_position_size = futures_metrics.get("position", 0.0)
        self.perpetual_position_size = perpetual_metrics.get("position", 0.0)
        
        # Calculate current spread value
        if self.futures_ticker and self.ticker:
            self.current_spread_value = self.futures_ticker.mark_price - self.ticker.mark_price
            
        return float(spread_pnl)
        
    except Exception as e:
        self.logger.error(f"Error calculating spread PnL: {str(e)}")
        return 0.0

def _is_valid_spread_position(self) -> bool:
    """Check if we have a valid spread position to monitor"""
    min_size = RISK_LIMITS.get("min_spread_position_size", 0.01)
    
    # Check if both legs exist and are roughly opposite
    if (abs(self.futures_position_size) < min_size or 
        abs(self.perpetual_position_size) < min_size):
        return False
        
    # Check if positions are roughly opposite (spread trade)
    position_ratio = abs(self.futures_position_size / self.perpetual_position_size)
    if not (0.8 <= position_ratio <= 1.2):  # Allow 20% tolerance
        self.logger.warning(f"Position sizes not balanced: futures={self.futures_position_size:.4f}, perp={self.perpetual_position_size:.4f}")
        
    return True

async def _check_spread_take_profit_conditions(self) -> bool:
    """Check if spread take profit conditions are met"""
    if not RISK_LIMITS.get("spread_take_profit_enabled", True):
        return False
        
    current_time = time.time()
    
    # Check cooldown
    if current_time < self.spread_cooldown_until:
        return False
        
    # Check interval
    if current_time - self.last_spread_check < RISK_LIMITS.get("spread_check_interval", 3):
        return False
        
    self.last_spread_check = current_time
    
    # Check if we have valid spread positions
    if not self._is_valid_spread_position():
        return False
    
    # Get current spread PnL
    current_spread_pnl = self._get_spread_pnl()
    self.last_spread_pnl = current_spread_pnl
    
    # Check threshold
    profit_threshold = RISK_LIMITS.get("spread_profit_threshold_usd", 50.0)
    
    if current_spread_pnl >= profit_threshold:
        self.logger.info(f"Spread take profit triggered: PnL ${current_spread_pnl:.2f} >= ${profit_threshold:.2f} (Spread: {self.current_spread_value:.2f})")
        return True
        
    return False
```

## Phase 3: Spread Position Flattening

### Step 3.1: Add Spread Flattening Method
**File**: `thalex_py/Thalex_modular/avellaneda_quoter.py`
**Location**: Add after existing `_execute_take_profit_flatten` method (keep existing one)

```python
async def _execute_spread_take_profit_flatten(self):
    """Execute spread take profit by flattening both legs"""
    try:
        if not self._is_valid_spread_position():
            self.logger.info("No valid spread position to flatten")
            return True
            
        self.logger.critical(f"EXECUTING SPREAD TAKE PROFIT: Spread PnL ${self.last_spread_pnl:.2f}, Futures: {self.futures_position_size:.4f}, Perp: {self.perpetual_position_size:.4f}")
        
        # Cancel all existing quotes first
        await self.cancel_quotes("Spread take profit execution")
        await asyncio.sleep(1)
        
        # Close futures position
        if abs(self.futures_position_size) >= RISK_LIMITS.get("min_spread_position_size", 0.01):
            futures_side = "sell" if self.futures_position_size > 0 else "buy"
            
            if hasattr(self, 'order_manager') and self.order_manager:
                futures_result = await self.order_manager.place_order(
                    instrument=self.futures_instrument_name,
                    direction=futures_side,
                    price=None,  # Market order
                    amount=abs(self.futures_position_size),
                    label="SpreadTakeProfit",
                    post_only=False  # Market orders can't be post-only
                )
                self.logger.info(f"Spread TP: Futures close order placed: {futures_result}")
        
        # Close perpetual position  
        if abs(self.perpetual_position_size) >= RISK_LIMITS.get("min_spread_position_size", 0.01):
            perp_side = "sell" if self.perpetual_position_size > 0 else "buy"
            
            if hasattr(self, 'order_manager') and self.order_manager:
                perp_result = await self.order_manager.place_order(
                    instrument=self.perp_name,
                    direction=perp_side,
                    price=None,  # Market order
                    amount=abs(self.perpetual_position_size),
                    label="SpreadTakeProfit",
                    post_only=False  # Market orders can't be post-only
                )
                self.logger.info(f"Spread TP: Perpetual close order placed: {perp_result}")
            
        # Set cooldown
        self.spread_cooldown_until = time.time() + RISK_LIMITS.get("spread_profit_cooldown", 60)
        self.spread_take_profit_active = True
        
        return True
        
    except Exception as e:
        self.logger.error(f"Error executing spread take profit: {str(e)}")
        return False
```

## Phase 4: Integration and Monitoring

### Step 4.1: Update Quote Task Integration
**File**: `thalex_py/Thalex_modular/avellaneda_quoter.py`
**Location**: Update existing take profit check in quote_task to support both systems

```python
# Update existing take profit check to support BOTH systems:
if self.active_trading and not self.risk_recovery_mode:
    # Check spread take profit first (if enabled and has priority)
    if (RISK_LIMITS.get("spread_take_profit_enabled", False) and 
        await self._check_spread_take_profit_conditions()):
        await self._execute_spread_take_profit_flatten()
        # Brief pause after spread take profit
        await asyncio.sleep(3.0)
        continue
    
    # Check simple UPNL take profit (if enabled and spread didn't trigger)
    elif (RISK_LIMITS.get("take_profit_enabled", False) and 
          await self._check_take_profit_conditions()):
        await self._execute_take_profit_flatten()
        # Brief pause after simple take profit
        await asyncio.sleep(2.0)
        continue
```

### Step 4.2: Add Dual Take Profit Monitoring to Risk Task
**File**: `thalex_py/Thalex_modular/avellaneda_quoter.py`
**Location**: Add to `_risk_monitoring_task` main loop

```python
# Add dual take profit monitoring (separate from individual risk checks):
if self.active_trading and not self.risk_recovery_mode:
    # Check spread take profit first (if enabled and has priority)
    if (RISK_LIMITS.get("spread_take_profit_enabled", False) and 
        await self._check_spread_take_profit_conditions()):
        await self._execute_spread_take_profit_flatten()
    
    # Check simple UPNL take profit (if enabled and spread didn't trigger)
    elif (RISK_LIMITS.get("take_profit_enabled", False) and 
          await self._check_take_profit_conditions()):
        await self._execute_take_profit_flatten()
```

### Step 4.3: Add Dual Take Profit Status Logging
**File**: `thalex_py/Thalex_modular/avellaneda_quoter.py`
**Location**: Add to status logging methods

```python
def _log_take_profit_status(self):
    """Log current take profit status for both systems"""
    # Log simple UPNL take profit status
    if RISK_LIMITS.get("take_profit_enabled", False):
        current_upnl = self._get_current_upnl()
        threshold = RISK_LIMITS.get("take_profit_threshold", 100.0)
        self.logger.info(f"Simple TP: UPNL ${current_upnl:.2f} / ${threshold:.2f}")
        
        if self.take_profit_active:
            cooldown_remaining = max(0, self.take_profit_cooldown_until - time.time())
            self.logger.info(f"Simple TP: Cooldown active for {cooldown_remaining:.1f}s")
    
    # Log spread take profit status  
    if RISK_LIMITS.get("spread_take_profit_enabled", False):
        if self._is_valid_spread_position():
            spread_pnl = self._get_spread_pnl()
            threshold = RISK_LIMITS.get("spread_profit_threshold_usd", 50.0)
            
            self.logger.info(f"Spread TP: PnL ${spread_pnl:.2f} / ${threshold:.2f} | "
                            f"Futures: {self.futures_position_size:.4f} | "
                            f"Perp: {self.perpetual_position_size:.4f} | "
                            f"Spread: {self.current_spread_value:.2f}")
        
        if self.spread_take_profit_active:
            cooldown_remaining = max(0, self.spread_cooldown_until - time.time())
            self.logger.info(f"Spread TP: Cooldown active for {cooldown_remaining:.1f}s")
```

## Implementation Summary

### Files to Modify:
1. **`market_config.py`** - Add 7 new spread parameters (keep existing 5 simple TP parameters)
2. **`avellaneda_quoter.py`** - Add 4 new methods, modify 2 existing methods, add 7 spread state variables

### Key Benefits:
- **Dual Take Profit Systems**: Both simple UPNL and spread-aware logic available
- **Independent Toggle Control**: Enable/disable each system separately via config
- **Priority System**: Spread take profit can take priority over simple TP when both triggered
- **Spread-Aware Logic**: Calculates PnL from both legs of futures trades
- **Position Validation**: Ensures balanced spread positions before monitoring
- **Dual-Leg Closure**: Closes both futures and perpetual positions simultaneously
- **Unified Integration**: Both systems work seamlessly in existing risk framework

### Testing Strategy:
1. Verify spread PnL calculation accuracy with both legs
2. Test position balance validation for spread trades  
3. Confirm dual-leg flattening executes correctly
4. Validate cooldown periods work properly
5. Check integration with existing futures trading logic

## Execution Status:
- [ ] Phase 1: Spread Configuration  
- [ ] Phase 2: Spread PnL Calculation
- [ ] Phase 3: Spread Position Flattening
- [ ] Phase 4: Integration and Monitoring

**Next Steps**: Begin Phase 1 implementation with spread-aware configuration. 