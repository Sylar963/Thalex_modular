

# Implementation Tasks: Trading System Enhancements

## ✅ COMPLETED: Risk Recovery System Implementation

### Summary of Completed Work
**Status**: All phases completed successfully ✅

The Risk Recovery System has been fully implemented with the following enhancements:

- **Recovery Configuration**: Added 5 new parameters to `market_config.py` for recovery behavior control
- **State Management**: Added recovery state variables to track halt/recovery status  
- **Enhanced Risk Breach Handler**: Modified to activate recovery mode instead of permanent halt
- **Recovery Logic**: Implemented gradual 3-step recovery process with cooldown periods
- **Quote Task Integration**: Updated to check for recovery conditions instead of exiting on halt
- **Risk Monitor Enhancement**: Added periodic recovery condition checking

**Key Features Delivered**:
- 5-minute cooldown after risk breach before recovery attempts
- 80% threshold for position/notional limits before recovery initiation
- 3-step gradual recovery to prevent immediate re-breach
- 30-second interval recovery condition checks
- Comprehensive logging for all recovery events
- Full backward compatibility with existing functionality

---

## 🎯 NEW TASK: Take Profit Logic Simplification

### Overview
Replace current take profit logic with simplified UPNL-based monetary thresholds. The new logic should monitor Unrealized PnL from position data and flatten positions when reaching specific dollar amounts.

### Current Challenge
The existing take profit mechanism needs to be replaced with a simpler approach that:
- Monitors UPNL (Unrealized PnL) from position data
- Triggers position flattening at specific monetary values
- Removes complex conditional logic in favor of direct dollar-based thresholds

## Phase 1: Add Take Profit Configuration

### Step 1.1: Add UPNL Take Profit Configuration
**File**: `thalex_py/Thalex_modular/config/market_config.py`
**Location**: Add to existing `"risk"` section

```python
# Add these to existing BOT_CONFIG["risk"]
"take_profit_enabled": True,           # Enable UPNL-based take profit
"take_profit_threshold": 100.0,        # Take profit at $100 UPNL
"take_profit_check_interval": 5,       # Check every 5 seconds
"flatten_position_enabled": True,       # Allow position flattening
"take_profit_cooldown": 30,            # 30 second cooldown after take profit
```

### Step 1.2: Add Take Profit State Variables
**File**: `thalex_py/Thalex_modular/avellaneda_quoter.py`
**Location**: In `__init__` method after existing risk recovery variables

```python
# Take profit state management (ADD AFTER risk recovery variables)
self.take_profit_active = False
self.last_take_profit_check = 0
self.take_profit_cooldown_until = 0
self.last_upnl_value = 0.0
```

## Phase 2: Implement UPNL Monitoring

### Step 2.1: Add UPNL Extraction Method
**File**: `thalex_py/Thalex_modular/avellaneda_quoter.py`
**Location**: Add new method after recovery methods

```python
def _get_current_upnl(self) -> float:
    """Extract UPNL from position data"""
    try:
        # Get position metrics from position tracker
        metrics = self.position_tracker.get_position_metrics()
        upnl = metrics.get("unrealized_pnl", 0.0)
        
        # Alternative: Get from position data if available
        if hasattr(self, 'position') and self.position:
            upnl = getattr(self.position, 'unrealized_pnl', upnl)
            
        # Alternative: Calculate from mark price if needed
        if upnl == 0.0 and hasattr(self, 'ticker') and self.ticker:
            position_size = metrics.get("position", 0.0)
            entry_price = metrics.get("average_entry_price", 0.0)
            mark_price = self.ticker.mark_price
            
            if position_size != 0 and entry_price > 0 and mark_price > 0:
                upnl = position_size * (mark_price - entry_price)
        
        return float(upnl)
        
    except Exception as e:
        self.logger.error(f"Error extracting UPNL: {str(e)}")
        return 0.0

async def _check_take_profit_conditions(self) -> bool:
    """Check if take profit conditions are met"""
    if not RISK_LIMITS.get("take_profit_enabled", True):
        return False
        
    current_time = time.time()
    
    # Check cooldown
    if current_time < self.take_profit_cooldown_until:
        return False
        
    # Check interval
    if current_time - self.last_take_profit_check < RISK_LIMITS.get("take_profit_check_interval", 5):
        return False
        
    self.last_take_profit_check = current_time
    
    # Get current UPNL
    current_upnl = self._get_current_upnl()
    self.last_upnl_value = current_upnl
    
    # Check threshold
    take_profit_threshold = RISK_LIMITS.get("take_profit_threshold", 100.0)
    
    if current_upnl >= take_profit_threshold:
        self.logger.info(f"Take profit triggered: UPNL {current_upnl:.2f} >= threshold {take_profit_threshold:.2f}")
        return True
        
    return False
```

## Phase 3: Implement Position Flattening

### Step 3.1: Add Position Flattening Method
**File**: `thalex_py/Thalex_modular/avellaneda_quoter.py`
**Location**: Add new method after UPNL monitoring methods

```python
async def _execute_take_profit_flatten(self):
    """Execute take profit by flattening position"""
    try:
        # Get current position
        metrics = self.position_tracker.get_position_metrics()
        position_size = metrics.get("position", 0.0)
        
        if abs(position_size) < 0.0001:  # Already flat
            self.logger.info("Position already flat, take profit action not needed")
            return True
            
        self.logger.critical(f"EXECUTING TAKE PROFIT: Flattening position {position_size:.4f} at UPNL ${self.last_upnl_value:.2f}")
        
        # Cancel all existing quotes first
        await self.cancel_quotes("Take profit execution")
        await asyncio.sleep(1)  # Brief pause
        
        # Determine market order side and size
        if position_size > 0:
            # Long position - sell to flatten
            order_side = "sell"
            order_size = abs(position_size)
        else:
            # Short position - buy to flatten
            order_side = "buy"
            order_size = abs(position_size)
            
        # Place market order to flatten
        if hasattr(self, 'order_manager') and self.order_manager:
            market_order = {
                "instrument": self.instrument,
                "side": order_side,
                "size": order_size,
                "type": "market",
                "reduce_only": True  # Ensure we only reduce position
            }
            
            order_result = await self.order_manager.place_order(market_order)
            self.logger.info(f"Take profit market order placed: {order_result}")
            
        # Set cooldown
        self.take_profit_cooldown_until = time.time() + RISK_LIMITS.get("take_profit_cooldown", 30)
        self.take_profit_active = True
        
        return True
        
    except Exception as e:
        self.logger.error(f"Error executing take profit flatten: {str(e)}")
        return False
```

## Phase 4: Remove/Replace Existing Take Profit Logic

### Step 4.1: Identify and Replace Current Take Profit Logic
**File**: `thalex_py/Thalex_modular/avellaneda_quoter.py`
**Location**: Find and replace existing take profit implementation

```python
# REPLACE existing take profit logic with simple call:
# (Search for current take profit methods and replace with)

async def _legacy_take_profit_check(self):
    """Replaced with simplified UPNL-based logic"""
    # This method is deprecated - see _check_take_profit_conditions()
    pass
```

### Step 4.2: Integration in Main Trading Loop
**File**: `thalex_py/Thalex_modular/avellaneda_quoter.py`
**Location**: Add to main quote task loop or risk monitoring task

```python
# ADD this check in quote_task main loop (after existing risk checks):
if self.active_trading and not self.risk_recovery_mode:
    # Check take profit conditions
    if await self._check_take_profit_conditions():
        await self._execute_take_profit_flatten()
        # Brief pause after take profit
        await asyncio.sleep(2.0)
        continue
```

## Phase 5: Integration and Testing

### Step 5.1: Add Take Profit to Risk Monitoring
**File**: `thalex_py/Thalex_modular/avellaneda_quoter.py`
**Location**: In `_risk_monitoring_task` main loop

```python
# ADD to risk monitoring loop:
# Check take profit conditions (separate from other risk checks)
if self.active_trading and not self.risk_recovery_mode:
    if await self._check_take_profit_conditions():
        await self._execute_take_profit_flatten()
```

### Step 5.2: Add Logging and Monitoring
**File**: `thalex_py/Thalex_modular/avellaneda_quoter.py`
**Location**: Add logging in status methods

```python
# ADD to status logging methods:
def _log_take_profit_status(self):
    """Log current take profit status"""
    current_upnl = self._get_current_upnl()
    threshold = RISK_LIMITS.get("take_profit_threshold", 100.0)
    
    if current_upnl > 0:
        self.logger.info(f"Take Profit Monitor: UPNL ${current_upnl:.2f} / ${threshold:.2f} threshold")
    
    if self.take_profit_active:
        cooldown_remaining = max(0, self.take_profit_cooldown_until - time.time())
        self.logger.info(f"Take Profit: Cooldown active for {cooldown_remaining:.1f}s")
```

## Implementation Summary

### Files to Modify:
1. **`market_config.py`** - Add 5 new take profit parameters
2. **`avellaneda_quoter.py`** - Add 4 new methods, modify 2 existing methods, add 4 state variables

### Key Benefits:
- **Simplified Logic**: Direct UPNL-based thresholds replace complex conditions
- **Real-time Monitoring**: Continuous UPNL tracking with configurable intervals  
- **Immediate Action**: Market orders for instant position flattening
- **Configurable Thresholds**: Easy adjustment of profit targets
- **Integration Friendly**: Works with existing risk management

### Testing Strategy:
1. Verify UPNL extraction from position data
2. Test threshold detection accuracy
3. Confirm position flattening execution
4. Validate cooldown periods
5. Check integration with existing risk systems

## Execution Status:
- [ ] Phase 1: Add Take Profit Configuration
- [ ] Phase 2: Implement UPNL Monitoring  
- [ ] Phase 3: Implement Position Flattening
- [ ] Phase 4: Remove/Replace Existing Take Profit Logic
- [ ] Phase 5: Integration and Testing

**Next Steps**: Begin Phase 1 implementation with configuration updates. 