# Portfolio-Level UPNL Implementation Guide

## Problem Statement

The current `AvellanedaQuoter` uses a single `PositionTracker` for multiple instruments (BTC-PERP and BTC-27JUL25), causing UPNL calculations to overwrite each other. This prevents proper combined P&L recognition for take profit decisions.

**Current Issue**: If BTC-PERP has $130 UPNL and BTC-27JUL25 has $150 UPNL, the system only sees $150 (the last calculated), not the combined $280.

## Solution Overview

Replace the single `PositionTracker` with `PortfolioTracker` to properly aggregate P&L across multiple instruments.

---

## Step-by-Step Implementation

### Step 1: Update Imports in `avellaneda_quoter.py`

**File**: `thalex_py/Thalex_modular/avellaneda_quoter.py`

**Location**: Around line 47 (in the imports section)

**Change**:
```python
# BEFORE:
from thalex_py.Thalex_modular.models.position_tracker import PositionTracker, Fill

# AFTER:
from thalex_py.Thalex_modular.models.position_tracker import PositionTracker, PortfolioTracker, Fill
```

### Step 2: Replace PositionTracker with PortfolioTracker in __init__

**File**: `thalex_py/Thalex_modular/avellaneda_quoter.py`

**Location**: Line 248 (in `__init__` method)

**Change**:
```python
# BEFORE:
self.position_tracker = PositionTracker() # Initialize PositionTracker first

# AFTER:
self.portfolio_tracker = PortfolioTracker() # Initialize PortfolioTracker for multi-instrument support
# Keep backward compatibility reference
self.position_tracker = self.portfolio_tracker  # For existing code compatibility
```

### Step 3: Register Instruments in start() Method

**File**: `thalex_py/Thalex_modular/avellaneda_quoter.py`

**Location**: Around line 900 (in `start()` method, after instrument setup)

**Add after line where `self.perp_name` is set**:
```python
# Register instruments with portfolio tracker
if self.perp_name:
    self.portfolio_tracker.register_instrument(self.perp_name)
    self.logger.info(f"Registered perpetual instrument: {self.perp_name}")

if self.futures_instrument_name:
    self.portfolio_tracker.register_instrument(self.futures_instrument_name)
    self.logger.info(f"Registered futures instrument: {self.futures_instrument_name}")
```

### Step 4: Update _get_current_upnl() Method

**File**: `thalex_py/Thalex_modular/avellaneda_quoter.py`

**Location**: Lines 501-527

**Replace entire method**:
```python
def _get_current_upnl(self) -> float:
    """Extract combined UPNL from portfolio across all instruments"""
    try:
        # Get total portfolio P&L across all instruments
        total_portfolio_pnl = self.portfolio_tracker.get_total_pnl()
        
        # Log detailed breakdown for debugging
        if hasattr(self, '_log_counter') and self._log_counter % 100 == 0:
            portfolio_metrics = self.portfolio_tracker.get_portfolio_metrics()
            instrument_pnls = portfolio_metrics.get("instrument_pnls", {})
            self.logger.info(f"Portfolio UPNL breakdown: {instrument_pnls}, Total: {total_portfolio_pnl:.2f}")
        
        return float(total_portfolio_pnl)
        
    except Exception as e:
        self.logger.error(f"Error extracting portfolio UPNL: {str(e)}")
        # Fallback to single instrument if portfolio tracker fails
        try:
            if hasattr(self, 'position_tracker') and hasattr(self.position_tracker, 'get_position_metrics'):
                metrics = self.position_tracker.get_position_metrics()
                return float(metrics.get("unrealized_pnl", 0.0))
        except Exception as fallback_error:
            self.logger.error(f"Fallback UPNL calculation also failed: {str(fallback_error)}")
        return 0.0
```

### Step 5: Update Risk Monitoring Task

**File**: `thalex_py/Thalex_modular/avellaneda_quoter.py`

**Location**: Lines 700-750 (in `_risk_monitoring_task()` method)

**Replace the UPNL update section**:
```python
# BEFORE:
# Update unrealized PnL for Perpetual
if current_price is not None and self.position_tracker and self.perp_name:
    self.position_tracker.update_unrealized_pnl(current_price, self.perp_name)

# Get futures price and update its PnL
current_price_futures = None
if self.futures_ticker and hasattr(self.futures_ticker, 'mark_price') and self.futures_ticker.mark_price > 0:
    current_price_futures = self.futures_ticker.mark_price
elif self.market_data and self.futures_instrument_name and len(self.market_data.get_prices(self.futures_instrument_name)) > 0:
    current_price_futures = self.market_data.get_prices(self.futures_instrument_name)[-1]

if current_price_futures is not None and self.position_tracker and self.futures_instrument_name:
    self.position_tracker.update_unrealized_pnl(current_price_futures, self.futures_instrument_name)

# AFTER:
# Update mark prices for portfolio-level P&L calculation
if current_price is not None and self.perp_name:
    self.portfolio_tracker.update_mark_price(self.perp_name, current_price)

# Get futures price and update its mark price
current_price_futures = None
if self.futures_ticker and hasattr(self.futures_ticker, 'mark_price') and self.futures_ticker.mark_price > 0:
    current_price_futures = self.futures_ticker.mark_price
elif self.market_data and self.futures_instrument_name and len(self.market_data.get_prices(self.futures_instrument_name)) > 0:
    current_price_futures = self.market_data.get_prices(self.futures_instrument_name)[-1]

if current_price_futures is not None and self.futures_instrument_name:
    self.portfolio_tracker.update_mark_price(self.futures_instrument_name, current_price_futures)
```

### Step 6: Update Position Updates in handle_portfolio_update()

**File**: `thalex_py/Thalex_modular/avellaneda_quoter.py`

**Location**: Lines 2300-2350 (in `handle_portfolio_update()` method)

**Add after position tracking updates**:
```python
# Update portfolio tracker for this instrument
if instrument in [self.perp_name, self.futures_instrument_name]:
    try:
        # Ensure instrument is registered
        if instrument not in self.portfolio_tracker.instrument_trackers:
            self.portfolio_tracker.register_instrument(instrument)
        
        # Update position in portfolio tracker
        if avg_price > 0:
            self.portfolio_tracker.update_position(instrument, size, avg_price)
        else:
            # If no average price, use current mark price as estimate
            current_mark = None
            if instrument == self.perp_name and self.ticker:
                current_mark = self.ticker.mark_price
            elif instrument == self.futures_instrument_name and self.futures_ticker:
                current_mark = self.futures_ticker.mark_price
            
            if current_mark and current_mark > 0:
                self.portfolio_tracker.update_position(instrument, size, current_mark)
                self.logger.info(f"Updated portfolio position for {instrument} using mark price {current_mark:.2f}")
        
    except Exception as e:
        self.logger.error(f"Error updating portfolio tracker for {instrument}: {str(e)}")
```

### Step 7: Update Fill Processing in handle_order_update()

**File**: `thalex_py/Thalex_modular/avellaneda_quoter.py`

**Location**: Lines 2150-2200 (in `handle_order_update()` method, after fill processing)

**Add after existing position tracker update**:
```python
# Update portfolio tracker with fill
if hasattr(self, 'portfolio_tracker'):
    try:
        # Determine which instrument this fill belongs to
        fill_instrument = self.perp_name  # Default assumption
        
        # If we have order instrument info, use it
        if hasattr(order, 'instrument_id') and order.instrument_id:
            fill_instrument = order.instrument_id
        
        # Ensure instrument is registered
        if fill_instrument not in self.portfolio_tracker.instrument_trackers:
            self.portfolio_tracker.register_instrument(fill_instrument)
        
        # Calculate estimated fee for this fill
        notional_value = order.price * order.amount
        estimated_fee = self.portfolio_tracker.calculate_trade_fee(
            notional_value, 
            is_maker=order_data.get("is_maker", True)
        )
        
        # Update portfolio position with fee tracking
        position_change = order.amount if order.direction.lower() == "buy" else -order.amount
        self.portfolio_tracker.update_position(
            fill_instrument, 
            position_change, 
            order.price, 
            fee=estimated_fee
        )
        
        self.logger.info(f"Updated portfolio tracker: {fill_instrument} position change {position_change:.4f} @ {order.price:.2f}, fee: {estimated_fee:.4f}")
        
    except Exception as e:
        self.logger.error(f"Error updating portfolio tracker with fill: {str(e)}")
```

### Step 8: Update _close_both_positions() Method

**File**: `thalex_py/Thalex_modular/avellaneda_quoter.py`

**Location**: Lines 3349-3438 (in `_close_both_positions()` method)

**Replace position size retrieval**:
```python
# BEFORE:
current_perp_position = self.position_tracker.get_position_metrics().get("position", 0.0)
current_futures_position = self.position_tracker.get_position_metrics().get("position", 0.0)

# AFTER:
# Get positions from portfolio tracker
portfolio_positions = self.portfolio_tracker.positions
current_perp_position = portfolio_positions.get(self.perp_name, 0.0)
current_futures_position = portfolio_positions.get(self.futures_instrument_name, 0.0) if self.futures_instrument_name else 0.0
```

### Step 9: Update Status Logging

**File**: `thalex_py/Thalex_modular/avellaneda_quoter.py`

**Location**: Lines 2900-2950 (in `log_status_task()` method)

**Add portfolio metrics to status logging**:
```python
# Add after existing position metrics
portfolio_metrics = self.portfolio_tracker.get_portfolio_metrics()
total_portfolio_pnl = portfolio_metrics.get("total_pnl", 0.0)
net_portfolio_pnl = portfolio_metrics.get("net_pnl_after_fees", 0.0)
instrument_positions = portfolio_metrics.get("instrument_positions", {})

# Update the status log to include portfolio info
self.logger.info(
    f"Status: Pos(MM)={pos_mm_str} PnL(MM)=[R:{rpnl_mm_str} U:{upnl_mm_str}] | "
    f"Pos(PT)={pt_pos_str} AvgEntry(PT)={pt_avg_entry_str} PnL(PT)=[R:{pt_rpnl_str} U:{pt_upnl_str}] Vol(PT)={pt_total_vol_str} Fills(PT)={pt_fills} | "
    f"Portfolio: Total_PnL={total_portfolio_pnl:.2f} Net_PnL={net_portfolio_pnl:.2f} Positions={instrument_positions} | "
    f"Price={mark_price_str} | Orders={active_orders} | "
    f"HFT=[Pools:{len(self.order_pool.items)}/{len(self.quote_pool.items)}]"
)
```

### Step 10: Add Configuration for Portfolio Take Profit

**File**: `thalex_py/Thalex_modular/config/market_config.py`

**Add to RISK_LIMITS section**:
```python
# Portfolio-level take profit settings
"portfolio_take_profit_enabled": True,
"portfolio_take_profit_threshold": 200.0,  # Combined UPNL threshold across all instruments
"portfolio_take_profit_check_interval": 5,  # Check every 5 seconds
"portfolio_take_profit_cooldown": 30,  # 30 second cooldown after execution
"portfolio_fee_buffer": 1.1,  # 10% buffer for estimated closing fees
```

### Step 11: Update Take Profit Logic

**File**: `thalex_py/Thalex_modular/avellaneda_quoter.py`

**Location**: Lines 527-564 (in `_check_take_profit_conditions()` method)

**Add portfolio-aware logic**:
```python
async def _check_take_profit_conditions(self) -> bool:
    """Check if take profit conditions are met using portfolio-level P&L"""
    if not RISK_LIMITS.get("take_profit_enabled", True):
        return False
        
    current_time = time.time()
    
    # Check cooldown (prevents immediate retries)
    if current_time < self.take_profit_cooldown_until:
        return False
        
    # Check interval (prevents too frequent checks)
    if current_time - self.last_take_profit_check < RISK_LIMITS.get("take_profit_check_interval", 5):
        return False
        
    self.last_take_profit_check = current_time
    
    # Check if we have any meaningful positions
    portfolio_positions = self.portfolio_tracker.positions
    has_meaningful_position = any(abs(pos) > 0.0001 for pos in portfolio_positions.values())
    
    if not has_meaningful_position:
        return False
    
    # Get current portfolio UPNL (including fees)
    current_upnl = self._get_current_upnl()
    net_upnl_after_fees = self.portfolio_tracker.get_net_profit_after_all_fees()
    self.last_upnl_value = current_upnl
    
    # Check threshold - use net P&L after fees for more accurate take profit
    take_profit_threshold = RISK_LIMITS.get("portfolio_take_profit_threshold", 
                                           RISK_LIMITS.get("take_profit_threshold", 100.0))
    
    if net_upnl_after_fees >= take_profit_threshold:
        # Log detailed breakdown
        fee_breakdown = self.portfolio_tracker.get_detailed_fee_breakdown()
        self.logger.info(
            f"Portfolio take profit triggered: "
            f"Gross_UPNL={current_upnl:.2f}, Net_UPNL_after_fees={net_upnl_after_fees:.2f} >= threshold={take_profit_threshold:.2f}"
        )
        self.logger.info(f"Fee breakdown: {fee_breakdown}")
        return True
        
    return False
```

---

## Testing and Validation

### Step 12: Add Debug Logging

Add temporary debug logging to verify the fix works:

```python
# Add to _get_current_upnl() method for testing
def _get_current_upnl(self) -> float:
    """Extract combined UPNL from portfolio across all instruments"""
    try:
        total_portfolio_pnl = self.portfolio_tracker.get_total_pnl()
        
        # TEMPORARY DEBUG LOGGING - Remove after testing
        portfolio_metrics = self.portfolio_tracker.get_portfolio_metrics()
        instrument_pnls = portfolio_metrics.get("instrument_pnls", {})
        positions = portfolio_metrics.get("instrument_positions", {})
        
        self.logger.info(f"DEBUG UPNL: Positions={positions}, Individual_PnLs={instrument_pnls}, Total={total_portfolio_pnl:.2f}")
        
        return float(total_portfolio_pnl)
        
    except Exception as e:
        self.logger.error(f"Error extracting portfolio UPNL: {str(e)}")
        return 0.0
```

### Step 13: Test Scenarios

1. **Single Instrument Test**: Verify existing functionality still works
2. **Multi-Instrument Test**: 
   - Open positions on both BTC-PERP and BTC-27JUL25
   - Verify combined P&L calculation
   - Test take profit triggers at combined threshold
3. **Edge Cases**:
   - One instrument profitable, other at loss
   - Both instruments at loss
   - Position flips (long to short)

---

## Expected Results

After implementation:

1. **Combined P&L Recognition**: $130 (BTC-PERP) + $150 (BTC-27JUL25) = $280 total
2. **Accurate Take Profit**: Triggers when combined UPNL exceeds threshold
3. **Fee-Aware Decisions**: Uses net P&L after estimated closing fees
4. **Multi-Instrument Support**: Works for any combination of long/short positions
5. **Backward Compatibility**: Existing single-instrument code continues to work

---

## Rollback Plan

If issues arise, rollback by:

1. Reverting `__init__` method to use `PositionTracker()`
2. Reverting `_get_current_upnl()` to original implementation
3. Removing portfolio-specific updates in risk monitoring
4. Keeping the `PortfolioTracker` import for future use

---

## Performance Considerations

- `PortfolioTracker` uses locks for thread safety
- Minimal performance impact due to efficient design
- Consider reducing debug logging frequency in production
- Monitor memory usage with multiple instruments

---

## Future Enhancements

1. **Dynamic Instrument Registration**: Auto-register instruments from portfolio updates
2. **Advanced Fee Modeling**: More sophisticated fee estimation
3. **Risk-Adjusted Take Profit**: Consider correlation between instruments
4. **Portfolio Rebalancing**: Automatic position sizing across instruments 