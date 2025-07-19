Here's a detailed prompt to guide an engineer LLM in implementing your hybrid Avellaneda-Stoikov (AS) and rescue trading strategy. This prompt outlines the necessary configuration, new methods, and modifications to existing files.

---

**Engineer LLM Prompt: Implementing Hybrid Avellaneda-Stoikov with Rescue Trades**

**Goal:** Implement a "rescue trade" logic in the existing market-making bot to actively manage underwater positions, while ensuring the core Avellaneda-Stoikov (AS) quoting strategy continues to operate in parallel.

**Current Problem:** The bot may accumulate inventory when the price moves adversely and currently lacks an aggressive, automated recovery mechanism for these "underwater" positions. Additionally, the existing `AvellanedaQuoter.place_quotes` function cancels *all* open orders, which would interfere with persistent rescue trades.

**Proposed Solution (Hybrid Strategy):**
When the bot holds an "underwater" position (e.g., a long position where the current market price is significantly below the average entry price), it should proactively place additional, aggressive limit orders ("rescue trades") on the adverse side (e.g., buy more if long) at the current best bid/ask. These rescue orders should be `post_only=True` to add liquidity.

The core AS quoting grid must continue to be managed independently; its orders should not be cancelled by, nor should they cancel, the rescue trades.

Once the averaged position (including original and rescue fills) becomes profitable by a small, configurable margin, the bot should attempt to flatten that specific inventory for a profit (`post_only=False` to ensure immediate fill), then reset the rescue state for that instrument and return to normal AS quoting for that portion of the inventory.

**Files to Modify:**
*   `thalex_py/Thalex_modular/avellaneda_quoter.py`
*   `thalex_py/Thalex_modular/components/order_manager.py`
*   `thalex_py/Thalex_modular/config/market_config.py`

**Detailed Implementation Steps:**

---

### **1. Configuration (`thalex_py/Thalex_modular/config/market_config.py`)**

Add a new `"rescue"` section under `TRADING_CONFIG`:

```python
# ... existing TRADING_CONFIG ...

"rescue": {
    "enabled": True,  # Boolean to enable/disable the rescue trade logic
    "threshold_pct": 0.003, # Decimal: Percentage price drop from average entry to trigger rescue (e.g., 0.003 for 0.3%)
    "profit_bps": 7.0,     # Float: Basis points profit target on the averaged position to exit rescue (e.g., 7.0 for 0.07%)
    "max_steps": 3,        # Integer: Maximum number of averaging-down steps allowed before halting rescue for the instrument
    "size_multiplier": 1.0, # Float: Multiplier for the base order size (TRADING_CONFIG["avellaneda"]["base_size"]) for each rescue order
    "min_interval_seconds": 5.0, # Float: Minimum time (seconds) between placing new rescue orders (to avoid spamming)
    "order_label": "RescueTrade", # String: Specific label for rescue orders to differentiate them from AS orders
    "exit_label": "RescueExit" # String: Specific label for rescue exit orders
},

# ... existing TRADING_CONFIG ...
```

---

### **2. `AvellanedaQuoter` Initialization (`thalex_py/Thalex_modular/avellaneda_quoter.py`)**

**A. Add new instance variables to `__slots__`:**

```python
# ... existing slots ...
        'rescue_mode_active', 'current_rescue_step', 'last_rescue_order_time',
        'active_rescue_orders', 'rescue_config', 'base_order_size' # Add base_order_size for convenience
```

**B. Initialize these variables in `__init__` (after `self.perp_name` and `self.futures_instrument_name` are potentially set):**

```python
# ... existing __init__ code ...

        # Rescue Trade State Management
        self.rescue_config = TRADING_CONFIG.get("rescue", {})
        # Initialize these as defaultdicts or explicitly for known instruments if they're always fixed.
        # For simplicity, assuming perp_name and futures_instrument_name are known and stable after setup.
        self.rescue_mode_active: Dict[str, bool] = {
            instr: False for instr in [self.perp_name, self.futures_instrument_name] if instr
        }
        self.current_rescue_step: Dict[str, int] = {
            instr: 0 for instr in [self.perp_name, self.futures_instrument_name] if instr
        }
        self.last_rescue_order_time: Dict[str, float] = {
            instr: 0.0 for instr in [self.perp_name, self.futures_instrument_name] if instr
        }
        # Stores active rescue orders by instrument, then by client_order_id
        self.active_rescue_orders: Dict[str, Dict[str, Order]] = {
            instr: {} for instr in [self.perp_name, self.futures_instrument_name] if instr
        }
        self.base_order_size = TRADING_CONFIG["avellaneda"].get("base_size", 0.01)

# ... rest of __init__ ...
```

---

### **3. Modify `OrderManager` (`thalex_py/Thalex_modular/components/order_manager.py`)**

**A. Add a method `cancel_orders_by_label(self, label: str)`:**

This method should iterate through `self.active_orders` (or `active_bids`, `active_asks`) and cancel only those orders whose `label` attribute matches the provided `label`.

**B. Add a method `cancel_all_but_labels(self, exclude_labels: List[str])`:**

This method should iterate through all active orders and cancel any order whose `label` *is NOT* in the `exclude_labels` list. This is crucial for keeping rescue orders alive.

```python
# Inside class OrderManager:
# ... existing methods ...

    async def cancel_orders_by_label(self, label: str):
        """Cancels all active orders with a specific label."""
        orders_to_cancel = [
            order_data for order_data in self.active_orders.values()
            if order_data.get("label") == label
        ]
        self.logger.info(f"Attempting to cancel {len(orders_to_cancel)} orders with label '{label}'.")
        for order_data in orders_to_cancel:
            try:
                # Assuming order_data has 'id' and 'instrument_name'
                await self.exchange_client.cancel(
                    order_id=order_data.get("id"),
                    instrument_name=order_data.get("instrument_name", self.default_instrument), # Use default if not found
                    id=random.randint(20000, 30000) # Unique client ID for cancellation request
                )
            except Exception as e:
                self.logger.error(f"Error cancelling order {order_data.get('id')} with label '{label}': {str(e)}")
        # Note: Actual removal from active_orders happens in handle_order_update when status changes to cancelled.

    async def cancel_all_but_labels(self, exclude_labels: List[str]):
        """Cancels all active orders except those with specified labels."""
        orders_to_cancel = [
            order_data for order_data in self.active_orders.values()
            if order_data.get("label") not in exclude_labels
        ]
        self.logger.info(f"Attempting to cancel {len(orders_to_cancel)} orders, excluding labels: {exclude_labels}.")
        for order_data in orders_to_cancel:
            try:
                # Assuming order_data has 'id' and 'instrument_name'
                await self.exchange_client.cancel(
                    order_id=order_data.get("id"),
                    instrument_name=order_data.get("instrument_name", self.default_instrument),
                    id=random.randint(20000, 30000)
                )
            except Exception as e:
                self.logger.error(f"Error cancelling order {order_data.get('id')} (label: {order_data.get('label')}): {str(e)}")
        # Note: Actual removal from active_orders happens in handle_order_update when status changes to cancelled.
```

---

### **4. Modify `cancel_quotes` in `AvellanedaQuoter` (`thalex_py/Thalex_modular/avellaneda_quoter.py`)**

Update the `cancel_quotes` method to use the new `cancel_all_but_labels` from `OrderManager`.

```python
# Inside class AvellanedaQuoter:
# ... existing methods ...

    async def cancel_quotes(self, reason: str):
        """Cancel all existing quotes, preserving rescue trades."""
        try:
            self.logger.info(f"Cancelling Avellaneda grid orders: {reason}")
            
            # Use the order manager to cancel all orders EXCEPT those labeled as rescue trades
            if hasattr(self, 'order_manager') and self.order_manager:
                await self.order_manager.cancel_all_but_labels(
                    exclude_labels=[self.rescue_config["order_label"], self.rescue_config["exit_label"]]
                )
                self.logger.info("Avellaneda grid orders cancelled successfully (rescue trades preserved).")
            else:
                self.logger.warning("Order manager not available, cannot cancel orders selectively.")
                
            return True
        except Exception as e:
            self.logger.error(f"Error cancelling Avellaneda quotes: {str(e)}")
            return False

# ... rest of the file ...
```

---

### **5. New Method: `_manage_rescue_trades(self, instrument_name: str)` in `AvellanedaQuoter` (`thalex_py/Thalex_modular/avellaneda_quoter.py`)**

This async method will contain the core rescue logic and should be called periodically.

```python
# Inside class AvellanedaQuoter:
# ... existing methods ...

    async def _manage_rescue_trades(self, instrument_name: str):
        """
        Manages the lifecycle of rescue trades for a specific instrument.
        Places new rescue orders if underwater, and flattens position if profitable.
        """
        if not self.rescue_config.get("enabled", False) or not self.active_trading:
            return

        if not self.ticker or self.ticker.mark_price <= 0: # Ensure valid price data
            if self.futures_instrument_name and self.futures_ticker and self.futures_ticker.mark_price > 0:
                # Use futures ticker if perp not available and we are managing futures rescue
                current_price = self.futures_ticker.mark_price
            else:
                self.logger.debug(f"Rescue trade manager: No valid ticker price for {instrument_name}.")
                return

        current_time = time.time()
        
        # Get position and PnL for the specific instrument
        position_metrics = self.portfolio_tracker.get_instrument_metrics(instrument_name)
        position = position_metrics.get("position", 0.0)
        average_entry = position_metrics.get("average_entry", 0.0)
        unrealized_pnl_usd = position_metrics.get("unrealized_pnl_usd", 0.0) # Using USD PnL for threshold
        
        # Determine if we are "underwater"
        is_underwater = False
        if abs(position) > ZERO_THRESHOLD and average_entry > 0 and self.ticker and self.ticker.mark_price > 0:
            # Calculate PnL based on mark price, not best bid/ask, for consistent check
            current_mark_price = self.ticker.mark_price if instrument_name == self.perp_name else (self.futures_ticker.mark_price if instrument_name == self.futures_instrument_name else 0.0)
            if current_mark_price <= 0: return # Cannot proceed without valid price

            # Calculate the percentage drop/rise for the current position from avg entry
            # For long position (position > 0), price drops means loss
            # For short position (position < 0), price rises means loss
            
            # Simple threshold check based on percentage of position value
            pnl_threshold_usd = self.rescue_config["threshold_pct"] * abs(position) * current_mark_price
            if unrealized_pnl_usd < -pnl_threshold_usd:
                is_underwater = True

            self.logger.debug(f"Rescue for {instrument_name}: Pos={position:.4f}, AvgEntry={average_entry:.2f}, UPNL=${unrealized_pnl_usd:.2f}, Threshold=${pnl_threshold_usd:.2f}. Underwater: {is_underwater}")
        
        # --- Handle placing new rescue orders ---
        if is_underwater:
            if not self.rescue_mode_active.get(instrument_name, False):
                self.logger.warning(f"🚨 Entering rescue mode for {instrument_name}! Pos: {position:.4f}, UPNL: ${unrealized_pnl_usd:.2f}")
                self.rescue_mode_active[instrument_name] = True
                self.current_rescue_step[instrument_name] = 0 # Reset steps on entering

            if self.current_rescue_step.get(instrument_name, 0) < self.rescue_config["max_steps"]:
                if current_time - self.last_rescue_order_time.get(instrument_name, 0.0) >= self.rescue_config["min_interval_seconds"]:
                    
                    # Determine direction and price for averaging down
                    # If long (position > 0) and underwater (price dropped), buy more at best bid.
                    # If short (position < 0) and underwater (price rose), sell more at best ask.
                    
                    side_to_average = ""
                    rescue_price = 0.0
                    
                    if position > 0: # Long position, price dropped. Need to BUY more.
                        side_to_average = "buy"
                        if self.ticker and self.ticker.best_bid_price > 0:
                            rescue_price = self.ticker.best_bid_price # User wants at best bid
                        elif current_mark_price > 0: # Fallback
                            rescue_price = current_mark_price * 0.999 # Slightly below mark
                    elif position < 0: # Short position, price rose. Need to SELL more.
                        side_to_average = "sell"
                        if self.ticker and self.ticker.best_ask_price > 0:
                            rescue_price = self.ticker.best_ask_price # User wants at best ask
                        elif current_mark_price > 0: # Fallback
                            rescue_price = current_mark_price * 1.001 # Slightly above mark
                    
                    if rescue_price <= 0:
                        self.logger.warning(f"Could not determine valid rescue price for {instrument_name}. Skipping rescue order.")
                        return

                    rescue_amount = self.base_order_size * self.rescue_config["size_multiplier"]
                    if rescue_amount <= ZERO_THRESHOLD:
                        self.logger.warning(f"Calculated rescue amount for {instrument_name} is too small ({rescue_amount}). Skipping.")
                        return
                    
                    # Align price to tick size
                    if instrument_name == self.perp_name:
                        tick_size_for_instrument = self.tick_size
                    elif instrument_name == self.futures_instrument_name:
                        tick_size_for_instrument = self.futures_tick_size
                    else:
                        tick_size_for_instrument = 1.0 # Default fallback
                    
                    if tick_size_for_instrument > 0:
                        if side_to_average == "buy":
                            rescue_price = math.floor(rescue_price / tick_size_for_instrument) * tick_size_for_instrument
                        else: # sell
                            rescue_price = math.ceil(rescue_price / tick_size_for_instrument) * tick_size_for_instrument
                    
                    client_id = f"{self.rescue_config['order_label']}_{instrument_name}_{self.current_rescue_step[instrument_name]}_{int(current_time * 1000)}"
                    
                    self.logger.info(
                        f"Attempting rescue trade for {instrument_name}: "
                        f"{side_to_average.upper()} {rescue_amount:.4f} @ {rescue_price:.2f} "
                        f"(Step: {self.current_rescue_step[instrument_name] + 1}/{self.rescue_config['max_steps']})"
                    )
                    
                    try:
                        await self.order_manager.place_order(
                            instrument=instrument_name,
                            direction=side_to_average,
                            price=rescue_price,
                            amount=rescue_amount,
                            label=self.rescue_config["order_label"],
                            post_only=True, # User wants to post a limit order
                            client_id=client_id
                        )
                        self.active_rescue_orders[instrument_name][client_id] = Order(
                            id=0, # Will be updated on fill
                            price=rescue_price,
                            amount=rescue_amount,
                            status=OrderStatus.PENDING,
                            direction=side_to_average,
                            instrument_id=instrument_name,
                            client_id=client_id,
                            label=self.rescue_config["order_label"]
                        )
                        self.current_rescue_step[instrument_name] += 1
                        self.last_rescue_order_time[instrument_name] = current_time
                    except Exception as e:
                        self.logger.error(f"Error placing rescue order for {instrument_name}: {str(e)}")
                else:
                    self.logger.info(f"Rescue trade for {instrument_name} is on cooldown. Time left: {self.rescue_config['min_interval_seconds'] - (current_time - self.last_rescue_order_time.get(instrument_name, 0.0)):.1f}s")
            else:
                self.logger.warning(f"Max rescue steps ({self.rescue_config['max_steps']}) reached for {instrument_name}. Halting further rescue trades.")
                self.rescue_mode_active[instrument_name] = False # Exit rescue mode if max steps reached
                self.current_rescue_step[instrument_name] = 0 # Reset for future entry if position becomes viable

        # --- Handle exiting rescue mode (flattening position) ---
        # Only check if currently in rescue mode AND have a position
        if self.rescue_mode_active.get(instrument_name, False) and abs(position) > ZERO_THRESHOLD:
            # Calculate profit target in USD
            profit_target_usd = (self.rescue_config["profit_bps"] / 10000) * abs(position) * (average_entry if average_entry > 0 else (self.ticker.mark_price if self.ticker else 0.0))
            
            self.logger.debug(f"Rescue Exit Check for {instrument_name}: UPNL=${unrealized_pnl_usd:.2f}, Target=${profit_target_usd:.2f}. Profitable? {unrealized_pnl_usd >= profit_target_usd}")

            if unrealized_pnl_usd >= profit_target_usd:
                self.logger.warning(
                    f"🎉 EXITING RESCUE MODE for {instrument_name}! "
                    f"Current UPNL: ${unrealized_pnl_usd:.2f} >= Target: ${profit_target_usd:.2f}"
                )
                
                # Determine direction and price to flatten
                side_to_flatten = ""
                exit_price = 0.0
                
                if position > 0: # Long position, sell to flatten
                    side_to_flatten = "sell"
                    if self.ticker and self.ticker.best_ask_price > 0:
                        exit_price = self.ticker.best_ask_price * (1 - 0.0001) # Slightly aggressive to ensure fill
                    elif current_mark_price > 0:
                        exit_price = current_mark_price * (1 - 0.0002) # Fallback, slightly below mark
                elif position < 0: # Short position, buy to flatten
                    side_to_flatten = "buy"
                    if self.ticker and self.ticker.best_bid_price > 0:
                        exit_price = self.ticker.best_bid_price * (1 + 0.0001) # Slightly aggressive to ensure fill
                    elif current_mark_price > 0:
                        exit_price = current_mark_price * (1 + 0.0002) # Fallback, slightly above mark

                if exit_price <= 0:
                    self.logger.warning(f"Could not determine valid exit price for {instrument_name}. Skipping rescue exit.")
                    return

                # Align price to tick size
                if tick_size_for_instrument > 0:
                    if side_to_flatten == "buy":
                        exit_price = math.ceil(exit_price / tick_size_for_instrument) * tick_size_for_instrument
                    else: # sell
                        exit_price = math.floor(exit_price / tick_size_for_instrument) * tick_size_for_instrument

                # First, cancel any pending rescue orders for this instrument
                if self.active_rescue_orders.get(instrument_name):
                    for client_id, order_obj in list(self.active_rescue_orders[instrument_name].items()):
                        self.logger.info(f"Cancelling pending rescue order {order_obj.id} ({client_id}) for {instrument_name}.")
                        try:
                            await self.order_manager.cancel_order(order_obj.id, instrument_name, client_id) # Assuming cancel_order exists
                        except Exception as e:
                            self.logger.error(f"Error cancelling pending rescue order {client_id}: {str(e)}")
                
                self.logger.warning(
                    f"Placing rescue exit order for {instrument_name}: "
                    f"{side_to_flatten.upper()} {abs(position):.4f} @ {exit_price:.2f}"
                )
                
                try:
                    await self.order_manager.place_order(
                        instrument=instrument_name,
                        direction=side_to_flatten,
                        price=exit_price,
                        amount=abs(position),
                        label=self.rescue_config["exit_label"],
                        post_only=False, # Allow taker to ensure immediate fill
                        client_id=f"{self.rescue_config['exit_label']}_{instrument_name}_{int(current_time * 1000)}"
                    )
                except Exception as e:
                    self.logger.error(f"Error placing rescue exit order for {instrument_name}: {str(e)}")

                # Reset rescue state for this instrument
                self.rescue_mode_active[instrument_name] = False
                self.current_rescue_step[instrument_name] = 0
                self.active_rescue_orders[instrument_name].clear()
                self.logger.info(f"Rescue mode exited for {instrument_name}.")

        elif self.rescue_mode_active.get(instrument_name, False) and abs(position) < ZERO_THRESHOLD:
            # If in rescue mode but position is now flat (e.g. manually closed or filled by AS), reset state
            self.logger.info(f"Position for {instrument_name} is flat, exiting rescue mode.")
            self.rescue_mode_active[instrument_name] = False
            self.current_rescue_step[instrument_name] = 0
            # Cancel any remaining pending rescue orders
            if self.active_rescue_orders.get(instrument_name):
                for client_id, order_obj in list(self.active_rescue_orders[instrument_name].items()):
                    self.logger.info(f"Cancelling pending rescue order {order_obj.id} ({client_id}) for {instrument_name} due to position flat.")
                    try:
                        await self.order_manager.cancel_order(order_obj.id, instrument_name, client_id)
                    except Exception as e:
                        self.logger.error(f"Error cancelling pending rescue order {client_id}: {str(e)}")
            self.active_rescue_orders[instrument_name].clear()

# ... rest of the file ...
```

---

### **6. Integrate `_manage_rescue_trades` Call in `AvellanedaQuoter`'s `quote_task`**

Call this new method for the perpetual and futures instruments within the main `quote_task` loop. This ensures the rescue logic is checked at every quoting cycle.

```python
# Inside class AvellanedaQuoter:
# ... existing methods ...

    async def quote_task(self):
        """Optimized quote task with reduced latency"""
        self.logger.info("Optimized quote task started - waiting for updates")
        
        if not self.active_trading:
            self.logger.critical("Quote task starting, but trading is already halted due to a prior risk breach. Will not proceed.")
            return # Exit task if trading already halted at start

        # Pre-allocate frequently used variables
        last_quote_time = 0
        min_quote_interval = 0.5  # Reduced from 1.0 for faster response
        force_quote_interval = 15.0  # Reduced from 30.0
        
        # Performance counters
        quote_counter = 0
        
        # Wait for initial setup to complete
        self.logger.info("Waiting for setup to complete before starting quote generation")
        await self.setup_complete.wait()
        
        while True:
            try:
                quote_counter += 1
                
                # Log status less frequently
                if quote_counter % 1200 == 0:  # Every ~10 minutes at 0.5s intervals
                    self.logger.info(f"Quote task alive - processed {quote_counter} cycles")
                
                if not self.active_trading:
                    if self.risk_recovery_mode:
                        # In recovery mode - check for recovery instead of exiting
                        if time.time() - self.last_recovery_check > RISK_LIMITS.get("recovery_check_interval", 30):
                            self.last_recovery_check = time.time()
                            if await self._check_risk_recovery():
                                await self._initiate_gradual_recovery()
                            else:
                                self.logger.info("Still in recovery mode - waiting for risk conditions to improve")
                        
                        # Sleep and continue loop instead of breaking
                        await asyncio.sleep(5.0)
                        continue
                    else:
                        self.logger.critical("Trading HALTED. Quote task will cease further operations.")
                        break
                
                # --- NEW: Call rescue trade management for relevant instruments ---
                if self.perp_name:
                    await self._manage_rescue_trades(self.perp_name)
                if self.futures_instrument_name:
                    await self._manage_rescue_trades(self.futures_instrument_name)
                # --- END NEW ---

                if not self.ticker:
                    self.logger.warning("No ticker data received yet")
                else:
                    self.logger.info(f"Current ticker: mark_price={self.ticker.mark_price}")
                
                # Wait for price updates with timeout
                try:
                    async with self.quote_cv:
                        # Wait with timeout to ensure we don't deadlock
                        await asyncio.wait_for(
                            self.quote_cv.wait_for(lambda: self.condition_met),
                            timeout=10.0  # Don't wait indefinitely
                        )
                        self.condition_met = False
                        self.logger.info("Quote task received notification")
                except asyncio.TimeoutError:
                    # If we timeout, still proceed with quote update if possible
                    self.logger.warning("Quote task timed out waiting for notification, proceeding anyway")
                
                current_time = time.time()
                
                # Force periodic quote updates regardless of ticker changes
                force_quote = (current_time - last_quote_time >= force_quote_interval)
                if force_quote:
                    self.logger.info(f"Forcing quote update after {current_time - last_quote_time:.1f}s since last update")
                
                # Enforce minimum interval between quote updates
                if not force_quote and current_time - last_quote_time < min_quote_interval:
                    self.logger.info(f"Quote update too frequent - waiting {min_quote_interval}s between updates")
                    await asyncio.sleep(0.5)
                    continue
                    
                # Check if we're in cooldown
                if self.cooldown_active and current_time < self.cooldown_until:
                    self.logger.info(f"In cooldown period - {(self.cooldown_until - current_time):.1f}s remaining")
                    await asyncio.sleep(0.5)
                    continue
                
                # Skip if no ticker data
                if not self.ticker:
                    self.logger.warning("Cannot quote - no ticker data available")
                    await asyncio.sleep(1.0)  # Shorter sleep to check again soon
                    continue
                
                # Generate default quotes with fallback values if needed
                if force_quote and (not hasattr(self, 'market_data') or len(self.market_data.prices) < 10):
                    self.logger.warning("Insufficient market data, using default values for quote generation")
                    # Create basic market conditions with default values
                    market_conditions = {
                        "volatility": TRADING_CONFIG["avellaneda"].get("fixed_volatility", 0.01),
                        "market_impact": 0.0
                    }
                    
                    # Create basic ticker data
                    ticker_data = {
                        "mid_price": self.ticker.mark_price
                    }
                    
                    # Generate quotes with default parameters
                    self.logger.info("Generating quotes with default parameters")
                    await self.update_quotes(ticker_data, market_conditions)
                    last_quote_time = current_time
                    continue
                
                # Check if we have enough price history
                min_samples = TRADING_CONFIG["volatility"].get("min_samples", 20)
                if not self.price_history_full and self.price_history_idx < min_samples:
                    self.logger.info(f"Need more price history before quoting: {self.price_history_idx}/{min_samples}")
                    continue
                
                # Get market conditions
                market_conditions = self.get_market_conditions()
                
                # Get ticker data
                ticker_data = {
                    "mid_price": (self.ticker.best_bid_price + self.ticker.best_ask_price) / 2
                    if hasattr(self.ticker, 'best_bid_price') and hasattr(self.ticker, 'best_ask_price')
                    else self.ticker.mark_price
                }
                
                # Update quotes
                with self.performance_tracer.trace("quote_update"):
                    try:
                        await self.update_quotes(ticker_data, market_conditions)
                    except Exception as e:
                        if "Ticker.__init__() got an unexpected keyword argument 'timestamp'" in str(e):
                            self.logger.error(f"Error generating quotes with market maker: {str(e)}")
                            # Create a temporary, valid ticker without the timestamp parameter
                            # that can be used by the market maker
                            if hasattr(self, 'ticker') and self.ticker:
                                ticker_dict = {
                                    "mark_price": self.ticker.mark_price,
                                    "best_bid_price": getattr(self.ticker, 'best_bid_price', self.ticker.mark_price * 0.999),
                                    "best_ask_price": getattr(self.ticker, 'best_ask_price', self.ticker.mark_price * 1.001),
                                    "mark_timestamp": time.time(),
                                    "index": getattr(self.ticker, 'index', self.ticker.mark_price)
                                }
                                # Skip timestamp parameter when creating the ticker
                                market_conditions["ticker"] = ticker_dict
                                # Try again with the fixed ticker data
                                await self.update_quotes(ticker_data, market_conditions)
                            else:
                                self.logger.error("Unable to create valid ticker for quote generation")
                        else:
                            # Re-raise other exceptions
                            raise
                
                last_quote_time = current_time
                
                # Add a delay after successfully updating quotes to prevent excessive updates
                await asyncio.sleep(0.5)
                    
            except asyncio.CancelledError:
                self.logger.info("Quote task cancelled")
                break
            except Exception as e:
                self.logger.error(f"Error in quote task: {str(e)}", exc_info=True)
                await asyncio.sleep(0.1)  # Shorter error recovery time

# ... rest of the file ...
```

---

### **7. Modify `handle_order_update` in `AvellanedaQuoter` (`thalex_py/Thalex_modular/avellaneda_quoter.py`)**

When an order with the rescue trade label (`"RescueTrade"`) or exit label (`"RescueExit"`) is filled or cancelled, remove it from the `self.active_rescue_orders` dictionary.

```python
# Inside class AvellanedaQuoter:
# ... existing methods ...

    async def handle_order_update(self, order_data: Dict):
        """Process order updates and trigger quote updates if necessary"""
        try:
            # Extract order info
            order = self.create_order_from_data(order_data)
            
            # Update order manager with the new order info
            if hasattr(self, 'order_manager') and self.order_manager is not None:
                await self.order_manager.update_order(order)
            
            # Handle fills
            if order_data.get("status") == "filled":
                # HFT-focused fill validation
                if order.direction is None:
                    return  # Skip invalid fills for HFT performance
                
                is_buy_flag = order.direction.lower() == "buy"
                significant_fill_threshold = TRADING_CONFIG["avellaneda"].get("significant_fill_threshold", 0.1)
                
                # Check if this is a trigger order fill (existing logic)
                order_label = order_data.get("label", "")
                client_id = order_data.get("client_id", "")
                is_trigger_order = (order_label == "TakeProfitTrigger" or 
                                  (client_id and client_id.startswith("trigger_")))
                
                # --- NEW: Handle Rescue Trade fills ---
                if order_label == self.rescue_config["order_label"] or order_label == self.rescue_config["exit_label"]:
                    self.logger.info(
                        f"RESCUE ORDER FILLED: {order.direction.upper()} {order.amount:.4f} @ {order.price:.2f} "
                        f"(Instrument: {order.instrument_id}, Label: {order_label})"
                    )
                    # Remove from active rescue orders tracking
                    if order.instrument_id in self.active_rescue_orders and order.client_id in self.active_rescue_orders[order.instrument_id]:
                        del self.active_rescue_orders[order.instrument_id][order.client_id]
                        self.logger.debug(f"Removed rescue order {order.client_id} from active tracking.")
                # --- END NEW ---

                if is_trigger_order:
                    # Handle trigger order fill
                    self.logger.info(
                        f"TRIGGER ORDER FILLED: {order.direction.upper()} {order.amount:.4f} @ {order.price:.2f} "
                        f"(order_id: {order.id}, client_id: {client_id})"
                    )
                    
                    # Remove from active trigger orders tracking
                    if hasattr(self, 'active_trigger_order_ids') and client_id in self.active_trigger_order_ids:
                        self.active_trigger_order_ids.remove(client_id)
                    
                    # Notify market maker about trigger order fill
                    if hasattr(self.market_maker, 'on_trigger_order_filled'):
                        self.market_maker.on_trigger_order_filled(str(order.id), order.price, order.amount)
                else:
                    # Handle normal grid order fill
                    self.logger.info(
                        f"GRID ORDER FILLED: {order.direction.upper()} {order.amount:.4f} @ {order.price:.2f} "
                        f"(order_id: {order.id})"
                    )
                
                # Update PositionTracker (for both trigger, grid, and rescue orders)
                if self.position_tracker:
                    try:
                        fill_timestamp_raw = order_data.get("timestamp", order_data.get("create_time", time.time()))
                        if isinstance(fill_timestamp_raw, (int, float)):
                            fill_time = datetime.fromtimestamp(fill_timestamp_raw, tz=timezone.utc)
                        elif isinstance(fill_timestamp_raw, str):
                            try:
                                fill_time = datetime.fromisoformat(fill_timestamp_raw.replace("Z", "+00:00"))
                            except ValueError:
                                fill_time = datetime.now(timezone.utc)
                        else:
                            fill_time = datetime.now(timezone.utc)

                        fill_object = Fill(
                            order_id=str(order.id),
                            fill_price=order.price,
                            fill_size=order.amount,
                            fill_time=fill_time,
                            side=order.direction.lower(),
                            is_maker=order_data.get("is_maker", True)
                        )
                        self.position_tracker.update_on_fill(fill_object)
                    except Exception:
                        pass  # Silent fail for HFT performance

                # CRITICAL FIX: Update PortfolioTracker for take profit logic (for all fills)
                if self.portfolio_tracker:
                    try:
                        # Extract instrument from order data
                        fill_instrument = getattr(order, 'instrument_id', None) or order_data.get("instrument_name", self.perp_name)
                        
                        # Ensure instrument is registered
                        if fill_instrument not in self.portfolio_tracker.instrument_trackers:
                            self.portfolio_tracker.register_instrument(fill_instrument)
                        
                        # Create fill object for portfolio tracker
                        fill_timestamp_raw = order_data.get("timestamp", order_data.get("create_time", time.time()))
                        if isinstance(fill_timestamp_raw, (int, float)):
                            fill_time = datetime.fromtimestamp(fill_timestamp_raw, tz=timezone.utc)
                        elif isinstance(fill_timestamp_raw, str):
                            try:
                                fill_time = datetime.fromisoformat(fill_timestamp_raw.replace("Z", "+00:00"))
                            except ValueError:
                                fill_time = datetime.now(timezone.utc)
                        else:
                            fill_time = datetime.now(timezone.utc)

                        portfolio_fill = Fill(
                            order_id=str(order.id),
                            fill_price=order.price,
                            fill_size=order.amount,
                            fill_time=fill_time,
                            side=order.direction.lower(),
                            is_maker=order_data.get("is_maker", True)
                        )
                        
                        # Update portfolio tracker with fill for the specific instrument
                        self.portfolio_tracker.update_instrument_fill(fill_instrument, portfolio_fill)
                        
                        self.logger.info(f"Updated portfolio tracker: {fill_instrument} {order.direction.upper()} {order.amount:.4f} @ {order.price:.2f}")
                        
                    except Exception as e:
                        self.logger.error(f"Error updating portfolio tracker on fill: {str(e)}")
                        pass  # Continue execution
                
                # Update market maker with fill information (for all orders)
                fill_instrument = getattr(order, 'instrument_id', None) or order_data.get("instrument_name", self.perp_name)
                
                if hasattr(self.market_maker, '_handle_trigger_order_on_fill'):
                    old_position = self.position_tracker.get_position_metrics().get("position", 0.0)
                    old_entry_price = self.position_tracker.get_position_metrics().get("entry_price", 0.0)
                    
                    self.market_maker.on_order_filled(str(order.id), order.price, order.amount, is_buy_flag)
                    self.market_maker._handle_trigger_order_on_fill(
                        order.price, order.amount, is_buy_flag, old_position, old_entry_price, fill_instrument
                    )
                else:
                    self.market_maker.on_order_filled(str(order.id), order.price, order.amount, is_buy_flag)
                
                # Sync position data after fill to ensure consistency
                self._sync_position_data()
                
                # Update risk manager with position information
                if hasattr(self.risk_manager, 'update_position_fill'):
                    await self.risk_manager.update_position_fill(order.direction, order.price, order.amount, time.time())
                
                # Force quote update after fills
                if self.ticker:
                    market_state = self.market_data.get_market_state()
                    market_conditions = {
                        "volatility": market_state.get("yz_volatility") or market_state.get("volatility") or TRADING_CONFIG["avellaneda"]["fixed_volatility"],
                        "market_impact": market_state.get("market_impact", 0.0),
                        "fill_impact": 0.5 if order.amount >= significant_fill_threshold else 0.2
                    }
                    
                    if order.amount >= significant_fill_threshold:
                        asyncio.create_task(self.update_quotes(None, market_conditions))
                    else:
                        async with self.quote_cv:
                            self.condition_met = True
                            self.quote_cv.notify()
                
        except Exception as e:
            self.logger.error(f"Error handling order update: {str(e)}", exc_info=True)

# ... rest of the file ...
```

---

This prompt provides a comprehensive plan for implementing the hybrid strategy, including new configuration, modifications to `OrderManager` for selective cancellation, new rescue trade management logic in `AvellanedaQuoter`, and integration points. 