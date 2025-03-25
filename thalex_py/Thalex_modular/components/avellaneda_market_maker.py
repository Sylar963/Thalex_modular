import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
import time
from collections import deque
from datetime import datetime
import math
import random

from thalex_py.Thalex_modular.config.market_config import TRADING_PARAMS, ORDERBOOK_CONFIG, TECHNICAL_PARAMS, RISK_LIMITS, TRADING_CONFIG
from thalex_py.Thalex_modular.models.data_models import Quote, Ticker
from thalex_py.Thalex_modular.models.position_tracker import PositionTracker, Fill

class AvellanedaMarketMaker:
    """
    Simplified implementation of the Avellaneda-Stoikov market making model.
    """
    
    def __init__(self):
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        
        # Avellaneda-Stoikov parameters
        self.gamma = TRADING_CONFIG["avellaneda"]["gamma"]  # Risk aversion
        self.inventory_weight = TRADING_CONFIG["avellaneda"]["inventory_weight"]  # Inventory skew
        self.position_fade_time = TRADING_CONFIG["avellaneda"]["position_fade_time"]  # Time to fade position
        self.order_flow_intensity = TRADING_CONFIG["avellaneda"]["order_flow_intensity"]  # Order flow intensity
        
        # Add critical missing parameters
        self.position_limit = TRADING_CONFIG["avellaneda"]["position_limit"]  # Position limit
        self.instrument = ""  # Instrument name must be set externally
        self.base_spread_factor = 1.0  # Base spread multiplier
        self.market_impact_factor = 0.5  # Market impact multiplier
        self.inventory_factor = 0.5  # Inventory adjustment factor
        self.volatility_multiplier = 0.2  # Volatility multiplier
        self.market_state = "normal"  # Market state: normal, trending, ranging
        
        # Market making state
        self.position_size = 0.0
        self.entry_price = None
        self.last_quote_time = 0.0
        self.tick_size = None
        self.min_tick = 0.0
        self.last_position_check = 0
        self.last_quote_update = 0
        
        # VAMP calculation variables
        self.vwap = 0.0
        self.total_volume = 0.0
        self.volume_price_sum = 0.0
        self.market_buys_volume = 0.0
        self.market_sells_volume = 0.0
        self.aggressive_buys_sum = 0.0
        self.aggressive_sells_sum = 0.0
        
        # Market conditions
        self.volatility = TRADING_CONFIG["volatility"]["floor"]  # Initialize with floor
        self.market_impact = 0.0
        
        # Quote tracking
        self.active_quotes: Dict[str, Quote] = {}
        self.current_bid_quotes: List[Quote] = []
        self.current_ask_quotes: List[Quote] = []
        
        # Position tracking
        self.position_tracker = PositionTracker()
        
        self.logger.info(f"Avellaneda-Stoikov market maker initialized with gamma={self.gamma}, inventory_weight={self.inventory_weight}")

    def set_tick_size(self, tick_size: float):
        """Set the tick size for the instrument"""
        self.tick_size = tick_size
        self.min_tick = tick_size
        self.logger.info(f"Tick size set to {tick_size}")

    def update_position(self, size: float, price: float):
        """Update current position information"""
        old_position = self.position_size
        self.position_size = size
        self.entry_price = price
        self.logger.info(f"Position updated: {old_position} -> {size} @ {price}")

    def update_market_conditions(self, volatility: float, market_impact: float):
        """Update market conditions"""
        self.volatility = max(volatility, TRADING_CONFIG["volatility"]["floor"])
        self.market_impact = market_impact
        self.logger.debug(f"Market conditions updated: vol={self.volatility:.6f}, impact={market_impact:.6f}")
    
    def update_vamp(self, price: float, volume: float, is_buy: bool, is_aggressive: bool=False):
        """Update Volume-Adjusted Market Price calculations"""
        # Update VWAP (Volume-Weighted Average Price)
        self.volume_price_sum += price * volume
        self.total_volume += volume
        if self.total_volume > 0:
            self.vwap = self.volume_price_sum / self.total_volume
        
        # Update aggressive volume tracking
        if is_aggressive:
            if is_buy:
                self.market_buys_volume += volume
                self.aggressive_buys_sum += price * volume
            else:
                self.market_sells_volume += volume
                self.aggressive_sells_sum += price * volume
        
        # Calculate VAMP
        vamp = self.calculate_vamp()
        self.logger.debug(f"VAMP updated: {vamp:.2f}, VWAP: {self.vwap:.2f}")
        return vamp
    
    def calculate_vamp(self) -> float:
        """Calculate Volume-Adjusted Market Price"""
        # If no volume, use mid_price
        if self.total_volume == 0:
            return 0.0
            
        # Basic VWAP if no aggressive trades
        if self.market_buys_volume == 0 and self.market_sells_volume == 0:
            return self.vwap
            
        # Calculate buy/sell pressure
        buy_vwap = self.aggressive_buys_sum / self.market_buys_volume if self.market_buys_volume > 0 else 0
        sell_vwap = self.aggressive_sells_sum / self.market_sells_volume if self.market_sells_volume > 0 else 0
        
        # If only one side has volume, use that side
        if self.market_buys_volume == 0:
            return sell_vwap
        if self.market_sells_volume == 0:
            return buy_vwap
            
        # Calculate VAMP using both sides
        buy_ratio = self.market_buys_volume / (self.market_buys_volume + self.market_sells_volume)
        sell_ratio = 1 - buy_ratio
        
        # VAMP is weighted average of buy and sell VWAPs
        vamp = (buy_vwap * buy_ratio) + (sell_vwap * sell_ratio)
        
        return vamp

    def calculate_optimal_spread(self, market_impact: float = 0) -> float:
        """
        Calculate the optimal bid-ask spread using Avellaneda-Stoikov model with enhanced adaptations
        
        Args:
            market_impact: Market impact parameter (0-1) to adjust spread in volatile markets
        
        Returns:
            float: Optimal spread in USDC
        """
        try:
            # Start with a minimum spread baseline based on tick size
            min_spread_ticks = ORDERBOOK_CONFIG.get("min_spread", 5)
            min_spread = min_spread_ticks * self.tick_size
            
            # Get volatility with safety fallback, ensure minimum is reasonable
            volatility = max(self.volatility, 0.01)  # At least 1% volatility
            
            # Base Avellaneda-Stoikov spread calculation components
            # 1. Gamma/risk aversion component
            gamma_component = 1.0 + self.gamma
            
            # 2. Volatility-based component (more volatile = wider spread)
            volatility_term = volatility * self.volatility_multiplier
            volatility_component = volatility_term * math.sqrt(self.position_fade_time)
            
            # 3. Market impact adjustment (high impact = wider spread)
            # Scale market impact by both volatility and order flow intensity for more responsive spreads
            market_impact_component = market_impact * self.market_impact_factor * (1 + volatility)
            
            # 4. Position/inventory risk component (larger position = wider spread)
            inventory_risk = abs(self.position_size) / max(self.position_limit, 0.001)
            inventory_component = self.inventory_factor * inventory_risk * volatility_term
            
            # 5. Market state component (trending/ranging adjustment)
            market_state_factor = 1.0
            if self.market_state == "trending":
                # Wider spreads in trending markets to avoid getting run over
                market_state_factor = 1.5
            elif self.market_state == "ranging":
                # Tighter spreads in ranging markets for more fills
                market_state_factor = 0.85
            
            # Calculate raw optimal spread
            base_spread = self.base_spread_factor * min_spread
            optimal_spread = (
                base_spread * gamma_component +
                volatility_component +
                market_impact_component +
                inventory_component
            ) * market_state_factor
            
            # Ensure minimum spread
            final_spread = max(optimal_spread, min_spread)
            
            # Log spread calculation components for debugging
            if random.random() < 0.05:  # Log occasionally to avoid spam
                self.logger.info(
                    f"Spread calc: base={base_spread:.2f}, volatility={volatility_component:.2f}, "
                    f"impact={market_impact_component:.2f}, inventory={inventory_component:.2f}, "
                    f"market_state={market_state_factor:.2f}, final={final_spread:.2f}"
                )
                
            return final_spread
        except Exception as e:
            self.logger.error(f"Error calculating optimal spread: {str(e)}")
            # Return safe default spread on error
            return 5 * self.tick_size

    def calculate_skewed_prices(self, mid_price: float, spread: float) -> Tuple[float, float]:
        """Calculate skewed bid and ask prices based on inventory"""
        try:
            # Simple inventory skew based on position size and limits
            position_limit = TRADING_CONFIG["avellaneda"]["position_limit"]
            inventory_skew_factor = self.inventory_weight * 0.5
            inventory_skew = 0
            
            if position_limit > 0:
                inventory_skew = (self.position_size / position_limit) * spread * inventory_skew_factor
            
            # Calculate half spread
            half_spread = spread / 2
            
            # Apply skew to bid and ask
            bid_price = mid_price - half_spread - inventory_skew
            ask_price = mid_price + half_spread - inventory_skew
            
            # Round to tick size
            bid_price = self.round_to_tick(bid_price)
            ask_price = self.round_to_tick(ask_price)
            
            # Ensure minimum spread
            min_spread_ticks = ORDERBOOK_CONFIG["min_spread"]
            min_spread = min_spread_ticks * self.tick_size
            
            if ask_price - bid_price < min_spread:
                center = (ask_price + bid_price) / 2
                half_min = min_spread / 2
                bid_price = self.round_to_tick(center - half_min)
                ask_price = self.round_to_tick(center + half_min)
            
            self.logger.info(f"Skewed prices: bid={bid_price:.2f}, ask={ask_price:.2f}, skew={inventory_skew:.4f}")
            return bid_price, ask_price
            
        except Exception as e:
            self.logger.error(f"Error calculating skewed prices: {str(e)}")
            half_spread = spread / 2
            return self.round_to_tick(mid_price - half_spread), self.round_to_tick(mid_price + half_spread)

    def calculate_quote_sizes(self, reference_price: float) -> Tuple[float, float]:
        """
        Calculate optimal quote sizes based on position, risk and market factors
        
        Args:
            reference_price: Current reference price for size calculations
            
        Returns:
            Tuple of (bid_size, ask_size)
        """
        try:
            # Base size from config - represents target exposure per quote
            base_size = ORDERBOOK_CONFIG.get("base_order_size", 0.01)  # Default 0.01 BTC
            
            # Get position limit from config
            position_limit = self.position_limit
            if position_limit <= 0:
                position_limit = TRADING_CONFIG["avellaneda"].get("position_limit", 0.1)
                
            # Calculate position ratio (how full our position capacity is)
            position_ratio = min(1.0, abs(self.position_size) / max(position_limit, 0.001))
            
            # Calculate base bid and ask sizes
            bid_size = base_size
            ask_size = base_size
            
            # 1. Adjust based on position direction (asymmetric sizing)
            if self.position_size > 0:  # Long position - reduce bids, increase asks
                # Decrease bid size as position grows (down to 20% of base size at position limit)
                bid_size = bid_size * max(0.2, (1.0 - position_ratio * 0.8))
                
                # Increase ask size as position grows (up to 150% of base size at position limit)
                ask_size = ask_size * min(1.5, (1.0 + position_ratio * 0.5))
                
            elif self.position_size < 0:  # Short position - increase bids, reduce asks
                # Increase bid size as position grows (up to 150% of base size at position limit)
                bid_size = bid_size * min(1.5, (1.0 + position_ratio * 0.5))
                
                # Decrease ask size as position grows (down to 20% of base size at position limit)
                ask_size = ask_size * max(0.2, (1.0 - position_ratio * 0.8))
            
            # 2. Adjust for market volatility - smaller sizes in high volatility
            volatility_factor = max(0.5, 1.0 - (self.volatility * 5.0))  # Scale down sizes in volatile markets
            bid_size *= volatility_factor
            ask_size *= volatility_factor
            
            # 3. Adjust for market direction - more size on momentum side
            if hasattr(self, 'market_state'):
                if self.market_state == "up_trend":
                    # In uptrend, larger bids (following trend), smaller asks (protection)
                    bid_size *= 1.2
                    ask_size *= 0.9
                elif self.market_state == "down_trend":
                    # In downtrend, smaller bids (protection), larger asks (following trend)
                    bid_size *= 0.9
                    ask_size *= 1.2
            
            # 4. Risk checks - ensure we don't exceed position limits
            # Calculate remaining capacity
            buy_capacity = position_limit - self.position_size
            sell_capacity = position_limit + self.position_size
            
            # Cap sizes to available capacity
            bid_size = min(bid_size, max(0, buy_capacity))
            ask_size = min(ask_size, max(0, sell_capacity))
            
            # Ensure minimum viable size
            min_order_size = ORDERBOOK_CONFIG.get("min_order_size", 0.001)
            bid_size = max(min_order_size, bid_size) if bid_size > 0 else 0
            ask_size = max(min_order_size, ask_size) if ask_size > 0 else 0
            
            # Round to precision (assume BTC = 8 decimals)
            bid_size = round(bid_size, 8)
            ask_size = round(ask_size, 8)
            
            # Create sizes tuple for bid and ask
            return bid_size, ask_size
            
        except Exception as e:
            self.logger.error(f"Error calculating quote sizes: {str(e)}")
            # Return conservative default sizes
            return 0.001, 0.001
            
    def can_increase_position(self) -> bool:
        """Check if we can increase our position (buy more)"""
        return self.position_size < self.position_limit
        
    def can_decrease_position(self) -> bool:
        """Check if we can decrease our position (sell some)"""
        return self.position_size > -self.position_limit

    def generate_quotes(self, ticker: Ticker, market_conditions: Dict) -> Tuple[List[Quote], List[Quote]]:
        """
        Generate optimized quotes using the Avellaneda-Stoikov model
        
        Args:
            ticker: Current market ticker data
            market_conditions: Dictionary of current market conditions (volatility, trend, etc.)
            
        Returns:
            Tuple of (bid_quotes, ask_quotes)
        """
        bid_quotes = []
        ask_quotes = []
        
        try:
            # 1. Extract price data with fallbacks
            # Determine valid mark price (with safety checks)
            mark_price = getattr(ticker, "mark_price", None)
            if not mark_price or mark_price <= 0:
                self.logger.warning("Invalid mark price, falling back to mid price")
                mark_price = None
            
            # Get bid/ask prices with validation
            best_bid = getattr(ticker, "best_bid_price", None)
            best_ask = getattr(ticker, "best_ask_price", None)
            
            # Validate bid/ask prices
            if not best_bid or best_bid <= 0:
                self.logger.warning("Invalid best bid price")
                best_bid = None
                
            if not best_ask or best_ask <= 0:
                self.logger.warning("Invalid best ask price")
                best_ask = None
            
            # Calculate mid price (with fallbacks)
            mid_price = None
            if best_bid and best_ask and best_bid < best_ask:
                mid_price = (best_bid + best_ask) / 2
                self.logger.debug(f"Using orderbook mid price: {mid_price}")
            
            # Determine reference price - prioritize VAMP if available
            reference_price = None
            
            # Try to use VAMP (volume-adjusted mid price) if suitable
            vamp = None
            if hasattr(self, 'calculate_vamp'):
                vamp = self.calculate_vamp()
                if vamp and vamp > 0:
                    reference_price = vamp
                    self.logger.debug(f"Using VAMP as reference: {reference_price}")
            
            # Fallback to mid price
            if not reference_price and mid_price:
                reference_price = mid_price
                self.logger.debug(f"Using mid price as reference: {reference_price}")
            
            # Last resort: mark price
            if not reference_price and mark_price:
                reference_price = mark_price
                self.logger.debug(f"Using mark price as reference: {reference_price}")
                
            # If we still don't have a reference price, we can't generate quotes
            if not reference_price or reference_price <= 0:
                self.logger.error("Could not determine valid reference price for quote generation")
                return [], []
                
            # 2. Calculate optimal spread with market impact factor
            market_impact = market_conditions.get("market_impact", 0)
            spread = self.calculate_optimal_spread(market_impact)
            
            # 3. Calculate skewed prices based on inventory
            bid_price, ask_price = self.calculate_skewed_prices(reference_price, spread)
            
            # 4. Safety check: ensure bid < ask and prices are positive
            if bid_price >= ask_price or bid_price <= 0 or ask_price <= 0:
                self.logger.warning(
                    f"Invalid prices calculated: bid={bid_price}, ask={ask_price}, " 
                    f"ref={reference_price}, spread={spread}"
                )
                # Recalculate with safe defaults
                min_spread_ticks = ORDERBOOK_CONFIG.get("min_spread", 5)
                safe_spread = min_spread_ticks * self.tick_size
                bid_price = reference_price - (safe_spread / 2)
                ask_price = reference_price + (safe_spread / 2)
            
            # 5. Align prices to tick size
            bid_price = self.align_price_to_tick(bid_price)
            ask_price = self.align_price_to_tick(ask_price)
            
            # 6. Calculate quote sizes (with position limit awareness)
            bid_size, ask_size = self.calculate_quote_sizes(reference_price)
            
            # Ensure sizes aren't too small
            min_size = ORDERBOOK_CONFIG.get("min_order_size", 0.001)
            if bid_size < min_size or ask_size < min_size:
                self.logger.warning(f"Quote sizes too small: bid={bid_size}, ask={ask_size}, using min size {min_size}")
                bid_size = max(bid_size, min_size)
                ask_size = max(ask_size, min_size)
                
            # 7. Create quote objects
            timestamp = datetime.now().timestamp()
            
            # Create bid quote if we have capacity to buy
            if self.can_increase_position():
                # Always create at least one bid quote
                bid_quote = Quote(
                    instrument=self.instrument,
                    side="buy",
                    price=bid_price,
                    amount=bid_size,
                    timestamp=timestamp
                )
                bid_quotes.append(bid_quote)
            else:
                self.logger.info("Skipping bid quote: position limit reached")
                
            # Create ask quote if we have position to sell
            if self.can_decrease_position():
                # Always create at least one ask quote
                ask_quote = Quote(
                    instrument=self.instrument,
                    side="sell",
                    price=ask_price,
                    amount=ask_size,
                    timestamp=timestamp
                )
                ask_quotes.append(ask_quote)
            else:
                self.logger.info("Skipping ask quote: no position to sell")
                
            # Log generated quotes
            realized_spread = ask_price - bid_price if bid_quotes and ask_quotes else None
            spread_display = f"{realized_spread:.2f}" if realized_spread is not None else "N/A"
            self.logger.info(
                f"Generated quotes: bid={bid_price:.2f} x {bid_size:.4f}, "
                f"ask={ask_price:.2f} x {ask_size:.4f}, "
                f"ref={reference_price:.2f}, spread={spread_display}"
            )
            
            return bid_quotes, ask_quotes
            
        except Exception as e:
            self.logger.error(f"Error generating quotes: {str(e)}")
            return [], []

    def should_update_quotes(self, current_quotes: Tuple[List[Quote], List[Quote]], mid_price: float) -> bool:
        """Enhanced quote update decision logic with comprehensive checks"""
        try:
            # Store current quotes tuple for reference
            bid_quotes, ask_quotes = current_quotes
            
            # 1. Always update if no quotes exist
            if not bid_quotes or not ask_quotes:
                self.logger.info("Should update quotes: No current quotes present")
                return True
            
            # 2. Check for stale or invalid quotes
            current_time = time.time()
            quote_lifetime = ORDERBOOK_CONFIG.get("quote_lifetime", 30)
            time_elapsed = current_time - self.last_quote_time
            
            if time_elapsed > quote_lifetime:
                self.logger.info(f"Should update quotes: Quotes are stale ({time_elapsed:.1f}s > {quote_lifetime}s lifetime)")
                return True
                
            # 3. Check if quotes are too far from mid price (urgent update needed)
            # Calculate current spread
            if bid_quotes and ask_quotes:
                current_bid = bid_quotes[0].price
                current_ask = ask_quotes[0].price
                current_mid = (current_bid + current_ask) / 2
                current_spread = current_ask - current_bid
                
                # Calculate urgency based on quote distance from current mid
                quote_mid_distance = abs(current_mid - mid_price)
                quote_mid_distance_pct = quote_mid_distance / mid_price
                
                # Base threshold from original quoter
                amend_threshold = ORDERBOOK_CONFIG.get("amend_threshold", 25) * self.tick_size
                price_move_threshold = amend_threshold / mid_price
                
                # Check if price moved significantly (at least 0.1% or 10 basis points)
                if quote_mid_distance_pct > max(price_move_threshold, 0.001):
                    self.logger.info(
                        f"Should update quotes: Significant price movement detected - "
                        f"Move: {quote_mid_distance_pct:.4%}, Threshold: {max(price_move_threshold, 0.001):.4%}"
                    )
                    return True
                    
                # Check if current spread is invalid (negative or too small)
                min_spread_ticks = ORDERBOOK_CONFIG.get("min_spread", 3)
                min_allowed_spread = min_spread_ticks * self.tick_size
                if current_spread < min_allowed_spread:
                    self.logger.info(
                        f"Should update quotes: Current spread too small - "
                        f"Spread: {current_spread:.2f}, Min allowed: {min_allowed_spread:.2f}"
                    )
                    return True
                    
                # Check if bid is above or ask is below mid price (crossed market)
                if current_bid > mid_price or current_ask < mid_price:
                    self.logger.info(
                        f"Should update quotes: Quotes crossed current market mid - "
                        f"Bid: {current_bid:.2f}, Ask: {current_ask:.2f}, Mid: {mid_price:.2f}"
                    )
                    return True
            
            # 4. Check for position-based update (significant position needs more frequent updates)
            position_check_interval = 5  # Check every 5 seconds with significant position
            significant_position_threshold = 0.05
            
            if abs(self.position_size) > significant_position_threshold and current_time - self.last_position_check > position_check_interval:
                self.last_position_check = current_time
                self.logger.info(f"Should update quotes: Position check with significant position {self.position_size:.2f}")
                return True
                
            # 5. Check for forced regular updates (ensure quotes refresh periodically)
            min_update_interval = ORDERBOOK_CONFIG.get("min_quote_interval", 2.0)
            max_update_interval = min(60.0, quote_lifetime * 0.8)  # 80% of quote lifetime or 60s max
            
            # More frequent updates with larger position
            if abs(self.position_size) > 0:
                # Scale update frequency based on position size
                position_scale_factor = min(1.0, abs(self.position_size) / TRADING_CONFIG["avellaneda"]["position_limit"])
                # More frequent updates as position grows (interval between min and max)
                update_interval = max_update_interval - (max_update_interval - min_update_interval) * position_scale_factor
                
                if time_elapsed > update_interval:
                    self.logger.info(
                        f"Should update quotes: Position-scaled interval reached - "
                        f"Elapsed: {time_elapsed:.1f}s, Interval: {update_interval:.1f}s, Position: {self.position_size:.3f}"
                    )
                    return True
            # Default update interval for no position
            elif time_elapsed > max_update_interval:
                self.logger.info(f"Should update quotes: Regular update interval reached ({time_elapsed:.1f}s > {max_update_interval:.1f}s)")
                return True
            
            # Do not update if no triggers were hit
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking if quotes should update: {str(e)}")
            # Default to updating quotes on error (safer)
            return True

    def validate_quotes(self, bid_quotes: List[Quote], ask_quotes: List[Quote]) -> Tuple[List[Quote], List[Quote]]:
        """Validate quotes before sending to exchange"""
        try:
            if not bid_quotes or not ask_quotes:
                return bid_quotes, ask_quotes
                
            # Check bid-ask spread
            best_bid = bid_quotes[0].price
            best_ask = ask_quotes[0].price
            
            if best_bid >= best_ask:
                self.logger.warning(f"Invalid spread detected: bid {best_bid} >= ask {best_ask}")
                
                # Fix by adjusting to minimum spread
                mid_price = (best_bid + best_ask) / 2
                min_spread_ticks = max(3, ORDERBOOK_CONFIG["min_spread"])
                half_min_spread = (min_spread_ticks * self.tick_size) / 2
                
                new_bid = self.round_to_tick(mid_price - half_min_spread)
                new_ask = self.round_to_tick(mid_price + half_min_spread)
                
                # Update quotes
                for quote in bid_quotes:
                    quote.price = new_bid
                
                for quote in ask_quotes:
                    quote.price = new_ask
                
                self.logger.info(f"Fixed quotes: new bid={new_bid:.2f}, new ask={new_ask:.2f}")
            
            # Apply risk limits to sizes
            max_quote_size = RISK_LIMITS.get("max_quote_size", 1.0)
            for quote in bid_quotes + ask_quotes:
                if quote.amount > max_quote_size:
                    quote.amount = max_quote_size
            
            return bid_quotes, ask_quotes
            
        except Exception as e:
            self.logger.error(f"Error validating quotes: {str(e)}")
            return [], []

    def round_to_tick(self, value: float) -> float:
        """Round price to nearest tick size"""
        if self.tick_size <= 0:
            return value
        return round(value / self.tick_size) * self.tick_size

    def on_order_filled(self, order_id: str, fill_price: float, fill_size: float, is_buy: bool) -> None:
        """Handle order fill events"""
        try:
            # Create fill object
            fill = Fill(
                order_id=order_id,
                fill_price=fill_price,
                fill_size=fill_size,
                fill_time=datetime.now(),
                side="buy" if is_buy else "sell",
                is_maker=True  # Assume maker fill for now
            )
            
            # Update position tracker
            self.position_tracker.update_on_fill(fill)
            
            # Update market maker state
            self.position_size = self.position_tracker.current_position
            self.entry_price = self.position_tracker.average_entry_price
            
            # Update VAMP calculations
            self.update_vamp(fill_price, fill_size, is_buy, False)
            
            self.logger.info(
                f"Order filled: {fill_size} @ {fill_price} ({'BUY' if is_buy else 'SELL'})"
                f" - New position: {self.position_size:.4f}, Avg entry: {self.entry_price:.2f}"
            )
            
        except Exception as e:
            self.logger.error(f"Error handling order fill: {str(e)}")

    def get_position_metrics(self) -> Dict:
        """Get position and performance metrics"""
        return {
            'position': self.position_size,
            'entry_price': self.entry_price,
            'realized_pnl': self.position_tracker.realized_pnl,
            'unrealized_pnl': self.position_tracker.unrealized_pnl,
            'total_pnl': self.position_tracker.realized_pnl + self.position_tracker.unrealized_pnl,
            'vwap': self.vwap,
            'vamp': self.calculate_vamp()
        }

    def align_price_to_tick(self, price: float) -> float:
        """
        Align price to the instrument's tick size
        
        Args:
            price: The price to align
            
        Returns:
            float: Price aligned to the instrument's tick size
        """
        if not self.tick_size or self.tick_size <= 0:
            self.logger.warning("Invalid tick size, using default alignment")
            return round(price, 2)  # Default to 2 decimal places
            
        # Round to nearest tick
        aligned_price = self.tick_size * round(price / self.tick_size)
        
        # Ensure we have at least minimum price
        min_price = self.tick_size
        aligned_price = max(aligned_price, min_price)
        
        return aligned_price 

    def set_instrument(self, instrument: str):
        """Set the instrument name"""
        self.instrument = instrument
        self.logger.info(f"Instrument set to {instrument}") 