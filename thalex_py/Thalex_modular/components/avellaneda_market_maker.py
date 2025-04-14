import numpy as np
from typing import Dict, List, Optional, Tuple
import time
from collections import deque
import math
import random
from datetime import datetime

from ..config.market_config import TRADING_CONFIG, RISK_LIMITS, ORDERBOOK_CONFIG
from ..models.data_models import Ticker, Quote
from ..models.position_tracker import PositionTracker, Fill
from ..thalex_logging import LoggerFactory
from ..ringbuffer.volume_candle_buffer import VolumeBasedCandleBuffer

class AvellanedaMarketMaker:
    """Avellaneda-Stoikov market making strategy implementation"""
    
    def __init__(self, exchange_client=None):
        # Initialize logger
        self.logger = LoggerFactory.configure_component_logger(
            "avellaneda_market_maker",
            log_file="market_maker.log",
            high_frequency=True
        )
        
        # Store the exchange client
        self.exchange_client = exchange_client
        
        # Trading parameters
        self.gamma = TRADING_CONFIG["avellaneda"]["gamma"]  # Risk aversion
        self.k_default = TRADING_CONFIG["avellaneda"]["kappa"]  # Inventory risk factor
        self.kappa = self.k_default
        
        # Position tracking
        self.position_tracker = PositionTracker()
        self.monetary_position = 0.0  # Monetary value of position (USD)
        
        # Market parameters
        self.volatility = TRADING_CONFIG["volatility"]["default"]
        self.tick_size = 0.0
        self.time_horizon = TRADING_CONFIG["avellaneda"]["time_horizon"]
        self.reservation_price_offset = 0.0
        
        # Performance tracking
        self.quote_count = 0
        self.quote_levels = TRADING_CONFIG["quoting"]["levels"]
        self.instrument = None
        
        # Strategy state
        self.last_mid_price = 0.0
        self.last_update_time = 0.0
        self.volatility_grid = []
        
        # VAMP (Volume Adjusted Market Pressure) tracking
        self.vamp_window = deque(maxlen=TRADING_CONFIG["vamp"]["window"])
        self.vamp_aggressive_window = deque(maxlen=TRADING_CONFIG["vamp"]["aggressive_window"])
        self.vamp_buy_volume = 0.0
        self.vamp_sell_volume = 0.0
        self.vamp_value = 0.0
        self.vamp_impact = 0.0
        self.vamp_impact_window = deque(maxlen=TRADING_CONFIG["vamp"]["impact_window"])
        
        # NEW: Volume candle buffer for predictive analysis
        self.volume_buffer = VolumeBasedCandleBuffer(
            volume_threshold=TRADING_CONFIG.get("volume_candle", {}).get("threshold", 1.0),
            max_candles=TRADING_CONFIG.get("volume_candle", {}).get("max_candles", 100),
            max_time_seconds=TRADING_CONFIG.get("volume_candle", {}).get("max_time_seconds", 300)
        )
        
        # NEW: Predictive state tracking
        self.predictive_adjustments = {
            "gamma_adjustment": 0.0,
            "kappa_adjustment": 0.0,
            "reservation_price_offset": 0.0,
            "trend_direction": 0,
            "volatility_adjustment": 0.0,
            "last_update_time": 0
        }
        
        self.logger.info("Avellaneda market maker initialized")
        
        # Avellaneda-Stoikov parameters
        self.inventory_weight = TRADING_CONFIG["avellaneda"]["inventory_weight"]  # Inventory skew
        self.position_fade_time = TRADING_CONFIG["avellaneda"]["position_fade_time"]  # Time to fade position
        self.order_flow_intensity = TRADING_CONFIG["avellaneda"]["order_flow_intensity"]  # Order flow intensity
        
        # Add critical missing parameters
        self.position_limit = TRADING_CONFIG["avellaneda"]["position_limit"]  # Position limit
        self.base_spread_factor = 1.0  # Base spread multiplier
        self.market_impact_factor = 0.5  # Market impact multiplier
        self.inventory_factor = 0.5  # Inventory adjustment factor
        self.volatility_multiplier = 0.2  # Volatility multiplier
        self.market_state = "normal"  # Market state: normal, trending, ranging
        
        # Market making state
        self.position_size = 0.0
        self.entry_price = None
        self.last_quote_time = 0.0
        self.min_tick = 0.0
        self.last_position_check = 0
        self.last_quote_update = 0
        
        # Enhanced position tracking
        self.position_start_time = 0  # Time when current position was opened
        self.unrealized_pnl = 0.0     # Current unrealized PnL
        self.position_value = 0.0     # Current position value in USD
        
        # VAMP calculation variables
        self.vwap = 0.0
        self.total_volume = 0.0
        self.volume_price_sum = 0.0
        self.market_buys_volume = 0.0
        self.market_sells_volume = 0.0
        self.aggressive_buys_sum = 0.0
        self.aggressive_sells_sum = 0.0
        
        # Market conditions
        self.market_impact = 0.0
        
        # Quote tracking
        self.active_quotes: Dict[str, Quote] = {}
        self.current_bid_quotes: List[Quote] = []
        self.current_ask_quotes: List[Quote] = []

        # Hedge manager integration
        self.hedge_manager = None
        self.use_hedging = TRADING_CONFIG.get("hedging", {}).get("enabled", False)
        if self.use_hedging:
            from .hedge import create_hedge_manager
            # Create hedge manager with default settings and the exchange client
            self.hedge_manager = create_hedge_manager(
                config_path=TRADING_CONFIG.get("hedging", {}).get("config_path"),
                exchange_client=self.exchange_client,
                strategy_type=TRADING_CONFIG.get("hedging", {}).get("strategy", "notional")
            )
            # Start the hedge manager's background thread for rebalancing
            self.hedge_manager.start()
            self.logger.info("Hedge manager initialized and started")
        else:
            self.logger.info("Hedging is disabled in configuration")

    def set_tick_size(self, tick_size: float):
        """Set the tick size for the instrument"""
        self.tick_size = tick_size
        self.min_tick = tick_size
        self.logger.info(f"Tick size set to {tick_size}")

    def update_position(self, size: float, price: float):
        """
        Update current position information with enhanced metrics tracking
        
        Args:
            size: Current position size
            price: Current position price/mark price
        """
        old_position = self.position_size
        old_position_value = getattr(self, 'position_value', 0.0)
        
        # Track position start time for new positions
        if self.position_size == 0 and size != 0:
            self.position_start_time = time.time()
            self.logger.info(f"Starting new position tracking at {time.strftime('%H:%M:%S')}")
        
        # Update core position data
        self.position_size = size
        self.entry_price = price if not hasattr(self, 'entry_price') else self.entry_price
        
        # Calculate position value (absolute value of position in USD)
        self.position_value = abs(size * price)
        
        # Calculate unrealized PnL if we have an entry price
        if hasattr(self, 'entry_price') and self.entry_price > 0:
            if size > 0:  # Long position
                self.unrealized_pnl = (price - self.entry_price) * size
            elif size < 0:  # Short position
                self.unrealized_pnl = (self.entry_price - price) * abs(size)
            else:  # No position
                self.unrealized_pnl = 0.0
        else:
            self.unrealized_pnl = 0.0
        
        # Calculate PnL as percentage for reference
        pnl_percentage = 0.0
        if self.position_value > 0 and self.unrealized_pnl != 0:
            pnl_percentage = self.unrealized_pnl / self.position_value
        
        # If position flipped from long to short or vice versa, reset entry price and track as new position
        if (old_position > 0 and size < 0) or (old_position < 0 and size > 0):
            self.entry_price = price
            self.position_start_time = time.time()
            self.logger.info(f"Position flipped from {old_position} to {size}, resetting tracking")
        
        # If position closed completely, reset position tracking
        if size == 0 and old_position != 0:
            self.position_start_time = 0
            self.entry_price = 0
            self.unrealized_pnl = 0.0
            self.position_value = 0.0
        
        # Extended logging with more metrics
        self.logger.info(
            f"Position updated: {old_position:.4f} -> {size:.4f} @ {price:.2f} | "
            f"Value: {self.position_value:.2f} | PnL: {self.unrealized_pnl:.2f} ({pnl_percentage:.2%})"
        )

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
            
            # Apply predictive volatility adjustment
            prediction_age = time.time() - self.predictive_adjustments.get("last_update_time", 0)
            
            # Original volatility for logging
            original_volatility = volatility
            
            if prediction_age < 300:  # Only use predictions less than 5 minutes old
                volatility_adjustment = self.predictive_adjustments.get("volatility_adjustment", 0)
                if abs(volatility_adjustment) > 0.05:  # Only apply if significant
                    volatility *= (1 + volatility_adjustment)
                    self.logger.debug(f"Adjusted volatility: {original_volatility:.4f} -> {volatility:.4f} (factor: {1 + volatility_adjustment:.2f})")
            
            # Base Avellaneda-Stoikov spread calculation components
            # 1. Gamma/risk aversion component
            base_gamma = self.gamma
            
            # Apply predictive gamma adjustment
            original_gamma = base_gamma
            if prediction_age < 300:
                gamma_adjustment = self.predictive_adjustments.get("gamma_adjustment", 0)
                if abs(gamma_adjustment) > 0.05:  # Only apply if significant
                    base_gamma = max(0.1, base_gamma * (1 + gamma_adjustment))
                    self.logger.debug(f"Adjusted gamma: {original_gamma:.3f} -> {base_gamma:.3f} (factor: {1 + gamma_adjustment:.2f})")
            
            gamma_component = 1.0 + base_gamma
            
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
            
            # Use predictive trend direction if available
            if prediction_age < 300:
                trend_direction = self.predictive_adjustments.get("trend_direction", 0)
                if trend_direction != 0:
                    # Only update market state if different from current
                    if self.market_state != "trending":
                        self.market_state = "trending"
                        self.logger.info(f"Market state changed to trending due to predicted trend direction: {trend_direction}")
                    market_state_factor = 1.5  # Wider spreads in trending markets
                elif self.market_state == "trending":
                    # Only reset if we were previously trending
                    self.market_state = "normal"
                    self.logger.info("Market state reset to normal as no trend detected")
            
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
            
            # Apply predictive reservation price offset (store original for logging)
            original_mid_price = mid_price
            prediction_age = time.time() - self.predictive_adjustments.get("last_update_time", 0)
            reservation_offset = 0.0
            
            if prediction_age < 300:  # Only use predictions less than 5 minutes old
                reservation_offset = self.predictive_adjustments.get("reservation_price_offset", 0.0)
                
                # Apply offset to mid price directly if significant
                if abs(reservation_offset) > 0.00005:
                    adjusted_mid = mid_price + reservation_offset * mid_price
                    self.logger.debug(
                        f"Adjusted reservation price: {original_mid_price:.2f} -> {adjusted_mid:.2f} "
                        f"(offset: {reservation_offset:.6f}, value: {reservation_offset * mid_price:.2f})"
                    )
                    mid_price = adjusted_mid
            
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
            
            # Add more detail to skew logging
            self.logger.info(
                f"Skewed prices: bid={bid_price:.2f}, ask={ask_price:.2f}, "
                f"skew={inventory_skew:.4f}, reservation_offset={reservation_offset:.6f}"
            )
            
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
        Generate quotes based on current market conditions and inventory
        
        Args:
            ticker: Current ticker data
            market_conditions: Dictionary containing market conditions
            
        Returns:
            Tuple of (bid_quotes, ask_quotes)
        """
        try:
            bid_quotes = []
            ask_quotes = []
            
            # Skip if invalid ticker
            if not ticker or not hasattr(ticker, 'mark_price') or ticker.mark_price <= 0:
                self.logger.warning("Cannot generate quotes: Invalid ticker")
                return [], []
            
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
            base_bid_price, base_ask_price = self.calculate_skewed_prices(reference_price, spread)
            
            # 4. Safety check: ensure bid < ask and prices are positive
            if base_bid_price >= base_ask_price or base_bid_price <= 0 or base_ask_price <= 0:
                self.logger.warning(
                    f"Invalid prices calculated: bid={base_bid_price}, ask={base_ask_price}, " 
                    f"ref={reference_price}, spread={spread}"
                )
                # Recalculate with safe defaults
                min_spread_ticks = ORDERBOOK_CONFIG.get("min_spread", 5)
                safe_spread = min_spread_ticks * self.tick_size
                base_bid_price = reference_price - (safe_spread / 2)
                base_ask_price = reference_price + (safe_spread / 2)
            
            # 5. Align prices to tick size
            base_bid_price = self.align_price_to_tick(base_bid_price)
            base_ask_price = self.align_price_to_tick(base_ask_price)
            
            # 6. Calculate base quote sizes
            base_bid_size, base_ask_size = self.calculate_quote_sizes(reference_price)
            
            # Ensure sizes aren't too small
            min_size = ORDERBOOK_CONFIG.get("min_order_size", 0.001)
            if base_bid_size < min_size or base_ask_size < min_size:
                self.logger.warning(f"Quote sizes too small: bid={base_bid_size}, ask={base_ask_size}, using min size {min_size}")
                base_bid_size = max(base_bid_size, min_size)
                base_ask_size = max(base_ask_size, min_size)
                
            # 7. Create quote objects for multiple levels
            timestamp = datetime.now().timestamp()
            levels = self.quote_levels
            
            # Get the bid/ask step sizes for additional levels
            bid_step = ORDERBOOK_CONFIG.get("bid_step", 25) * self.tick_size
            ask_step = ORDERBOOK_CONFIG.get("ask_step", 25) * self.tick_size
            
            # Get the size multipliers for different levels - using Fibonacci-based sizing
            size_multipliers = self._calculate_fibonacci_size_multipliers(levels)
            
            # Create bid quotes at multiple levels if we have capacity to buy
            if self.can_increase_position():
                for level in range(levels):
                    # Calculate price for this level (further from mid for higher levels)
                    level_price = self._calculate_level_price(base_bid_price, level, bid_step, is_bid=True)
                    
                    # Calculate size for this level (use multiplier)
                    level_size = base_bid_size * size_multipliers[level] if level < len(size_multipliers) else base_bid_size * 0.1
                    
                    # Ensure minimum size
                    if level_size < min_size:
                        level_size = min_size
                    
                    # Create quote
                    bid_quote = Quote(
                        instrument=self.instrument,
                        side="buy",
                        price=level_price,
                        amount=level_size,
                        timestamp=timestamp
                    )
                    bid_quotes.append(bid_quote)
                    
                    self.logger.debug(f"Created level {level} bid: {level_size:.4f} @ {level_price:.2f}")
            else:
                self.logger.info("Skipping bid quotes: position limit reached")
                
            # Create ask quotes at multiple levels if we have position to sell
            if self.can_decrease_position():
                for level in range(levels):
                    # Calculate price for this level (further from mid for higher levels)
                    level_price = self._calculate_level_price(base_ask_price, level, ask_step, is_bid=False)
                    
                    # Calculate size for this level (use multiplier)
                    level_size = base_ask_size * size_multipliers[level] if level < len(size_multipliers) else base_ask_size * 0.1
                    
                    # Ensure minimum size
                    if level_size < min_size:
                        level_size = min_size
                    
                    # Create quote
                    ask_quote = Quote(
                        instrument=self.instrument,
                        side="sell",
                        price=level_price,
                        amount=level_size,
                        timestamp=timestamp
                    )
                    ask_quotes.append(ask_quote)
                    
                    self.logger.debug(f"Created level {level} ask: {level_size:.4f} @ {level_price:.2f}")
            else:
                self.logger.info("Skipping ask quotes: no position to sell")
                
            # Log generated quotes
            realized_spread = base_ask_price - base_bid_price if bid_quotes and ask_quotes else None
            spread_display = f"{realized_spread:.2f}" if realized_spread is not None else "N/A"
            self.logger.info(
                f"Generated quotes: {len(bid_quotes)} bids and {len(ask_quotes)} asks, "
                f"base bid={base_bid_price:.2f}, base ask={base_ask_price:.2f}, "
                f"ref={reference_price:.2f}, spread={spread_display}"
            )
            
            return bid_quotes, ask_quotes
            
        except Exception as e:
            self.logger.error(f"Error generating quotes: {str(e)}")
            return [], []

    def _calculate_level_price(self, base_price: float, level: int, step: float, is_bid: bool) -> float:
        """
        Calculate price for a specific quote level using Fibonacci grid spacing
        
        Args:
            base_price: Base price for level 0
            level: Level number (0 = closest to mid)
            step: Step size between levels
            is_bid: True if calculating for bid, False for ask
            
        Returns:
            float: Price for this level
        """
        if level == 0:
            return base_price
            
        # Fibonacci sequence: 1, 1, 2, 3, 5, 8, 13, 21, 34, ...
        # We'll use a simplified approach with pre-computed Fibonacci multipliers
        fib_multipliers = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89]
        
        # Use the appropriate multiplier based on level
        # For levels beyond our pre-computed list, use the last multiplier
        fib_multiplier = fib_multipliers[level] if level < len(fib_multipliers) else fib_multipliers[-1]
        
        # Adjust step size with Fibonacci multiplier
        fib_step = step * fib_multiplier
        
        # Bids get lower as level increases, asks get higher
        if is_bid:
            return self.align_price_to_tick(base_price - fib_step)
        else:
            return self.align_price_to_tick(base_price + fib_step)

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
        """Validate and filter quotes"""
        valid_bid_quotes = []
        valid_ask_quotes = []
        
        # Ensure minimum size is respected (Thalex minimum is 0.01)
        min_quote_size = 0.01  # Default minimum quote size for Thalex
        
        # Get min_size from trading config if available
        try:
            if 'trading_strategy' in TRADING_CONFIG and 'avellaneda' in TRADING_CONFIG['trading_strategy']:
                min_quote_size = max(0.01, TRADING_CONFIG['trading_strategy']['avellaneda'].get('base_size', 0.01))
        except (KeyError, TypeError):
            self.logger.warning("Could not get min_size from config, using default of 0.01")
        
        # Calculate maximum price deviation from ticker (3% by default)
        max_deviation_pct = 0.03
        
        # Get the market price from the bid/ask quotes if available
        market_price = 0
        if len(bid_quotes) > 0 and len(ask_quotes) > 0:
            # Use midpoint of best bid/ask as reference
            market_price = (bid_quotes[0].price + ask_quotes[0].price) / 2
        
        # Calculate absolute max deviation
        max_deviation = market_price * max_deviation_pct if market_price > 0 else 1000
        
        # Minimum and maximum valid prices
        min_valid_price = market_price * (1 - max_deviation_pct) if market_price > 0 else 1
        max_valid_price = market_price * (1 + max_deviation_pct) if market_price > 0 else float('inf')
        
        for quote in bid_quotes:
            # Validate bid quote
            if quote.price <= 0:
                self.logger.warning(f"Invalid bid price: {quote.price}")
                continue
                
            if quote.amount < min_quote_size:
                self.logger.debug(f"Increasing bid size from {quote.amount} to minimum {min_quote_size}")
                quote.amount = min_quote_size
                
            # Ensure price is not too far from market
            if market_price > 0 and (quote.price < min_valid_price or quote.price > market_price):
                # For bids, price should be below market price but not too far
                if quote.price > market_price:
                    self.logger.warning(f"Bid price {quote.price} above market {market_price}, adjusting")
                    quote.price = market_price - (self.tick_size * 2)  # 2 ticks below market
                elif quote.price < min_valid_price:
                    self.logger.warning(f"Bid price {quote.price} too far from market {market_price}, adjusting")
                    quote.price = min_valid_price
            
            valid_bid_quotes.append(quote)
            
        for quote in ask_quotes:
            # Validate ask quote
            if quote.price <= 0:
                self.logger.warning(f"Invalid ask price: {quote.price}")
                continue
                
            if quote.amount < min_quote_size:
                self.logger.debug(f"Increasing ask size from {quote.amount} to minimum {min_quote_size}")
                quote.amount = min_quote_size
                
            # Ensure price is not too far from market
            if market_price > 0 and (quote.price > max_valid_price or quote.price < market_price):
                # For asks, price should be above market price but not too far
                if quote.price < market_price:
                    self.logger.warning(f"Ask price {quote.price} below market {market_price}, adjusting")
                    quote.price = market_price + (self.tick_size * 2)  # 2 ticks above market
                elif quote.price > max_valid_price:
                    self.logger.warning(f"Ask price {quote.price} too far from market {market_price}, adjusting")
                    quote.price = max_valid_price
            
            valid_ask_quotes.append(quote)
            
        return valid_bid_quotes, valid_ask_quotes

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
            
            # NEW: Update volume candle buffer with fill data
            self.volume_buffer.update(fill_price, fill_size, is_buy, int(time.time() * 1000))
            
            # Update monetary position (which will trigger hedge updates if needed)
            self.update_monetary_position()
            
            # Check if we're using the hedge manager
            if self.use_hedging and self.hedge_manager is not None:
                self._process_fill_for_hedging(order_id, fill_price, fill_size, is_buy)
            
            self.logger.info(
                f"Order filled: {fill_size} @ {fill_price} ({'BUY' if is_buy else 'SELL'})"
                f" - New position: {self.position_size:.4f}, Avg entry: {self.entry_price:.2f}"
            )
            
        except Exception as e:
            self.logger.error(f"Error handling order fill: {str(e)}")

    def _process_fill_for_hedging(self, order_id: str, fill_price: float, fill_size: float, is_buy: bool):
        """
        Process a fill through the hedge manager
        
        Args:
            order_id: Order ID
            fill_price: Fill price
            fill_size: Fill size
            is_buy: Whether it was a buy (True) or sell (False)
        """
        try:
            # Create a fill object suitable for the hedge manager
            class HedgeFill:
                def __init__(self, instrument, price, size, is_buy, order_id):
                    self.instrument = instrument
                    self.price = price
                    self.size = size
                    self.is_buy = is_buy
                    self.fill_id = order_id
            
            # Make sure we have a valid instrument
            if not self.instrument or self.instrument == "unknown":
                self.logger.warning(f"No instrument set for fill, using BTC-PERPETUAL as default")
                instrument = "BTC-PERPETUAL"
            else:
                instrument = self.instrument
                
            # Create the fill object
            fill = HedgeFill(
                instrument=instrument,
                price=fill_price,
                size=fill_size,
                is_buy=is_buy,
                order_id=order_id
            )
            
            # Process the fill through hedge manager
            self.hedge_manager.on_fill(fill)
            self.logger.info(f"Processed fill for hedging: {instrument} {'BUY' if is_buy else 'SELL'} {fill_size} @ {fill_price}")
            
            # Get current position status for logging
            hedge_position = self.hedge_manager.get_hedged_position(instrument)
            if hedge_position:
                self.logger.info(
                    f"Current hedge: {hedge_position.primary_position} {instrument} hedged with "
                    f"{hedge_position.hedge_position} {hedge_position.hedge_asset} (ratio: {hedge_position.hedge_ratio:.2f})"
                )
        except Exception as e:
            self.logger.error(f"Error processing fill for hedging: {e}", exc_info=True)

    def get_position_metrics(self) -> Dict:
        """Get position and performance metrics"""
        metrics = {
            'position': self.position_size,
            'entry_price': self.entry_price,
            'realized_pnl': self.position_tracker.realized_pnl,
            'unrealized_pnl': self.position_tracker.unrealized_pnl,
            'total_pnl': self.position_tracker.realized_pnl + self.position_tracker.unrealized_pnl,
            'vwap': self.vwap,
            'vamp': self.calculate_vamp()
        }
        
        # NEW: Add predictive signals to metrics
        try:
            if hasattr(self, 'volume_buffer'):
                metrics['predictive_signals'] = self.volume_buffer.get_signal_metrics()['signals']
                
                # Add prediction accuracy if available
                if hasattr(self, 'last_mid_price') and self.last_mid_price > 0:
                    metrics['prediction_accuracy'] = self.volume_buffer.evaluate_prediction_accuracy(self.last_mid_price)
        except Exception as e:
            self.logger.error(f"Error getting predictive metrics: {str(e)}")
            
        return metrics

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

    def calculate_dynamic_gamma(self, volatility: float, market_impact: float) -> float:
        """
        Calculate optimal gamma value dynamically based on market conditions
        
        Args:
            volatility: Current market volatility
            market_impact: Current market impact estimate
            
        Returns:
            Optimized gamma value
        """
        try:
            self.logger.info(f"Starting dynamic gamma calculation with: volatility={volatility}, impact={market_impact}")
            
            # Base gamma from config
            base_gamma = TRADING_CONFIG["avellaneda"]["gamma"]
            
            # Higher volatility = higher gamma (wider spreads to compensate for risk)
            # Lower volatility = lower gamma (tighter spreads to capture more flow)
            volatility_factor = volatility / TRADING_CONFIG["volatility"]["default"]
            volatility_factor = min(2.0, max(0.5, volatility_factor))
            
            # Higher market impact = higher gamma (wider spreads to compensate for market impact)
            impact_factor = 1.0
            if market_impact > 0:
                impact_factor = market_impact / TRADING_CONFIG["avellaneda"]["order_flow_intensity"]
                impact_factor = min(2.0, max(0.5, impact_factor))
            
            # Enhanced position risk management
            position_factor = 1.0
            profitability_factor = 1.0
            
            if hasattr(self, 'position_limit') and self.position_limit > 0:
                # Calculate position utilization ratio (how much of our limit we're using)
                position_utilization = min(1.0, abs(self.position_size) / self.position_limit)
                
                # Progressive scaling as position grows (more aggressive scaling)
                # This creates a non-linear increase in gamma as we approach position limits
                if position_utilization <= 0.2:  # Small position
                    position_factor = 1.0
                elif position_utilization <= 0.5:  # Medium position
                    position_factor = 1.0 + (position_utilization - 0.2) * 1.5  # Scales from 1.0 to 1.45
                elif position_utilization <= 0.8:  # Large position
                    position_factor = 1.45 + (position_utilization - 0.5) * 2.0  # Scales from 1.45 to 2.05
                else:  # Very large position (near limit)
                    position_factor = 2.05 + (position_utilization - 0.8) * 3.0  # Scales from 2.05 to 2.95
                
                # Additional profitability-based adjustment
                # If we have a profitable position, we can be more aggressive (lower gamma)
                # If we have a losing position, we need to be more conservative (higher gamma)
                if hasattr(self, 'unrealized_pnl') and hasattr(self, 'position_value') and self.position_value > 0:
                    pnl_percentage = self.unrealized_pnl / self.position_value
                    
                    # Adjust gamma based on profitability
                    if pnl_percentage >= 0.01:  # >= 1% profit
                        # Profitable position - can be more aggressive (lower gamma)
                        profitability_factor = max(0.8, 1.0 - pnl_percentage * 5.0)  # Can reduce gamma up to 20%
                    elif pnl_percentage <= -0.005:  # >= 0.5% loss
                        # Losing position - be more conservative (higher gamma)
                        profitability_factor = min(1.5, 1.0 + abs(pnl_percentage) * 10.0)  # Can increase gamma up to 50%
                    
                    self.logger.info(f"Position profitability factor: {profitability_factor:.2f} (PnL %: {pnl_percentage:.2%})")
            
            # Add time-based risk ramp-up for positions held too long
            time_factor = 1.0
            if hasattr(self, 'position_start_time') and self.position_size != 0:
                position_duration = time.time() - self.position_start_time
                # If position held more than 30 minutes, start increasing gamma
                if position_duration > 1800:  # 30 minutes
                    time_multiplier = min(2.0, 1.0 + (position_duration - 1800) / 7200)  # Increase up to 2x over 2 hours
                    time_factor = time_multiplier
                    self.logger.info(f"Position duration factor: {time_factor:.2f} (duration: {position_duration/60:.1f} min)")
            
            # Calculate dynamic gamma with all factors
            dynamic_gamma = base_gamma * volatility_factor * impact_factor * position_factor * profitability_factor * time_factor
            
            # Clamp to reasonable range to avoid extreme values
            min_gamma = 0.1
            max_gamma = 0.5
            dynamic_gamma = min(max_gamma, max(min_gamma, dynamic_gamma))
            
            self.logger.info(
                f"Dynamic gamma calculation details:\n"
                f"  - Base gamma:          {base_gamma:.3f}\n"
                f"  - Volatility:          {volatility:.5f} (factor: {volatility_factor:.2f})\n"
                f"  - Market impact:       {market_impact:.5f} (factor: {impact_factor:.2f})\n"
                f"  - Position:            {self.position_size if hasattr(self, 'position_size') else 0} (factor: {position_factor:.2f})\n"
                f"  - Profitability:       factor: {profitability_factor:.2f}\n"
                f"  - Time-based risk:     factor: {time_factor:.2f}\n"
                f"  - Result (unclamped):  {base_gamma * volatility_factor * impact_factor * position_factor * profitability_factor * time_factor:.3f}\n"
                f"  - Final gamma:         {dynamic_gamma:.3f} (after min/max clamping)"
            )
            
            return dynamic_gamma
            
        except Exception as e:
            self.logger.error(f"Error calculating dynamic gamma: {str(e)}", exc_info=True)
            return TRADING_CONFIG["avellaneda"]["gamma"]  # Fallback to config value
            
    def update_gamma(self, gamma: float):
        """
        Update the gamma value used by the market maker
        
        Args:
            gamma: New gamma value to use
        """
        old_gamma = self.gamma
        self.gamma = gamma
        self.logger.info(f"Updated gamma: {old_gamma:.3f} -> {self.gamma:.3f}") 

    def update_market_data(self, price: float, volume: float = 0, is_buy: Optional[bool] = None, is_trade: bool = False) -> None:
        """
        Update market data with new information
        
        Args:
            price: Current price
            volume: Volume of trade or order
            is_buy: Whether it's a buy (True) or sell (False)
            is_trade: Whether this is a trade (True) or other market data (False)
        """
        try:
            # Update last price
            if price > 0:
                self.last_mid_price = price
                
            # Update VAMP if this is a trade
            if is_trade and volume > 0 and is_buy is not None:
                self.update_vamp(price, volume, is_buy, False)
                
                # Update volume candles for predictive analysis
                completed_candle = self.volume_buffer.update(price, volume, is_buy, int(time.time() * 1000))
                if completed_candle:
                    # Get updated predictions when a candle completes
                    self._update_predictive_parameters()
                    
                    # Log completion of new candle with its delta ratio
                    self.logger.debug(
                        f"Volume candle completed: V={completed_candle.volume:.3f}, "
                        f"Δ={completed_candle.delta_ratio:.2f}, "
                        f"P={completed_candle.close_price:.2f}"
                    )
            
            # Update last update time
            self.last_update_time = time.time()
            
        except Exception as e:
            self.logger.error(f"Error updating market data: {str(e)}")

    def _update_predictive_parameters(self) -> None:
        """Update predictive parameters based on volume candle analysis"""
        try:
            # Get predictions from volume candle buffer
            predictions = self.volume_buffer.get_predicted_parameters()
            
            # Save old values for logging changes
            old_predictions = self.predictive_adjustments.copy()
            
            # Apply sensitivity multipliers from config
            sensitivity = TRADING_CONFIG.get("volume_candle", {}).get("sensitivity", {})
            
            # Apply sensitivity multipliers to each adjustment
            if sensitivity:
                if "momentum" in sensitivity and "reservation_price" in sensitivity:
                    momentum_factor = sensitivity.get("momentum", 1.0)
                    price_factor = sensitivity.get("reservation_price", 1.0)
                    predictions["reservation_price_offset"] *= momentum_factor * price_factor
                
                if "volatility" in sensitivity:
                    volatility_factor = sensitivity.get("volatility", 1.0)
                    predictions["volatility_adjustment"] *= volatility_factor
                    
                if "reversal" in sensitivity:
                    reversal_factor = sensitivity.get("reversal", 1.0)
                    predictions["gamma_adjustment"] *= reversal_factor
            
            # Set minimum thresholds for adjustments to prevent noise
            if abs(predictions["gamma_adjustment"]) < 0.05:
                predictions["gamma_adjustment"] = 0.0
                
            if abs(predictions["kappa_adjustment"]) < 0.05:
                predictions["kappa_adjustment"] = 0.0
                
            if abs(predictions["reservation_price_offset"]) < 0.00005:
                predictions["reservation_price_offset"] = 0.0
                
            if abs(predictions["volatility_adjustment"]) < 0.05:
                predictions["volatility_adjustment"] = 0.0
            
            # Update predictive adjustments
            self.predictive_adjustments = predictions
            self.predictive_adjustments["last_update_time"] = time.time()
            
            # Log all significant parameter changes
            changes = []
            if old_predictions.get("gamma_adjustment", 0) != predictions["gamma_adjustment"]:
                changes.append(f"γ_adj: {old_predictions.get('gamma_adjustment', 0):.3f}->{predictions['gamma_adjustment']:.3f}")
                
            if old_predictions.get("kappa_adjustment", 0) != predictions["kappa_adjustment"]:
                changes.append(f"κ_adj: {old_predictions.get('kappa_adjustment', 0):.3f}->{predictions['kappa_adjustment']:.3f}")
                
            if old_predictions.get("reservation_price_offset", 0) != predictions["reservation_price_offset"]:
                changes.append(f"r_offset: {old_predictions.get('reservation_price_offset', 0):.6f}->{predictions['reservation_price_offset']:.6f}")
                
            if old_predictions.get("trend_direction", 0) != predictions["trend_direction"]:
                changes.append(f"trend: {old_predictions.get('trend_direction', 0)}->{predictions['trend_direction']}")
                
            if old_predictions.get("volatility_adjustment", 0) != predictions["volatility_adjustment"]:
                changes.append(f"vol_adj: {old_predictions.get('volatility_adjustment', 0):.3f}->{predictions['volatility_adjustment']:.3f}")
            
            # Log changes if any adjustments were made
            if changes:
                self.logger.info(f"Applied predictive adjustments: {', '.join(changes)}")
                
                # If significant changes were made, log the signals that led to them
                if len(changes) > 1 or abs(predictions["gamma_adjustment"]) > 0.2 or abs(predictions["reservation_price_offset"]) > 0.0002:
                    signals = self.volume_buffer.signals
                    self.logger.info(
                        f"Signal basis: momentum={signals['momentum']:.2f}, reversal={signals['reversal']:.2f}, "
                        f"volatility={signals['volatility']:.2f}, exhaustion={signals['exhaustion']:.2f}"
                    )
                    
        except Exception as e:
            self.logger.error(f"Error updating predictive parameters: {str(e)}") 

    def _calculate_fibonacci_size_multipliers(self, levels: int) -> List[float]:
        """
        Calculate size multipliers based on Fibonacci ratios
        
        Args:
            levels: Number of quote levels to generate
            
        Returns:
            List of size multipliers for each level
        """
        # Start with default config multipliers
        default_multipliers = ORDERBOOK_CONFIG.get("bid_sizes", [1.0, 0.8, 0.6, 0.4, 0.2, 0.1])
        
        # If we don't need many levels, just return the defaults
        if levels <= len(default_multipliers):
            return default_multipliers[:levels]
            
        # For Fibonacci-based sizing, we use the inverse of the Fibonacci ratios
        # This makes orders closer to the mid price larger, and orders further away smaller
        fib_sequence = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89]
        
        # Calculate multipliers based on inverse Fibonacci ratios
        fib_multipliers = []
        max_fib = max(fib_sequence[:levels]) if levels < len(fib_sequence) else max(fib_sequence)
        
        for i in range(levels):
            # Use actual Fibonacci number if available, otherwise use the last one
            fib_value = fib_sequence[i] if i < len(fib_sequence) else fib_sequence[-1]
            # Invert and normalize to get multiplier (higher for lower levels)
            multiplier = 1.0 - (fib_value / max_fib * 0.9)  # Allow minimum of 0.1
            fib_multipliers.append(round(multiplier, 2))
            
        # Ensure the first level is always 1.0
        if fib_multipliers:
            fib_multipliers[0] = 1.0
            
        return fib_multipliers 

    def cleanup(self):
        """Clean up resources when shutting down"""
        # Stop hedge manager if it's running
        if self.use_hedging and self.hedge_manager is not None:
            self.hedge_manager.stop()
            self.logger.info("Hedge manager stopped")
        
        # Add other cleanup tasks as needed 

    def update_monetary_position(self):
        """Update and log the monetary position value"""
        if self.last_mid_price > 0:
            old_monetary = self.monetary_position
            self.monetary_position = self.position_size * self.last_mid_price
            
            # Log the monetary position
            self.logger.info(f"Monetary position: ${self.monetary_position:.2f} from {self.position_size} {self.instrument} @ {self.last_mid_price}")
            
            # Check if monetary position changed significantly
            if abs(self.monetary_position - old_monetary) > 100:  # $100 change threshold
                # Update hedge if enabled
                if self.use_hedging and self.hedge_manager is not None:
                    self.hedge_manager.update_position(
                        self.instrument,
                        self.position_size,
                        self.last_mid_price
                    )
                    # Report hedge status after update
                    self.report_hedge_status()
    
    def report_hedge_status(self):
        """Report on monetary position and hedge status"""
        if not self.use_hedging or self.hedge_manager is None:
            return
        
        # Calculate and log monetary position
        monetary_position = self.position_size * self.last_mid_price
        
        # Get hedge position
        hedge_position = self.hedge_manager.get_hedged_position(self.instrument)
        if hedge_position:
            hedge_monetary = hedge_position.hedge_position * hedge_position.hedge_price
            net_monetary = monetary_position + hedge_monetary
            
            self.logger.info(f"=== HEDGE STATUS ===")
            self.logger.info(f"Primary: {self.position_size} {self.instrument} @ {self.last_mid_price:.2f} = ${monetary_position:.2f}")
            self.logger.info(f"Hedge: {hedge_position.hedge_position} {hedge_position.hedge_asset} @ {hedge_position.hedge_price:.2f} = ${hedge_monetary:.2f}")
            self.logger.info(f"Net monetary exposure: ${net_monetary:.2f}")
            if monetary_position != 0:
                self.logger.info(f"Hedge ratio: {abs(hedge_monetary/monetary_position):.2f} (target: {hedge_position.hedge_ratio:.2f})")
            self.logger.info(f"Current P&L: ${hedge_position.pnl:.2f}")
        else:
            self.logger.info(f"No active hedge for {self.instrument} position (${monetary_position:.2f})")
    
    def force_hedge_rebalance(self):
        """Force an immediate rebalance of all hedges"""
        if self.use_hedging and self.hedge_manager is not None:
            # Get current position information
            current_position = self.position_size
            current_price = self.last_mid_price
            
            # Update the position to force hedge calculation
            result = self.hedge_manager.update_position(
                self.instrument, 
                current_position,
                current_price
            )
            
            self.logger.info(f"Forced hedge rebalance for {self.instrument} position: {current_position} @ {current_price}")
            
            # Report on hedge operations
            if result["hedges"]:
                for hedge in result["hedges"]:
                    self.logger.info(f"Hedge operation: {hedge['side']} {hedge['size']} {hedge['hedge_asset']} @ {hedge['price']}")
            
            # Report hedge status
            self.report_hedge_status() 