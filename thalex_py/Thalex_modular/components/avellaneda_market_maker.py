import numpy as np
from typing import Dict, List, Optional, Tuple, Any, NamedTuple
import time
from collections import deque
import math
import random
from datetime import datetime
import logging

from ..config.market_config import TRADING_CONFIG, RISK_LIMITS, ORDERBOOK_CONFIG
from ..models.data_models import Ticker, Quote
from ..models.position_tracker import PositionTracker, Fill
from ..thalex_logging import LoggerFactory
from ..ringbuffer.volume_candle_buffer import VolumeBasedCandleBuffer

# Constants
DEFAULT_RESERVATION_PRICE_BOUND = TRADING_CONFIG["avellaneda"].get("reservation_price_bound", 0.005)
MIN_SPREAD_TICK_MULTIPLIER = 2  # Minimum spread as a multiple of tick size

class AvellanedaMarketMaker:
    """Avellaneda-Stoikov market making strategy implementation"""
    
    def __init__(self, exchange_client=None, position_tracker: PositionTracker = None):
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
        self.position_tracker = position_tracker
        
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
        self.price_history = deque(maxlen=TRADING_CONFIG["volatility"]["window"])
        
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
            volume_threshold=TRADING_CONFIG["volume_candle"].get("threshold", 1.0),
            max_candles=TRADING_CONFIG["volume_candle"].get("max_candles", 100),
            max_time_seconds=TRADING_CONFIG["volume_candle"].get("max_time_seconds", 300),
            exchange_client=exchange_client,
            instrument=None,  # Will be set via set_instrument method
            use_exchange_data=TRADING_CONFIG["volume_candle"].get("use_exchange_data", False),
            fetch_interval_seconds=TRADING_CONFIG["volume_candle"].get("fetch_interval_seconds", 60),
            lookback_hours=TRADING_CONFIG["volume_candle"].get("lookback_hours", 1)
        )
        
        # NEW: Predictive state tracking - initialize at the start to avoid None issues
        self.predictive_adjustments = {
            "gamma_adjustment": 0.0,
            "kappa_adjustment": 0.0,
            "reservation_price_offset": 0.0,
            "trend_direction": 0,
            "volatility_adjustment": 0.0,
            "last_update_time": 0
        }
        
        # Add parameters for real-time grid updates
        self.last_prediction_time = time.time()
        self.prediction_interval = TRADING_CONFIG["volume_candle"].get("prediction_update_interval", 10.0)
        self.last_grid_update_time = time.time()
        self.grid_update_interval = TRADING_CONFIG["quoting"].get("grid_update_interval", 3.0)
        self.force_grid_update = False
        
        self.logger.info("Avellaneda market maker initialized")
        
        # Avellaneda-Stoikov parameters
        self.inventory_weight = TRADING_CONFIG["avellaneda"]["inventory_weight"]  # Inventory skew
        self.position_fade_time = TRADING_CONFIG["avellaneda"]["position_fade_time"]  # Time to fade position
        self.order_flow_intensity = TRADING_CONFIG["avellaneda"]["order_flow_intensity"]  # Order flow intensity
        
        # Add critical missing parameters
        self.position_limit = TRADING_CONFIG["avellaneda"]["position_limit"]  # Position limit
        self.base_spread_factor = TRADING_CONFIG["avellaneda"].get("base_spread_factor", 1.0)  # Base spread multiplier
        self.market_impact_factor = TRADING_CONFIG["avellaneda"].get("market_impact_factor", 0.5)  # Market impact multiplier
        self.inventory_factor = TRADING_CONFIG["avellaneda"].get("inventory_factor", 0.5)  # Inventory adjustment factor
        self.volatility_multiplier = TRADING_CONFIG["avellaneda"].get("volatility_multiplier", 0.2)  # Volatility multiplier
        self.market_state = "normal"  # Market state: normal, trending, ranging
        
        # Market making state
        self.last_quote_time = 0.0
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
        self.market_impact = 0.0
        
        # Quote tracking
        self.active_quotes: Dict[str, Quote] = {}
        self.current_bid_quotes: List[Quote] = []
        self.current_ask_quotes: List[Quote] = []

        # Hedge manager integration
        self.hedge_manager = None
        self.use_hedging = False  # Always disable hedging
        
        # Skip the hedge manager initialization to avoid the exchange_api_key error
        if False and self.use_hedging:  # Force this to never execute
            from .hedge import create_hedge_manager
            # Create hedge manager with default settings and the exchange client
            hedge_config_dict = {}
            hedge_strategy = TRADING_CONFIG.get("hedging", {}).get("strategy", "notional")
            # Add strategy to config dict if specified
            if hedge_strategy:
                hedge_config_dict["strategy"] = hedge_strategy
                
            self.hedge_manager = create_hedge_manager(
                config_path=TRADING_CONFIG.get("hedging", {}).get("config_path"),
                config_dict=hedge_config_dict if hedge_config_dict else None,
                exchange_client=self.exchange_client
            )
            # Start the hedge manager's background thread for rebalancing
            self.hedge_manager.start()
            self.logger.info("Hedge manager initialized and started")
        else:
            self.logger.info("Hedging is disabled")

    def set_tick_size(self, tick_size: float):
        """Set the tick size for the instrument"""
        self.tick_size = tick_size
        self.min_tick = tick_size
        self.logger.info(f"Tick size set to {tick_size}")

    def update_market_conditions(self, volatility: float, market_impact: float):
        """Update the market conditions used for quote generation"""
        self.volatility = volatility
        self.market_impact = market_impact
        self.logger.debug(f"Market conditions updated: vol={self.volatility:.6f}, impact={market_impact:.6f}")
    
    def update_vamp(self, price: float, volume: float, is_buy: bool, is_aggressive: bool=False):
        """Update Volume-Adjusted Market Price calculations"""
        # Log entry
        side_str = "BUY" if is_buy else "SELL"
        agg_str = "aggressive" if is_aggressive else "passive"
        self.logger.debug(f"Updating VAMP with {side_str} {volume:.6f} @ {price:.2f} ({agg_str})")
        
        # Save old values for comparison
        old_vwap = getattr(self, 'vwap', 0)
        old_vamp_value = getattr(self, 'vamp_value', 0)
        
        # Update VWAP (Volume-Weighted Average Price)
        self.volume_price_sum += price * volume
        self.total_volume += volume
        if self.total_volume > 0:
            self.vwap = self.volume_price_sum / self.total_volume
            self.logger.debug(f"VWAP updated: {old_vwap:.2f} → {self.vwap:.2f} (Δ: {self.vwap - old_vwap:+.2f})")
        
        # Update aggressive volume tracking with detailed logging
        if is_aggressive:
            if is_buy:
                old_buys = self.market_buys_volume
                self.market_buys_volume += volume
                self.aggressive_buys_sum += price * volume
                self.logger.debug(f"Aggressive buys: {old_buys:.6f} → {self.market_buys_volume:.6f} (Δ: +{volume:.6f})")
            else:
                old_sells = self.market_sells_volume
                self.market_sells_volume += volume
                self.aggressive_sells_sum += price * volume
                self.logger.debug(f"Aggressive sells: {old_sells:.6f} → {self.market_sells_volume:.6f} (Δ: +{volume:.6f})")
            
            # Record impact in the impact window for time-weighted impact decay
            current_time = time.time()
            impact_value = volume * (1 if is_buy else -1)
            self.vamp_impact_window.append((current_time, impact_value))
            
            # Calculate time-weighted impact for market making adjustments
            self.vamp_impact = self._calculate_time_weighted_impact()
            self.logger.debug(f"VAMP impact updated: {self.vamp_impact:.4f} (window size: {len(self.vamp_impact_window)})")
        
        # Calculate VAMP
        vamp = self.calculate_vamp()
        
        # Record the change
        vamp_change = vamp - old_vamp_value
        self.vamp_value = vamp
        
        # More detailed logging with value changes
        self.logger.info(
            f"VAMP updated: {old_vamp_value:.2f} → {vamp:.2f} (Δ: {vamp_change:+.2f}), "
            f"VWAP: {self.vwap:.2f}, Total Vol: {self.total_volume:.6f}, "
            f"Buy/Sell Ratio: {self._get_buy_sell_ratio():.2f}, Impact: {self.vamp_impact:.4f}"
        )
        
        return vamp
    
    def _get_buy_sell_ratio(self) -> float:
        """Calculate buy/sell ratio for logging"""
        if self.market_buys_volume + self.market_sells_volume == 0:
            return 1.0  # Neutral if no volume
        return self.market_buys_volume / (self.market_buys_volume + self.market_sells_volume)
    
    def _calculate_time_weighted_impact(self) -> float:
        """Calculate time-weighted impact from the impact window"""
        if not self.vamp_impact_window:
            return 0.0
            
        # Current time
        current_time = time.time()
        
        # Calculate decay factor - more recent trades have higher weight
        total_weight = 0.0
        weighted_impact = 0.0
        
        # Max age to consider (default 5 minutes)
        max_age = TRADING_CONFIG["vamp"].get("max_impact_age", 300)
        
        for timestamp, impact in self.vamp_impact_window:
            age = current_time - timestamp
            if age > max_age:
                continue  # Skip too old values
                
            # Linear decay weight (1.0 for newest, approaching 0 for oldest)
            weight = max(0, 1.0 - (age / max_age))
            weighted_impact += impact * weight
            total_weight += weight
            
        # Normalize by total weight
        if total_weight > 0:
            return weighted_impact / total_weight
        return 0.0
    
    def calculate_vamp(self) -> float:
        """Calculate Volume-Adjusted Market Price"""
        # If no volume, use mid_price
        if self.total_volume == 0:
            self.logger.debug("No volume data available for VAMP calculation")
            return 0.0
            
        # Basic VWAP if no aggressive trades
        if self.market_buys_volume == 0 and self.market_sells_volume == 0:
            self.logger.debug("No aggressive trades, using VWAP as VAMP")
            return self.vwap
            
        # Calculate buy/sell pressure with detailed logging
        buy_vwap = 0
        sell_vwap = 0
        
        if self.market_buys_volume > 0:
            buy_vwap = self.aggressive_buys_sum / self.market_buys_volume
            self.logger.debug(f"Buy VWAP: {buy_vwap:.2f} from {self.market_buys_volume:.6f} volume")
            
        if self.market_sells_volume > 0:
            sell_vwap = self.aggressive_sells_sum / self.market_sells_volume
            self.logger.debug(f"Sell VWAP: {sell_vwap:.2f} from {self.market_sells_volume:.6f} volume")
        
        # If only one side has volume, use that side
        if self.market_buys_volume == 0:
            self.logger.debug("Only sell-side aggressive volume, using sell VWAP")
            return sell_vwap
            
        if self.market_sells_volume == 0:
            self.logger.debug("Only buy-side aggressive volume, using buy VWAP")
            return buy_vwap
            
        # Calculate VAMP using both sides
        buy_ratio = self.market_buys_volume / (self.market_buys_volume + self.market_sells_volume)
        sell_ratio = 1 - buy_ratio
        
        # VAMP is weighted average of buy and sell VWAPs
        vamp = (buy_vwap * buy_ratio) + (sell_vwap * sell_ratio)
        
        # Detailed calculation explanation
        self.logger.debug(
            f"VAMP calculation: ({buy_vwap:.2f} × {buy_ratio:.2f}) + ({sell_vwap:.2f} × {sell_ratio:.2f}) = {vamp:.2f}, "
            f"Buy bias: {'+' if buy_ratio > 0.5 else '-'}{abs(buy_ratio - 0.5) * 2:.1f}x"
        )
        
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
            min_spread_ticks = ORDERBOOK_CONFIG["min_spread"]
            
            # Ensure tick_size is valid
            if self.tick_size <= 0:
                self.logger.warning("Invalid tick_size detected in calculate_optimal_spread. Using default of 1.0")
                self.tick_size = 1.0
                
            min_spread = min_spread_ticks * self.tick_size
            
            # Get volatility with safety fallback, ensure minimum is reasonable
            volatility = max(self.volatility, 0.01)  # At least 1% volatility
            
            # Apply predictive volatility adjustment
            prediction_age = time.time() - self.predictive_adjustments.get("last_update_time", 0)
            
            # Original volatility for logging
            original_volatility = volatility
            
            # Fetch thresholds from config
            pa_prediction_max_age_seconds = TRADING_CONFIG["avellaneda"].get("pa_prediction_max_age_seconds", 300)
            pa_volatility_adj_threshold = TRADING_CONFIG["avellaneda"].get("pa_volatility_adj_threshold", 0.05)

            if prediction_age < pa_prediction_max_age_seconds:  # Only use predictions less than configured max age
                volatility_adjustment = self.predictive_adjustments.get("volatility_adjustment", 0)
                if abs(volatility_adjustment) > pa_volatility_adj_threshold:  # Only apply if significant
                    volatility *= (1 + volatility_adjustment)
                    self.logger.debug(f"Adjusted volatility: {original_volatility:.4f} -> {volatility:.4f} (factor: {1 + volatility_adjustment:.2f}, threshold: {pa_volatility_adj_threshold:.2f})")
            
            # Base Avellaneda-Stoikov spread calculation components
            # 1. Gamma/risk aversion component
            base_gamma = self.gamma
            
            # Ensure gamma is positive to avoid negative spread components
            if base_gamma <= 0:
                self.logger.warning(f"Invalid gamma value {base_gamma}, using default of 0.1")
                base_gamma = 0.1
            
            # Apply predictive gamma adjustment
            original_gamma = base_gamma
            # Fetch threshold from config
            pa_gamma_adj_threshold = TRADING_CONFIG["avellaneda"].get("pa_gamma_adj_threshold", 0.05)

            if prediction_age < pa_prediction_max_age_seconds: # Already fetched pa_prediction_max_age_seconds
                gamma_adjustment = self.predictive_adjustments.get("gamma_adjustment", 0)
                if abs(gamma_adjustment) > pa_gamma_adj_threshold:  # Only apply if significant
                    base_gamma = max(0.1, base_gamma * (1 + gamma_adjustment))
                    self.logger.debug(f"Adjusted gamma: {original_gamma:.3f} -> {base_gamma:.3f} (factor: {1 + gamma_adjustment:.2f}, threshold: {pa_gamma_adj_threshold:.2f})")
            
            gamma_component = 1.0 + base_gamma
            
            # 2. Volatility-based component (more volatile = wider spread)
            volatility_term = volatility * self.volatility_multiplier
            
            # Ensure time horizon is not zero to prevent division by zero
            if not hasattr(self, 'position_fade_time') or self.position_fade_time <= 0:
                self.position_fade_time = TRADING_CONFIG["avellaneda"].get("position_fade_time", 3600)
                self.logger.warning(f"Invalid position_fade_time, using default: {self.position_fade_time}")
                
            volatility_component = volatility_term * math.sqrt(self.position_fade_time)
            
            # 3. Market impact adjustment (high impact = wider spread)
            # Ensure market_impact is not negative
            market_impact_arg = max(0, market_impact) # Renamed original market_impact to avoid confusion
            
            # VAMP integration for spread adjustment
            vamp_spread_sensitivity = TRADING_CONFIG["avellaneda"].get("vamp_spread_sensitivity", 0.0) # Default to 0 if not found
            vamp_effect_on_spread = abs(self.vamp_impact) * vamp_spread_sensitivity
            effective_market_impact = market_impact_arg + vamp_effect_on_spread

            self.logger.debug(
                f"VAMP effect on spread: base_impact_arg={market_impact_arg:.4f}, "
                f"vamp_impact_abs={abs(self.vamp_impact):.4f}, sensitivity={vamp_spread_sensitivity:.2f}, "
                f"vamp_effect={vamp_effect_on_spread:.4f}, effective_impact={effective_market_impact:.4f}"
            )

            # Scale market impact by both volatility and order flow intensity for more responsive spreads
            market_impact_component = effective_market_impact * self.market_impact_factor * (1 + volatility)
            
            # 4. Position/inventory risk component (larger position = wider spread)
            # Ensure position_limit is not zero to prevent division by zero
            if not hasattr(self, 'position_limit') or self.position_limit <= 0:
                self.position_limit = TRADING_CONFIG["avellaneda"].get("position_limit", 0.1)
                if self.position_limit <= 0:
                    self.position_limit = 0.1  # Minimum valid value
                    self.logger.warning(f"Invalid position_limit, using minimum: {self.position_limit}")
            
            inventory_risk = abs(self.position_tracker.current_position) / max(self.position_limit, 0.001)
            inventory_component = self.inventory_factor * inventory_risk * volatility_term
            
            # 5. Market state component (trending/ranging adjustment)
            market_state_factor = 1.0
            
            # Use predictive trend direction if available
            # pa_prediction_max_age_seconds already fetched
            if prediction_age < pa_prediction_max_age_seconds:
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
                    f"impact_arg={market_impact_arg:.2f}, vamp_effect={vamp_effect_on_spread:.2f}, effective_impact_comp={market_impact_component:.2f}, "
                    f"inventory={inventory_component:.2f}, market_state_factor={market_state_factor:.2f}, final={final_spread:.2f}"
                )
                
            return final_spread
        except Exception as e:
            self.logger.error(f"Error calculating optimal spread: {str(e)}", exc_info=True)
            # Return safe default spread on error
            min_spread_ticks = ORDERBOOK_CONFIG.get("min_spread", 5)
            if self.tick_size <= 0:
                self.tick_size = 1.0
            return min_spread_ticks * self.tick_size

    def calculate_skewed_prices(self, mid_price: float, spread: float) -> Tuple[float, float]:
        """Calculate skewed bid and ask prices based on inventory"""
        try:
            # Simple inventory skew based on position size and limits
            position_limit = TRADING_CONFIG["avellaneda"]["position_limit"]
            inventory_skew_factor = self.inventory_weight * 0.5
            inventory_skew = 0
            
            if position_limit > 0:
                inventory_skew = (self.position_tracker.current_position / position_limit) * spread * inventory_skew_factor
            
            # Apply predictive reservation price offset (store original for logging)
            original_mid_price = mid_price
            prediction_age = time.time() - self.predictive_adjustments.get("last_update_time", 0)
            reservation_offset = 0.0
            
            # Fetch thresholds from config
            pa_prediction_max_age_seconds = TRADING_CONFIG["avellaneda"].get("pa_prediction_max_age_seconds", 300)
            pa_res_price_offset_adj_threshold = TRADING_CONFIG["avellaneda"].get("pa_res_price_offset_adj_threshold", 0.00005)

            if prediction_age < pa_prediction_max_age_seconds:  # Only use predictions less than configured max age
                reservation_offset = self.predictive_adjustments.get("reservation_price_offset", 0.0)
                
                # Apply offset to mid price directly if significant
                if abs(reservation_offset) > pa_res_price_offset_adj_threshold:
                    adjusted_mid = mid_price + reservation_offset * mid_price
                    self.logger.debug(
                        f"Adjusted reservation price: {original_mid_price:.2f} -> {adjusted_mid:.2f} "
                        f"(offset: {reservation_offset:.6f}, value: {reservation_offset * mid_price:.2f}, threshold: {pa_res_price_offset_adj_threshold:.6f})"
                    )
                    mid_price = adjusted_mid
            
            # VAMP integration for skew adjustment
            vamp_skew_sensitivity = TRADING_CONFIG["avellaneda"].get("vamp_skew_sensitivity", 0.0) # Default to 0 if not found
            if abs(self.vamp_impact) > 1e-9: # Apply only if vamp_impact is non-trivial
                vamp_induced_price_offset = self.vamp_impact * vamp_skew_sensitivity
                original_mid_price_before_vamp_skew = mid_price
                mid_price += vamp_induced_price_offset
                self.logger.debug(
                    f"VAMP effect on skew: vamp_impact={self.vamp_impact:.4f}, sensitivity={vamp_skew_sensitivity:.6f}, "
                    f"vamp_induced_offset={vamp_induced_price_offset:.4f}, "
                    f"mid_price: {original_mid_price_before_vamp_skew:.2f} -> {mid_price:.2f}"
                )
            else:
                self.logger.debug(f"Skipping VAMP skew adjustment due to near-zero vamp_impact: {self.vamp_impact:.4f}")

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
                f"inventory_skew={inventory_skew:.4f}, pred_reservation_offset={reservation_offset:.6f}, "
                f"vamp_induced_offset={vamp_induced_price_offset if 'vamp_induced_price_offset' in locals() else 0.0:.4f}"
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
            position_ratio = min(1.0, abs(self.position_tracker.current_position) / max(position_limit, 0.001))
            
            # Calculate base bid and ask sizes
            bid_size = base_size
            ask_size = base_size
            
            # 1. Adjust based on position direction (asymmetric sizing)
            if self.position_tracker.current_position > 0:  # Long position - reduce bids, increase asks
                # Decrease bid size as position grows (down to 20% of base size at position limit)
                bid_size = bid_size * max(0.2, (1.0 - position_ratio * 0.8))
                
                # Increase ask size as position grows (up to 150% of base size at position limit)
                ask_size = ask_size * min(1.5, (1.0 + position_ratio * 0.5))
                
            elif self.position_tracker.current_position < 0:  # Short position - increase bids, reduce asks
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
            buy_capacity = position_limit - self.position_tracker.current_position
            sell_capacity = position_limit + self.position_tracker.current_position
            
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
        return self.position_tracker.current_position < self.position_limit
        
    def can_decrease_position(self) -> bool:
        """Check if we can decrease our position (sell some)"""
        return self.position_tracker.current_position > -self.position_limit

    def generate_quotes(self, ticker: Ticker, market_conditions: Dict, max_orders: int = None) -> Tuple[List[Quote], List[Quote]]:
        """Generate quotes using Avellaneda-Stoikov market making model with dynamic grid updates"""
        try:
            # Validate tick size before generating quotes
            if self.tick_size <= 0:
                self.logger.warning("Invalid tick size detected in generate_quotes. Setting default value of 1.0")
                self.tick_size = 1.0
                
            # Extract ticker data
            timestamp = getattr(ticker, 'timestamp', time.time())
            
            # Debug: Log all ticker attributes
            self.logger.debug(f"Ticker data: {vars(ticker) if hasattr(ticker, '__dict__') else 'No ticker data'}")
            
            mark_price = getattr(ticker, 'mark_price', 0)
            if mark_price <= 0:
                self.logger.error(f"Invalid mark price from ticker: {mark_price}, cannot generate quotes.")
                return [], []

            if ticker.best_bid_price is None or ticker.best_ask_price is None:
                self.logger.warning(
                    f"Ticker missing best_bid_price ({ticker.best_bid_price}) or best_ask_price ({ticker.best_ask_price}). Cannot generate quotes reliably."
                )
                return [], []

            bid_price = ticker.best_bid_price
            ask_price = ticker.best_ask_price

            if not (bid_price and ask_price and bid_price > 0 and ask_price > 0 and ask_price > bid_price):
                self.logger.error(f"Invalid BBO prices from ticker: bid={bid_price}, ask={ask_price}. Cannot generate quotes.")
                return [], []

            mid_price = (bid_price + ask_price) / 2
            if mid_price <= 0:
                self.logger.error(f"Calculated mid_price ({mid_price}) is invalid. Cannot generate quotes.")
                return [], []
            
            self.logger.debug(f"Using BBO for mid_price: bid={bid_price}, ask={ask_price}, mid={mid_price}")
            
            # Update market mid price
            self.last_mid_price = mid_price
            
            # Extract volatility
            try:
                volatility = market_conditions.get("volatility", self.volatility)
                self.logger.debug(f"Extracted volatility: {volatility}")
                market_impact = market_conditions.get("market_impact", 0.0)
                self.logger.debug(f"Extracted market_impact: {market_impact}")
            except Exception as e:
                self.logger.error(f"Error extracting market conditions: {str(e)}")
                return [], []
            
            # Update market conditions
            try:
                self.update_market_conditions(volatility, market_impact)
                self.logger.debug("Market conditions updated successfully")
            except Exception as e:
                self.logger.error(f"Error updating market conditions: {str(e)}")
                return [], []
            
            # Calculate optimal spread using Avellaneda-Stoikov model
            try:
                spread = self.calculate_optimal_spread(market_impact)
                self.logger.debug(f"Calculated spread: {spread}")
            except Exception as e:
                self.logger.error(f"Error calculating optimal spread: {str(e)}")
                return [], []
            
            # Apply reservation price offset from volume candle prediction if available
            try:
                reservation_offset = self.reservation_price_offset
                if abs(reservation_offset) > 0.00001:
                    # Apply as a percentage offset to the mid price
                    offset_amount = mid_price * reservation_offset
                    self.logger.debug(f"Applying reservation price offset: {offset_amount:.2f} ({reservation_offset:.6f})")
                    mid_price += offset_amount
            except Exception as e:
                self.logger.error(f"Error applying reservation price offset: {str(e)}")
                # Continue as this is not critical
            
            # Calculate base bid and ask prices with inventory skew
            try:
                base_bid_price, base_ask_price = self.calculate_skewed_prices(mid_price, spread)
                self.logger.debug(f"Calculated base prices: bid={base_bid_price}, ask={base_ask_price}")
            except Exception as e:
                self.logger.error(f"Error calculating skewed prices: {str(e)}")
                return [], []
            
            # Calculate optimal sizes for base bid and ask
            try:
                base_bid_size, base_ask_size = self.calculate_quote_sizes(mid_price)
                self.logger.debug(f"Calculated base sizes: bid={base_bid_size}, ask={base_ask_size}")
            except Exception as e:
                self.logger.error(f"Error calculating quote sizes: {str(e)}")
                return [], []
            
            # Round prices to tick size
            try:
                base_bid_price = self.round_to_tick(base_bid_price)
                base_ask_price = self.round_to_tick(base_ask_price)
                self.logger.debug(f"Rounded prices: bid={base_bid_price}, ask={base_ask_price}")
            except Exception as e:
                self.logger.error(f"Error rounding prices to tick size: {str(e)}")
                return [], []
            
            # Enforce minimum spread 
            try:
                min_spread_ticks = ORDERBOOK_CONFIG["min_spread"]
                min_spread = min_spread_ticks * self.tick_size
                actual_spread = base_ask_price - base_bid_price
                
                if actual_spread < min_spread:
                    # Adjust prices to ensure minimum spread
                    half_min_spread = min_spread / 2
                    base_bid_price = self.round_to_tick(mid_price - half_min_spread)
                    base_ask_price = self.round_to_tick(mid_price + half_min_spread)
                    self.logger.debug(f"Enforcing minimum spread: {min_spread} ticks, adjusted spread: {base_ask_price - base_bid_price}")
            except Exception as e:
                self.logger.error(f"Error enforcing minimum spread: {str(e)}")
                # Continue as this is not critical
            
            # Generate quotes at multiple price levels
            bid_quotes = []
            ask_quotes = []
            
            # Maximum number of levels to quote
            try:
                levels = min(self.quote_levels, ORDERBOOK_CONFIG.get("levels", 6))
                self.logger.debug(f"Using {levels} quote levels")
            except Exception as e:
                self.logger.error(f"Error determining quote levels: {str(e)}")
                levels = 1  # Default to 1 level
            
            # Get appropriate step sizes for grid spacing
            try:
                bid_step = ORDERBOOK_CONFIG.get("bid_step", 10) * self.tick_size
                ask_step = ORDERBOOK_CONFIG.get("ask_step", 10) * self.tick_size
                self.logger.debug(f"Base step sizes: bid={bid_step}, ask={ask_step}")
            except Exception as e:
                self.logger.error(f"Error calculating step sizes: {str(e)}")
                # Use reasonable defaults
                bid_step = self.tick_size * 10
                ask_step = self.tick_size * 10
            
            # Dynamic step size adjustment based on volatility
            try:
                # Need to handle possible division by zero
                default_vol = max(TRADING_CONFIG["volatility"]["default"], 0.001)
                # Add additional safeguard against division by zero
                if default_vol <= 0:
                    default_vol = 0.001
                    self.logger.warning("Default volatility was zero or negative, using 0.001 as fallback")
                
                # Ensure volatility is also positive
                volatility = max(volatility, 0.0001)
                
                # Calculate with safeguard against division by zero
                volatility_factor = min(3.0, max(0.5, volatility / default_vol))
                self.logger.debug(f"Volatility factor calculation: {volatility} / {default_vol} = {volatility_factor}")
            except Exception as e:
                self.logger.error(f"Error calculating volatility factor: {str(e)}")
                volatility_factor = 1.0  # Use safe default
            
            # Apply volume candle prediction to step size if available
            try:
                if hasattr(self, 'predictive_adjustments'):
                    vol_adj = self.predictive_adjustments.get('volatility_adjustment', 0.0)
                    if abs(vol_adj) > 0.05:
                        old_factor = volatility_factor
                        volatility_factor *= (1 + vol_adj)
                        self.logger.debug(f"Applied predictive volatility adjustment: {old_factor} * (1 + {vol_adj}) = {volatility_factor}")
            except Exception as e:
                self.logger.error(f"Error applying volume prediction to volatility: {str(e)}")
                # No adjustment needed, continue with existing volatility_factor
            
            # Adjust step size based on volatility factor
            try:
                bid_step *= volatility_factor
                ask_step *= volatility_factor
                self.logger.debug(f"Final grid spacing: bid_step={bid_step:.2f}, ask_step={ask_step:.2f}, volatility factor={volatility_factor:.2f}")
            except Exception as e:
                self.logger.error(f"Error adjusting step sizes: {str(e)}")
                # Reset to reasonable values
                bid_step = self.tick_size * 10
                ask_step = self.tick_size * 10
            
            # Get minimum quote size from config
            try:
                min_size = ORDERBOOK_CONFIG.get("min_size", 0.001)
                self.logger.debug(f"Minimum quote size: {min_size}")
            except Exception as e:
                self.logger.error(f"Error getting minimum quote size: {str(e)}")
                min_size = 0.001  # Safe default
            
            # Get size multipliers for grid levels using Fibonacci sequence
            size_multipliers = self._calculate_fibonacci_size_multipliers(levels)
            
            # Track used prices to avoid duplicates
            bid_price_set = set()
            ask_price_set = set()
            
            # Create bid quotes at multiple levels if we have position to add
            if self.can_increase_position():
                for level in range(levels):
                    # Calculate price for this level (further from mid for higher levels)
                    level_price = self._calculate_level_price(base_bid_price, level, bid_step, is_bid=True)
                    
                    # Calculate minimum price difference based on level - higher levels need more separation
                    min_price_diff = self.tick_size * (1 + level // 2)  # Scales with level
                    
                    # Skip if this price is a duplicate or too close to existing prices
                    duplicate = False
                    closest_existing = None
                    min_distance = float('inf')
                    
                    for existing_price in bid_price_set:
                        distance = abs(level_price - existing_price)
                        if distance < min_distance:
                            min_distance = distance
                            closest_existing = existing_price
                            
                        if distance <= min_price_diff:
                            duplicate = True
                            self.logger.debug(f"Skipping duplicate bid price at level {level}: {level_price} "
                                              f"(too close to {existing_price}, diff={distance}, min required={min_price_diff})")
                            break
                            
                    if duplicate:
                        self.logger.info(f"Level {level} bid price {level_price:.2f} too close to existing price {closest_existing:.2f} "
                                        f"(diff={min_distance:.2f}, min required={min_price_diff:.2f})")
                        continue
                    
                    # Add to set of used prices
                    bid_price_set.add(level_price)
                    
                    # Calculate size for this level (use multiplier)
                    level_size = base_bid_size * size_multipliers[level] if level < len(size_multipliers) else base_bid_size * 0.1
                    
                    # Ensure minimum size (at least 0.01)
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
                self.logger.info("Skipping bid quotes: at position limit or no capacity to buy")
                
            # Create ask quotes at multiple levels if we have position to sell
            if self.can_decrease_position():
                for level in range(levels):
                    # Calculate price for this level (further from mid for higher levels)
                    level_price = self._calculate_level_price(base_ask_price, level, ask_step, is_bid=False)
                    
                    # Calculate minimum price difference based on level - higher levels need more separation
                    min_price_diff = self.tick_size * (1 + level // 2)  # Scales with level
                    
                    # Skip if this price is a duplicate or too close to existing prices
                    duplicate = False
                    closest_existing = None
                    min_distance = float('inf')
                    
                    for existing_price in ask_price_set:
                        distance = abs(level_price - existing_price)
                        if distance < min_distance:
                            min_distance = distance
                            closest_existing = existing_price
                            
                        if distance <= min_price_diff:
                            duplicate = True
                            self.logger.debug(f"Skipping duplicate ask price at level {level}: {level_price} "
                                              f"(too close to {existing_price}, diff={distance}, min required={min_price_diff})")
                            break
                            
                    if duplicate:
                        self.logger.info(f"Level {level} ask price {level_price:.2f} too close to existing price {closest_existing:.2f} "
                                        f"(diff={min_distance:.2f}, min required={min_price_diff:.2f})")
                        continue
                    
                    # Add to set of used prices
                    ask_price_set.add(level_price)
                    
                    # Calculate size for this level (use multiplier)
                    level_size = base_ask_size * size_multipliers[level] if level < len(size_multipliers) else base_ask_size * 0.1
                    
                    # Ensure minimum size (at least 0.01)
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
                f"ref={mid_price:.2f}, spread={spread_display}"
            )
            
            return bid_quotes, ask_quotes
            
        except Exception as e:
            self.logger.error(f"Error generating quotes: {str(e)}")
            return [], []

    def _calculate_level_price(self, base_price: float, level: int, step: float, is_bid: bool) -> float:
        """
        Calculate price for a specific quote level using improved Fibonacci grid spacing
        
        Args:
            base_price: Base price for level 0
            level: Level number (0 = closest to mid)
            step: Step size between levels
            is_bid: True if calculating for bid, False for ask
            
        Returns:
            float: Price for this level
        """
        # Ensure base price is valid
        if base_price <= 0:
            self.logger.warning(f"Invalid base_price {base_price} in _calculate_level_price. Using fallback of 1.0")
            base_price = 1.0
            
        # Ensure tick size is valid    
        if self.tick_size <= 0:
            self.logger.warning(f"Invalid tick_size {self.tick_size} in _calculate_level_price. Using default of 1.0")
            self.tick_size = 1.0
        
        # Early return for level 0
        if level == 0:
            return base_price
        
        # Modified Fibonacci sequence to avoid duplicates at early levels
        # Start at 1, 2, 3, 5, 8, 13, 21, 34, 55, 89
        fib_multipliers = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89]
        
        # Use the appropriate multiplier based on level
        # For levels beyond our pre-computed list, use exponential growth
        if level < len(fib_multipliers):
            fib_multiplier = fib_multipliers[level-1]  # level-1 since we handle level 0
        else:
            # Use much more aggressive exponential growth for higher levels
            # Previously: fib_multipliers[-1] * (1.5 ** (level - len(fib_multipliers)))
            # New: Steeper exponential growth with base 2.0 instead of 1.5
            fib_multiplier = fib_multipliers[-1] * (2.0 ** (level - len(fib_multipliers) + 1))
        
        # Ensure step is positive
        if step <= 0:
            self.logger.warning(f"Invalid step size {step} in _calculate_level_price. Using default of {self.tick_size * 10}")
            step = self.tick_size * 10
            
        # Adjust step size with Fibonacci multiplier
        fib_step = step * fib_multiplier
        
        # Ensure minimum step distance is at least 2 ticks to avoid duplicates
        min_step = 2 * self.tick_size
        if fib_step < min_step:
            fib_step = min_step
        
        # Add additional spacing between levels based on level number to prevent duplicates
        # Use exponential spacing for higher levels to prevent duplicates
        if level <= 10:
            level_spacing_factor = 1 + (level * 0.05)  # 5% increase per level for first 10 levels
        else:
            # More aggressive spacing for higher levels (15% increase per level after level 10)
            level_spacing_factor = 1.5 + ((level - 10) * 0.15)
        
        fib_step *= level_spacing_factor
        
        # Ensure the step is a multiple of tick size to avoid rounding issues
        # Use a higher multiple for higher levels to prevent rounding to the same price
        tick_multiple = max(2, level // 2)  # At least 2, increases with level
        
        # Prevent division by zero in the calculation below
        if self.tick_size * tick_multiple <= 0:
            self.logger.warning(f"Invalid tick multiple calculation: tick_size={self.tick_size}, multiple={tick_multiple}")
            tick_multiple = 2  # Safe default
            
        fib_step = max(round(fib_step / (self.tick_size * tick_multiple)) * (self.tick_size * tick_multiple), min_step)
        
        # Bids get lower as level increases, asks get higher
        if is_bid:
            new_price = base_price - fib_step
            # Ensure we don't go negative or too close to zero
            if new_price <= self.tick_size * 2:
                new_price = self.tick_size * 2
            return self.align_price_to_tick(new_price)
        else:
            new_price = base_price + fib_step
            return self.align_price_to_tick(new_price)

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
            position_check_interval = TRADING_CONFIG["quoting"].get("position_check_interval", 5.0)
            significant_position_threshold = 0.05
            
            if abs(self.position_tracker.current_position) > significant_position_threshold and current_time - self.last_position_check > position_check_interval:
                self.last_position_check = current_time
                self.logger.info(f"Should update quotes: Position check with significant position {self.position_tracker.current_position:.2f}")
                return True
                
            # 5. Check for forced regular updates (ensure quotes refresh periodically)
            min_update_interval = ORDERBOOK_CONFIG.get("min_quote_interval", 2.0)
            max_update_interval = min(60.0, quote_lifetime * 0.8)  # 80% of quote lifetime or 60s max
            
            # More frequent updates with larger position
            if abs(self.position_tracker.current_position) > 0:
                # Scale update frequency based on position size
                position_scale_factor = min(1.0, abs(self.position_tracker.current_position) / TRADING_CONFIG["avellaneda"]["position_limit"])
                # More frequent updates as position grows (interval between min and max)
                update_interval = max_update_interval - (max_update_interval - min_update_interval) * position_scale_factor
                
                if time_elapsed > update_interval:
                    self.logger.info(
                        f"Should update quotes: Position-scaled interval reached - "
                        f"Elapsed: {time_elapsed:.1f}s, Interval: {update_interval:.1f}s, Position: {self.position_tracker.current_position:.3f}"
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
            if 'avellaneda' in TRADING_CONFIG: # Check if 'avellaneda' key exists
                min_quote_size = max(0.01, TRADING_CONFIG["avellaneda"].get('base_size', 0.01))
        except (KeyError, TypeError):
            self.logger.warning("Could not get min_size from config, using default of 0.01")
        
        # Calculate maximum price deviation from ticker (5% by default, increased from 3%)
        max_deviation_pct = 0.05
        
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
            self.logger.warning(f"Invalid tick size ({self.tick_size}) in round_to_tick. Using default of 1.0")
            self.tick_size = 1.0
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
            # self.position_size = self.position_tracker.current_position # REMOVED - No longer needed as direct access will use position_tracker
            # self.entry_price = self.position_tracker.average_entry_price # REMOVED - No longer needed as direct access will use position_tracker
            
            # Update VAMP calculations
            self.update_vamp(fill_price, fill_size, is_buy, False)
            
            # NEW: Update volume candle buffer with fill data
            self.volume_buffer.update(fill_price, fill_size, is_buy, int(time.time() * 1000))
            
            # Update monetary position (which will trigger hedge updates if needed)
            # self.update_monetary_position() # REMOVED - Method was deleted
            
            # Check if we're using the hedge manager and have a valid instrument
            if self.use_hedging and self.hedge_manager is not None:
                # Ensure we have a valid instrument before processing for hedging
                if not self.instrument or self.instrument == "unknown":
                    # Try one more time to determine the instrument from exchange client
                    if self.exchange_client and hasattr(self.exchange_client, 'get_current_instrument'):
                        current_instrument = self.exchange_client.get_current_instrument()
                        if current_instrument:
                            self.instrument = current_instrument
                            self.logger.info(f"Setting instrument from current exchange context: {self.instrument}")
                        else:
                            self.logger.error(f"No instrument set for fill - cannot process trade. Aborting fill processing.")
                            return  # Stop processing if no instrument is set
                    else:
                        self.logger.error(f"No instrument set for fill and no way to determine it - cannot process trade. Aborting fill processing.")
                        return  # Stop processing if no instrument is set
                
                # Now process the fill for hedging
                self._process_fill_for_hedging(order_id, fill_price, fill_size, is_buy)
            
            self.logger.info(
                f"Order filled: {fill_size} @ {fill_price} ({'BUY' if is_buy else 'SELL'})"
                f" - New position: {self.position_tracker.current_position:.4f}, Avg entry: {self.position_tracker.average_entry_price:.2f}" # MODIFIED
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
            # Make sure we have a valid instrument
            if not self.instrument or self.instrument == "unknown":
                # Try one more time to determine the instrument from exchange client
                if self.exchange_client and hasattr(self.exchange_client, 'get_current_instrument'):
                    current_instrument = self.exchange_client.get_current_instrument()
                    if current_instrument:
                        self.instrument = current_instrument
                        self.logger.info(f"Setting instrument from current exchange context: {self.instrument}")
                    else:
                        self.logger.error(f"No instrument set for fill - cannot process trade. Aborting fill processing.")
                        return  # Stop processing if no instrument is set
                else:
                    self.logger.error(f"No instrument set for fill and no way to determine it - cannot process trade. Aborting fill processing.")
                    return  # Stop processing if no instrument is set
            
            # Create a fill object suitable for the hedge manager
            class HedgeFill:
                def __init__(self, instrument, price, size, is_buy, order_id):
                    self.instrument = instrument
                    self.price = price
                    self.size = size
                    self.is_buy = is_buy
                    self.fill_id = order_id
            
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
            'position': self.position_tracker.current_position,
            'entry_price': self.position_tracker.average_entry_price,
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
        # Ensure price is positive
        if price <= 0:
            self.logger.warning(f"Invalid price {price} in align_price_to_tick. Using minimum tick size.")
            return self.tick_size  # Return minimum valid price
            
        if not self.tick_size or self.tick_size <= 0:
            self.logger.warning("Invalid tick size, using default alignment")
            # Store previous tick size for logging
            old_tick_size = self.tick_size
            # Set a valid tick size for future use
            self.tick_size = 1.0
            self.logger.info(f"Updated tick size from {old_tick_size} to {self.tick_size}")
            return round(price, 2)  # Default to 2 decimal places
            
        # Round to nearest tick
        aligned_price = self.tick_size * round(price / self.tick_size)
        
        # Ensure we have at least minimum price
        min_price = self.tick_size
        aligned_price = max(aligned_price, min_price)
        
        return aligned_price

    def set_instrument(self, instrument: str):
        """Set the instrument being traded"""
        self.instrument = instrument
        self.logger.info(f"Setting instrument to {instrument}")
        
        # Update instrument in volume candle buffer if available
        if hasattr(self, 'volume_buffer') and self.volume_buffer:
            self.volume_buffer.instrument = instrument
            self.logger.info(f"Updated volume candle buffer instrument to {instrument}")

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
            volatility_factor = volatility / max(TRADING_CONFIG["volatility"]["default"], 0.001)
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
                position_utilization = min(1.0, abs(self.position_tracker.current_position) / self.position_limit)
                
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
                if self.position_tracker.unrealized_pnl is not None and self.position_tracker.current_value_usd != 0: # MODIFIED (checking current_value_usd != 0)
                    pnl_percentage = self.position_tracker.unrealized_pnl / self.position_tracker.current_value_usd # MODIFIED
                    
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
            if self.position_tracker.current_position != 0 and self.position_tracker.position_start_time is not None: # MODIFIED
                position_duration = time.time() - self.position_tracker.position_start_time # MODIFIED
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
                f"  - Position:            {self.position_tracker.current_position if hasattr(self, 'position_tracker') else 0} (factor: {position_factor:.2f})\n"
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
        """Update market data with new price and potentially volume information"""
        current_time = time.time()
        
        # Always update price history
        if price > 0:
            self.price_history.append(price)
            if not self.last_mid_price:
                self.last_mid_price = price
            
            # Only update mid price if it's a significant change (avoid noise)
            if abs(price - self.last_mid_price) / self.last_mid_price > 0.0001:
                self.last_mid_price = price
        
        # Update volume-based indicators if trade data is provided
        if is_trade and volume > 0 and is_buy is not None:
            buy_sell_str = "BUY" if is_buy else "SELL"
            self.logger.debug(f"Processing trade: {buy_sell_str} {volume:.6f} @ {price:.2f}")
            
            # Update VAMP
            self.update_vamp(price, volume, is_buy)
            
            # Update volume candle buffer if available
            if self.volume_buffer:
                # Convert to milliseconds for the volume candle buffer
                timestamp = int(current_time * 1000)
                
                # Log details about the trade being added to candle
                self.logger.debug(
                    f"Adding trade to volume candle: {buy_sell_str} {volume:.6f} @ {price:.2f}, "
                    f"current candle volume: {getattr(self.volume_buffer.current_candle, 'volume', 0):.6f}/{self.volume_buffer.volume_threshold:.6f}"
                )
                
                # Update the candle with trade data
                candle = self.volume_buffer.update(price, volume, is_buy, timestamp)
                
                # If a candle was completed, update predictive parameters
                if candle and candle.is_complete:
                    self.logger.info(
                        f"Volume candle completed: O:{candle.open_price:.2f}, H:{candle.high_price:.2f}, "
                        f"L:{candle.low_price:.2f}, C:{candle.close_price:.2f}, V:{candle.volume:.6f}, "
                        f"Δ:{candle.delta_ratio:.2f}, triggering parameter update"
                    )
                    self._update_predictive_parameters()
                    self.force_grid_update = True  # Force grid update when new candle is complete
                    self.logger.info("Forcing grid update due to new candle completion")
        
        # Periodically update predictive parameters
        if self.volume_buffer and current_time - self.last_prediction_time > self.prediction_interval:
            self.logger.info(f"Periodic prediction update triggered after {current_time - self.last_prediction_time:.1f}s")
            self._update_predictive_parameters()
            self.last_prediction_time = current_time
            
        # Check if we should update the grid (based on time or forced update)
        if self.force_grid_update or current_time - self.last_grid_update_time > self.grid_update_interval:
            self.last_grid_update_time = current_time
            self.force_grid_update = False
            
            # Log the update
            update_reason = "forced update" if self.force_grid_update else f"time-based update ({self.grid_update_interval}s interval)"
            self.logger.debug(f"Triggering grid update due to {update_reason}")
            
            # This will signal that grids should be updated on the next quote generation
            return True
        
        return False

    def _update_predictive_parameters(self) -> None:
        """Update predictive parameters from volume candle buffer"""
        if not self.volume_buffer:
            return
        
        try:
            # Get predictions from volume candle buffer
            predictions = self.volume_buffer.get_predicted_parameters()
            
            # Always log that we're checking predictions
            self.logger.info(f"Checking volume candle predictions at {time.strftime('%H:%M:%S')}")
            
            if predictions:
                # Log all raw prediction values
                self.logger.info(
                    f"Raw predictions: gamma_adj={predictions.get('gamma_adjustment', 0):.3f}, "
                    f"kappa_adj={predictions.get('kappa_adjustment', 0):.3f}, "
                    f"res_price_offset={predictions.get('reservation_price_offset', 0):.6f}, "
                    f"trend_dir={predictions.get('trend_direction', 0)}, "
                    f"vol_adj={predictions.get('volatility_adjustment', 0):.3f}"
                )
                
                # Store the adjustments for use in other components
                self.predictive_adjustments = predictions
                self.predictive_adjustments["last_update_time"] = time.time()
                
                # Update Avellaneda parameters based on predictions
                # Track changes to summarize at the end
                changes_made = []
                
                # Fetch thresholds from config
                pa_gamma_adj_threshold = TRADING_CONFIG["avellaneda"].get("pa_gamma_adj_threshold", 0.05)
                pa_kappa_adj_threshold = TRADING_CONFIG["avellaneda"].get("pa_kappa_adj_threshold", 0.05)
                # As discussed, using the same threshold as calculate_optimal_spread/calculate_skewed_prices for consistency here.
                pa_res_price_offset_adj_threshold = TRADING_CONFIG["avellaneda"].get("pa_res_price_offset_adj_threshold", 0.00005) 
                pa_volatility_adj_threshold = TRADING_CONFIG["avellaneda"].get("pa_volatility_adj_threshold", 0.05)

                # 1. Adjust gamma (risk aversion) based on prediction
                gamma_adjustment = predictions.get("gamma_adjustment", 0)
                if abs(gamma_adjustment) > pa_gamma_adj_threshold:  # Only apply significant adjustments
                    old_gamma = self.gamma
                    new_gamma = self.gamma * (1 + gamma_adjustment)
                    self.update_gamma(new_gamma)
                    changes_made.append(f"gamma: {old_gamma:.3f} → {new_gamma:.3f} (adj: {gamma_adjustment:+.3f})")
                else:
                    self.logger.debug(f"Skipping gamma adjustment: {gamma_adjustment:.3f} (below threshold {pa_gamma_adj_threshold:.2f})")
                    
                # 2. Adjust kappa (market depth) based on prediction
                kappa_adjustment = predictions.get("kappa_adjustment", 0)
                old_kappa = self.kappa
                if abs(kappa_adjustment) > pa_kappa_adj_threshold:  # Only apply significant adjustments
                    self.kappa = self.k_default * (1 + kappa_adjustment)
                    changes_made.append(f"kappa: {old_kappa:.3f} → {self.kappa:.3f} (adj: {kappa_adjustment:+.3f})")
                else:
                    self.logger.debug(f"Skipping kappa adjustment: {kappa_adjustment:.3f} (below threshold {pa_kappa_adj_threshold:.2f})")
                    
                # 3. Set reservation price offset based on prediction
                old_offset = getattr(self, 'reservation_price_offset', 0)
                self.reservation_price_offset = predictions.get("reservation_price_offset", 0)
                if abs(self.reservation_price_offset) > pa_res_price_offset_adj_threshold: # Changed from 0.00001 to configured value
                    changes_made.append(f"res_price_offset: {old_offset:.6f} → {self.reservation_price_offset:.6f}")
                else:
                    self.logger.debug(f"Reservation price offset: {self.reservation_price_offset:.6f} (minor adjustment, below threshold {pa_res_price_offset_adj_threshold:.6f})")
                    
                # 4. Track trend direction for grid spacing adjustment
                old_trend = getattr(self, 'trend_direction', 0)
                trend_direction = predictions.get("trend_direction", 0)
                if trend_direction != old_trend:
                    self.trend_direction = trend_direction
                    changes_made.append(f"trend_direction: {old_trend} → {trend_direction}")
                
                # 5. Apply volatility adjustment
                vol_adjustment = predictions.get("volatility_adjustment", 0)
                old_vol = self.volatility
                if abs(vol_adjustment) > pa_volatility_adj_threshold:
                    adjusted_vol = self.volatility * (1 + vol_adjustment)
                    # Clamp to reasonable bounds
                    adjusted_vol = max(min(adjusted_vol, TRADING_CONFIG["volatility"]["ceiling"]), TRADING_CONFIG["volatility"]["floor"])
                    self.volatility = adjusted_vol
                    changes_made.append(f"volatility: {old_vol:.4f} → {adjusted_vol:.4f} (adj: {vol_adjustment:+.3f})")
                else:
                    self.logger.debug(f"Skipping volatility adjustment: {vol_adjustment:.3f} (below threshold {pa_volatility_adj_threshold:.2f})")
                
                # Log the signals from volume candles
                signals = self.volume_buffer.signals if hasattr(self.volume_buffer, 'signals') else {}
                if signals:
                    self.logger.info(
                        f"Signal basis: momentum={signals.get('momentum', 0):.2f}, "
                        f"reversal={signals.get('reversal', 0):.2f}, "
                        f"volatility={signals.get('volatility', 0):.2f}, "
                        f"exhaustion={signals.get('exhaustion', 0):.2f}"
                    )
                
                # Log a summary of all changes
                if changes_made:
                    self.logger.info(f"Applied {len(changes_made)} prediction-based adjustments: {', '.join(changes_made)}")
                else:
                    self.logger.info("No significant parameter adjustments needed based on predictions")
                    
                # Log the current state of all key parameters
                self.logger.info(
                    f"Current parameters: gamma={self.gamma:.3f}, kappa={self.kappa:.3f}, "
                    f"res_price_offset={self.reservation_price_offset:.6f}, "
                    f"vol={self.volatility:.4f}, trend={getattr(self, 'trend_direction', 0)}"
                )
                    
        except Exception as e:
            self.logger.error(f"Error updating predictive parameters: {str(e)}", exc_info=True)

    def _calculate_fibonacci_size_multipliers(self, levels: int) -> List[float]:
        """
        Calculate size multipliers based on Fibonacci ratios
        
        Args:
            levels: Number of quote levels to generate
            
        Returns:
            List of size multipliers for each level
        """
        # Use the configured size multipliers directly from ORDERBOOK_CONFIG
        configured_multipliers = ORDERBOOK_CONFIG.get("bid_sizes", [0.3, 0.5, 0.7, 0.9, 1.0, 1.2])
        
        # If we don't need many levels, just return the configured values
        if levels <= len(configured_multipliers):
            return configured_multipliers[:levels]
        
        # For additional levels beyond config, use Fibonacci sequence
        fib_sequence = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89]
        
        # Initialize multipliers with configured values
        fib_multipliers = configured_multipliers.copy()
        
        # Calculate scaling factor to ensure smooth transition
        base_scale = configured_multipliers[-1] / 0.7  # Scale to match the last configured multiplier
        
        # Add additional multipliers for levels beyond configuration
        for i in range(len(configured_multipliers), levels):
            idx = min(i - len(configured_multipliers) + 2, len(fib_sequence) - 1)
            fib_value = fib_sequence[idx]
            # Scale based on both the fibonacci value and base scale
            multiplier = base_scale * (fib_value / 8)  # Normalized by fib(6) = 8
            fib_multipliers.append(round(multiplier, 2))
        
        # Make sure the base level is not too small
        if fib_multipliers and fib_multipliers[0] < 0.1:
            fib_multipliers[0] = 0.1
        
        # Ensure we have exactly the number of levels requested
        return fib_multipliers[:levels]

    def cleanup(self):
        """Clean up resources when shutting down"""
        self.logger.info("Cleaning up Avellaneda market maker resources")
        
        # Clean up hedge manager if active
        if self.use_hedging and self.hedge_manager:
            self.hedge_manager.stop()
            
        # Clean up volume candle buffer
        if hasattr(self, 'volume_buffer') and self.volume_buffer:
            if hasattr(self.volume_buffer, 'stop'):
                self.volume_buffer.stop()
                
        self.logger.info("Cleanup complete")

    def report_hedge_status(self):
        """Report on monetary position and hedge status"""
        if not self.use_hedging or self.hedge_manager is None:
            return
        
        # Calculate and log monetary position
        monetary_position = self.position_tracker.current_position * self.last_mid_price
        
        # Get hedge position
        hedge_position = self.hedge_manager.get_hedged_position(self.instrument)
        if hedge_position:
            hedge_monetary = hedge_position.hedge_position * hedge_position.hedge_price
            net_monetary = monetary_position + hedge_monetary
            
            self.logger.info(f"=== HEDGE STATUS ===")
            self.logger.info(f"Primary: {self.position_tracker.current_position} {self.instrument} @ {self.last_mid_price:.2f} = ${monetary_position:.2f}")
            self.logger.info(f"Hedge: {hedge_position.hedge_position} {hedge_position.hedge_asset} @ {hedge_position.hedge_price:.2f} = ${hedge_monetary:.2f}")
            self.logger.info(f"Net monetary exposure: ${net_monetary:.2f}")
            if monetary_position != 0:
                self.logger.info(f"Hedge ratio: {abs(hedge_monetary/monetary_position):.2f} (target: {hedge_position.hedge_ratio:.2f})")
            self.logger.info(f"Current P&L: ${hedge_position.pnl:.2f}")
        else:
            self.logger.info(f"No active hedge for {self.instrument} position (${monetary_position:.2f})")
