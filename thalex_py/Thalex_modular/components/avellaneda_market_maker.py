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
from ..models.position_tracker import PositionTracker, PortfolioTracker, Fill
from ..thalex_logging import LoggerFactory
from ..ringbuffer.volume_candle_buffer import VolumeBasedCandleBuffer

# Constants - Pre-computed for performance
DEFAULT_RESERVATION_PRICE_BOUND = TRADING_CONFIG["avellaneda"].get("reservation_price_bound", 0.005)
MIN_SPREAD_TICK_MULTIPLIER = 2  # Minimum spread as a multiple of tick size
SQRT_2PI = math.sqrt(2 * math.pi)  # Pre-computed constant
LOG_SAMPLING_RATE = 0.05  # Log only 5% of operations for performance

class AvellanedaMarketMaker:
    """Avellaneda-Stoikov market making strategy implementation"""
    
    # Use __slots__ for memory efficiency while maintaining all functionality
    __slots__ = [
        'logger', 'exchange_client', 'position_tracker', 'gamma', 'k_default', 'kappa',
        'volatility', 'tick_size', 'time_horizon', 'reservation_price_offset',
        'quote_count', 'quote_levels', 'instrument', 'last_mid_price', 'last_update_time',
        'volatility_grid', 'price_history', 'vamp_window', 'vamp_aggressive_window',
        'vamp_buy_volume', 'vamp_sell_volume', 'vamp_value', 'vamp_impact', 'vamp_impact_window',
        'volume_buffer', 'predictive_adjustments', 'last_prediction_time', 'prediction_interval',
        'last_grid_update_time', 'grid_update_interval', 'force_grid_update',
        'inventory_weight', 'position_fade_time', 'order_flow_intensity', 'position_limit',
        'base_spread_factor', 'market_impact_factor', 'inventory_factor', 'volatility_multiplier',
        'market_state', 'last_quote_time', 'min_tick', 'last_position_check', 'last_quote_update',
        'vwap', 'total_volume', 'volume_price_sum', 'market_buys_volume', 'market_sells_volume',
        'aggressive_buys_sum', 'aggressive_sells_sum', 'market_impact', 'active_quotes',
        'current_bid_quotes', 'current_ask_quotes',
        # Performance optimization caches
        '_cached_spread', '_cached_mid_price', '_cached_volatility_factor', '_cached_position', '_cache_timestamp',
        '_log_counter', '_last_major_log',
        # Take profit trigger order system
        'active_trigger_orders', 'take_profit_spread_bps', 'trigger_order_enabled',
        'last_trigger_update', 'pending_trigger_cancellations',
        # Multi-instrument support
        'portfolio_tracker',
        # Multi-instrument arbitrage take profit system
        'arbitrage_trigger_enabled', 'arbitrage_profit_threshold_usd', 'single_instrument_profit_threshold_usd',
        'spread_profit_threshold_bps', 'last_arbitrage_check', 'arbitrage_check_interval',
        # Trading mode detection
        'trading_mode', 'last_mode_detection', 'mode_detection_interval',
        # Instrument configurations
        'perp_instrument', 'futures_instrument', 'perp_tick_size', 'futures_tick_size',
        # Predictive trend tracking
        'trend_direction'
    ]
    
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
        # Portfolio tracker for multi-instrument support (will be set by quoter)
        self.portfolio_tracker = None
        
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
        
        # VAMP (Volume Adjusted Market Pressure) tracking - using defaults
        vamp_config = TRADING_CONFIG.get("vamp", {})
        self.vamp_window = deque(maxlen=vamp_config.get("window", 50))
        self.vamp_aggressive_window = deque(maxlen=vamp_config.get("aggressive_window", 20))
        self.vamp_buy_volume = 0.0
        self.vamp_sell_volume = 0.0
        self.vamp_value = 0.0
        self.vamp_impact = 0.0
        self.vamp_impact_window = deque(maxlen=vamp_config.get("impact_window", 30))
        
        # Volume candle buffer - will be shared from quoter to avoid duplication
        self.volume_buffer = None  # Will be set by quoter
        
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

        # Performance optimization: Initialize caches
        self._cached_spread = 0.0
        self._cached_mid_price = 0.0
        self._cached_volatility_factor = 1.0
        self._cached_position = 0.0
        self._cache_timestamp = 0.0
        self._log_counter = 0
        self._last_major_log = 0.0

        # Enhanced multi-instrument take profit trigger order system
        self.active_trigger_orders = {}  # Dict[str, Quote] - order_id -> trigger order
        self.take_profit_spread_bps = TRADING_CONFIG["avellaneda"].get("take_profit_spread_bps", 7.0)  # 6-8 bps default
        self.trigger_order_enabled = TRADING_CONFIG["avellaneda"].get("enable_take_profit_triggers", True)
        self.last_trigger_update = 0.0
        self.pending_trigger_cancellations = set()  # Set of order IDs pending cancellation

        # NEW: Multi-instrument arbitrage take profit system
        self.arbitrage_trigger_enabled = TRADING_CONFIG["avellaneda"].get("enable_arbitrage_triggers", True)
        self.arbitrage_profit_threshold_usd = TRADING_CONFIG["avellaneda"].get("arbitrage_profit_threshold_usd", 10.0)
        self.single_instrument_profit_threshold_usd = TRADING_CONFIG["avellaneda"].get("single_instrument_profit_threshold_usd", 5.0)
        self.spread_profit_threshold_bps = TRADING_CONFIG["avellaneda"].get("spread_profit_threshold_bps", 15.0)  # 15 bps for spread profits
        self.last_arbitrage_check = 0.0
        self.arbitrage_check_interval = TRADING_CONFIG["avellaneda"].get("arbitrage_check_interval", 2.0)  # Check every 2 seconds
        
        # Trading mode detection
        self.trading_mode = "unknown"  # "single_perp", "single_futures", "arbitrage", "unknown"
        self.last_mode_detection = 0.0
        self.mode_detection_interval = 5.0  # Check trading mode every 5 seconds
        
        # Instrument configurations (will be updated by quoter)
        self.perp_instrument = None
        self.futures_instrument = None
        self.perp_tick_size = 1.0
        self.futures_tick_size = 1.0

    def set_tick_size(self, tick_size: float):
        """Set the tick size for the instrument"""
        self.tick_size = tick_size
        self.min_tick = tick_size
        self.logger.info(f"Tick size set to {tick_size}")

    def _get_default_tick_size(self) -> float:
        """Get appropriate default tick size based on instrument type"""
        # Check if we have instrument information to determine proper tick size
        if hasattr(self, 'instrument') and self.instrument:
            instrument_name = str(self.instrument).upper()
            if 'BTC' in instrument_name:
                return 1.0  # BTC derivatives typically have 1 USD tick size
            elif 'ETH' in instrument_name:
                return 0.1  # ETH derivatives typically have 0.1 USD tick size
            else:
                return 1.0  # Default for unknown instruments
        
        # Check if we have perp or futures instrument names
        if hasattr(self, 'perp_instrument') and self.perp_instrument:
            if 'BTC' in str(self.perp_instrument).upper():
                return 1.0
            elif 'ETH' in str(self.perp_instrument).upper():
                return 0.1
        
        if hasattr(self, 'futures_instrument') and self.futures_instrument:
            if 'BTC' in str(self.futures_instrument).upper():
                return 1.0
            elif 'ETH' in str(self.futures_instrument).upper():
                return 0.1
        
        # Fallback to BTC default
        return 1.0

    def update_market_conditions(self, volatility: float, market_impact: float):
        """Update the market conditions used for quote generation"""
        self.volatility = volatility
        self.market_impact = market_impact
        if random.random() < LOG_SAMPLING_RATE:
            self.logger.debug(f"Market conditions updated: vol={self.volatility:.6f}, impact={market_impact:.6f}")
    
    def update_vamp(self, price: float, volume: float, is_buy: bool, is_aggressive: bool=False):
        """Update Volume-Adjusted Market Price calculations with improved edge case handling"""
        # Input validation to prevent corrupted VAMP calculations
        if not isinstance(price, (int, float)) or not np.isfinite(price) or price <= 0:
            self.logger.warning(f"Invalid price {price} in VAMP update, skipping")
            return self.vamp_value if hasattr(self, 'vamp_value') else 0.0
            
        if not isinstance(volume, (int, float)) or not np.isfinite(volume) or volume <= 0:
            self.logger.warning(f"Invalid volume {volume} in VAMP update, skipping")
            return self.vamp_value if hasattr(self, 'vamp_value') else 0.0
        
        if random.random() < LOG_SAMPLING_RATE:
            side_str = "BUY" if is_buy else "SELL"
            agg_str = "aggressive" if is_aggressive else "passive"
            self.logger.debug(f"Updating VAMP with {side_str} {volume:.6f} @ {price:.2f} ({agg_str})")
        
        old_vwap = getattr(self, 'vwap', 0.0)
        old_vamp_value = getattr(self, 'vamp_value', 0.0)
        
        # Initialize attributes safely if they don't exist
        if not hasattr(self, 'volume_price_sum'):
            self.volume_price_sum = 0.0
        if not hasattr(self, 'total_volume'):
            self.total_volume = 0.0
        
        # Update VWAP (Volume-Weighted Average Price) with overflow protection
        price_volume = price * volume
        if not np.isfinite(price_volume):
            self.logger.warning(f"Price*volume overflow: {price}*{volume}, skipping VAMP update")
            return old_vamp_value
            
        self.volume_price_sum += price_volume
        self.total_volume += volume
        
        # Prevent accumulation of excessive data that could cause precision issues
        max_total_volume = 1e6  # Reset after 1M volume to maintain precision
        if self.total_volume > max_total_volume:
            # Scale down proportionally to maintain VWAP accuracy
            scale_factor = 0.5
            self.volume_price_sum *= scale_factor
            self.total_volume *= scale_factor
            self.market_buys_volume *= scale_factor
            self.market_sells_volume *= scale_factor
            self.aggressive_buys_sum *= scale_factor
            self.aggressive_sells_sum *= scale_factor
            if random.random() < LOG_SAMPLING_RATE:
                self.logger.info(f"VAMP data scaled down by {scale_factor} to maintain precision")
        
        if self.total_volume > 0:
            self.vwap = self.volume_price_sum / self.total_volume
            if random.random() < LOG_SAMPLING_RATE:
                self.logger.debug(f"VWAP updated: {old_vwap:.2f} → {self.vwap:.2f} (Δ: {self.vwap - old_vwap:+.2f})")
        
        # Update aggressive volume tracking
        if is_aggressive:
            if is_buy:
                old_buys = self.market_buys_volume
                self.market_buys_volume += volume
                self.aggressive_buys_sum += price * volume
                if random.random() < LOG_SAMPLING_RATE:
                    self.logger.debug(f"Aggressive buys: {old_buys:.6f} → {self.market_buys_volume:.6f} (Δ: +{volume:.6f})")
            else:
                old_sells = self.market_sells_volume
                self.market_sells_volume += volume
                self.aggressive_sells_sum += price * volume
                if random.random() < LOG_SAMPLING_RATE:
                    self.logger.debug(f"Aggressive sells: {old_sells:.6f} → {self.market_sells_volume:.6f} (Δ: +{volume:.6f})")
            
            # Record impact in the impact window
            current_time = time.time()
            impact_value = volume * (1 if is_buy else -1)
            self.vamp_impact_window.append((current_time, impact_value))
            
            # Calculate time-weighted impact
            self.vamp_impact = self._calculate_time_weighted_impact()
            self.logger.debug(f"VAMP impact updated: {self.vamp_impact:.4f} (window size: {len(self.vamp_impact_window)})")
        
        # Calculate VAMP
        vamp = self.calculate_vamp()
        vamp_change = vamp - old_vamp_value
        self.vamp_value = vamp
        
        self.logger.info(
            f"VAMP updated: {old_vamp_value:.2f} → {vamp:.2f} (Δ: {vamp_change:+.2f}), "
            f"VWAP: {self.vwap:.2f}, Total Vol: {self.total_volume:.6f}, "
            f"Buy/Sell Ratio: {self._get_buy_sell_ratio():.2f}, Impact: {self.vamp_impact:.4f}"
        )
        
        return vamp
    
    def _get_buy_sell_ratio(self) -> float:
        """Calculate buy/sell ratio for logging"""
        if self.market_buys_volume + self.market_sells_volume == 0:
            return 1.0
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
        vamp_config = TRADING_CONFIG.get("vamp", {})
        max_age = vamp_config.get("max_impact_age", 300)
        
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
        # Performance: Check cache first to avoid expensive recalculation
        current_time = time.time()
        # Cache is only valid if key parameters haven't changed significantly
        try:
            current_position = getattr(self.position_tracker, 'current_position', 0.0)
            cache_valid = (
                current_time - self._cache_timestamp < 0.1 and  # Time-based expiry
                abs(market_impact) < 0.001 and  # Market impact hasn't changed much
                self._cached_spread > 0 and  # Valid cached value exists
                abs(self.volatility - getattr(self, '_cached_volatility_factor', self.volatility)) < 0.001 and  # Volatility stable
                abs(current_position - getattr(self, '_cached_position', current_position)) < 0.001  # Position stable
            )
        except Exception:
            # If cache validation fails, invalidate cache
            cache_valid = False
        
        if cache_valid:
            return self._cached_spread
            
        try:
            min_spread_ticks = ORDERBOOK_CONFIG["min_spread"]
            
            if self.tick_size <= 0:
                # Use proper tick size based on instrument type
                default_tick_size = self._get_default_tick_size()
                self.logger.warning(f"Invalid tick_size detected in calculate_optimal_spread. Using instrument default: {default_tick_size}")
                self.tick_size = default_tick_size
                
            min_spread = min_spread_ticks * self.tick_size
            
            # Calculate minimum spread needed for profitability
            trading_fees_config = TRADING_CONFIG.get("trading_fees", {})
            maker_fee_rate = trading_fees_config.get("maker_fee_rate", 0.0002)
            profit_margin_rate = trading_fees_config.get("profit_margin_rate", 0.0005)
            fee_coverage_multiplier = trading_fees_config.get("fee_coverage_multiplier", 1.2)
            
            if hasattr(self, 'last_mid_price') and self.last_mid_price > 0:
                reference_price = self.last_mid_price
            else:
                reference_price = 84000  # Fallback estimate for BTC
                
            fee_based_min_spread = reference_price * (maker_fee_rate * 2 + profit_margin_rate) * fee_coverage_multiplier
            fee_coverage_spread = max(min_spread, fee_based_min_spread)
            
            self.logger.debug(f"Fee calculations: reference_price={reference_price:.2f}, "
                             f"fee_min_spread=${fee_based_min_spread:.2f}, "
                             f"config_min_spread=${min_spread:.2f}, "
                             f"final_fee_coverage=${fee_coverage_spread:.2f}, "
                             f"coverage_multiplier={fee_coverage_multiplier:.2f}")
            
            volatility = max(self.volatility, 0.01)
            
            # Apply predictive volatility adjustment
            prediction_age = time.time() - self.predictive_adjustments.get("last_update_time", 0)
            original_volatility = volatility
            
            pa_prediction_max_age_seconds = TRADING_CONFIG["avellaneda"].get("pa_prediction_max_age_seconds", 300)
            pa_volatility_adj_threshold = TRADING_CONFIG["avellaneda"].get("pa_volatility_adj_threshold", 0.05)

            if prediction_age < pa_prediction_max_age_seconds:
                volatility_adjustment = self.predictive_adjustments.get("volatility_adjustment", 0)
                if abs(volatility_adjustment) > pa_volatility_adj_threshold:
                    volatility *= (1 + volatility_adjustment)
                    self.logger.debug(f"Adjusted volatility: {original_volatility:.4f} -> {volatility:.4f} (factor: {1 + volatility_adjustment:.2f}, threshold: {pa_volatility_adj_threshold:.2f})")
            
            # Base Avellaneda-Stoikov spread calculation components
            base_gamma = self.gamma
            
            if base_gamma <= 0:
                self.logger.warning(f"Invalid gamma value {base_gamma}, using default of 0.1")
                base_gamma = 0.1
            
            # Apply predictive gamma adjustment
            original_gamma = base_gamma
            pa_gamma_adj_threshold = TRADING_CONFIG["avellaneda"].get("pa_gamma_adj_threshold", 0.05)

            if prediction_age < pa_prediction_max_age_seconds:
                gamma_adjustment = self.predictive_adjustments.get("gamma_adjustment", 0)
                if abs(gamma_adjustment) > pa_gamma_adj_threshold:
                    base_gamma = max(0.1, base_gamma * (1 + gamma_adjustment))
                    self.logger.debug(f"Adjusted gamma: {original_gamma:.3f} -> {base_gamma:.3f} (factor: {1 + gamma_adjustment:.2f}, threshold: {pa_gamma_adj_threshold:.2f})")
            
            gamma_component = 1.0 + base_gamma
            volatility_term = volatility * self.volatility_multiplier
            
            if not hasattr(self, 'position_fade_time') or self.position_fade_time <= 0:
                self.position_fade_time = TRADING_CONFIG["avellaneda"].get("position_fade_time", 3600)
                self.logger.warning(f"Invalid position_fade_time, using default: {self.position_fade_time}")
                
            volatility_component = volatility_term * math.sqrt(self.position_fade_time)
            
            market_impact_arg = max(0, market_impact)
            
            # VAMP integration for spread adjustment
            vamp_spread_sensitivity = TRADING_CONFIG["avellaneda"].get("vamp_spread_sensitivity", 0.0)
            vamp_effect_on_spread = abs(self.vamp_impact) * vamp_spread_sensitivity
            effective_market_impact = market_impact_arg + vamp_effect_on_spread

            self.logger.debug(
                f"VAMP effect on spread: base_impact_arg={market_impact_arg:.4f}, "
                f"vamp_impact_abs={abs(self.vamp_impact):.4f}, sensitivity={vamp_spread_sensitivity:.2f}, "
                f"vamp_effect={vamp_effect_on_spread:.4f}, effective_impact={effective_market_impact:.4f}"
            )

            market_impact_component = effective_market_impact * self.market_impact_factor * (1 + volatility)
            
            if not hasattr(self, 'position_limit') or self.position_limit <= 0:
                self.position_limit = TRADING_CONFIG["avellaneda"].get("position_limit", 0.1)
                if self.position_limit <= 0:
                    self.position_limit = 0.1
                    self.logger.warning(f"Invalid position_limit, using minimum: {self.position_limit}")
            
            # Safe division with proper bounds checking and validation
            current_position = getattr(self.position_tracker, 'current_position', 0.0)
            if not isinstance(current_position, (int, float)) or not np.isfinite(current_position):
                self.logger.warning(f"Invalid current_position value: {current_position}, using 0.0")
                current_position = 0.0
            
            # Ensure position_limit is always positive and reasonable
            safe_position_limit = max(0.001, abs(self.position_limit))
            inventory_risk = abs(current_position) / safe_position_limit
            
            # Cap inventory risk to prevent extreme values
            inventory_risk = min(inventory_risk, 10.0)  # Cap at 10x position limit
            inventory_component = self.inventory_factor * inventory_risk * volatility_term
            
            market_state_factor = 1.0
            
            if prediction_age < pa_prediction_max_age_seconds:
                trend_direction = self.predictive_adjustments.get("trend_direction", 0)
                if trend_direction != 0:
                    if self.market_state != "trending":
                        self.market_state = "trending"
                        self.logger.info(f"Market state changed to trending due to predicted trend direction: {trend_direction}")
                    market_state_factor = 1.5
                elif self.market_state == "trending":
                    self.market_state = "normal"
                    self.logger.info("Market state reset to normal as no trend detected")
            
            if self.market_state == "trending":
                market_state_factor = 1.5
            elif self.market_state == "ranging":
                market_state_factor = 0.85
            
            base_spread = self.base_spread_factor * fee_coverage_spread
            optimal_spread = (
                base_spread * gamma_component +
                volatility_component +
                market_impact_component +
                inventory_component
            ) * market_state_factor
            
            final_spread = max(optimal_spread, fee_coverage_spread)
            
            # Performance: Cache the result with current state
            self._cached_spread = final_spread
            self._cached_volatility_factor = volatility
            self._cached_position = getattr(self.position_tracker, 'current_position', 0.0)
            self._cache_timestamp = current_time
            
            if random.random() < LOG_SAMPLING_RATE:
                self.logger.info(
                    f"Spread calc: fee_coverage=${fee_coverage_spread:.2f}, base={base_spread:.2f}, "
                    f"volatility={volatility_component:.2f}, impact_comp={market_impact_component:.2f}, "
                    f"inventory={inventory_component:.2f}, market_state_factor={market_state_factor:.2f}, "
                    f"final=${final_spread:.2f}"
                )
                
            return final_spread
        except Exception as e:
            self.logger.error(f"Error calculating optimal spread: {str(e)}", exc_info=True)
            maker_fee_rate = TRADING_CONFIG.get("trading_fees", {}).get("maker_fee_rate", 0.0002)
            reference_price = 84000
            fee_based_fallback = reference_price * (maker_fee_rate * 2 + 0.0005)
            min_spread_ticks = ORDERBOOK_CONFIG.get("min_spread", 35)
            if self.tick_size <= 0:
                self.tick_size = self._get_default_tick_size()
            return max(min_spread_ticks * self.tick_size, fee_based_fallback)

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
            if self.tick_size <= 0:
                default_tick = self._get_default_tick_size()
                self.logger.warning(f"Invalid tick size detected in generate_quotes. Setting default value of {default_tick}")
                self.tick_size = default_tick
                
            timestamp = getattr(ticker, 'timestamp', time.time())
            
            if random.random() < LOG_SAMPLING_RATE:
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
            
            self._cached_mid_price = mid_price
            
            if random.random() < LOG_SAMPLING_RATE * 0.1:
                self.logger.debug(f"Using BBO for mid_price: bid={bid_price}, ask={ask_price}, mid={mid_price}")
            
            self.last_mid_price = mid_price
            
            # Extract market conditions
            try:
                volatility = market_conditions.get("volatility", self.volatility)
                market_impact = market_conditions.get("market_impact", 0.0)
                
                if random.random() < LOG_SAMPLING_RATE:
                    self.logger.debug(f"Extracted volatility: {volatility}")
                    self.logger.debug(f"Extracted market_impact: {market_impact}")
                
                # Extract volume candle signals for enhanced quote generation
                volume_momentum = market_conditions.get("volume_momentum", 0.0)
                volume_reversal = market_conditions.get("volume_reversal", 0.0)
                volume_volatility = market_conditions.get("volume_volatility", 0.0)
                volume_exhaustion = market_conditions.get("volume_exhaustion", 0.0)
                
                if (any(abs(sig) > 0.1 for sig in [volume_momentum, volume_reversal, volume_volatility, volume_exhaustion]) 
                    and random.random() < LOG_SAMPLING_RATE * 5):
                    self.logger.info(f"Volume signals detected: momentum={volume_momentum:.3f}, reversal={volume_reversal:.3f}, "
                                   f"volatility={volume_volatility:.3f}, exhaustion={volume_exhaustion:.3f}")
                    
            except Exception as e:
                self.logger.error(f"Error extracting market conditions: {str(e)}")
                return [], []
            
            # Update market conditions
            try:
                self.update_market_conditions(volatility, market_impact)
                if random.random() < LOG_SAMPLING_RATE:
                    self.logger.debug("Market conditions updated successfully")
            except Exception as e:
                self.logger.error(f"Error updating market conditions: {str(e)}")
                return [], []
            
            # Calculate optimal spread
            try:
                base_spread = self.calculate_optimal_spread(market_impact)
                
                if random.random() < LOG_SAMPLING_RATE:
                    self.logger.debug(f"Calculated base spread: {base_spread}")
                
                # Enhance spread based on volume candle signals
                spread_adjustment = 1.0
                
                if volume_volatility > 0.3:
                    spread_adjustment *= (1.0 + volume_volatility * 0.3)
                    if random.random() < LOG_SAMPLING_RATE:
                        self.logger.debug(f"Increased spread by {volume_volatility * 30:.1f}% due to volume volatility")
                
                if volume_reversal > 0.4:
                    spread_adjustment *= (1.0 + volume_reversal * 0.2)
                    if random.random() < LOG_SAMPLING_RATE:
                        self.logger.debug(f"Increased spread by {volume_reversal * 20:.1f}% due to reversal signal")
                
                spread = base_spread * spread_adjustment
                
                if random.random() < LOG_SAMPLING_RATE:
                    self.logger.debug(f"Final spread after volume adjustments: {spread} (adjustment: {spread_adjustment:.3f})")
                
            except Exception as e:
                self.logger.error(f"Error calculating optimal spread: {str(e)}")
                return [], []
            
            # Apply reservation price offset
            try:
                reservation_offset = self.reservation_price_offset
                if abs(reservation_offset) > 0.00001:
                    offset_amount = mid_price * reservation_offset
                    self.logger.debug(f"Applying reservation price offset: {offset_amount:.2f} ({reservation_offset:.6f})")
                    mid_price += offset_amount
            except Exception as e:
                self.logger.error(f"Error applying reservation price offset: {str(e)}")
            
            # Calculate base bid and ask prices with inventory skew and volume signals
            try:
                base_bid_price, base_ask_price = self.calculate_skewed_prices(mid_price, spread)
                self.logger.debug(f"Calculated inventory-skewed prices: bid={base_bid_price}, ask={base_ask_price}")
                
                # Apply volume candle momentum and exhaustion signals to price skewing
                volume_skew_adjustment = 0.0
                
                if abs(volume_momentum) > 0.05:
                    momentum_skew = volume_momentum * spread * 0.3
                    volume_skew_adjustment += momentum_skew
                    self.logger.debug(f"Applied momentum skew: {momentum_skew:.4f} (momentum: {volume_momentum:.3f})")
                
                if abs(volume_exhaustion) > 0.1:
                    exhaustion_skew = -volume_exhaustion * spread * 0.2
                    volume_skew_adjustment += exhaustion_skew
                    self.logger.debug(f"Applied exhaustion skew: {exhaustion_skew:.4f} (exhaustion: {volume_exhaustion:.3f})")
                
                if abs(volume_skew_adjustment) > 0.0001:
                    base_bid_price += volume_skew_adjustment
                    base_ask_price += volume_skew_adjustment
                    self.logger.info(f"Applied total volume skew: {volume_skew_adjustment:.4f} to both sides")
                
                self.logger.debug(f"Final volume-enhanced prices: bid={base_bid_price}, ask={base_ask_price}")
                
            except Exception as e:
                self.logger.error(f"Error calculating skewed prices: {str(e)}")
                return [], []
            
            # Calculate optimal sizes for base bid and ask with volume enhancements
            try:
                base_bid_size, base_ask_size = self.calculate_quote_sizes(mid_price)
                self.logger.debug(f"Calculated base sizes: bid={base_bid_size}, ask={base_ask_size}")
                
                # NEW: Adjust quote sizes based on volume candle signals (FIXED)
                size_adjustment_factor = 1.0
                
                # Reduce sizes on high volatility or reversal signals to limit exposure
                if volume_volatility > 0.4 or volume_reversal > 0.5:
                    risk_reduction = max(volume_volatility, volume_reversal) * 0.3  # Up to 30% reduction
                    size_adjustment_factor *= (1.0 - risk_reduction)
                    self.logger.debug(f"Reduced quote sizes by {risk_reduction * 100:.1f}% due to volume risk signals")
                
                # Adjust individual sides based on momentum and exhaustion
                bid_size_factor = size_adjustment_factor
                ask_size_factor = size_adjustment_factor
                
                # Strong buy momentum: increase ask sizes (more willing to sell), reduce bid sizes
                if volume_momentum > 0.1:  # Lowered threshold from 0.3 to 0.1
                    ask_size_factor *= (1.0 + volume_momentum * 0.2)  # Up to 20% increase
                    bid_size_factor *= (1.0 - volume_momentum * 0.1)  # Up to 10% decrease
                    self.logger.debug(f"Buy momentum detected: increased ask sizes, reduced bid sizes")
                    
                # Strong sell momentum: increase bid sizes (more willing to buy), reduce ask sizes  
                elif volume_momentum < -0.1:  # Lowered threshold from -0.3 to -0.1
                    bid_size_factor *= (1.0 + abs(volume_momentum) * 0.2)  # Up to 20% increase
                    ask_size_factor *= (1.0 - abs(volume_momentum) * 0.1)  # Up to 10% decrease
                    self.logger.debug(f"Sell momentum detected: increased bid sizes, reduced ask sizes")
                
                # Apply size adjustments
                base_bid_size *= bid_size_factor
                base_ask_size *= ask_size_factor
                
                self.logger.debug(f"Volume-enhanced sizes: bid={base_bid_size:.4f} (factor: {bid_size_factor:.3f}), "
                                f"ask={base_ask_size:.4f} (factor: {ask_size_factor:.3f})")
                
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
                    
                    # Reduce debug logging for HFT performance
                if random.random() < LOG_SAMPLING_RATE * 0.1:  # 10x less frequent
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
                    
                    # Reduce debug logging for HFT performance
                if random.random() < LOG_SAMPLING_RATE * 0.1:  # 10x less frequent
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
            default_tick = self._get_default_tick_size()
            self.logger.warning(f"Invalid tick_size {self.tick_size} in _calculate_level_price. Using default of {default_tick}")
            self.tick_size = default_tick
        
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
        # HFT BUG FIX: Handle NaN values that cause "cannot convert float NaN to integer"
        if not (value and math.isfinite(value)):
            return 0.0  # Return safe default for NaN/inf values
            
        # Performance optimization: Fast path for valid tick size
        if self.tick_size > 0:
            return round(value / self.tick_size) * self.tick_size
        
        # Fallback path with occasional logging
        default_tick = self._get_default_tick_size()
        if random.random() < LOG_SAMPLING_RATE:
            self.logger.warning(f"Invalid tick size ({self.tick_size}) in round_to_tick. Using default of {default_tick}")
        self.tick_size = default_tick
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
            old_position = self.position_tracker.current_position
            old_entry_price = self.position_tracker.average_entry_price
            self.position_tracker.update_on_fill(fill)
            
            # Update VAMP calculations
            self.update_vamp(fill_price, fill_size, is_buy, False)
            
            # NEW: Update volume candle buffer with fill data
            self.volume_buffer.update(fill_price, fill_size, is_buy, int(time.time() * 1000))
            
            # NOTE: Trigger order logic is now handled separately by the quoter
            # with instrument-specific information via _handle_trigger_order_on_fill
            
            # Performance: Always log fills but with reduced detail for frequent small fills  
            if fill_size >= 0.01 or random.random() < LOG_SAMPLING_RATE * 10:  # Always log significant fills
                self.logger.info(
                    f"Order filled: {fill_size} @ {fill_price} ({'BUY' if is_buy else 'SELL'})"
                    f" - New position: {self.position_tracker.current_position:.4f}, Avg entry: {self.position_tracker.average_entry_price:.2f}"
                )
            
        except Exception as e:
            self.logger.error(f"Error handling order fill: {str(e)}")

    def _handle_trigger_order_on_fill(self, fill_price: float, fill_size: float, is_buy: bool, 
                                     old_position: float, old_entry_price: float, instrument: str = None) -> None:
        """
        Handle trigger order creation/update when a grid order is filled
        
        Args:
            fill_price: Price at which the order was filled
            fill_size: Size of the fill
            is_buy: True if it was a buy order
            old_position: Position before this fill
            old_entry_price: Entry price before this fill
            instrument: The instrument that was filled (NEW)
        """
        try:
            # Use the instrument from the fill, or fallback to current instrument
            target_instrument = instrument or self.instrument
            if not target_instrument:
                self.logger.warning("No instrument specified for trigger order creation")
                return
            
            # Get current position for this specific instrument
            current_position = old_position  # Use old position as fallback
            
            # Try to get current position from portfolio tracker if available
            if hasattr(self, 'portfolio_tracker') and self.portfolio_tracker:
                if target_instrument in self.portfolio_tracker.instrument_trackers:
                    instrument_tracker = self.portfolio_tracker.instrument_trackers[target_instrument]
                    current_position = instrument_tracker.current_position
                else:
                    self.logger.warning(f"Instrument {target_instrument} not found in portfolio tracker")
                    return
            else:
                # Fallback to single position tracker (but this won't be instrument-specific)
                current_position = self.position_tracker.current_position
            
            # Skip if no position (shouldn't happen but safety check)
            if abs(current_position) < 0.001:
                self.logger.debug(f"No position after fill for {target_instrument}, skipping trigger order creation")
                return
            
            # Determine if this is a new position or addition to existing position
            position_direction_changed = (old_position >= 0) != (current_position >= 0)
            position_increased = abs(current_position) > abs(old_position)
            
            # If position direction changed, cancel all existing triggers for this instrument
            if position_direction_changed:
                self._cancel_trigger_orders_by_instrument(target_instrument, "Position direction changed")
            
            # If position increased (new inventory added), update/create trigger orders
            if position_increased:
                # Cancel existing trigger orders for this side and instrument
                side_to_cancel = "sell" if current_position > 0 else "buy"
                self._cancel_trigger_orders_by_side_and_instrument(side_to_cancel, target_instrument, "Position increased, updating trigger")
                
                # Create new trigger order at updated entry price + spread
                self._create_trigger_order_for_instrument(target_instrument)
                
                self.logger.info(
                    f"Updated trigger for {target_instrument}: position {old_position:.4f} → {current_position:.4f}"
                )
                
            self.last_trigger_update = time.time()
            
        except Exception as e:
            self.logger.error(f"Error handling trigger order on fill: {str(e)}", exc_info=True)

    def _create_trigger_order_for_instrument(self, instrument: str) -> None:
        """Create a take profit trigger order for a specific instrument"""
        try:
            # Get position for this specific instrument from portfolio tracker
            if not hasattr(self, 'portfolio_tracker') or not self.portfolio_tracker:
                self.logger.warning("No portfolio tracker available for instrument-specific trigger orders")
                return
                
            # Get instrument tracker
            if instrument not in self.portfolio_tracker.instrument_trackers:
                self.logger.warning(f"Instrument {instrument} not found in portfolio tracker")
                return
                
            instrument_tracker = self.portfolio_tracker.instrument_trackers[instrument]
            current_position = instrument_tracker.current_position
            
            # Skip if no significant position
            if abs(current_position) < 0.001:
                self.logger.debug(f"No significant position for {instrument}: {current_position:.6f}")
                return
                
            entry_price = instrument_tracker.average_entry_price
            if entry_price <= 0:
                self.logger.warning(f"Invalid entry price for {instrument}: {entry_price}")
                return
            
            # Calculate take profit price (entry + spread in basis points)
            spread_multiplier = self.take_profit_spread_bps / 10000.0  # Convert bps to decimal
            
            if current_position > 0:  # Long position - sell trigger above entry
                trigger_price = entry_price * (1 + spread_multiplier)
                trigger_side = "sell"
                trigger_size = abs(current_position)
            else:  # Short position - buy trigger below entry
                trigger_price = entry_price * (1 - spread_multiplier)
                trigger_side = "buy"
                trigger_size = abs(current_position)
            
            # Round to tick size (use instrument-specific tick size if available)
            trigger_price = self.round_to_tick(trigger_price)
            
            # Ensure minimum order size
            min_size = ORDERBOOK_CONFIG.get("min_order_size", 0.001)
            trigger_size = max(min_size, trigger_size)
            
            # Create trigger quote
            trigger_quote = Quote(
                instrument=instrument,  # Use specific instrument
                side=trigger_side,
                price=trigger_price,
                amount=trigger_size,
                timestamp=time.time(),
                is_trigger_order=True  # Mark as trigger order
            )
            
            # Store the trigger order with instrument-specific ID
            trigger_id = f"trigger_{instrument}_{trigger_side}_{int(time.time() * 1000)}"
            self.active_trigger_orders[trigger_id] = trigger_quote
            
            self.logger.info(
                f"Created trigger order for {instrument}: {trigger_side.upper()} {trigger_size:.4f} @ {trigger_price:.2f} "
                f"(entry: {entry_price:.2f}, spread: {self.take_profit_spread_bps:.1f}bps, "
                f"position: {current_position:.4f})"
            )
            
        except Exception as e:
            self.logger.error(f"Error creating trigger order for {instrument}: {str(e)}", exc_info=True)

    def _cancel_trigger_orders_by_instrument(self, instrument: str, reason: str) -> None:
        """Cancel all trigger orders for a specific instrument"""
        try:
            orders_to_cancel = []
            for order_id, trigger_order in self.active_trigger_orders.items():
                if trigger_order.instrument == instrument:
                    orders_to_cancel.append(order_id)
            
            for order_id in orders_to_cancel:
                self.pending_trigger_cancellations.add(order_id)
                del self.active_trigger_orders[order_id]
                self.logger.info(f"Cancelled trigger order {order_id} for {instrument}: {reason}")
                
        except Exception as e:
            self.logger.error(f"Error cancelling trigger orders for {instrument}: {str(e)}", exc_info=True)

    def _cancel_trigger_orders_by_side_and_instrument(self, side: str, instrument: str, reason: str) -> None:
        """Cancel trigger orders for a specific side and instrument"""
        try:
            orders_to_cancel = []
            for order_id, trigger_order in self.active_trigger_orders.items():
                if trigger_order.side == side and trigger_order.instrument == instrument:
                    orders_to_cancel.append(order_id)
            
            for order_id in orders_to_cancel:
                self.pending_trigger_cancellations.add(order_id)
                del self.active_trigger_orders[order_id]
                self.logger.info(f"Cancelled trigger order {order_id} ({side}) for {instrument}: {reason}")
                
        except Exception as e:
            self.logger.error(f"Error cancelling trigger orders by side for {instrument}: {str(e)}", exc_info=True)

    def check_and_create_missing_trigger_orders(self) -> None:
        """Enhanced check for missing trigger orders with multi-instrument support"""
        try:
            if not self.trigger_order_enabled:
                return
            
            # Use enhanced trigger order creation
            new_triggers = self.create_enhanced_trigger_orders()
            
            if new_triggers:
                self.logger.info(f"Created {len(new_triggers)} enhanced trigger orders")
            else:
                self.logger.debug("No new trigger orders needed")
                
        except Exception as e:
            self.logger.error(f"Error checking and creating missing trigger orders: {str(e)}", exc_info=True)

    def _cancel_all_trigger_orders(self, reason: str) -> None:
        """Cancel all active trigger orders"""
        try:
            cancelled_count = 0
            for order_id in list(self.active_trigger_orders.keys()):
                self.pending_trigger_cancellations.add(order_id)
                del self.active_trigger_orders[order_id]
                cancelled_count += 1
                
            if cancelled_count > 0:
                self.logger.info(f"Cancelled {cancelled_count} trigger orders: {reason}")
                
        except Exception as e:
            self.logger.error(f"Error cancelling all trigger orders: {str(e)}", exc_info=True)

    def _cancel_trigger_orders_by_side(self, side: str, reason: str) -> None:
        """Cancel trigger orders for a specific side (all instruments)"""
        try:
            orders_to_cancel = []
            for order_id, trigger_order in self.active_trigger_orders.items():
                if trigger_order.side == side:
                    orders_to_cancel.append(order_id)
            
            for order_id in orders_to_cancel:
                self.pending_trigger_cancellations.add(order_id)
                del self.active_trigger_orders[order_id]
                self.logger.info(f"Cancelled trigger order {order_id} ({side}): {reason}")
                
        except Exception as e:
            self.logger.error(f"Error cancelling trigger orders by side: {str(e)}", exc_info=True)

    def get_active_trigger_orders(self) -> Dict[str, Quote]:
        """Get current active trigger orders for the quoter to place"""
        return self.active_trigger_orders.copy()

    def on_trigger_order_filled(self, order_id: str, fill_price: float, fill_size: float) -> None:
        """Handle when a trigger order gets filled"""
        try:
            # Remove from active triggers
            if order_id in self.active_trigger_orders:
                trigger_order = self.active_trigger_orders[order_id]
                del self.active_trigger_orders[order_id]
                
                self.logger.info(
                    f"Trigger order filled: {trigger_order.side.upper()} {fill_size:.4f} @ {fill_price:.2f} "
                    f"(target: {trigger_order.price:.2f})"
                )
                
                # The fill will be handled by the normal on_order_filled method
                # which will update position tracker
                
        except Exception as e:
            self.logger.error(f"Error handling trigger order fill: {str(e)}", exc_info=True)

    def on_trigger_order_cancelled(self, order_id: str) -> None:
        """Handle when a trigger order gets cancelled"""
        try:
            if order_id in self.pending_trigger_cancellations:
                self.pending_trigger_cancellations.remove(order_id)
                
            if order_id in self.active_trigger_orders:
                del self.active_trigger_orders[order_id]
                
        except Exception as e:
            self.logger.error(f"Error handling trigger order cancellation: {str(e)}", exc_info=True)

    def get_position_metrics(self) -> Dict:
        """Enhanced position metrics with multi-instrument support"""
        try:
            # Get trading mode and profit metrics
            mode = self.detect_trading_mode()
            
            base_metrics = {
                'position': self.position_tracker.current_position,
                'entry_price': self.position_tracker.average_entry_price,
                'realized_pnl': self.position_tracker.realized_pnl,
                'unrealized_pnl': self.position_tracker.unrealized_pnl,
                'total_pnl': self.position_tracker.realized_pnl + self.position_tracker.unrealized_pnl,
                'vwap': self.vwap,
                'vamp': self.calculate_vamp(),
                'trading_mode': mode
            }
            
            # Add multi-instrument metrics
            if self.portfolio_tracker:
                portfolio_metrics = self.portfolio_tracker.get_portfolio_metrics()
                base_metrics.update({
                    'portfolio_total_pnl': portfolio_metrics.get('total_pnl', 0.0),
                    'portfolio_net_pnl': portfolio_metrics.get('net_pnl_after_fees', 0.0),
                    'portfolio_positions': portfolio_metrics.get('instrument_positions', {}),
                    'estimated_closing_fees': portfolio_metrics.get('estimated_closing_fees', 0.0)
                })
                
                # Add mode-specific profit metrics
                if mode == "arbitrage":
                    arb_profit = self.calculate_arbitrage_profit()
                    base_metrics.update({
                        'arbitrage_profit_usd': arb_profit['total_profit_usd'],
                        'spread_profit_bps': arb_profit['spread_profit_bps'],
                        'arbitrage_net_profit': arb_profit['net_profit_after_fees']
                    })
                elif mode in ["single_perp", "single_futures"]:
                    instrument = self.perp_instrument if mode == "single_perp" else self.futures_instrument
                    if instrument:
                        single_profit = self.calculate_single_instrument_profit(instrument)
                        base_metrics.update({
                            f'{mode}_profit_usd': single_profit['profit_usd'],
                            f'{mode}_profit_pct': single_profit.get('profit_percentage', 0.0)
                        })
            
            # Add trigger decision info
            try:
                trigger_decision = self.should_trigger_take_profit()
                base_metrics.update({
                    'should_trigger_take_profit': trigger_decision['should_trigger'],
                    'trigger_reason': trigger_decision.get('reason', ''),
                    'active_trigger_orders': len(self.active_trigger_orders)
                })
            except:
                # Don't fail if trigger logic has issues
                pass
            
            # NEW: Add predictive signals to metrics
            try:
                if hasattr(self, 'volume_buffer') and self.volume_buffer:
                    base_metrics['predictive_signals'] = self.volume_buffer.get_signal_metrics()['signals']
                    
                    # Add prediction accuracy if available
                    if hasattr(self, 'last_mid_price') and self.last_mid_price > 0:
                        base_metrics['prediction_accuracy'] = self.volume_buffer.evaluate_prediction_accuracy(self.last_mid_price)
            except Exception as e:
                self.logger.error(f"Error getting predictive metrics: {str(e)}")
                
            return base_metrics
            
        except Exception as e:
            self.logger.error(f"Error getting position metrics: {str(e)}", exc_info=True)
            # Return basic metrics as fallback
            return {
                'position': getattr(self.position_tracker, 'current_position', 0),
                'entry_price': getattr(self.position_tracker, 'average_entry_price', 0),
                'realized_pnl': getattr(self.position_tracker, 'realized_pnl', 0),
                'unrealized_pnl': getattr(self.position_tracker, 'unrealized_pnl', 0),
                'total_pnl': 0,
                'vwap': self.vwap,
                'vamp': self.calculate_vamp(),
                'trading_mode': 'error'
            }

    def align_price_to_tick(self, price: float) -> float:
        """
        Align price to the instrument's tick size
        
        Args:
            price: The price to align
            
        Returns:
            float: Price aligned to the instrument's tick size
        """
        # Performance: Fast path for valid inputs (most common case)
        if price > 0 and self.tick_size > 0:
            aligned_price = self.tick_size * round(price / self.tick_size)
            return max(aligned_price, self.tick_size)  # Ensure minimum price
        
        # Fallback path with occasional logging
        if price <= 0:
            if random.random() < LOG_SAMPLING_RATE:
                self.logger.warning(f"Invalid price {price} in align_price_to_tick. Using minimum tick size.")
            return self.tick_size if self.tick_size > 0 else self._get_default_tick_size()
            
        if not self.tick_size or self.tick_size <= 0:
            if random.random() < LOG_SAMPLING_RATE:
                self.logger.warning("Invalid tick size, using default alignment")
                old_tick_size = self.tick_size
                self.tick_size = self._get_default_tick_size()
                self.logger.info(f"Updated tick size from {old_tick_size} to {self.tick_size}")
            else:
                self.tick_size = self._get_default_tick_size()  # Set default without logging
            return round(price, 2)  # Default to 2 decimal places

    def set_instrument(self, instrument: str):
        """Set the instrument being traded"""
        self.instrument = instrument
        
        # Performance: Only log occasionally unless instrument actually changes
        if random.random() < LOG_SAMPLING_RATE * 10:  # More frequent for instrument changes
            self.logger.info(f"Setting instrument to {instrument}")
        
        # Update instrument in volume candle buffer if available
        if hasattr(self, 'volume_buffer') and self.volume_buffer:
            self.volume_buffer.instrument = instrument
            # Performance: Only log occasionally
            if random.random() < LOG_SAMPLING_RATE * 5:
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
        # Performance: Always log cleanup start/end since it's infrequent but important
        self.logger.info("Cleaning up Avellaneda market maker resources")
        
        # Clean up trigger orders
        if hasattr(self, 'active_trigger_orders') and self.active_trigger_orders:
            self.logger.info(f"Cleaning up {len(self.active_trigger_orders)} active trigger orders")
            self._cancel_all_trigger_orders("System shutdown cleanup")
        
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

    def set_instrument_config(self, perp_instrument: str = None, futures_instrument: str = None, 
                             perp_tick_size: float = 1.0, futures_tick_size: float = 1.0):
        """
        Configure instruments for multi-instrument take profit logic
        
        Args:
            perp_instrument: Perpetual contract instrument name (e.g. "BTC-PERPETUAL")
            futures_instrument: Futures contract instrument name (e.g. "BTC-25JUL25")
            perp_tick_size: Tick size for perpetual contract
            futures_tick_size: Tick size for futures contract
        """
        self.perp_instrument = perp_instrument
        self.futures_instrument = futures_instrument
        self.perp_tick_size = perp_tick_size
        self.futures_tick_size = futures_tick_size
        
        self.logger.info(
            f"Instrument config updated: Perp={perp_instrument} (tick:{perp_tick_size}), "
            f"Futures={futures_instrument} (tick:{futures_tick_size})"
        )

    def detect_trading_mode(self) -> str:
        """
        Detect current trading mode based on active positions
        
        Returns:
            str: "single_perp", "single_futures", "arbitrage", or "unknown"
        """
        try:
            current_time = time.time()
            
            # Don't check too frequently for performance
            if current_time - self.last_mode_detection < self.mode_detection_interval:
                return self.trading_mode
                
            self.last_mode_detection = current_time
            
            if not self.portfolio_tracker:
                self.logger.debug("No portfolio tracker available for mode detection")
                return "unknown"
            
            # Get positions for both instruments
            perp_position = 0.0
            futures_position = 0.0
            
            if self.perp_instrument and self.perp_instrument in self.portfolio_tracker.instrument_trackers:
                perp_position = self.portfolio_tracker.instrument_trackers[self.perp_instrument].current_position
                
            if self.futures_instrument and self.futures_instrument in self.portfolio_tracker.instrument_trackers:
                futures_position = self.portfolio_tracker.instrument_trackers[self.futures_instrument].current_position
            
            # Position thresholds
            position_threshold = 0.001  # Minimum position to consider significant
            
            # Detect mode based on positions
            has_perp = abs(perp_position) > position_threshold
            has_futures = abs(futures_position) > position_threshold
            
            if has_perp and has_futures:
                # Check if positions are opposite (arbitrage) or same direction
                if (perp_position > 0 and futures_position < 0) or (perp_position < 0 and futures_position > 0):
                    new_mode = "arbitrage"
                else:
                    # Both positions in same direction - likely hedged or directional multi-leg
                    new_mode = "arbitrage"  # Still treat as arbitrage for profit calculation
            elif has_perp and not has_futures:
                new_mode = "single_perp"
            elif has_futures and not has_perp:
                new_mode = "single_futures"
            else:
                new_mode = "unknown"  # No significant positions
            
            # Log mode changes
            if new_mode != self.trading_mode:
                self.logger.info(
                    f"Trading mode changed: {self.trading_mode} → {new_mode} "
                    f"(Perp: {perp_position:.4f}, Futures: {futures_position:.4f})"
                )
                self.trading_mode = new_mode
            
            return self.trading_mode
            
        except Exception as e:
            self.logger.error(f"Error detecting trading mode: {str(e)}", exc_info=True)
            return "unknown"

    def calculate_single_instrument_profit(self, instrument: str) -> Dict[str, float]:
        """
        Calculate profit metrics for a single instrument position
        
        Args:
            instrument: Instrument name
            
        Returns:
            Dict with profit metrics
        """
        try:
            if not self.portfolio_tracker or instrument not in self.portfolio_tracker.instrument_trackers:
                return {"profit_usd": 0.0, "position_size": 0.0, "entry_price": 0.0, "mark_price": 0.0}
            
            tracker = self.portfolio_tracker.instrument_trackers[instrument]
            position_size = tracker.current_position
            
            if abs(position_size) < 0.001:
                return {"profit_usd": 0.0, "position_size": 0.0, "entry_price": 0.0, "mark_price": 0.0}
            
            entry_price = tracker.average_entry_price
            mark_price = self.portfolio_tracker.mark_prices.get(instrument, 0.0)
            
            if entry_price <= 0 or mark_price <= 0:
                return {"profit_usd": 0.0, "position_size": position_size, "entry_price": entry_price, "mark_price": mark_price}
            
            # Calculate profit
            profit_per_unit = mark_price - entry_price if position_size > 0 else entry_price - mark_price
            profit_usd = profit_per_unit * abs(position_size)
            
            return {
                "profit_usd": profit_usd,
                "position_size": position_size,
                "entry_price": entry_price,
                "mark_price": mark_price,
                "profit_per_unit": profit_per_unit,
                "profit_percentage": (profit_per_unit / entry_price) * 100 if entry_price > 0 else 0.0
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating single instrument profit for {instrument}: {str(e)}", exc_info=True)
            return {"profit_usd": 0.0, "position_size": 0.0, "entry_price": 0.0, "mark_price": 0.0}

    def calculate_arbitrage_profit(self) -> Dict[str, Any]:
        """
        Calculate combined profit metrics for arbitrage positions
        
        Returns:
            Dict with combined profit metrics
        """
        try:
            if not self.portfolio_tracker:
                return {"total_profit_usd": 0.0, "spread_profit_bps": 0.0, "legs": {}}
            
            # Get individual leg profits
            perp_metrics = self.calculate_single_instrument_profit(self.perp_instrument) if self.perp_instrument else {}
            futures_metrics = self.calculate_single_instrument_profit(self.futures_instrument) if self.futures_instrument else {}
            
            # Calculate combined metrics
            total_profit_usd = perp_metrics.get("profit_usd", 0.0) + futures_metrics.get("profit_usd", 0.0)
            
            # Calculate spread profit in basis points
            spread_profit_bps = 0.0
            if perp_metrics.get("position_size", 0) != 0 and futures_metrics.get("position_size", 0) != 0:
                # Use the larger position size as the base for BPS calculation
                base_notional = max(
                    abs(perp_metrics.get("position_size", 0) * perp_metrics.get("mark_price", 0)),
                    abs(futures_metrics.get("position_size", 0) * futures_metrics.get("mark_price", 0))
                )
                if base_notional > 0:
                    spread_profit_bps = (total_profit_usd / base_notional) * 10000
            
            # Estimate closing fees
            closing_fees = 0.0
            if hasattr(self.portfolio_tracker, 'estimate_closing_fees'):
                closing_fees = self.portfolio_tracker.estimate_closing_fees()
            
            return {
                "total_profit_usd": total_profit_usd,
                "spread_profit_bps": spread_profit_bps,
                "estimated_closing_fees": closing_fees,
                "net_profit_after_fees": total_profit_usd - closing_fees,
                "legs": {
                    "perp": perp_metrics,
                    "futures": futures_metrics
                },
                "combined_position_value": (
                    abs(perp_metrics.get("position_size", 0) * perp_metrics.get("mark_price", 0)) +
                    abs(futures_metrics.get("position_size", 0) * futures_metrics.get("mark_price", 0))
                )
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating arbitrage profit: {str(e)}", exc_info=True)
            return {"total_profit_usd": 0.0, "spread_profit_bps": 0.0, "legs": {}}

    def should_trigger_take_profit(self) -> Dict[str, Any]:
        """
        Determine if take profit should be triggered based on current positions and profits
        
        Returns:
            Dict with trigger decision and details
        """
        try:
            current_time = time.time()
            
            # Don't check too frequently for performance
            if current_time - self.last_arbitrage_check < self.arbitrage_check_interval:
                return {"should_trigger": False, "reason": "rate_limited"}
            
            self.last_arbitrage_check = current_time
            
            # Detect current trading mode
            mode = self.detect_trading_mode()
            
            if mode == "unknown":
                return {"should_trigger": False, "reason": "no_significant_positions", "mode": mode}
            
            result = {
                "should_trigger": False,
                "reason": "",
                "mode": mode,
                "trigger_type": "",
                "instruments_to_close": [],
                "profit_metrics": {}
            }
            
            if mode == "arbitrage":
                # Arbitrage mode - check combined profit
                arb_profit = self.calculate_arbitrage_profit()
                result["profit_metrics"] = arb_profit
                
                total_profit = arb_profit["total_profit_usd"]
                net_profit = arb_profit["net_profit_after_fees"]
                spread_bps = arb_profit["spread_profit_bps"]
                
                self.logger.debug(
                    f"Arbitrage profit check: Total=${total_profit:.2f}, Net=${net_profit:.2f}, "
                    f"Spread={spread_bps:.1f}bps, Threshold=${self.arbitrage_profit_threshold_usd:.2f}"
                )
                
                # Check if arbitrage profit exceeds threshold
                if net_profit >= self.arbitrage_profit_threshold_usd or spread_bps >= self.spread_profit_threshold_bps:
                    result["should_trigger"] = True
                    result["trigger_type"] = "arbitrage"
                    result["reason"] = f"arbitrage_profit_threshold (${net_profit:.2f} >= ${self.arbitrage_profit_threshold_usd:.2f} OR {spread_bps:.1f}bps >= {self.spread_profit_threshold_bps:.1f}bps)"
                    
                    # Close both legs
                    if self.perp_instrument and abs(arb_profit["legs"]["perp"].get("position_size", 0)) > 0.001:
                        result["instruments_to_close"].append(self.perp_instrument)
                    if self.futures_instrument and abs(arb_profit["legs"]["futures"].get("position_size", 0)) > 0.001:
                        result["instruments_to_close"].append(self.futures_instrument)
                        
            elif mode in ["single_perp", "single_futures"]:
                # Single instrument mode
                instrument = self.perp_instrument if mode == "single_perp" else self.futures_instrument
                if not instrument:
                    return {"should_trigger": False, "reason": "instrument_not_configured", "mode": mode}
                
                single_profit = self.calculate_single_instrument_profit(instrument)
                result["profit_metrics"] = {instrument: single_profit}
                
                profit_usd = single_profit["profit_usd"]
                
                self.logger.debug(
                    f"Single instrument profit check ({instrument}): ${profit_usd:.2f}, "
                    f"Threshold=${self.single_instrument_profit_threshold_usd:.2f}"
                )
                
                # Check if single instrument profit exceeds threshold
                if profit_usd >= self.single_instrument_profit_threshold_usd:
                    result["should_trigger"] = True
                    result["trigger_type"] = "single_instrument"
                    result["reason"] = f"single_instrument_threshold (${profit_usd:.2f} >= ${self.single_instrument_profit_threshold_usd:.2f})"
                    result["instruments_to_close"] = [instrument]
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error checking take profit trigger: {str(e)}", exc_info=True)
            return {"should_trigger": False, "reason": "error", "error": str(e)}

    def create_enhanced_trigger_orders(self) -> List[Quote]:
        """
        Create enhanced trigger orders based on current trading mode and positions
        
        Returns:
            List of trigger orders to place
        """
        try:
            # Check if we should trigger take profit
            trigger_decision = self.should_trigger_take_profit()
            
            if not trigger_decision["should_trigger"]:
                # No immediate trigger needed, create standard trigger orders for existing positions
                return self._create_standard_trigger_orders()
            
            # Create immediate trigger orders to close positions
            trigger_orders = []
            instruments_to_close = trigger_decision.get("instruments_to_close", [])
            
            self.logger.info(
                f"Creating enhanced trigger orders: {trigger_decision['trigger_type']} trigger "
                f"for {len(instruments_to_close)} instruments - {trigger_decision['reason']}"
            )
            
            for instrument in instruments_to_close:
                if not self.portfolio_tracker or instrument not in self.portfolio_tracker.instrument_trackers:
                    continue
                
                tracker = self.portfolio_tracker.instrument_trackers[instrument]
                position_size = tracker.current_position
                
                if abs(position_size) < 0.001:
                    continue
                
                # Get current mark price
                mark_price = self.portfolio_tracker.mark_prices.get(instrument, 0.0)
                if mark_price <= 0:
                    self.logger.warning(f"No mark price for {instrument}, skipping trigger order")
                    continue
                
                # Determine appropriate tick size with safety check
                tick_size = self.perp_tick_size if instrument == self.perp_instrument else self.futures_tick_size
                if tick_size <= 0:
                    self.logger.warning(f"Invalid tick size {tick_size} for {instrument}, using default 1.0")
                    tick_size = 1.0
                
                # Create aggressive trigger order to close position quickly
                # Use a small buffer to ensure fills (0.1% for market-like execution)
                price_buffer = 0.001  # 10 basis points
                
                if position_size > 0:  # Long position - create sell trigger
                    trigger_price = mark_price * (1 - price_buffer)
                    trigger_side = "sell"
                else:  # Short position - create buy trigger
                    trigger_price = mark_price * (1 + price_buffer)
                    trigger_side = "buy"
                
                # Round to appropriate tick size
                trigger_price = round(trigger_price / tick_size) * tick_size
                trigger_size = abs(position_size)
                
                # Create trigger order
                trigger_quote = Quote(
                    instrument=instrument,
                    side=trigger_side,
                    price=trigger_price,
                    amount=trigger_size,
                    timestamp=time.time(),
                    is_trigger_order=True
                )
                
                # Generate unique trigger ID
                trigger_id = f"enhanced_trigger_{instrument}_{trigger_side}_{int(time.time() * 1000)}"
                
                # Store trigger order
                self.active_trigger_orders[trigger_id] = trigger_quote
                trigger_orders.append(trigger_quote)
                
                self.logger.info(
                    f"Created enhanced trigger: {instrument} {trigger_side.upper()} {trigger_size:.4f} @ {trigger_price:.2f} "
                    f"(mark: {mark_price:.2f}, buffer: {price_buffer:.1%})"
                )
            
            return trigger_orders
            
        except Exception as e:
            self.logger.error(f"Error creating enhanced trigger orders: {str(e)}", exc_info=True)
            return []

    def _create_standard_trigger_orders(self) -> List[Quote]:
        """
        Create standard trigger orders for existing positions (non-urgent)
        
        Returns:
            List of standard trigger orders
        """
        try:
            if not self.portfolio_tracker:
                return []
            
            trigger_orders = []
            
            for instrument, tracker in self.portfolio_tracker.instrument_trackers.items():
                position_size = tracker.current_position
                
                if abs(position_size) < 0.001:
                    continue
                
                entry_price = tracker.average_entry_price
                if entry_price <= 0:
                    continue
                
                # Calculate standard take profit price (entry + spread)
                spread_multiplier = self.take_profit_spread_bps / 10000.0
                
                if position_size > 0:  # Long position - sell trigger above entry
                    trigger_price = entry_price * (1 + spread_multiplier)
                    trigger_side = "sell"
                else:  # Short position - buy trigger below entry
                    trigger_price = entry_price * (1 - spread_multiplier)
                    trigger_side = "buy"
                
                # Determine appropriate tick size with safety check
                tick_size = self.perp_tick_size if instrument == self.perp_instrument else self.futures_tick_size
                if tick_size <= 0:
                    self.logger.warning(f"Invalid tick size {tick_size} for {instrument}, using default 1.0")
                    tick_size = 1.0
                trigger_price = round(trigger_price / tick_size) * tick_size
                
                # Check if we already have a similar trigger order
                existing_similar = False
                for existing_id, existing_quote in self.active_trigger_orders.items():
                    if (existing_quote.instrument == instrument and 
                        existing_quote.side == trigger_side and
                        abs(existing_quote.price - trigger_price) < tick_size * 2):
                        existing_similar = True
                        break
                
                if existing_similar:
                    continue
                
                # Create standard trigger order
                trigger_quote = Quote(
                    instrument=instrument,
                    side=trigger_side,
                    price=trigger_price,
                    amount=abs(position_size),
                    timestamp=time.time(),
                    is_trigger_order=True
                )
                
                trigger_id = f"standard_trigger_{instrument}_{trigger_side}_{int(time.time() * 1000)}"
                self.active_trigger_orders[trigger_id] = trigger_quote
                trigger_orders.append(trigger_quote)
                
                self.logger.debug(
                    f"Created standard trigger: {instrument} {trigger_side.upper()} {abs(position_size):.4f} @ {trigger_price:.2f} "
                    f"(entry: {entry_price:.2f}, spread: {self.take_profit_spread_bps:.1f}bps)"
                )
            
            return trigger_orders
            
        except Exception as e:
            self.logger.error(f"Error creating standard trigger orders: {str(e)}", exc_info=True)
            return []
