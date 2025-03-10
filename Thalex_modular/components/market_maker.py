import numpy as np
from typing import Dict, List, Optional, Tuple
import time
import logging

from thalex_py.logs.Thalex_modular.config.market_config import TRADING_PARAMS, ORDERBOOK_CONFIG
from thalex_py.logs.Thalex_modular.models.data_models import Quote

class MarketMaker:
    def __init__(self):
        # Avellaneda-Stoikov parameters
        self.gamma = TRADING_PARAMS["position_management"]["gamma"]
        self.inventory_weight = TRADING_PARAMS["position_management"]["inventory_weight"]
        self.position_fade_time = TRADING_PARAMS["position_management"]["position_fade_time"]
        self.order_flow_intensity = TRADING_PARAMS["position_management"]["order_flow_intensity"]
        
        # Market making state
        self.position_size = 0.0
        self.entry_price = 0.0
        self.last_quote_time = 0.0
        self.tick_size = 0.0
        
        # Market conditions
        self.volatility = 0.0
        self.market_impact = 0.0
        
        # Quote tracking
        self.active_quotes: Dict[str, Quote] = {}
        self.quote_history: List[Dict] = []
        
        # Performance tracking
        self.quote_performance = {
            "total_quotes": 0,
            "filled_quotes": 0,
            "cancelled_quotes": 0,
            "amended_quotes": 0
        }

    def set_tick_size(self, tick_size: float):
        """Set the tick size for the instrument"""
        self.tick_size = tick_size

    def update_position(self, size: float, price: float):
        """Update current position information"""
        self.position_size = size
        self.entry_price = price

    def update_market_conditions(self, volatility: float, market_impact: float):
        """Update market conditions"""
        self.volatility = volatility
        self.market_impact = market_impact

    def calculate_optimal_size(self, side: str, q: float, volatility: float) -> float:
        """Calculate optimal size for quotes based on market conditions"""
        try:
            # Get base size from config
            side_key = "bid_sizes" if side == "bid" else "ask_sizes"
            base_sizes = ORDERBOOK_CONFIG.get(side_key, [0.2, 0.8])
            base_size = base_sizes[0] if base_sizes else 0.2  # Use first level size as base
            
            # Adjust for position
            position_utilization = abs(q)  # How much of position limit is used
            size_scalar = 1.0 - position_utilization  # Reduce size as position grows
            
            # Adjust for volatility - handle missing keys gracefully
            vol_scalar = 1.0
            vol_floor = TRADING_PARAMS.get("volatility", {}).get("vol_floor", 0.001)
            if volatility > vol_floor:
                vol_scalar = min(1.0, vol_floor / volatility)
            
            # Adjust for position direction
            direction_scalar = 1.0
            position_limit = TRADING_PARAMS.get("position_management", {}).get("position_limit", 50000)
            if (side == "bid" and self.position_size > 0) or (side == "ask" and self.position_size < 0):
                direction_scalar = max(0.2, 1.0 - abs(self.position_size) / position_limit)
            
            # Calculate final size
            size = base_size * size_scalar * vol_scalar * direction_scalar
            
            # Apply limits
            max_size = ORDERBOOK_CONFIG.get("threshold", 1.0)
            size = min(max_size, max(0.01, size))  # Ensure minimum size of 0.01
            
            logging.debug(f"Calculated {side} size: {size} (base={base_size}, pos_util={position_utilization:.2f}, vol={vol_scalar:.2f})")
            return size
            
        except Exception as e:
            logging.error(f"Error calculating optimal size: {str(e)}")
            return 0.1  # Fallback to conservative size

    def calculate_optimal_spread(self, mid_price: float) -> float:
        """Calculate optimal spread using enhanced Avellaneda-Stoikov model"""
        try:
            # Base spread from A-S model
            base_spread = (
                self.gamma * self.volatility**2 * self.position_fade_time +
                2/self.gamma * np.log(1 + self.gamma/self.order_flow_intensity)
            )
            
            # Scale spread with price level
            base_spread = base_spread * mid_price
            
            # Adjust for market impact
            impact_adjustment = self.market_impact * mid_price
            
            # Adjust for inventory - handle missing keys gracefully
            position_limit = TRADING_PARAMS.get("position_management", {}).get("position_limit", 50000)
            inventory_ratio = abs(self.position_size) / position_limit
            inventory_adjustment = (
                self.inventory_weight * 
                inventory_ratio * 
                self.tick_size * 
                mid_price
            )
            
            # Calculate final spread
            spread = base_spread + impact_adjustment + inventory_adjustment
            
            # Apply spread limits - handle missing keys gracefully
            min_spread_ticks = ORDERBOOK_CONFIG.get("min_spread", 0.3)
            max_spread_ticks = ORDERBOOK_CONFIG.get("max_spread", 2.0)
            min_spread = min_spread_ticks * self.tick_size * mid_price
            max_spread = max_spread_ticks * self.tick_size * mid_price
            spread = np.clip(spread, min_spread, max_spread)
            
            logging.debug(f"Spread calculation: base={base_spread:.2f}, impact={impact_adjustment:.2f}, inv={inventory_adjustment:.2f}, final={spread:.2f}")
            return spread
            
        except Exception as e:
            logging.error(f"Error calculating optimal spread: {str(e)}")
            default_spread = ORDERBOOK_CONFIG.get("spread", 0.5) * self.tick_size * mid_price
            return default_spread

    def calculate_skewed_prices(
        self,
        mid_price: float,
        spread: float,
        market_conditions: Dict
    ) -> Tuple[float, float]:
        """Calculate skewed bid and ask prices based on inventory and market conditions"""
        try:
            # Base prices
            half_spread = spread / 2
            base_bid = mid_price - half_spread
            base_ask = mid_price + half_spread
            
            # Calculate inventory skew
            inventory_skew = (
                self.position_size * 
                self.inventory_weight * 
                self.tick_size
            )
            
            # Adjust for market conditions
            trend_adjustment = 0.0
            if market_conditions["is_trending"]:
                trend_factor = 0.5 if market_conditions["trend_direction"] else -0.5
                trend_adjustment = trend_factor * market_conditions["trend_strength"] * self.tick_size
            
            # Apply mean reversion adjustment
            mean_reversion_adjustment = 0.0
            if market_conditions["mean_reverting_signal"]:
                mean_reversion_adjustment = -market_conditions["zscore"] * self.tick_size
            
            # Calculate final prices
            bid_price = base_bid - inventory_skew + trend_adjustment + mean_reversion_adjustment
            ask_price = base_ask - inventory_skew + trend_adjustment + mean_reversion_adjustment
            
            # Round to tick size
            bid_price = round(bid_price / self.tick_size) * self.tick_size
            ask_price = round(ask_price / self.tick_size) * self.tick_size
            
            return bid_price, ask_price
            
        except Exception as e:
            print(f"Error calculating skewed prices: {str(e)}")
            return mid_price - spread/2, mid_price + spread/2

    def calculate_quote_sizes(
        self,
        bid_price: float,
        ask_price: float,
        market_conditions: Dict
    ) -> Tuple[List[float], List[float]]:
        """Calculate optimal quote sizes for each level"""
        try:
            # Get base sizes
            base_bid_sizes = ORDERBOOK_CONFIG["bid_sizes"]
            base_ask_sizes = ORDERBOOK_CONFIG["ask_sizes"]
            
            # Adjust for volatility
            vol_factor = 1.0
            if market_conditions["is_volatile"]:
                vol_factor = 0.7  # Reduce size in volatile markets
            
            # Adjust for volume
            volume_factor = 1.0
            if market_conditions["high_volume"]:
                volume_factor = 1.2  # Increase size in high volume
            
            # Adjust for trend
            trend_factor = 1.0
            if market_conditions["is_trending"]:
                # Reduce size against trend, increase with trend
                trend_factor = 1.2 if market_conditions["trend_direction"] else 0.8
            
            # Calculate final sizes
            bid_sizes = [
                size * vol_factor * volume_factor * trend_factor
                for size in base_bid_sizes
            ]
            ask_sizes = [
                size * vol_factor * volume_factor * trend_factor
                for size in base_ask_sizes
            ]
            
            return bid_sizes, ask_sizes
            
        except Exception as e:
            print(f"Error calculating quote sizes: {str(e)}")
            return ORDERBOOK_CONFIG["bid_sizes"], ORDERBOOK_CONFIG["ask_sizes"]

    def generate_quotes(
        self,
        mid_price: float,
        market_conditions: Dict
    ) -> Tuple[List[Quote], List[Quote]]:
        """Generate optimal quotes using Avellaneda-Stoikov model"""
        try:
            # Calculate optimal spread
            spread = self.calculate_optimal_spread(mid_price)
            
            # Calculate skewed prices
            bid_price, ask_price = self.calculate_skewed_prices(
                mid_price, spread, market_conditions
            )
            
            # Calculate quote sizes
            bid_sizes, ask_sizes = self.calculate_quote_sizes(
                bid_price, ask_price, market_conditions
            )
            
            # Generate bid quotes
            bid_quotes = []
            current_bid = bid_price
            for size in bid_sizes:
                bid_quotes.append(Quote(price=current_bid, amount=size))
                current_bid -= ORDERBOOK_CONFIG["bid_step"] * self.tick_size
            
            # Generate ask quotes
            ask_quotes = []
            current_ask = ask_price
            for size in ask_sizes:
                ask_quotes.append(Quote(price=current_ask, amount=size))
                current_ask += ORDERBOOK_CONFIG["ask_step"] * self.tick_size
            
            # Update quote tracking
            self.quote_performance["total_quotes"] += len(bid_quotes) + len(ask_quotes)
            self.last_quote_time = time.time()
            
            return bid_quotes, ask_quotes
            
        except Exception as e:
            print(f"Error generating quotes: {str(e)}")
            return [], []

    def should_update_quotes(
        self,
        current_quotes: Tuple[List[Quote], List[Quote]],
        mid_price: float
    ) -> bool:
        """Determine if quotes should be updated"""
        if not current_quotes[0] and not current_quotes[1]:
            return True
            
        current_time = time.time()
        
        # Check quote lifetime
        if current_time - self.last_quote_time > ORDERBOOK_CONFIG["quote_lifetime"]:
            return True
            
        # Check price movement
        if any(current_quotes[0]):  # If we have bid quotes
            best_bid = current_quotes[0][0].price
            if abs(best_bid - (mid_price - self.calculate_optimal_spread(mid_price)/2)) > ORDERBOOK_CONFIG["amend_threshold"] * self.tick_size:
                return True
                
        if any(current_quotes[1]):  # If we have ask quotes
            best_ask = current_quotes[1][0].price
            if abs(best_ask - (mid_price + self.calculate_optimal_spread(mid_price)/2)) > ORDERBOOK_CONFIG["amend_threshold"] * self.tick_size:
                return True
                
        return False

    def validate_quotes(
        self,
        bid_quotes: List[Quote],
        ask_quotes: List[Quote],
        market_conditions: Dict
    ) -> bool:
        """Validate generated quotes"""
        try:
            if not bid_quotes or not ask_quotes:
                return False
                
            # Check spread
            spread = ask_quotes[0].price - bid_quotes[0].price
            min_spread = ORDERBOOK_CONFIG["min_spread"] * self.tick_size
            if spread < min_spread:
                logging.warning(f"Quote spread {spread} is less than minimum {min_spread}")
                return False
                
            # Check quote sizes - handle missing threshold gracefully
            max_size = ORDERBOOK_CONFIG.get("threshold", 1.0)  # Default to 1.0 if missing
            if any(q.amount > max_size for q in bid_quotes + ask_quotes):
                logging.warning(f"Quote size exceeds maximum {max_size}")
                return False
                
            # Additional checks in volatile markets
            if market_conditions.get("is_volatile", False):  # Handle missing key
                if spread < min_spread * 2:  # Double minimum spread in volatile markets
                    logging.warning("Spread too small for volatile market conditions")
                    return False
                    
            return True
            
        except Exception as e:
            logging.error(f"Error validating quotes: {str(e)}")
            return False

    def calculate_optimal_quotes(self, mid_price: float) -> Tuple[float, float, float, float]:
        """Calculate optimal quotes using enhanced Avellaneda-Stoikov model"""
        try:
            if mid_price <= 0:
                logging.warning(f"Invalid mid price: {mid_price}")
                return 0, 0, 0, 0

            # Calculate position utilization - handle missing keys gracefully
            position_limit = TRADING_PARAMS.get("position_management", {}).get("position_limit", 50000)
            q = self.position_size / position_limit
            
            # Calculate reservation price with dampened skew for large positions
            skew_factor = np.tanh(q)  # Use tanh to dampen extreme positions
            r = mid_price - skew_factor * self.gamma * self.volatility**2 * self.inventory_weight
            
            # Calculate optimal spread in ticks
            base_spread = (
                self.gamma * self.volatility**2 * self.position_fade_time +
                2/self.gamma * np.log(1 + self.gamma/self.order_flow_intensity)
            )
            
            # Convert spread to price units and apply limits
            min_spread_ticks = ORDERBOOK_CONFIG.get("min_spread", 0.3)
            max_spread_ticks = ORDERBOOK_CONFIG.get("max_spread", 2.0)
            
            # Ensure spread is reasonable (in ticks)
            spread_in_ticks = max(min_spread_ticks, min(base_spread, max_spread_ticks))
            
            # Convert to price
            spread = spread_in_ticks * self.tick_size
            
            # Calculate bid and ask prices
            bid_price = self.round_to_tick(r - spread/2)
            ask_price = self.round_to_tick(r + spread/2)
            
            # Ensure minimum distance from mid price
            min_distance = min_spread_ticks * self.tick_size
            bid_price = min(bid_price, mid_price - min_distance)
            ask_price = max(ask_price, mid_price + min_distance)
            
            # Ensure bid and ask are on opposite sides of mid price
            if bid_price >= mid_price:
                bid_price = self.round_to_tick(mid_price - min_distance)
            if ask_price <= mid_price:
                ask_price = self.round_to_tick(mid_price + min_distance)
            
            # Calculate optimal sizes with position awareness
            bid_size = self.calculate_optimal_size("bid", q, self.volatility)
            ask_size = self.calculate_optimal_size("ask", q, self.volatility)
            
            # Log the calculations for debugging
            logging.debug(f"Optimal quotes: bid={bid_price}, ask={ask_price}, bid_size={bid_size}, ask_size={ask_size}")
            logging.debug(f"Market conditions: pos={self.position_size}, vol={self.volatility:.4f}, impact={self.market_impact:.4f}")
            
            return bid_price, ask_price, bid_size, ask_size
            
        except Exception as e:
            logging.error(f"Error calculating optimal quotes: {str(e)}")
            return 0, 0, 0, 0

    def round_to_tick(self, value: float) -> float:
        """Round price to valid tick size"""
        if self.tick_size <= 0:
            return value
        return self.tick_size * round(value / self.tick_size) 