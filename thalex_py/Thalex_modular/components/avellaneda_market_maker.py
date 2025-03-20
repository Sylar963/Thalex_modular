import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
import time
from collections import deque

from thalex_py.Thalex_modular.config.market_config import TRADING_PARAMS, ORDERBOOK_CONFIG, TECHNICAL_PARAMS, RISK_LIMITS
from thalex_py.Thalex_modular.models.data_models import Quote

class AvellanedaMarketMaker:
    """
    Implementation of the Avellaneda-Stoikov market making model.
    
    This model optimizes quotes based on inventory risk and market conditions,
    using the mathematical framework from the paper:
    "High-frequency Trading in a Limit Order Book" by Avellaneda and Stoikov.
    """
    
    def __init__(self):
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        
        # Avellaneda-Stoikov parameters
        self.gamma = TRADING_PARAMS["position_management"]["gamma"]
        self.inventory_weight = TRADING_PARAMS["position_management"]["inventory_weight"]
        self.position_fade_time = TRADING_PARAMS["position_management"]["position_fade_time"]
        self.order_flow_intensity = TRADING_PARAMS["position_management"]["order_flow_intensity"]
        
        # Market making state
        self.position_size = 0.0
        self.entry_price = None
        self.last_quote_time = 0.0
        self.tick_size = None
        self.min_tick = 0.0
        self.last_position_check = 0
        self.last_quote_update = 0
        self.last_rebalance_time = 0
        self.last_stop_loss = 0
        self.last_take_profit = 0
        self.last_close_attempt = 0
        self.last_emergency_close = 0
        self.orders_lock = None
        self.last_inventory_update = 0
        self.inventory_holding_cost = 0.0
        self.inventory_imbalance = 0.0
        self.price_history = deque(maxlen=100)
        self.sigma = 0.0  # Market volatility (dynamic)
        self.T = 1.0     # Time horizon
        
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
        
        self.logger.info("Avellaneda-Stoikov market maker initialized")

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
        self.volatility = max(volatility, TRADING_PARAMS["volatility"]["vol_floor"])
        self.market_impact = market_impact
        self.logger.debug(f"Market conditions updated: vol={self.volatility:.6f}, impact={market_impact:.6f}")

    def calculate_optimal_spread(self, mid_price: float) -> float:
        """
        Calculate optimal spread using Avellaneda-Stoikov model
        
        The spread is calculated as:
        spread = gamma * sigma^2 * T + (2/gamma) * log(1 + gamma/k)
        
        Where:
        - gamma: risk aversion parameter
        - sigma: volatility
        - T: time horizon
        - k: order flow intensity
        """
        try:
            # Base spread from A-S model with standard gamma (no longer reduced)
            adjusted_gamma = self.gamma  # Use full gamma value for wider spreads
            base_spread = (
                adjusted_gamma * self.volatility**2 * self.position_fade_time +
                2/adjusted_gamma * np.log(1 + adjusted_gamma/self.order_flow_intensity)
            )
            
            # Scale spread with price level - increased impact factor
            base_spread = base_spread * mid_price * 0.2  # Increased from 0.1
            
            # Adjust for market impact with larger factor
            impact_adjustment = self.market_impact * mid_price * 0.3  # Increased from 0.1
            
            # Adjust for inventory with stronger bounds
            position_limit = TRADING_PARAMS["position_management"]["position_limit"]
            inventory_ratio = abs(self.position_size) / position_limit if position_limit > 0 else 0
            inventory_adjustment = (
                self.inventory_weight * 0.2 *  # Increased from 0.1
                inventory_ratio * 
                self.tick_size * 
                mid_price
            )
            
            # Calculate final spread with volatility multiplier
            vol_factor = 1.0 + (self.volatility / TRADING_PARAMS["volatility"]["vol_floor"]) * 0.1
            spread = (base_spread + impact_adjustment + inventory_adjustment) * vol_factor
            
            # Apply wider spread limits - no longer halving the max spread
            min_spread_ticks = max(3, ORDERBOOK_CONFIG["min_spread"])  # Increased minimum spread
            max_spread_ticks = ORDERBOOK_CONFIG["max_spread"]  # Use full maximum spread
            min_spread = min_spread_ticks * self.tick_size
            max_spread = max_spread_ticks * self.tick_size
            spread = np.clip(spread, min_spread, max_spread)
            
            self.logger.info(f"Spread calculation: base={base_spread:.4f}, impact={impact_adjustment:.4f}, vol_factor={vol_factor:.2f}, final={spread:.4f}")
            return spread
            
        except Exception as e:
            self.logger.error(f"Error calculating optimal spread: {str(e)}")
            default_spread = ORDERBOOK_CONFIG["spread"] * self.tick_size  # Use full default spread
            return default_spread

    def calculate_reservation_price(self, mid_price: float) -> float:
        """
        Calculate the reservation price based on inventory position
        
        r = mid_price - inventory_position * gamma * volatility^2 * time_horizon
        """
        try:
            # Calculate inventory risk component
            inventory_risk = self.gamma * self.volatility**2 * self.position_fade_time
            
            # Calculate reservation price (mid price adjusted for inventory)
            reservation_price = mid_price - self.position_size * inventory_risk
            
            # Round to tick size
            reservation_price = self.round_to_tick(reservation_price)
            
            self.logger.debug(f"Reservation price: {reservation_price:.4f} (mid={mid_price:.4f}, pos={self.position_size})")
            return reservation_price
            
        except Exception as e:
            self.logger.error(f"Error calculating reservation price: {str(e)}")
            return mid_price

    def calculate_skewed_prices(
        self,
        mid_price: float,
        spread: float,
        market_conditions: Dict
    ) -> Tuple[float, float]:
        """
        Calculate skewed bid and ask prices based on inventory and market conditions
        """
        try:
            # Calculate reservation price with standard inventory impact
            inventory_risk = self.gamma * self.volatility**2 * self.position_fade_time
            reservation_price = mid_price - self.position_size * inventory_risk * 0.2  # Increased from 0.1
            
            # Base prices from reservation price and spread
            half_spread = spread / 2
            base_bid = reservation_price - half_spread
            base_ask = reservation_price + half_spread
            
            # Adjust for market conditions with stronger impact
            trend_adjustment = 0.0
            if market_conditions.get("is_trending", False):
                trend_factor = 0.3 if market_conditions.get("trend_direction", False) else -0.3  # Increased from 0.2
                trend_strength = market_conditions.get("trend_strength", 0.0)
                trend_adjustment = trend_factor * trend_strength * self.tick_size * 2  # Doubled impact
            
            # Apply mean reversion adjustment with stronger impact
            mean_reversion_adjustment = 0.0
            if market_conditions.get("mean_reverting_signal", False):
                zscore = market_conditions.get("zscore", 0.0)
                mean_reversion_factor = TECHNICAL_PARAMS["zscore"]["mean_reversion_factor"] * 0.2  # Increased from 0.1
                mean_reversion_adjustment = -zscore * mean_reversion_factor * self.tick_size
            
            # Volatility-based additional adjustment
            vol_adjustment = 0.0
            if market_conditions.get("is_volatile", False):
                vol_ratio = min(3.0, self.volatility / TRADING_PARAMS["volatility"]["vol_floor"])
                vol_adjustment = vol_ratio * self.tick_size * 5  # Add volatility premium
                self.logger.debug(f"Adding volatility premium: {vol_adjustment:.4f} ticks")
            
            # Calculate final prices with all adjustments
            bid_price = base_bid + trend_adjustment + mean_reversion_adjustment - vol_adjustment
            ask_price = base_ask + trend_adjustment + mean_reversion_adjustment + vol_adjustment
            
            # Round to tick size
            bid_price = self.round_to_tick(bid_price)
            ask_price = self.round_to_tick(ask_price)
            
            # Ensure minimum spread
            min_spread_ticks = max(3, ORDERBOOK_CONFIG["min_spread"])  # Increased minimum spread
            min_spread = min_spread_ticks * self.tick_size
            
            if ask_price - bid_price < min_spread:
                # Adjust prices to maintain minimum spread
                mid = (ask_price + bid_price) / 2
                half_min_spread = min_spread / 2
                bid_price = self.round_to_tick(mid - half_min_spread)
                ask_price = self.round_to_tick(mid + half_min_spread)
            
            self.logger.debug(f"Skewed prices: bid={bid_price:.4f}, ask={ask_price:.4f} (r={reservation_price:.4f})")
            return bid_price, ask_price
            
        except Exception as e:
            self.logger.error(f"Error calculating skewed prices: {str(e)}")
            half_spread = spread / 2
            return self.round_to_tick(mid_price - half_spread), self.round_to_tick(mid_price + half_spread)

    def calculate_optimal_size(self, side: str, q: float, volatility: float) -> float:
        """
        Calculate optimal size for quotes based on market conditions and inventory
        
        The size is adjusted based on:
        - Current position relative to limits
        - Market volatility
        - Quote side (bid/ask) relative to current position
        """
        try:
            # Get base size from config
            side_key = "bid_sizes" if side == "bid" else "ask_sizes"
            base_sizes = ORDERBOOK_CONFIG.get(side_key, [0.2, 0.8])
            base_size = base_sizes[0] if base_sizes else 0.2  # Use first level size as base
            
            # Adjust for position
            position_limit = TRADING_PARAMS["position_management"]["position_limit"]
            position_utilization = abs(q) / position_limit if position_limit > 0 else 0
            size_scalar = 1.0 - position_utilization  # Reduce size as position grows
            
            # Adjust for volatility
            vol_scalar = 1.0
            vol_floor = TRADING_PARAMS["volatility"]["vol_floor"]
            vol_ceiling = TRADING_PARAMS["volatility"]["vol_ceiling"]
            
            if volatility > vol_floor:
                # Normalize volatility between floor and ceiling
                norm_vol = min(volatility, vol_ceiling) / vol_floor
                vol_scalar = 1.0 / norm_vol  # Inverse relationship with volatility
            
            # Adjust for position direction (reduce size on side that increases position)
            direction_scalar = 1.0
            if (side == "bid" and self.position_size > 0) or (side == "ask" and self.position_size < 0):
                # We already have position in this direction, reduce size
                direction_scalar = max(0.2, 1.0 - abs(self.position_size) / position_limit)
            elif (side == "bid" and self.position_size < 0) or (side == "ask" and self.position_size > 0):
                # We have opposite position, slightly increase size to rebalance
                direction_scalar = min(1.5, 1.0 + abs(self.position_size) / (2 * position_limit))
            
            # Calculate final size
            size = base_size * size_scalar * vol_scalar * direction_scalar
            
            # Apply limits
            max_size = ORDERBOOK_CONFIG.get("threshold", 1.0)
            min_size = 0.01  # Minimum quote size
            size = min(max_size, max(min_size, size))
            
            self.logger.debug(f"Calculated {side} size: {size:.4f} (base={base_size}, pos_util={position_utilization:.2f}, vol={vol_scalar:.2f}, dir={direction_scalar:.2f})")
            return size
            
        except Exception as e:
            self.logger.error(f"Error calculating optimal size: {str(e)}")
            return 0.1  # Fallback to conservative size

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
            if market_conditions.get("is_volatile", False):
                vol_factor = 0.7  # Reduce size in volatile markets
            
            # Adjust for volume
            volume_factor = 1.0
            if market_conditions.get("high_volume", False):
                volume_factor = 1.2  # Increase size in high volume
            
            # Adjust for trend
            trend_factor = 1.0
            if market_conditions.get("is_trending", False):
                # Reduce size against trend, increase with trend
                trend_factor = 1.2 if market_conditions.get("trend_direction", False) else 0.8
            
            # Adjust for inventory
            inventory_factor = 1.0
            position_limit = TRADING_PARAMS["position_management"]["position_limit"]
            if position_limit > 0:
                inventory_ratio = abs(self.position_size) / position_limit
                
                # Reduce size on side that would increase position
                if self.position_size > 0:  # Long position
                    bid_inventory_factor = 1.0 - inventory_ratio
                    ask_inventory_factor = 1.0
                else:  # Short position
                    bid_inventory_factor = 1.0
                    ask_inventory_factor = 1.0 - inventory_ratio
                
                # Apply inventory factors
                bid_sizes = [
                    size * vol_factor * volume_factor * trend_factor * bid_inventory_factor
                    for size in base_bid_sizes
                ]
                ask_sizes = [
                    size * vol_factor * volume_factor * trend_factor * ask_inventory_factor
                    for size in base_ask_sizes
                ]
            else:
                # No position limit, apply uniform factors
                bid_sizes = [
                    size * vol_factor * volume_factor * trend_factor
                    for size in base_bid_sizes
                ]
                ask_sizes = [
                    size * vol_factor * volume_factor * trend_factor
                    for size in base_ask_sizes
                ]
            
            # Ensure minimum size
            min_size = 0.01
            bid_sizes = [max(min_size, size) for size in bid_sizes]
            ask_sizes = [max(min_size, size) for size in ask_sizes]
            
            self.logger.debug(f"Quote sizes - bids: {bid_sizes}, asks: {ask_sizes}")
            return bid_sizes, ask_sizes
            
        except Exception as e:
            self.logger.error(f"Error calculating quote sizes: {str(e)}")
            return ORDERBOOK_CONFIG["bid_sizes"], ORDERBOOK_CONFIG["ask_sizes"]

    def generate_quotes(
        self,
        mid_price: float,
        market_conditions: Dict
    ) -> Tuple[List[Quote], List[Quote]]:
        """Generate optimal quotes using Avellaneda-Stoikov model"""
        try:
            # Check position against risk limits
            current_notional = abs(self.position_size * mid_price)
            max_notional = RISK_LIMITS["max_notional"]
            max_position = RISK_LIMITS["max_position"]
            
            # If we're near limits, only quote on reducing side
            if current_notional >= max_notional * 0.95 or abs(self.position_size) >= max_position * 0.95:
                self.logger.warning(f"Near risk limits - Position: {self.position_size:.3f}, Notional: {current_notional:.2f}")
                if self.position_size > 0:
                    # Only generate ask quotes to reduce long position
                    return [], self._generate_ask_quotes(mid_price, market_conditions)
                else:
                    # Only generate bid quotes to reduce short position
                    return self._generate_bid_quotes(mid_price, market_conditions), []

            # Calculate optimal spread
            spread = self.calculate_optimal_spread(mid_price)
            
            # Calculate skewed prices
            bid_price, ask_price = self.calculate_skewed_prices(
                mid_price, spread, market_conditions
            )
            
            # Generate bid quotes
            bid_quotes = []
            current_bid = bid_price
            for i, size in enumerate(ORDERBOOK_CONFIG["bid_sizes"]):
                # Check if this quote would exceed limits
                potential_position = self.position_size + size
                potential_notional = abs(potential_position * mid_price)
                
                if potential_notional > max_notional or abs(potential_position) > max_position:
                    self.logger.info(f"Skipping bid quote that would exceed limits - Size: {size:.3f}")
                    continue
                
                # Adjust size based on position and volatility
                adjusted_size = self.calculate_optimal_size(
                    "bid", self.position_size, self.volatility
                ) if i == 0 else size
                
                bid_quotes.append(Quote(price=current_bid, amount=adjusted_size))
                current_bid -= ORDERBOOK_CONFIG["bid_step"] * self.tick_size
            
            # Generate ask quotes
            ask_quotes = []
            current_ask = ask_price
            for i, size in enumerate(ORDERBOOK_CONFIG["ask_sizes"]):
                # Check if this quote would exceed limits
                potential_position = self.position_size - size  # Subtract for sells
                potential_notional = abs(potential_position * mid_price)
                
                if potential_notional > max_notional or abs(potential_position) > max_position:
                    self.logger.info(f"Skipping ask quote that would exceed limits - Size: {size:.3f}")
                    continue
                
                # Adjust size based on position and volatility
                adjusted_size = self.calculate_optimal_size(
                    "ask", self.position_size, self.volatility
                ) if i == 0 else size
                
                ask_quotes.append(Quote(price=current_ask, amount=adjusted_size))
                current_ask += ORDERBOOK_CONFIG["ask_step"] * self.tick_size
            
            # Update performance tracking
            self.quote_performance["total_quotes"] += len(bid_quotes) + len(ask_quotes)
            self.last_quote_time = time.time()
            
            self.logger.debug(f"Generated {len(bid_quotes)} bid quotes and {len(ask_quotes)} ask quotes")
            return bid_quotes, ask_quotes
            
        except Exception as e:
            self.logger.error(f"Error generating quotes: {str(e)}")
            # Return minimal quotes in case of error
            return [Quote(price=mid_price * 0.99, amount=0.1)], [Quote(price=mid_price * 1.01, amount=0.1)]

    def _generate_bid_quotes(self, mid_price: float, market_conditions: Dict) -> List[Quote]:
        """Generate only bid quotes for position reduction"""
        spread = self.calculate_optimal_spread(mid_price)
        bid_price, _ = self.calculate_skewed_prices(mid_price, spread, market_conditions)
        
        bid_quotes = []
        current_bid = bid_price
        for i, size in enumerate(ORDERBOOK_CONFIG["bid_sizes"]):
            adjusted_size = self.calculate_optimal_size("bid", self.position_size, self.volatility)
            bid_quotes.append(Quote(price=current_bid, amount=adjusted_size))
            current_bid -= ORDERBOOK_CONFIG["bid_step"] * self.tick_size
        return bid_quotes

    def _generate_ask_quotes(self, mid_price: float, market_conditions: Dict) -> List[Quote]:
        """Generate only ask quotes for position reduction"""
        spread = self.calculate_optimal_spread(mid_price)
        _, ask_price = self.calculate_skewed_prices(mid_price, spread, market_conditions)
        
        ask_quotes = []
        current_ask = ask_price
        for i, size in enumerate(ORDERBOOK_CONFIG["ask_sizes"]):
            adjusted_size = self.calculate_optimal_size("ask", self.position_size, self.volatility)
            ask_quotes.append(Quote(price=current_ask, amount=adjusted_size))
            current_ask += ORDERBOOK_CONFIG["ask_step"] * self.tick_size
        return ask_quotes

    def should_update_quotes(
        self,
        current_quotes: Tuple[List[Quote], List[Quote]],
        mid_price: float
    ) -> bool:
        """Determine if quotes should be updated based on market conditions"""
        if not current_quotes or not current_quotes[0] or not current_quotes[1]:
            return True
            
        # Check time since last quote
        min_quote_interval = ORDERBOOK_CONFIG["min_quote_interval"]
        time_since_last_quote = time.time() - self.last_quote_time
        if time_since_last_quote < min_quote_interval:
            return False
            
        # Check price movement
        best_bid = current_quotes[0][0].price
        best_ask = current_quotes[1][0].price
        current_mid = (best_bid + best_ask) / 2
        
        price_change_pct = abs(current_mid - mid_price) / mid_price
        amend_threshold = ORDERBOOK_CONFIG["amend_threshold"] * self.tick_size / mid_price
        
        # Update if price moved significantly
        if price_change_pct > amend_threshold:
            self.logger.debug(f"Updating quotes due to price movement: {price_change_pct:.6f} > {amend_threshold:.6f}")
            return True
            
        # Check if position changed significantly
        position_check_interval = 5.0  # Check position every 5 seconds
        if time.time() - self.last_position_check > position_check_interval:
            self.last_position_check = time.time()
            return True
            
        return False

    def validate_quotes(
        self,
        bid_quotes: List[Quote],
        ask_quotes: List[Quote],
        market_conditions: Dict
    ) -> Tuple[List[Quote], List[Quote]]:
        """Validate and adjust quotes if necessary"""
        if not bid_quotes or not ask_quotes:
            self.logger.warning("Empty quotes detected in validation")
            return bid_quotes, ask_quotes
            
        try:
            # Ensure bid-ask spread is valid
            best_bid = bid_quotes[0].price
            best_ask = ask_quotes[0].price
            
            if best_bid >= best_ask:
                self.logger.warning(f"Invalid spread detected: bid {best_bid} >= ask {best_ask}")
                mid_price = (best_bid + best_ask) / 2
                min_spread_ticks = ORDERBOOK_CONFIG["min_spread"]
                min_spread = min_spread_ticks * self.tick_size
                
                # Adjust prices to ensure valid spread
                new_bid = self.round_to_tick(mid_price - min_spread/2)
                new_ask = self.round_to_tick(mid_price + min_spread/2)
                
                # Update quotes
                bid_quotes[0].price = new_bid
                ask_quotes[0].price = new_ask
                self.logger.info(f"Adjusted quotes to ensure valid spread: bid={new_bid}, ask={new_ask}")
            
            # Check for extreme volatility
            if market_conditions.get("is_volatile", False) and market_conditions.get("volatility", 0) > TECHNICAL_PARAMS["atr"]["threshold"] * 3:
                # Reduce sizes in extreme volatility
                vol_reduction = 0.5
                for quote in bid_quotes:
                    quote.amount *= vol_reduction
                for quote in ask_quotes:
                    quote.amount *= vol_reduction
                self.logger.info(f"Reduced quote sizes due to extreme volatility")
            
            return bid_quotes, ask_quotes
            
        except Exception as e:
            self.logger.error(f"Error validating quotes: {str(e)}")
            return bid_quotes, ask_quotes

    def round_to_tick(self, value: float) -> float:
        """Round price to nearest tick size"""
        if self.tick_size <= 0:
            return value
        return round(value / self.tick_size) * self.tick_size 