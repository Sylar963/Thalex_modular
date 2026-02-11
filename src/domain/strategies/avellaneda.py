import numpy as np
import time
import math
import logging
from typing import Optional, List, Dict, Any
from enum import Enum
from ..interfaces import Strategy
from ..entities import MarketState, Position, Order, OrderSide, OrderType, Ticker

logger = logging.getLogger(__name__)


class TradingMode(str, Enum):
    MARKET_MAKING = "market_making"
    RESCUE = "rescue"


class AvellanedaStoikovStrategy(Strategy):
    """
    Domain implementation of the Avellaneda-Stoikov market making strategy.

    This implementation ports the robust heuristic logic from the legacy
    'avellaneda_market_maker.py' component to ensure behavioral parity
    while fitting into the new clean architecture. It also includes
    regime-aware parameter adjustments.
    """

    __slots__ = [
        "gamma",
        "volatility",
        "base_spread_factor",
        "min_spread_ticks",
        "position_limit",
        "inventory_weight",
        "position_fade_time",
        "volatility_multiplier",
        "market_impact_component",
        "inventory_factor",
        "fee_coverage_multiplier",
        "maker_fee_rate",
        "profit_margin_rate",
        "tick_size",
        "order_size",
        "recalc_interval",
        "last_recalc_time",
    ]

    def __init__(self):
        # Default parameters matching legacy defaults
        self.gamma = 0.1
        self.volatility = 0.05
        self.base_spread_factor = 1.0
        self.min_spread_ticks = 20  # Conservative default
        self.position_limit = 1.0
        self.inventory_weight = 0.5  # Inventory skew vs spread
        self.position_fade_time = 3600
        self.volatility_multiplier = 0.2
        self.market_impact_component = 0.0
        self.inventory_factor = 0.5

        # Fee params
        self.fee_coverage_multiplier = 1.2
        self.maker_fee_rate = 0.0002
        self.profit_margin_rate = 0.0005

        self.tick_size = 0.5
        self.order_size = 0.001
        self.recalc_interval = 0.1
        self.last_recalc_time = 0

        # Multi-level params
        self.quote_levels = 1
        self.level_spacing_factor = 0.5  # Fraction of spread to add for next level

    def setup(self, config: Dict[str, Any]):
        """Initialize strategy parameters from configuration dictionary."""
        # Extract Avellaneda specific config
        av_config = config.get("avellaneda", {})

        self.gamma = config.get("gamma", getattr(self, "gamma", 0.1))
        self.volatility = config.get("volatility", 0.05)
        self.position_limit = config.get("position_limit", 1.0)
        self.order_size = config.get("order_size", 0.001)
        self.min_spread_ticks = config.get("min_spread", 20)

        # Respect tick_size from config if provided
        if "tick_size" in config:
            self.tick_size = float(config["tick_size"])

        self.quote_levels = config.get("quote_levels", 1)

        self.level_spacing_factor = config.get("level_spacing_factor", 0.5)

        # Heuristic Params
        self.base_spread_factor = av_config.get("base_spread_factor", 1.0)
        self.position_fade_time = av_config.get("position_fade_time", 3600)
        self.inventory_weight = av_config.get("inventory_weight", 0.5)
        self.volatility_multiplier = av_config.get("volatility_multiplier", 0.2)
        self.inventory_factor = av_config.get("inventory_factor", 0.5)

        logger.info(
            f"Avellaneda Strategy Configured: PosLimit={self.position_limit}, Gamma={self.gamma}, Vol={self.volatility}"
        )

    # --- Constants for Heuristic Dampening ---
    MAX_INVENTORY_RISK_SCORE = 5.0
    MAX_SKEW_RATIO = 3.0
    DAMPENING_THRESHOLD_RATIO = 1.0

    def calculate_quotes(
        self,
        market_state: MarketState,
        position: Position,
        regime: Any = None,
        tick_size: Optional[float] = None,
        exchange: Optional[str] = None,
    ) -> List[Order]:
        """
        Calculate optimal bid and ask orders using Heuristic Avellaneda-Stoikov logic.

        Key Improvements over legacy:
        - Geometric Dampening: Uses square root scaling for inventory risk ratios > 1.0
          to prevent spread explosion at high leverage (e.g. 10x).
        - Skew Capping: Caps absolute skew ratio to prevent runaway quotes.
        - Unit Corrections: Properly scales volatility and inventory components by price.
        """
        if not market_state.ticker or market_state.ticker.mid_price <= 0:
            return []

        ticker = market_state.ticker
        mid_price = ticker.mid_price
        timestamp = market_state.timestamp or time.time()

        # --- LEAD-LAG ORACLE INTEGRATION ---
        # If a Fair Price signal exists and is fresh, use it as the anchor.
        # Otherwise, fall back to local mid-price.
        fair_price = market_state.signals.get("fair_price")

        # Check freshness? For now assume signal engine handles it or we trust the value.
        # Ideally we check timestamp delta.
        # But 'fair_price' is just a float value here.
        # If we need metadata, we should look for 'fair_price_meta' or similar.

        anchor_price = fair_price if fair_price and fair_price > 0 else mid_price

        if anchor_price != mid_price:
            pass  # We could log this div, but it happens every tick.

        # Determine tick size
        if tick_size:
            self.tick_size = tick_size
        elif ticker.symbol and "BTC" in ticker.symbol:
            self.tick_size = 0.5

        # Apply venue-specific adjustments
        venue_gamma_multiplier = 1.0
        venue_volatility_multiplier = 1.0

        if exchange:
            if exchange.lower() == "thalex":
                # Thalex-specific adjustments for better performance
                venue_gamma_multiplier = 0.8  # Slightly more aggressive spreads
                venue_volatility_multiplier = (
                    1.1  # Adjust for Thalex's volatility characteristics
                )
            elif exchange.lower() == "bybit":
                venue_gamma_multiplier = 1.0
                venue_volatility_multiplier = 1.0

        # --- 0. Regime & Mode Adjustment ---
        gamma = self.gamma * venue_gamma_multiplier
        volatility_mult = self.volatility_multiplier * venue_volatility_multiplier
        inventory_factor = self.inventory_factor

        # Mode Adjustment
        mode = (
            regime.trading_mode
            if regime and hasattr(regime, "trading_mode")
            else TradingMode.MARKET_MAKING
        )

        if mode == TradingMode.RESCUE:
            # In rescue mode, be extremely aggressive with skew to reduce exposure
            inventory_factor *= 5.0
            # Narrow spreads to get filled faster
            gamma *= 0.5

        if regime:
            # Handle regime as a dictionary
            regime_name = regime.get("name", "Quiet")
            vol_delta = regime.get("vol_delta", 0.0)
            is_overpriced = regime.get("is_overpriced", False)

            if regime_name == "Volatile":
                # Widen spreads in high vol
                gamma *= 1.5
                volatility_mult *= 1.5
            elif regime_name == "Trending":
                # Skew heavily against inventory in trends
                inventory_factor *= 2.0
            elif regime_name == "Illiquid":
                # Be more conservative
                gamma *= 1.2
            elif regime_name == "OverpricedVol" or is_overpriced:
                # IV >> RV: Market is paying up for options.
                # Widen spreads to capture premium, but maybe trade more aggressive size?
                # For now, just widen slightly to be safe and capture edge.
                gamma *= 1.25

            # Dynamic Volatility Scaling based on Option Market Implieds
            # If IV is significantly higher than RV (positive vol_delta), we might want to
            # increase our volatility estimate.
            # However, if RV is spiking (Expansionary) and IV hasn't caught up, we are in danger.

            # If vol_delta is negative (RV > EM), it means realized movement is EXCEEDING expected.
            # This is dangerous. Widen significantly.
            if vol_delta < -0.05:  # RV is 5% higher than EM
                gamma *= 1.5
                logger.warning(
                    f"Expansionary Volatility Detected (Delta={vol_delta:.2f}). Widening spreads."
                )

        # --- 1. Spread Calculation (Heuristic) ---

        # A. Fee-based Minimum Spread
        # spread >= price * (2*fee + margin) * multiplier
        fee_based_min = (
            anchor_price
            * (self.maker_fee_rate * 2 + self.profit_margin_rate)
            * self.fee_coverage_multiplier
        )
        hard_min = self.min_spread_ticks * self.tick_size
        fee_coverage_spread = max(hard_min, fee_based_min)

        base_spread = self.base_spread_factor * fee_coverage_spread

        # B. Components
        gamma_component = 1.0 + gamma
        volatility_term = self.volatility * volatility_mult
        # Volatility Component (Corrected Units)
        # Convert fade time to days (since vol is daily)
        # Scale by price (since vol is percentage)
        time_scaling = math.sqrt(self.position_fade_time / 86400.0)
        volatility_component = (volatility_term * anchor_price) * time_scaling

        # Inventory Risk Component
        current_pos = position.size
        safe_pos_limit = max(0.001, self.position_limit)

        # Calculate raw ratio
        raw_risk_ratio = abs(current_pos) / safe_pos_limit

        # Dampened Risk Logic:
        # Linear for ratio <= 1.0 (Normal operation)
        # Square root for ratio > 1.0 (High leverage / Overload)
        # This prevents spread explosion at 10x leverage (Ratio 10.0)
        if raw_risk_ratio <= self.DAMPENING_THRESHOLD_RATIO:
            inventory_risk = raw_risk_ratio
        else:
            # Ratio 10.0 -> 1.0 + sqrt(9.0) = 4.0
            # Reduces impact from 10x to 4x equivalent
            inventory_risk = self.DAMPENING_THRESHOLD_RATIO + math.sqrt(
                raw_risk_ratio - self.DAMPENING_THRESHOLD_RATIO
            )

        inventory_risk = min(
            inventory_risk, self.MAX_INVENTORY_RISK_SCORE
        )  # Cap risk score

        inventory_component = (
            inventory_factor * inventory_risk * volatility_term * anchor_price
        )

        # Market Impact (Placeholder from signals)
        impact_signal = market_state.signals.get("market_impact", 0.0)
        # Assuming impact_signal is a relative factor (e.g. 0.01 for 1% impact)
        # We scale it by price to get dollar impact
        market_impact_comp = impact_signal * 0.5 * (1 + self.volatility) * anchor_price

        toxicity_score = market_state.signals.get("toxicity_score", 0.0)
        toxicity_spread_mult = 1.0 + (toxicity_score * 0.5)

        optimal_spread = (
            base_spread * gamma_component
            + volatility_component
            + market_impact_comp
            + inventory_component
        )

        final_spread = max(optimal_spread, fee_coverage_spread) * toxicity_spread_mult

        # --- 2. Reservation Price / Skew Calculation ---

        # Linear Inventory Skew
        # Shift = (CurrentPos / Limit) * Spread * Factor

        # FIX: Decouple skew from inventory-widened spread to prevent quadratic explosion
        # Use optimal_spread WITHOUT inventory_component as base
        skew_base_spread = (
            base_spread * gamma_component + volatility_component + market_impact_comp
        )
        # Also ensure we respect min spread for skew base
        skew_base_spread = max(skew_base_spread, fee_coverage_spread)

        inventory_skew_factor = self.inventory_weight * 0.5

        # FIX: Cap the skew ratio to prevent extreme leverage from pushing quotes to infinity
        # At 10x leverage, ratio could be 10.0. We cap at 3.0 (heuristic).
        raw_skew_ratio = current_pos / safe_pos_limit
        capped_skew_ratio = max(
            min(raw_skew_ratio, self.MAX_SKEW_RATIO), -self.MAX_SKEW_RATIO
        )

        inventory_skew = capped_skew_ratio * skew_base_spread * inventory_skew_factor

        # Signal Offsets
        res_offset = (
            market_state.signals.get("reservation_price_offset", 0.0) * anchor_price
        )

        # --- 3. Final Prices ---
        half_spread = final_spread / 2.0

        # Anchor price is now potentially FairPrice, not just mid_price
        raw_bid = anchor_price - half_spread - inventory_skew + res_offset
        raw_ask = anchor_price + half_spread - inventory_skew + res_offset

        # --- 4. Rounding & Sanity ---
        bid_price = self._round_to_tick(raw_bid)
        ask_price = self._round_to_tick(raw_ask)

        # Check min spread again after rounding
        if ask_price - bid_price < hard_min:
            center = (ask_price + bid_price) / 2
            bid_price = self._round_to_tick(center - hard_min / 2)
            ask_price = self._round_to_tick(center + hard_min / 2)

        # --- 5. Order Generation (Multi-Level) ---
        orders = []

        half_fee_cover = fee_coverage_spread / 2.0

        suppress_bids = market_state.signals.get("suppress_bids", 0.0)
        suppress_asks = market_state.signals.get("suppress_asks", 0.0)

        bid_l0 = bid_price
        ask_l0 = ask_price

        level_step = final_spread * self.level_spacing_factor

        for i in range(self.quote_levels):
            if current_pos < self.position_limit and suppress_bids < 0.8:
                # Level i
                p_bid = bid_l0 - (i * level_step)
                p_bid = self._round_to_tick(p_bid)

                # Check minimal profitability (distance from mid)
                # If skew is extreme, we might bid above mid? (Not with our skew logic)
                # But just in case:
                if mid_price - p_bid < half_fee_cover:
                    # Too close to mid (or crossed), clamp it
                    p_bid = self._round_to_tick(mid_price - half_fee_cover)

                orders.append(
                    Order(
                        id=f"bid_{i}_{int(timestamp * 1000)}",
                        symbol=ticker.symbol,
                        side=OrderSide.BUY,
                        type=OrderType.LIMIT,
                        price=p_bid,
                        size=self.order_size,
                        timestamp=timestamp,
                        post_only=True,
                    )
                )

            # Ask Logic
            if current_pos > -self.position_limit and suppress_asks < 0.8:
                # Level i
                p_ask = ask_l0 + (i * level_step)
                p_ask = self._round_to_tick(p_ask)

                if p_ask - mid_price < half_fee_cover:
                    p_ask = self._round_to_tick(mid_price + half_fee_cover)

                orders.append(
                    Order(
                        id=f"ask_{i}_{int(timestamp * 1000)}",
                        symbol=ticker.symbol,
                        side=OrderSide.SELL,
                        type=OrderType.LIMIT,
                        price=p_ask,
                        size=self.order_size,
                        timestamp=timestamp,
                        post_only=True,
                    )
                )

        # --- 6. Metrics Capture ---
        self._last_metrics = {
            "timestamp": timestamp,
            "mid_price": mid_price,
            "anchor_price": anchor_price,
            "reservation_price": (raw_ask + raw_bid) / 2.0,  # Approximate Res Price
            "bid_price": bid_price,
            "ask_price": ask_price,
            "spread": final_spread,
            "gamma": gamma,
            "volatility": volatility_component,
            "inventory_risk": inventory_risk,
            "inventory_skew": inventory_skew,
            "position": current_pos,
            "quote_levels": self.quote_levels,
        }

        return orders

    def get_last_metrics(self) -> Dict[str, Any]:
        """
        Return internal metrics from the last calculation cycle.
        Useful for monitoring 'Gamma Band', Inventory Risk, and Reservation Price.
        """
        return getattr(self, "_last_metrics", {})

    def _round_to_tick(self, price: float) -> float:
        return round(price / self.tick_size) * self.tick_size
