import numpy as np
import time
import math
import logging
from typing import List, Dict, Tuple, Any, Optional
from ..interfaces import Strategy
from ..entities import MarketState, Position, Order, OrderSide, OrderType, Ticker

logger = logging.getLogger(__name__)


class AvellanedaStoikovStrategy(Strategy):
    """
    Domain implementation of the Avellaneda-Stoikov market making strategy.

    This implementation ports the robust heuristic logic from the legacy
    'avellaneda_market_maker.py' component to ensure behavioral parity
    while fitting into the new clean architecture.
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

    def setup(self, config: Dict[str, Any]):
        """Initialize strategy parameters from configuration dictionary."""
        # Extract Avellaneda specific config
        av_config = config.get("avellaneda", {})

        self.gamma = config.get("gamma", getattr(self, "gamma", 0.1))
        self.volatility = config.get("volatility", 0.05)
        self.position_limit = config.get("position_limit", 1.0)
        self.order_size = config.get("order_size", 0.001)
        self.min_spread_ticks = config.get("min_spread", 20)

        # Heuristic Params
        self.base_spread_factor = av_config.get("base_spread_factor", 1.0)
        self.position_fade_time = av_config.get("position_fade_time", 3600)
        self.inventory_weight = av_config.get("inventory_weight", 0.5)
        self.volatility_multiplier = av_config.get("volatility_multiplier", 0.2)
        self.inventory_factor = av_config.get("inventory_factor", 0.5)

        logger.info(
            f"Avellaneda Strategy Configured: PosLimit={self.position_limit}, Gamma={self.gamma}, Vol={self.volatility}"
        )

    def calculate_quotes(
        self, market_state: MarketState, position: Position
    ) -> List[Order]:
        """
        Calculate optimal bid and ask orders using legacy heuristic logic.
        """
        if not market_state.ticker:
            return []

        ticker = market_state.ticker
        mid_price = ticker.mid_price
        timestamp = market_state.timestamp or time.time()

        # Determine tick size from ticker if possible, else default
        if ticker.symbol and "BTC" in ticker.symbol:
            self.tick_size = 0.5

        # --- 1. Spread Calculation (Heuristic) ---

        # A. Fee-based Minimum Spread
        # spread >= price * (2*fee + margin) * multiplier
        fee_based_min = (
            mid_price
            * (self.maker_fee_rate * 2 + self.profit_margin_rate)
            * self.fee_coverage_multiplier
        )
        hard_min = self.min_spread_ticks * self.tick_size
        fee_coverage_spread = max(hard_min, fee_based_min)

        base_spread = self.base_spread_factor * fee_coverage_spread

        # B. Components
        gamma_component = 1.0 + self.gamma
        volatility_term = self.volatility * self.volatility_multiplier
        volatility_component = volatility_term * math.sqrt(self.position_fade_time)

        # Inventory Risk Component
        current_pos = position.size
        safe_pos_limit = max(0.001, self.position_limit)
        inventory_risk = abs(current_pos) / safe_pos_limit
        inventory_risk = min(inventory_risk, 10.0)  # Cap risk

        inventory_component = self.inventory_factor * inventory_risk * volatility_term

        # Market Impact (Placeholder from signals)
        impact_signal = market_state.signals.get("market_impact", 0.0)
        market_impact_comp = (
            impact_signal * 0.5 * (1 + self.volatility)
        )  # Simplified from legacy

        # Total Optimal Spread
        optimal_spread = (
            base_spread * gamma_component
            + volatility_component
            + market_impact_comp
            + inventory_component
        )

        # Ensure at least fee coverage
        final_spread = max(optimal_spread, fee_coverage_spread)

        # --- 2. Reservation Price / Skew Calculation ---

        # Linear Inventory Skew
        # Shift = (CurrentPos / Limit) * Spread * Factor
        # Positive Pos (Long) -> Shift > 0 -> Lower Prices (Sell harder, Buy less)
        # But wait:
        # If Long: logic says "lower prices" to dump inventory?
        # Logic in legacy:
        # bid = mid - half - skew
        # ask = mid + half - skew
        # If skew > 0 (Long), Bid drops, Ask drops. Correct.

        inventory_skew_factor = self.inventory_weight * 0.5
        inventory_skew = (
            (current_pos / safe_pos_limit) * final_spread * inventory_skew_factor
        )

        # Signal Offsets
        res_offset = (
            market_state.signals.get("reservation_price_offset", 0.0) * mid_price
        )

        # --- 3. Final Prices ---
        half_spread = final_spread / 2.0

        raw_bid = mid_price - half_spread - inventory_skew + res_offset
        raw_ask = mid_price + half_spread - inventory_skew + res_offset

        # --- 4. Rounding & Sanity ---
        bid_price = self._round_to_tick(raw_bid)
        ask_price = self._round_to_tick(raw_ask)

        # Check min spread again after rounding
        if ask_price - bid_price < hard_min:
            center = (ask_price + bid_price) / 2
            bid_price = self._round_to_tick(center - hard_min / 2)
            ask_price = self._round_to_tick(center + hard_min / 2)

        # --- 5. Order Generation ---
        orders = []

        # Bid
        if current_pos < self.position_limit:
            orders.append(
                Order(
                    id=f"bid_{int(timestamp * 1000)}",
                    symbol=ticker.symbol,
                    side=OrderSide.BUY,
                    type=OrderType.LIMIT,
                    price=bid_price,
                    size=self.order_size,
                    timestamp=timestamp,
                )
            )

        # Ask
        if current_pos > -self.position_limit:
            orders.append(
                Order(
                    id=f"ask_{int(timestamp * 1000)}",
                    symbol=ticker.symbol,
                    side=OrderSide.SELL,
                    type=OrderType.LIMIT,
                    price=ask_price,
                    size=self.order_size,
                    timestamp=timestamp,
                )
            )

        return orders

    def _round_to_tick(self, price: float) -> float:
        return round(price / self.tick_size) * self.tick_size
