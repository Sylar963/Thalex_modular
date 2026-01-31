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
    Pure domain implementation of the Avellaneda-Stoikov market making strategy.
    Optimized for readability and strict type safety.
    """

    __slots__ = [
        "gamma",
        "kappa",
        "volatility",
        "time_horizon",
        "base_spread",
        "min_spread",
        "max_spread",
        "position_limit",
        "inventory_risk_aversion",
        "order_size",
        "tick_size",
        "last_recalc_time",
        "recalc_interval",
    ]

    def __init__(self):
        self.gamma = 0.1
        self.kappa = 1.5
        self.volatility = 0.05
        self.time_horizon = 1.0
        self.base_spread = 0.001
        self.min_spread = 0.001
        self.max_spread = 0.05
        self.position_limit = 1.0
        self.inventory_risk_aversion = 0.1
        self.order_size = 0.001
        self.tick_size = 0.5  # Default, should be updated from instrument
        self.last_recalc_time = 0
        self.recalc_interval = 0.1  # 100ms

    def setup(self, config: Dict[str, Any]):
        """Initialize strategy parameters from configuration dictionary."""
        self.gamma = config.get("gamma", 0.1)
        self.kappa = config.get("kappa", 1.5)
        self.volatility = config.get("volatility", 0.05)
        self.time_horizon = config.get("time_horizon", 1.0)
        self.base_spread = config.get("base_spread", 0.001)
        self.min_spread = config.get("min_spread", 0.001)
        self.max_spread = config.get("max_spread", 0.05)
        self.position_limit = config.get("position_limit", 1.0)
        self.order_size = config.get("order_size", 0.001)
        self.recalc_interval = config.get("recalc_interval", 0.1)
        logger.info(
            f"Avellaneda Strategy configured: gamma={self.gamma}, kappa={self.kappa}"
        )

    def calculate_quotes(
        self, market_state: MarketState, position: Position
    ) -> List[Order]:
        """
        Calculate optimal bid and ask orders based on current market state and position.
        """
        if not market_state.ticker:
            return []

        ticker = market_state.ticker
        mid_price = ticker.mid_price
        timestamp = market_state.timestamp or time.time()

        # Update tick size if available from config context (omitted for purity here, assuming setup sets it or constant)
        # Ideally tick_size comes from Instrument entity passed in setup or context.

        # 1. Calculate Optimal Spread (S)
        spread = self._calculate_optimal_spread(market_state, position)

        # 2. Calculate Reservation Price (r)
        reservation_price = self._calculate_reservation_price(
            mid_price, position, market_state.signals
        )

        # 3. Determine Bid/Ask Prices
        half_spread = spread / 2.0
        optimal_bid = reservation_price - half_spread
        optimal_ask = reservation_price + half_spread

        # 4. Round to Tick Size
        bid_price = self._round_to_tick(optimal_bid)
        ask_price = self._round_to_tick(optimal_ask)

        # 5. Sanity Check: Ensure min spread
        if ask_price - bid_price < self.min_spread:
            center = (ask_price + bid_price) / 2
            bid_price = self._round_to_tick(center - self.min_spread / 2)
            ask_price = self._round_to_tick(center + self.min_spread / 2)

        # 6. Generate Orders
        orders = []

        # Bid Order
        if position.size < self.position_limit:
            orders.append(
                Order(
                    id=f"bid_{int(timestamp * 1000)}",
                    symbol=ticker.symbol,
                    side=OrderSide.BUY,
                    type=OrderType.LIMIT,
                    price=bid_price,
                    size=self.order_size,  # Could be dynamic
                    timestamp=timestamp,
                )
            )

        # Ask Order
        if position.size > -self.position_limit:
            orders.append(
                Order(
                    id=f"ask_{int(timestamp * 1000)}",
                    symbol=ticker.symbol,
                    side=OrderSide.SELL,
                    type=OrderType.LIMIT,
                    price=ask_price,
                    size=self.order_size,  # Could be dynamic
                    timestamp=timestamp,
                )
            )

        return orders

    def _calculate_optimal_spread(
        self, market_state: MarketState, position: Position
    ) -> float:
        """
        Avellaneda-Stoikov spread formula:
        s = gamma * sigma^2 * (T - t) + (2 / gamma) * ln(1 + (gamma / kappa))
        """
        # Extract signals if available
        volatility_adj = market_state.signals.get("volatility_adjustment", 0.0)
        gamma_adj = market_state.signals.get("gamma_adjustment", 0.0)

        effective_sigma = self.volatility * (1 + volatility_adj)
        effective_gamma = self.gamma * (1 + gamma_adj)

        # Simplified time factor (T-t normalized to 1 for infinite horizon approximation or fixed window)
        time_factor = 1.0

        spread = (effective_gamma * (effective_sigma**2) * time_factor) + (
            (2 * np.log(1 + (effective_gamma / self.kappa))) / effective_gamma
        )

        return max(self.min_spread, spread)

    def _calculate_reservation_price(
        self, mid_price: float, position: Position, signals: Dict[str, float]
    ) -> float:
        """
        Reservation price r = s - q * gamma * sigma^2 * (T - t)

        Where:
        s = mid price
        q = current inventory (position size)
        """
        # Inventory skew
        inventory_skew = (
            position.size * self.gamma * (self.volatility**2) * self.time_horizon
        )

        # Predictive Adjustments
        reservation_offset = signals.get("reservation_price_offset", 0.0)

        # VAMP Adjustment (if passed in signals)
        vamp_offset = signals.get("vamp_offset", 0.0)

        return (
            mid_price - inventory_skew + (mid_price * reservation_offset) + vamp_offset
        )

    def _round_to_tick(self, price: float) -> float:
        return round(price / self.tick_size) * self.tick_size
