import logging
from typing import Dict, Any, Optional, List, Union
from ..interfaces import RiskManager
from ..entities import Order, Position, OrderSide, Portfolio

logger = logging.getLogger(__name__)


class BasicRiskManager(RiskManager):
    def __init__(self, max_position: float = 10.0, max_order_size: float = 1.0):
        self.max_position = max_position
        self.max_order_size = max_order_size
        self.enabled = True
        self.venue_limits: Dict[str, float] = {}
        self._breached = False
        self._positions: Dict[str, Position] = {}

    def setup(self, config: Dict[str, Any]):
        self.max_position = config.get("max_position", 10.0)
        self.max_order_size = config.get("max_order_size", 1.0)
        self.enabled = config.get("enabled", True)
        self.venue_limits = config.get("venue_limits", {})

    def validate_order(
        self,
        order: Order,
        position: Union[Position, Portfolio],
        active_orders: Optional[List[Order]] = None,
    ) -> bool:
        if not self.enabled:
            logger.warning("Order REJECTED: Risk Manager is disabled")
            return False

        if order.size <= 0:
            logger.warning(f"Order REJECTED: Invalid size {order.size}")
            return False

        if order.size > self.max_order_size:
            logger.warning(
                f"Order REJECTED: Size {order.size} > Max {self.max_order_size}"
            )
            return False

        if isinstance(position, Portfolio):
            current_exposure = position.get_aggregate_exposure(order.symbol)
        else:
            current_exposure = position.size

        active_exposure_change = 0.0
        if active_orders:
            for o in active_orders:
                if o.side == "buy" or o.side == OrderSide.BUY:
                    if order.side == "buy" or order.side == OrderSide.BUY:
                        active_exposure_change += o.size
                else:
                    if order.side == "sell" or order.side == OrderSide.SELL:
                        active_exposure_change -= o.size

        new_impact = (
            order.size
            if (order.side == "buy" or order.side == OrderSide.BUY)
            else -order.size
        )

        projected_total = current_exposure + active_exposure_change + new_impact

        max_limit = self._get_limit_for_order(order)

        # Add more detailed logging for Thalex
        exchange = getattr(order, "exchange", "")
        if exchange and exchange.lower() == "thalex":
            logger.info(
                f"Thalex risk check - Order: {order.side.value} {order.size}@{order.price}, "
                f"Current: {current_exposure:.4f}, Active change: {active_exposure_change:.4f}, "
                f"New impact: {new_impact:.4f}, Projected: {projected_total:.4f}, Limit: {max_limit:.4f}"
            )

        if abs(projected_total) > max_limit:
            # Check if this order is reducing our exposure (even if it doesn't clear the breach)
            if abs(projected_total) < abs(current_exposure):
                logger.info(
                    f"Risk: Allowing Reducing order even if still in breach. "
                    f"Proj: {projected_total:.4f}, Curr: {current_exposure:.4f}"
                )
                return True

            logger.warning(
                f"Order REJECTED: Projected {projected_total:.4f} exceeds limit {max_limit}. "
                f"(Curr: {current_exposure} + Active: {active_exposure_change} + New: {new_impact})"
            )
            self._breached = True
            return False

        return True

    def update_position(self, position: Position) -> None:
        key = f"{position.exchange}:{position.symbol}"
        self._positions[key] = position

        limit = self._get_limit(position.exchange, position.symbol)
        if abs(position.size) > limit:
            logger.warning(f"Risk Breach: {key} size {position.size} > limit {limit}")
            self._breached = True

    def _get_limit_for_order(self, order: Order) -> float:
        exchange = getattr(order, "exchange", "")
        return self._get_limit(exchange, order.symbol)

    def _get_limit(self, exchange: str, symbol: str) -> float:
        # 1. Try symbol-specific override (e.g., "bybit:BTCUSDT")
        composite_key = f"{exchange}:{symbol}"
        if composite_key in self.venue_limits:
            return self.venue_limits[composite_key]
        
        # 2. Try exchange-wide default
        if exchange in self.venue_limits:
            return self.venue_limits[exchange]
            
        # 3. Fallback to global max_position
        return self.max_position

    def get_diagnostics(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "max_position": self.max_position,
            "max_order_size": self.max_order_size,
            "venue_limits": self.venue_limits,
        }

    def check_position_limits(self, position: Union[Position, Portfolio]) -> bool:
        if isinstance(position, Portfolio):
            for pos in position.all_positions():
                limit = self._get_limit(pos.exchange, pos.symbol)
                if abs(pos.size) > limit:
                    return False
            return True

        # Helper for single position check
        limit = self._get_limit(position.exchange, position.symbol)
        return abs(position.size) <= limit

    def can_trade(self) -> bool:
        return self.enabled and not self._breached

    def has_breached(self) -> bool:
        return self._breached

    def get_risk_state(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "breached": self._breached,
            "max_position": self.max_position,
            "max_order_size": self.max_order_size,
            "venue_limits": self.venue_limits,
            "positions": {k: p.size for k, p in self._positions.items()},
        }

    def reset_breach(self) -> None:
        self._breached = False
        logger.info("Risk breach status reset manually.")
