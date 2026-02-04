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
        if abs(projected_total) > max_limit:
            logger.warning(
                f"Order REJECTED: Projected {projected_total:.4f} exceeds limit {max_limit}. "
                f"(Curr: {current_exposure} + Active: {active_exposure_change} + New: {new_impact})"
            )
            return False

        return True

    def _get_limit_for_order(self, order: Order) -> float:
        exchange = getattr(order, "exchange", "")
        if exchange and exchange in self.venue_limits:
            return self.venue_limits[exchange]
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
                limit = self.venue_limits.get(pos.exchange, self.max_position)
                if abs(pos.size) > limit:
                    return False
            return True
        return abs(position.size) <= self.max_position

    def can_trade(self) -> bool:
        return self.enabled
