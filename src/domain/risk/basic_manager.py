import logging
from typing import Dict, Any, Optional, List
from ..interfaces import RiskManager
from ..entities import Order, Position, OrderSide

logger = logging.getLogger(__name__)


class BasicRiskManager(RiskManager):
    """
    Basic implementation of logic controls.
    """

    def __init__(self, max_position: float = 10.0, max_order_size: float = 1.0):
        self.max_position = max_position
        self.max_order_size = max_order_size
        self.enabled = True

    def setup(self, config: Dict[str, Any]):
        self.max_position = config.get("max_position", 10.0)
        self.max_order_size = config.get("max_order_size", 1.0)
        self.enabled = config.get("enabled", True)

    def validate_order(
        self,
        order: Order,
        position: Position,
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

        # Calculate projected position with active orders
        current_exposure = position.size

        # Add up all active orders that increase exposure in the same direction
        # OR just simpler conservative approach: worst case exposure

        # Let's verify active_orders
        active_exposure_change = 0.0
        if active_orders:
            for o in active_orders:
                # We only care about orders that add to the position in the direction we are checking?
                # Actually, standard MM risk check is:
                # Max Long = Current Pos + Open Buys + New Buy
                # Max Short = Current Pos - Open Sells - New Sell (absolute)

                if o.side == "buy" or o.side == OrderSide.BUY:
                    if order.side == "buy" or order.side == OrderSide.BUY:
                        active_exposure_change += o.size
                else:  # sell
                    if order.side == "sell" or order.side == OrderSide.SELL:
                        active_exposure_change -= o.size

        # New order impact
        new_impact = (
            order.size
            if (order.side == "buy" or order.side == OrderSide.BUY)
            else -order.size
        )

        projected_total = current_exposure + active_exposure_change + new_impact

        if abs(projected_total) > self.max_position:
            logger.warning(
                f"Order REJECTED: Projected {projected_total:.4f} exceeds limit {self.max_position}. "
                f"(Curr: {current_exposure} + Active: {active_exposure_change} + New: {new_impact})"
            )
            return False

        return True

    def get_diagnostics(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "max_position": self.max_position,
            "max_order_size": self.max_order_size,
        }

    def check_position_limits(self, position: Position) -> bool:
        return abs(position.size) <= self.max_position

    def can_trade(self) -> bool:
        return self.enabled
