from typing import Dict, Any
from ..interfaces import RiskManager
from ..entities import Order, Position


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

    def validate_order(self, order: Order, position: Position) -> bool:
        if order.size <= 0:
            return False

        if order.size > self.max_order_size:
            return False

        # Check projected position
        # Note: This is a loose check.
        # Ideally we check (current_pos + open_orders_same_side + new_order)
        # But for basic domain logic, we assume order size is within bounds relative to current.
        projected_size = (
            position.size + order.size
            if order.side == "buy"
            else position.size - order.size
        )

        if abs(projected_size) > self.max_position:
            return False

        return True

    def check_position_limits(self, position: Position) -> bool:
        return abs(position.size) <= self.max_position

    def can_trade(self) -> bool:
        return self.enabled
