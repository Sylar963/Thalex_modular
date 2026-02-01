import time
from src.domain.tracking.position_tracker import PositionTracker, PortfolioTracker, Fill


def verify_position_tracker():
    tracker = PositionTracker()
    port_tracker = PortfolioTracker()

    # Test 1: Simple Buy/Sell (Profit)
    print("Test 1: Long 1.0 @ 10000, Sell 1.0 @ 10100")
    tracker.update_on_fill(Fill("1", 10000, 1.0, time.time(), "buy"))
    tracker.update_on_fill(Fill("2", 10100, 1.0, time.time(), "sell"))

    realized = tracker.realized_pnl
    print(f"Realized PnL: {realized} (Exp: 100.0)")

    # Test 2: FIFO Accounting
    tracker.reset()
    print("\nTest 2: FIFO - Buy 1@10k, Buy 1@11k, Sell 1@12k")
    tracker.update_on_fill(Fill("3", 10000, 1.0, time.time(), "buy"))
    tracker.update_on_fill(Fill("4", 11000, 1.0, time.time(), "buy"))
    tracker.update_on_fill(Fill("5", 12000, 1.0, time.time(), "sell"))

    print(f"Realized PnL: {tracker.realized_pnl} (Exp: 2000.0 from first lot)")
    print(f"Remaining Pos: {tracker.current_position}")
    print(f"Avg Entry: {tracker.average_entry_price} (Exp: 11000.0)")


if __name__ == "__main__":
    verify_position_tracker()
