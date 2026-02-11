import math
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from src.domain.strategies.avellaneda import AvellanedaStoikovStrategy
from src.domain.entities import MarketState, Ticker, Position


def verify_skew_fix():
    print("--- Verifying Skew Fix ---")

    # Setup Strategy
    strategy = AvellanedaStoikovStrategy()
    config = {
        "gamma": 0.0,  # disable gamma for clarity
        "volatility": 0.05,
        "min_spread": 0,
        "tick_size": 0.0001,
        "avellaneda": {
            "position_fade_time": 3600,
            "volatility_multiplier": 1.0,
            "inventory_weight": 0.5,  # Factor 0.5
            "base_spread_factor": 1.0,
            "inventory_factor": 0.5,
        },
        "maker_fee_rate": 0.0,
        "profit_margin_rate": 0.0,
    }
    strategy.setup(config)

    # Scenario 1: Short 10x leverage on SUI ($0.89)
    # Pos = -1000, Limit = 100.
    test_cases = [
        {"symbol": "SUIUSDT", "price": 0.8971, "pos": -1000.0, "limit": 100.0},
        {"symbol": "BTCUSDT", "price": 90000.0, "pos": -1.0, "limit": 0.1},
    ]

    for case in test_cases:
        price = case["price"]
        pos = case["pos"]
        limit = case["limit"]
        symbol = case["symbol"]

        print(f"\nScenario: {symbol} @ ${price}")

        ticker = Ticker(
            symbol=symbol,
            bid=price - 0.0001,
            ask=price + 0.0001,
            bid_size=100.0,
            ask_size=100.0,
            last=price,
            volume=10000.0,
            exchange="bybit",
            timestamp=1000.0,
        )
        market_state = MarketState(ticker=ticker, timestamp=1000)
        strategy.position_limit = limit

        position = Position(symbol=symbol, size=pos, entry_price=price)

        # Calculate
        strategy.calculate_quotes(market_state, position, exchange="bybit")
        metrics = strategy.get_last_metrics()

        print(f"Position: {pos} (Limit {limit}) -> Ratio: {pos / limit:.1f}")
        print(f"Final Spread: {metrics['spread']:.4f}")
        print(f"Skew: {metrics['inventory_skew']:.4f}")
        print(f"Bid: {metrics['bid_price']:.4f}")
        print(f"Ask: {metrics['ask_price']:.4f}")

        anchor = metrics["anchor_price"]
        ask_dist_abs = metrics["ask_price"] - anchor
        ask_dist_pct = (ask_dist_abs / anchor) * 100

        print(f"Ask Distance: ${ask_dist_abs:.4f} ({ask_dist_pct:.2f}%)")

        # Logic Check
        # SUI ($0.89) -> Dist should be small (previously 0.80)
        # BTC ($90k) -> Dist should be proportional

        # With skew capped at 3.0 and decoupled from inventory spread:
        # We expect the Ask to be pushed away, but not insanely.

        if symbol == "SUIUSDT":
            if ask_dist_abs < 0.20:
                print("PASS: SUI Ask distance is reasonable (< $0.20)")
            else:
                print(f"FAIL: SUI Ask distance is still too high (${ask_dist_abs:.4f})")

        if symbol == "BTCUSDT":
            # Should be similar percentage
            if 1.0 < ask_dist_pct < 5.0:
                print(f"PASS: BTC Ask distance is reasonable % ({ask_dist_pct:.2f}%)")


if __name__ == "__main__":
    verify_skew_fix()
