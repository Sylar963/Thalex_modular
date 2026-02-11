import math
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from src.domain.strategies.avellaneda import AvellanedaStoikovStrategy
from src.domain.entities import MarketState, Ticker, Position


def verify_inventory_dampening():
    print("--- Verifying Inventory Dampening ---")

    # Setup Strategy
    strategy = AvellanedaStoikovStrategy()
    config = {
        "gamma": 0.0,  # Disable gamma for cleaner spread check
        "volatility": 0.05,
        "min_spread": 0,  # Disable min spread
        "tick_size": 0.0001,
        "avellaneda": {
            "position_fade_time": 3600,
            "volatility_multiplier": 1.0,
            "inventory_weight": 0.0,
            "base_spread_factor": 0.0,
            "inventory_factor": 0.5,
        },
        "maker_fee_rate": 0.0,
        "profit_margin_rate": 0.0,
    }
    strategy.setup(config)

    # Scenarios
    # 1. Normal (0.5x risk) -> Linear
    # 2. High (5x risk) -> Dampened: 1 + sqrt(4) = 3.0 (vs 5.0)
    # 3. Max (10x risk) -> Dampened: 1 + sqrt(9) = 4.0 (vs 10.0)

    scenarios = [
        {"symbol": "NORMAL", "price": 100.0, "pos": 5.0, "limit": 10.0},  # Ratio 0.5
        {"symbol": "HIGH_5X", "price": 100.0, "pos": 50.0, "limit": 10.0},  # Ratio 5.0
        {
            "symbol": "MAX_10X",
            "price": 100.0,
            "pos": 100.0,
            "limit": 10.0,
        },  # Ratio 10.0
    ]

    for sc in scenarios:
        price = sc["price"]
        limit = sc["limit"]
        pos = sc["pos"]

        raw_ratio = pos / limit

        # Expected Risk Score (Dampened)
        if raw_ratio <= 1.0:
            exp_risk = raw_ratio
        else:
            exp_risk = 1.0 + math.sqrt(raw_ratio - 1.0)

        # Expected Spread Component
        # inv_factor(0.5) * risk * vol(0.05) * price
        exp_spread_add = 0.5 * exp_risk * 0.05 * price
        exp_spread_pct = (exp_spread_add / price) * 100

        # Expected Linear Spread (Old)
        linear_risk = raw_ratio
        linear_spread_add = 0.5 * linear_risk * 0.05 * price
        linear_spread_pct = (linear_spread_add / price) * 100

        print(f"\nScenario: {sc['symbol']} (Ratio {raw_ratio:.1f})")
        print(
            f"Expected Risk Score: {exp_risk:.2f} (Linear would be {linear_risk:.1f})"
        )
        print(
            f"Expected Spread Add: {exp_spread_pct:.2f}% (Linear would be {linear_spread_pct:.2f}%)"
        )

        # Run Strategy
        ticker = Ticker(
            symbol=sc["symbol"],
            bid=price - 0.01,
            ask=price + 0.01,
            bid_size=10,
            ask_size=10,
            last=price,
            volume=1000,
            exchange="bybit",
            timestamp=1000,
        )
        market_state = MarketState(ticker=ticker, timestamp=1000)
        strategy.position_limit = limit
        position = Position(symbol=sc["symbol"], size=pos, entry_price=price)

        strategy.calculate_quotes(market_state, position, exchange="bybit")
        metrics = strategy.get_last_metrics()

        actual_spread = metrics["spread"]
        actual_risk = metrics["inventory_risk"]

        print(f"ACTUAL Risk Score: {actual_risk:.2f}")
        print(
            f"ACTUAL Spread: {actual_spread:.4f} ({(actual_spread / price) * 100:.2f}%)"
        )

        if abs(actual_risk - exp_risk) < 0.1:
            print("PASS: Risk score matches dampened curve.")
        else:
            print("FAIL: Risk score mismatch.")


if __name__ == "__main__":
    verify_inventory_dampening()
