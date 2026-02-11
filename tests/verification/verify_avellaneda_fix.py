import math
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from src.domain.strategies.avellaneda import AvellanedaStoikovStrategy
from src.domain.entities import MarketState, Ticker, Position


def verify_fix():
    print("--- Verifying Avellaneda Fix ---")

    # 1. Setup Strategy
    strategy = AvellanedaStoikovStrategy()
    config = {
        "gamma": 0.5,
        "volatility": 0.0566,  # 5.66% daily
        "min_spread": 1,  # Low min spread to see the vol component
        "tick_size": 0.0001,
        "avellaneda": {
            "position_fade_time": 3600,  # 1 hour
            "volatility_multiplier": 1.0,
            "inventory_weight": 0.0,  # Disable inventory effects for this test
            "base_spread_factor": 0.0,  # Disable base spread to isolate components
        },
        "maker_fee_rate": 0.0,
        "profit_margin_rate": 0.0,
    }
    strategy.setup(config)

    # 2. Mock Market State
    price = 3.20
    ticker = Ticker(
        symbol="SUIUSDT",
        bid=price - 0.01,
        ask=price + 0.01,
        bid_size=100.0,
        ask_size=100.0,
        last=price,
        volume=10000.0,
        exchange="bybit",
        timestamp=1000.0,
    )
    market_state = MarketState(ticker=ticker, timestamp=1000)
    position = Position(symbol="SUIUSDT", size=0.0, entry_price=0.0)

    # 3. Calculate Quotes
    orders = strategy.calculate_quotes(market_state, position, exchange="bybit")

    # 4. Inspect internals
    metrics = strategy.get_last_metrics()

    vol_component = metrics.get("volatility")
    spread = metrics.get("spread")

    print(f"Price: {price}")
    print(f"Daily Vol: {strategy.volatility}")
    print(f"Fade Time: {strategy.position_fade_time}")
    print(f"Vol Component (Calculated): {vol_component}")

    # Expected: (0.0566 * 3.20) * sqrt(3600/86400)
    # = 0.18112 * sqrt(0.04166)
    # = 0.18112 * 0.20412
    # = 0.03697

    expected = (0.0566 * 1.0 * price) * math.sqrt(3600 / 86400.0)
    print(f"Expected: {expected}")

    diff = abs(vol_component - expected)
    print(f"Diff: {diff}")

    if diff < 0.001:
        if vol_component < 0.1:
            print("PASS: Volatility component is reasonable and matches expectation.")
        else:
            print(
                "FAIL: Volatility component matches expectation but is suspiciously large (Formula issue?)."
            )
    else:
        print("FAIL: Value mismatch from expected formula.")


if __name__ == "__main__":
    verify_fix()
