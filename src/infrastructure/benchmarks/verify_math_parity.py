import asyncio
import time
import math
import numpy as np
from typing import Dict, List, Any
from dataclasses import dataclass

# Import New Strategy
from src.domain.strategies.avellaneda import AvellanedaStoikovStrategy
from src.domain.entities import MarketState, Position, Ticker, OrderSide


# Mock Entities
@dataclass
class MockTicker:
    symbol: str
    bid: float
    ask: float
    last: float
    timestamp: float

    @property
    def mid_price(self):
        return (self.bid + self.ask) / 2.0


class LegacyAvellanedaLogic:
    """
    Replication of the heuristic logic found in Thalex_modular/components/avellaneda_market_maker.py
    for comparison purposes.
    """

    def __init__(self, config: Dict[str, Any]):
        self.gamma = config.get("gamma", 0.1)
        self.volatility = config.get("volatility", 0.05)
        self.base_spread_factor = config.get("base_spread_factor", 1.0)
        self.fee_coverage_spread = 10.0  # simplified
        self.market_impact_component = 0.0  # simplified
        self.inventory_factor = 0.5
        self.position_limit = config.get("position_limit", 1.0)
        self.inventory_weight = 0.5

        # Derived for heuristic
        self.volatility_multiplier = 0.2
        self.position_fade_time = 3600

    def calculate_quotes(self, mid_price: float, position: float) -> tuple:
        # 1. Spread Calculation (Heuristic)
        gamma_component = 1.0 + self.gamma
        volatility_term = self.volatility * self.volatility_multiplier
        volatility_component = volatility_term * math.sqrt(self.position_fade_time)

        inventory_risk = abs(position) / max(0.001, self.position_limit)
        inventory_risk = min(inventory_risk, 10.0)
        inventory_component = self.inventory_factor * inventory_risk * volatility_term

        base_spread = self.base_spread_factor * self.fee_coverage_spread

        optimal_spread = (
            base_spread * gamma_component
            + volatility_component
            + self.market_impact_component
            + inventory_component
        )

        # 2. Skew Calculation
        inventory_skew_factor = self.inventory_weight * 0.5
        inventory_skew = (
            (position / self.position_limit) * optimal_spread * inventory_skew_factor
        )

        bid_price = mid_price - (optimal_spread / 2) - inventory_skew
        ask_price = mid_price + (optimal_spread / 2) - inventory_skew

        return bid_price, ask_price, optimal_spread


async def main():
    print("=== Avellaneda Strategy Verification ===")
    print("Comparing New (Theoretical) vs Old (Heuristic) models\n")

    # Configuration
    config = {
        "gamma": 0.5,
        "kappa": 1.5,
        "volatility": 0.02,  # 2% daily vol
        "time_horizon": 1 / 24,  # 1 hour
        "position_limit": 10.0,
        "min_spread": 5.0,
    }

    # 1. Setup New Strategy
    new_strat = AvellanedaStoikovStrategy()
    new_strat.setup(config)

    # 2. Setup Old Logic
    old_logic = LegacyAvellanedaLogic(config)

    # 3. Test Scenarios
    scenarios = [
        {"desc": "Flat Position", "pos": 0.0},
        {"desc": "Long Small", "pos": 2.0},
        {"desc": "Long Heavy", "pos": 8.0},
        {"desc": "Short Small", "pos": -2.0},
        {"desc": "Short Heavy", "pos": -8.0},
    ]

    price = 10000.0
    ticker = Ticker("BTC-PERP", 9995, 10005, 10000, 10000, 10000, 100, time.time())

    print(
        f"{'Scenario':<15} | {'Model':<10} | {'Bid':<10} | {'Ask':<10} | {'Spread':<10} | {'Mid':<10}"
    )
    print("-" * 80)

    for sc in scenarios:
        pos_val = sc["pos"]
        position = Position("BTC-PERP", pos_val, pos_val * price)
        ms = MarketState(ticker=ticker)

        # Calculate New
        new_orders = new_strat.calculate_quotes(ms, position)
        new_bid = next((o.price for o in new_orders if o.side == OrderSide.BUY), 0)
        new_ask = next((o.price for o in new_orders if o.side == OrderSide.SELL), 0)
        new_spread = new_ask - new_bid if (new_bid and new_ask) else 0
        new_mid = (new_bid + new_ask) / 2 if (new_bid and new_ask) else 0

        # Calculate Old
        old_bid, old_ask, old_spread = old_logic.calculate_quotes(price, pos_val)
        old_mid = (old_bid + old_ask) / 2

        print(
            f"{sc['desc']:<15} | {'NEW':<10} | {new_bid:<10.2f} | {new_ask:<10.2f} | {new_spread:<10.2f} | {new_mid:<10.2f}"
        )
        print(
            f"{'Pos: ' + str(pos_val):<15} | {'OLD':<10} | {old_bid:<10.2f} | {old_ask:<10.2f} | {old_spread:<10.2f} | {old_mid:<10.2f}"
        )
        print("-" * 80)


if __name__ == "__main__":
    asyncio.run(main())
