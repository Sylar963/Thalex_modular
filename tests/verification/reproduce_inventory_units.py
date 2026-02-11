import math


def calculate_inventory_component(price, vol_daily, position_limit, current_pos):
    print(f"--- Inputs ---")
    print(f"Price: {price}")
    print(f"Daily Volatility: {vol_daily} ({vol_daily * 100}%)")
    print(f"Position Limit: {position_limit}")
    print(f"Current Position: {current_pos}")

    # Defaults from code
    inventory_factor = 0.5
    volatility_mult = 1.0  # simplified

    # Logic from avellaneda.py
    # volatility_term = self.volatility * volatility_mult
    volatility_term = vol_daily * volatility_mult

    safe_pos_limit = max(0.001, position_limit)
    inventory_risk = abs(current_pos) / safe_pos_limit
    inventory_risk = min(inventory_risk, 10.0)  # Cap risk

    # Current Implementation
    # inventory_component = inventory_factor * inventory_risk * volatility_term
    current_inv_component = inventory_factor * inventory_risk * volatility_term

    print(f"\n[Current Implementation]")
    print(f"Risk Score: {inventory_risk}")
    print(f"Vol Term: {volatility_term}")
    print(f"Inventory Component (Added to Spread): {current_inv_component}")
    print(f"Spread Impact ($): ${current_inv_component:.4f}")
    if price > 0:
        print(f"Spread Impact (%): {(current_inv_component / price) * 100:.2f}%")

    # Proposed Correction:
    # Scale by price?
    corrected_inv_component = current_inv_component * price

    print(f"\n[Hypothetical Corrected Implementation (Scaled by Price)]")
    print(f"Spread Impact ($): ${corrected_inv_component:.4f}")
    print(f"Spread Impact (%): {(corrected_inv_component / price) * 100:.2f}%")


if __name__ == "__main__":
    # Scenario: Price $0.89, Vol 5%, Position Maxed out (Risk=10 if significantly over limit, or just 1.0 if at limit)
    # If standard Avellaneda, risk q is raw quantity. The code uses ratio `current_pos / limit`.
    # Let's assume we are at 5x limit (Risk=5) or max cap (Risk=10).

    calculate_inventory_component(
        price=0.8971, vol_daily=0.05, position_limit=100, current_pos=500
    )
