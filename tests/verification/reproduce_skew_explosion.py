import math


def calculate_skew_explosion(price, vol_daily, position_limit, current_pos):
    print(f"--- Inputs ---")
    print(f"Price: {price}")
    print(f"Position Limit: {position_limit}")
    print(f"Current Position: {current_pos}")

    # Ratios
    safe_pos_limit = max(0.001, position_limit)
    risk_ratio = current_pos / safe_pos_limit
    abs_risk_ratio = abs(risk_ratio)

    # --- 1. Spread Calculation ---
    # Simplified from Avellaneda

    base_spread_factor = 1.0
    fee_min = price * 0.001  # 0.1% fee spread
    base_spread = base_spread_factor * fee_min

    # Components
    vol_term = vol_daily  # simplified mult=1.0
    # Vol Component (Assuming Fixed units)
    vol_component = (vol_term * price) * math.sqrt(3600 / 86400)

    # Inventory Component (Fixed units)
    inventory_factor = 0.5
    # Capped at 10 in spread calculation
    capped_risk = min(abs_risk_ratio, 10.0)
    inventory_component = inventory_factor * capped_risk * vol_term * price

    # Final Spread
    optimal_spread = base_spread + vol_component + inventory_component
    final_spread = max(optimal_spread, fee_min)

    print(f"\n[Spread Calc]")
    print(f"Base Spread: {base_spread:.4f}")
    print(f"Vol Component: {vol_component:.4f}")
    print(f"Inventory Comp (Risk={capped_risk}): {inventory_component:.4f}")
    print(f"Final Spread: {final_spread:.4f}")

    # --- 2. Skew Calculation ---
    # inventory_skew = (current_pos / safe_pos_limit) * final_spread * inventory_skew_factor
    inventory_weight = 0.5
    inventory_skew_factor = inventory_weight * 0.5  # 0.25

    # Issue: risk_ratio is NOT capped here in current code
    skew = risk_ratio * final_spread * inventory_skew_factor

    print(f"\n[Skew Calc]")
    print(f"Risk Ratio (Uncapped): {risk_ratio:.2f}")
    print(f"Skew Factor: {inventory_skew_factor}")
    print(f"Skew: {skew:.4f}")

    # --- 3. Prices ---
    anchor = price
    half_spread = final_spread / 2.0

    # Logic: bid = anchor - half - skew
    # If Long (risk > 0): Skew > 0. Bid lowers, Ask lowers.
    # If Short (risk < 0): Skew < 0. Bid raises. Ask raises.

    raw_bid = anchor - half_spread - skew
    raw_ask = anchor + half_spread - skew

    print(f"\n[Prices]")
    print(f"Anchor: {anchor}")
    print(f"Raw Bid: {raw_bid:.4f} (Dist: {anchor - raw_bid:.4f})")
    print(f"Raw Ask: {raw_ask:.4f} (Dist: {raw_ask - anchor:.4f})")

    if risk_ratio < 0:  # Short
        dist_ask = raw_ask - anchor
        print(f"\nShort Scenario: Ask Distance = {dist_ask:.4f}")
        if dist_ask > 0.30:
            print("ALERT: Ask is massive distance (>0.30) away!")

    if risk_ratio > 0:  # Long
        dist_bid = anchor - raw_bid
        print(f"\nLong Scenario: Bid Distance = {dist_bid:.4f}")


if __name__ == "__main__":
    # Scenario: Short 10x leverage on SUI ($0.89)
    # Pos = -1000, Limit = 100.
    calculate_skew_explosion(
        price=0.8971, vol_daily=0.0566, position_limit=100, current_pos=-1000
    )
