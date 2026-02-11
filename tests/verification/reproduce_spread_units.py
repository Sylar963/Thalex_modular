import math


def calculate_spread_component(price, vol_daily, fade_seconds):
    print(f"--- Inputs ---")
    print(f"Price: {price}")
    print(f"Daily Volatility: {vol_daily} ({vol_daily * 100}%)")
    print(f"Fade Time: {fade_seconds} seconds")

    # Current Implementation Logic
    # volatility_component = volatility_term * math.sqrt(self.position_fade_time)
    # assuming volatility_term = vol_daily

    current_vol_component = vol_daily * math.sqrt(fade_seconds)
    print(f"\n[Current Implementation]")
    print(f"Component Value: {current_vol_component}")
    print(f"Is this added to spread ($)? : Yes (based on code review)")
    print(f"Resulting Spread Add-on: ${current_vol_component:.4f}")

    # Proposed Correction:
    # 1. Convert seconds to days (since vol is daily)
    # 2. Multiply by Price (since vol is %)

    fade_days = fade_seconds / 86400.0
    corrected_component_pct = vol_daily * math.sqrt(fade_days)
    corrected_component_usd = corrected_component_pct * price

    print(f"\n[Corrected Implementation]")
    print(f"Fade Days: {fade_days:.4f}")
    print(
        f"Vol over fade window (%): {corrected_component_pct:.4f} ({corrected_component_pct * 100:.2f}%)"
    )
    print(f"Vol over fade window ($): ${corrected_component_usd:.4f}")

    return current_vol_component, corrected_component_usd


if __name__ == "__main__":
    # SUI Params
    calculate_spread_component(price=3.20, vol_daily=0.0566, fade_seconds=3600)
