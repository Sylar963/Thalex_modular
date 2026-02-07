def calculate_budget():
    limit_req_per_sec = 50  # ME Limit with COD
    cancel_req_per_sec = 1000  # Cancel Limit

    levels_per_side = 24
    total_orders = levels_per_side * 2

    # scenario 1: Individual Orders (Old)
    # 1 full refresh = 48 places + 48 cancels = 96 ops
    # ME Limit is bottleneck for places.
    # Max refreshes per second = 50 / 48 ~= 1.04 Hz (Best case, if no other traffic)

    # scenario 2: Batching (Conservative - 1 op = 1 token)
    # Same token usage, but much lower latency/network overhead.
    # Burst capability improved.

    # scenario 3: Smart Delta (Market moves 1 tick)
    # Only spread (2 orders) + maybe 2 neighbors update.
    # 4 updates = 4 tokens.
    # Max refreshes per second = 50 / 4 = 12.5 Hz!

    # scenario 4: Smart Delta (Market crashes - Full Refresh)
    # 48 updates.
    # We can handle 1 burst per second.

    print("--- THALEX BUDGET CALCULATION ---")
    print(f"Total Active Orders: {total_orders} ({levels_per_side} x 2)")
    print(f"ME Rate Limit: {limit_req_per_sec} req/s")

    print("\n[Strategy Sustainability]")
    print(f"Full Refresh Cost: {total_orders} tokens")
    print(f"Max Full Refresh Freq: {limit_req_per_sec / total_orders:.2f} Hz")

    print("\n[Smart Delta Efficiency]")
    print(f"Minor Move Cost (Spread only): 2 tokens")
    print(f"Max Minor Refresh Freq: {limit_req_per_sec / 2:.2f} Hz")

    print("\n[Conclusion]")
    print("With 24 levels, you CANNOT sustain >1Hz full refresh.")
    print("With 'Smart Delta', you can sustain >10Hz for small moves.")
    print(
        "Recommendation: Keep 1Hz base resolution, allow high-freq updates ONLY for BBO."
    )


if __name__ == "__main__":
    calculate_budget()
