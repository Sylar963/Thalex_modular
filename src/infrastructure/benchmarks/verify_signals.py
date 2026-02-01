import time
import numpy as np
from src.domain.signals.volume_candle import VolumeCandleSignalEngine
from src.domain.entities import Trade, OrderSide


def verify_signals():
    print("=== Signal Engine Verification (VAMP & Regime) ===")

    # Setup Engine
    engine = VolumeCandleSignalEngine(volume_threshold=100.0, max_candles=10)

    # 1. Simulate Buying Pressure (Trending Up)
    print("\n--- Scenario 1: Aggressive Buying (Trend Up) ---")
    start_price = 10000.0
    for i in range(150):  # 150 trades
        price = start_price + (i * 0.5)  # Price climbing
        trade = Trade(
            id=f"t_{i}",
            order_id="ext_order_id",
            symbol="BTC-PERP",
            price=price,
            size=2.0 + (i % 3),  # Variable size
            side=OrderSide.BUY if i % 10 != 0 else OrderSide.SELL,  # 90% Buys
            timestamp=time.time() + i,
        )
        engine.update_trade(trade)

    signals = engine.get_signals()
    print(f"Market Impact (Exp > 0): {signals['market_impact']:.4f}")
    print(f"Res Price Offset (Exp > 0): {signals['reservation_price_offset']:.6f}")
    print(f"Gamma Adj (Exp != 0): {signals['gamma_adjustment']:.4f}")
    print(f"VAMP Value: {signals['vamp_value']:.2f}")

    # 2. Simulate High Volatility (Choppy)
    print("\n--- Scenario 2: High Volatility (Choppy) ---")
    current_price = 10000.0
    for i in range(200):
        # Sine wave price action
        current_price = 10000.0 + (np.sin(i / 5) * 50.0)
        trade = Trade(
            id=f"t_v_{i}",
            order_id="ext_order_id_v",
            symbol="BTC-PERP",
            price=current_price,
            size=5.0,
            side=OrderSide.BUY if i % 2 == 0 else OrderSide.SELL,
            timestamp=time.time() + 200 + i,
        )
        engine.update_trade(trade)

    signals = engine.get_signals()
    print(f"Volatility (Exp High): {signals['volatility']:.4f}")
    print(f"Gamma Adj (Exp > 0 for Vol): {signals['gamma_adjustment']:.4f}")

    print("\nVerification Complete.")


if __name__ == "__main__":
    verify_signals()
