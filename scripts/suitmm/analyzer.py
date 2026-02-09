import math
import numpy as np
from typing import Dict, List, Tuple


class MarketAnalyzer:
    def __init__(self, maker_fee=0.0002, taker_fee=0.00055):
        self.maker_fee = maker_fee
        self.taker_fee = taker_fee

    def calculate_volatility(self, klines: List) -> Tuple[float, float, float]:
        closes = [float(k[4]) for k in klines if k[4]]
        if len(closes) < 2:
            return 0.0, 0.0, 0.0

        returns = []
        for i in range(1, len(closes)):
            if closes[i - 1] > 0:
                returns.append(math.log(closes[i] / closes[i - 1]))

        if not returns:
            return 0.0, 0.0, 0.0

        hourly_vol = (sum(r**2 for r in returns) / len(returns)) ** 0.5
        daily_vol = hourly_vol * math.sqrt(24)
        annual_vol = hourly_vol * math.sqrt(24 * 365)

        return hourly_vol, daily_vol, annual_vol

    def calculate_volume_profile(self, klines: List) -> Tuple[float, float]:
        volumes_usd = []
        for k in klines:
            close_price = float(k[4])
            volume = float(k[5])
            volumes_usd.append(close_price * volume)

        if not volumes_usd:
            return 0.0, 0.0

        hourly_avg = sum(volumes_usd) / len(volumes_usd)
        daily_avg = hourly_avg * 24
        return hourly_avg, daily_avg

    def analyze_orderbook(self, ob: Dict) -> Dict:
        bids = ob.get("b", [])
        asks = ob.get("a", [])

        if not bids or not asks:
            return {}

        best_bid = float(bids[0][0])
        best_ask = float(asks[0][0])
        mid = (best_bid + best_ask) / 2
        native_spread = best_ask - best_bid
        native_spread_pct = (native_spread / mid) * 100

        # Depth Analysis
        def get_depth(orders, limit_cnt=5):
            return sum(float(o[1]) * float(o[0]) for o in orders[:limit_cnt])

        return {
            "best_bid": best_bid,
            "best_ask": best_ask,
            "mid": mid,
            "native_spread": native_spread,
            "native_spread_pct": native_spread_pct,
            "bid_depth_5_usd": get_depth(bids, 5),
            "ask_depth_5_usd": get_depth(asks, 5),
            "bid_depth_10_usd": get_depth(bids, 10),
            "ask_depth_10_usd": get_depth(asks, 10),
        }

    def analyze_fills(self, fills: List[Dict]) -> Optional[Dict]:
        if not fills:
            return None

        buys = [(f["price"], f["size"], f["fee"]) for f in fills if f["side"] == "buy"]
        sells = [
            (f["price"], f["size"], f["fee"]) for f in fills if f["side"] == "sell"
        ]

        total_buy_vol = sum(s for _, s, _ in buys)
        total_sell_vol = sum(s for _, s, _ in sells)
        total_fees = sum(f["fee"] for f in fills)

        round_trips = min(len(buys), len(sells))
        gross_pnl = 0.0

        # Simple FIFOish PnL for estimation
        for i in range(round_trips):
            buy_price = buys[i][0]
            sell_price = sells[i][0]
            size = min(buys[i][1], sells[i][1])
            gross_pnl += (sell_price - buy_price) * size

        net_pnl = gross_pnl - total_fees

        first_time = fills[0]["time"]
        last_time = fills[-1]["time"]

        # Determine duration
        if isinstance(first_time, (int, float)):
            # timestamp
            duration_hours = max((last_time - first_time) / 3600, 0.001)
        else:
            # datetime object
            duration_hours = max((last_time - first_time).total_seconds() / 3600, 0.001)

        fills_per_hour = len(fills) / duration_hours

        return {
            "total_fills": len(fills),
            "buys": len(buys),
            "sells": len(sells),
            "round_trips": round_trips,
            "gross_pnl": gross_pnl,
            "total_fees": total_fees,
            "net_pnl": net_pnl,
            "fills_per_hour": fills_per_hour,
            "duration_hours": duration_hours,
            "avg_buy_price": sum(p for p, _, _ in buys) / len(buys) if buys else 0,
            "avg_sell_price": sum(p for p, _, _ in sells) / len(sells) if sells else 0,
        }

    def recommend_params(
        self,
        price,
        tick_size,
        daily_vol,
        hourly_vol_usd,
        capital,
        min_order_size,
        target_return=0.02,
    ) -> Dict:
        fee_cost_per_unit = price * (self.maker_fee * 2)
        min_profitable_spread = fee_cost_per_unit * 1.5
        min_profitable_ticks = math.ceil(min_profitable_spread / tick_size)

        vol_spread = price * daily_vol * 0.15
        vol_spread_ticks = math.ceil(vol_spread / tick_size)

        recommended_spread_ticks = max(min_profitable_ticks, vol_spread_ticks)
        spread_dollars = recommended_spread_ticks * tick_size

        profit_per_rt = (spread_dollars - fee_cost_per_unit) * min_order_size

        daily_target = capital * target_return
        rts_needed = (
            math.ceil(daily_target / profit_per_rt)
            if profit_per_rt > 0
            else float("inf")
        )

        # Position Sizing
        max_position_value = capital * 0.15
        max_position_size = max_position_value / price
        # Snap to min order size
        max_position_size = max(
            math.floor(max_position_size / min_order_size) * min_order_size,
            min_order_size,
        )
        position_limit = max_position_size / min_order_size
        gamma_rec = min(0.5, max(0.1, daily_vol * 5))

        return {
            "min_spread_ticks": recommended_spread_ticks,
            "spread_dollars": spread_dollars,
            "spread_pct": (spread_dollars / price) * 100,
            "fee_cost_per_rt": fee_cost_per_unit * min_order_size,
            "profit_per_rt": profit_per_rt,
            "rts_for_2pct": rts_needed,
            "rts_per_hour_needed": round(rts_needed / 24, 1),
            "order_size": min_order_size,
            "position_limit": position_limit,
            "max_position_value": max_position_value,
            "gamma": round(gamma_rec, 2),
            "quote_levels": 3 if hourly_vol_usd > 500000 else 2,
            "level_spacing_factor": 0.5,
        }

    def generate_profitability_matrix(
        self, price, tick_size, daily_vol, min_qty, capital
    ):
        # Ranges
        fee_cost_per_unit = price * (self.maker_fee * 2)

        # Spread Range: from 1 tick to 3x recommended vol spread
        vol_spread = price * daily_vol * 0.2
        max_spread_ticks = max(20, math.ceil(vol_spread / tick_size) * 3)
        spread_ticks = np.linspace(1, max_spread_ticks, 20)

        # Size Range: from min_qty to 20% of capital
        max_size = (capital * 0.2) / price
        if max_size <= min_qty:
            sizes = [min_qty]
        else:
            sizes = np.linspace(min_qty, max_size, 20)

        matrix = []
        for s_tick in spread_ticks:
            row = []
            s_tick = int(s_tick)
            spread_val = s_tick * tick_size
            spread_pct = spread_val / price

            # Probability Model (Heuristic): Prob = exp(-k * spread_pct / daily_vol)
            # Assuming daily_vol is ~1 StdDev of daily moves.
            # If spread is 1 daily vol, prob is low.
            # We want prob per hour?
            # Let's use Normalized Spread = Spread / (Price * HourlyVol)
            # High volatile -> smaller normalized spread -> higher prob.
            hourly_vol = daily_vol / math.sqrt(24)
            normalized_spread = spread_pct / hourly_vol
            prob_fill = math.exp(-0.5 * normalized_spread)  # Decay factor

            for size in sizes:
                # Profit per RT
                profit = (spread_val - fee_cost_per_unit) * size

                # Expected Hourly Profit = Profit * (MaxTradesPerHour * Prob)
                # Assume MaxTrades could be high, say 60/hr for tightest spread?
                # This is just for relative heatmap coloring
                expected_profit_hr = profit * (60 * prob_fill)

                if profit < 0:
                    expected_profit_hr = profit * 10  # Penalize negative

                row.append(
                    {
                        "spread_ticks": s_tick,
                        "spread_pct": spread_pct * 100,
                        "size": size,
                        "profit_per_rt": profit,
                        "expected_profit_hr": expected_profit_hr,
                        "prob_fill": prob_fill,
                    }
                )
            matrix.append(row)

        return matrix, spread_ticks, sizes
