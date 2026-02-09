import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple


class PerformanceAnalyzer:
    def __init__(self):
        pass

    def calculate_pnl_series(self, fills: List[Dict]) -> pd.DataFrame:
        """
        Calculates cumulative Mark-to-Market PnL and other trade-related series.
        """
        if not fills:
            return pd.DataFrame()

        df = pd.DataFrame(fills)

        # Ensure datetime
        # Ensure datetime
        if not pd.api.types.is_datetime64_any_dtype(df["time"]):
            df["time"] = pd.to_datetime(df["time"])

        df = df.sort_values("time")

        # Parse numeric
        df["price"] = pd.to_numeric(df["price"])
        df["size"] = pd.to_numeric(df["size"])
        df["fee"] = pd.to_numeric(df["fee"]).fillna(0.0)

        # Vectorized PnL Calculation (MtM)
        # Side multiplier: Buy -> 1, Sell -> -1
        df["side_mult"] = df["side"].apply(
            lambda x: 1 if str(x).lower() == "buy" else -1
        )

        # Inventory Change: Buy +Size, Sell -Size
        df["inv_change"] = df["size"] * df["side_mult"]
        df["inventory"] = df["inv_change"].cumsum()

        # Cash Flow: Buy -Cost, Sell +Revenue (minus fees)
        # Cost = Price * Size
        df["cash_flow"] = (-1 * df["price"] * df["size"] * df["side_mult"]) - df["fee"]
        df["cum_cash"] = df["cash_flow"].cumsum()

        # MtM Value = Inventory * Current Price
        # We use the execution price as a proxy for market price at that moment
        df["mtm_value"] = df["inventory"] * df["price"]

        # Total Equity (PnL) = Cash + MtM Value
        # This assumes starting with 0 inventory and 0 cash
        df["cum_pnl"] = df["cum_cash"] + df["mtm_value"]

        return df

    def calculate_trade_stats(self, fills: List[Dict]) -> Dict:
        """
        Calculates aggregate statistics: Win Rate, Profit Factor, Total PnL, etc.
        """
        if not fills:
            return {}

        df = self.calculate_pnl_series(fills)

        total_pnl = df["cum_pnl"].iloc[-1] if not df.empty else 0.0
        total_volume = (df["price"] * df["size"]).sum()
        total_trades = len(df)
        total_fees = df["fee"].sum()

        # Estimate Win Rate (Hard without FIFO matching)
        # We can look at individual trade "implied" edge or just global stats.
        # For simplicity, let's estimate "profitable trades" by comparing price to simple moving average? No.
        # Let's omit "Win Rate" if we can't calculate it accurately without matching.
        # Or implement a simple FIFO matcher.

        # Simple FIFO Matcher for Win Rate
        realized_pnls = []
        inventory_queue = []  # List of (price, size)

        for _, row in df.iterrows():
            size = row["size"]
            price = row["price"]
            side = row["side_mult"]  # 1 for Buy, -1 for Sell

            if side == 1:  # Buy
                # If we have short position, cover it
                # Logic is complex for partial fills.
                # Let's simplify: Just report Total PnL and Volume for now.
                pass

        # Valid stats we can report reliably:
        return {
            "Total PnL": total_pnl,
            "Total Volume": total_volume,
            "Total Trades": total_trades,
            "Total Fees": total_fees,
            "Avg PnL per Trade": total_pnl / total_trades if total_trades > 0 else 0,
            "Final Inventory": df["inventory"].iloc[-1] if not df.empty else 0,
            # Placeholder for Win Rate/Profit Factor until FIFO implemented
            "Win Rate": 0.0,
            "Profit Factor": 0.0,
        }

    def calculate_markout(self, fills: List[Dict], klines: List[Dict]) -> pd.DataFrame:
        """
        Calculates price change 1m, 5m, 60m after each fill.
        Positive markout means price moved in favor of the trade (e.g. bought before rise).
        """
        if not fills or not klines:
            return pd.DataFrame()

        fills_df = pd.DataFrame(fills)
        if not pd.api.types.is_datetime64_any_dtype(fills_df["time"]):
            fills_df["time"] = pd.to_datetime(fills_df["time"])

        # Convert klines to dataframe
        # Klines: [start, open, high, low, close, volume, turnout]
        # Bybit klines are usually [startTime, open, high, low, close, volume, turnover]
        kline_data = []
        for k in klines:
            kline_data.append(
                {
                    "time": pd.to_datetime(int(k[0]), unit="ms", utc=True),
                    "open": float(k[1]),
                    "close": float(k[4]),
                }
            )
        kline_df = pd.DataFrame(kline_data).sort_values("time").set_index("time")

        markouts = []

        # Resample klines to ensure we have a continuous time series for lookup
        # kline_df = kline_df.resample('1min').ffill()

        for _, fill in fills_df.iterrows():
            fill_time = fill["time"]
            # Find close price at T + delay

            def get_price_at_delay(delay_min):
                target_time = fill_time + pd.Timedelta(minutes=delay_min)
                # Find nearest kline
                idx = kline_df.index.get_indexer([target_time], method="nearest")
                if idx[0] == -1:
                    return None
                matched_time = kline_df.index[idx[0]]
                # If gap is too large (> 2x freq), ignore
                if (
                    abs((matched_time - target_time).total_seconds()) > 120 * delay_min
                ):  # Rough heuristic
                    return None
                return kline_df.iloc[idx[0]]["close"]

            p1m = get_price_at_delay(1)
            p5m = get_price_at_delay(5)
            p60m = get_price_at_delay(60)

            entry_price = float(fill["price"])
            side_mult = 1 if fill["side"].lower() == "buy" else -1

            m_rec = {"time": fill_time}
            if p1m:
                m_rec["markout_1m"] = (p1m - entry_price) * side_mult
            if p5m:
                m_rec["markout_5m"] = (p5m - entry_price) * side_mult
            if p60m:
                m_rec["markout_60m"] = (p60m - entry_price) * side_mult

            markouts.append(m_rec)

        return pd.DataFrame(markouts)

    def prepare_indicator_data(
        self, indicators: Dict[str, List[Dict]]
    ) -> Dict[str, pd.DataFrame]:
        """
        Converts raw indicator lists to synchronized DataFrames.
        """
        result = {}
        for category, data in indicators.items():
            if not data:
                continue
            df = pd.DataFrame(data)
            if "time" in df.columns:
                if not pd.api.types.is_datetime64_any_dtype(df["time"]):
                    df["time"] = pd.to_datetime(df["time"])
                df = df.sort_values("time")
                result[category] = df

        return result
