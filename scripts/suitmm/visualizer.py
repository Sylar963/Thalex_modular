import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np


class MMVisualizer:
    def plot_profitability_heatmap(self, matrix_data, symbol):
        """
        Expects matrix_data as list of list of dicts.
        Rows = Spread Ticks
        Cols = Order Sizes
        """
        # Flatten for DataFrame
        data = []
        for row in matrix_data:
            for item in row:
                data.append(item)

        df = pd.DataFrame(data)

        # We want a pivot table for the heatmap
        # Pivot: index=order_size, columns=spread_ticks, values=expected_profit_hr (or profit_per_rt)
        # But wait, heatmap x=spread, y=size

        pivot_val = df.pivot(
            index="size", columns="spread_ticks", values="profit_per_rt"
        )

        # Create Heatmap
        fig = go.Figure(
            data=go.Heatmap(
                z=pivot_val.values,
                x=pivot_val.columns,
                y=pivot_val.index,
                colorscale="RdBu",
                zmid=0,
                hovertemplate="Spread: %{x} ticks<br>Size: %{y}<br>Profit/RT: $%{z:.4f}<extra></extra>",
            )
        )

        fig.update_layout(
            title=f"Profit Per Round Trip (Net Fees) - {symbol}",
            xaxis_title="Spread (Ticks)",
            yaxis_title="Order Size (Coins)",
            template="plotly_dark",
        )
        return fig

    def plot_volatility_cone(self, klines, symbol):
        # Calculate trailing volatility for different windows
        closes = pd.Series([float(k[4]) for k in klines if k[4]])
        returns = np.log(closes / closes.shift(1))

        windows = [24, 7 * 24, 14 * 24, 30 * 24]  # 1d, 7d, 14d, 30d
        vols = []
        labels = []

        for w in windows:
            if len(returns) > w:
                vol = returns.rolling(window=w).std() * np.sqrt(24 * 365)  # Annualized
                vols.append(vol.iloc[-1])
                labels.append(f"{w // 24} Days")

        if not vols:
            return go.Figure()

        fig = go.Figure(
            data=[
                go.Bar(
                    x=labels,
                    y=[v * 100 for v in vols],
                    text=[f"{v * 100:.1f}%" for v in vols],
                    textposition="auto",
                )
            ]
        )

        fig.update_layout(
            title=f"Annualized Rolling Volatility Cone - {symbol}",
            xaxis_title="Lookback Window",
            yaxis_title="Volatility (%)",
            template="plotly_dark",
        )
        return fig

    def plot_depth_profile(self, ob, symbol):
        bids = pd.DataFrame(ob.get("b", []), columns=["price", "size"], dtype=float)
        asks = pd.DataFrame(ob.get("a", []), columns=["price", "size"], dtype=float)

        if bids.empty or asks.empty:
            return go.Figure()

        bids["value"] = bids["price"] * bids["size"]
        asks["value"] = asks["price"] * asks["size"]

        bids["cum_value"] = bids["value"].cumsum()
        asks["cum_value"] = asks["value"].cumsum()

        fig = go.Figure()

        # Bids (Green area)
        fig.add_trace(
            go.Scatter(
                x=bids["price"],
                y=bids["cum_value"],
                fill="tozeroy",
                name="Bids",
                line=dict(color="green"),
            )
        )

        # Asks (Red area) - Sort by price ascending
        asks = asks.sort_values("price", ascending=True)
        # Cumulative should be from best ask up
        asks["cum_value"] = asks["value"].cumsum()

        fig.add_trace(
            go.Scatter(
                x=asks["price"],
                y=asks["cum_value"],
                fill="tozeroy",
                name="Asks",
                line=dict(color="red"),
            )
        )

        current_price = (bids.iloc[0]["price"] + asks.iloc[0]["price"]) / 2

        fig.update_layout(
            title=f"Market Depth (Cumulative USD) - {symbol}",
            xaxis_title="Price",
            yaxis_title="Cumulative Value ($)",
            template="plotly_dark",
            xaxis_range=[current_price * 0.95, current_price * 1.05],  # Zoom to +/- 5%
        )
        return fig
