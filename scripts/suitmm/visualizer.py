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

    def plot_cumulative_pnl(self, df, symbol):
        if df.empty or "cum_pnl" not in df.columns:
            return go.Figure()

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=df["time"],
                y=df["cum_pnl"],
                mode="lines",
                name="Cumulative PnL",
                line=dict(color="blue", width=2),
                fill="tozeroy",
            )
        )
        fig.update_layout(
            title=f"Cumulative Realized PnL - {symbol}",
            xaxis_title="Time",
            yaxis_title="PnL (USD)",
            template="plotly_dark",
        )
        return fig

    def plot_trade_executions(self, klines, fills_df, symbol):
        # Klines to OHLC dataframe
        if not klines:
            return go.Figure()

        # Parse klines
        # [startTime, open, high, low, close, volume, turnover]
        k_data = []
        for k in klines:
            k_data.append(
                {
                    "time": pd.to_datetime(int(k[0]), unit="ms", utc=True),
                    "open": float(k[1]),
                    "high": float(k[2]),
                    "low": float(k[3]),
                    "close": float(k[4]),
                }
            )
        ohlc = pd.DataFrame(k_data).sort_values("time")

        fig = go.Figure(
            data=[
                go.Candlestick(
                    x=ohlc["time"],
                    open=ohlc["open"],
                    high=ohlc["high"],
                    low=ohlc["low"],
                    close=ohlc["close"],
                    name="Price",
                )
            ]
        )

        if not fills_df.empty:
            buys = fills_df[fills_df["side"] == "Buy"]
            sells = fills_df[fills_df["side"] == "Sell"]

            if not buys.empty:
                fig.add_trace(
                    go.Scatter(
                        x=buys["time"],
                        y=buys["price"],
                        mode="markers",
                        name="Buy Fill",
                        marker=dict(
                            symbol="triangle-up", size=10, color="green", line_width=1
                        ),
                    )
                )

            if not sells.empty:
                fig.add_trace(
                    go.Scatter(
                        x=sells["time"],
                        y=sells["price"],
                        mode="markers",
                        name="Sell Fill",
                        marker=dict(
                            symbol="triangle-down", size=10, color="red", line_width=1
                        ),
                    )
                )

        fig.update_layout(
            title=f"Trade Executions - {symbol}",
            xaxis_title="Time",
            yaxis_title="Price",
            template="plotly_dark",
            xaxis_rangeslider_visible=False,
        )
        return fig

    def plot_markout_distribution(self, markouts, symbol):
        if markouts.empty:
            return go.Figure()

        fig = go.Figure()

        for col in ["markout_1m", "markout_5m", "markout_60m"]:
            if col in markouts.columns:
                fig.add_trace(go.Box(y=markouts[col], name=col, boxmean=True))

        fig.update_layout(
            title=f"Post-Trade Markout Distribution (PnL per Unit) - {symbol}",
            yaxis_title="Price Change in Favor (USD)",
            template="plotly_dark",
        )
        return fig

    def plot_indicators(self, indicators, symbol):
        """
        Plots key indicators: Liquidity, Trend, and Toxicity.
        Expects indicators dict with 'regimes' and 'hft' keys containing DataFrames.
        """
        from plotly.subplots import make_subplots

        regimes = indicators.get("regimes")
        hft = indicators.get("hft")

        if (regimes is None or regimes.empty) and (hft is None or hft.empty):
            return go.Figure()

        # Create subplots
        fig = make_subplots(
            rows=3,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=[
                "Trend Strength (Regimes)",
                "Liquidity Score",
                "HFT Toxicity",
            ],
        )

        # 1. Trend
        if regimes is not None and not regimes.empty:
            for col, color in [
                ("trend_fast", "cyan"),
                ("trend_mid", "yellow"),
                ("trend_slow", "magenta"),
            ]:
                if col in regimes.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=regimes["time"],
                            y=regimes[col],
                            mode="lines",
                            name=col,
                            line=dict(color=color, width=1),
                        ),
                        row=1,
                        col=1,
                    )

            # 2. Liquidity
            if "liquidity_score" in regimes.columns:
                fig.add_trace(
                    go.Scatter(
                        x=regimes["time"],
                        y=regimes["liquidity_score"],
                        mode="lines",
                        name="Liquidity Score",
                        line=dict(color="lime", width=1),
                        fill="tozeroy",
                    ),
                    row=2,
                    col=1,
                )

        # 3. Toxicity
        if hft is not None and not hft.empty:
            if "toxicity_score" in hft.columns:
                fig.add_trace(
                    go.Scatter(
                        x=hft["time"],
                        y=hft["toxicity_score"],
                        mode="lines",
                        name="Toxicity",
                        line=dict(color="red", width=1),
                    ),
                    row=3,
                    col=1,
                )

        fig.update_layout(
            title=f"Market Regime & Risk Indicators - {symbol}",
            template="plotly_dark",
            height=800,
        )
        return fig
