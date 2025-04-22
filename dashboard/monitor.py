# dashboard/monitor.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import time
import os
import datetime as dt
import numpy as np
import sys
from datetime import datetime, timedelta

# Add parent directory to path to import Thalex client
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

# Try to import Thalex client
try:
    import thalex as th
    THALEX_AVAILABLE = True
except ImportError:
    THALEX_AVAILABLE = False
    st.warning("Thalex client not available. Install with `pip install thalex`")

st.set_page_config(layout="wide", page_title="Thalex Market Maker Dashboard")

# Store last modification time to detect changes
last_mod_time = 0

# Simulate volume candles from time-series data
class VolumeCandle:
    """Simple container for volume-based candle data"""
    def __init__(self):
        self.open_price = 0.0
        self.high_price = 0.0
        self.low_price = float('inf')
        self.close_price = 0.0
        self.volume = 0.0
        self.buy_volume = 0.0
        self.sell_volume = 0.0
        self.volume_delta = 0.0  # buy_volume - sell_volume
        self.delta_ratio = 0.0   # normalized delta (-1 to 1)
        self.trade_count = 0
        self.start_time = 0
        self.end_time = 0
        self.is_complete = False
        self.start_timestamp = None
        self.end_timestamp = None

    def update(self, price: float, volume: float, is_buy: bool, timestamp: int) -> None:
        """Update candle with new trade data"""
        # Update prices
        if self.trade_count == 0:
            self.open_price = price
            self.high_price = price
            self.low_price = price
            self.start_time = timestamp
            self.start_timestamp = datetime.fromtimestamp(timestamp/1000)
        else:
            self.high_price = max(self.high_price, price)
            self.low_price = min(self.low_price, price)
        
        self.close_price = price
        self.end_time = timestamp
        self.end_timestamp = datetime.fromtimestamp(timestamp/1000)
        
        # Update volumes
        self.volume += volume
        if is_buy:
            self.buy_volume += volume
        else:
            self.sell_volume += volume
        
        # Update volume delta
        self.volume_delta = self.buy_volume - self.sell_volume
        if self.volume > 0:
            self.delta_ratio = self.volume_delta / self.volume
        
        self.trade_count += 1

def fetch_exchange_trades(instrument: str = "BTC-PERPETUAL", hours: int = 24, network: str = "testnet"):
    """
    Fetch trades directly from Thalex exchange
    
    Args:
        instrument: Instrument to fetch trades for (default: BTC-PERPETUAL)
        hours: How many hours of trade history to fetch
        network: Which network to connect to (testnet or mainnet)
        
    Returns:
        List of trades or None if error
    """
    if not THALEX_AVAILABLE:
        st.error("Thalex client not available. Cannot fetch trades directly.")
        return None
        
    try:
        # Initialize Thalex client
        client = th.Thalex(network=network)
        
        # Calculate start time (UTC)
        end_time = int(time.time() * 1000)
        start_time = end_time - (hours * 60 * 60 * 1000)
        
        # Fetch trade history
        st.info(f"Fetching trade history for {instrument} from Thalex {network}...")
        trades = client.get_trade_history(
            instrument=instrument,
            start_time=start_time,
            end_time=end_time,
            limit=1000  # Maximum allowed by API
        )
        
        if trades and len(trades) > 0:
            st.success(f"Successfully fetched {len(trades)} trades from Thalex")
            return trades
        else:
            st.warning(f"No trades found for {instrument} in the last {hours} hours")
            return None
            
    except Exception as e:
        st.error(f"Error fetching trades from Thalex: {str(e)}")
        return None

def create_volume_candles_from_trades(trades, volume_threshold=1.0, max_time_seconds=300):
    """
    Create volume candles from real trade data
    
    Args:
        trades: List of trade objects from Thalex API
        volume_threshold: Volume required to complete a candle
        max_time_seconds: Maximum time before forcing candle completion
        
    Returns:
        List of VolumeCandle objects
    """
    if not trades:
        return []
        
    candles = []
    current_candle = VolumeCandle()
    
    # Sort trades by timestamp (newest first from API, we want oldest first)
    sorted_trades = sorted(trades, key=lambda x: x.get('timestamp', 0))
    
    for trade in sorted_trades:
        # Extract trade data
        price = float(trade.get('price', 0))
        volume = float(trade.get('amount', 0))
        is_buy = trade.get('side', '').lower() == 'buy'
        timestamp = trade.get('timestamp', 0)
        
        # Skip invalid trades
        if price <= 0 or volume <= 0 or timestamp <= 0:
            continue
            
        # Update current candle with this trade
        current_candle.update(price, volume, is_buy, timestamp)
        
        # Check if candle should complete
        if (current_candle.volume >= volume_threshold or 
            (timestamp - current_candle.start_time) >= max_time_seconds * 1000):
            
            current_candle.is_complete = True
            candles.append(current_candle)
            current_candle = VolumeCandle()
    
    # Add the last candle if it has data
    if current_candle.trade_count > 0:
        current_candle.is_complete = True
        candles.append(current_candle)
        
    return candles

def generate_volume_candles(df, volume_threshold=0.2, max_time_seconds=180, market_threshold=0.05):
    """
    Generate volume candles from time-series metric data
    
    Args:
        df: DataFrame with 'datetime', 'mid_price', etc.
        volume_threshold: Volume needed to complete a candle
        max_time_seconds: Maximum time before forcing candle completion
        market_threshold: Threshold percentage to classify as market order
        
    Returns:
        List of VolumeCandle objects
    """
    if df.empty or 'mid_price' not in df.columns:
        return []
        
    candles = []
    current_candle = VolumeCandle()
    
    # Sort by datetime to ensure chronological processing
    df = df.sort_values('datetime').reset_index(drop=True)
    
    # Process each row sequentially
    prev_price = None
    market_order_count = 0
    total_orders = 0
    
    for i in range(len(df)):
        row = df.iloc[i]
        price = row['mid_price']
        total_orders += 1
        
        # Estimate trade volume and detect market orders
        if prev_price is not None and prev_price > 0:
            price_change_pct = abs(price / prev_price - 1)
            
            # Classify as market order if price change exceeds threshold
            is_market_order = price_change_pct > (market_threshold / 100)
            
            if is_market_order:
                # Market orders have higher volume impact
                estimated_volume = max(0.05, min(0.5, price_change_pct * 200))
                market_order_count += 1
                # We could add this info to the candle if we enhance the VolumeCandle class
            else:
                # Limit orders have less volume impact
                estimated_volume = max(0.01, min(0.1, price_change_pct * 100))
        else:
            estimated_volume = 0.01
            is_market_order = False
            
        # Estimate if it's a buy or sell based on price movement
        is_buy = prev_price is None or price >= prev_price
        
        # Update for next iteration
        prev_price = price
            
        # Convert datetime to timestamp for volume candle
        timestamp = int(row['datetime'].timestamp() * 1000)
        
        # Update current candle
        current_candle.update(price, estimated_volume, is_buy, timestamp)
        
        # Check if candle should complete
        if (current_candle.volume >= volume_threshold or 
            (timestamp - current_candle.start_time) >= max_time_seconds * 1000):
            
            current_candle.is_complete = True
            candles.append(current_candle)
            current_candle = VolumeCandle()
    
    # Add the last candle if it has data
    if current_candle.trade_count > 0:
        current_candle.is_complete = True
        candles.append(current_candle)
    
    # Print market order stats
    if total_orders > 0:
        market_order_percentage = (market_order_count / total_orders) * 100
        print(f"Detected {market_order_count} potential market orders out of {total_orders} total updates ({market_order_percentage:.1f}%)")
        
    return candles

def plot_volume_candles(candles, title="Volume-Based Candles"):
    """Create a plot of volume-based candles"""
    if not candles:
        return go.Figure()
        
    # Extract timestamps for x-axis
    timestamps = [candle.start_timestamp for candle in candles]
    
    # Create figure
    fig = go.Figure()
    
    # Add candlestick trace
    fig.add_trace(go.Candlestick(
        x=timestamps,
        open=[candle.open_price for candle in candles],
        high=[candle.high_price for candle in candles],
        low=[candle.low_price for candle in candles],
        close=[candle.close_price for candle in candles],
        name="Price",
        increasing_line_color='green', 
        decreasing_line_color='red'
    ))
    
    # Add volume as bar chart on secondary y-axis
    fig.add_trace(go.Bar(
        x=timestamps,
        y=[candle.volume for candle in candles],
        name="Volume",
        marker_color=['rgba(0,150,0,0.7)' if candle.volume_delta >= 0 else 'rgba(150,0,0,0.7)' 
                     for candle in candles],
        yaxis="y2"
    ))
    
    # Add delta ratio line
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=[candle.delta_ratio for candle in candles],
        name="Buy/Sell Ratio",
        line=dict(width=2, color='blue'),
        yaxis="y3"
    ))
    
    # Set up layout with multiple y-axes
    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis=dict(
            title="Price",
            domain=[0.3, 1.0]
        ),
        yaxis2=dict(
            title="Volume",
            domain=[0, 0.25],
            anchor="x"
        ),
        yaxis3=dict(
            title="Buy/Sell Ratio",
            domain=[0, 0.25],
            anchor="x",
            overlaying="y2",
            side="right",
            range=[-1.1, 1.1]
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        height=700
    )
    
    # Add zero line for delta ratio
    fig.add_hline(y=0, line_width=1, line_dash="dash", line_color="gray", row=2, col=1)
    
    return fig

def load_data():
    # Use the absolute path to your metrics directory
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    metrics_path = os.path.join(base_dir, 'metrics', 'metrics.csv')
    
    global last_mod_time
    current_mod_time = 0
    
    if os.path.exists(metrics_path):
        current_mod_time = os.path.getmtime(metrics_path)
        if current_mod_time == last_mod_time:
            st.sidebar.warning("⚠️ Metrics file hasn't been updated since last check")
        last_mod_time = current_mod_time
        
        st.sidebar.info(f"Last file update: {datetime.fromtimestamp(current_mod_time).strftime('%Y-%m-%d %H:%M:%S')}")
        
        metrics_df = pd.read_csv(metrics_path)
        if 'datetime' in metrics_df.columns:
            metrics_df['datetime'] = pd.to_datetime(metrics_df['datetime'])
        elif 'timestamp' in metrics_df.columns:
            metrics_df['datetime'] = pd.to_datetime(metrics_df['timestamp'], unit='s')
        return metrics_df
    return pd.DataFrame()

def create_sidebar():
    st.sidebar.title("Dashboard Controls")
    st.sidebar.info("This dashboard displays trading metrics from your Thalex bot.")
    
    # Add refresh button in sidebar
    if st.sidebar.button("🔄 Refresh Data"):
        st.sidebar.success("Refreshing data...")
        st.rerun()
    
    # Add metrics file info
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    metrics_path = os.path.join(base_dir, 'metrics', 'metrics.csv')
    
    if os.path.exists(metrics_path):
        file_size = os.path.getsize(metrics_path) / 1024  # KB
        last_modified = datetime.fromtimestamp(os.path.getmtime(metrics_path))
        time_since_update = datetime.now() - last_modified
        
        st.sidebar.info(f"Metrics file size: {file_size:.2f} KB")
        st.sidebar.info(f"Last updated: {last_modified.strftime('%Y-%m-%d %H:%M:%S')} ({time_since_update.seconds // 60} minutes ago)")
        
        if time_since_update.seconds > 3600:  # More than an hour
            st.sidebar.warning("⚠️ Data may be stale (>1 hour old)")
    else:
        st.sidebar.error("Metrics file not found!")
    
    # Add time range selector
    st.sidebar.subheader("Time Range")
    time_range = st.sidebar.selectbox(
        "Select time period",
        ["Last 24 hours", "Last 12 hours", "Last 6 hours", "Last hour", "All data"]
    )
    
    # Add volume candle settings
    st.sidebar.subheader("Volume Candle Settings")
    volume_threshold = st.sidebar.slider(
        "Volume Threshold (BTC)",
        min_value=0.1,
        max_value=2.0,
        value=0.2,  # Changed default to 0.2 for better visualization
        step=0.1,
        help="Volume required to complete a candle (similar to your bot configuration)"
    )
    
    max_time = st.sidebar.slider(
        "Max Candle Time (sec)",
        min_value=60,
        max_value=900,
        value=180,  # Changed default to 180 seconds
        step=60,
        help="Maximum time before forcing candle completion"
    )
    
    # Add market order detection settings
    st.sidebar.subheader("Market Order Detection")
    market_threshold = st.sidebar.slider(
        "Market Order Threshold (%)",
        min_value=0.01,
        max_value=0.5,
        value=0.05,
        step=0.01,
        format="%.2f%%",
        help="Price change percentage to classify as market order"
    )
    
    # Add exchange settings if Thalex is available
    use_exchange_data = False
    instrument = "BTC-PERPETUAL"
    network = "testnet"
    fetch_hours = 24
    
    if THALEX_AVAILABLE:
        st.sidebar.subheader("Exchange Data")
        use_exchange_data = st.sidebar.checkbox(
            "Pull data from Thalex exchange",
            value=False,
            help="Fetch real trading data from Thalex exchange instead of using metrics"
        )
        
        if use_exchange_data:
            instrument = st.sidebar.text_input("Instrument", value="BTC-PERPETUAL")
            network = st.sidebar.selectbox("Network", ["testnet", "mainnet"], index=0)
            fetch_hours = st.sidebar.slider("Fetch hours", min_value=1, max_value=48, value=24)
    
    return time_range, volume_threshold, max_time, market_threshold, use_exchange_data, instrument, network, fetch_hours

def filter_by_time(df, time_range):
    if df.empty or 'datetime' not in df.columns:
        return df
        
    now = datetime.now()
    
    if time_range == "Last 24 hours":
        cutoff = now - timedelta(hours=24)
    elif time_range == "Last 12 hours":
        cutoff = now - timedelta(hours=12)
    elif time_range == "Last 6 hours":
        cutoff = now - timedelta(hours=6)
    elif time_range == "Last hour":
        cutoff = now - timedelta(hours=1)
    else:  # All data
        return df
        
    return df[df['datetime'] > cutoff]

def create_dashboard():
    st.title("Thalex Market Maker Dashboard")
    
    # Setup sidebar and get parameters
    time_range, volume_threshold, max_time, market_threshold, use_exchange_data, instrument, network, fetch_hours = create_sidebar()
    
    # Create tab structure first (so we can use it for progress reporting)
    tab1, tab2 = st.tabs(["Metrics Dashboard", "Volume Candles"])
    
    # Load metrics data unless we're only using exchange data
    if not use_exchange_data:
        metrics_df = load_data()
        
        if metrics_df.empty:
            with tab1:
                st.warning("No metrics data found. Make sure your trading bot is running and writing to metrics/metrics.csv")
                
                # Display help information
                st.error("⚠️ Performance monitoring not active in your bot")
                st.markdown("""
                ## Troubleshooting
                
                The performance monitor in your trading bot isn't actively writing to the metrics file.
                
                Your options:
                1. Try using exchange data directly (enable in sidebar)
                2. Run the bot with the proper flags to enable metrics collection
                """)
            return
        
        # Filter by selected time range
        filtered_df = filter_by_time(metrics_df, time_range)
        
        if filtered_df.empty:
            with tab1:
                st.warning(f"No data available for the selected time range: {time_range}")
                # Fall back to all data
                filtered_df = metrics_df
                st.info("Showing all available data instead")
        
        with tab1:
            # Display summary statistics
            st.subheader("Summary Statistics")
            latest = filtered_df.iloc[-1] if not filtered_df.empty else None
            
            if latest is not None:
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Current PnL", f"${latest['pnl']:.2f}")
                
                with col2:
                    st.metric("Position", f"{latest['position']:.4f}")
                    
                with col3:
                    st.metric("Current Price", f"${latest['mid_price']:.2f}")
                    
                with col4:
                    st.metric("Current Spread", f"${latest['spread']:.2f}")
            
            # Create charts
            st.subheader("Performance Charts")
            col1, col2 = st.columns(2)
            
            # PnL Chart
            with col1:
                st.subheader("PnL Over Time")
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=filtered_df['datetime'], y=filtered_df['pnl'], mode='lines', name='PnL'))
                st.plotly_chart(fig, use_container_width=True)
            
            # Position Chart
            with col2:
                st.subheader("Position Size")
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=filtered_df['datetime'], y=filtered_df['position'], mode='lines', name='Position'))
                st.plotly_chart(fig, use_container_width=True)
            
            # Mid Price and Spread
            col3, col4 = st.columns(2)
            
            with col3:
                st.subheader("Mid Price")
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=filtered_df['datetime'], y=filtered_df['mid_price'], mode='lines', name='Mid Price'))
                st.plotly_chart(fig, use_container_width=True)
            
            with col4:
                st.subheader("Spread")
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=filtered_df['datetime'], y=filtered_df['spread'], mode='lines', name='Spread'))
                st.plotly_chart(fig, use_container_width=True)
            
            # Display last few records as a table
            st.subheader("Recent Metrics")
            st.dataframe(filtered_df.tail(10).sort_values('datetime', ascending=False))
    
    with tab2:
        st.subheader("Volume-Based Candles")
        
        # Add information about forecasting
        with st.expander("ℹ️ How can this improve bot forecasting?"):
            st.markdown("""
            ### Improving Bot Forecasting
            
            The volume candles displayed here are identical to how your trading bot processes market data for its predictive analysis.
            
            **How to improve your bot's forecasting:**
            
            1. **Parameter Tuning**: Use this dashboard to find optimal volume thresholds and time settings that capture market patterns
            
            2. **Backtesting**: Analyze historical candles to see how signals would have predicted price movements
            
            3. **Direct Integration**: 
               - The `create_volume_candles_from_trades()` function here could be integrated directly into your bot's code
               - This would allow your bot to use exchange data instead of relying on its own trade observations
               - This is especially useful for low-volume markets where your bot might not see all trades
            
            4. **Signal Analysis**: Compare the generated candles with actual price movements to refine the signal calculations in your bot
            
            You can modify `volume_candle_buffer.py` to fetch and process exchange data directly, which would give your bot a more complete view of the market and potentially improve its forecasting accuracy.
            """)
        
        if use_exchange_data:
            st.info(f"""
            Fetching real trade data from Thalex {network} for {instrument}.
            Creating volume candles with: threshold={volume_threshold} BTC, max time={max_time}s
            """)
            
            # Fetch trades from exchange
            trades = fetch_exchange_trades(instrument, fetch_hours, network)
            
            if trades:
                # Create volume candles from real trade data
                candles = create_volume_candles_from_trades(trades, volume_threshold, max_time)
                
                if not candles:
                    st.warning("Could not create any volume candles from trade data.")
                else:
                    st.success(f"Created {len(candles)} volume candles from {len(trades)} trades")
                    
                    # Plot candles
                    fig = plot_volume_candles(candles, 
                        title=f"Volume Candles from Exchange Data ({instrument}, Threshold: {volume_threshold} BTC, Max Time: {max_time}s)")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display candle data as table
                    st.subheader("Volume Candle Data")
                    candle_data = []
                    for i, candle in enumerate(candles):
                        candle_data.append({
                            "Number": i+1,
                            "Start Time": candle.start_timestamp.strftime('%Y-%m-%d %H:%M:%S') if candle.start_timestamp else "Unknown",
                            "Open": f"{candle.open_price:.2f}",
                            "High": f"{candle.high_price:.2f}",
                            "Low": f"{candle.low_price:.2f}",
                            "Close": f"{candle.close_price:.2f}",
                            "Volume": f"{candle.volume:.4f}",
                            "Delta Ratio": f"{candle.delta_ratio:.2f}",
                            "Trade Count": f"{candle.trade_count}",
                            "Duration": f"{(candle.end_time - candle.start_time)/1000:.1f}s"
                        })
                    
                    candle_df = pd.DataFrame(candle_data)
                    st.dataframe(candle_df)
        else:
            st.info(f"""
            These volume candles are reconstructed from your metrics data, simulating how your bot's 
            VolumeBasedCandleBuffer works. Each candle completes when either:
            1. The volume threshold is reached (currently set to {volume_threshold} BTC)
            2. Maximum time has passed (currently set to {max_time} seconds)
            
            Market order detection threshold: {market_threshold}% price change
            
            For more accurate candles, enable "Pull data from Thalex exchange" in the sidebar.
            """)
            
            # Generate volume candles from metrics data
            candles = generate_volume_candles(filtered_df, volume_threshold, max_time, market_threshold)
            
            if not candles:
                st.warning("Not enough data to generate volume candles.")
            else:
                # Plot candles
                st.write(f"Generated {len(candles)} volume candles from available data")
                fig = plot_volume_candles(candles, title=f"Volume Candles (Threshold: {volume_threshold} BTC, Max Time: {max_time}s)")
                st.plotly_chart(fig, use_container_width=True)
                
                # Display candle data as table
                st.subheader("Volume Candle Data")
                candle_data = []
                for i, candle in enumerate(candles):
                    candle_data.append({
                        "Number": i+1,
                        "Start Time": candle.start_timestamp.strftime('%Y-%m-%d %H:%M:%S') if candle.start_timestamp else "Unknown",
                        "Open": f"{candle.open_price:.2f}",
                        "High": f"{candle.high_price:.2f}",
                        "Low": f"{candle.low_price:.2f}",
                        "Close": f"{candle.close_price:.2f}",
                        "Volume": f"{candle.volume:.4f}",
                        "Delta Ratio": f"{candle.delta_ratio:.2f}",
                        "Duration (sec)": f"{(candle.end_time - candle.start_time)/1000:.1f}"
                    })
                
                candle_df = pd.DataFrame(candle_data)
                st.dataframe(candle_df)

if __name__ == "__main__":
    create_dashboard()
    
    # Add auto-refresh functionality
    # Uncomment to enable auto-refresh every 60 seconds
    # time.sleep(60)
    # st.rerun()  # Changed from st.experimental_rerun()