# dashboard/monitor_enhanced.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import os
import datetime as dt
import numpy as np
import sys
from datetime import datetime, timedelta
import json
from typing import Dict, List, Optional

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

# Try to import bot modules
try:
    from thalex_py.Thalex_modular.performance_monitor import PerformanceMonitor
    BOT_MODULES_AVAILABLE = True
except ImportError:
    BOT_MODULES_AVAILABLE = False

# Try to import Thalex client
try:
    import thalex as th
    THALEX_AVAILABLE = True
except ImportError:
    THALEX_AVAILABLE = False

st.set_page_config(
    layout="wide", 
    page_title="Thalex Avellaneda-Stoikov Dashboard",
    page_icon="📈",
    initial_sidebar_state="expanded"
)

# Custom CSS for TradingView-like styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #ffffff;
        text-align: center;
        margin-bottom: 1rem;
        background: linear-gradient(90deg, #1f4e79, #2e86de);
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .status-indicator {
        width: 12px;
        height: 12px;
        border-radius: 50%;
        display: inline-block;
        margin-right: 8px;
    }
    
    .status-online { background-color: #2ecc71; }
    .status-offline { background-color: #e74c3c; }
</style>
""", unsafe_allow_html=True)

class TradingViewChart:
    """Create TradingView-like charts"""
    
    @staticmethod
    def create_price_chart(df: pd.DataFrame, title: str = "Price Chart") -> go.Figure:
        """Create TradingView-style price chart"""
        # Make a copy to avoid SettingWithCopyWarning
        df = df.copy()
        
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.02,
            row_heights=[0.6, 0.2, 0.2],
            subplot_titles=('Price', 'Volume', 'Volatility')
        )
        
        # Main price line
        fig.add_trace(
            go.Scatter(
                x=df['datetime'],
                y=df['mid_price'],
                mode='lines',
                name='Mid Price',
                line=dict(color='#2196F3', width=2)
            ), row=1, col=1
        )
        
        # Volume estimation - now safe to modify
        if 'volume' not in df.columns:
            df['volume'] = np.random.uniform(0.1, 2.0, len(df))
        
        colors = ['#26a69a' if i % 2 == 0 else '#ef5350' for i in range(len(df))]
        fig.add_trace(
            go.Bar(
                x=df['datetime'],
                y=df['volume'],
                name='Volume',
                marker_color=colors,
                opacity=0.7
            ), row=2, col=1
        )
        
        # Volatility if available
        if 'volatility' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df['datetime'],
                    y=df['volatility'],
                    mode='lines',
                    name='Volatility',
                    line=dict(color='#ff9800', width=2)
                ), row=3, col=1
            )
        
        # Style the chart
        fig.update_layout(
            title=dict(text=title, font=dict(size=20, color='white')),
            paper_bgcolor='#1e1e1e',
            plot_bgcolor='#1e1e1e',
            font=dict(color='white'),
            xaxis_rangeslider_visible=False,
            height=700,
            margin=dict(l=50, r=50, t=80, b=50)
        )
        
        fig.update_xaxes(gridcolor='#333333', linecolor='#666666')
        fig.update_yaxes(gridcolor='#333333', linecolor='#666666')
        
        return fig
    
    @staticmethod
    def create_pnl_chart(df: pd.DataFrame) -> go.Figure:
        """Create PnL chart with position overlay"""
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.7, 0.3],
            subplot_titles=('PnL & Position', 'Drawdown')
        )
        
        # PnL line
        fig.add_trace(
            go.Scatter(
                x=df['datetime'],
                y=df['pnl'],
                mode='lines',
                name='PnL',
                line=dict(color='#00E676', width=3),
                fill='tonexty' if df['pnl'].iloc[0] >= 0 else 'tozeroy',
                fillcolor='rgba(0, 230, 118, 0.2)'
            ), row=1, col=1
        )
        
        # Position bars
        colors = ['#26a69a' if pos >= 0 else '#ef5350' for pos in df['position']]
        fig.add_trace(
            go.Bar(
                x=df['datetime'],
                y=df['position'],
                name='Position',
                marker_color=colors,
                opacity=0.6,
                yaxis='y2'
            ), row=1, col=1
        )
        
        # Calculate drawdown
        cumulative = df['pnl'].cumsum()
        running_max = cumulative.expanding().max()
        drawdown = cumulative - running_max
        
        fig.add_trace(
            go.Scatter(
                x=df['datetime'],
                y=drawdown,
                mode='lines',
                name='Drawdown',
                line=dict(color='#f44336', width=2),
                fill='tonexty',
                fillcolor='rgba(244, 67, 54, 0.3)'
            ), row=2, col=1
        )
        
        fig.update_layout(
            title=dict(text="PnL Analysis", font=dict(size=20, color='white')),
            paper_bgcolor='#1e1e1e',
            plot_bgcolor='#1e1e1e',
            font=dict(color='white'),
            height=600,
            yaxis2=dict(overlaying='y', side='right', title='Position')
        )
        
        fig.update_xaxes(gridcolor='#333333', linecolor='#666666')
        fig.update_yaxes(gridcolor='#333333', linecolor='#666666')
        
        return fig

class BotDataLoader:
    """Load and process bot data"""
    
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.metrics_path = os.path.join(self.base_dir, 'metrics', 'metrics.csv')
        
    def load_metrics(self) -> pd.DataFrame:
        """Load metrics with error handling"""
        if not os.path.exists(self.metrics_path):
            return pd.DataFrame()
        
        try:
            df = pd.read_csv(self.metrics_path)
            
            # Handle datetime
            if 'datetime' in df.columns:
                df['datetime'] = pd.to_datetime(df['datetime'])
            elif 'timestamp' in df.columns:
                df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
            
            return df.sort_values('datetime').reset_index(drop=True)
            
        except Exception as e:
            st.error(f"Error loading metrics: {str(e)}")
            return pd.DataFrame()
    
    def get_bot_status(self) -> Dict:
        """Get current bot status with detailed analysis"""
        status = {
            'online': False, 
            'last_update': None, 
            'data_age_minutes': None,
            'file_size_kb': 0,
            'recent_records': 0,
            'status_reason': 'File not found'
        }
        
        if os.path.exists(self.metrics_path):
            try:
                # File modification time
                last_modified = datetime.fromtimestamp(os.path.getmtime(self.metrics_path))
                status['last_update'] = last_modified
                time_diff = datetime.now() - last_modified
                status['data_age_minutes'] = time_diff.total_seconds() / 60
                
                # File size
                status['file_size_kb'] = os.path.getsize(self.metrics_path) / 1024
                
                # Check recent data activity
                df = pd.read_csv(self.metrics_path)
                if not df.empty:
                    # Count records from last 10 minutes
                    if 'timestamp' in df.columns:
                        recent_time = time.time() - 600  # 10 minutes ago
                        status['recent_records'] = len(df[df['timestamp'] > recent_time])
                    elif 'datetime' in df.columns:
                        df['datetime'] = pd.to_datetime(df['datetime'])
                        recent_time = datetime.now() - timedelta(minutes=10)
                        status['recent_records'] = len(df[df['datetime'] > recent_time])
                
                # Determine online status with multiple criteria
                file_fresh = status['data_age_minutes'] < 10  # Increased from 5 to 10 minutes
                has_recent_data = status['recent_records'] > 0
                file_not_empty = status['file_size_kb'] > 0
                
                if file_fresh and has_recent_data and file_not_empty:
                    status['online'] = True
                    status['status_reason'] = 'Active'
                elif file_fresh and file_not_empty:
                    status['online'] = True
                    status['status_reason'] = 'Recently updated'
                elif has_recent_data:
                    status['online'] = True
                    status['status_reason'] = 'Has recent data'
                elif not file_not_empty:
                    status['status_reason'] = 'Empty file'
                else:
                    status['status_reason'] = f'Stale data ({status["data_age_minutes"]:.1f}m old)'
                    
            except Exception as e:
                status['status_reason'] = f'Error reading file: {str(e)}'
        
        return status

def create_sidebar():
    """Enhanced sidebar"""
    st.sidebar.markdown("""
    <div style='text-align: center; padding: 1rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; margin-bottom: 1rem;'>
        <h2 style='color: white; margin: 0;'>🤖 Avellaneda-Stoikov</h2>
        <p style='color: white; margin: 0; opacity: 0.9;'>Market Maker Dashboard</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Bot status with detailed information
    loader = BotDataLoader()
    status = loader.get_bot_status()
    
    status_color = "online" if status['online'] else "offline"
    status_text = "ONLINE" if status['online'] else "OFFLINE"
    
    st.sidebar.markdown(f"""
    <div style='padding: 1rem; background: #2c3e50; border-radius: 8px; margin-bottom: 1rem;'>
        <p style='color: white; margin: 0;'>
            <span class='status-indicator status-{status_color}'></span>
            Bot Status: <strong>{status_text}</strong>
        </p>
        <p style='color: #bdc3c7; margin: 0; font-size: 0.8rem;'>
            {status['status_reason']}
        </p>
        {f"<p style='color: #bdc3c7; margin: 0; font-size: 0.8rem;'>Data age: {status['data_age_minutes']:.1f}m | Recent records: {status['recent_records']}</p>" if status['last_update'] else ""}
    </div>
    """, unsafe_allow_html=True)
    
    # Controls
    if st.sidebar.button("🔄 Refresh Data", type="primary"):
        st.rerun()
    
    # Time range
    time_ranges = {
        "Last hour": 1,
        "Last 6 hours": 6,
        "Last 24 hours": 24,
        "All data": None
    }
    
    selected_range = st.sidebar.selectbox("📅 Time Range", list(time_ranges.keys()), index=2)
    
    return {'time_range': time_ranges[selected_range]}

def filter_by_time(df: pd.DataFrame, hours: Optional[int]) -> pd.DataFrame:
    """Filter data by time"""
    if df.empty or hours is None:
        return df
    
    cutoff = datetime.now() - timedelta(hours=hours)
    return df[df['datetime'] > cutoff]

def create_dashboard():
    """Main dashboard"""
    st.markdown('<div class="main-header">Thalex Avellaneda-Stoikov Market Maker</div>', 
                unsafe_allow_html=True)
    
    settings = create_sidebar()
    
    # Load data
    loader = BotDataLoader()
    df = loader.load_metrics()
    
    if df.empty:
        st.error("⚠️ No metrics data found!")
        return
    
    # Filter data
    filtered_df = filter_by_time(df, settings['time_range'])
    if filtered_df.empty:
        filtered_df = df.tail(100)
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Overview", "💰 PnL", "📊 Market", "⚙️ System"])
    
    with tab1:
        create_overview_tab(filtered_df)
    
    with tab2:
        create_pnl_tab(filtered_df)
    
    with tab3:
        create_market_tab(filtered_df)
    
    with tab4:
        create_system_tab(filtered_df)

def create_overview_tab(df: pd.DataFrame):
    """Trading overview tab"""
    if df.empty:
        return
    
    # Metrics
    latest = df.iloc[-1]
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        pnl_change = latest['pnl'] - df.iloc[-2]['pnl'] if len(df) > 1 else 0
        st.metric("Current PnL", f"${latest['pnl']:.2f}", f"${pnl_change:.2f}")
    
    with col2:
        st.metric("Position", f"{latest['position']:.4f} BTC")
    
    with col3:
        st.metric("Mid Price", f"${latest['mid_price']:.2f}")
    
    with col4:
        st.metric("Spread", f"${latest['spread']:.2f}")
    
    # Main chart
    chart = TradingViewChart()
    fig = chart.create_price_chart(df, "BTC-PERPETUAL Price Action")
    st.plotly_chart(fig, use_container_width=True)

def create_pnl_tab(df: pd.DataFrame):
    """PnL analysis tab"""
    if df.empty:
        return
    
    # PnL metrics
    total_pnl = df['pnl'].iloc[-1]
    max_pnl = df['pnl'].max()
    min_pnl = df['pnl'].min()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total PnL", f"${total_pnl:.2f}")
    with col2:
        st.metric("Max PnL", f"${max_pnl:.2f}")
    with col3:
        st.metric("Min PnL", f"${min_pnl:.2f}")
    
    # PnL chart
    chart = TradingViewChart()
    fig = chart.create_pnl_chart(df)
    st.plotly_chart(fig, use_container_width=True)

def create_market_tab(df: pd.DataFrame):
    """Market analysis tab"""
    if df.empty:
        return
    
    st.subheader("🌊 Market Volatility")
    
    if 'volatility' in df.columns:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df['datetime'],
            y=df['volatility'],
            mode='lines',
            name='Volatility',
            line=dict(color='#ff9800', width=2),
            fill='tonexty',
            fillcolor='rgba(255, 152, 0, 0.2)'
        ))
        
        fig.update_layout(
            paper_bgcolor='#1e1e1e',
            plot_bgcolor='#1e1e1e',
            font=dict(color='white'),
            height=400
        )
        fig.update_xaxes(gridcolor='#333333')
        fig.update_yaxes(gridcolor='#333333')
        st.plotly_chart(fig, use_container_width=True)

def create_system_tab(df: pd.DataFrame):
    """System health tab"""
    st.subheader("🔧 System Health")
    
    loader = BotDataLoader()
    status = loader.get_bot_status()
    
    # Enhanced metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Status", "ONLINE" if status['online'] else "OFFLINE")
    with col2:
        st.metric("Data Age", f"{status['data_age_minutes']:.1f}m" if status['data_age_minutes'] else "N/A")
    with col3:
        st.metric("File Size", f"{status['file_size_kb']:.1f} KB")
    with col4:
        st.metric("Recent Records", status['recent_records'])
    
    # Status details
    st.subheader("📊 Diagnostic Information")
    
    diag_data = {
        "Metric": ["Status Reason", "Last Update", "File Exists", "Total Records", "Data Freshness"],
        "Value": [
            status['status_reason'],
            status['last_update'].strftime('%Y-%m-%d %H:%M:%S') if status['last_update'] else "N/A",
            "✅ Yes" if os.path.exists(loader.metrics_path) else "❌ No",
            len(df),
            "✅ Fresh" if status['data_age_minutes'] and status['data_age_minutes'] < 10 else "⚠️ Stale"
        ]
    }
    
    diag_df = pd.DataFrame(diag_data)
    st.dataframe(diag_df, use_container_width=True, hide_index=True)
    
    # Recent data
    st.subheader("📋 Recent Data")
    if not df.empty:
        recent = df.tail(10)[['datetime', 'pnl', 'position', 'mid_price', 'spread']].copy()
        recent['datetime'] = recent['datetime'].dt.strftime('%H:%M:%S')
        st.dataframe(recent, use_container_width=True)

if __name__ == "__main__":
    create_dashboard() 