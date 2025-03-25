import pandas as pd
import numpy as np
from datetime import datetime
import os

class ThalexDataLoader:
    """Data loader for Thalex trading data"""
    
    def __init__(self, metrics_dir: str = "metrics"):
        """Initialize the data loader
        
        Args:
            metrics_dir: Directory containing the metrics files
        """
        self.metrics_dir = metrics_dir
        self.metrics_file = os.path.join(metrics_dir, "performance_metrics.csv")
        self.trade_file = os.path.join(metrics_dir, "trade_history.csv")
        self.quotes_file = os.path.join(metrics_dir, "quote_metrics.csv")
        
        # Initialize DataFrames
        self.metrics_df = None
        self.trades_df = None
        self.quotes_df = None
        
        # Load data
        self.load_data()
        
    def load_data(self):
        """Load data from CSV files"""
        try:
            # Load metrics data
            if os.path.exists(self.metrics_file):
                print(f"Loading metrics from {self.metrics_file}")
                self.metrics_df = pd.read_csv(self.metrics_file)
                if 'timestamp' in self.metrics_df.columns:
                    self.metrics_df['timestamps'] = pd.to_datetime(self.metrics_df['timestamp'])
                print(f"Loaded metrics shape: {self.metrics_df.shape}")
            else:
                print(f"Metrics file not found: {self.metrics_file}")
                self.metrics_df = pd.DataFrame()
                
            # Load trade data
            if os.path.exists(self.trade_file):
                print(f"Loading trades from {self.trade_file}")
                self.trades_df = pd.read_csv(self.trade_file)
                if 'timestamp' in self.trades_df.columns:
                    self.trades_df['timestamps'] = pd.to_datetime(self.trades_df['timestamp'])
                print(f"Loaded trades shape: {self.trades_df.shape}")
            else:
                print(f"Trades file not found: {self.trade_file}")
                self.trades_df = pd.DataFrame()
                
            # Load quotes data
            if os.path.exists(self.quotes_file):
                print(f"Loading quotes from {self.quotes_file}")
                self.quotes_df = pd.read_csv(self.quotes_file)
                if 'timestamp' in self.quotes_df.columns:
                    self.quotes_df['timestamps'] = pd.to_datetime(self.quotes_df['timestamp'])
                print(f"Loaded quotes shape: {self.quotes_df.shape}")
            else:
                print(f"Quotes file not found: {self.quotes_file}")
                self.quotes_df = pd.DataFrame()
                
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            self.metrics_df = pd.DataFrame()
            self.trades_df = pd.DataFrame()
            self.quotes_df = pd.DataFrame()
            
    def to_dataframe(self) -> pd.DataFrame:
        """Convert data to a single DataFrame for visualization
        
        Returns:
            DataFrame containing all metrics and trading data
        """
        try:
            # Start with metrics data
            if self.metrics_df is None or self.metrics_df.empty:
                print("No metrics data available")
                return pd.DataFrame()
                
            df = self.metrics_df.copy()
            print(f"Initial metrics columns: {df.columns.tolist()}")
            
            # Ensure we have timestamps
            if 'timestamps' not in df.columns:
                if 'timestamp' in df.columns:
                    print("Converting timestamp to datetime")
                    df['timestamps'] = pd.to_datetime(df['timestamp'])
                    df = df.drop('timestamp', axis=1)
                else:
                    print("No timestamp column found")
                    return pd.DataFrame()
                    
            # Remove duplicate timestamps, keeping the latest entry
            before_dedup = len(df)
            df = df.sort_values('timestamps').drop_duplicates('timestamps', keep='last')
            print(f"Removed {before_dedup - len(df)} duplicate timestamps")
            
            # Add trade data if available
            if self.trades_df is not None and not self.trades_df.empty:
                trade_data = self.trades_df.copy()
                if 'timestamps' not in trade_data.columns and 'timestamp' in trade_data.columns:
                    trade_data['timestamps'] = pd.to_datetime(trade_data['timestamp'])
                    trade_data = trade_data.drop('timestamp', axis=1)
                trade_data = trade_data.sort_values('timestamps').drop_duplicates('timestamps', keep='last')
                df = df.merge(trade_data, on='timestamps', how='left')
            
            # Add quote data if available
            if self.quotes_df is not None and not self.quotes_df.empty:
                quote_data = self.quotes_df.copy()
                if 'timestamps' not in quote_data.columns and 'timestamp' in quote_data.columns:
                    quote_data['timestamps'] = pd.to_datetime(quote_data['timestamp'])
                    quote_data = quote_data.drop('timestamp', axis=1)
                quote_data = quote_data.sort_values('timestamps').drop_duplicates('timestamps', keep='last')
                df = df.merge(quote_data, on='timestamps', how='left')
            
            # Sort by timestamp
            df = df.sort_values('timestamps')
            
            # Forward fill missing values
            df = df.ffill()
            
            # Calculate total PnL if components exist
            if 'realized_pnl' in df.columns and 'unrealized_pnl' in df.columns:
                df['pnl'] = df['realized_pnl'] + df['unrealized_pnl']
            
            # Ensure required columns exist and convert to float
            required_columns = ['mark_price', 'position_size', 'pnl', 'volatility', 'zscore']
            for col in required_columns:
                if col not in df.columns:
                    print(f"Warning: Required column '{col}' not found in data")
                    df[col] = 0.0
                else:
                    print(f"Column {col} exists with dtype: {df[col].dtype}")
                    # Convert to float if needed
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
            
            print(f"Final DataFrame shape: {df.shape}")
            print(f"Final columns: {df.columns.tolist()}")
            print(f"Sample of timestamps: {df['timestamps'].head().tolist()}")
            
            return df
            
        except Exception as e:
            print(f"Error creating DataFrame: {str(e)}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame() 