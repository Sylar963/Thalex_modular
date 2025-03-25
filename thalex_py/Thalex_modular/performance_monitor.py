import os
import time
import pandas as pd
import numpy as np
import logging
import json
from datetime import datetime
from typing import Dict, List, Optional
import asyncio

class PerformanceMonitor:
    """
    Performance monitoring module for the Avellaneda market maker.
    This module collects metrics from various components and saves them to CSV files.
    """
    
    def __init__(self, output_dir: str = "metrics"):
        """Initialize the performance monitor"""
        self.output_dir = output_dir
        self.metrics_file = os.path.join(output_dir, "performance_metrics.csv")
        self.trade_file = os.path.join(output_dir, "trade_history.csv")
        self.quotes_file = os.path.join(output_dir, "quote_metrics.csv")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize metrics storage
        self.metrics_data = []
        self.trade_data = []
        self.quote_data = []
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Performance monitor initialized with output dir: {output_dir}")
        
        # Recording frequency
        self.record_interval = 60  # Record metrics every 60 seconds
        self.last_record_time = 0
        
    async def start_recording(self, quoter):
        """Start background task for recording metrics"""
        self.logger.info("Starting metrics recording task")
        try:
            while True:
                current_time = time.time()
                if current_time - self.last_record_time >= self.record_interval:
                    await self.record_metrics(quoter)
                    self.last_record_time = current_time
                await asyncio.sleep(5)  # Check every 5 seconds
        except Exception as e:
            self.logger.error(f"Error in metrics recording task: {str(e)}")
    
    async def record_metrics(self, quoter):
        """Record current performance metrics"""
        try:
            # Get current timestamp
            timestamp = datetime.now().isoformat()
            
            # Basic metrics
            metrics = {
                "timestamp": timestamp,
                "mark_price": getattr(quoter.ticker, "mark_price", 0),
                "index_price": getattr(quoter, "index", 0),
            }
            
            # Risk metrics
            if hasattr(quoter, "risk_manager"):
                try:
                    risk_metrics = quoter.risk_manager.get_risk_metrics()
                    for key, value in risk_metrics.items():
                        if key == "limit_breaches":  # Handle nested dict
                            for breach_key, breach_value in value.items():
                                metrics[f"breach_{breach_key}"] = breach_value
                        else:
                            metrics[key] = value
                            
                    # Calculate total PnL if available from position tracker
                    if hasattr(quoter, "position_tracker"):
                        try:
                            position_metrics = quoter.position_tracker.get_position_metrics()
                            metrics["realized_pnl"] = position_metrics.get("realized_pnl", 0.0)
                            metrics["unrealized_pnl"] = position_metrics.get("unrealized_pnl", 0.0)
                            metrics["total_pnl"] = position_metrics.get("total_pnl", 0.0)
                        except Exception as e:
                            self.logger.warning(f"Error getting position metrics: {str(e)}")
                except Exception as e:
                    self.logger.warning(f"Error processing risk metrics: {str(e)}")
            
            # Order metrics
            if hasattr(quoter, "order_manager"):
                try:
                    order_metrics = quoter.order_manager.get_order_metrics()
                    for key, value in order_metrics.items():
                        metrics[f"order_{key}"] = value
                except Exception as e:
                    self.logger.warning(f"Error processing order metrics: {str(e)}")
            
            # Technical metrics
            if hasattr(quoter, "technical_analysis"):
                try:
                    metrics["volatility"] = quoter.technical_analysis.get_volatility()
                    metrics["zscore"] = quoter.technical_analysis.get_zscore()
                    metrics["is_trending"] = int(quoter.technical_analysis.is_trending())
                    metrics["is_volatile"] = int(quoter.technical_analysis.is_volatile())
                    metrics["trend_strength"] = quoter.technical_analysis.get_trend_strength()
                    metrics["market_impact"] = quoter.technical_analysis.get_market_impact()
                except Exception as e:
                    self.logger.warning(f"Error processing technical metrics: {str(e)}")
            
            # Quote metrics
            if hasattr(quoter, "market_maker"):
                try:
                    # Current quotes
                    bid_quotes, ask_quotes = quoter.current_quotes
                    if bid_quotes and ask_quotes:
                        metrics["best_bid"] = bid_quotes[0].price
                        metrics["best_ask"] = ask_quotes[0].price
                        metrics["bid_ask_spread"] = ask_quotes[0].price - bid_quotes[0].price
                        metrics["mid_price"] = (ask_quotes[0].price + bid_quotes[0].price) / 2
                        
                        # Record individual quotes for detailed analysis
                        for i, quote in enumerate(bid_quotes):
                            self.quote_data.append({
                                "timestamp": timestamp,
                                "side": "bid",
                                "level": i,
                                "price": quote.price,
                                "amount": quote.amount
                            })
                        
                        for i, quote in enumerate(ask_quotes):
                            self.quote_data.append({
                                "timestamp": timestamp,
                                "side": "ask",
                                "level": i,
                                "price": quote.price,
                                "amount": quote.amount
                            })
                except Exception as e:
                    self.logger.warning(f"Error processing quote metrics: {str(e)}")
            
            # Append metrics
            self.metrics_data.append(metrics)
            
            # Save data periodically (every 10 records)
            if len(self.metrics_data) % 10 == 0:
                self.save_metrics()
                self.save_quotes()
            
            self.logger.debug(f"Recorded metrics at {timestamp}")
            
        except Exception as e:
            self.logger.error(f"Error recording metrics: {str(e)}")
            # Log the full error details for easier debugging
            import traceback
            self.logger.error(f"Full error details: {traceback.format_exc()}")
    
    def record_trade(self, trade_data: Dict):
        """Record a trade execution"""
        try:
            # Ensure timestamp is present
            if "timestamp" not in trade_data:
                trade_data["timestamp"] = datetime.now().isoformat()
            
            # Append trade data
            self.trade_data.append(trade_data)
            
            # Save trades immediately
            self.save_trades()
            
            self.logger.info(f"Recorded trade: {trade_data.get('direction', '')} {trade_data.get('amount', 0)} @ {trade_data.get('price', 0)}")
            
        except Exception as e:
            self.logger.error(f"Error recording trade: {str(e)}")
    
    def save_metrics(self):
        """Save metrics to CSV file"""
        try:
            if not self.metrics_data:
                return
                
            df = pd.DataFrame(self.metrics_data)
            
            # Check if file exists to determine if header is needed
            file_exists = os.path.isfile(self.metrics_file)
            
            # Save to CSV
            df.to_csv(
                self.metrics_file, 
                mode='a',
                header=not file_exists,
                index=False
            )
            
            self.logger.info(f"Saved {len(self.metrics_data)} metrics records to {self.metrics_file}")
            
        except Exception as e:
            self.logger.error(f"Error saving metrics: {str(e)}")
    
    def save_trades(self):
        """Save trade history to CSV file"""
        try:
            if not self.trade_data:
                return
                
            df = pd.DataFrame(self.trade_data)
            
            # Check if file exists to determine if header is needed
            file_exists = os.path.isfile(self.trade_file)
            
            # Save to CSV
            df.to_csv(
                self.trade_file, 
                mode='a',
                header=not file_exists,
                index=False
            )
            
            self.logger.info(f"Saved {len(self.trade_data)} trade records to {self.trade_file}")
            
            # Clear trade data after saving
            self.trade_data = []
            
        except Exception as e:
            self.logger.error(f"Error saving trades: {str(e)}")
    
    def save_quotes(self):
        """Save quote data to CSV file"""
        try:
            if not self.quote_data:
                return
                
            df = pd.DataFrame(self.quote_data)
            
            # Check if file exists to determine if header is needed
            file_exists = os.path.isfile(self.quotes_file)
            
            # Save to CSV
            df.to_csv(
                self.quotes_file, 
                mode='a',
                header=not file_exists,
                index=False
            )
            
            self.logger.info(f"Saved {len(self.quote_data)} quote records to {self.quotes_file}")
            
            # Clear quote data after saving
            self.quote_data = []
            
        except Exception as e:
            self.logger.error(f"Error saving quotes: {str(e)}") 