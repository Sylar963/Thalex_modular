import asyncio
import csv
import os
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
import json
import random
import numpy as np
from .thalex_logging import LoggerFactory

class PerformanceMonitor:
    """
    Monitors and records performance metrics for the trading system.
    Provides tools for analyzing trading performance and system health.
    """
    
    def __init__(self, output_dir: str = "metrics", record_interval: int = 60, buffer_size: int = 10):
        """
        Initialize performance monitor.
        
        Args:
            output_dir: Directory to save performance data
            record_interval: How often to record metrics (seconds)
            buffer_size: Number of records to buffer before writing to disk
        """
        self.output_dir = output_dir
        self.record_interval = record_interval
        self.buffer_size = buffer_size
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize logger
        self.logger = LoggerFactory.configure_component_logger(
            "performance_monitor",
            log_file="performance.log",  # Now uses centralized logging structure
            high_frequency=False
        )
        
        # Metrics storage
        self.metrics_history = []
        self.trade_history = []
        self.position_history = []
        
        # Buffers to reduce disk writes
        self.metrics_buffer = []
        self.trades_buffer = []
        self.system_stats_buffer = []
        
        # Current metrics snapshot
        self.current_metrics = {
            "timestamp": time.time(),
            "pnl": 0.0,
            "position": 0.0,
            "mid_price": 0.0,
            "spread": 0.0,
            "quote_count": 0,
            "active_quotes": 0,
        }
        
        # File paths
        self.metrics_file = os.path.join(output_dir, "metrics.csv")
        self.trades_file = os.path.join(output_dir, "trades.csv")
        self.system_file = os.path.join(output_dir, "system_stats.csv")
        
        # Recording frequency
        self.last_record_time = 0
        self.last_system_stats_time = 0
        self.system_stats_interval = record_interval * 5  # Less frequent for system stats
        
        # Create files with headers if they don't exist
        self._initialize_files()
        
        self.logger.info(f"Performance monitor initialized with output to {output_dir}")
    
    def _initialize_files(self):
        """Initialize files with headers if they don't exist"""
        try:
            # Initialize metrics file
            if not os.path.isfile(self.metrics_file):
                with open(self.metrics_file, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        "timestamp", "datetime", "pnl", "position", "mid_price", 
                        "spread", "quote_count", "active_quotes"
                    ])
                    
            # Initialize trades file
            if not os.path.isfile(self.trades_file):
                with open(self.trades_file, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        "timestamp", "datetime", "order_id", "direction", 
                        "price", "amount", "pnl", "position_after"
                    ])
                    
            # Initialize system stats file
            if not os.path.isfile(self.system_file):
                with open(self.system_file, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        "timestamp", "datetime", "cpu_usage", "memory_usage", 
                        "request_rate"
                    ])
        except Exception as e:
            self.logger.error(f"Error initializing files: {str(e)}")

    async def start_recording(self, quoter):
        """Start background task for recording metrics"""
        self.logger.info("Starting performance recording task")
        
        try:
            # Start continuous recording
            while True:
                try:
                    # Record metrics
                    self.record_metrics(quoter)
                    
                    # Flush buffers if needed
                    self._flush_buffers_if_needed()
                    
                    # Sleep until next recording
                    await asyncio.sleep(self.record_interval)
                except Exception as e:
                    self.logger.error(f"Error in recording loop: {str(e)}")
                    await asyncio.sleep(5)
        except Exception as e:
            self.logger.error(f"Error in recording task: {str(e)}")
    
    def _flush_buffers_if_needed(self):
        """Flush buffers to disk if they reach buffer size"""
        try:
            # Flush metrics buffer
            if len(self.metrics_buffer) >= self.buffer_size:
                self._flush_metrics_buffer()
                
            # Flush trades buffer
            if len(self.trades_buffer) >= self.buffer_size:
                self._flush_trades_buffer()
                
            # Flush system stats buffer
            if len(self.system_stats_buffer) >= self.buffer_size:
                self._flush_system_stats_buffer()
        except Exception as e:
            self.logger.error(f"Error flushing buffers: {str(e)}")
    
    def _flush_metrics_buffer(self):
        """Flush metrics buffer to disk"""
        if not self.metrics_buffer:
            return
            
        try:
            with open(self.metrics_file, 'a', newline='') as f:
                writer = csv.writer(f)
                for metric in self.metrics_buffer:
                    writer.writerow([
                        metric["timestamp"],
                        datetime.fromtimestamp(metric["timestamp"]).strftime('%Y-%m-%d %H:%M:%S'),
                        metric["pnl"],
                        metric["position"],
                        metric["mid_price"],
                        metric["spread"],
                        metric["quote_count"],
                        metric["active_quotes"]
                    ])
            # Clear buffer after writing
            self.metrics_buffer = []
        except Exception as e:
            self.logger.error(f"Error flushing metrics buffer: {str(e)}")
    
    def _flush_trades_buffer(self):
        """Flush trades buffer to disk"""
        if not self.trades_buffer:
            return
            
        try:
            with open(self.trades_file, 'a', newline='') as f:
                writer = csv.writer(f)
                for trade in self.trades_buffer:
                    writer.writerow([
                        trade["timestamp"],
                        datetime.fromtimestamp(trade["timestamp"]).strftime('%Y-%m-%d %H:%M:%S'),
                        trade["order_id"],
                        trade["direction"],
                        trade["price"],
                        trade["amount"],
                        trade["pnl"],
                        trade["position_after"]
                    ])
            # Clear buffer after writing
            self.trades_buffer = []
        except Exception as e:
            self.logger.error(f"Error flushing trades buffer: {str(e)}")
    
    def _flush_system_stats_buffer(self):
        """Flush system stats buffer to disk"""
        if not self.system_stats_buffer:
            return
            
        try:
            with open(self.system_file, 'a', newline='') as f:
                writer = csv.writer(f)
                for stat in self.system_stats_buffer:
                    writer.writerow([
                        stat["timestamp"],
                        datetime.fromtimestamp(stat["timestamp"]).strftime('%Y-%m-%d %H:%M:%S'),
                        stat.get("cpu_usage", 0.0),
                        stat.get("memory_usage", 0.0),
                        stat.get("request_rate", 0.0)
                    ])
            # Clear buffer after writing
            self.system_stats_buffer = []
        except Exception as e:
            self.logger.error(f"Error flushing system stats buffer: {str(e)}")
    
    def record_metrics(self, quoter):
        """Record current metrics to buffer"""
        try:
            # Avoid recording too frequently
            current_time = time.time()
            if current_time - self.last_record_time < self.record_interval:
                return
                
            self.last_record_time = current_time
            
            # Collect metrics from quoter
            metrics = self._collect_metrics(quoter)
            
            # Add to buffer
            self.metrics_buffer.append(metrics)
            
            # Add to history
            self.metrics_history.append(metrics)
            
            # Limit history size to prevent memory issues
            if len(self.metrics_history) > 1000:
                self.metrics_history = self.metrics_history[-1000:]
            
            # Log summary (less frequently to reduce log spam)
            if random.random() < 0.2:  # Only log about 20% of metric collections
                self.logger.info(
                    f"Metrics collected: PnL=${metrics['pnl']:.2f}, "
                    f"Pos={metrics['position']:.4f}"
                )
                
        except Exception as e:
            self.logger.error(f"Error recording metrics: {str(e)}")
    
    def _collect_metrics(self, quoter) -> Dict:
        """Collect essential metrics from quoter object"""
        try:
            metrics = {
                "timestamp": time.time(),
                "pnl": 0.0,
                "position": 0.0,
                "mid_price": 0.0,
                "spread": 0.0,
                "quote_count": len(quoter.current_quotes[0]) + len(quoter.current_quotes[1]),
                "active_quotes": len(quoter.order_manager.active_bids) + len(quoter.order_manager.active_asks),
            }
            
            # Get position and PnL if available
            if hasattr(quoter, 'market_maker') and quoter.market_maker:
                position_metrics = quoter.market_maker.get_position_metrics()
                metrics["position"] = position_metrics.get("position", 0.0)
                metrics["pnl"] = position_metrics.get("total_pnl", 0.0)
                
            # Get market data if available
            if hasattr(quoter, 'ticker') and quoter.ticker:
                metrics["mid_price"] = quoter.ticker.mark_price
                if quoter.ticker.best_bid_price and quoter.ticker.best_ask_price:
                    metrics["spread"] = quoter.ticker.best_ask_price - quoter.ticker.best_bid_price
                    
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error collecting metrics: {str(e)}")
            return self.current_metrics
    
    def record_trade(self, trade_data: Dict):
        """Record trade information to buffer"""
        try:
            # Extract trade data
            timestamp = time.time()
            order_id = trade_data.get("order_id", "unknown")
            direction = trade_data.get("direction", "unknown")
            price = trade_data.get("price", 0.0)
            amount = trade_data.get("amount", 0.0)
            
            # Calculate position and PnL if market maker is available
            position_after = 0.0
            pnl = 0.0
            
            # Add to buffer
            trade_record = {
                "timestamp": timestamp,
                "order_id": order_id,
                "direction": direction,
                "price": price,
                "amount": amount,
                "pnl": pnl,
                "position_after": position_after
            }
            
            self.trades_buffer.append(trade_record)
            
            # Add to history (limited size)
            self.trade_history.append(trade_record)
            if len(self.trade_history) > 1000:
                self.trade_history = self.trade_history[-1000:]
            
            self.logger.info(f"Trade recorded: {direction} {amount:.4f} @ {price:.2f}")
                
        except Exception as e:
            self.logger.error(f"Error recording trade: {str(e)}")
            
    def save_system_stats(self, stats: Dict):
        """Save system performance statistics (with reduced frequency)"""
        try:
            # Check if we should record system stats
            current_time = time.time()
            if current_time - self.last_system_stats_time < self.system_stats_interval:
                return
                
            self.last_system_stats_time = current_time
            
            # Add timestamp
            stats["timestamp"] = current_time
            
            # Add to buffer
            self.system_stats_buffer.append(stats)
                
        except Exception as e:
            self.logger.error(f"Error saving system stats: {str(e)}")

    def get_performance_summary(self) -> Dict:
        """Get summary of trading performance (in-memory, no disk operations)"""
        try:
            if not self.metrics_history:
                return {"status": "No data available"}
                
            # Calculate summary statistics
            latest = self.metrics_history[-1]
            
            # Calculate daily PnL if enough data
            daily_pnl = 0.0
            if len(self.metrics_history) > (24 * 60 * 60) // self.record_interval:
                start_idx = -((24 * 60 * 60) // self.record_interval)
                start_pnl = self.metrics_history[start_idx]["pnl"]
                daily_pnl = latest["pnl"] - start_pnl
            
            return {
                "timestamp": latest["timestamp"],
                "current_pnl": latest["pnl"],
                "current_position": latest["position"],
                "daily_pnl": daily_pnl,
                "trade_count": len(self.trade_history),
                "active_quotes": latest["active_quotes"]
            }
            
        except Exception as e:
            self.logger.error(f"Error getting performance summary: {str(e)}")
            return {"status": "Error", "error": str(e)} 