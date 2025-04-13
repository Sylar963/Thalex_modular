"""
Orderbook Manager for Thalex

This module provides a wrapper around the Numba-accelerated ThalexHFTOrderbook class,
adding logging, error handling, and integration with the rest of the system.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import time
import asyncio

from .thalex_hft_orderbook import ThalexHFTOrderbook
from ..logging import LoggerFactory
from ..config.market_config import MARKET_CONFIG


class OrderbookManager:
    """
    Manages the high-performance orderbook for Thalex exchange.
    
    This class wraps the Numba-accelerated ThalexHFTOrderbook class and provides
    additional functionality like logging, error handling, and integration with
    the rest of the system.
    """
    
    def __init__(self, tick_size: float = 0.01, lot_size: float = 0.001, num_levels: int = 2500):
        """
        Initialize the orderbook manager.
        
        Args:
            tick_size: Minimum price increment
            lot_size: Minimum size increment
            num_levels: Maximum number of price levels to track (default: 2500)
        """
        # Configure logger
        self.logger = LoggerFactory.configure_component_logger(
            "orderbook_manager",
            log_file="orderbook_manager.log",
            high_frequency=True
        )
        
        # Create the high-performance orderbook
        self.orderbook = ThalexHFTOrderbook(tick_size, lot_size, num_levels)
        
        # State tracking
        self.last_update_time = 0.0
        self.snapshot_requested = False
        self.snapshot_request_time = 0.0
        self.update_count = 0
        self.sequence_mismatches = 0
        self.book_crosses = 0
        self.instrument_name = MARKET_CONFIG.get("underlying", "BTCUSD")
        
        self.logger.info(f"OrderbookManager initialized for {self.instrument_name} with tick_size={tick_size}, lot_size={lot_size}")
    
    def process_update(self, message: Dict[str, Any]) -> bool:
        """
        Process an incoming orderbook update message.
        
        Args:
            message: Parsed WebSocket message containing orderbook update
            
        Returns:
            bool: True if update was processed successfully
        """
        try:
            # Fast path - most common case first for better branch prediction
            # Directly delegate to the high-performance orderbook to minimize overhead
            self.orderbook.ingest_thalex_update(message)
            
            # Update only essential metrics - minimize work during critical path
            self.last_update_time = time.time()
            self.update_count += 1
            
            # Log only occasionally to avoid logging overhead
            if self.update_count % 1000 == 0:  # Reduced logging frequency from 100 to 1000
                self.logger.debug(f"Processed {self.update_count} updates")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error processing orderbook update: {str(e)}", exc_info=True)
            return False
    
    def get_orderbook_metrics(self) -> Dict[str, Any]:
        """
        Get current orderbook metrics for monitoring and analytics.
        
        Returns:
            Dict containing key orderbook metrics
        """
        # Use a more efficient approach by calculating metrics only when needed
        current_time = time.time()
        
        # Only compute the most essential metrics by default
        metrics = {
            "best_bid": self.orderbook.get_best_bid(),
            "best_ask": self.orderbook.get_best_ask(),
            "mid_price": self.orderbook.get_mid_price(),
            "spread": self.orderbook.get_spread(),
            "bid_levels": self.orderbook._bid_len,
            "ask_levels": self.orderbook._ask_len,
            "update_count": self.update_count,
            "last_update_age": current_time - self.last_update_time,
        }
        
        return metrics
    
    def get_slippage_impact(self, is_buy: bool, size: float) -> Tuple[float, float]:
        """
        Calculate slippage impact for a given order size.
        
        Args:
            is_buy: True if calculating for a buy order, False for sell
            size: Order size in base currency
            
        Returns:
            Tuple of (average execution price, slippage percentage from mid)
        """
        # Get current mid price as reference
        mid_price = self.orderbook.get_mid_price()
        if mid_price == 0:
            return 0.0, 0.0
            
        # Calculate average execution price
        avg_price = self.orderbook.get_slippage(is_buy, size)
        if avg_price == 0:
            return 0.0, 0.0
            
        # Calculate slippage as percentage from mid
        slippage_pct = ((avg_price - mid_price) / mid_price) * 100 if is_buy else ((mid_price - avg_price) / mid_price) * 100
            
        return avg_price, slippage_pct
    
    def get_book_depth(self, levels: int = 10) -> Dict[str, List]:
        """
        Get a snapshot of the orderbook to a specified depth.
        
        Args:
            levels: Number of price levels to return
            
        Returns:
            Dict with bids and asks arrays
        """
        # Fast path for empty book
        if self.orderbook._bid_len == 0 and self.orderbook._ask_len == 0:
            return {"bids": [], "asks": []}
        
        # Pre-allocate arrays to the right size for better performance
        bids = self.orderbook.get_bids()
        asks = self.orderbook.get_asks()
        
        # Use more efficient slicing
        bid_levels = min(levels, len(bids))
        ask_levels = min(levels, len(asks))
        
        # Convert to list only once at the end
        return {
            "bids": bids[:bid_levels].tolist() if isinstance(bids, np.ndarray) else bids[:bid_levels],
            "asks": asks[:ask_levels].tolist() if isinstance(asks, np.ndarray) else asks[:ask_levels],
        }
    
    def is_ready(self) -> bool:
        """
        Check if the orderbook is initialized and ready for use.
        
        Returns:
            bool: True if the orderbook is ready
        """
        return self.orderbook.warmed_up
    
    def reset(self) -> None:
        """Reset the orderbook state."""
        self.orderbook.reset()
        self.last_update_time = 0.0
        self.snapshot_requested = False
        self.snapshot_request_time = 0.0
        self.update_count = 0
        self.logger.info("Orderbook reset")
    
    async def request_snapshot(self, request_callback) -> None:
        """
        Request a fresh snapshot of the orderbook.
        
        Args:
            request_callback: Async function to call to request the snapshot
        """
        if self.snapshot_requested and time.time() - self.snapshot_request_time < 5:
            self.logger.debug("Snapshot already requested recently, skipping")
            return
            
        self.logger.info("Requesting orderbook snapshot")
        self.snapshot_requested = True
        self.snapshot_request_time = time.time()
        
        try:
            await request_callback()
        except Exception as e:
            self.logger.error(f"Error requesting orderbook snapshot: {str(e)}")
            self.snapshot_requested = False 