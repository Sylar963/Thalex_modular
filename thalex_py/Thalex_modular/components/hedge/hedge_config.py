"""
Configuration settings for the hedge manager and strategies.
This file defines hedge pairs, correlation factors, and execution parameters.
"""

from typing import Dict, List, Tuple, Optional
import json
import os

# Default hedge configuration
DEFAULT_HEDGE_CONFIG = {
    # Define asset pairs for hedging (primary -> hedge)
    "hedge_pairs": {
        "BTC-PERP": {
            "hedge_assets": ["ETH-PERP"],
            "correlation_factors": [0.85],  # Correlation-based hedge ratio
            "min_hedge_size": 0.01,  # Minimum size for hedge orders
            "slippage_tolerance": 0.001,  # Max acceptable slippage (0.1%)
        },
        "ETH-PERP": {
            "hedge_assets": ["BTC-PERP"],
            "correlation_factors": [1.18],  # Inverse of BTC->ETH correlation
            "min_hedge_size": 0.1,
            "slippage_tolerance": 0.001,
        }
    },
    
    # General hedge settings
    "hedge_settings": {
        "enabled": True,
        "execution_mode": "market",  # "market" or "limit"
        "execution_timeout": 30,  # Seconds to wait for limit orders
        "rebalance_frequency": 300,  # How often to rebalance hedges (seconds)
        "deviation_threshold": 0.05,  # Rebalance if hedge deviates by 5%
        "profit_target": 0.02,  # Take profit at 2% for hedge positions
        "stop_loss": 0.05,  # Stop loss at 5% for hedge positions
        "max_hedge_ratio": 1.0,  # Maximum hedge ratio (1.0 = full hedge)
        "price_feed_timeout": 5.0,  # Seconds to wait for price data
    },
    
    # Delta calculation settings
    "delta_settings": {
        "calculation_method": "notional",  # "notional" or "risk_based"
        "use_mark_price": True,  # Use mark price for calculations
        "delta_threshold": 0.01,  # Min delta change to trigger hedge update
        "portfolio_delta_target": 0.0,  # Target portfolio delta (0 = neutral)
    }
}

class HedgeConfig:
    """Manages configuration for the hedge system"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize hedge configuration
        
        Args:
            config_path: Optional path to JSON configuration file
        """
        self.config = DEFAULT_HEDGE_CONFIG.copy()
        
        # Load custom config if provided
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    custom_config = json.load(f)
                    self._merge_config(custom_config)
            except Exception as e:
                print(f"Error loading hedge config: {e}")
    
    def _merge_config(self, custom_config: Dict) -> None:
        """Merge custom config with default config"""
        if "hedge_pairs" in custom_config:
            for pair, settings in custom_config["hedge_pairs"].items():
                if pair in self.config["hedge_pairs"]:
                    self.config["hedge_pairs"][pair].update(settings)
                else:
                    self.config["hedge_pairs"][pair] = settings
        
        if "hedge_settings" in custom_config:
            self.config["hedge_settings"].update(custom_config["hedge_settings"])
            
        if "delta_settings" in custom_config:
            self.config["delta_settings"].update(custom_config["delta_settings"])
    
    def get_hedge_assets(self, primary_asset: str) -> List[str]:
        """Get list of hedge assets for a primary asset"""
        if primary_asset in self.config["hedge_pairs"]:
            return self.config["hedge_pairs"][primary_asset]["hedge_assets"]
        return []
    
    def get_correlation_factors(self, primary_asset: str) -> List[float]:
        """Get correlation factors for hedge assets"""
        if primary_asset in self.config["hedge_pairs"]:
            return self.config["hedge_pairs"][primary_asset]["correlation_factors"]
        return []
    
    def get_hedge_pair_config(self, primary_asset: str) -> Dict:
        """Get complete config for a hedge pair"""
        if primary_asset in self.config["hedge_pairs"]:
            return self.config["hedge_pairs"][primary_asset]
        return {}
    
    def is_hedging_enabled(self) -> bool:
        """Check if hedging is enabled globally"""
        return self.config["hedge_settings"]["enabled"]
    
    def get_hedge_settings(self) -> Dict:
        """Get general hedge settings"""
        return self.config["hedge_settings"]
    
    def get_delta_settings(self) -> Dict:
        """Get delta calculation settings"""
        return self.config["delta_settings"] 