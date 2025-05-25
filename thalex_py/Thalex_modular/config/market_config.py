from thalex import Network
import threading
import time
import json
import os
from typing import Dict, Any, Callable, List

# =============================================================================
# THALEX MARKET MAKER CONFIGURATION
# =============================================================================
"""
This file defines the configuration for the Thalex Avellaneda-Stoikov market maker.
It uses a layered approach to configuration:

1. PRIMARY CONFIGURATION (BOT_CONFIG):
   - Single source of truth for all settings
   - Any changes should be made here
   
2. CONSOLIDATED CONFIGURATIONS:
   - TRADING_CONFIG: Order parameters, quoting settings, and Avellaneda model
   - RISK_CONFIG: Risk limits and inventory management
   - PERFORMANCE_CONFIG: Performance metrics and thresholds
   - Other direct references (MARKET_CONFIG, TECHNICAL_PARAMS, etc.)
   
3. LEGACY CONFIGURATIONS (for backward compatibility):
   - Original configuration variables that existing code may use
   - These reference the consolidated configs rather than BOT_CONFIG directly
   - Should gradually migrate code to use consolidated configs instead

USAGE EXAMPLES:
   - New code should use consolidated configs: TRADING_CONFIG["order"]["spread"]
   - Legacy code can continue using: ORDERBOOK_CONFIG["spread"]
"""

# SUGGESTED ALTERNATIVE CONFIG STRUCTURE USING CLASSES
# This approach would reduce redundancy and provide better type checking
# You could implement this in the future if desired
"""
from thalex import Network
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Union

@dataclass
class MarketConfig:
    underlying: str
    network: Network
    label: str

@dataclass
class OrderConfig:
    spread: float
    min_spread: float
    max_spread: float
    bid_step: int
    ask_step: int
    bid_sizes: List[float]
    ask_sizes: List[float]
    threshold: float
    amend_threshold: int
    post_only: bool

@dataclass
class ThalexConfig:
    market: MarketConfig
    order: OrderConfig
    # ... other config classes
    
    # This method would give you backward compatibility
    def to_legacy_configs(self) -> Dict[str, Dict[str, Any]]:
        return {
            "MARKET_CONFIG": vars(self.market),
            "ORDERBOOK_CONFIG": {
                "spread": self.order.spread,
                # ... other mappings
            },
            # ... other legacy configs
        }

# Example usage:
config = ThalexConfig(
    market=MarketConfig("BTCUSD", Network.TEST, "P"),
    order=OrderConfig(2.0, 1.5, 5.0, 25, 25, [0.2, 0.8], [0.2, 0.8], 0.5, 25, True),
    # ... other configs
)

# For backward compatibility
MARKET_CONFIG = vars(config.market)
# etc.
"""

def validate_config():
    """Validate all BOT_CONFIG parameters for type checking, range validation, and logical consistency"""
    
    # Type checking for numeric parameters
    numeric_params = [
        ("trading_strategy.avellaneda.gamma", BOT_CONFIG["trading_strategy"]["avellaneda"]["gamma"], float),
        ("trading_strategy.avellaneda.kappa", BOT_CONFIG["trading_strategy"]["avellaneda"]["kappa"], float),
        ("trading_strategy.avellaneda.time_horizon", BOT_CONFIG["trading_strategy"]["avellaneda"]["time_horizon"], (int, float)),
        ("trading_strategy.avellaneda.base_spread", BOT_CONFIG["trading_strategy"]["avellaneda"]["base_spread"], (int, float)),
        ("trading_strategy.avellaneda.min_spread", BOT_CONFIG["trading_strategy"]["avellaneda"]["min_spread"], (int, float)),
        ("trading_strategy.avellaneda.max_spread", BOT_CONFIG["trading_strategy"]["avellaneda"]["max_spread"], (int, float)),
        ("risk.max_position", BOT_CONFIG["risk"]["max_position"], (int, float)),
        ("risk.max_notional", BOT_CONFIG["risk"]["max_notional"], (int, float)),
        ("connection.rate_limit", BOT_CONFIG["connection"]["rate_limit"], int),
    ]
    
    for param_path, value, expected_type in numeric_params:
        if not isinstance(value, expected_type):
            raise TypeError(f"Parameter {param_path} must be {expected_type}, got {type(value)}")
    
    # Range validation for positive values
    positive_params = [
        ("trading_strategy.avellaneda.gamma", BOT_CONFIG["trading_strategy"]["avellaneda"]["gamma"]),
        ("trading_strategy.avellaneda.kappa", BOT_CONFIG["trading_strategy"]["avellaneda"]["kappa"]),
        ("trading_strategy.avellaneda.time_horizon", BOT_CONFIG["trading_strategy"]["avellaneda"]["time_horizon"]),
        ("risk.max_position", BOT_CONFIG["risk"]["max_position"]),
        ("risk.max_notional", BOT_CONFIG["risk"]["max_notional"]),
        ("connection.rate_limit", BOT_CONFIG["connection"]["rate_limit"]),
    ]
    
    for param_path, value in positive_params:
        if value <= 0:
            raise ValueError(f"Parameter {param_path} must be positive, got {value}")
    
    # Logical consistency checks
    min_spread = BOT_CONFIG["trading_strategy"]["avellaneda"]["min_spread"]
    max_spread = BOT_CONFIG["trading_strategy"]["avellaneda"]["max_spread"]
    if min_spread >= max_spread:
        raise ValueError(f"min_spread ({min_spread}) must be less than max_spread ({max_spread})")
    
    base_spread = BOT_CONFIG["trading_strategy"]["avellaneda"]["base_spread"]
    if not (min_spread <= base_spread <= max_spread):
        raise ValueError(f"base_spread ({base_spread}) must be between min_spread ({min_spread}) and max_spread ({max_spread})")
    
    max_position_notional = BOT_CONFIG["risk"]["max_position_notional"]
    max_notional = BOT_CONFIG["risk"]["max_notional"]
    if max_position_notional > max_notional:
        raise ValueError(f"max_position_notional ({max_position_notional}) cannot exceed max_notional ({max_notional})")
    
    return True


# =============================================================================
# DYNAMIC CONFIGURATION RELOADING
# =============================================================================

class ConfigReloader:
    """Thread-safe configuration reloader with file watching and component notifications"""
    
    def __init__(self):
        self._lock = threading.RLock()
        self._observers: List[Callable[[Dict[str, Any]], None]] = []
        self._config_file_path = None
        self._last_reload_time = 0
        self._reload_cooldown = 2.0  # Minimum 2 seconds between reloads
        
    def add_observer(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Add a callback to be notified when configuration changes"""
        with self._lock:
            self._observers.append(callback)
    
    def remove_observer(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Remove a configuration change observer"""
        with self._lock:
            if callback in self._observers:
                self._observers.remove(callback)
    
    def _notify_observers(self, old_config: Dict[str, Any], new_config: Dict[str, Any]) -> None:
        """Notify all observers of configuration changes"""
        changes = self._get_config_changes(old_config, new_config)
        if changes:
            for observer in self._observers:
                try:
                    observer(changes)
                except Exception as e:
                    print(f"Error notifying config observer: {e}")
    
    def _get_config_changes(self, old_config: Dict[str, Any], new_config: Dict[str, Any]) -> Dict[str, Any]:
        """Get the differences between old and new configuration"""
        changes = {}
        
        def compare_dicts(old_dict, new_dict, path=""):
            for key in set(old_dict.keys()) | set(new_dict.keys()):
                current_path = f"{path}.{key}" if path else key
                
                if key not in old_dict:
                    changes[current_path] = {"action": "added", "new_value": new_dict[key]}
                elif key not in new_dict:
                    changes[current_path] = {"action": "removed", "old_value": old_dict[key]}
                elif isinstance(old_dict[key], dict) and isinstance(new_dict[key], dict):
                    compare_dicts(old_dict[key], new_dict[key], current_path)
                elif old_dict[key] != new_dict[key]:
                    changes[current_path] = {
                        "action": "modified",
                        "old_value": old_dict[key],
                        "new_value": new_dict[key]
                    }
        
        compare_dicts(old_config, new_config)
        return changes
    
    def reload_config(self, new_config: Dict[str, Any] = None) -> bool:
        """Reload configuration with thread-safe updates and validation"""
        current_time = time.time()
        
        with self._lock:
            # Rate limiting (only for successful reloads)
            if current_time - self._last_reload_time < self._reload_cooldown:
                print(f"Config reload rate limited. Last reload was {current_time - self._last_reload_time:.1f}s ago")
                return False
            
            try:
                # Save current configuration (deep copy)
                import copy
                old_config = copy.deepcopy(BOT_CONFIG)
                
                if new_config:
                    # Create a temporary copy for validation
                    temp_config = {}
                    for key, value in BOT_CONFIG.items():
                        if isinstance(value, dict):
                            temp_config[key] = dict(value)
                        else:
                            temp_config[key] = value
                    
                    # Update temp config with new values
                    self._deep_update(temp_config, new_config)
                    
                    # Validate temp configuration by temporarily updating global
                    original_bot_config = dict(BOT_CONFIG)
                    BOT_CONFIG.clear()
                    BOT_CONFIG.update(temp_config)
                    
                    try:
                        validate_config()
                    except Exception as validation_error:
                        # Restore original config on validation failure
                        BOT_CONFIG.clear()
                        BOT_CONFIG.update(original_bot_config)
                        raise validation_error
                
                # Update derived configurations
                self._update_derived_configs()
                
                # Notify observers of changes
                self._notify_observers(old_config, BOT_CONFIG)
                
                self._last_reload_time = current_time
                print(f"Configuration reloaded successfully at {time.strftime('%Y-%m-%d %H:%M:%S')}")
                return True
                
            except Exception as e:
                print(f"Configuration reload failed: {e}")
                return False
    
    def _deep_update(self, target: Dict[str, Any], source: Dict[str, Any]) -> None:
        """Deep update target dictionary with source values"""
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._deep_update(target[key], value)
            else:
                target[key] = value
    
    def _update_derived_configs(self) -> None:
        """Update all derived configuration objects"""
        global MARKET_CONFIG, TRADING_CONFIG, RISK_CONFIG, PERFORMANCE_CONFIG
        global ORDERBOOK_CONFIG, RISK_LIMITS, TRADING_PARAMS, PERFORMANCE_METRICS
        global INVENTORY_CONFIG, QUOTING_CONFIG, CONNECTION_CONFIG, CALL_IDS
        
        # Update simple pass-through configs
        MARKET_CONFIG.clear()
        MARKET_CONFIG.update(BOT_CONFIG["market"])
        
        CONNECTION_CONFIG.clear()
        CONNECTION_CONFIG.update(BOT_CONFIG["connection"])
        
        CALL_IDS.clear()
        CALL_IDS.update(BOT_CONFIG["call_ids"])
        
        # Update consolidated configs (these are more complex and would need full reconstruction)
        # For now, we'll just log that they need manual update
        print("Note: Consolidated configs (TRADING_CONFIG, RISK_CONFIG, etc.) may need manual restart for full effect")


# Global configuration reloader instance
_config_reloader = ConfigReloader()

def reload_config(new_config: Dict[str, Any] = None) -> bool:
    """Public interface for reloading configuration"""
    return _config_reloader.reload_config(new_config)

def add_config_observer(callback: Callable[[Dict[str, Any]], None]) -> None:
    """Add a callback to be notified when configuration changes"""
    _config_reloader.add_observer(callback)

def remove_config_observer(callback: Callable[[Dict[str, Any]], None]) -> None:
    """Remove a configuration change observer"""
    _config_reloader.remove_observer(callback)


# =============================================================================
# PRIMARY CONFIGURATION - SINGLE SOURCE OF TRUTH
# =============================================================================
BOT_CONFIG = {
    # Market parameters
    "market": {
        "underlying": "BTCUSD",
        "network": Network.TEST,
        "label": "P",
    },
    
    # Trading strategy parameters
    "trading_strategy": {
        # Avellaneda-Stoikov model parameters
        "avellaneda": {
            # Core model parameters
            "gamma": 0.3,                  # Risk aversion (increased from 0.2 for wider spreads)
            "kappa": 0.5,                  # Inventory risk factor
            "time_horizon": 3600,          # Time horizon in seconds (1 hour)
            "order_flow_intensity": 2.0,   # Order flow intensity parameter
            
            # Spread management
            "base_spread": 5.0,            # Base spread in ticks
            "min_spread": 3.0,             # Minimum spread in ticks
            "max_spread": 25.0,            # Maximum spread in ticks
            "spread_multiplier": 1.0,      # Dynamic spread adjustment factor
            
            # Position and inventory management
            "inventory_weight": 0.7,       # Inventory skew factor
            "inventory_cost_factor": 0.0001, # Cost of holding inventory
            "position_fade_time": 300,     # Time to fade position (seconds)
            "adverse_selection_threshold": 0.002,  # Adverse selection threshold
            "min_profit_rebalance": 0.01,  # Minimum profit to trigger rebalance
            "max_loss_threshold": 0.03,    # Maximum loss before gradual exit
            
            # Quote sizing and levels
            "base_size": 0.1,             # Base quote size
            "size_multipliers": [1.0, 2.0, 3.0, 2.0, 1.0, 1.0],  # Size multipliers for each level
            "max_levels": 6,              # Maximum number of quote levels
            "level_spacing": 35,          # Base spacing between levels in ticks
        },
        
        # Order execution parameters
        "execution": {
            "post_only": True,             # Use post-only orders
            "min_size": 0.001,             # Minimum quote size
            "max_size": 1.0,               # Maximum quote size
            "size_increment": 0.001,       # Size increment
            "price_decimals": 2,           # Price decimal places
            "size_decimals": 3,            # Size decimal places
            "max_retries": 2,              # Maximum retry attempts
        },
        
        # Quote management
        "quote_timing": {
            "min_interval": 1.0,           # Minimum time between quotes
            "max_lifetime": 10,            # Maximum quote lifetime in seconds
            "operation_interval": 0.2,     # Time between order operations
            "max_pending": 5,              # Maximum concurrent operations
        },
        
        # Market impact and cancellation
        "market_impact": {
            "threshold": 0.01,             # Market impact threshold
            "fast_cancel_threshold": 0.005, # Price movement for fast cancellation
            "aggressive_cancel": False,     # Whether to cancel aggressively
        },
        
        # VAMP (Volume Adjusted Market Pressure)
        "vamp": {
            "price_window": 50,            # Number of price-volume samples
            "aggressive_window": 20,        # Number of aggressive trade samples
            "impact_window": 30,           # Number of market impact samples
        }
    },
    
    # Risk management
    "risk": {
        "max_position": 1.0,           # Maximum position size
        "stop_loss_pct": 0.06,         # Stop loss percentage
        "take_profit_pct": 0.0023,     # Take profit percentage (set to 0.23%)
        "max_drawdown": 0.10,          # Maximum drawdown
        "max_consecutive_losses": 5,   # Maximum consecutive losses
        "inventory_target": 0.0,       # Target inventory level
        "inventory_imbalance_limit": 0.7, # Maximum inventory imbalance
        # Additional risk parameters
        "position_rebalance_threshold": 0.8, # Position utilization for rebalance
        "market_impact_threshold": 0.0025, # Market impact threshold for rebalance
        "rebalance_cooldown": 300,     # Rebalance cooldown period (5 min)
        
        # Take profit configuration
        "base_take_profit_pct": 0.02,  # Base take profit percentage
        "max_take_profit_pct": 0.05,   # Maximum take profit percentage
        "min_take_profit_pct": 0.002,  # Minimum take profit percentage (0.2%)
        
        # Position management
        "max_notional": 50000,  # Maximum notional value in USD
        "max_position_notional": 40000,  # Maximum notional position (80% of max_notional)
    },
    
    # Connection parameters
    "connection": {
        "rate_limit": 180,            # Maximum requests per minute (increased from 60)
        "retry_delay": 5,            # Delay between retries in seconds
        "heartbeat_interval": 30,    # Heartbeat interval in seconds
        "max_reconnect_attempts": 3  # Maximum reconnection attempts
    },

    # Performance monitoring parameters
    "performance": {
        "window_size": 100,          # Window size for performance metrics
        "min_trades": 10,            # Minimum trades for metrics calculation
        "metrics_update_interval": 60, # Update interval for metrics in seconds
        "max_position_time": 3600,    # Maximum position holding time in seconds
        "profit_factor_threshold": 1.5, # Minimum profit factor
        "win_rate_threshold": 0.55,   # Minimum win rate
        "risk_reward_ratio": 1.5,     # Target risk/reward ratio
        "drawdown_recovery_factor": 2.0 # Drawdown recovery factor
    },

    # Call IDs for API requests
    "call_ids": {
        # Core API calls
        "instruments": 1000,
        "ticker": 1001,
        "index": 1002,
        "portfolio": 1003,
        "orders": 1004,
        "trades": 1005,
        
        # Authentication and connection management
        "login": 1006,
        "set_cod": 1007,
        
        # Subscription and data management
        "subscribe": 1008,
        "unsubscribe": 1009,
        "heartbeat": 1010,
        "public_subscribe": 1011,
        "private_subscribe": 1012,
        
        # Order management
        "place_order": 1013,
        "cancel_order": 1014,
        "amend_order": 1015,
        "cancel_all": 1016,
        
        # Account management
        "account_info": 1017,
        "positions": 1018,
        
        # Additional subscriptions
        "subscribe_ticker": 1019,
        "subscribe_index": 1020,
        "subscribe_orders": 1021,
        "subscribe_portfolio": 1022,
        "subscribe_trades": 1023
    }
}

# =============================================================================
# CONSOLIDATED CONFIGURATIONS
# =============================================================================
"""
These configs provide a more logical organization of parameters while
maintaining references to BOT_CONFIG as the single source of truth.
New code should prefer these over the legacy configs.
"""

# Simple pass-through configs (direct references to BOT_CONFIG sections)
MARKET_CONFIG = BOT_CONFIG["market"]
CALL_IDS = BOT_CONFIG["call_ids"]

# Consolidated trading configuration (combines order parameters, quoting, and Avellaneda model)
TRADING_CONFIG = {
    # Core order parameters
    "order": {
        "spread": BOT_CONFIG["trading_strategy"]["avellaneda"]["base_spread"],
        "min_spread": BOT_CONFIG["trading_strategy"]["avellaneda"]["min_spread"],
        "max_spread": BOT_CONFIG["trading_strategy"]["avellaneda"]["max_spread"],
        "bid_step": BOT_CONFIG["trading_strategy"]["avellaneda"]["level_spacing"],
        "ask_step": BOT_CONFIG["trading_strategy"]["avellaneda"]["level_spacing"],
        "bid_sizes": BOT_CONFIG["trading_strategy"]["avellaneda"]["size_multipliers"],
        "ask_sizes": BOT_CONFIG["trading_strategy"]["avellaneda"]["size_multipliers"],
        "threshold": BOT_CONFIG["trading_strategy"]["avellaneda"]["adverse_selection_threshold"],
        "amend_threshold": BOT_CONFIG["trading_strategy"]["avellaneda"]["max_loss_threshold"],
        "post_only": BOT_CONFIG["trading_strategy"]["execution"]["post_only"]
    },
    
    # Quote management
    "quoting": {
        "min_quote_interval": BOT_CONFIG["trading_strategy"]["quote_timing"]["min_interval"],
        "quote_lifetime": BOT_CONFIG["trading_strategy"]["quote_timing"]["max_lifetime"],
        "refresh_interval": BOT_CONFIG["trading_strategy"]["quote_timing"]["min_interval"] / 2,
        "max_quote_age": BOT_CONFIG["trading_strategy"]["quote_timing"]["max_lifetime"] * 3,
        "operation_interval": BOT_CONFIG["trading_strategy"]["quote_timing"]["operation_interval"],
        "max_pending_operations": BOT_CONFIG["trading_strategy"]["quote_timing"]["max_pending"],
        "fast_cancel_threshold": BOT_CONFIG["trading_strategy"]["market_impact"]["fast_cancel_threshold"],
        "market_impact_threshold": BOT_CONFIG["trading_strategy"]["market_impact"]["threshold"],
        "min_size": BOT_CONFIG["trading_strategy"]["execution"]["min_size"],
        "max_size": BOT_CONFIG["trading_strategy"]["execution"]["max_size"],
        "size_increment": BOT_CONFIG["trading_strategy"]["execution"]["size_increment"],
        "price_decimals": BOT_CONFIG["trading_strategy"]["execution"]["price_decimals"],
        "size_decimals": BOT_CONFIG["trading_strategy"]["execution"]["size_decimals"],
        "max_retries": BOT_CONFIG["trading_strategy"]["execution"]["max_retries"],
        "buffer": BOT_CONFIG["trading_strategy"]["vamp"]["price_window"],
        "aggressive_cancel": BOT_CONFIG["trading_strategy"]["market_impact"]["aggressive_cancel"],
        "levels": BOT_CONFIG["trading_strategy"]["avellaneda"]["max_levels"],
    },
    
    # Avellaneda model parameters
    "avellaneda": {
        "gamma": BOT_CONFIG["trading_strategy"]["avellaneda"]["gamma"],
        "inventory_weight": BOT_CONFIG["trading_strategy"]["avellaneda"]["inventory_weight"], 
        "inventory_skew_factor": BOT_CONFIG["trading_strategy"]["avellaneda"]["inventory_weight"] / 2,
        "position_fade_time": BOT_CONFIG["trading_strategy"]["avellaneda"]["position_fade_time"],
        "order_flow_intensity": BOT_CONFIG["trading_strategy"]["avellaneda"]["order_flow_intensity"],
        "inventory_cost_factor": BOT_CONFIG["trading_strategy"]["avellaneda"]["inventory_cost_factor"],
        "adverse_selection_threshold": BOT_CONFIG["trading_strategy"]["avellaneda"]["adverse_selection_threshold"],
        "min_profit_rebalance": BOT_CONFIG["trading_strategy"]["avellaneda"]["min_profit_rebalance"],
        "gradual_exit_steps": BOT_CONFIG["trading_strategy"]["avellaneda"]["max_loss_threshold"],
        "max_loss_threshold": BOT_CONFIG["trading_strategy"]["avellaneda"]["max_loss_threshold"],
        "position_limit": BOT_CONFIG["risk"]["max_position"],  # Add position limit from risk config
        "kappa": BOT_CONFIG["trading_strategy"]["avellaneda"]["kappa"],
        "time_horizon": BOT_CONFIG["trading_strategy"]["avellaneda"]["time_horizon"],
        # Adding a fixed volatility value to replace dynamic calculation
        "fixed_volatility": 0.03,  # Fixed volatility value of 3% (increased for wider spreads)
        # Additional Avellaneda-specific parameters (used for dynamic spread calculation)
        "k": 2.0,               # Order flow intensity (increased from 1.5)
        "window_size": 100,     # Window for volatility estimation
        # Remove the fixed spread parameters to let Avellaneda model calculate them dynamically
    },
    
    # Volatility parameters (simplified)
    "volatility": {
        "default": 0.03,  # Default volatility (3%, increased from 2%)
        "floor": 0.02,    # Minimum volatility (2%, increased from 1%)
        "ceiling": 0.15,  # Maximum volatility (15%, increased from 10%)
        "cache_duration": 60,  # Cache duration in seconds
        "window": 100,    # Lookback window for volatility calculation
        "min_samples": 20, # Minimum samples required
        "scaling": 1.2,   # Volatility scaling factor (increased from 1.0)
        "ewm_span": 20    # Exponential weighted moving average span
    },
    
    # VAMP parameters
    "vamp": {
        "window": BOT_CONFIG["trading_strategy"]["vamp"]["price_window"],
        "aggressive_window": BOT_CONFIG["trading_strategy"]["vamp"]["aggressive_window"],
        "impact_window": BOT_CONFIG["trading_strategy"]["vamp"]["impact_window"],
    },
}

# Consolidated risk configuration (combines risk limits and inventory management)
RISK_CONFIG = {
    # Position limits
    "limits": {
        "max_position": BOT_CONFIG["risk"]["max_position"],
        "max_notional": BOT_CONFIG["risk"]["max_notional"],  # Now coming from BOT_CONFIG
        "max_position_notional": BOT_CONFIG["risk"]["max_position_notional"],  # Now coming from BOT_CONFIG
        "stop_loss_pct": BOT_CONFIG["risk"]["stop_loss_pct"],
        "take_profit_pct": BOT_CONFIG["risk"]["take_profit_pct"],
        "max_drawdown": BOT_CONFIG["risk"]["max_drawdown"],
        "max_consecutive_losses": BOT_CONFIG["risk"]["max_consecutive_losses"],
        # Take profit parameters
        "base_take_profit_pct": BOT_CONFIG["risk"]["base_take_profit_pct"],
        "max_take_profit_pct": BOT_CONFIG["risk"]["max_take_profit_pct"],
        "min_take_profit_pct": BOT_CONFIG["risk"]["min_take_profit_pct"],
        # Other default values
        "vol_scaling_factor": 1.0,  # Default volatility scaling factor
        "baseline_volatility": 0.02,  # Default baseline volatility of 2%
        "min_position_limit_factor": 0.5,  # Default minimum position limit factor
        "hard_vol_limit_factor": 2.0,  # Default hard volatility limit factor
        "max_position_duration": 24 * 60 * 60,  # Default 24 hours (in seconds)
        "max_profit_position_duration": 12 * 60 * 60,  # Default 12 hours (in seconds)
        "position_rebalance_threshold": BOT_CONFIG["risk"].get("position_rebalance_threshold", 0.8),
        "market_impact_threshold": BOT_CONFIG["risk"].get("market_impact_threshold", 0.0025),
        "rebalance_cooldown": BOT_CONFIG["risk"].get("rebalance_cooldown", 300),
        "max_quote_size": BOT_CONFIG["trading_strategy"]["execution"]["max_size"]
    },
    
    # Inventory management
    "inventory": {
        "target": BOT_CONFIG["risk"]["inventory_target"],
        "max_imbalance": BOT_CONFIG["risk"]["inventory_imbalance_limit"],
        "fade_time": BOT_CONFIG["trading_strategy"]["avellaneda"]["position_fade_time"],
        "adverse_selection_threshold": BOT_CONFIG["trading_strategy"]["avellaneda"]["adverse_selection_threshold"],
        "min_profit_rebalance": BOT_CONFIG["trading_strategy"]["avellaneda"]["min_profit_rebalance"]
    }
}

# Consolidated performance monitoring configuration
PERFORMANCE_CONFIG = {
    "metrics": {
        "window_size": BOT_CONFIG["performance"]["window_size"],
        "min_trades": BOT_CONFIG["performance"]["min_trades"],
        "update_interval": BOT_CONFIG["performance"]["metrics_update_interval"],
        "max_position_time": BOT_CONFIG["performance"]["max_position_time"]
    },
    
    "thresholds": {
        "profit_factor": BOT_CONFIG["performance"]["profit_factor_threshold"],
        "win_rate": BOT_CONFIG["performance"]["win_rate_threshold"],
        "risk_reward_ratio": BOT_CONFIG["performance"]["risk_reward_ratio"],
        "target_profit": BOT_CONFIG["risk"]["take_profit_pct"],
        "max_drawdown": BOT_CONFIG["risk"]["max_drawdown"],
        "max_consecutive_losses": BOT_CONFIG["risk"]["max_consecutive_losses"],
        "drawdown_recovery_factor": BOT_CONFIG["performance"]["drawdown_recovery_factor"]
    }
}

# Connection configuration
CONNECTION_CONFIG = BOT_CONFIG["connection"]

# =============================================================================
# LEGACY CONFIGURATIONS (BACKWARD COMPATIBILITY)
# =============================================================================
"""
These configurations maintain the original variable names used throughout the codebase.
They now reference the consolidated configs rather than BOT_CONFIG directly.
Over time, code should be migrated to use the consolidated configs instead.
"""

# Core backward compatibility configs
ORDERBOOK_CONFIG = {
    "spread": BOT_CONFIG["trading_strategy"]["avellaneda"]["base_spread"],
    "min_spread": BOT_CONFIG["trading_strategy"]["avellaneda"]["min_spread"],
    "max_spread": BOT_CONFIG["trading_strategy"]["avellaneda"]["max_spread"],
    "bid_step": BOT_CONFIG["trading_strategy"]["avellaneda"]["level_spacing"],
    "ask_step": BOT_CONFIG["trading_strategy"]["avellaneda"]["level_spacing"],
    "bid_sizes": BOT_CONFIG["trading_strategy"]["avellaneda"]["size_multipliers"],
    "ask_sizes": BOT_CONFIG["trading_strategy"]["avellaneda"]["size_multipliers"],
    "threshold": BOT_CONFIG["trading_strategy"]["avellaneda"]["adverse_selection_threshold"],
    "amend_threshold": BOT_CONFIG["trading_strategy"]["avellaneda"]["max_loss_threshold"],
    "market_impact_threshold": BOT_CONFIG["trading_strategy"]["market_impact"]["threshold"],
    "quote_lifetime": BOT_CONFIG["trading_strategy"]["quote_timing"]["max_lifetime"],
    "min_quote_interval": BOT_CONFIG["trading_strategy"]["quote_timing"]["min_interval"],
    "error_retry_interval": BOT_CONFIG["connection"]["retry_delay"],
    "max_pending_operations": BOT_CONFIG["trading_strategy"]["quote_timing"]["max_pending"],
    "fast_cancel_threshold": BOT_CONFIG["trading_strategy"]["market_impact"]["fast_cancel_threshold"],
    "base_order_size": BOT_CONFIG["trading_strategy"]["execution"]["min_size"],
    "min_order_size": BOT_CONFIG["trading_strategy"]["execution"]["min_size"],
    "levels": BOT_CONFIG["trading_strategy"]["avellaneda"]["max_levels"],
}

RISK_LIMITS = RISK_CONFIG["limits"]

TRADING_PARAMS = {
    "position_management": {
        "gamma": TRADING_CONFIG["avellaneda"]["gamma"],
        "inventory_weight": TRADING_CONFIG["avellaneda"]["inventory_weight"],
        "position_fade_time": TRADING_CONFIG["avellaneda"]["position_fade_time"],
        "order_flow_intensity": TRADING_CONFIG["avellaneda"]["order_flow_intensity"],
        "inventory_cost_factor": TRADING_CONFIG["avellaneda"]["inventory_cost_factor"],
        "max_inventory_imbalance": RISK_CONFIG["inventory"]["max_imbalance"],
        "adverse_selection_threshold": TRADING_CONFIG["avellaneda"]["adverse_selection_threshold"],
        "position_limit": RISK_CONFIG["limits"]["max_notional"],  # Using hardcoded value from RISK_CONFIG
        "kappa": TRADING_CONFIG["avellaneda"]["kappa"],
    },
    # Use values from the volatility section in TRADING_CONFIG
    "volatility": {
        "default": TRADING_CONFIG["volatility"]["default"],
        "window": TRADING_CONFIG["volatility"]["window"],
        "min_samples": TRADING_CONFIG["volatility"]["min_samples"],
        "scaling": TRADING_CONFIG["volatility"]["scaling"], 
        "floor": TRADING_CONFIG["volatility"]["floor"],
        "ceiling": TRADING_CONFIG["volatility"]["ceiling"],
        "ewm_span": TRADING_CONFIG["volatility"]["ewm_span"],
        "cache_duration": TRADING_CONFIG["volatility"]["cache_duration"]
    },
}

PERFORMANCE_METRICS = {
    "window_size": PERFORMANCE_CONFIG["metrics"]["window_size"],
    "min_trades": PERFORMANCE_CONFIG["metrics"]["min_trades"],
    "target_profit": PERFORMANCE_CONFIG["thresholds"]["target_profit"],
    "max_drawdown": PERFORMANCE_CONFIG["thresholds"]["max_drawdown"],
    "max_position_time": PERFORMANCE_CONFIG["metrics"]["max_position_time"],
    "metrics_update_interval": PERFORMANCE_CONFIG["metrics"]["update_interval"],
    "profit_factor_threshold": PERFORMANCE_CONFIG["thresholds"]["profit_factor"],
    "win_rate_threshold": PERFORMANCE_CONFIG["thresholds"]["win_rate"],
    "risk_reward_ratio": PERFORMANCE_CONFIG["thresholds"]["risk_reward_ratio"],
    "max_consecutive_losses": PERFORMANCE_CONFIG["thresholds"]["max_consecutive_losses"],
    "drawdown_recovery_factor": PERFORMANCE_CONFIG["thresholds"]["drawdown_recovery_factor"]
}

INVENTORY_CONFIG = {
    "max_inventory_imbalance": RISK_CONFIG["inventory"]["max_imbalance"],
    "target_inventory": RISK_CONFIG["inventory"]["target"],
    "inventory_fade_time": RISK_CONFIG["inventory"]["fade_time"],
    "adverse_selection_threshold": RISK_CONFIG["inventory"]["adverse_selection_threshold"],
    "inventory_skew_factor": TRADING_CONFIG["avellaneda"]["inventory_skew_factor"],
    "max_position_notional": RISK_CONFIG["limits"]["max_position_notional"],
    "min_profit_rebalance": RISK_CONFIG["inventory"]["min_profit_rebalance"],
    "gradual_exit_steps": TRADING_CONFIG["avellaneda"]["max_loss_threshold"],
    "inventory_cost_factor": TRADING_CONFIG["avellaneda"]["inventory_cost_factor"],
}

# Quoting configuration
QUOTING_CONFIG = {
    "min_quote_interval": BOT_CONFIG["trading_strategy"]["quote_timing"]["min_interval"],
    "quote_lifetime": BOT_CONFIG["trading_strategy"]["quote_timing"]["max_lifetime"],
    "order_operation_interval": BOT_CONFIG["trading_strategy"]["quote_timing"]["operation_interval"],
    "max_pending_operations": BOT_CONFIG["trading_strategy"]["quote_timing"]["max_pending"],
    "fast_cancel_threshold": BOT_CONFIG["trading_strategy"]["market_impact"]["fast_cancel_threshold"],
    "market_impact_threshold": BOT_CONFIG["trading_strategy"]["market_impact"]["threshold"],
    "min_quote_size": BOT_CONFIG["trading_strategy"]["execution"]["min_size"],
    "max_quote_size": BOT_CONFIG["trading_strategy"]["execution"]["max_size"],
    "size_increment": BOT_CONFIG["trading_strategy"]["execution"]["size_increment"],
    "price_decimals": BOT_CONFIG["trading_strategy"]["execution"]["price_decimals"],
    "size_decimals": BOT_CONFIG["trading_strategy"]["execution"]["size_decimals"],
    "max_retries": BOT_CONFIG["trading_strategy"]["execution"]["max_retries"],
    "quote_buffer": BOT_CONFIG["trading_strategy"]["vamp"]["price_window"],
    "aggressive_cancel": BOT_CONFIG["trading_strategy"]["market_impact"]["aggressive_cancel"],
    "post_only": BOT_CONFIG["trading_strategy"]["execution"]["post_only"],
    "error_retry_interval": BOT_CONFIG["connection"]["retry_delay"],
    "levels": BOT_CONFIG["trading_strategy"]["avellaneda"]["max_levels"],
    # Add base_levels for compatibility with existing code
    "base_levels": [
        {"size": 0.1, "spread_multiplier": 1.0},  # Level 0 (closest to mid)
        {"size": 0.2, "spread_multiplier": 1.2},  # Level 1
        {"size": 0.3, "spread_multiplier": 1.5},  # Level 2
        {"size": 0.2, "spread_multiplier": 1.8},  # Level 3 
        {"size": 0.1, "spread_multiplier": 2.2},  # Level 4
        {"size": 0.1, "spread_multiplier": 2.5},  # Level 5 (furthest from mid)
    ]
}

# =============================================================================
# CONFIGURATION VALIDATION
# =============================================================================
# Validate configuration on module import
validate_config()
