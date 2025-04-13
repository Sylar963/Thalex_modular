from thalex import Network
from typing import Dict, Any
import warnings

# =============================================================================
# THALEX MARKET MAKER CONFIGURATION
# =============================================================================
"""
This file defines the configuration for the Thalex Avellaneda-Stoikov market maker.
"""

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
            "gamma": 0.2,                  # Risk aversion (reduced from 0.3 for more frequent executions)
            "kappa": 0.5,                  # Inventory risk factor
            "time_horizon": 3600,          # Time horizon in seconds (1 hour)
            "order_flow_intensity": 2.0,   # Order flow intensity parameter
            
            # Spread management
            "base_spread": 5.0,            # Base spread in ticks
            "min_spread": 3.0,             # Minimum spread in ticks
            "max_spread": 25.0,            # Maximum spread in ticks
            "spread_multiplier": 1.0,      # Dynamic spread adjustment factor
            
            # Position and inventory management
            "inventory_weight": 0.8,       # Inventory skew factor (increased from 0.7 for better skewing)
            "inventory_cost_factor": 0.0001, # Cost of holding inventory
            "position_fade_time": 300,     # Time to fade position (seconds)
            "adverse_selection_threshold": 0.002,  # Adverse selection threshold
            "min_profit_rebalance": 0.01,  # Minimum profit to trigger rebalance
            "max_loss_threshold": 0.03,    # Maximum loss before gradual exit
            
            # Quote sizing and levels
            "base_size": 0.1,             # Base quote size
            "size_multipliers": [1.0, 2.5, 3.5, 2.5, 0.8, 0.5],  # Size multipliers optimized for better fill probability
            "max_levels": 6,              # Maximum number of quote levels
            "level_spacing": 30,          # Base spacing between levels in ticks (reduced from 35)
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
            "max_lifetime": 5,             # Maximum quote lifetime in seconds (reduced from 10 to 5)
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
            "window": 50,                   # Number of price-volume samples
            "aggressive_window": 20,        # Number of aggressive trade samples
            "impact_window": 30,           # Number of market impact samples
        }
    },
    
    # Risk management
    "risk": {
        "max_position": 1.0,           # Maximum position size
        "stop_loss_pct": 0.06,         # Stop loss percentage
        "take_profit_pct": 0.0018,     # Take profit percentage (reduced from 0.23% to 0.18% for faster profit capture)
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
        "subscribe_trades": 1023,
        "mass_quote": 1024
    }
}

# =============================================================================
# DERIVATIVE CONFIGS - USED DIRECTLY BY THE APPLICATION
# =============================================================================

# Essential directly referenced configs
MARKET_CONFIG = BOT_CONFIG["market"]
CALL_IDS = BOT_CONFIG["call_ids"]

# Default volatility settings - used directly in several places
DEFAULT_VOLATILITY_CONFIG = {
    "default": 0.025,    # Default volatility (2.5%)
    "window": 75,        # Lookback window for volatility calculation
    "min_samples": 5,    # Minimum samples required 
    "scaling": 1.3,      # Volatility scaling factor
    "floor": 0.015,      # Minimum volatility (1.5%)
    "ceiling": 0.18,     # Maximum volatility (18%)
    "ewm_span": 15,      # Exponential weighted moving average span
    "cache_duration": 30 # Cache duration in seconds
}

# Main trading configuration
TRADING_CONFIG = {
    "avellaneda": {
        "gamma": BOT_CONFIG["trading_strategy"]["avellaneda"]["gamma"],
        "kappa": BOT_CONFIG["trading_strategy"]["avellaneda"]["kappa"],
        "time_horizon": BOT_CONFIG["trading_strategy"]["avellaneda"]["time_horizon"],
        "min_spread": BOT_CONFIG["trading_strategy"]["avellaneda"]["min_spread"],
        "max_spread": BOT_CONFIG["trading_strategy"]["avellaneda"]["max_spread"],
        "inventory_weight": BOT_CONFIG["trading_strategy"]["avellaneda"]["inventory_weight"],
        "inventory_cost_factor": BOT_CONFIG["trading_strategy"]["avellaneda"]["inventory_cost_factor"],
        "max_loss_threshold": BOT_CONFIG["trading_strategy"]["avellaneda"]["max_loss_threshold"],
        "min_profit_rebalance": BOT_CONFIG["trading_strategy"]["avellaneda"]["min_profit_rebalance"],
        "fixed_volatility": 0.01,  # Default fallback
        "position_fade_time": BOT_CONFIG["trading_strategy"]["avellaneda"]["position_fade_time"],
        "order_flow_intensity": BOT_CONFIG["trading_strategy"]["avellaneda"]["order_flow_intensity"],
        "adverse_selection_threshold": BOT_CONFIG["trading_strategy"]["avellaneda"]["adverse_selection_threshold"],
        "position_limit": BOT_CONFIG["risk"]["max_position"]
    },
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
        "levels": BOT_CONFIG["trading_strategy"]["avellaneda"]["max_levels"],
        "post_only": BOT_CONFIG["trading_strategy"]["execution"]["post_only"]
    },
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
    "volatility": DEFAULT_VOLATILITY_CONFIG,
    "vamp": {
        "window": BOT_CONFIG["trading_strategy"]["vamp"]["window"],
        "aggressive_window": BOT_CONFIG["trading_strategy"]["vamp"]["aggressive_window"],
        "impact_window": BOT_CONFIG["trading_strategy"]["vamp"]["impact_window"],
    },
    "volume_candle": {
        "threshold": 0.1,                # Volume required to complete a candle (BTC) - reduced for more frequent candles
        "max_candles": 200,              # Maximum number of candles to store
        "max_time_seconds": 180,         # Maximum time before forcing candle close - reduced to 3 minutes
        "enable_predictions": True,      # Whether to enable predictive features
        "prediction_update_interval": 5, # How often to update predictions (in seconds)
        "sensitivity": {                 # Sensitivity parameters for predictive signals
            "momentum": 1.5,             # How sensitive momentum signals should be (increased from 1.0)
            "reversal": 1.2,             # How sensitive reversal signals should be (increased from 1.0)
            "volatility": 1.3,           # How sensitive volatility predictions should be (increased from 1.0)
            "reservation_price": 1.5     # How much to adjust reservation price (increased from 1.0)
        }
    }
}

# Risk management configuration
RISK_LIMITS = {
    "max_position": BOT_CONFIG["risk"]["max_position"],
    "max_notional": BOT_CONFIG["risk"]["max_notional"],
    "max_position_notional": BOT_CONFIG["risk"]["max_position_notional"],
    "stop_loss_pct": BOT_CONFIG["risk"]["stop_loss_pct"],
    "take_profit_pct": BOT_CONFIG["risk"]["take_profit_pct"],
    "max_drawdown": BOT_CONFIG["risk"]["max_drawdown"],
    "max_consecutive_losses": BOT_CONFIG["risk"]["max_consecutive_losses"],
    "volatility_threshold": 0.05  # 5% volatility threshold
}

# =============================================================================
# LEGACY CONFIGS - KEPT FOR BACKWARD COMPATIBILITY
# =============================================================================

# Display warning about using legacy configs
def _legacy_config_warning(name: str) -> None:
    warnings.warn(
        f"Using legacy config '{name}'. This is deprecated and will be removed in a future version. "
        f"Please use TRADING_CONFIG or RISK_LIMITS instead.",
        DeprecationWarning, 
        stacklevel=2
    )

# Create proxy dictionary that warns when accessed
class DeprecatedConfigDict(dict):
    def __init__(self, data: Dict[str, Any], name: str):
        super().__init__(data)
        self.name = name
        
    def __getitem__(self, key):
        _legacy_config_warning(self.name)
        return super().__getitem__(key)

# Define direct backward compatibility mappings
ORDERBOOK_CONFIG = TRADING_CONFIG["order"]
CONNECTION_CONFIG = BOT_CONFIG["connection"]

# Create legacy configurations with deprecation warnings
QUOTING_CONFIG = DeprecatedConfigDict({
    **TRADING_CONFIG["quoting"],
    "post_only": BOT_CONFIG["trading_strategy"]["execution"]["post_only"],
    "error_retry_interval": BOT_CONFIG["connection"]["retry_delay"],
    # Add base_levels for compatibility with existing code
    "base_levels": [
        {"size": 0.1, "spread_multiplier": 1.0},  # Level 0 (closest to mid)
        {"size": 0.2, "spread_multiplier": 1.2},  # Level 1
        {"size": 0.3, "spread_multiplier": 1.5},  # Level 2
        {"size": 0.2, "spread_multiplier": 1.8},  # Level 3 
        {"size": 0.1, "spread_multiplier": 2.2},  # Level 4
        {"size": 0.1, "spread_multiplier": 2.5},  # Level 5 (furthest from mid)
    ]
}, "QUOTING_CONFIG")

RISK_CONFIG = DeprecatedConfigDict({
    # Position limits
    "limits": RISK_LIMITS,
    
    # Inventory management
    "inventory": {
        "target": BOT_CONFIG["risk"]["inventory_target"],
        "max_imbalance": BOT_CONFIG["risk"]["inventory_imbalance_limit"],
        "fade_time": BOT_CONFIG["trading_strategy"]["avellaneda"]["position_fade_time"],
        "adverse_selection_threshold": BOT_CONFIG["trading_strategy"]["avellaneda"]["adverse_selection_threshold"],
        "min_profit_rebalance": BOT_CONFIG["trading_strategy"]["avellaneda"]["min_profit_rebalance"]
    }
}, "RISK_CONFIG")

INVENTORY_CONFIG = DeprecatedConfigDict({
    "max_inventory_imbalance": RISK_CONFIG["inventory"]["max_imbalance"],
    "target_inventory": RISK_CONFIG["inventory"]["target"],
    "inventory_fade_time": RISK_CONFIG["inventory"]["fade_time"],
    "adverse_selection_threshold": RISK_CONFIG["inventory"]["adverse_selection_threshold"],
    "inventory_skew_factor": TRADING_CONFIG["avellaneda"]["inventory_weight"],
    "max_position_notional": RISK_LIMITS["max_position_notional"],
    "min_profit_rebalance": RISK_CONFIG["inventory"]["min_profit_rebalance"],
    "gradual_exit_steps": TRADING_CONFIG["avellaneda"]["max_loss_threshold"],
    "inventory_cost_factor": TRADING_CONFIG["avellaneda"]["inventory_cost_factor"],
}, "INVENTORY_CONFIG")

PERFORMANCE_CONFIG = DeprecatedConfigDict({
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
}, "PERFORMANCE_CONFIG")

PERFORMANCE_METRICS = DeprecatedConfigDict({
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
}, "PERFORMANCE_METRICS")

TRADING_PARAMS = DeprecatedConfigDict({
    "position_management": {
        "gamma": BOT_CONFIG["trading_strategy"]["avellaneda"]["gamma"],
        "inventory_weight": BOT_CONFIG["trading_strategy"]["avellaneda"]["inventory_weight"],
        "position_fade_time": BOT_CONFIG["trading_strategy"]["avellaneda"]["position_fade_time"],
        "order_flow_intensity": BOT_CONFIG["trading_strategy"]["avellaneda"]["order_flow_intensity"],
        "inventory_cost_factor": BOT_CONFIG["trading_strategy"]["avellaneda"]["inventory_cost_factor"],
        "max_inventory_imbalance": BOT_CONFIG["risk"]["inventory_imbalance_limit"],
        "adverse_selection_threshold": BOT_CONFIG["trading_strategy"]["avellaneda"]["adverse_selection_threshold"],
        "kappa": BOT_CONFIG["trading_strategy"]["avellaneda"]["kappa"],
    },
    "volatility": DEFAULT_VOLATILITY_CONFIG,
    "position_limit": BOT_CONFIG["risk"]["max_position"],
}, "TRADING_PARAMS")
