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
            "base_size": 0.001,             # Base quote size increased from 0.01
            "size_multipliers": [0.2, 0.4, 0.6, 1.0, 1.6, 2.0],  # Size multipliers with Fibonacci-like progression
            "max_levels": 12,              # Maximum number of quote levels (reduced from 21)
            "level_spacing": 100,          # Base spacing between levels in ticks (increased from 50 to 100)
            "vamp_spread_sensitivity": 0.5,  # Sensitivity of spread to VAMP impact
            "vamp_skew_sensitivity": 0.001,  # Sensitivity of quote skew to VAMP impact
            # Predictive adjustment thresholds
            "pa_gamma_adj_threshold": 0.05,
            "pa_kappa_adj_threshold": 0.05,
            "pa_res_price_offset_adj_threshold": 0.00005,
            "pa_volatility_adj_threshold": 0.05,
            "pa_prediction_max_age_seconds": 300,
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
            "impact_window": 30            # Number of market impact samples
        },

        # Volume Candle configuration for predictive analysis
        "volume_candle": {
            "threshold": 1.0,               # Volume threshold for candle formation (e.g., in BTC)
            "max_candles": 100,             # Max number of candles to store
            "max_time_seconds": 300,        # Max time for a candle to form if volume threshold not met
            "use_exchange_data": False,     # Whether to use exchange's kline data as a source
            "fetch_interval_seconds": 60,   # Interval to fetch exchange data if use_exchange_data is True
            "lookback_hours": 1,            # Lookback hours for initial data fetch
            "prediction_update_interval": 10.0 # How often to query for new predictions
        }
    },
    
    # Orderbook specific parameters
    "orderbook": {
        "min_spread": 3,                # Minimum spread in ticks
        "base_order_size": 0.01,        # Base order size for quoting
        "min_order_size": 0.001,        # Absolute minimum order size
        "levels": 6,                    # Number of order book levels for quoting logic
        "bid_step": 10,                 # Default step for placing bid orders (in ticks)
        "ask_step": 10,                 # Default step for placing ask orders (in ticks)
        "min_size": 0.001,              # Minimum size for individual orders (used by AMM)
        "quote_lifetime": 30,           # Max lifetime for quotes in seconds
        "amend_threshold": 25,          # Price movement (in ticks) to trigger quote amendment
        "min_quote_interval": 2.0,      # Minimum interval between quote updates in seconds
        "bid_sizes": [0.3, 0.5, 0.7, 0.9, 1.0, 1.2] # Size multipliers for bid side levels
    },
    
    # Risk management
    "risk": {
        "max_position": 0.02,           # Maximum position size
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
        "max_notional": 1000,  # Maximum notional value in USD
        "max_position_notional": 800,  # Maximum notional position (80% of max_notional)
    },
    
    # Hedging configuration
    "hedging": {
        "enabled": False,  # Explicitly disable hedging
        "strategy": "notional",  # Hedging strategy (notional or delta_neutral)
        "config_path": "thalex_py/Thalex_modular/config/hedge/hedge_config.json",  # Custom config file path
        
        # Asset correlation pairs
        "pairs": {
            "BTC-PERPETUAL": {
                "hedge_assets": ["ETH-PERPETUAL"],
                "correlation_factors": [0.85]
            },
            "ETH-PERPETUAL": {
                "hedge_assets": ["BTC-PERPETUAL"],
                "correlation_factors": [1.18]
            }
        },
        
        # Execution settings
        "execution": {
            "mode": "market",  # market or limit
            "timeout": 30,  # seconds to wait for limit orders
            "slippage_tolerance": 0.001  # 0.1% slippage tolerance
        },
        
        # Rebalancing settings
        "rebalance": {
            "frequency": 300,  # seconds between rebalances
            "threshold": 0.05  # deviation threshold to trigger rebalance
        }
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

# Orderbook specific configuration
ORDERBOOK_CONFIG = BOT_CONFIG["orderbook"]

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
        "order_flow_intensity": BOT_CONFIG["trading_strategy"]["avellaneda"]["order_flow_intensity"],
        "base_spread": BOT_CONFIG["trading_strategy"]["avellaneda"]["base_spread"],
        "min_spread": BOT_CONFIG["trading_strategy"]["avellaneda"]["min_spread"],
        "max_spread": BOT_CONFIG["trading_strategy"]["avellaneda"]["max_spread"],
        "spread_multiplier": BOT_CONFIG["trading_strategy"]["avellaneda"]["spread_multiplier"],
        "inventory_weight": BOT_CONFIG["trading_strategy"]["avellaneda"]["inventory_weight"],
        "inventory_cost_factor": BOT_CONFIG["trading_strategy"]["avellaneda"]["inventory_cost_factor"],
        "position_fade_time": BOT_CONFIG["trading_strategy"]["avellaneda"]["position_fade_time"],
        "adverse_selection_threshold": BOT_CONFIG["trading_strategy"]["avellaneda"]["adverse_selection_threshold"],
        "min_profit_rebalance": BOT_CONFIG["trading_strategy"]["avellaneda"]["min_profit_rebalance"],
        "max_loss_threshold": BOT_CONFIG["trading_strategy"]["avellaneda"]["max_loss_threshold"],
        "base_size": BOT_CONFIG["trading_strategy"]["avellaneda"]["base_size"],
        "size_multipliers": BOT_CONFIG["trading_strategy"]["avellaneda"]["size_multipliers"],
        "max_levels": BOT_CONFIG["trading_strategy"]["avellaneda"]["max_levels"],
        "level_spacing": BOT_CONFIG["trading_strategy"]["avellaneda"]["level_spacing"],
        "vamp_spread_sensitivity": BOT_CONFIG["trading_strategy"]["avellaneda"]["vamp_spread_sensitivity"],
        "vamp_skew_sensitivity": BOT_CONFIG["trading_strategy"]["avellaneda"]["vamp_skew_sensitivity"],
        "pa_gamma_adj_threshold": BOT_CONFIG["trading_strategy"]["avellaneda"]["pa_gamma_adj_threshold"],
        "pa_kappa_adj_threshold": BOT_CONFIG["trading_strategy"]["avellaneda"]["pa_kappa_adj_threshold"],
        "pa_res_price_offset_adj_threshold": BOT_CONFIG["trading_strategy"]["avellaneda"]["pa_res_price_offset_adj_threshold"],
        "pa_volatility_adj_threshold": BOT_CONFIG["trading_strategy"]["avellaneda"]["pa_volatility_adj_threshold"],
        "pa_prediction_max_age_seconds": BOT_CONFIG["trading_strategy"]["avellaneda"]["pa_prediction_max_age_seconds"],
        "position_limit": BOT_CONFIG["risk"]["max_position"], 
        "exchange_fee_rate": 0.0001, 
        "fixed_volatility": 0.01, 
    },
    "quoting": {
        "levels": BOT_CONFIG["trading_strategy"]["avellaneda"]["max_levels"],
        "min_interval": BOT_CONFIG["trading_strategy"]["quote_timing"]["min_interval"],
        "max_lifetime": BOT_CONFIG["trading_strategy"]["quote_timing"]["max_lifetime"],
        "operation_interval": BOT_CONFIG["trading_strategy"]["quote_timing"]["operation_interval"],
        "max_pending": BOT_CONFIG["trading_strategy"]["quote_timing"]["max_pending"],
    },
    "order": {
        "bid_step": BOT_CONFIG["orderbook"]["bid_step"], 
        "ask_step": BOT_CONFIG["orderbook"]["ask_step"], 
        "price_decimals": BOT_CONFIG["trading_strategy"]["execution"]["price_decimals"],
        "size_decimals": BOT_CONFIG["trading_strategy"]["execution"]["size_decimals"],
        "size_increment": BOT_CONFIG["trading_strategy"]["execution"]["size_increment"],
    },
    "execution": BOT_CONFIG["trading_strategy"]["execution"],
    "volume_candle": BOT_CONFIG["trading_strategy"]["volume_candle"],
    "volatility": DEFAULT_VOLATILITY_CONFIG,
    "vamp": BOT_CONFIG["trading_strategy"]["vamp"],
}

# Risk limits configuration
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
# LEGACY CONFIGS - REMOVED - All usages should now point to TRADING_CONFIG, 
# RISK_LIMITS, or BOT_CONFIG directly.
# =============================================================================

# DeprecatedConfigDict class and _legacy_config_warning function can also be removed 
# if no other DeprecatedConfigDict instances remain.
# For now, keeping them in case other legacy configs are added/managed temporarily.

# Display warning about using legacy configs
# def _legacy_config_warning(name: str) -> None:
#     warnings.warn(
#         f"Using legacy config '{name}'. This is deprecated and will be removed in a future version. "
#         f"Please use TRADING_CONFIG or RISK_LIMITS instead.",
#         DeprecationWarning, 
#         stacklevel=2
#     )

# Create proxy dictionary that warns when accessed
# class DeprecatedConfigDict(dict):
#     def __init__(self, data: Dict[str, Any], name: str):
#         super().__init__(data)
#         self.name = name
#         
#     def __getitem__(self, key):
#         _legacy_config_warning(self.name)
#         return super().__getitem__(key)

# REMOVED: ORDERBOOK_CONFIG = TRADING_CONFIG["order"]
CONNECTION_CONFIG = BOT_CONFIG["connection"] # This one is fine as it's a direct alias to a part of BOT_CONFIG

# REMOVED Legacy Config Definitions:
# QUOTING_CONFIG
# RISK_CONFIG
# INVENTORY_CONFIG
# PERFORMANCE_CONFIG
# PERFORMANCE_METRICS
# TRADING_PARAMS

# Ensure BOT_CONFIG is accessible if needed for direct lookups by modules that previously used these.
# (It's already imported and used for TRADING_CONFIG and RISK_LIMITS, so it's available)

# Example of how a module might now access a previously proxied value:
# Old: val = QUOTING_CONFIG["min_spread"]
# New direct: val = TRADING_CONFIG["avellaneda"]["min_spread"]
# New via BOT_CONFIG: val = BOT_CONFIG["trading_strategy"]["avellaneda"]["min_spread"]

# Old: val = RISK_CONFIG["limits"]["max_position"]
# New direct: val = RISK_LIMITS["max_position"]
# New via BOT_CONFIG: val = BOT_CONFIG["risk"]["max_position"]

# The rest of the file, if any, continues below...
