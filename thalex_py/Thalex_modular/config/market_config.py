from thalex import Network

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
    
    # Trading strategy parameters (merged order, quoting and Avellaneda parameters)
    "trading_strategy": {
        # Core order parameters
        "spread": 2.0,                 # Base spread in ticks
        "min_spread": 1.5,             # Minimum spread in tick size
        "max_spread": 5.0,             # Maximum spread in tick size
        "bid_step": 25,                # Price step between bid levels
        "ask_step": 25,                # Price step between ask levels
        "bid_sizes": [0.2, 0.8],       # Size for each bid level
        "ask_sizes": [0.2, 0.8],       # Size for each ask level
        "threshold": 0.5,              # Maximum size per quote
        "amend_threshold": 25,         # Minimum price change to amend orders
        "post_only": True,             # Use post-only orders
        
        # Quoting parameters
        "min_quote_interval": 5.0,     # Minimum time between quotes
        "quote_lifetime": 15,          # Maximum quote lifetime in seconds
        "operation_interval": 0.5,     # Time between order operations
        "max_pending_operations": 3,   # Maximum concurrent operations
        "fast_cancel_threshold": 0.005, # Price movement for fast cancellation
        "market_impact_threshold": 0.01, # Market impact threshold
        "min_quote_size": 0.001,       # Minimum quote size
        "max_quote_size": 1.0,         # Maximum quote size
        "size_increment": 0.001,       # Size increment
        "price_decimals": 2,           # Price decimal places
        "size_decimals": 3,            # Size decimal places
        "max_retries": 2,              # Maximum retry attempts
        "quote_buffer": 5,             # Quote buffer
        "aggressive_cancel": False,    # Whether to cancel aggressively
        "levels": 2,                   # Number of quote levels
        
        # Avellaneda-Stoikov model parameters
        "gamma": 0.2,                  # Risk aversion
        "inventory_weight": 0.7,       # Inventory skew factor
        "position_fade_time": 300,     # Time to fade position (seconds)
        "order_flow_intensity": 1.5,   # Order flow intensity parameter
        "inventory_cost_factor": 0.0001, # Cost of holding inventory
        "adverse_selection_threshold": 0.002, # Adverse selection threshold
        "min_profit_rebalance": 0.01,  # Minimum profit to trigger rebalance
        "gradual_exit_steps": 4,       # Number of steps for gradual position exit
        "max_loss_threshold": 0.03,    # Maximum loss before gradual exit
        "kappa": 0.3,                  # Inventory risk factor
        "time_horizon": 3600,          # Time horizon in seconds (1 hour)
        
        # VAMP (Volume Adjusted Market Pressure) parameters
        "vamp_window": 50,             # Number of price-volume samples to keep
        "vamp_aggressive_window": 20,  # Number of aggressive trade samples to keep
        "vamp_impact_window": 30,      # Number of market impact samples to keep
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
        "take_profit_levels": [
            {"percentage": 0.0023, "size": 0.6},  # Take 60% profit at 0.23%
            {"percentage": 0.01, "size": 0.2},    # Take 20% profit at 1%
            {"percentage": 0.02, "size": 0.1},    # Take 10% profit at 2%
            {"percentage": 0.03, "size": 0.1},    # Take remaining 10% at 3%
        ],
        
        # Trailing stop configuration
        "trailing_stop_activation": 0.015,  # Activate trailing at 1.5% profit
        "trailing_stop_distance": 0.01,     # 1% trailing distance
        "trailing_stop_levels": [
            {"activation": 0.015, "distance": 0.01},  # First trailing stop
            {"activation": 0.03, "distance": 0.015},  # Second trailing stop
            {"activation": 0.05, "distance": 0.02},   # Final trailing stop
        ],
        
        # Position management
        "max_notional": 50000,  # Maximum notional value in USD
        "max_position_notional": 40000,  # Maximum notional position (80% of max_notional)
    },
    
    # Technical analysis parameters
    "technical": {
        "trend": {
            "short_period": 10,        # Short-term trend period
            "long_period": 30,         # Long-term trend period
            "confirmation_threshold": 0.6, # Trend confirmation threshold
        },
        # Additional technical analysis parameters
        "zscore": {
            "window": 100,
            "threshold": 2.0,
            "mean_reversion_factor": 0.5
        },
        "atr": {
            "period": 14,
            "multiplier": 1.0,
            "smoothing": 0.1,
            "threshold": 0.005
        },
        "volume": {
            "ma_period": 20,
            "threshold": 1.5
        },
        
        # Signal configuration
        "signal": {
            "bbands_period": 20,
            "bbands_std": 2,
            "momentum_period": 10,
            "volume_ma_period": 20,
            "min_signal_strength": 0.3,
            "signal_cooldown": 300,
            "trend_confirmation_threshold": 0.6,
            "max_position_increase": 0.2,  # Maximum position increase per signal
            "notional_utilization_threshold": 0.8,  # 80% of max notional
            "signal_size_dampening": 0.5,  # Dampen signal impact on size
            "min_trade_interval": 5,  # Minimum seconds between trades
            "max_trade_count": 3,  # Maximum trades per interval
        }
    },
    
    # Connection parameters
    "connection": {
        "rate_limit": 60,            # Maximum requests per minute
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
TECHNICAL_PARAMS = BOT_CONFIG["technical"]

# Consolidated trading configuration (combines order parameters, quoting, and Avellaneda model)
TRADING_CONFIG = {
    # Core order parameters
    "order": {
        "spread": BOT_CONFIG["trading_strategy"]["spread"],
        "min_spread": BOT_CONFIG["trading_strategy"]["min_spread"],
        "max_spread": BOT_CONFIG["trading_strategy"]["max_spread"],
        "bid_step": BOT_CONFIG["trading_strategy"]["bid_step"],
        "ask_step": BOT_CONFIG["trading_strategy"]["ask_step"],
        "bid_sizes": BOT_CONFIG["trading_strategy"]["bid_sizes"],
        "ask_sizes": BOT_CONFIG["trading_strategy"]["ask_sizes"],
        "threshold": BOT_CONFIG["trading_strategy"]["threshold"],
        "amend_threshold": BOT_CONFIG["trading_strategy"]["amend_threshold"],
        "post_only": BOT_CONFIG["trading_strategy"]["post_only"]
    },
    
    # Quote management
    "quoting": {
        "min_quote_interval": BOT_CONFIG["trading_strategy"]["min_quote_interval"],
        "quote_lifetime": BOT_CONFIG["trading_strategy"]["quote_lifetime"],
        "refresh_interval": BOT_CONFIG["trading_strategy"]["min_quote_interval"] / 2,
        "max_quote_age": BOT_CONFIG["trading_strategy"]["quote_lifetime"] * 3,
        "operation_interval": BOT_CONFIG["trading_strategy"]["operation_interval"],
        "max_pending_operations": BOT_CONFIG["trading_strategy"]["max_pending_operations"],
        "fast_cancel_threshold": BOT_CONFIG["trading_strategy"]["fast_cancel_threshold"],
        "market_impact_threshold": BOT_CONFIG["trading_strategy"]["market_impact_threshold"],
        "min_size": BOT_CONFIG["trading_strategy"]["min_quote_size"],
        "max_size": BOT_CONFIG["trading_strategy"]["max_quote_size"],
        "size_increment": BOT_CONFIG["trading_strategy"]["size_increment"],
        "price_decimals": BOT_CONFIG["trading_strategy"]["price_decimals"],
        "size_decimals": BOT_CONFIG["trading_strategy"]["size_decimals"],
        "max_retries": BOT_CONFIG["trading_strategy"]["max_retries"],
        "buffer": BOT_CONFIG["trading_strategy"]["quote_buffer"],
        "aggressive_cancel": BOT_CONFIG["trading_strategy"]["aggressive_cancel"],
        "levels": BOT_CONFIG["trading_strategy"]["levels"],
    },
    
    # Avellaneda model parameters
    "avellaneda": {
        "gamma": BOT_CONFIG["trading_strategy"]["gamma"],
        "inventory_weight": BOT_CONFIG["trading_strategy"]["inventory_weight"], 
        "inventory_skew_factor": BOT_CONFIG["trading_strategy"]["inventory_weight"] / 2,
        "position_fade_time": BOT_CONFIG["trading_strategy"]["position_fade_time"],
        "order_flow_intensity": BOT_CONFIG["trading_strategy"]["order_flow_intensity"],
        "inventory_cost_factor": BOT_CONFIG["trading_strategy"]["inventory_cost_factor"],
        "adverse_selection_threshold": BOT_CONFIG["trading_strategy"]["adverse_selection_threshold"],
        "min_profit_rebalance": BOT_CONFIG["trading_strategy"]["min_profit_rebalance"],
        "gradual_exit_steps": BOT_CONFIG["trading_strategy"]["gradual_exit_steps"],
        "max_loss_threshold": BOT_CONFIG["trading_strategy"]["max_loss_threshold"],
        "position_limit": BOT_CONFIG["risk"]["max_position"],  # Add position limit from risk config
        "kappa": BOT_CONFIG["trading_strategy"]["kappa"],
        "time_horizon": BOT_CONFIG["trading_strategy"]["time_horizon"],
        # Adding a fixed volatility value to replace dynamic calculation
        "fixed_volatility": 0.02,  # Fixed volatility value of 2%
        # Additional Avellaneda-specific parameters
        "k": 1.5,               # Order flow intensity
        "window_size": 100,     # Window for volatility estimation
        "reservation_spread": 0.002,  # Base spread as percentage
        "vol_window": 20,       # Window for volatility calculation
        "min_spread": 0.001,    # Minimum spread
        "max_spread": 0.01,     # Maximum spread
    },
    
    # Volatility parameters (simplified)
    "volatility": {
        "default": 0.02,  # Default volatility (2%)
        "floor": 0.01,    # Minimum volatility (1%)
        "ceiling": 0.10,  # Maximum volatility (10%)
        "cache_duration": 60,  # Cache duration in seconds
        "window": 100,    # Lookback window for volatility calculation
        "min_samples": 20, # Minimum samples required
        "scaling": 1.0,   # Volatility scaling factor
        "ewm_span": 20    # Exponential weighted moving average span
    },
    
    # VAMP parameters
    "vamp": {
        "window": BOT_CONFIG["trading_strategy"]["vamp_window"],
        "aggressive_window": BOT_CONFIG["trading_strategy"]["vamp_aggressive_window"],
        "impact_window": BOT_CONFIG["trading_strategy"]["vamp_impact_window"],
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
        "take_profit_levels": BOT_CONFIG["risk"]["take_profit_levels"],
        # Trailing stop parameters
        "trailing_stop_activation": BOT_CONFIG["risk"]["trailing_stop_activation"],
        "trailing_stop_distance": BOT_CONFIG["risk"]["trailing_stop_distance"],
        "trailing_stop_levels": BOT_CONFIG["risk"]["trailing_stop_levels"],
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
        "max_quote_size": BOT_CONFIG["trading_strategy"]["max_quote_size"]
    },
    
    # Inventory management
    "inventory": {
        "target": BOT_CONFIG["risk"]["inventory_target"],
        "max_imbalance": BOT_CONFIG["risk"]["inventory_imbalance_limit"],
        "fade_time": BOT_CONFIG["trading_strategy"]["position_fade_time"],
        "adverse_selection_threshold": BOT_CONFIG["trading_strategy"]["adverse_selection_threshold"],
        "min_profit_rebalance": BOT_CONFIG["trading_strategy"]["min_profit_rebalance"]
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
    "spread": BOT_CONFIG["trading_strategy"]["spread"],
    "min_spread": BOT_CONFIG["trading_strategy"]["min_spread"],
    "max_spread": BOT_CONFIG["trading_strategy"]["max_spread"],
    "bid_step": BOT_CONFIG["trading_strategy"]["bid_step"],
    "ask_step": BOT_CONFIG["trading_strategy"]["ask_step"],
    "bid_sizes": BOT_CONFIG["trading_strategy"]["bid_sizes"],
    "ask_sizes": BOT_CONFIG["trading_strategy"]["ask_sizes"],
    "threshold": BOT_CONFIG["trading_strategy"]["threshold"],
    "amend_threshold": BOT_CONFIG["trading_strategy"]["amend_threshold"],
    "market_impact_threshold": BOT_CONFIG["trading_strategy"]["market_impact_threshold"],
    "quote_lifetime": BOT_CONFIG["trading_strategy"]["quote_lifetime"],
    "min_quote_interval": BOT_CONFIG["trading_strategy"]["min_quote_interval"],
    "error_retry_interval": BOT_CONFIG["connection"]["retry_delay"],
    "max_pending_operations": BOT_CONFIG["trading_strategy"]["max_pending_operations"],
    "fast_cancel_threshold": BOT_CONFIG["trading_strategy"]["fast_cancel_threshold"],
    "base_order_size": 0.01,  # Default base order size for quote calculations
    "min_order_size": 0.001,  # Minimum order size
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
    "gradual_exit_steps": TRADING_CONFIG["avellaneda"]["gradual_exit_steps"],
    "max_loss_threshold": TRADING_CONFIG["avellaneda"]["max_loss_threshold"],
    "inventory_cost_factor": TRADING_CONFIG["avellaneda"]["inventory_cost_factor"],
}

# Quoting configuration
QUOTING_CONFIG = {
    "min_quote_interval": BOT_CONFIG["trading_strategy"]["min_quote_interval"],
    "quote_lifetime": BOT_CONFIG["trading_strategy"]["quote_lifetime"],
    "order_operation_interval": BOT_CONFIG["trading_strategy"]["operation_interval"],
    "max_pending_operations": BOT_CONFIG["trading_strategy"]["max_pending_operations"],
    "fast_cancel_threshold": BOT_CONFIG["trading_strategy"]["fast_cancel_threshold"],
    "market_impact_threshold": BOT_CONFIG["trading_strategy"]["market_impact_threshold"],
    "min_quote_size": BOT_CONFIG["trading_strategy"]["min_quote_size"],
    "max_quote_size": BOT_CONFIG["trading_strategy"]["max_quote_size"],
    "size_increment": BOT_CONFIG["trading_strategy"]["size_increment"],
    "price_decimals": BOT_CONFIG["trading_strategy"]["price_decimals"],
    "size_decimals": BOT_CONFIG["trading_strategy"]["size_decimals"],
    "max_retries": BOT_CONFIG["trading_strategy"]["max_retries"],
    "quote_buffer": BOT_CONFIG["trading_strategy"]["quote_buffer"],
    "aggressive_cancel": BOT_CONFIG["trading_strategy"]["aggressive_cancel"],
    "post_only": BOT_CONFIG["trading_strategy"]["post_only"],
    "error_retry_interval": BOT_CONFIG["connection"]["retry_delay"]
}
