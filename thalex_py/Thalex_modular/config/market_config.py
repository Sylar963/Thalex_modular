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
        "underlying": "BTC-22MAY25",
        "network": Network.TEST,
        "label": "F",
    },
    
    # Trading strategy parameters
    "trading_strategy": {
        # Avellaneda-Stoikov model parameters
        "avellaneda": {
            # Core model parameters
            "gamma": 0.1,                  # Risk aversion (reduced from 0.2 to 0.1 for less aggressive trading) # Updated 2024-12-19
            "kappa": 0.5,                  # Inventory risk factor
            "time_horizon": 3600,          # Time horizon in seconds (1 hour)
            "order_flow_intensity": 2.0,   # Order flow intensity parameter
            
            # Spread management
            "base_spread": 5.0,            # Base spread in ticks
            "max_spread": 25.0,            # Maximum spread in ticks
            "spread_multiplier": 1.0,      # Dynamic spread adjustment factor
            "base_spread_factor": 1.0,     # Multiplier for the base_spread in optimal spread calculation
            "market_impact_factor": 0.5,   # Factor for market impact component in spread calculation
            "inventory_factor": 0.5,       # Factor for inventory component in spread calculation
            "volatility_multiplier": 0.2,  # Multiplier for volatility component in spread calculation
            
            # Position and inventory management
            "inventory_weight": 0.8,       # Inventory skew factor (increased from 0.7 for better skewing)
            "inventory_cost_factor": 0.0001, # Cost of holding inventory
            "position_fade_time": 300,     # Time to fade position (seconds)
            "adverse_selection_threshold": 0.002,  # Adverse selection threshold
            "min_profit_rebalance": 0.01,  # Minimum profit to trigger rebalance
            "max_loss_threshold": 0.03,    # Maximum loss before gradual exit
            
            # Quote sizing and levels
            "base_size": 0.1,             # Base quote size increased from 0.01
            "size_multipliers": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],  # Size multipliers with Fibonacci-like progression
            "max_levels": 12,              # Maximum number of quote levels (reduced from 21)
            "level_spacing": 100,          # Base spacing between levels in ticks (increased from 50 to 100)
            "vamp_spread_sensitivity": 0.5,  # Sensitivity of spread to VAMP impact
            "vamp_skew_sensitivity": 0.001,  # Sensitivity of quote skew to VAMP impact
            # Predictive adjustment thresholds
            "pa_gamma_adj_threshold": 0.05,
            "pa_kappa_adj_threshold": 0.05,
            "pa_res_price_offset_adj_threshold": 0.00005,
            "pa_volatility_adj_threshold": 0.05,
            "pa_prediction_max_age_seconds": 5,
        },
        
        # Order execution parameters
        "execution": {
            "post_only": True,             # Use post-only orders
            "min_size": 0.1,             # Minimum quote size
            "max_size": 1.0,               # Maximum quote size
            "size_increment": 0.1,       # Size increment
            "price_decimals": 2,           # Price decimal places
            "size_decimals": 1,            # Size decimal places
            "max_retries": 2,              # Maximum retry attempts
        },
        
        # Quote management
        "quote_timing": {
            "min_interval": 1.0,           # Minimum time between quotes
            "max_lifetime": 5,             # Maximum quote lifetime in seconds (reduced from 10 to 5)
            "operation_interval": 0.2,     # Time between order operations
            "max_pending": 5,              # Maximum concurrent operations
            "grid_update_interval": 3.0,   # Interval for updating quote grid based on market data
            "position_check_interval": 5.0 # Interval for position-based quote updates
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
            "max_time_seconds": 300,        # Max time for a candle toi form if volume threshold not met
            "use_exchange_data": False,     # Whether to use exchange's kline data as a source
            "fetch_interval_seconds": 60,   # Interval to fetch exchange data if use_exchange_data is True
            "lookback_hours": 1,            # Lookback hours for initial data fetch
            "prediction_update_interval": 10.0 # How often to query for new predictions
        }
    },
    
    # Orderbook specific parameters
    "orderbook": {
        "min_spread": 3,                # Minimum spread in ticks
        "base_order_size": 0.1,        # Base order size for quoting
        "min_order_size": 0.1,        # Absolute minimum order size
        "levels": 6,                    # Number of order book levels for quoting logic
        "bid_step": 10,                 # Default step for placing bid orders (in ticks)
        "ask_step": 10,                 # Default step for placing ask orders (in ticks)
        "min_size": 0.1,              # Minimum size for individual orders (used by AMM)
        "quote_lifetime": 30,           # Max lifetime for quotes in seconds
        "amend_threshold": 25,          # Price movement (in ticks) to trigger quote amendment
        "min_quote_interval": 2.0,      # Minimum interval between quote updates in seconds
        "bid_sizes": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], # Size multipliers for bid side levels
        "ask_sizes": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]  # Size multipliers for ask side levels (mirrors bid_sizes)
    },
    
    # Risk management
    "risk": {
        "max_position": 2,           # Maximum position size
        "stop_loss_pct": 0.06,         # Stop loss percentage
        "take_profit_pct": 0.0022,       # Take profit percentage (NEW - Changed to 0.22%)
        "max_drawdown": 0.10,          # Maximum drawdown
        "max_consecutive_losses": 5,   # Maximum consecutive losses
        "inventory_target": 0.0,       # Target inventory level
        "inventory_imbalance_limit": 0.7, # Maximum inventory imbalance
        # Additional risk parameters
        "position_rebalance_threshold": 0.8, # Position utilization for rebalance
        "market_impact_threshold": 0.0025, # Market impact threshold for rebalance
        "rebalance_cooldown": 300,     # Rebalance cooldown period (5 min)
        
        # Position management
        "max_notional": 1000,  # Maximum notional value in USD
        "max_position_notional": 800,  # Maximum notional position (80% of max_notional)
        "max_daily_loss_pct": 0.05,  # Maximum daily loss percentage (5%) - Added 2024-12-19
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
MARKET_CONFIG = BOT_CONFIG["market"]
CALL_IDS = BOT_CONFIG["call_ids"]
ORDERBOOK_CONFIG = BOT_CONFIG["orderbook"]

DEFAULT_VOLATILITY_CONFIG = {
    "default": 0.025,
    "window": 75,
    "min_samples": 5,
    "scaling": 1.3,
    "floor": 0.015,
    "ceiling": 0.18,
    "ewm_span": 15,
    "cache_duration": 30,
    "ewma_alpha": 0.06,  # EWMA decay factor for improved volatility calculation - Added 2024-12-19
}

# VaR (Value at Risk) Configuration - Added 2024-12-19
VAR_CONFIG = {
    "confidence_levels": [0.95, 0.99],  # Standard confidence levels
    "time_horizons": [1.0, 4.0, 24.0],  # Time horizons in hours
    "calculation_interval": 60,  # Calculate VaR every 60 seconds
    "alert_threshold_95": 500.0,  # Alert if 95% VaR exceeds $500
    "alert_threshold_99": 1000.0,  # Alert if 99% VaR exceeds $1000
    "max_var_position_ratio": 0.1,  # Maximum VaR as ratio of max position
}

TRADING_CONFIG = {
    "avellaneda": {
        "gamma": BOT_CONFIG["trading_strategy"]["avellaneda"]["gamma"],
        "kappa": BOT_CONFIG["trading_strategy"]["avellaneda"]["kappa"],
        "time_horizon": BOT_CONFIG["trading_strategy"]["avellaneda"]["time_horizon"],
        "order_flow_intensity": BOT_CONFIG["trading_strategy"]["avellaneda"]["order_flow_intensity"],
        "base_spread": BOT_CONFIG["trading_strategy"]["avellaneda"]["base_spread"],
        "max_spread": BOT_CONFIG["trading_strategy"]["avellaneda"]["max_spread"],
        "spread_multiplier": BOT_CONFIG["trading_strategy"]["avellaneda"]["spread_multiplier"],
        "base_spread_factor": BOT_CONFIG["trading_strategy"]["avellaneda"]["base_spread_factor"],
        "market_impact_factor": BOT_CONFIG["trading_strategy"]["avellaneda"]["market_impact_factor"],
        "inventory_factor": BOT_CONFIG["trading_strategy"]["avellaneda"]["inventory_factor"],
        "volatility_multiplier": BOT_CONFIG["trading_strategy"]["avellaneda"]["volatility_multiplier"],
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
        # Enhanced spread calculation parameters - Added 2024-12-19
        "collar_buffer_pct": 0.005,  # Price collar buffer percentage
        "desired_margin_rate_above_fee": 0.00025,  # Desired margin above exchange fees
        "min_spread": 3.0,  # Minimum spread in ticks
        "max_spread": 25.0,  # Maximum spread in ticks
    },
    "quoting": {
        "levels": BOT_CONFIG["trading_strategy"]["avellaneda"]["max_levels"],
        "min_interval": BOT_CONFIG["trading_strategy"]["quote_timing"]["min_interval"],
        "max_lifetime": BOT_CONFIG["trading_strategy"]["quote_timing"]["max_lifetime"],
        "operation_interval": BOT_CONFIG["trading_strategy"]["quote_timing"]["operation_interval"],
        "max_pending": BOT_CONFIG["trading_strategy"]["quote_timing"]["max_pending"],
        "grid_update_interval": BOT_CONFIG["trading_strategy"]["quote_timing"]["grid_update_interval"],
        "position_check_interval": BOT_CONFIG["trading_strategy"]["quote_timing"]["position_check_interval"],
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
    # New configurations for Phase 3 enhancements - Added 2024-12-19
    "var": VAR_CONFIG,  # Value at Risk configuration
    "market_impact": {
        "threshold": BOT_CONFIG["risk"]["market_impact_threshold"],
        "fast_cancel_threshold": 0.005,  # Fast cancel threshold for price movement
    },
}

# RISK_LIMITS - Direct access to risk parameters
RISK_LIMITS: Dict[str, Any] = {
    "max_position": BOT_CONFIG["risk"]["max_position"],
    "max_notional": BOT_CONFIG["risk"]["max_notional"],
    "max_position_notional": BOT_CONFIG["risk"]["max_position_notional"],
    "stop_loss_pct": BOT_CONFIG["risk"]["stop_loss_pct"],
    "take_profit_pct": BOT_CONFIG["risk"]["take_profit_pct"],
    "max_drawdown": BOT_CONFIG["risk"]["max_drawdown"],
    "max_consecutive_losses": BOT_CONFIG["risk"]["max_consecutive_losses"],
    "inventory_target": BOT_CONFIG["risk"]["inventory_target"],
    "inventory_imbalance_limit": BOT_CONFIG["risk"]["inventory_imbalance_limit"],
    "position_rebalance_threshold": BOT_CONFIG["risk"]["position_rebalance_threshold"],
    "market_impact_threshold": BOT_CONFIG["risk"]["market_impact_threshold"],
    "rebalance_cooldown": BOT_CONFIG["risk"]["rebalance_cooldown"],
    "max_daily_loss_pct": BOT_CONFIG["risk"]["max_daily_loss_pct"],  # Added 2024-12-19
    # Ensure all keys from BOT_CONFIG["risk"] are present if needed by RiskManager
}

CONNECTION_CONFIG = BOT_CONFIG["connection"]
