from thalex import Network
from typing import Dict, Any
import warnings
import logging

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
        # Maintain backward compatibility
        "underlying": "BTC-25JUL25",
        #"underlying2": "BTC-20JUN25",
        "network": Network.TEST,
        "label": "F",
    
    },
    
    # Trading strategy parameters
    "trading_strategy": {
        # Avellaneda-Stoikov model parameters
        "avellaneda": {
            # Core model parameters
            "gamma": 0.4,                  # Risk aversion (reduced from 0.3 for more frequent executions)
            "kappa": 0.5,                  # Inventory risk factor
            "time_horizon": 3600,          # Time horizon in seconds (1 hour)
            "order_flow_intensity": 2.0,   # Order flow intensity parameter
            
            # Spread management
            "base_spread": 14.0,            # INCREASED: Base spread in ticks - must cover fees
            "max_spread": 75.0,            # INCREASED: Maximum spread in ticks
            "spread_multiplier": 1.2,      # INCREASED: Dynamic spread adjustment factor
            "base_spread_factor": 1.5,     # INCREASED: Multiplier for the base_spread in optimal spread calculation
            "market_impact_factor": 0.7,   # INCREASED: Factor for market impact component in spread calculation
            "inventory_factor": 0.7,       # INCREASED: Factor for inventory component in spread calculation
            "volatility_multiplier": 0.4,  # INCREASED: Multiplier for volatility component in spread calculation
            
            # Position and inventory management
            "inventory_weight": 0.8,       # Inventory skew factor (increased from 0.7 for better skewing)
            "inventory_cost_factor": 0.0001, # Cost of holding inventory
            "position_fade_time": 300,     # INCREASED: Time to fade position (seconds) - was 30
            "adverse_selection_threshold": 0.002,  # Adverse selection threshold
            "min_profit_rebalance": 0.01,  # Minimum profit to trigger rebalance
            "max_loss_threshold": 0.03,    # Maximum loss before gradual exit
            
            # Quote sizing and levels - REDUCED for profitability
            "base_size": 0.1,            
            "size_multipliers": [1.0, 1.5, 2.0, 2.5, 3.0],  # REDUCED: Less aggressive size progression
            "max_levels": 3,               # REDUCED: Maximum number of quote levels - was 6
            "level_spacing": 150,          # INCREASED: Base spacing between levels in ticks - was 100
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
            "min_size": 0.1,              # CORRECTED: Minimum order size for BTC futures (0.1 BTC)
            "max_size": 5.0,              # Maximum quote size
            "size_increment": 0.001,      # CORRECTED: Volume tick size for BTC futures (0.001 BTC)
            "price_decimals": 0,          # CORRECTED: Price in whole USD (no decimals)
            "size_decimals": 3,           # CORRECTED: Size decimals for 0.001 BTC increments
            "max_retries": 2,             # Maximum retry attempts
        },
        
        # Quote management
        "quote_timing": {
            "min_interval": 8.0,           # INCREASED: Minimum time between quotes - was 5.0
            "max_lifetime": 60,             # INCREASED: Maximum quote lifetime in seconds - was 10
            "operation_interval": 0.5,     # INCREASED: Time between order operations - was 0.2
            "max_pending": 6,              # REDUCED: Maximum concurrent operations - was 9
            "grid_update_interval": 10.0,   # INCREASED: Interval for updating quote grid - was 5.0
            "position_check_interval": 5.0 # INCREASED: Interval for position-based quote updates - was 2.0
        },
        
        # Market impact and cancellation
        "market_impact": {
            "threshold": 0.1,             # Market impact threshold
            "fast_cancel_threshold": 0.005, # Price movement for fast cancellation
            "aggressive_cancel": True,     # Whether to cancel aggressively
        },
        
        # VAMP (Volume Adjusted Market Pressure)
        "vamp": {
            "window": 50,                   # Number of price-volume samples
            "aggressive_window": 20,        # Number of aggressive trade samples
            "impact_window": 30            # Number of market impact samples
        },

        # Volume Candle configuration for predictive analysis (FULLY IMPLEMENTED)
        # This system now properly integrates volume-based candle formation with quote generation
        # Volume signals (momentum, reversal, volatility, exhaustion) enhance spread and size calculations
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
        "min_spread": 14,                # INCREASED: Minimum spread in ticks - CRITICAL for profitability
        "base_order_size": 0.02,        # REDUCED: Base order size for quoting - was 0.05
        "min_order_size": 0.01,        # REDUCED: Absolute minimum order size - was 0.02
        "levels": 3,                    # REDUCED: Number of order book levels for quoting logic - was 6
        "bid_step": 15,                 # INCREASED: Default step for placing bid orders (in ticks) - was 10
        "ask_step": 15,                 # INCREASED: Default step for placing ask orders (in ticks) - was 10
        "min_size": 0.05,              # REDUCED: Minimum size for individual orders (used by AMM) - was 0.1
        "quote_lifetime": 45,           # INCREASED: Max lifetime for quotes in seconds - was 30
        "amend_threshold": 35,          # INCREASED: Price movement (in ticks) to trigger quote amendment - was 25
        "min_quote_interval": 3.0,      # INCREASED: Minimum interval between quote updates in seconds - was 2.0
        "bid_sizes": [1.0, 1.5, 2.0], # REDUCED: Size multipliers for bid side levels
        "ask_sizes": [1.0, 1.5, 2.0]  # REDUCED: Size multipliers for ask side levels
    },
    
    # Risk management
    "risk": {
        "max_position": 20,           # Maximum position size
        "stop_loss_pct": 0.06,         # Stop loss percentage
        "take_profit_pct": 0.0022,       # Take profit percentage (NEW - Changed to 0.22%)
        "max_drawdown": 0.10,          # Maximum drawdown
        "max_consecutive_losses": 5,   # Maximum consecutive losses
        "inventory_target": 0.0,       # Target inventory level
        "inventory_imbalance_limit": 0.7, # Maximum inventory imbalance
        # Additional risk parameters
        "position_rebalance_threshold": 0.8, # Position utilization for rebalance
        "market_impact_threshold": 0.0025, # Market impact threshold for rebalance
        "rebalance_cooldown": 30,     # Rebalance cooldown period (5 min)
        
        # Position management
        "max_notional": 100000000,  # Maximum notional value in USD
        "max_position_notional": 80000000,  # Maximum notional position (80% of max_notional)
        
        # Risk recovery system
        "recovery_enabled": True,              # Enable recovery mechanism
        "recovery_cooldown_seconds": 9,      # 5 minutes cooldown after breach
        "recovery_check_interval": 30,         # Check recovery conditions every 30s
        "risk_recovery_threshold": 0.8,        # Resume at 80% of risk limits
        "gradual_recovery_steps": 3,           # Number of steps to full recovery
        
        # UPNL Take Profit Configuration
        "take_profit_enabled": True,           # Enable UPNL-based take profit
        "take_profit_threshold": 5.0,        # INCREASED: Take profit at $5 UPNL - was 2.0
        "take_profit_check_interval": 2,       # INCREASED: Check every 2 seconds - was 1
        "flatten_position_enabled": True,       # Allow position flattening
        "take_profit_cooldown": 3,            # INCREASED: 3 second cooldown after take profit - was 1
    },
    
    # Portfolio-wide take profit configuration
    "portfolio_take_profit": {
        "enable_portfolio_tp": True,         # Master enable/disable
        "min_profit_usd": 3.0,              # INCREASED: Minimum profit threshold in USD - was 1.1
        "profit_after_fees": True,          # Whether threshold applies after fees
        "check_interval_seconds": 3.0,       # INCREASED: How often to check portfolio P&L - was 2.0
        "max_position_age_hours": 24,        # Maximum time to hold positions
        "emergency_close_threshold": -15.0,  # INCREASED: Emergency close if loss exceeds this - was -10.0
        "partial_profit_threshold": 1.0,     # INCREASED: Take partial profits at this level - was 0.5
        "position_correlation_check": True,  # Monitor position correlation
        "fee_estimation_buffer": 1.2        # INCREASED: Multiply estimated fees by this factor - was 1.1
    },
    
    # Trading fees configuration
    "trading_fees": {
        "maker_fee_rate": 0.0002,    # 0.02% for maker orders
        "taker_fee_rate": 0.0005,    # 0.05% for taker orders  
        "minimum_fee_usd": 0.0001,   # Minimum fee per trade
        "fee_estimation_buffer": 1.1, # Safety buffer for fee estimates
        "profit_margin_rate": 0.0005, # ADDED: Minimum profit margin above fees (0.05%)
        "fee_coverage_multiplier": 1.2, # ADDED: Multiplier for fee coverage (20% safety margin)
        "min_profit_per_trade_pct": 0.001, # ADDED: Minimum profit per trade (0.1%)
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
    "cache_duration": 30
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
    "portfolio_take_profit": BOT_CONFIG["portfolio_take_profit"],
    "trading_fees": BOT_CONFIG["trading_fees"],
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
    # Risk recovery parameters
    "recovery_enabled": BOT_CONFIG["risk"]["recovery_enabled"],
    "recovery_cooldown_seconds": BOT_CONFIG["risk"]["recovery_cooldown_seconds"],
    "recovery_check_interval": BOT_CONFIG["risk"]["recovery_check_interval"],
    "risk_recovery_threshold": BOT_CONFIG["risk"]["risk_recovery_threshold"],
    "gradual_recovery_steps": BOT_CONFIG["risk"]["gradual_recovery_steps"],
    # UPNL Take Profit parameters
    "take_profit_enabled": BOT_CONFIG["risk"]["take_profit_enabled"],
    "take_profit_threshold": BOT_CONFIG["risk"]["take_profit_threshold"],
    "take_profit_check_interval": BOT_CONFIG["risk"]["take_profit_check_interval"],
    "flatten_position_enabled": BOT_CONFIG["risk"]["flatten_position_enabled"],
    "take_profit_cooldown": BOT_CONFIG["risk"]["take_profit_cooldown"],
    # Ensure all keys from BOT_CONFIG["risk"] are present if needed by RiskManager
}

CONNECTION_CONFIG = BOT_CONFIG["connection"]


def validate_portfolio_take_profit_config(config: Dict) -> bool:
    """Validate portfolio take profit configuration"""
    required_fields = ["min_profit_usd", "check_interval_seconds"]
    
    for field in required_fields:
        if field not in config:
            logging.error(f"Missing required portfolio take profit field: {field}")
            return False
    
    if config["min_profit_usd"] <= 0:
        logging.error("min_profit_usd must be positive")
        return False
        
    if config["check_interval_seconds"] < 1.0:
        logging.error("check_interval_seconds must be at least 1.0")
        return False
    
    return True
