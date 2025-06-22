from thalex import Network
from typing import Dict, Any
import warnings
import logging

# =============================================================================
# THALEX MARKET MAKER CONFIGURATION
# =============================================================================
"""
Simplified configuration for the Thalex Avellaneda-Stoikov market maker.
"""

# =============================================================================
# PRIMARY CONFIGURATION - SINGLE SOURCE OF TRUTH
# =============================================================================
BOT_CONFIG = {
    # Market parameters
    "market": {
        "underlying": "BTC-PERPETUAL",
        "futures_instrument": "BTC-25JUL25",
        "network": Network.TEST,
        "label": "F",
    },
    
    # Trading strategy parameters
    "trading_strategy": {
        # Avellaneda-Stoikov model parameters
        "avellaneda": {
            # Core model parameters
            "gamma": 0.1,                           # Risk aversion parameter
            "kappa": 1.5,                           # Inventory risk factor
            "time_horizon": 1.0,                    # Time horizon for optimization
            "order_flow_intensity": 1.0,            # Order flow intensity
            
            # Spread management
            "base_spread": 14.0,            # Base spread in ticks
            "max_spread": 75.0,            # Maximum spread in ticks
            "spread_multiplier": 1.2,      # Dynamic spread adjustment factor
            "base_spread_factor": 1.0,              # Base spread multiplier
            "market_impact_factor": 0.5,            # Market impact adjustment
            "inventory_factor": 0.5,                # Inventory risk adjustment
            "volatility_multiplier": 0.2,           # Volatility adjustment factor
            
            # Position and inventory management
            "inventory_weight": 0.3,                # Inventory skew factor
            "position_fade_time": 3600,             # Time to fade position (seconds)
            
            # Quote sizing and levels
            "base_size": 0.1,            
            "size_multipliers": [1.0, 1.5, 2.0, 2.5, 3.0],
            "max_levels": 3,               # Maximum number of quote levels
            "level_spacing": 150,          # Base spacing between levels in ticks
            
            # Fixed volatility for when calculation fails
            "fixed_volatility": 0.01,
            "position_limit": 1.0,                  # Maximum position size
            "exchange_fee_rate": 0.0001,
            
            # Take profit trigger order configuration
            "enable_take_profit_triggers": True,          # Enable basic trigger orders
            "enable_arbitrage_triggers": True,            # Enable enhanced arbitrage triggers
            "take_profit_spread_bps": 7.0,               # Basic take profit spread (6-8 bps)
            "arbitrage_profit_threshold_usd": 10.0,      # Trigger arbitrage close at $10 profit
            "single_instrument_profit_threshold_usd": 5.0, # Trigger single instrument close at $5 profit
            "spread_profit_threshold_bps": 15.0,         # Trigger close at 15 bps spread profit
            "arbitrage_check_interval": 2.0,             # Check trigger conditions every 2 seconds
            
            # Predictive adjustment thresholds
            "pa_prediction_max_age_seconds": 300,        # Max age for predictions (5 minutes)
            "pa_gamma_adj_threshold": 0.05,              # Minimum gamma adjustment to apply
            "pa_kappa_adj_threshold": 0.05,              # Minimum kappa adjustment to apply  
            "pa_res_price_offset_adj_threshold": 0.00005, # Minimum reservation price offset to apply
            "pa_volatility_adj_threshold": 0.05,         # Minimum volatility adjustment to apply
        },
        
        # Order execution parameters
        "execution": {
            "post_only": True,             # Use post-only orders
            "min_size": 0.1,              # Minimum order size for BTC futures (0.1 BTC)
            "max_size": 5.0,              # Maximum quote size
            "size_increment": 0.001,      # Volume tick size for BTC futures (0.001 BTC)
            "price_decimals": 0,          # Price in whole USD (no decimals)
            "size_decimals": 3,           # Size decimals for 0.001 BTC increments
            "max_retries": 2,             # Maximum retry attempts
        },
        
        # Quote management
        "quote_timing": {
            "min_interval": 8.0,           # Minimum time between quotes
            "max_lifetime": 60,             # Maximum quote lifetime in seconds
            "operation_interval": 0.5,     # Time between order operations
            "max_pending": 6,              # Maximum concurrent operations
            "grid_update_interval": 10.0,   # Interval for updating quote grid
            "position_check_interval": 5.0 # Interval for position-based quote updates
        },
        
        # Volume Candle configuration
        "volume_candle": {
            "threshold": 1.0,               # Volume threshold for candle formation (e.g., in BTC)
            "max_candles": 100,             # Max number of candles to store
            "max_time_seconds": 300,        # Max time for a candle to form if volume threshold not met
            "use_exchange_data": False,     # Whether to use exchange's kline data as a source
            "fetch_interval_seconds": 60,   # Interval to fetch exchange data if use_exchange_data is True
            "lookback_hours": 1,            # Lookback hours for initial data fetch
        }
    },
    
    # Orderbook specific parameters
    "orderbook": {
        "min_spread": 14,                # Minimum spread in ticks
        "base_order_size": 0.02,        # Base order size for quoting
        "min_order_size": 0.01,        # Absolute minimum order size
        "levels": 3,                    # Number of order book levels for quoting logic
        "bid_step": 15,                 # Default step for placing bid orders (in ticks)
        "ask_step": 15,                 # Default step for placing ask orders (in ticks)
        "quote_lifetime": 45,           # Max lifetime for quotes in seconds
        "amend_threshold": 35,          # Price movement (in ticks) to trigger quote amendment
        "min_quote_interval": 3.0,      # Minimum interval between quote updates in seconds
        "bid_sizes": [1.0, 1.5, 2.0], # Size multipliers for bid side levels
        "ask_sizes": [1.0, 1.5, 2.0]  # Size multipliers for ask side levels
    },
    
    # Risk management
    "risk": {
        "max_position": 20,           # Maximum position size
        "stop_loss_pct": 0.06,         # Stop loss percentage

        "max_drawdown": 0.10,          # Maximum drawdown
        "max_consecutive_losses": 5,   # Maximum consecutive losses
        "inventory_target": 0.0,       # Target inventory level
        "inventory_imbalance_limit": 0.7, # Maximum inventory imbalance
        
        # Position management
        "max_notional": 100000000,  # Maximum notional value in USD
        "max_position_notional": 80000000,  # Maximum notional position (80% of max_notional)
        
        # Risk recovery system
        "recovery_enabled": True,              # Enable recovery mechanism
        "recovery_cooldown_seconds": 9,      # Cooldown after breach
        "recovery_check_interval": 30,         # Check recovery conditions every 30s
        "risk_recovery_threshold": 0.8,        # Resume at 80% of risk limits
        "gradual_recovery_steps": 1,           # Number of steps to full recovery
        

    },
    

    
    # Trading fees configuration
    "trading_fees": {
        "maker_fee_rate": 0.0002,    # 0.02% for maker orders
        "taker_fee_rate": 0.0005,    # 0.05% for taker orders  
        "minimum_fee_usd": 0.0001,   # Minimum fee per trade
        "fee_estimation_buffer": 1.1, # Safety buffer for fee estimates
        "profit_margin_rate": 0.0005, # Minimum profit margin above fees (0.05%)
        "fee_coverage_multiplier": 1.2, # Multiplier for fee coverage (20% safety margin)
        "min_profit_per_trade_pct": 0.001, # Minimum profit per trade (0.1%)
    },
    
    # Connection parameters
    "connection": {
        "rate_limit": 180,            # Maximum requests per minute
        "retry_delay": 5,            # Delay between retries in seconds
        "heartbeat_interval": 30,    # Heartbeat interval in seconds
        "max_reconnect_attempts": 3  # Maximum reconnection attempts
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
    "avellaneda": BOT_CONFIG["trading_strategy"]["avellaneda"],
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

    "trading_fees": BOT_CONFIG["trading_fees"],
}

# RISK_LIMITS - Direct access to risk parameters
RISK_LIMITS: Dict[str, Any] = BOT_CONFIG["risk"]

CONNECTION_CONFIG = BOT_CONFIG["connection"]



