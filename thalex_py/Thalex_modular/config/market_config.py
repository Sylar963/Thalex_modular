from thalex import Network

# Main configuration for the bot
BOT_CONFIG = {
    # Market parameters
    "market": {
        "underlying": "BTCUSD",
        "network": Network.TEST,
        "label": "P",
    },
    
    # Order parameters
    "order": {
        "spread": 2.0,          # Base spread in ticks (increased from 0.5)
        "min_spread": 1.5,      # Minimum spread in tick size (increased from 0.3)
        "max_spread": 5.0,      # Maximum spread in tick size (increased from 2.0)
        "bid_step": 25,         # Price step between bid levels
        "ask_step": 25,         # Price step between ask levels
        "bid_sizes": [0.2, 0.8], # Size for each bid level
        "ask_sizes": [0.2, 0.8], # Size for each ask level
        "threshold": 0.5,       # Maximum size per quote
        "amend_threshold": 5,   # Minimum price change to amend orders
        "post_only": True       # Use post-only orders
    },
    
    # Quote management
    "quoting": {
        "min_quote_interval": 2.0,     # Minimum time between quotes
        "quote_lifetime": 9,           # Maximum quote lifetime in seconds
        "order_operation_interval": 0.1, # Time between order operations
        "max_pending_operations": 5,   # Maximum concurrent operations
        "fast_cancel_threshold": 0.005, # Price movement for fast cancellation
        "market_impact_threshold": 0.01, # Market impact threshold
        "min_quote_size": 0.001,       # Minimum quote size
        "max_quote_size": 1.0,         # Maximum quote size 
    },
    
    # Risk management
    "risk": {
        "max_position": 1.0,           # Maximum position size
        "max_notional": 50000,         # Maximum notional exposure (USD)
        "stop_loss_pct": 0.06,         # Stop loss percentage
        "take_profit_pct": 0.03,       # Take profit percentage
        "max_daily_loss": 0.05,        # Maximum daily loss
        "max_drawdown": 0.10,          # Maximum drawdown
        "max_consecutive_losses": 5,   # Maximum consecutive losses
        "inventory_target": 0.0,       # Target inventory level
        "inventory_imbalance_limit": 0.7, # Maximum inventory imbalance
    },
    
    # Technical analysis parameters
    "technical": {
        "volatility": {
            "window": 100,             # Volatility calculation window
            "min_samples": 20,         # Minimum samples for calculation
            "scaling": 1.0,            # Volatility scaling factor
            "floor": 0.001,            # Minimum volatility
            "ceiling": 5.0,            # Maximum volatility
        },
        "trend": {
            "short_period": 10,        # Short-term trend period
            "long_period": 30,         # Long-term trend period
            "confirmation_threshold": 0.6, # Trend confirmation threshold
        },
    },
    
    # Avellaneda-Stoikov model parameters
    "avellaneda": {
        "gamma": 0.2,                 # Risk aversion (increased from 0.1)
        "inventory_weight": 0.7,      # Inventory skew factor (increased from 0.5)
        "position_fade_time": 300,    # Time to fade position (seconds)
        "order_flow_intensity": 1.5   # Order flow intensity parameter
    },
    
    # Connection parameters
    "connection": {
        "max_retries": 5,             # Maximum connection retry attempts
        "retry_delay": 1,             # Initial retry delay
        "heartbeat_interval": 30,     # Heartbeat interval in seconds
        "rate_limit": 300,            # Maximum requests per minute
    },
    
    # System call IDs
    "call_ids": {
        "instruments": 0,
        "instrument": 1,
        "subscribe": 2,
        "login": 3,
        "cancel_session": 4,
        "set_cod": 5
    }
}

# For backward compatibility
MARKET_CONFIG = BOT_CONFIG["market"]
ORDERBOOK_CONFIG = {
    "spread": BOT_CONFIG["order"]["spread"],
    "min_spread": BOT_CONFIG["order"]["min_spread"],
    "max_spread": BOT_CONFIG["order"]["max_spread"],
    "bid_step": BOT_CONFIG["order"]["bid_step"],
    "ask_step": BOT_CONFIG["order"]["ask_step"],
    "bid_sizes": BOT_CONFIG["order"]["bid_sizes"],
    "ask_sizes": BOT_CONFIG["order"]["ask_sizes"],
    "threshold": BOT_CONFIG["order"]["threshold"],
    "amend_threshold": BOT_CONFIG["order"]["amend_threshold"],
    "market_impact_threshold": BOT_CONFIG["quoting"]["market_impact_threshold"],
    "quote_lifetime": BOT_CONFIG["quoting"]["quote_lifetime"],
    "min_quote_interval": BOT_CONFIG["quoting"]["min_quote_interval"],
    "error_retry_interval": BOT_CONFIG["connection"]["retry_delay"],
    "max_pending_operations": BOT_CONFIG["quoting"]["max_pending_operations"],
    "fast_cancel_threshold": BOT_CONFIG["quoting"]["fast_cancel_threshold"],
}

RISK_LIMITS = {
    "max_position": BOT_CONFIG["risk"]["max_position"],
    "max_notional": BOT_CONFIG["risk"]["max_notional"],
    "stop_loss_pct": BOT_CONFIG["risk"]["stop_loss_pct"],
    "take_profit_pct": BOT_CONFIG["risk"]["take_profit_pct"],
    "max_daily_loss": BOT_CONFIG["risk"]["max_daily_loss"],
    "max_drawdown": BOT_CONFIG["risk"]["max_drawdown"],
    "max_consecutive_losses": BOT_CONFIG["risk"]["max_consecutive_losses"],
    # Legacy parameters maintained for compatibility
    "base_take_profit_pct": 0.02,
    "max_take_profit_pct": 0.05,
    "min_take_profit_pct": 0.01,
    "rebalance_threshold": 0.8,
    "trailing_stop_activation": 0.015,
    "trailing_stop_distance": 0.01,
    "take_profit_levels": [
        {"percentage": 0.01, "size": 0.2},
        {"percentage": 0.02, "size": 0.3},
        {"percentage": 0.03, "size": 0.3},
        {"percentage": 0.05, "size": 0.2},
    ],
    "trailing_stop_levels": [
        {"activation": 0.015, "distance": 0.01},
        {"activation": 0.03, "distance": 0.015},
        {"activation": 0.05, "distance": 0.02},
    ],
}

TRADING_PARAMS = {
    "position_management": {
        "gamma": BOT_CONFIG["avellaneda"]["gamma"],
        "inventory_weight": BOT_CONFIG["avellaneda"]["inventory_weight"],
        "position_fade_time": BOT_CONFIG["avellaneda"]["position_fade_time"],
        "order_flow_intensity": BOT_CONFIG["avellaneda"]["order_flow_intensity"],
        "inventory_cost_factor": 0.0001,
        "max_inventory_imbalance": BOT_CONFIG["risk"]["inventory_imbalance_limit"],
        "adverse_selection_threshold": 0.002,
        "position_limit": BOT_CONFIG["risk"]["max_notional"],
    },
    "volatility": {
        "window": BOT_CONFIG["technical"]["volatility"]["window"],
        "min_samples": BOT_CONFIG["technical"]["volatility"]["min_samples"],
        "scaling": BOT_CONFIG["technical"]["volatility"]["scaling"],
        "vol_floor": BOT_CONFIG["technical"]["volatility"]["floor"],
        "vol_ceiling": BOT_CONFIG["technical"]["volatility"]["ceiling"],
        "ewm_span": 20,
        "cache_duration": 60,
    }
}

TECHNICAL_PARAMS = {
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
    "momentum": {
        "period": 10,
        "threshold": 0.02
    },
    "volume": {
        "ma_period": 20,
        "threshold": 1.5
    },
    "trend": {
        "short_period": BOT_CONFIG["technical"]["trend"]["short_period"],
        "long_period": BOT_CONFIG["technical"]["trend"]["long_period"],
        "confirmation_threshold": BOT_CONFIG["technical"]["trend"]["confirmation_threshold"]
    }
}

CALL_IDS = BOT_CONFIG["call_ids"]

# Simplified performance metrics
PERFORMANCE_METRICS = {
    "window_size": 100,
    "min_trades": 10,
    "target_profit": BOT_CONFIG["risk"]["take_profit_pct"],
    "max_drawdown": BOT_CONFIG["risk"]["max_drawdown"],
    "max_position_time": 3600,
    "metrics_update_interval": 60,
    "profit_factor_threshold": 1.5,
    "win_rate_threshold": 0.55,
    "risk_reward_ratio": 1.5,
    "max_consecutive_losses": BOT_CONFIG["risk"]["max_consecutive_losses"],
    "drawdown_recovery_factor": 2.0
}

# Simplified inventory config
INVENTORY_CONFIG = {
    "max_inventory_imbalance": BOT_CONFIG["risk"]["inventory_imbalance_limit"],
    "target_inventory": BOT_CONFIG["risk"]["inventory_target"],
    "inventory_fade_time": BOT_CONFIG["avellaneda"]["position_fade_time"],
    "adverse_selection_threshold": 0.002,
    "inventory_skew_factor": BOT_CONFIG["avellaneda"]["inventory_weight"] / 2,
    "max_position_notional": BOT_CONFIG["risk"]["max_notional"] * 0.8,
    "min_profit_rebalance": 0.01,
    "gradual_exit_steps": 4,
    "max_loss_threshold": 0.03,
    "inventory_cost_factor": 0.0001,
}

# Simplified quoting config
QUOTING_CONFIG = {
    "order_operation_interval": BOT_CONFIG["quoting"]["order_operation_interval"],
    "quote_refresh_interval": BOT_CONFIG["quoting"]["min_quote_interval"] / 2,
    "max_quote_age": BOT_CONFIG["quoting"]["quote_lifetime"] * 3,
    "min_quote_size": BOT_CONFIG["quoting"]["min_quote_size"],
    "max_quote_size": BOT_CONFIG["quoting"]["max_quote_size"],
    "size_increment": 0.001,
    "price_decimals": 2,
    "size_decimals": 3,
    "max_retries": 3,
    "retry_delay": BOT_CONFIG["connection"]["retry_delay"],
    "quote_buffer": 5,
    "aggressive_cancel": True,
    "post_only": BOT_CONFIG["order"]["post_only"]
}
