from thalex import Network

MARKET_CONFIG = {
    "underlying": "BTCUSD",
    "network": Network.TEST,
    "label": "P",
}

ORDERBOOK_CONFIG = {
    "spread": 0.5,          # Base spread in ticks
    "amend_threshold": 25,  # Minimum price change to amend orders
    "bid_step": 25,        # Price step between bid levels
    "ask_step": 25,        # Price step between ask levels
    "bid_sizes": [0.2, 0.8],  # Size for each bid level
    "ask_sizes": [0.2, 0.8],  # Size for each ask level
    "min_spread": 0.3,     # Minimum spread in tick size
    "max_spread": 2.0,     # Maximum spread in tick size
    "market_impact_threshold": 0.01,  # 1% price impact threshold
    "quote_lifetime": 30,   # Maximum quote lifetime in seconds
    "min_quote_interval": 0.5,  # Minimum time between quotes
    "error_retry_interval": 1.0,  # Time between error retries
    "max_pending_operations": 5,  # Maximum concurrent order operations
    "fast_cancel_threshold": 0.005,  # Price movement for fast cancellation
    "threshold": 1.0,      # Maximum size per quote
}

RISK_LIMITS = {
    "max_position": 1.0,     # Maximum position size
    "max_notional": 50000,   # Maximum notional exposure (USD)
    "stop_loss_pct": 0.06,   # Stop loss percentage
    "base_take_profit_pct": 0.02,  # Base take profit target
    "max_take_profit_pct": 0.05,   # Maximum take profit
    "min_take_profit_pct": 0.01,   # Minimum take profit
    "rebalance_threshold": 0.8,  # Rebalance trigger
    "take_profit_pct": 0.03,  # Default take profit percentage
    "trailing_stop_activation": 0.015,  # Activate trailing stop at 1.5% profit
    "trailing_stop_distance": 0.01,    # Trailing stop follows at 1% distance
    "take_profit_levels": [
        {"percentage": 0.01, "size": 0.2},  # Take 20% profit at 1%
        {"percentage": 0.02, "size": 0.3},  # Take 30% profit at 2%
        {"percentage": 0.03, "size": 0.3},  # Take 30% profit at 3%
        {"percentage": 0.05, "size": 0.2},  # Take remaining 20% at 5%
    ],
    "trailing_stop_levels": [
        {"activation": 0.015, "distance": 0.01},  # First trailing stop
        {"activation": 0.03, "distance": 0.015},  # Second trailing stop
        {"activation": 0.05, "distance": 0.02},   # Final trailing stop
    ],
    "position_limit": 50000,  # Notional position limit in USD
}

TRADING_PARAMS = {
    "trailing_stop": {
        "activation": 0.015,  # Activate at 1.5% profit
        "distance": 0.01,     # 1% trailing distance
    },
    "position_management": {
        "gamma": 0.1,               # Risk aversion (Avellaneda-Stoikov)
        "inventory_weight": 0.5,    # Inventory skew factor
        "position_fade_time": 300,  # Time to fade position (seconds)
        "inventory_cost_factor": 0.0001,  # Cost of holding inventory
        "max_inventory_imbalance": 0.7,  # Maximum inventory imbalance
        "adverse_selection_threshold": 0.002,  # Price move threshold
        "position_limit": 50000,    # Position limit in USD
        "order_flow_intensity": 1.5  # Order flow intensity parameter for A-S model
    },
    "volatility": {
        "window": 100,        # Volatility calculation window
        "min_samples": 20,    # Minimum samples for vol calc
        "scaling": 1.0,       # Volatility scaling factor
        "vol_floor": 0.001,   # Minimum volatility
        "vol_ceiling": 5.0,   # Maximum volatility
        "ewm_span": 20,       # Span for exponential weighting
        "cache_duration": 60,  # Cache duration in seconds
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
        "threshold": 0.005  # Add missing threshold for volatility detection
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
        "short_period": 10,
        "long_period": 30,
        "confirmation_threshold": 0.6
    }
}

CALL_IDS = {
    "instruments": 0,
    "instrument": 1,
    "subscribe": 2,
    "login": 3,
    "cancel_session": 4,
    "set_cod": 5
}

PERFORMANCE_METRICS = {
    "window_size": 100,        # Window for calculating metrics
    "min_trades": 10,         # Minimum trades for metrics calculation
    "target_profit": 0.02,    # Target profit percentage
    "max_drawdown": 0.05,     # Maximum allowable drawdown
    "max_position_time": 3600,  # Maximum position holding time in seconds
    "metrics_update_interval": 60,  # Update interval in seconds
    "profit_factor_threshold": 1.5,  # Minimum profit factor
    "win_rate_threshold": 0.55,  # Minimum win rate
    "risk_reward_ratio": 1.5,  # Target risk/reward ratio
    "max_consecutive_losses": 5,  # Maximum consecutive losses
    "drawdown_recovery_factor": 2.0  # Required profit factor after drawdown
}
