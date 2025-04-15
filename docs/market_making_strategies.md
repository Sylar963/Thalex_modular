# Market Making Strategies

This guide explains different market making strategies that can be implemented with the Thalex SimpleQuoter.

## Basic Market Making Concepts

### Spread and Levels

The SimpleQuoter places orders at multiple price levels around the mid-price:

- **Spread**: The difference between bid and ask prices
- **Levels**: Number of price points at which orders are placed on each side of the book
- **Min/Max Spread**: Limits for how tight or wide your quotes can be

### Inventory Management

The SimpleQuoter uses inventory management to maintain a balanced position:

- **Inventory Risk Aversion**: Controls how aggressively the bot adjusts prices based on inventory
- **Max Position Size**: Limits the maximum position the bot can hold

## Strategy Components

### 1. Volatility-Based Spreads

The bot can adjust spreads based on market volatility:

```
If market_volatility > vol_threshold:
    spread = max_spread
Else:
    spread = min_spread + (max_spread - min_spread) * (market_volatility / vol_threshold)
```

#### Implementation

```json
"quoter": {
  "min_spread_bps": 10,
  "max_spread_bps": 100,
  "vol_threshold": 0.05
}
```

### 2. Inventory-Based Skew

The bot skews prices based on current inventory to balance positions:

```
If inventory > 0:  // Long position
    bid_price -= skew_adjustment
    ask_price -= skew_adjustment
If inventory < 0:  // Short position
    bid_price += skew_adjustment
    ask_price += skew_adjustment
```

Where `skew_adjustment` increases with inventory size and `inventory_risk_aversion`.

#### Implementation

```json
"trading": {
  "max_position_size": {
    "BTC-PERP": 1.0
  }
},
"quoter": {
  "inventory_risk_aversion": 0.9
}
```

### 3. Multi-Level Order Placement

The bot places orders at multiple price levels:

```
For i = 1 to levels:
    bid_price[i] = mid_price * (1 - base_spread - (i-1) * level_spacing)
    ask_price[i] = mid_price * (1 + base_spread + (i-1) * level_spacing)
```

#### Implementation

```json
"quoter": {
  "levels": 10
}
```

## Common Strategies

### 1. Tight Spread Market Making

Focus on high volume with tight spreads.

```json
"quoter": {
  "levels": 5,
  "min_spread_bps": 5,
  "max_spread_bps": 50,
  "vol_threshold": 0.03,
  "inventory_risk_aversion": 0.8,
  "order_size_factor": 1.2
}
```

Best for:
- Highly liquid markets
- Low volatility conditions
- High trading volume

### 2. Wide Spread Market Making

Focus on capturing larger spreads with less frequent fills.

```json
"quoter": {
  "levels": 8,
  "min_spread_bps": 30,
  "max_spread_bps": 200,
  "vol_threshold": 0.05,
  "inventory_risk_aversion": 0.9,
  "order_size_factor": 1.0
}
```

Best for:
- Less liquid markets
- Higher volatility conditions
- Prioritizing profit per trade over volume

### 3. Volatility-Responsive Market Making

Dynamically adapts to changing market conditions.

```json
"quoter": {
  "levels": 12,
  "min_spread_bps": 10,
  "max_spread_bps": 250,
  "vol_threshold": 0.07,
  "inventory_risk_aversion": 0.85,
  "order_size_factor": 0.9
}
```

Best for:
- Markets with changing volatility
- 24/7 operation across different market conditions

### 4. Conservative Market Making

Focus on risk management with controlled inventory.

```json
"quoter": {
  "levels": 6,
  "min_spread_bps": 15,
  "max_spread_bps": 150,
  "vol_threshold": 0.04,
  "inventory_risk_aversion": 0.95,
  "order_size_factor": 0.7,
  "order_size_limit": 0.5
}
```

Best for:
- Risk-averse operators
- Markets with potential for sharp moves
- Lower capital allocation

## Performance Monitoring and Tuning

### Key Metrics to Monitor

1. **Profit and Loss (PnL)**
   - Total PnL
   - PnL per trade
   - PnL by time of day

2. **Execution Metrics**
   - Fill rate (orders filled / orders placed)
   - Time to fill
   - Spread capture percentage

3. **Risk Metrics**
   - Maximum drawdown
   - Inventory turnover
   - Position holding time

### Tuning Process

1. Start with conservative settings
2. Monitor performance for at least 1-2 days
3. Adjust one parameter at a time
4. Compare performance metrics before/after changes
5. Gradually optimize for your specific market conditions

## Strategy Development Tips

1. Backtest strategies when possible
2. Start in testnet mode before trading real funds
3. Implement circuit breakers for extreme market conditions
4. Consider market-specific factors (contract type, liquidity, volatility)
5. Regularly review and adjust your strategy as market conditions change 