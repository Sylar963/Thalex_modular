# Tuning Guide for Thalex SimpleQuoter

This guide helps you optimize the SimpleQuoter bot for different market conditions, trading styles, and risk profiles. Proper tuning of your market maker is essential for profitability and risk management.

## Performance Metrics to Monitor

Before tuning your bot, understand the key metrics to monitor:

1. **P&L (Profit and Loss)**: Total profit/loss over time
2. **Fill Ratio**: Percentage of quotes that get filled
3. **Bid-Ask Capture**: How much of the spread you capture on round-trip trades
4. **Inventory Position**: Current exposure in each instrument
5. **Spread Width**: Average spread provided to the market
6. **Order Refresh Rate**: How frequently orders are updated
7. **Latency**: Time between market changes and quote updates

## Basic Tuning Parameters

### Spread Settings

The most fundamental tuning parameters are your spread settings:

```json
"quoter": {
  "min_spread_bps": 20,
  "max_spread_bps": 200
}
```

- **Narrow spreads** (10-30 bps): Higher fill rates but lower profit per trade
- **Wide spreads** (50-300 bps): Lower fill rates but higher profit per trade

**Recommendation**: Start with wider spreads and gradually decrease until you reach your target fill rate.

### Order Size

```json
"trading": {
  "order_size": {
    "BTC-PERP": 0.01
  }
}
```

- **Small orders**: Easier to manage risk, lower impact, but smaller potential profits
- **Large orders**: Higher potential profit, but higher risk and market impact

**Recommendation**: Start with 1-5% of the average volume at your price level.

### Position Limits

```json
"risk": {
  "max_position": {
    "BTC-PERP": 0.05
  }
}
```

- Sets the maximum position in each instrument
- Should align with your capital and risk tolerance

**Recommendation**: Set to an amount you're comfortable holding if you had to stop the bot.

## Advanced Tuning

### Volatility-Responsive Parameters

If using the volatility-responsive strategy:

```json
"volatility": {
  "lookback_periods": 20,
  "vol_multiplier": 1.2,
  "min_vol_bps": 30,
  "max_vol_bps": 1000
}
```

- **lookback_periods**: Shorter periods react faster to volatility changes but may be noisier
- **vol_multiplier**: Higher values widen spreads more aggressively during volatility
- **min_vol_bps** and **max_vol_bps**: Bounds for calculated volatility

**Recommendation**: Set lookback_periods to 15-30 minutes for most instruments; adjust vol_multiplier based on your risk tolerance (1.0-2.0 range).

### Order Levels

```json
"quoter": {
  "order_levels": 3,
  "level_spacing_bps": 20,
  "size_multiplier_per_level": 1.5
}
```

- **Multiple levels**: Increases fill probability and position management flexibility
- **Level spacing**: Tighter spacing increases fill probability but at lower profit
- **Size multiplier**: Increasing size at further levels can improve inventory management

**Recommendation**: Start with 1-2 levels until comfortable, then expand to 3-5 levels with size_multiplier around 1.5-2.0.

### Inventory Management

```json
"advanced": {
  "inventory_management": true,
  "inventory_target_percentage": 0,
  "inventory_skew_coefficient": 0.5
}
```

- **inventory_target_percentage**: 0 means neutral; positive favors long position
- **inventory_skew_coefficient**: Higher values adjust quotes more aggressively as inventory deviates from target

**Recommendation**: Start with coefficient at 0.3-0.5; increase if inventory is not being managed effectively.

## Market-Specific Tuning

### High Volatility Markets (e.g., during major news events)

```json
"quoter": {
  "strategy": "volatility_responsive",
  "min_spread_bps": 50,
  "max_spread_bps": 500,
  "order_levels": 2
},
"volatility": {
  "vol_multiplier": 1.5,
  "lookback_periods": 10
},
"trading": {
  "order_size": {
    "BTC-PERP": 0.005  // 50% of normal size
  }
},
"risk": {
  "max_position": {
    "BTC-PERP": 0.02  // 40% of normal position limit
  }
}
```

### Low Volatility Markets (e.g., range-bound periods)

```json
"quoter": {
  "strategy": "basic",
  "min_spread_bps": 10,
  "max_spread_bps": 100,
  "order_levels": 3,
  "level_spacing_bps": 10
},
"trading": {
  "order_size": {
    "BTC-PERP": 0.02  // 200% of normal size
  }
}
```

### Illiquid Markets

```json
"quoter": {
  "min_spread_bps": 40,
  "max_spread_bps": 300,
  "order_levels": 1
},
"trading": {
  "order_size": {
    "BTC-PERP": 0.005  // Smaller size
  },
  "cancel_threshold_bps": 20,
  "post_only": true
}
```

## Risk Profile Tuning

### Conservative Profile

For traders who prioritize capital preservation:

```json
"quoter": {
  "min_spread_bps": 40,
  "max_spread_bps": 300,
  "order_levels": 2
},
"trading": {
  "order_size": {
    "BTC-PERP": 0.005
  },
  "post_only": true
},
"risk": {
  "max_position": {
    "BTC-PERP": 0.02
  },
  "max_notional_position_usd": 5000,
  "max_daily_loss_usd": 500
}
```

### Balanced Profile

For traders seeking a middle ground:

```json
"quoter": {
  "min_spread_bps": 20,
  "max_spread_bps": 200,
  "order_levels": 3
},
"trading": {
  "order_size": {
    "BTC-PERP": 0.01
  }
},
"risk": {
  "max_position": {
    "BTC-PERP": 0.05
  },
  "max_notional_position_usd": 10000,
  "max_daily_loss_usd": 1000
}
```

### Aggressive Profile

For traders looking to maximize opportunities:

```json
"quoter": {
  "min_spread_bps": 10,
  "max_spread_bps": 150,
  "order_levels": 5,
  "level_spacing_bps": 15
},
"trading": {
  "order_size": {
    "BTC-PERP": 0.02
  }
},
"risk": {
  "max_position": {
    "BTC-PERP": 0.1
  },
  "max_notional_position_usd": 20000,
  "max_daily_loss_usd": 2000
}
```

## Tuning Process

Follow this iterative process to optimize your configuration:

1. **Start Conservative**: Begin with wider spreads, smaller sizes, and stricter limits
2. **Monitor for 24-48 Hours**: Collect performance data without making changes
3. **Analyze Key Metrics**: Review P&L, fill rates, and inventory management
4. **Adjust One Parameter**: Change only one parameter at a time
5. **Monitor for Impact**: Run for at least 24 hours to assess the change
6. **Document Results**: Keep a record of changes and corresponding performance
7. **Repeat**: Continue the process until performance meets your goals

## A/B Testing Configuration Changes

For scientific optimization, consider A/B testing:

1. Run the bot with Configuration A for a week
2. Run the bot with Configuration B (one parameter changed) for a week
3. Compare performance metrics
4. Keep the better configuration and test the next parameter change

## Common Tuning Scenarios

### Problem: Orders rarely getting filled

**Solution**:
- Decrease `min_spread_bps` and `max_spread_bps`
- Increase `order_levels`
- Decrease `level_spacing_bps`

### Problem: Inventory building up too much on one side

**Solution**:
- Increase `inventory_skew_coefficient`
- Ensure `inventory_management` is set to true
- Consider setting `inventory_target_percentage` to 0

### Problem: P&L is negative despite good fill rates

**Solution**:
- Increase `min_spread_bps`
- Consider using `volatility_responsive` strategy
- Reduce `order_size` to minimize adverse selection

### Problem: High volatility causing large losses

**Solution**:
- Increase `vol_multiplier` to widen spreads during volatility
- Reduce `max_position` limits
- Consider reducing `order_levels` during volatile periods

## Conclusion

Successful market making requires continuous tuning and adaptation. Start conservatively, make data-driven adjustments, and prioritize risk management. Remember that market conditions change, so regular review and adjustment of your configuration is essential.

For any questions or assistance with tuning, refer to Thalex support or community resources. 