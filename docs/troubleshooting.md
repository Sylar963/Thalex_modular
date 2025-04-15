# Troubleshooting Guide

This guide helps you identify and resolve common issues with the Thalex SimpleQuoter.

## Connectivity Issues

### Exchange API Connection Failures

**Symptoms:**
- Error messages related to API connectivity
- Order placement failures
- Missing market data

**Solutions:**
1. Check API key permissions
   ```
   Error: "Permission denied" or "Invalid API key"
   ```
   Ensure your API key has proper trading and reading permissions.

2. Verify network connectivity
   ```
   Error: "Connection timeout" or "Network error"
   ```
   Check your internet connection and firewall settings. Ensure the bot has outbound access.

3. API rate limiting
   ```
   Error: "Rate limit exceeded"
   ```
   Reduce the frequency of API calls in your configuration or implement backoff strategies.

## Order Placement Issues

### Orders Not Being Placed

**Symptoms:**
- Bot is running but no orders appear on exchange
- Log shows order creation attempts but no confirmations

**Solutions:**
1. Check minimum order size
   ```
   Error: "Order size below minimum"
   ```
   Increase your order size or check exchange requirements for minimum order sizes.

2. Insufficient funds
   ```
   Error: "Insufficient balance"
   ```
   Ensure your account has sufficient funds for the positions you're trying to open.

3. Market access issues
   ```
   Error: "Market currently closed" or "Trading suspended"
   ```
   Verify the instrument is available for trading on the exchange.

### Orders Getting Instantly Filled

**Symptoms:**
- Orders are filled immediately after placement
- Position size grows rapidly

**Solutions:**
1. Increase spread settings
   ```json
   "quoter": {
     "min_spread_bps": 20,  // Increase from lower value
     "max_spread_bps": 150  // Increase from lower value
   }
   ```

2. Verify price calculation logic
   Check that the bot is receiving accurate market data and calculating prices correctly.

## Performance Issues

### High CPU/Memory Usage

**Symptoms:**
- System becomes sluggish
- Bot operation slows down

**Solutions:**
1. Reduce update frequency
   ```json
   "quoter": {
     "quote_update_interval_ms": 1000  // Increase from lower value
   }
   ```

2. Decrease the number of instruments
   Focus on fewer trading pairs if resources are constrained.

3. Optimize logging
   Reduce log verbosity if not needed for debugging.

### Slow Response to Market Changes

**Symptoms:**
- Bot quotes lag behind market movements
- Positions build up in trending markets

**Solutions:**
1. Decrease quote update interval
   ```json
   "quoter": {
     "quote_update_interval_ms": 500  // Decrease from higher value
   }
   ```

2. Adjust volatility thresholds
   ```json
   "quoter": {
     "vol_threshold": 0.03  // Decrease for more responsive spreads
   }
   ```

## Risk Management Issues

### Excessive Position Building

**Symptoms:**
- Position size grows beyond comfortable levels
- One-sided position building

**Solutions:**
1. Enforce stricter position limits
   ```json
   "trading": {
     "max_position_size": {
       "BTC-PERP": 0.5  // Decrease from higher value
     }
   }
   ```

2. Increase inventory risk aversion
   ```json
   "quoter": {
     "inventory_risk_aversion": 0.95  // Increase from lower value
   }
   ```

3. Implement price skewing based on inventory
   Ensure inventory-based price skewing is properly configured.

### Excessive Losses

**Symptoms:**
- PnL decreases rapidly
- Multiple losing trades in sequence

**Solutions:**
1. Implement loss circuit breakers
   Configure the bot to pause trading after a certain loss threshold.

2. Widen spreads during volatility
   ```json
   "quoter": {
     "max_spread_bps": 300,  // Increase from lower value
     "vol_threshold": 0.04   // Decrease from higher value
   }
   ```

3. Review and adjust strategy
   Your strategy may need adjustments based on current market conditions.

## Configuration Issues

### Configuration Not Taking Effect

**Symptoms:**
- Changes to configuration don't affect bot behavior
- Bot continues with previous settings

**Solutions:**
1. Verify configuration file path
   Ensure the bot is loading the correct configuration file.

2. Check command-line overrides
   Command-line arguments override configuration file settings.

3. Restart the bot
   Some configuration changes require a bot restart.

### Invalid Configuration

**Symptoms:**
- Bot crashes at startup
- Error messages related to configuration

**Solutions:**
1. Validate JSON format
   Check for syntax errors like missing commas or braces.

2. Verify parameter types
   Ensure parameters have correct types (string, number, boolean).

3. Check required fields
   Ensure all required configuration fields are present.

## Logging and Debugging

### Enable Verbose Logging

For detailed troubleshooting, enable verbose logging:

```json
"general": {
  "log_level": "DEBUG"
}
```

### Common Log Messages

**Informational:**
```
INFO:root:Orders placed successfully: 8 bids, 8 asks for BTC-PERP
```
Normal operation, no action needed.

**Warning:**
```
WARNING:root:Order placement failed, retrying (attempt 2/3)
```
Temporary issue, monitor for resolution.

**Error:**
```
ERROR:root:Failed to connect to exchange API after 3 attempts
```
Requires intervention using solutions mentioned above.

## Recovery Procedures

### Safe Restart Procedure

1. Cancel all open orders
   ```bash
   # Command to cancel all orders (if available in your bot)
   ./thalex_simplequoter --cancel-all
   ```

2. Check positions
   Verify current positions before restarting.

3. Restart with conservative settings
   Use wider spreads and smaller sizes initially.

### Emergency Shutdown

If the bot is behaving erratically or causing losses:

1. Stop the bot process
   ```bash
   # Find the process ID
   ps aux | grep thalex_simplequoter
   
   # Kill the process
   kill -9 [PID]
   ```

2. Cancel all open orders manually on the exchange

3. Assess positions and manage manually if needed

## Getting Support

If you continue experiencing issues after trying these solutions:

1. Check documentation for updates
2. Review exchange status pages for any known issues
3. Provide the following when seeking support:
   - Configuration file (with sensitive information removed)
   - Relevant log files
   - Error messages and timestamps
   - Description of the issue and steps to reproduce 