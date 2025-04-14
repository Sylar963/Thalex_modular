# Thalex Hedge Manager Security Fix

## Security Alert
This repository previously contained hardcoded API keys and private keys in the source code. 
These have been removed and replaced with a more secure approach using environment variables.

## Setup Instructions

### 1. Set up environment variables

Copy the `.env.sample` file to a new file named `.env`:

```bash
cp .env.sample .env
```

Edit the `.env` file and add your real API credentials:

```
# Testnet credentials
THALEX_TEST_KEY_ID=your_actual_key_id
THALEX_TEST_PRIVATE_KEY="-----BEGIN RSA PRIVATE KEY-----
YOUR_ACTUAL_PRIVATE_KEY_CONTENT
-----END RSA PRIVATE KEY-----"

# Alternative credentials for hedge manager
THALEX_API_KEY=your_api_key
THALEX_API_SECRET=your_api_secret
```

**IMPORTANT:** 
- Never commit your `.env` file to source control
- Add `.env` to your `.gitignore` file
- Protect access to your API keys

### 2. Install required packages

```bash
pip install python-dotenv
```

### 3. Run the hedge manager

```bash
python fix_eth_price_fetching.py
```

This will:
1. Load your API credentials from the `.env` file
2. Connect to Thalex exchange API
3. Fetch real-time ETH and BTC prices
4. Properly execute hedge trades for delta neutrality

## Alternative Usage

You can also pass your API credentials directly as command-line arguments:

```bash
python fix_eth_price_fetching.py your_api_key your_api_secret
```

## Key Improvements

1. **Security**: API keys are now loaded from environment variables instead of being hardcoded
2. **Real-time prices**: ETH and BTC prices are now fetched in real-time from the exchange
3. **Trade execution**: Trades are now properly executed on the exchange
4. **Delta neutrality**: The hedge manager now maintains proper delta-neutral positions 