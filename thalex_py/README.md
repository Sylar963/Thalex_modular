# Thalex Python Client

This is a Python client library for interacting with the [Thalex](https://www.thalex.com) derivatives exchange API.

It is provided free of charge, as is, with no warranties and no recommendations – see MIT license.

See also the [API documentation](https://www.thalex.com/docs/).

## Installation

```bash
pip install thalex
```

Alternatively, if you want the examples as well:
```bash
git clone https://github.com/thalextech/thalex_py.git
cd thalex_py
pip install -e ./
```

## Usage

The client provides a function for every WebSocket endpoint with type hints and a receive function 
that returns messages from the exchange one by one.

```python
import thalex as th
from thalex import Network

# Initialize client
client = th.Thalex(network=Network.TEST)  # Use Network.PROD for production

# Connect to WebSocket
await client.connect()

# Get instruments
await client.instruments(id=1)
response = await client.receive()
```

## Examples

There are examples in the `examples` folder showing how to use this library.

If you want to run the examples, rename/copy `_keys.py` to `keys.py`, 
create API keys in the Thalex UI and add them to `keys.py`.

## Cloud Deployment

See [this guide](https://thalex.com/blog/how-to-run-a-thalex-bot-on-aws) 
about how you can deploy a trading bot in the cloud.

## Disclaimer

This library is provided as-is with no warranty. It is not financial advice.

## Support

If you spot any errors/bugs please report or create a pull request.

You can also reach out at thalex_py@thalex.com
