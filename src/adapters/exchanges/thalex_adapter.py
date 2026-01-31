import asyncio
import logging
import json
import time
from typing import Dict, List, Optional, Callable, Any
from dataclasses import replace

# Native Thalex client
try:
    from thalex.thalex import Thalex, Network, Direction, OrderType as ThOrderType
except ImportError:
    # Creating a mock if library is missing during development/refactor
    class Thalex:
        def __init__(self, network=None):
            pass

        async def connect(self):
            pass

        async def receive(self):
            await asyncio.sleep(1)
            return None

        def connected(self):
            return True

        async def login(self, k, s):
            pass

        async def public_subscribe(self, channels):
            pass

    class Network:
        TESTNET = "testnet"
        MAINNET = "mainnet"

    class Direction:
        BUY = "buy"
        SELL = "sell"

    class ThOrderType:
        MARKET = "market"
        LIMIT = "limit"


from ...domain.interfaces import ExchangeGateway
from ...domain.entities import (
    Order,
    OrderSide,
    OrderType,
    OrderStatus,
    Position,
    Ticker,
    Trade,
)

logger = logging.getLogger(__name__)


class ThalexAdapter(ExchangeGateway):
    """
    Adapter for the Thalex exchange using the official python library.
    """

    def __init__(self, api_key: str, api_secret: str, testnet: bool = True):
        self.api_key = api_key
        self.api_secret = api_secret
        self.network = Network.TESTNET if testnet else Network.MAINNET
        self.client = Thalex(network=self.network)

        self.connected = False
        self.ticker_callback: Optional[Callable] = None
        self.trade_callback: Optional[Callable] = None

        # Local state cache
        self.positions: Dict[str, Position] = {}
        self.orders: Dict[str, Order] = {}

        self.msg_loop_task: Optional[asyncio.Task] = None

    async def connect(self):
        logger.info("Connecting to Thalex...")
        await self.client.connect()
        if not self.client.connected():
            raise ConnectionError("Failed to connect to Thalex")

        logger.info("Logging in...")
        await self.client.login(self.api_key, self.api_secret)
        self.connected = True

        # Start message processing loop
        self.msg_loop_task = asyncio.create_task(self._msg_loop())
        logger.info("Thalex Adapter connected and listening.")

    async def disconnect(self):
        if self.msg_loop_task:
            self.msg_loop_task.cancel()
        # Thalex lib doesn't have explicit disconnect?
        # Assuming socket close happens on gc or we can close underlying ws if needed.
        self.connected = False
        logger.info("Thalex Adapter disconnected.")

    async def place_order(self, order: Order) -> Order:
        if not self.connected:
            raise ConnectionError("Not connected to exchange")

        direction = Direction.BUY if order.side == OrderSide.BUY else Direction.SELL
        th_type = (
            ThOrderType.LIMIT if order.type == OrderType.LIMIT else ThOrderType.MARKET
        )

        try:
            if order.side == OrderSide.BUY:
                response = await self.client.buy(
                    instrument_name=order.symbol,
                    amount=order.size,
                    price=order.price,
                    order_type=th_type,
                    label=order.id,  # Use client ID as label if possible
                )
            else:
                response = await self.client.sell(
                    instrument_name=order.symbol,
                    amount=order.size,
                    price=order.price,
                    order_type=th_type,
                    label=order.id,
                )

            # Map response to Order
            # Response usually contains 'order_id', 'status', etc.
            # We update the original order with exchange ID
            exchange_id = str(response.get("id", response.get("order_id", "")))
            updated_order = replace(
                order, exchange_id=exchange_id, status=OrderStatus.OPEN
            )

            # Cache order
            self.orders[exchange_id] = updated_order
            return updated_order

        except Exception as e:
            logger.error(f"Failed to place order {order}: {e}")
            return replace(order, status=OrderStatus.REJECTED)

    async def cancel_order(self, order_id: str) -> bool:
        if not self.connected:
            return False
        try:
            await self.client.cancel(order_id=order_id)
            if order_id in self.orders:
                self.orders[order_id] = replace(
                    self.orders[order_id], status=OrderStatus.CANCELLED
                )
            return True
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return False

    async def subscribe_ticker(self, symbol: str):
        channel = f"ticker.{symbol}.raw"  # Based on legacy code
        await self.client.public_subscribe(channels=[channel])
        # Also subscribe to trades if needed? public_subscribe handles list.
        # Legacy also subscribed to ticker.{symbol}.raw
        # Assuming trades come via different channel or part of ticker?
        # Usually trades are 'trades.{symbol}'.
        # For VolumeCandles we NEED trades.
        trade_channel = f"trades.{symbol}"
        await self.client.public_subscribe(channels=[trade_channel])

    async def get_position(self, symbol: str) -> Position:
        return self.positions.get(symbol, Position(symbol, 0.0, 0.0))

    def set_ticker_callback(self, callback):
        self.ticker_callback = callback

    def set_trade_callback(self, callback):
        self.trade_callback = callback

    async def _msg_loop(self):
        while self.connected:
            try:
                msg = await self.client.receive()
                if not msg:
                    continue

                if isinstance(msg, str):
                    msg = json.loads(msg)

                if not isinstance(msg, dict):
                    continue

                # Delegate processing
                await self._process_message(msg)

            except Exception as e:
                logger.error(f"Error in msg loop: {e}")
                await asyncio.sleep(1)

    async def _process_message(self, msg: Dict):
        # Handle Ticker Updates
        # Structure based on legacy: method='subscription', params={'channel': 'ticker...', 'data': ...}
        if msg.get("method") == "subscription":
            params = msg.get("params", {})
            channel = params.get("channel", "")
            data = params.get("data", {})

            if channel.startswith("ticker."):
                symbol = channel.split(".")[1]
                ticker = self._map_ticker(symbol, data)
                if self.ticker_callback:
                    await self.ticker_callback(ticker)

            elif channel.startswith("trades."):
                symbol = channel.split(".")[1]
                # trades data is usually a list of trades
                # legacy 'trades' channel structure assumption
                for t_data in data:
                    trade = self._map_trade(symbol, t_data)
                    if self.trade_callback:
                        await self.trade_callback(trade)
                    # Also update position if it's OUR fill (private)?
                    # No, public trades are market data. private fills come via 'private' channel usually.

        # Handle Private Fills (if applicable, for position tracking)
        # Assuming thalex has a 'private_notifications' or similar.
        # Legacy code didn't explicitly show private sub in init, maybe login enables it?
        # If we receive execution report:
        # Check msg type for fills.

    def _map_ticker(self, symbol: str, data: Dict) -> Ticker:
        return Ticker(
            symbol=symbol,
            bid=float(data.get("best_bid_price", 0)),
            ask=float(data.get("best_ask_price", 0)),
            bid_size=float(data.get("best_bid_amount", 0)),
            ask_size=float(data.get("best_ask_amount", 0)),
            last=float(data.get("last_price", 0)),
            volume=float(data.get("volume_24h", 0)),  # approximate mapping
            timestamp=float(data.get("timestamp", time.time())) / 1000.0
            if data.get("timestamp") > 1e10
            else time.time(),
        )

    def _map_trade(self, symbol: str, data: Dict) -> Trade:
        # Map public trade
        return Trade(
            id=str(data.get("id", "")),
            order_id="",  # Public trade has no order_id for us
            symbol=symbol,
            side=OrderSide.BUY
            if data.get("direction", data.get("side")) == "buy"
            else OrderSide.SELL,
            price=float(data.get("price", 0)),
            size=float(data.get("amount", 0)),
            timestamp=float(data.get("timestamp", time.time())) / 1000.0,
        )
