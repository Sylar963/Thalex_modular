import asyncio
import logging
import time
import json
from typing import Dict, Optional, Callable, List, Any, Union
from dataclasses import replace

import aiohttp
import websockets
import pandas as pd

from ...domain.interfaces import ExchangeGateway, TimeSyncManager
from .base_adapter import TokenBucket, BaseExchangeAdapter
from ...domain.entities import (
    Order,
    OrderSide,
    OrderType,
    OrderStatus,
    Position,
    Ticker,
    Trade,
    Balance,
)

logger = logging.getLogger(__name__)


class DeriveAdapter(BaseExchangeAdapter):
    """
    Adapter for Derive (formerly Lyra) exchange.
    Follows the 'Hybrid' approach: Custom WebSocket for data, ready for SDK integration for trading.
    """

    def __init__(
        self,
        api_key: str = "",
        api_secret: str = "",
        testnet: bool = False,
        time_sync_manager: Optional[TimeSyncManager] = None,
    ):
        super().__init__(api_key, api_secret, testnet, time_sync_manager)

        self.testnet = testnet
        if testnet:
            self.http_url = "https://api-demo.lyra.finance"
            self.ws_url = "wss://api-demo.lyra.finance/ws"
        else:
            self.http_url = "https://api.lyra.finance"
            self.ws_url = "wss://api.lyra.finance/ws"

        self.ws: Optional[websockets.WebSocketClientProtocol] = None
        self.msg_loop_task: Optional[asyncio.Task] = None
        self.keepalive_task: Optional[asyncio.Task] = None
        
        self.ticker_map: Dict[str, Ticker] = {}
        self.session: Optional[aiohttp.ClientSession] = None
        self.pending_requests: Dict[int, asyncio.Future] = {}
        self._request_id = int(time.time() * 1000)

        # Rate limits (conservative defaults)
        self.ws_rate_limiter = TokenBucket(capacity=10, fill_rate=5) 

    @property
    def name(self) -> str:
        return "derive"

    def _get_next_id(self) -> int:
        self._request_id += 1
        return self._request_id

    async def _rpc_request_ws(self, method: str, params: Dict, timeout: float = 10.0) -> Dict:
        """Execute a JSON-RPC request over the established WebSocket."""
        if not self.ws or not self.connected:
            raise ConnectionError("WebSocket not connected")

        req_id = self._get_next_id()
        future = asyncio.Future()
        self.pending_requests[req_id] = future

        payload = {
            "jsonrpc": "2.0",
            "id": req_id,
            "method": method,
            "params": params
        }

        try:
            await self.ws.send(json.dumps(payload))
            return await asyncio.wait_for(future, timeout=timeout)
        except Exception as e:
            if req_id in self.pending_requests:
                del self.pending_requests[req_id]
            logger.error(f"WS RPC failed ({method}): {e}")
            raise

    async def connect(self):
        """Establish WebSocket connection and start message loop."""
        if self.connected:
            return

        logger.info(f"Connecting to Derive ({self.ws_url})...")
        try:
            self.session = aiohttp.ClientSession()
            self.ws = await websockets.connect(self.ws_url, ping_interval=20, ping_timeout=20)
            self.connected = True
            
            self.msg_loop_task = asyncio.create_task(self._msg_loop())
            logger.info("Derive Adapter connected.")
            
        except Exception as e:
            logger.error(f"Failed to connect to Derive: {e}")
            self.connected = False
            if self.session:
                await self.session.close()
            raise ConnectionError(f"Derive connection failed: {e}")

    async def disconnect(self):
        """Close connections."""
        self.connected = False
        if self.msg_loop_task:
            self.msg_loop_task.cancel()
        
        if self.ws:
            await self.ws.close()
        
        if self.session:
            await self.session.close()
            
        logger.info("Derive Adapter disconnected.")

    async def _rpc_request_http(self, method: str, params: Dict) -> Dict:
        """Execute a raw JSON-RPC request over HTTP."""
        if not self.session:
            self.session = aiohttp.ClientSession()

        payload = {
            "jsonrpc": "2.0",
            "id": int(time.time() * 1000),
            "method": method,
            "params": params
        }
        
        try:
            async with self.session.post(self.http_url, json=payload) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    raise Exception(f"HTTP {resp.status}: {text}")
                
                return await resp.json()
        except Exception as e:
            logger.error(f"HTTP RPC failed: {e}")
            raise

    async def get_instruments(self, currency: str = "HYPE", instrument_type: str = "option", expired: bool = False) -> List[Dict]:
        """Fetch instruments via WebSocket (preferred) or REST."""
        params = {
            "currency": currency,
            "instrument_type": instrument_type,
            "expired": expired
        }
        try:
            if self.connected:
                response = await self._rpc_request_ws("public/get_instruments", params)
                if "result" in response:
                    return response["result"]
            
            # Fallback to HTTP (suppress error logs if WS is preferred)
            response = await self._rpc_request_http("public/get_instruments", params)
            if "result" in response:
                return response["result"]
            return []
        except Exception:
            # Silent fallback/fail for public data if one method worked
            return []
            
    async def get_tickers(self, currency: str = "HYPE", instrument_type: str = "option", expiry: Optional[str] = None) -> List[Dict]:
        """Fetch initial tickers via WebSocket (preferred) or REST."""
        params = {
            "currency": currency,
            "instrument_type": instrument_type
        }
        if expiry:
            params["expiry"] = expiry

        try:
            if self.connected:
                response = await self._rpc_request_ws("public/get_tickers", params)
                if "result" in response:
                    return response["result"]
            
            # Fallback to HTTP
            response = await self._rpc_request_http("public/get_tickers", params)
            if "result" in response:
                return response["result"]
            return []
        except Exception:
            return []

    async def subscribe_ticker(self, instrument_names: Union[str, List[str]], interval: str = "100ms"):
        """Subscribe to ticker updates."""
        if isinstance(instrument_names, str):
            instrument_names = [instrument_names]
            
        channels = [f"ticker.{name}.{interval}" for name in instrument_names]
        
        subscribe_msg = {
            "jsonrpc": "2.0",
            "id": self._get_next_id(),
            "method": "subscribe",
            "params": {
                "channels": channels
            }
        }
        
        if self.ws and self.connected:
            await self.ws.send(json.dumps(subscribe_msg))
            logger.info(f"Subscribed to {len(channels)} ticker channels")
        else:
            logger.warning("Cannot subscribe: WebSocket not connected")

    async def _msg_loop(self):
        """Process incoming WebSocket messages."""
        while self.connected:
            try:
                msg = await self.ws.recv()
                data = json.loads(msg)
                
                # Handle RPC response
                msg_id = data.get("id")
                if msg_id is not None and msg_id in self.pending_requests:
                    self.pending_requests[msg_id].set_result(data)
                    del self.pending_requests[msg_id]
                    continue

                # Handle different message types
                method = data.get("method")
                params = data.get("params", {})
                
                if method == "subscription":
                    channel = params.get("channel", "")
                    payload = params.get("data", {})
                    
                    if "ticker" in channel:
                        await self._handle_ticker_update(channel, payload)
                        
            except websockets.exceptions.ConnectionClosed:
                logger.warning("Derive WS Connection Closed")
                self.connected = False
                break
            except Exception as e:
                logger.error(f"Error in Derive msg loop: {e}")
                await asyncio.sleep(1)

    async def _handle_ticker_update(self, channel: str, data: Dict):
        """Process ticker update and fire callback."""
        instrument_name = data.get("instrument_name")
        if not instrument_name:
            return

        # Map to internal Ticker entity
        ticker = Ticker(
            symbol=instrument_name,
            bid=self._safe_float(data.get("best_bid_price")),
            ask=self._safe_float(data.get("best_ask_price")),
            bid_size=self._safe_float(data.get("best_bid_amount")),
            ask_size=self._safe_float(data.get("best_ask_amount")),
            last=self._safe_float(data.get("last_price")),
            volume=self._safe_float(data.get("volume_24h")),
            mark_price=self._safe_float(data.get("mark_price")),
            timestamp=int(time.time() * 1000) # Local time for now, or parse server time
        )
        
        self.ticker_map[instrument_name] = ticker
        
        if self.ticker_callback:
            await self.ticker_callback(ticker)

    # Implement other abstract methods with stubs or specific logic
    async def place_order(self, order: Order) -> Order:
        raise NotImplementedError("Trading not yet implemented for Derive")
        
    async def cancel_order(self, order_id: str) -> bool:
        raise NotImplementedError("Trading not yet implemented for Derive")
        
    async def get_open_orders(self, symbol: str) -> List[Order]:
        return []
    
    async def get_balances(self) -> List[Balance]:
        return []
    
    async def get_position(self, symbol: str) -> Position:
        return Position(symbol, 0, 0)

    async def get_server_time(self) -> int:
        """Fetch server time. For now, return local time."""
        # TODO: Implement actual server time fetch if available
        return int(time.time() * 1000)

    async def place_orders_batch(self, orders: List[Order]) -> List[Order]:
        raise NotImplementedError("Trading not yet implemented for Derive")

    async def cancel_orders_batch(self, order_ids: List[str]) -> List[bool]:
        raise NotImplementedError("Trading not yet implemented for Derive")

    async def build_options_chain(self, currency: str = "HYPE") -> pd.DataFrame:
        """Fetch instruments and tickers to build a complete chain."""
        instruments = await self.get_instruments(currency=currency)
        if not instruments:
            return pd.DataFrame()

        # Group instruments by expiry to fetch tickers efficiently
        expiries = set()
        inst_by_name = {}
        for inst in instruments:
            name = inst['instrument_name']
            inst_by_name[name] = inst
            parts = name.split('-')
            if len(parts) >= 2:
                expiries.add(parts[1])

        all_tickers = []
        for expiry in expiries:
            tickers = await self.get_tickers(currency=currency, expiry=expiry)
            all_tickers.extend(tickers)
            await asyncio.sleep(0.2) # Avoid flooding
        
        ticker_dict = {t['instrument_name']: t for t in all_tickers}
        
        chain_data = []
        for name, inst in inst_by_name.items():
            ticker = ticker_dict.get(name, {})
            
            # Parse instrument name: HYPE-20240329-10-C
            parts = name.split('-')
            if len(parts) >= 4:
                expiry = parts[1]
                strike = parts[2]
                opt_type = parts[3]
                
                chain_data.append({
                    'instrument_name': name,
                    'expiry': expiry,
                    'strike': float(strike),
                    'type': 'Call' if 'C' in opt_type else 'Put',
                    'mark_price': self._safe_float(ticker.get('mark_price')),
                    'bid': self._safe_float(ticker.get('best_bid_price')),
                    'ask': self._safe_float(ticker.get('best_ask_price')),
                    'iv': self._safe_float(ticker.get('mark_iv')),
                    'delta': self._safe_float(ticker.get('greeks', {}).get('delta')),
                })
                
        return pd.DataFrame(chain_data)
