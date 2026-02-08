import abc
import logging
import asyncio
from datetime import datetime, timezone
from typing import List, Dict, Optional, Any
import aiohttp

try:
    import orjson
except ImportError:
    import json as orjson

logger = logging.getLogger(__name__)


class HistoricalSource(abc.ABC):
    def __init__(self, session: aiohttp.ClientSession):
        self.session = session

    @property
    @abc.abstractmethod
    def exchange_name(self) -> str:
        pass

    @abc.abstractmethod
    async def fetch_history(self, symbol: str, start_ts: int, end_ts: int) -> List[Dict]:
        """
        Fetch historical market data.
        Returns a list of dictionaries with keys corresponding to market_tickers columns:
        time, symbol, exchange, bid, ask, last, volume.
        """
        pass


class ThalexHistoricalSource(HistoricalSource):
    BASE_URL = "https://thalex.com/api/v2/public"

    @property
    def exchange_name(self) -> str:
        return "thalex"

    async def fetch_history(self, symbol: str, start_ts: int, end_ts: int) -> List[Dict]:
        """
        Fetches mark price history from Thalex.
        """
        # Thalex usually uses 'BTC-PERPETUAL' for perps.
        # If symbol is just 'BTC', we might mean index?
        # The loader uses 'BTCUSD' for index and 'BTC-PERPETUAL' for perp.
        
        endpoint = "mark_price_historical_data"
        params = {
            "instrument_name": symbol,
            "from": start_ts,
            "to": end_ts,
            "resolution": "1m",
        }

        # Special handling for Index
        if symbol == "BTCUSD":
             endpoint = "index_price_historical_data"
             params = {
                "index_name": symbol,
                "from": start_ts,
                "to": end_ts,
                "resolution": "1m",
            }

        url = f"{self.BASE_URL}/{endpoint}"
        
        try:
            async with self.session.get(url, params=params) as resp:
                if resp.status != 200:
                    logger.error(f"Thalex API Error {resp.status} for {url}: {await resp.text()}")
                    return []
                
                data = await resp.json(loads=orjson.loads)
                result = data.get("result", [])
                
                # Handle nested dict response structure if any
                if isinstance(result, dict):
                    # Thalex sometimes wraps list in 'mark_price_history' or similar keys
                    for val in result.values():
                        if isinstance(val, list):
                            result = val
                            break
                
                if not isinstance(result, list):
                    return []

                # Transform to standard format
                # Thalex format: [time, open, high, low, close, volume] (standard OHLCV often)
                # Or sometimes: [t, o, h, l, c, v, ...]
                # Based on previous loader: index 4 is close/mark.
                
                tickers = []
                for point in result:
                    if isinstance(point, list) and len(point) >= 5:
                        ts = point[0]
                        price = float(point[4])
                        # volume = float(point[5]) if len(point) > 5 else 0.0 # Volume might be 0 for mark price
                        
                        tickers.append({
                            "time": datetime.fromtimestamp(ts, tz=timezone.utc),
                            "symbol": symbol,
                            "exchange": self.exchange_name,
                            "bid": price,
                            "ask": price,
                            "last": price,
                            "volume": 0.0 # Thalex mark price history often has no volume
                        })
                    elif isinstance(point, dict):
                         ts = point.get("time")
                         price = float(point.get("close", 0) or point.get("price", 0))
                         if ts:
                            tickers.append({
                                "time": datetime.fromtimestamp(ts, tz=timezone.utc),
                                "symbol": symbol,
                                "exchange": self.exchange_name,
                                "bid": price,
                                "ask": price,
                                "last": price,
                                "volume": 0.0
                            })
                return tickers

        except Exception as e:
            logger.error(f"Failed to fetch Thalex history for {symbol}: {e}")
            return []


class BybitHistoricalSource(HistoricalSource):
    BASE_URL = "https://api.bybit.com/v5/market"

    @property
    def exchange_name(self) -> str:
        return "bybit"

    async def fetch_history(self, symbol: str, start_ts: int, end_ts: int) -> List[Dict]:
        """
        Fetch history from Bybit.
        Uses mark-price-kline for mark price history (best for trend/simulation).
        Handles pagination (limit 1000).
        """
        # Map symbol: 'BTC-PERP' -> 'BTCUSDT' (Linear)
        mapped_symbol = symbol
        if symbol in ["BTC-PERP", "BTC-PERPETUAL"]:
            mapped_symbol = "BTCUSDT"
        
        # Bybit uses milliseconds
        req_start_ms = start_ts * 1000
        req_end_ms = end_ts * 1000
        
        endpoint = "mark-price-kline"
        all_tickers = []
        
        current_end_ms = req_end_ms
        
        while True:
            # Safety break if we went past start
            if current_end_ms < req_start_ms:
                break

            params = {
                "category": "linear",
                "symbol": mapped_symbol,
                "interval": "1", # 1 minute
                "start": req_start_ms,
                "end": current_end_ms,
                "limit": 1000 
            }

            url = f"{self.BASE_URL}/{endpoint}"
            
            try:
                async with self.session.get(url, params=params) as resp:
                    if resp.status != 200:
                        logger.error(f"Bybit API Error {resp.status} for {url}: {await resp.text()}")
                        break

                    data = await resp.json(loads=orjson.loads)
                    if data.get("retCode") != 0:
                        logger.error(f"Bybit API Error: {data.get('retMsg')}")
                        break

                    result = data.get("result", {}).get("list", [])
                    
                    if not result:
                        break
                    
                    # Bybit returns [start, open, high, low, close] usually sorted by startTime DESC (Newest first)
                    # We process them and add to list
                    
                    batch_min_ts = None
                    
                    for point in result:
                        ts_ms = int(point[0])
                        close_price = float(point[4])
                        
                        if batch_min_ts is None or ts_ms < batch_min_ts:
                            batch_min_ts = ts_ms

                        # Only add if within absolute bounds (just in case)
                        if ts_ms >= req_start_ms and ts_ms <= req_end_ms:
                            all_tickers.append({
                                "time": datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc),
                                "symbol": symbol, 
                                "exchange": self.exchange_name,
                                "bid": close_price,
                                "ask": close_price,
                                "last": close_price,
                                "volume": 0.0
                            })
                    
                    # Pagination Logic
                    # If we got fewer than limit, we are done
                    if len(result) < 1000:
                        break
                    
                    # Prepare next page
                    # Since it returns newest first, the oldest in this batch is the 'cut off'
                    # Next request should end before this batch's oldest
                    if batch_min_ts:
                        current_end_ms = batch_min_ts - 1
                    else:
                        break # Should not happen if result is not empty
                    
                    # Safety: avoid infinite loops if API returns same data
                    if current_end_ms >= req_end_ms: 
                         # This implies we didn't move backwards. API might be behaving oddly or returning ASC.
                         # Bybit V5 usually DESC. If batch_min_ts is close to current_end_ms, it worked.
                         # But if current_end_ms didn't decrease, break.
                         break

            except Exception as e:
                logger.error(f"Failed to fetch Bybit history for {symbol}: {e}")
                break
                
        return all_tickers
