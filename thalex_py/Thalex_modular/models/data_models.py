"""
Data models for the Thalex market maker
"""
from enum import Enum
from typing import Optional, Dict, Any
from dataclasses import dataclass


class OrderStatus(Enum):
    """Order status enumeration"""
    PENDING = "pending"
    OPEN = "open"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class Ticker:
    """Market ticker data"""
    def __init__(self, data: Dict[str, Any]):
        self.mark_price = data.get("mark_price", 0.0)
        self.best_bid_price = data.get("best_bid_price")
        self.best_ask_price = data.get("best_ask_price")
        self.last_price = data.get("last_price", 0.0)
        self.volume = data.get("volume", 0.0)
        self.timestamp = data.get("timestamp", 0)


@dataclass
class Order:
    """Order data model"""
    def __init__(self, order_data: Dict[str, Any]):
        self.id = order_data.get("order_id", "")
        self.direction = order_data.get("direction", "")
        self.amount = order_data.get("amount", 0.0)
        self.price = order_data.get("price", 0.0)
        self.status = OrderStatus(order_data.get("status", "pending"))
        self.instrument_name = order_data.get("instrument_name", "")
        self.timestamp = order_data.get("timestamp", 0)


@dataclass
class Quote:
    """Quote data model"""
    def __init__(self, instrument_name: str, bid_price: Optional[float] = None, 
                 ask_price: Optional[float] = None, bid_size: Optional[float] = None, 
                 ask_size: Optional[float] = None):
        self.instrument_name = instrument_name
        self.bid_price = bid_price
        self.ask_price = ask_price
        self.bid_size = bid_size
        self.ask_size = ask_size
        self.timestamp = 0 