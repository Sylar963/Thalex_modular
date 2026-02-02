from abc import abstractmethod
from typing import Optional, Callable
from ...domain.interfaces import ExchangeGateway


class BaseExchangeAdapter(ExchangeGateway):
    def __init__(self, api_key: str, api_secret: str, testnet: bool = True):
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        self.connected = False

        self.ticker_callback: Optional[Callable] = None
        self.trade_callback: Optional[Callable] = None
        self.order_callback: Optional[Callable] = None
        self.position_callback: Optional[Callable] = None

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    def set_ticker_callback(self, callback: Callable):
        self.ticker_callback = callback

    def set_trade_callback(self, callback: Callable):
        self.trade_callback = callback

    def set_order_callback(self, callback: Callable):
        self.order_callback = callback

    def set_position_callback(self, callback: Callable):
        self.position_callback = callback
