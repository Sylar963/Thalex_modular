from dataclasses import dataclass
from typing import List, Optional
from enum import Enum


class HistoryType(Enum):
    TICKER = "ticker"
    TRADE = "trade"
    LOB = "lob"


@dataclass(slots=True)
class HistoryConfig:
    symbol: str
    venue: str
    start_time: float
    end_time: float
    resolution: str = "1m"
    include_lob: bool = False
