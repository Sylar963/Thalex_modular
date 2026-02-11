from collections import deque
import time
import logging
from typing import Dict, Deque, Optional
from ..interfaces import SignalEngine
from ..entities import Ticker, Trade

logger = logging.getLogger(__name__)


class CanarySensor(SignalEngine):
    __slots__ = (
        "_l1_history",
        "_trade_history",
        "_pull_rate",
        "_quote_stability",
        "_size_asymmetry",
        "_toxicity_score",
        "_last_bid",
        "_last_ask",
        "_last_bid_size",
        "_last_ask_size",
        "_window_ms",
        "_last_log_time",
    )

    def __init__(self, window_ms: int = 5000):
        self._window_ms = window_ms
        self._l1_history: Deque[Dict] = deque(maxlen=200)
        self._trade_history: Deque[Dict] = deque(maxlen=100)

        self._last_bid: float = 0.0
        self._last_ask: float = 0.0
        self._last_bid_size: float = 0.0
        self._last_ask_size: float = 0.0

        self._pull_rate: float = 0.0
        self._quote_stability: float = 1.0
        self._size_asymmetry: float = 0.0
        self._toxicity_score: float = 0.0
        self._last_log_time: float = 0.0

    def update(self, ticker: Ticker) -> None:
        now = ticker.timestamp * 1000 if ticker.timestamp < 1e12 else ticker.timestamp

        bid_pulled = False
        ask_pulled = False

        if self._last_bid > 0:
            bid_delta = abs(ticker.bid - self._last_bid)
            ask_delta = abs(ticker.ask - self._last_ask)

            if (
                self._last_bid_size > ticker.bid_size * 2
                and bid_delta < ticker.bid * 0.001
            ):
                bid_pulled = True

            if (
                self._last_ask_size > ticker.ask_size * 2
                and ask_delta < ticker.ask * 0.001
            ):
                ask_pulled = True

        self._l1_history.append(
            {
                "ts": now,
                "bid": ticker.bid,
                "ask": ticker.ask,
                "bid_size": ticker.bid_size,
                "ask_size": ticker.ask_size,
                "bid_pulled": bid_pulled,
                "ask_pulled": ask_pulled,
            }
        )

        self._last_bid = ticker.bid
        self._last_ask = ticker.ask
        self._last_bid_size = ticker.bid_size
        self._last_ask_size = ticker.ask_size

        self._calculate_metrics(now)

    def update_trade(self, trade: Trade) -> None:
        now = trade.timestamp * 1000 if trade.timestamp < 1e12 else trade.timestamp

        self._trade_history.append(
            {
                "ts": now,
                "size": trade.size,
                "side": trade.side.value,
            }
        )

    def get_signals(self, symbol: Optional[str] = None) -> Dict[str, float]:
        return {
            "toxicity_score": self._toxicity_score,
            "pull_rate": self._pull_rate,
            "quote_stability": self._quote_stability,
            "size_asymmetry": self._size_asymmetry,
        }

    def _calculate_metrics(self, now_ms: float) -> None:
        cutoff = now_ms - self._window_ms
        recent = [s for s in self._l1_history if s["ts"] >= cutoff]

        if len(recent) < 5:
            return

        total_samples = len(recent)
        pulls = sum(1 for s in recent if s["bid_pulled"] or s["ask_pulled"])
        self._pull_rate = pulls / total_samples if total_samples > 0 else 0.0

        bid_changes = 0
        ask_changes = 0
        for i in range(1, len(recent)):
            if recent[i]["bid"] != recent[i - 1]["bid"]:
                bid_changes += 1
            if recent[i]["ask"] != recent[i - 1]["ask"]:
                ask_changes += 1

        max_changes = total_samples - 1
        if max_changes > 0:
            churn_rate = (bid_changes + ask_changes) / (2 * max_changes)
            self._quote_stability = 1.0 - min(churn_rate, 1.0)
        else:
            self._quote_stability = 1.0

        avg_displayed = (
            sum(s["bid_size"] + s["ask_size"] for s in recent) / total_samples
        )
        recent_trades = [t for t in self._trade_history if t["ts"] >= cutoff]
        total_filled = sum(t["size"] for t in recent_trades)

        if avg_displayed > 0 and total_filled > 0:
            fill_ratio = total_filled / avg_displayed
            self._size_asymmetry = max(0.0, 1.0 - fill_ratio)
        else:
            self._size_asymmetry = 0.0

        self._toxicity_score = (
            self._pull_rate * 0.35
            + (1.0 - self._quote_stability) * 0.30
            + self._size_asymmetry * 0.35
        )
        self._toxicity_score = min(1.0, max(0.0, self._toxicity_score))

        if now_ms / 1000 - self._last_log_time > 60:
            if self._toxicity_score > 0.1:
                logger.info(
                    f"Canary Sensor: toxicity={self._toxicity_score:.2f}, "
                    f"pull_rate={self._pull_rate:.2f}, stability={self._quote_stability:.2f}"
                )
            self._last_log_time = now_ms / 1000
