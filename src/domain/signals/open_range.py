import time
import logging
from datetime import datetime, timezone
from typing import Dict, Optional, List
from dataclasses import dataclass, field
from ..interfaces import SignalEngine
from ..entities import Ticker, Trade


logger = logging.getLogger(__name__)


@dataclass
class TargetLevel:
    price: float
    label: str
    hit: bool = False
    hit_time: Optional[float] = None


@dataclass
class OpenRangeState:
    session_start_hour: int = 20
    session_start_minute: int = 0
    session_end_hour: int = 20
    session_end_minute: int = 15
    target_pct_from_mid: float = 0.0149
    subsequent_target_pct_of_range: float = 2.2

    orh: float = 0.0
    orl: float = float("inf")
    orm: float = 0.0
    orw: float = 0.0
    prev_orm: float = 0.0

    or_sesh: bool = False
    or_token: bool = False
    session_date: Optional[str] = None

    first_up_target: float = 0.0
    first_down_target: float = 0.0
    first_up_hit: bool = False
    first_down_hit: bool = False

    day_dir: int = 0
    up_check: bool = True
    down_check: bool = True

    breakout_direction: Optional[str] = None
    up_signal: bool = False
    down_signal: bool = False

    up_targets: List[TargetLevel] = field(default_factory=list)
    down_targets: List[TargetLevel] = field(default_factory=list)

    up_count: int = 0
    down_count: int = 0

    hst: float = 0.0
    lst: float = 0.0


class OpenRangeSignalEngine(SignalEngine):
    def __init__(
        self,
        session_start_utc: str = "20:00",
        session_end_utc: str = "20:15",
        target_pct_from_mid: float = 1.49,
        subsequent_target_pct_of_range: float = 220,
        use_bias: bool = False,
    ):
        start_parts = session_start_utc.split(":")
        end_parts = session_end_utc.split(":")

        self.state = OpenRangeState(
            session_start_hour=int(start_parts[0]),
            session_start_minute=int(start_parts[1]) if len(start_parts) > 1 else 0,
            session_end_hour=int(end_parts[0]),
            session_end_minute=int(end_parts[1]) if len(end_parts) > 1 else 0,
            target_pct_from_mid=target_pct_from_mid / 100.0,
            subsequent_target_pct_of_range=subsequent_target_pct_of_range / 100.0,
        )

        self.use_bias = use_bias
        self.signals: Dict[str, float] = {}
        self._last_close = 0.0
        self._last_high = 0.0
        self._last_low = 0.0
        self._prev_close = 0.0
        self._session_just_completed = False

    def _is_in_session(self, ts: float) -> bool:
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        current_minutes = dt.hour * 60 + dt.minute
        start_minutes = (
            self.state.session_start_hour * 60 + self.state.session_start_minute
        )
        end_minutes = self.state.session_end_hour * 60 + self.state.session_end_minute

        return start_minutes <= current_minutes < end_minutes

    def _get_session_date(self, ts: float) -> str:
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        return dt.strftime("%Y-%m-%d")

    def update(self, ticker: Ticker):
        ts = ticker.timestamp
        price = ticker.mid_price
        high = ticker.ask if ticker.ask > 0 else price
        low = ticker.bid if ticker.bid > 0 else price

        self._prev_close = self._last_close
        self._last_close = price
        self._last_high = high
        self._last_low = low
        self._session_just_completed = False

        current_date = self._get_session_date(ts)
        was_in_session = self.state.or_sesh
        in_session = self._is_in_session(ts)

        or_start = in_session and not was_in_session
        or_end = was_in_session and not in_session

        if current_date != self.state.session_date:
            or_start = True

        if or_start:
            self._on_or_start(current_date, high, low)

        if in_session:
            self._during_or_session(high, low)

        if or_end and self.state.or_token:
            self._on_or_end()

        if not in_session and self.state.or_token:
            self._outside_or_session(price, high, low)

        self.state.or_sesh = in_session
        self._update_signals()

    def _on_or_start(self, date: str, high: float, low: float):
        self.state.prev_orm = self.state.orm

        self.state.up_targets.clear()
        self.state.down_targets.clear()

        self.state.orh = high
        self.state.orl = low
        self.state.session_date = date
        self.state.up_count = 0
        self.state.down_count = 0
        self.state.up_check = True
        self.state.down_check = True
        self.state.first_up_hit = False
        self.state.first_down_hit = False
        self.state.up_signal = False
        self.state.down_signal = False
        self.state.breakout_direction = None

        self.state.or_token = False

        logger.info(f"Open Range session started for {date}")

    def _during_or_session(self, high: float, low: float):
        if high > self.state.orh:
            self.state.orh = high
        if low < self.state.orl:
            self.state.orl = low

        if self.state.orh != self.state.orl:
            self.state.or_token = True

    def _on_or_end(self):
        self.state.orm = (self.state.orh + self.state.orl) / 2.0
        self.state.orw = abs(self.state.orh - self.state.orl)

        self.state.first_up_target = self.state.orm + (
            self.state.orm * self.state.target_pct_from_mid
        )
        self.state.first_down_target = self.state.orm - (
            self.state.orm * self.state.target_pct_from_mid
        )

        self.state.hst = self.state.first_up_target + (
            self.state.orw * self.state.subsequent_target_pct_of_range
        )
        self.state.lst = self.state.first_down_target - (
            self.state.orw * self.state.subsequent_target_pct_of_range
        )

        if self.state.prev_orm > 0:
            if self.state.orm > self.state.prev_orm:
                self.state.day_dir = 1
            elif self.state.orm < self.state.prev_orm:
                self.state.day_dir = -1
            else:
                self.state.day_dir = 0

        self.state.up_targets.append(
            TargetLevel(price=self.state.first_up_target, label="T1")
        )
        self.state.down_targets.append(
            TargetLevel(price=self.state.first_down_target, label="T1")
        )

        self._session_just_completed = True

        logger.info(
            f"OR End: ORH={self.state.orh:.2f}, ORL={self.state.orl:.2f}, "
            f"ORM={self.state.orm:.2f}, ORW={self.state.orw:.2f}, "
            f"T1_UP={self.state.first_up_target:.2f}, T1_DN={self.state.first_down_target:.2f}, "
            f"day_dir={self.state.day_dir}"
        )

    def _outside_or_session(self, close: float, high: float, low: float):
        h_src = close
        l_src = close

        if not self.state.first_up_hit and h_src > self.state.first_up_target:
            self.state.first_up_hit = True
            self.state.hst = h_src
            self.state.up_targets[0].hit = True
            logger.info(f"First UP target hit at {h_src:.2f}")

        if not self.state.first_down_hit and l_src < self.state.first_down_target:
            self.state.first_down_hit = True
            self.state.lst = l_src
            self.state.down_targets[0].hit = True
            logger.info(f"First DOWN target hit at {l_src:.2f}")

        if self.state.first_up_hit and h_src > self.state.hst:
            self.state.hst = h_src
        if self.state.first_down_hit and l_src < self.state.lst:
            self.state.lst = l_src

        up_max = 0
        down_max = 0
        if self.state.first_up_hit and self.state.orw > 0:
            up_max = self._get_1up(
                (self.state.hst - self.state.first_up_target)
                / (self.state.orw * self.state.subsequent_target_pct_of_range)
            )
        if self.state.first_down_hit and self.state.orw > 0:
            down_max = self._get_1up(
                (self.state.first_down_target - self.state.lst)
                / (self.state.orw * self.state.subsequent_target_pct_of_range)
            )

        if self.state.first_up_hit and self.state.up_count < up_max:
            for i in range(self.state.up_count + 1, up_max + 1):
                target_price = self.state.first_up_target + (
                    self.state.orw * self.state.subsequent_target_pct_of_range * i
                )
                self.state.up_targets.append(
                    TargetLevel(price=target_price, label=f"T{i + 1}")
                )
            self.state.up_count = up_max

        if self.state.first_down_hit and self.state.down_count < down_max:
            for i in range(self.state.down_count + 1, down_max + 1):
                target_price = self.state.first_down_target - (
                    self.state.orw * self.state.subsequent_target_pct_of_range * i
                )
                self.state.down_targets.append(
                    TargetLevel(price=target_price, label=f"T{i + 1}")
                )
            self.state.down_count = down_max

        self._check_breakout_signals(close)

    def _check_breakout_signals(self, close: float):
        if close > self.state.orm and not self.state.down_check:
            self.state.down_check = True

        xdown = self._prev_close >= self.state.orl and close < self.state.orl
        xdown2 = self._prev_close >= (
            self.state.orl - self.state.orw * self.state.subsequent_target_pct_of_range
        ) and close < (
            self.state.orl - self.state.orw * self.state.subsequent_target_pct_of_range
        )

        if self.use_bias:
            trigger_down = (self.state.day_dir != 1 and xdown) or (
                self.state.day_dir == 1 and xdown2
            )
        else:
            trigger_down = xdown

        if trigger_down and self.state.down_check:
            self.state.down_signal = True
            self.state.down_check = False
            self.state.breakout_direction = "DOWN"
            logger.info(f"Bearish breakout signal at {close:.2f}")

        if close < self.state.orm and not self.state.up_check:
            self.state.up_check = True

        xup = self._prev_close <= self.state.orh and close > self.state.orh
        xup2 = self._prev_close <= (
            self.state.orh + self.state.orw * self.state.subsequent_target_pct_of_range
        ) and close > (
            self.state.orh + self.state.orw * self.state.subsequent_target_pct_of_range
        )

        if self.use_bias:
            trigger_up = (self.state.day_dir != -1 and xup) or (
                self.state.day_dir == -1 and xup2
            )
        else:
            trigger_up = xup

        if trigger_up and self.state.up_check:
            self.state.up_signal = True
            self.state.up_check = False
            self.state.breakout_direction = "UP"
            logger.info(f"Bullish breakout signal at {close:.2f}")

    def _get_1up(self, val: float) -> int:
        frac = val - int(val)
        if frac > 0:
            return int(val) + 1
        return int(val)

    def update_trade(self, trade: Trade):
        pass

    def get_signals(self) -> Dict[str, float]:
        return self.signals.copy()

    def _update_signals(self):
        self.signals["orh"] = self.state.orh if self.state.orh > 0 else 0.0
        self.signals["orl"] = self.state.orl if self.state.orl < float("inf") else 0.0
        self.signals["orm"] = self.state.orm
        self.signals["orw"] = self.state.orw
        self.signals["prev_orm"] = self.state.prev_orm
        self.signals["day_dir"] = float(self.state.day_dir)

        self.signals["session_active"] = 1.0 if self.state.or_sesh else 0.0
        self.signals["or_token"] = 1.0 if self.state.or_token else 0.0

        self.signals["first_up_target"] = self.state.first_up_target
        self.signals["first_down_target"] = self.state.first_down_target
        self.signals["first_up_hit"] = 1.0 if self.state.first_up_hit else 0.0
        self.signals["first_down_hit"] = 1.0 if self.state.first_down_hit else 0.0

        self.signals["hst"] = self.state.hst
        self.signals["lst"] = self.state.lst
        self.signals["up_count"] = float(self.state.up_count)
        self.signals["down_count"] = float(self.state.down_count)

        if self.state.up_signal:
            self.signals["breakout_signal"] = 1.0
        elif self.state.down_signal:
            self.signals["breakout_signal"] = -1.0
        else:
            self.signals["breakout_signal"] = 0.0

        self.signals["up_targets_json"] = len(self.state.up_targets)
        self.signals["down_targets_json"] = len(self.state.down_targets)

    def is_session_just_completed(self) -> bool:
        return self._session_just_completed

    def get_chart_levels(self) -> Dict:
        return {
            "orh": self.state.orh if self.state.orh > 0 else None,
            "orl": self.state.orl if self.state.orl < float("inf") else None,
            "orm": self.state.orm if self.state.orm > 0 else None,
            "day_dir": self.state.day_dir,
            "up_targets": [
                {"price": t.price, "label": t.label, "hit": t.hit}
                for t in self.state.up_targets
            ],
            "down_targets": [
                {"price": t.price, "label": t.label, "hit": t.hit}
                for t in self.state.down_targets
            ],
            "up_signal": self.state.up_signal,
            "down_signal": self.state.down_signal,
            "session_active": self.state.or_sesh,
        }
