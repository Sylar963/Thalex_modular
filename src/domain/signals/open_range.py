import logging
from datetime import datetime
import zoneinfo
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
    timezone: str = "UTC"
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
    session_start_timestamp: Optional[float] = None

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

    # Session MA
    session_ma: float = 0.0
    bars_since_start: int = 0
    ma_length: int = 20


class OpenRangeSignalEngine(SignalEngine):
    def __init__(
        self,
        session_start_utc: str = "20:00",
        session_end_utc: str = "20:15",
        target_pct_from_mid: float = 1.49,
        subsequent_target_pct_of_range: float = 220,
        timezone: str = "UTC",
        use_bias: bool = False,
        validation_threshold: float = 0.20,
        symbol_configs: Optional[Dict[str, Dict]] = None,
    ):
        self.validation_threshold = validation_threshold
        self.symbol_configs = symbol_configs or {}
        self._config = {
            "session_start_hour": int(session_start_utc.split(":")[0]),
            "session_start_minute": int(session_start_utc.split(":")[1])
            if ":" in session_start_utc
            else 0,
            "session_end_hour": int(session_end_utc.split(":")[0]),
            "session_end_minute": int(session_end_utc.split(":")[1])
            if ":" in session_end_utc
            else 0,
            "timezone": timezone,
            "target_pct_from_mid": target_pct_from_mid / 100.0,
            "subsequent_target_pct_of_range": subsequent_target_pct_of_range / 100.0,
        }

        self.states: Dict[str, OpenRangeState] = {}
        self.use_bias = use_bias
        self.signals: Dict[str, Dict[str, float]] = {}
        self._last_closes: Dict[str, float] = {}
        self._last_closes: Dict[str, float] = {}
        self._prev_closes: Dict[str, float] = {}
        self._last_valid_prices: Dict[str, float] = {}
        self._session_just_completed = False

    def _get_state(self, symbol: str) -> OpenRangeState:
        if symbol not in self.states:
            config = self._config.copy()

            # Apply per-symbol overrides
            if symbol in self.symbol_configs:
                sym_cfg = self.symbol_configs[symbol]
                if "target_pct_from_mid" in sym_cfg:
                    config["target_pct_from_mid"] = (
                        sym_cfg["target_pct_from_mid"] / 100.0
                    )
                if "subsequent_target_pct_of_range" in sym_cfg:
                    config["subsequent_target_pct_of_range"] = (
                        sym_cfg["subsequent_target_pct_of_range"] / 100.0
                    )
                # Session times could be overridden here too if needed

            self.states[symbol] = OpenRangeState(**config)
        return self.states[symbol]

    def _is_in_session(self, ts: float, state: OpenRangeState) -> bool:
        tz = zoneinfo.ZoneInfo(state.timezone)
        dt = datetime.fromtimestamp(ts, tz=tz)
        current_minutes = dt.hour * 60 + dt.minute
        start_minutes = state.session_start_hour * 60 + state.session_start_minute
        end_minutes = state.session_end_hour * 60 + state.session_end_minute

        return start_minutes <= current_minutes < end_minutes

    def _get_session_date(self, ts: float, state: OpenRangeState) -> str:
        tz = zoneinfo.ZoneInfo(state.timezone)
        dt = datetime.fromtimestamp(ts, tz=tz)
        return dt.strftime("%Y-%m-%d")

    def _on_or_start(self, symbol: str, date: str, high: float, low: float):
        state = self._get_state(symbol)
        state.prev_orm = state.orm
        state.orm = 0.0

        state.up_targets.clear()
        state.down_targets.clear()

        state.orh = high
        state.orl = low
        state.session_date = date
        # Capture precise start time if available?
        # But date is just YYYY-MM-DD.
        # We need the timestamp of the start of the session.
        # Ideally passed in. But _on_or_start signature is (symbol, date, high, low).
        # We can calculate it from date + config time.

        # Construct localized start time
        try:
            tz = zoneinfo.ZoneInfo(state.timezone)
            # Parse YYYY-MM-DD
            d = datetime.strptime(date, "%Y-%m-%d")
            # Set time
            session_start = d.replace(
                hour=state.session_start_hour,
                minute=state.session_start_minute,
                second=0,
                microsecond=0,
                tzinfo=tz,
            )
            state.session_start_timestamp = session_start.timestamp()
        except Exception as e:
            logger.error(f"Failed to calculate session timestamp: {e}")
            state.session_start_timestamp = None

        state.up_count = 0
        state.down_count = 0
        state.up_check = True
        state.down_check = True
        state.first_up_hit = False
        state.first_down_hit = False
        state.up_signal = False
        state.down_signal = False
        state.breakout_direction = None

        state.or_token = False

        logger.info(f"Open Range session started for {symbol} on {date}")

    def _during_or_session(self, symbol: str, high: float, low: float):
        state = self._get_state(symbol)
        if high > state.orh:
            state.orh = high
        if low < state.orl:
            state.orl = low

        if state.orh != state.orl:
            state.or_token = True

    def _on_or_end(self, symbol: str):
        state = self._get_state(symbol)
        state.orm = (state.orh + state.orl) / 2.0
        state.orw = abs(state.orh - state.orl)

        state.first_up_target = state.orm + (state.orm * state.target_pct_from_mid)
        state.first_down_target = state.orm - (state.orm * state.target_pct_from_mid)

        state.hst = state.first_up_target + (
            state.orw * state.subsequent_target_pct_of_range
        )
        state.lst = state.first_down_target - (
            state.orw * state.subsequent_target_pct_of_range
        )

        if state.prev_orm > 0:
            if state.orm > state.prev_orm:
                state.day_dir = 1
            elif state.orm < state.prev_orm:
                state.day_dir = -1
            else:
                state.day_dir = 0

        state.up_targets.append(TargetLevel(price=state.first_up_target, label="T1"))
        state.down_targets.append(
            TargetLevel(price=state.first_down_target, label="T1")
        )

        self._session_just_completed = True

        logger.info(
            f"OR End for {symbol}: ORH={state.orh:.2f}, ORL={state.orl:.2f}, "
            f"ORM={state.orm:.2f}, ORW={state.orw:.2f}, "
            f"T1_UP={state.first_up_target:.2f}, T1_DN={state.first_down_target:.2f}, "
            f"day_dir={state.day_dir}"
        )

    def _outside_or_session(self, symbol: str, close: float, high: float, low: float):
        state = self._get_state(symbol)
        h_src = close
        l_src = close

        if not state.first_up_hit and h_src > state.first_up_target:
            state.first_up_hit = True
            state.hst = h_src
            state.up_targets[0].hit = True
            logger.info(f"First UP target hit for {symbol} at {h_src:.2f}")

        if not state.first_down_hit and l_src < state.first_down_target:
            state.first_down_hit = True
            state.lst = l_src
            state.down_targets[0].hit = True
            logger.info(f"First DOWN target hit for {symbol} at {l_src:.2f}")

        if state.first_up_hit and h_src > state.hst:
            state.hst = h_src
        if state.first_down_hit and l_src < state.lst:
            state.lst = l_src

        up_max = 0
        down_max = 0
        if state.first_up_hit and state.orw > 0:
            up_max = self._get_1up(
                (state.hst - state.first_up_target)
                / (state.orw * state.subsequent_target_pct_of_range)
            )
        if state.first_down_hit and state.orw > 0:
            down_max = self._get_1up(
                (state.first_down_target - state.lst)
                / (state.orw * state.subsequent_target_pct_of_range)
            )

        if state.first_up_hit and state.up_count < up_max:
            for i in range(state.up_count + 1, up_max + 1):
                target_price = state.first_up_target + (
                    state.orw * state.subsequent_target_pct_of_range * i
                )
                state.up_targets.append(
                    TargetLevel(price=target_price, label=f"T{i + 1}")
                )
            state.up_count = up_max

        if state.first_down_hit and state.down_count < down_max:
            for i in range(state.down_count + 1, down_max + 1):
                target_price = state.first_down_target - (
                    state.orw * state.subsequent_target_pct_of_range * i
                )
                state.down_targets.append(
                    TargetLevel(price=target_price, label=f"T{i + 1}")
                )
            state.down_count = down_max

        self._check_breakout_signals(symbol, close)

    def _check_breakout_signals(self, symbol: str, close: float):
        state = self._get_state(symbol)
        prev_close = self._prev_closes.get(symbol, 0.0)

        if close > state.orm and not state.down_check:
            state.down_check = True

        xdown = prev_close >= state.orl and close < state.orl
        xdown2 = prev_close >= (
            state.orl - state.orw * state.subsequent_target_pct_of_range
        ) and close < (state.orl - state.orw * state.subsequent_target_pct_of_range)

        if self.use_bias:
            trigger_down = (state.day_dir != 1 and xdown) or (
                state.day_dir == 1 and xdown2
            )
        else:
            trigger_down = xdown

        if trigger_down and state.down_check:
            state.down_signal = True
            state.down_check = False
            state.breakout_direction = "DOWN"
            logger.info(f"Bearish breakout signal for {symbol} at {close:.2f}")

        if close < state.orm and not state.up_check:
            state.up_check = True

        xup = prev_close <= state.orh and close > state.orh
        xup2 = prev_close <= (
            state.orh + state.orw * state.subsequent_target_pct_of_range
        ) and close > (state.orh + state.orw * state.subsequent_target_pct_of_range)

        if self.use_bias:
            trigger_up = (state.day_dir != -1 and xup) or (state.day_dir == -1 and xup2)
        else:
            trigger_up = xup

        if trigger_up and state.up_check:
            state.up_signal = True
            state.up_check = False
            state.breakout_direction = "UP"
            logger.info(f"Bullish breakout signal for {symbol} at {close:.2f}")

    def _update_session_ma(self, state: OpenRangeState, price: float):
        # Pine Script logic:
        # bs_nd = fz(ta.barssince(_start)) -> bars since start (1-based)
        # v_len = bs_nd < l ? bs_nd : l -> min(bars, length)
        # EMA: k = 2/(v_len + 1)
        # ma := (s*k) + (nz(ma[1])*(1-k))

        length = min(state.bars_since_start, state.ma_length)
        if length == 0:
            length = 1

        k = 2 / (length + 1)

        if state.session_ma == 0.0:
            state.session_ma = price
        else:
            state.session_ma = (price * k) + (state.session_ma * (1 - k))

    def _get_1up(self, val: float) -> int:
        frac = val - int(val)
        if frac > 0:
            return int(val) + 1
        return int(val)

    async def update_batch(self, tickers: List[Ticker]):
        """Efficiently process a batch of tickers to reconstruct session state."""
        for ticker in tickers:
            self.update(ticker)

    def update_trade(self, trade: Trade):
        """Update using a trade (as a fallback or for precision)."""
        # We can wrap trade in a dummy ticker or handle it directly
        ticker = Ticker(
            symbol=trade.symbol,
            bid=trade.price,
            ask=trade.price,
            bid_size=0,
            ask_size=0,
            last=trade.price,
            volume=trade.size,
            exchange=trade.exchange,
            timestamp=trade.timestamp,
        )
        self.update(ticker)

    def update_candle(
        self,
        symbol: str,
        timestamp: float,
        open_: float,
        high: float,
        low: float,
        close: float,
    ):
        """Update state using a historical candle."""
        self._update_core(symbol, timestamp, close, high, low)

    def _update_core(
        self, symbol: str, ts: float, price: float, high: float, low: float
    ):
        # --- Validation Layer ---
        if price <= 0:
            logger.warning(
                f"Validation Error: Rejecting non-positive price {price} for {symbol}"
            )
            return

        last_valid = self._last_valid_prices.get(symbol)
        if last_valid is not None and last_valid > 0:
            # Check deviation
            deviation = abs(price - last_valid) / last_valid
            if deviation > self.validation_threshold:
                logger.warning(
                    f"Band Collar Triggered: Rejecting price {price} for {symbol} "
                    f"(Last Valid: {last_valid}, Deviation: {deviation:.2%})"
                )
                return

        # Price is valid, update state
        self._last_valid_prices[symbol] = price

        state = self._get_state(symbol)
        self._prev_closes[symbol] = self._last_closes.get(symbol, 0.0)
        self._last_closes[symbol] = price
        self._session_just_completed = False

        current_date = self._get_session_date(ts, state)
        was_in_session = state.or_sesh
        in_session = self._is_in_session(ts, state)

        or_start = in_session and not was_in_session
        or_end = was_in_session and not in_session

        # if current_date != state.session_date:
        #     or_start = True

        if or_start:
            self._on_or_start(symbol, current_date, high, low)
            state.bars_since_start = 0
            state.session_ma = 0.0

        if or_start or (state.or_token and not or_start):
            state.bars_since_start += 1
            self._update_session_ma(state, price)

        if in_session:
            self._during_or_session(symbol, high, low)

        if or_end and state.or_token:
            self._on_or_end(symbol)

        if not in_session and state.or_token:
            self._outside_or_session(symbol, price, high, low)

        state.or_sesh = in_session
        self._update_signals(symbol)

    def update(self, ticker: Ticker):
        ts = ticker.timestamp
        price = ticker.mid_price
        high = ticker.ask if ticker.ask > 0 else price
        low = ticker.bid if ticker.bid > 0 else price

        self._update_core(ticker.symbol, ts, price, high, low)

    def is_session_just_completed(self) -> bool:
        result = self._session_just_completed
        self._session_just_completed = False
        return result

    def get_signals(self) -> Dict[str, float]:
        return self.signals.copy()

    def _update_signals(self, symbol: str):
        state = self._get_state(symbol)
        if symbol not in self.signals:
            self.signals[symbol] = {}

        sig = self.signals[symbol]
        sig["orh"] = state.orh if state.orh > 0 else 0.0
        sig["orl"] = state.orl if state.orl < float("inf") else 0.0
        sig["orm"] = state.orm
        sig["orw"] = state.orw
        sig["prev_orm"] = state.prev_orm
        sig["day_dir"] = float(state.day_dir)

        sig["session_active"] = 1.0 if state.or_sesh else 0.0
        sig["or_token"] = 1.0 if state.or_token else 0.0

        sig["first_up_target"] = state.first_up_target
        sig["first_down_target"] = state.first_down_target
        sig["first_up_hit"] = 1.0 if state.first_up_hit else 0.0
        sig["first_down_hit"] = 1.0 if state.first_down_hit else 0.0

        sig["hst"] = state.hst
        sig["lst"] = state.lst
        sig["up_count"] = float(state.up_count)
        sig["down_count"] = float(state.down_count)

        if state.up_signal:
            sig["breakout_signal"] = 1.0
        elif state.down_signal:
            sig["breakout_signal"] = -1.0
        else:
            sig["breakout_signal"] = 0.0

        sig["up_targets_json"] = len(state.up_targets)
        sig["down_targets_json"] = len(state.down_targets)

    def get_chart_levels(self, symbol: str) -> Dict:
        state = self._get_state(symbol)
        return {
            "orh": state.orh if state.orh > 0 else None,
            "orl": state.orl if state.orl < float("inf") else None,
            "orm": state.orm if state.orm > 0 else None,
            "day_dir": state.day_dir,
            "up_targets": [
                {"price": t.price, "label": t.label, "hit": t.hit}
                for t in state.up_targets
            ],
            "down_targets": [
                {"price": t.price, "label": t.label, "hit": t.hit}
                for t in state.down_targets
            ],
            "up_signal": state.up_signal,
            "down_signal": state.down_signal,
            "session_active": state.or_sesh,
            "session_ma": state.session_ma if state.session_ma > 0 else None,
            "session_start_timestamp": state.session_start_timestamp,
        }
