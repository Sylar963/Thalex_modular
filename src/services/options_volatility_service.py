import asyncio
import logging
from datetime import datetime, timedelta
from typing import Optional, Tuple, Dict, Any

from thalex.thalex import Thalex, Network

logger = logging.getLogger(__name__)


class OptionsVolatilityService:
    TARGET_DTE = 3

    def __init__(self, thalex: Thalex, underlying: str = "BTC"):
        self._thalex = thalex
        self._underlying = underlying
        self._cached_em_pct: float = 0.0
        self._cached_atm_iv: float = 0.0
        self._last_fetch_time: float = 0.0
        self._cache_ttl: float = 10.0

    async def get_expected_move(self) -> Tuple[float, float]:
        now = asyncio.get_event_loop().time()
        if now - self._last_fetch_time < self._cache_ttl:
            return self._cached_em_pct, self._cached_atm_iv

        try:
            em_pct, atm_iv = await self._fetch_3dte_straddle()
            self._cached_em_pct = em_pct
            self._cached_atm_iv = atm_iv
            self._last_fetch_time = now
        except Exception as e:
            logger.warning(f"OptionsVolatilityService fetch failed: {e}")

        return self._cached_em_pct, self._cached_atm_iv

    async def _fetch_3dte_straddle(self) -> Tuple[float, float]:
        index_price = await self._get_index_price()
        if index_price == 0.0:
            return 0.0, 0.0

        call_instr, put_instr = await self._find_atm_options(index_price)
        if not call_instr or not put_instr:
            return 0.0, 0.0

        call_ticker = await self._thalex.ticker(
            instrument_name=call_instr["instrument_name"]
        )
        put_ticker = await self._thalex.ticker(
            instrument_name=put_instr["instrument_name"]
        )

        c_mark = float(call_ticker["result"]["mark_price"])
        p_mark = float(put_ticker["result"]["mark_price"])
        c_iv = float(call_ticker["result"].get("mark_iv", 0.0))

        straddle_price = c_mark + p_mark
        em_pct = straddle_price / index_price if index_price > 0 else 0.0

        return em_pct, c_iv

    async def _get_index_price(self) -> float:
        try:
            ticker = await self._thalex.ticker(
                instrument_name=f"{self._underlying}-PERP"
            )
            if not ticker or "result" not in ticker:
                return 0.0
            return float(ticker["result"]["index_price"])
        except Exception as e:
            logger.error(f"Failed to fetch index price: {e}")
            return 0.0

    async def _find_atm_options(
        self, index_price: float
    ) -> Tuple[Optional[Dict], Optional[Dict]]:
        instruments = await self._thalex.public_instruments()
        options = [
            i
            for i in instruments["result"]
            if i.get("base_currency") == self._underlying and i.get("type") == "option"
        ]

        if not options:
            return None, None

        now = datetime.utcnow()
        target_ts = (now + timedelta(days=self.TARGET_DTE)).timestamp()

        expiries = sorted(set(o["expiration_timestamp"] for o in options))
        closest_expiry = min(expiries, key=lambda x: abs(x - target_ts))

        expiry_options = [
            o for o in options if o["expiration_timestamp"] == closest_expiry
        ]

        strikes = sorted(set(o["strike_price"] for o in expiry_options))
        atm_strike = min(strikes, key=lambda x: abs(x - index_price))

        call_instr = next(
            (
                o
                for o in expiry_options
                if o["strike_price"] == atm_strike and o["option_type"] == "call"
            ),
            None,
        )
        put_instr = next(
            (
                o
                for o in expiry_options
                if o["strike_price"] == atm_strike and o["option_type"] == "put"
            ),
            None,
        )

        return call_instr, put_instr
