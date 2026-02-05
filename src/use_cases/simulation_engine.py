import logging
import asyncio
import time
from typing import List, Dict, Any, Optional
from dataclasses import replace

from ..domain.interfaces import Strategy, RiskManager, StorageGateway
from ..domain.entities import (
    MarketState,
    Ticker,
    Position,
    Order,
    OrderStatus,
    OrderSide,
)
from ..domain.entities.pnl import SimulationResult, EquitySnapshot, FillEffect, SimStats
from ..domain.history_provider import IHistoryProvider, HistoryConfig
from ..domain.lob_match_engine import LOBMatchEngine
from ..domain.stats_engine import StatsEngine

logger = logging.getLogger(__name__)


class SimulationEngine:
    def __init__(
        self,
        strategy: Strategy,
        risk_manager: RiskManager,
        data_provider: IHistoryProvider,
        history_db: StorageGateway,
        initial_balance: float = 23.0,
        maker_fee: float = -0.0001,
        taker_fee: float = 0.0003,
    ):
        self.strategy = strategy
        self.risk_manager = risk_manager
        self.data_provider = data_provider
        self.history_db = history_db
        self.initial_balance = initial_balance

        self.match_engine = LOBMatchEngine(maker_fee=maker_fee, taker_fee=taker_fee)
        self.stats_engine = StatsEngine()

        self.match_engine.fill_callback = self._on_sim_fill
        self._current_mid = 0.0
        self._current_vamp = None

    async def _on_sim_fill(self, fill: FillEffect):
        self.stats_engine.record_fill(fill, self._current_mid, self._current_vamp)

    async def run_simulation(
        self, symbol: str, start_time: float, end_time: float, venue: str = "bybit"
    ) -> SimulationResult:
        logger.info(f"Starting alpha simulation for {symbol} on {venue}")

        config = HistoryConfig(
            symbol=symbol, venue=venue, start_time=start_time, end_time=end_time
        )

        await self.data_provider.fetch_and_persist(config)

        self.match_engine.balance = self.initial_balance
        equity_curve = []

        async for ticker in self.data_provider.get_tickers(config):
            self._current_mid = (ticker.bid + ticker.ask) / 2

            self.match_engine.on_ticker(ticker)

            market_state = MarketState(ticker=ticker, timestamp=ticker.timestamp)
            pos = Position(
                symbol,
                self.match_engine.position_size,
                self.match_engine.position_entry_price,
            )

            quotes = self.strategy.calculate_quotes(market_state, pos)

            self.match_engine.bid_book.clear()
            self.match_engine.ask_book.clear()
            for q in quotes:
                if self.risk_manager.validate_order(q, pos):
                    self.match_engine.submit_order(q, ticker.timestamp)

            self.stats_engine.update_future_mids(ticker.timestamp, self._current_mid)

            equity = self.match_engine.get_equity(self._current_mid)
            equity_curve.append(
                EquitySnapshot(
                    ticker.timestamp, self.match_engine.balance, 0, equity, 0
                )
            )

        alpha_stats = self.stats_engine.calculate_alpha()

        return SimulationResult(
            run_id=f"alpha_sim_{int(time.time())}",
            start_time=start_time,
            end_time=end_time,
            config={},
            equity_curve=equity_curve,
            fills=self.match_engine.fills,
            stats=alpha_stats,
        )

    def _calculate_stats(
        self, fills: List[FillEffect], equity_curve: List[EquitySnapshot]
    ) -> SimStats:
        if not fills:
            return SimStats(0, 0, 0, 0, 0, 0, 0, 0, 0)

        total_pnl = equity_curve[-1].equity - self.initial_balance
        total_trades = len(fills)
        winning_trades = len([f for f in fills if f.realized_pnl > 0])
        losing_trades = len([f for f in fills if f.realized_pnl < 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0

        max_equity = self.initial_balance
        max_dd = 0.0
        for eq in equity_curve:
            if eq.equity > max_equity:
                max_equity = eq.equity
            dd = (max_equity - eq.equity) / max_equity if max_equity > 0 else 0
            if dd > max_dd:
                max_dd = dd

        return SimStats(
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            total_pnl=total_pnl,
            sharpe_ratio=1.0,
            max_drawdown=max_dd,
            profit_factor=2.0,
            avg_trade_pnl=total_pnl / total_trades if total_trades > 0 else 0,
        )
