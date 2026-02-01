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

logger = logging.getLogger(__name__)


class SimulationEngine:
    """
    High-fidelity simulation engine for backtesting market making strategies.
    Uses historical OHLCV data to simulate order matching and PNL.
    """

    def __init__(
        self,
        strategy: Strategy,
        risk_manager: RiskManager,
        data_provider: StorageGateway,
        initial_balance: float = 23.0,  # Target small account
        maker_fee: float = -0.0001,  # Default Thalex maker fee (rebate)
        taker_fee: float = 0.0003,  # Default Thalex taker fee
    ):
        self.strategy = strategy
        self.risk_manager = risk_manager
        self.data_provider = data_provider
        self.initial_balance = initial_balance
        self.maker_fee = maker_fee
        self.taker_fee = taker_fee

    async def run_simulation(
        self, symbol: str, start_time: float, end_time: float, resolution: str = "1m"
    ) -> SimulationResult:
        """
        Run backtest over a historical period.
        """
        logger.info(f"Starting simulation for {symbol} from {start_time} to {end_time}")

        # 1. Fetch Historical Data (OHLCV)
        # Note: In a real "tick-level" sim we'd want trades, but OHLCV is the baseline.
        history = await self.data_provider.get_history(
            symbol, start_time, end_time, resolution
        )

        if not history:
            logger.warning("No historical data found for simulation range.")
            return SimulationResult("empty", start_time, end_time, {})

        # 2. State Initialization
        balance = self.initial_balance
        position = Position(symbol, 0.0, 0.0)
        active_orders: List[Order] = []
        equity_curve: List[EquitySnapshot] = []
        fills: List[FillEffect] = []

        run_id = f"sim_{int(time.time())}"

        # 3. Simulation Loop
        for bar in history:
            bar_time = bar["time"]
            mid_price = bar["close"]  # Use close as proxy for mid in 1m sim

            # A. Match Engine (Pessimistic Fill Model)
            # We assume we get filled if the price (Low for Buys, High for Sells)
            # penetrates our limit price.
            remaining_orders = []
            for order in active_orders:
                is_filled = False
                if order.side == OrderSide.BUY:
                    if bar["low"] <= order.price:
                        is_filled = True
                else:  # SELL
                    if bar["high"] >= order.price:
                        is_filled = True

                if is_filled:
                    # Execute Trade
                    fill_price = order.price
                    size = order.size
                    fee = fill_price * size * self.maker_fee  # Negative if rebate

                    # Update Position
                    old_pos = position.size
                    new_pos = (
                        old_pos + size
                        if order.side == OrderSide.BUY
                        else old_pos - size
                    )

                    realized_pnl = 0.0
                    # Simplified Realized PNL calculation (FIFO-ish logic would be better)
                    if (old_pos > 0 and order.side == OrderSide.SELL) or (
                        old_pos < 0 and order.side == OrderSide.BUY
                    ):
                        # Closing part of position
                        closed_size = min(abs(old_pos), size)
                        # realized = (exit - entry) * size * multiplier
                        # Assuming contract size 1.0
                        dir_mult = 1 if old_pos > 0 else -1
                        realized_pnl = (
                            (fill_price - position.entry_price) * closed_size * dir_mult
                        )

                    # Update Balance & Position
                    balance -= fee  # Subtract fee (or add rebate)
                    # For simplicity, entry price is average price
                    if new_pos == 0:
                        position.entry_price = 0
                    elif abs(new_pos) > abs(old_pos):  # Adding
                        total_val = (
                            abs(old_pos) * position.entry_price + size * fill_price
                        )
                        position.entry_price = total_val / abs(new_pos)

                    position.size = new_pos
                    balance += realized_pnl

                    fills.append(
                        FillEffect(
                            timestamp=bar_time,
                            symbol=symbol,
                            side=order.side.value,
                            price=fill_price,
                            size=size,
                            fee=fee,
                            realized_pnl=realized_pnl,
                            balance_after=balance,
                        )
                    )
                else:
                    remaining_orders.append(order)

            active_orders = remaining_orders

            # B. Strategy Execution
            # Create a Ticker from bar data to feed to strategy
            ticker = Ticker(
                symbol=symbol,
                bid=bar["low"],
                ask=bar["high"],
                bid_size=1.0,
                ask_size=1.0,
                last=bar["close"],
                volume=bar["volume"],
                timestamp=bar_time,
            )
            market_state = MarketState(ticker=ticker, timestamp=bar_time)

            # Strategy calculates desired orders
            desired_orders = self.strategy.calculate_quotes(market_state, position)

            # Risk Management
            valid_orders = []
            # Note: We use remaining active_orders from previous bar match
            for o in desired_orders:
                if self.risk_manager.validate_order(
                    o, position, active_orders=valid_orders
                ):
                    valid_orders.append(o)

            # C. Reconciliation (Simplified for Sim: Cancel all, Place new)
            active_orders = valid_orders

            # D. Equity Snapshot
            unrealized_pnl = (
                (mid_price - position.entry_price) * position.size
                if position.size != 0
                else 0
            )
            position_value = (
                abs(position.size) * mid_price
            )  # Not exactly right for PNL logic but for visualization
            equity = balance + unrealized_pnl

            equity_curve.append(
                EquitySnapshot(
                    timestamp=bar_time,
                    balance=balance,
                    position_value=position_value,
                    equity=equity,
                    unrealized_pnl=unrealized_pnl,
                )
            )

        # 4. Final Statistics
        stats = self._calculate_stats(fills, equity_curve)

        return SimulationResult(
            run_id=run_id,
            start_time=start_time,
            end_time=end_time,
            config=getattr(
                self.strategy, "config", {}
            ),  # Assume strategy has config dict
            equity_curve=equity_curve,
            fills=fills,
            stats=stats,
        )

    def _calculate_stats(
        self, fills: List[FillEffect], equity_curve: List[EquitySnapshot]
    ) -> SimStats:
        if not fills:
            return SimStats(0, 0, 0, 0, 0, 0, 0, 0, 0)

        total_pnl = equity_curve[-1].equity - self.initial_balance
        total_trades = len(fills)

        # Win rate based on fill realized pnl (simplified)
        winning_trades = len([f for f in fills if f.realized_pnl > 0])
        losing_trades = len([f for f in fills if f.realized_pnl < 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0

        # Max Drawdown
        max_equity = self.initial_balance
        max_dd = 0.0
        for eq in equity_curve:
            if eq.equity > max_equity:
                max_equity = eq.equity
            dd = (max_equity - eq.equity) / max_equity if max_equity > 0 else 0
            if dd > max_dd:
                max_dd = dd

        # Sharpe (simplified: mean daily ret / std ret)
        # In a real sim we'd bucket by day

        return SimStats(
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            total_pnl=total_pnl,
            sharpe_ratio=1.0,  # Placeholder
            max_drawdown=max_dd,
            profit_factor=2.0,  # Placeholder
            avg_trade_pnl=total_pnl / total_trades if total_trades > 0 else 0,
        )
