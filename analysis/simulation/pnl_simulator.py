import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import logging

from src.domain.strategies.avellaneda import AvellanedaStoikovStrategy
from src.domain.tracking.position_tracker import PositionTracker, Fill
from src.domain.entities import MarketState, Ticker, Order, OrderSide
from src.domain.market.regime_detector import RegimeDetector

logger = logging.getLogger(__name__)


class PNLSimulator:
    """
    Simulates trading performance under different market regimes.
    """

    def __init__(
        self, strategy: AvellanedaStoikovStrategy, regime_detector: RegimeDetector
    ):
        self.strategy = strategy
        self.regime_detector = regime_detector
        self.tracker = PositionTracker("SIM", "BTC-PERP")
        self.history: List[Dict] = []
        self.db_conn = None

    def connect_db(
        self,
        host="localhost",
        dbname="thalex_trading",
        user="postgres",
        password="password",
    ):
        import psycopg2

        self.db_conn = psycopg2.connect(
            host=host, database=dbname, user=user, password=password
        )

    def load_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Loads historical market data merged with options metrics."""
        query = f"""
            SELECT t.time, t.bid, t.ask, t.last, m.expected_move_pct, m.implied_vol
            FROM market_tickers t
            LEFT JOIN LATERAL (
                SELECT expected_move_pct, implied_vol
                FROM options_live_metrics
                WHERE time <= t.time
                ORDER BY time DESC
                LIMIT 1
            ) m ON TRUE
            WHERE t.time BETWEEN '{start_date}' AND '{end_date}'
            ORDER BY t.time ASC
        """
        return pd.read_sql(query, self.db_conn)

    def run_simulation(self, market_data: pd.DataFrame):
        """
        Runs the strategy against data containing ticker and options metrics.
        """
        logger.info("Starting PNL Simulation...")

        for index, row in market_data.iterrows():
            # 1. Create Market State
            ticker = Ticker(
                symbol="BTC-PERP",
                bid=row["bid"],
                ask=row["ask"],
                bid_size=1.0,
                ask_size=1.0,
                last=row["last"],
                volume=0,
                timestamp=row["time"].timestamp(),
            )
            market_state = MarketState(ticker=ticker, timestamp=ticker.timestamp)

            # 2. Update Regime Detector with Expected Move
            self.regime_detector.set_expected_move(
                em=row["expected_move_pct"],
                is_overpriced=False,  # TODO: Logic to compare IV with RV
            )
            regime = self.regime_detector.update(market_state)

            # 3. Get Quotes
            position = self.tracker.get_position()
            orders = self.strategy.calculate_quotes(
                market_state, position, regime=regime
            )

            # 5. Simulate Fills (Simple assumption: if price crosses quote)
            # This is a simplification; a real backtest needs more complex matching
            self._simulate_execution(orders, ticker)

            # 6. Record State
            self.history.append(
                {
                    "timestamp": row["timestamp"],
                    "mid_price": ticker.mid_price,
                    "pnl": self.tracker.realized_pnl
                    + self.tracker.unrealized_pnl(ticker.mid_price),
                    "position": self.tracker.current_position,
                    "regime": regime.name,
                    "expected_move": regime.expected_move,
                }
            )

        logger.info("Simulation Complete.")

    def _simulate_execution(self, orders: List[Order], ticker: Ticker):
        """
        Checks if orders would be filled against the current ticker.
        """
        for order in orders:
            filled = False
            fill_price = 0.0

            if order.side == OrderSide.BUY and order.price >= ticker.ask:
                filled = True
                fill_price = ticker.ask  # Aggressive fill assumption
            elif order.side == OrderSide.SELL and order.price <= ticker.bid:
                filled = True
                fill_price = ticker.bid

            if filled:
                fill = Fill(
                    id="sim_fill",
                    order_id=order.id,
                    symbol=order.symbol,
                    side=order.side,
                    price=fill_price,
                    size=order.size,
                    fee=0.0,  # Ignore fees for now
                    timestamp=ticker.timestamp,
                )
                self.tracker.update_on_fill(fill)

    def analyze_patterns(self) -> Dict[str, Any]:
        """
        Analyzes PNL performance relative to market regimes and expected move.
        """
        df = self.generate_report()
        if df.empty:
            return {}

        df["pnl_diff"] = df["pnl"].diff().fillna(0)

        # 1. Performance by Regime
        regime_perf = (
            df.groupby("regime")["pnl_diff"]
            .agg(["sum", "mean", "std"])
            .to_dict("index")
        )

        # 2. Performance relative to Expected Move
        # High EM usually means higher risk but also higher potential premium
        em_bins = pd.cut(df["expected_move"], bins=5)
        em_perf = df.groupby(em_bins)["pnl_diff"].sum().to_dict()

        # 3. Sharpe Ratio (Simple)
        sharpe = (
            (df["pnl_diff"].mean() / df["pnl_diff"].std() * np.sqrt(365 * 24 * 12))
            if df["pnl_diff"].std() > 0
            else 0
        )

        return {
            "regime_performance": regime_perf,
            "expected_move_impact": em_perf,
            "total_pnl": df["pnl"].iloc[-1] if not df["pnl"].empty else 0,
            "sharpe_ratio": sharpe,
        }
