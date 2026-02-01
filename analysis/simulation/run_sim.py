import asyncio
import logging
import pandas as pd
from datetime import datetime, timedelta

from src.domain.strategies.avellaneda import AvellanedaStoikovStrategy
from src.domain.market.regime_detector import RegimeDetector
from analysis.simulation.pnl_simulator import PNLSimulator

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RunSim")


def main():
    # 1. Initialize Components
    strategy = AvellanedaStoikovStrategy()
    # Mock config for strategy
    strategy.setup(
        {
            "gamma": 0.1,
            "volatility": 0.02,
            "position_fade_time": 30,
            "position_limit": 1.0,
            "base_spread_factor": 1.0,
            "inventory_weight": 0.5,
            "maker_fee_rate": 0.0001,
            "profit_margin_rate": 0.0001,
            "fee_coverage_multiplier": 1.0,
            "min_spread_ticks": 1,
            "volatility_multiplier": 1.0,
            "inventory_factor": 1.0,
            "tick_size": 0.5,
        }
    )

    detector = RegimeDetector(window_size=50)
    sim = PNLSimulator(strategy, detector)

    # 2. Connect to DB and Load Data
    try:
        sim.connect_db(password="password")

        # Load last 1 hour of data for testing
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(hours=1)

        market_data = sim.load_data(
            start_date=start_date.strftime("%Y-%m-%d %H:%M:%S"),
            end_date=end_date.strftime("%Y-%m-%d %H:%M:%S"),
        )

        if market_data.empty:
            logger.warning("No data found for the specified period.")
            return

        logger.info(f"Loaded {len(market_data)} rows of data.")

        # 3. Run Simulation
        sim.run_simulation(market_data)

        # 4. Analyze and Report
        report = sim.analyze_patterns()

        print("\n=== PNL Simulation Report ===")
        print(f"Total PNL: {report['total_pnl']:.4f}")
        print(f"Sharpe Ratio: {report['sharpe_ratio']:.2f}")

        print("\nRegime Performance:")
        for regime, stats in report["regime_performance"].items():
            print(f"  {regime}: Sum PNL={stats['sum']:.4f}, Mean={stats['mean']:.6f}")

        print("\nExpected Move Impact (Bins):")
        for em_bin, impact in report["expected_move_impact"].items():
            print(f"  {em_bin}: Total PNL={impact:.4f}")

    except Exception as e:
        logger.error(f"Simulation failed: {e}")


if __name__ == "__main__":
    main()
