import asyncio
import sys
import argparse
import aiohttp
from .data_fetcher import MarketDataFetcher
from .analyzer import MarketAnalyzer
from .performance_analyzer import PerformanceAnalyzer
from .visualizer import MMVisualizer
from .report_generator import ReportGenerator


async def analyze_symbol(
    symbol,
    capital,
    min_qty_override,
    fetcher,
    analyzer,
    visualizer,
    reporter,
    spread_factor=0.15,
    dump_config=False,
):
    print(f"\nAnalyzing {symbol}...")

    # 1. Fetch Data
    ticker_data, klines, ob, inst_info = await asyncio.gather(
        fetcher.fetch_ticker(symbol),
        fetcher.fetch_klines(symbol, interval="60", limit=336),  # 14 days of hourly
        fetcher.fetch_orderbook(symbol),
        fetcher.fetch_instrument_info(symbol),
    )

    if not ticker_data or not inst_info:
        print(f"Skipping {symbol}: Missing data")
        return None

    # 2. Parse Basic Info
    price = float(ticker_data["lastPrice"])
    tick_size = float(inst_info["priceFilter"]["tickSize"])
    min_qty = float(inst_info["lotSizeFilter"]["minOrderQty"])

    if min_qty_override:
        print(f"  Overriding min_qty {min_qty} -> {min_qty_override}")
        min_qty = float(min_qty_override)

    # 3. Analyze
    vol_data = analyzer.calculate_volatility(klines)
    vol_profile = analyzer.calculate_volume_profile(klines)
    ob_analysis = analyzer.analyze_orderbook(ob)

    params = analyzer.recommend_params(
        price,
        tick_size,
        vol_data[1],
        vol_profile[0],
        capital,
        min_qty,
        vol_spread_factor=spread_factor,
    )

    if dump_config:
        import json

        s_conf = {
            "type": "avellaneda",
            "params": {
                "gamma": params["gamma"],
                "volatility": round(vol_data[1], 4),
                "position_limit": float(params["position_limit"]),
                "order_size": float(params["order_size"]),
                "min_spread": int(params["min_spread_ticks"]),
                "quote_levels": int(params["quote_levels"]),
                "level_spacing_factor": float(params["level_spacing_factor"]),
                "recalc_interval": 0.5,
            },
        }
        print(f"\n--- CONFIG SNIPPET FOR {symbol} ---")
        print(json.dumps(s_conf, indent=4))
        print("-----------------------------------\n")

    # 4. Heatmap Data
    matrix, spreads, sizes = analyzer.generate_profitability_matrix(
        price, tick_size, vol_data[1], min_qty, capital
    )

    # 5. Visualize
    figs = []

    # Heatmap
    figs.append(visualizer.plot_profitability_heatmap(matrix, symbol))

    # Volatility Cone
    figs.append(visualizer.plot_volatility_cone(klines, symbol))

    # Depth Profile
    figs.append(visualizer.plot_depth_profile(ob, symbol))

    # 6. Generate Report
    filename = f"analysis_{symbol}.html"
    reporter.generate_report(symbol, params, figs, filename)

    return params


async def analyze_performance(symbol, fetcher, perf_analyzer, visualizer, reporter):
    print(f"\nAnalyzing Performance for {symbol}...")

    # 1. Fetch Fills (DB)
    fills = await fetcher.fetch_fills_from_db(symbol)
    if not fills:
        print(f"No fills found for {symbol} in database.")
        return None

    # 2. Fetch Market Data and Indicators
    # Fetch klines for context (last 14 days hourly)
    # Fetch indicators (last 14 days)
    klines, indicators = await asyncio.gather(
        fetcher.fetch_klines(symbol, interval="60", limit=336),
        fetcher.fetch_indicators(symbol, days=14),
    )

    # 3. Analyze
    pnl_df = perf_analyzer.calculate_pnl_series(fills)
    stats = perf_analyzer.calculate_trade_stats(fills)

    # Use klines for markout? 60m klines might be too coarse for 1m markout
    # But for visual context they are fine.
    # Ideally we'd fetch 1m klines, but let's use what we have for now.
    markouts = perf_analyzer.calculate_markout(fills, klines)

    ind_dfs = perf_analyzer.prepare_indicator_data(indicators)

    # 4. Visualize
    figs = []

    # Cumulative PnL
    figs.append(visualizer.plot_cumulative_pnl(pnl_df, symbol))

    # Execution Map
    figs.append(visualizer.plot_trade_executions(klines, pnl_df, symbol))

    # Indicator Correlation
    figs.append(visualizer.plot_indicators(ind_dfs, symbol))

    # Execution Quality
    figs.append(visualizer.plot_markout_distribution(markouts, symbol))

    # 5. Generate Report
    filename = f"performance_{symbol}.html"
    reporter.generate_performance_report(symbol, stats, figs, filename)

    return stats


async def main():
    parser = argparse.ArgumentParser(
        description="SuitMM: Visual Market Making Analysis Suite"
    )
    parser.add_argument("symbols", nargs="+", help="Symbols to analyze (e.g. BTCUSDT)")
    parser.add_argument(
        "--capital", type=float, default=233.0, help="Trading capital in USDT"
    )
    parser.add_argument(
        "--size", type=float, help="Override minimum order size (e.g. 0.1)"
    )
    parser.add_argument(
        "--spread-factor",
        type=float,
        default=0.15,
        help="Volatility spread factor (default 0.15). Lower for tighter spreads.",
    )
    parser.add_argument(
        "--dump-config",
        action="store_true",
        help="Dump config.json snippet for the strategy.",
    )
    parser.add_argument(
        "--mode",
        choices=["analyze", "performance"],
        default="analyze",
        help="Mode: 'analyze' for market feasibility, 'performance' for bot stats",
    )

    args = parser.parse_args()

    async with aiohttp.ClientSession() as session:
        fetcher = MarketDataFetcher(session)
        analyzer = MarketAnalyzer()
        perf_analyzer = PerformanceAnalyzer()
        visualizer = MMVisualizer()
        reporter = ReportGenerator()

        results = {}

        for symbol in args.symbols:
            if args.mode == "analyze":
                res = await analyze_symbol(
                    symbol.upper(),
                    args.capital,
                    args.size,
                    fetcher,
                    analyzer,
                    visualizer,
                    reporter,
                    spread_factor=args.spread_factor,
                    dump_config=args.dump_config,
                )
            else:
                res = await analyze_performance(
                    symbol.upper(), fetcher, perf_analyzer, visualizer, reporter
                )

            if res:
                results[symbol] = res

        # CLI Summary
        if len(results) > 0 and args.mode == "analyze":
            print(f"\n{'=' * 70}")
            print(f"  COMPARISON RANKING (by RTs needed for 2%/day)")
            print(f"{'=' * 70}")
            ranked = sorted(results.items(), key=lambda x: x[1]["rts_for_2pct"])
            for i, (sym, p) in enumerate(ranked, 1):
                feasibility = (
                    "✅"
                    if p["rts_per_hour_needed"] < 10
                    else "⚠️ "
                    if p["rts_per_hour_needed"] < 30
                    else "❌"
                )
                print(
                    f"  {i}. {feasibility} {sym:15s} | {p['rts_for_2pct']:5d} RTs/day | "
                    f"${p['profit_per_rt']:.4f}/RT | Spread: {p['spread_pct']:.3f}%"
                )
            print(f"{'=' * 70}\n")
            print("Check generated .html files for visual reports.")
        elif len(results) > 0 and args.mode == "performance":
            print(f"\n{'=' * 70}")
            print("  PERFORMANCE SUMMARY")
            print(f"{'=' * 70}")
            for sym, stats in results.items():
                pnl = stats.get("Total PnL", 0)
                wr = stats.get("Win Rate", 0) * 100
                print(f"  {sym:15s} | PnL: ${pnl:.4f} | Win Rate: {wr:.1f}%")
            print(f"{'=' * 70}\n")
            print("Check generated performance_*.html files.")


if __name__ == "__main__":
    asyncio.run(main())
