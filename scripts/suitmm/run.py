import asyncio
import sys
import argparse
import aiohttp
from .data_fetcher import MarketDataFetcher
from .analyzer import MarketAnalyzer
from .visualizer import MMVisualizer
from .report_generator import ReportGenerator


async def analyze_symbol(
    symbol, capital, min_qty_override, fetcher, analyzer, visualizer, reporter
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
        price, tick_size, vol_data[1], vol_profile[0], capital, min_qty
    )

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

    args = parser.parse_args()

    async with aiohttp.ClientSession() as session:
        fetcher = MarketDataFetcher(session)
        analyzer = MarketAnalyzer()
        visualizer = MMVisualizer()
        reporter = ReportGenerator()

        results = {}

        for symbol in args.symbols:
            params = await analyze_symbol(
                symbol.upper(),
                args.capital,
                args.size,
                fetcher,
                analyzer,
                visualizer,
                reporter,
            )
            if params:
                results[symbol] = params

        # CLI Summary
        if len(results) > 0:
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


if __name__ == "__main__":
    asyncio.run(main())
