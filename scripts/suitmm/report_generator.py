import json
import plotly.io as pio
from datetime import datetime


class ReportGenerator:
    def generate_report(self, symbol, params, visual_figs, output_path):
        """
        Generates HTML report.
        params: Dict from analyzer.recommend_params
        visual_figs: List of Plotly Figures
        """

        # Convert figures to HTML divs
        charts_html = ""
        for fig in visual_figs:
            charts_html += pio.to_html(fig, include_plotlyjs="cdn", full_html=False)
            charts_html += "<br><hr><br>"

        # Format params for display
        feasibility_color = (
            "green"
            if params["rts_per_hour_needed"] < 10
            else "orange"
            if params["rts_per_hour_needed"] < 30
            else "red"
        )
        feasibility_text = (
            "FEASIBLE"
            if params["rts_per_hour_needed"] < 10
            else "MARGINAL"
            if params["rts_per_hour_needed"] < 30
            else "UNLIKELY"
        )

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>MM Analysis: {symbol}</title>
            <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
            <style>
                body {{ background-color: #1e1e1e; color: #f0f0f0; }}
                .card {{ background-color: #2d2d2d; border: 1px solid #444; margin-bottom: 20px; }}
                .metric-label {{ color: #aaa; font-size: 0.9em; }}
                .metric-value {{ font-size: 1.2em; font-weight: bold; }}
                .highlight {{ color: {feasibility_color}; }}
            </style>
        </head>
        <body>
            <div class="container mt-4">
                <h1 class="mb-4">Market Making Analysis: <span class="text-primary">{symbol}</span></h1>
                <p class="text-muted">Generated at {timestamp}</p>
                
                <div class="row">
                    <div class="col-md-4">
                        <div class="card p-3">
                            <div class="metric-label">Feasibility</div>
                            <div class="metric-value highlight">{feasibility_text}</div>
                        </div>
                    </div>
                    <div class="col-md-4">
                        <div class="card p-3">
                            <div class="metric-label">Profit Per Round Trip</div>
                            <div class="metric-value text-success">${params["profit_per_rt"]:.4f}</div>
                        </div>
                    </div>
                    <div class="col-md-4">
                        <div class="card p-3">
                            <div class="metric-label">RTs Needed (2% Returns)</div>
                            <div class="metric-value">{params["rts_for_2pct"]} / day</div>
                            <div class="small text-muted">({params["rts_per_hour_needed"]} / hour)</div>
                        </div>
                    </div>
                </div>

                <div class="row">
                     <div class="col-md-6">
                        <div class="card p-3">
                            <h5>Recommended Parameters</h5>
                            <table class="table table-dark table-sm">
                                <tr><td>Min Spread</td><td>{params["min_spread_ticks"]} ticks (${params["spread_dollars"]:.4f})</td></tr>
                                <tr><td>Order Size</td><td>{params["order_size"]}</td></tr>
                                <tr><td>Position Limit</td><td>{params["position_limit"]:.1f} units</td></tr>
                                <tr><td>Gamma</td><td>{params["gamma"]}</td></tr>
                                <tr><td>Quote Levels</td><td>{params["quote_levels"]}</td></tr>
                            </table>
                        </div>
                    </div>
                    <div class="col-md-6">
                        <div class="card p-3">
                           <h5>Config Snippet (JSON)</h5>
                           <pre class="pre-scrollable text-white">
{{
    "gamma": {params["gamma"]},
    "position_limit": {params["position_limit"]},
    "min_spread": {params["min_spread_ticks"]},
    "quote_levels": {params["quote_levels"]},
    "order_size": {params["order_size"]}
}}
                           </pre>
                        </div>
                    </div>
                </div>
                
                <div class="row">
                    <div class="col-12">
                        <div class="card p-3">
                            <h3 class="mb-3">Visual Analysis</h3>
                            {charts_html}
                        </div>
                    </div>
                </div>
            </div>
        </body>
        </html>
        """

        with open(output_path, "w") as f:
            f.write(html_content)

        print(f"Report saved to: {output_path}")
