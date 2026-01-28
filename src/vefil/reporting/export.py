"""Export functionality for CSV, JSON, and HTML."""

import json
from typing import Any, List

import pandas as pd

from ..simulation.runner import SimulationResult


def export_csv(result: SimulationResult, filepath: str):
    """Export simulation results to CSV."""
    # Convert states to DataFrame
    data = []
    for state in result.states:
        data.append({
            't_days': state.t,
            't_years': state.t / 365.25,
            'total_supply': state.total_supply,
            'circulating': state.circulating,
            'locked_vefil': state.locked_vefil,
            'reserve': state.reserve,
            'lending_pool': state.lending_pool,
            'sp_collateral': state.sp_collateral
        })

    # Add metrics
    for i, metrics in enumerate(result.metrics_over_time):
        if i < len(data):
            data[i].update(metrics)

    df = pd.DataFrame(data)
    df.to_csv(filepath, index=False)


def export_json(result: SimulationResult, filepath: str):
    """Export simulation results to JSON."""
    export_data = {
        'config': result.config.to_dict(),
        'config_hash': result.config.compute_hash(),
        'states': [
            {
                't': state.t,
                'total_supply': state.total_supply,
                'circulating': state.circulating,
                'locked_vefil': state.locked_vefil,
                'reserve': state.reserve,
                'lending_pool': state.lending_pool,
                'sp_collateral': state.sp_collateral
            }
            for state in result.states
        ],
        'metrics_over_time': result.metrics_over_time,
        'final_metrics': result.final_metrics
    }

    with open(filepath, 'w') as f:
        json.dump(export_data, f, indent=2)


def export_html_report(result: SimulationResult, filepath: str, charts: List[Any] = None):
    """Export HTML report with charts."""
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>veFIL Simulation Report</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1 {{ color: #333; }}
            .metric {{ margin: 10px 0; padding: 10px; background: #f5f5f5; }}
            .chart {{ margin: 20px 0; }}
        </style>
    </head>
    <body>
        <h1>veFIL Tokenomics Simulation Report</h1>
        
        <div class="metric">
            <h2>Configuration Hash</h2>
            <p>{result.config.compute_hash()}</p>
        </div>
        
        <div class="metric">
            <h2>Final Metrics</h2>
            <ul>
                <li>Final Circulating: {result.final_metrics.get('final_circulating', 0):,.0f} FIL</li>
                <li>Final Locked: {result.final_metrics.get('final_locked', 0):,.0f} FIL</li>
                <li>Final Reserve: {result.final_metrics.get('final_reserve', 0):,.0f} FIL</li>
                <li>Reserve Runway: {result.final_metrics.get('reserve_runway_years', 0):.2f} years</li>
            </ul>
        </div>
        
        <div class="metric">
            <h2>Configuration</h2>
            <pre>{json.dumps(result.config.to_dict(), indent=2)}</pre>
        </div>
    </body>
    </html>
    """

    with open(filepath, 'w') as f:
        f.write(html)
