"""HTML report generation with Plotly charts."""

import json
from datetime import datetime

import plotly.graph_objects as go


def _summary_card(results: dict) -> str:
    """Generate the summary card HTML."""
    ap = results.get("access_patterns", {})
    training = results.get("training", {})

    # Random access speedup
    ra = ap.get("random_access", {})
    lance_ra = ra.get("lance", {}).get("wall_clock_mean", 1)
    parquet_ra = ra.get("parquet", {}).get("wall_clock_mean", 1)
    ra_speedup = parquet_ra / lance_ra if lance_ra > 0 else 0

    # PyTorch total speedup
    pt = training.get("pytorch", {})
    lance_pt_total = pt.get("lance", {}).get("total", 0)
    parquet_pt_total = pt.get("parquet", {}).get("total", 0)
    pt_speedup = parquet_pt_total / lance_pt_total if lance_pt_total > 0 else 0

    return f"""
    <div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
                padding: 30px; border-radius: 12px; margin-bottom: 30px; color: white;">
        <h2 style="margin-top: 0;">Summary</h2>
        <div style="display: flex; gap: 40px; flex-wrap: wrap;">
            <div style="text-align: center;">
                <div style="font-size: 48px; font-weight: bold; color: #4ecdc4;">
                    {ra_speedup:.1f}x
                </div>
                <div>Lance faster at random access</div>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 48px; font-weight: bold; color: #4ecdc4;">
                    {pt_speedup:.1f}x
                </div>
                <div>Lance faster for PyTorch training (total)</div>
            </div>
        </div>
    </div>
    """


def _access_pattern_chart(results: dict) -> str:
    """Generate grouped bar chart for access patterns."""
    ap = results.get("access_patterns", {})
    patterns = ["random_access", "sequential_scan", "column_subset"]
    labels = ["Random Access", "Sequential Scan", "Column Subset"]

    lance_times = [
        ap.get(p, {}).get("lance", {}).get("wall_clock_mean", 0) for p in patterns
    ]
    parquet_times = [
        ap.get(p, {}).get("parquet", {}).get("wall_clock_mean", 0) for p in patterns
    ]

    fig = go.Figure(
        data=[
            go.Bar(name="Lance", x=labels, y=lance_times, marker_color="#4ecdc4"),
            go.Bar(name="Parquet", x=labels, y=parquet_times, marker_color="#ff6b6b"),
        ]
    )
    fig.update_layout(
        title="Access Pattern: Wall-Clock Time (lower is better)",
        yaxis_title="Seconds",
        barmode="group",
        template="plotly_dark",
        height=400,
    )
    return fig.to_html(full_html=False, include_plotlyjs=True)


def _memory_chart(results: dict) -> str:
    """Generate memory comparison bar chart."""
    ap = results.get("access_patterns", {})
    patterns = ["random_access", "sequential_scan", "column_subset"]
    labels = ["Random Access", "Sequential Scan", "Column Subset"]

    lance_mem = [
        ap.get(p, {}).get("lance", {}).get("peak_memory_mb", 0) for p in patterns
    ]
    parquet_mem = [
        ap.get(p, {}).get("parquet", {}).get("peak_memory_mb", 0) for p in patterns
    ]

    fig = go.Figure(
        data=[
            go.Bar(name="Lance", x=labels, y=lance_mem, marker_color="#4ecdc4"),
            go.Bar(name="Parquet", x=labels, y=parquet_mem, marker_color="#ff6b6b"),
        ]
    )
    fig.update_layout(
        title="Peak Memory Usage (lower is better)",
        yaxis_title="MB",
        barmode="group",
        template="plotly_dark",
        height=400,
    )
    return fig.to_html(full_html=False, include_plotlyjs=False)


def _cpu_chart(results: dict) -> str:
    """Generate CPU utilization bar chart."""
    ap = results.get("access_patterns", {})
    patterns = ["random_access", "sequential_scan", "column_subset"]
    labels = ["Random Access", "Sequential Scan", "Column Subset"]

    lance_cpu = [
        ap.get(p, {}).get("lance", {}).get("cpu_percent", 0) for p in patterns
    ]
    parquet_cpu = [
        ap.get(p, {}).get("parquet", {}).get("cpu_percent", 0) for p in patterns
    ]

    fig = go.Figure(
        data=[
            go.Bar(name="Lance", x=labels, y=lance_cpu, marker_color="#4ecdc4"),
            go.Bar(name="Parquet", x=labels, y=parquet_cpu, marker_color="#ff6b6b"),
        ]
    )
    fig.update_layout(
        title="CPU Utilization (%)",
        yaxis_title="%",
        barmode="group",
        template="plotly_dark",
        height=400,
    )
    return fig.to_html(full_html=False, include_plotlyjs=False)


def _training_breakdown_chart(results: dict) -> str:
    """Generate stacked bar chart for training step breakdown."""
    training = results.get("training", {})

    xgb_lance = training.get("xgboost", {}).get("lance", {})
    xgb_parquet = training.get("xgboost", {}).get("parquet", {})
    pt_lance = training.get("pytorch", {}).get("lance", {})
    pt_parquet = training.get("pytorch", {}).get("parquet", {})

    categories = [
        "XGBoost (Lance)",
        "XGBoost (Parquet)",
        "PyTorch (Lance)",
        "PyTorch (Parquet)",
    ]

    data_load = [
        xgb_lance.get("data_load", 0),
        xgb_parquet.get("data_load", 0),
        pt_lance.get("data_load", 0),
        pt_parquet.get("data_load", 0),
    ]
    convert = [
        xgb_lance.get("to_dmatrix", 0),
        xgb_parquet.get("to_dmatrix", 0),
        pt_lance.get("dataloader_init", 0),
        pt_parquet.get("dataloader_init", 0),
    ]
    train = [
        xgb_lance.get("train", 0),
        xgb_parquet.get("train", 0),
        pt_lance.get("train_5_epochs", 0),
        pt_parquet.get("train_5_epochs", 0),
    ]

    fig = go.Figure(
        data=[
            go.Bar(name="Data Load", x=categories, y=data_load, marker_color="#4ecdc4"),
            go.Bar(name="Convert/Init", x=categories, y=convert, marker_color="#45b7d1"),
            go.Bar(name="Train", x=categories, y=train, marker_color="#96ceb4"),
        ]
    )
    fig.update_layout(
        title="Training Step Breakdown (seconds)",
        yaxis_title="Seconds",
        barmode="stack",
        template="plotly_dark",
        height=400,
    )
    return fig.to_html(full_html=False, include_plotlyjs=False)


def _storage_table(results: dict) -> str:
    """Generate HTML table for storage efficiency."""
    fs = results.get("file_sizes", {})
    rows = ""
    for key, val in fs.items():
        label = key.replace("_mb", " (MB)").replace("_", " ").title()
        rows += f"<tr><td>{label}</td><td>{val:.2f}</td></tr>"

    return f"""
    <div style="margin: 20px 0;">
        <h3>Storage Efficiency</h3>
        <table style="width: 100%; border-collapse: collapse;
                      background: #1e1e2e; color: white;">
            <thead>
                <tr style="border-bottom: 2px solid #4ecdc4;">
                    <th style="padding: 10px; text-align: left;">Dataset</th>
                    <th style="padding: 10px; text-align: right;">Size (MB)</th>
                </tr>
            </thead>
            <tbody>{rows}</tbody>
        </table>
    </div>
    """


def _environment_section(results: dict) -> str:
    """Generate environment info section."""
    env = results.get("environment", {})
    items = "".join(f"<li><strong>{k}:</strong> {v}</li>" for k, v in env.items())
    return f"""
    <div style="margin: 20px 0; padding: 20px; background: #1e1e2e;
                border-radius: 8px; color: #ccc;">
        <h3>Environment</h3>
        <ul>{items}</ul>
    </div>
    """


def _data_scale_section(results: dict) -> str:
    """Generate data scale info section."""
    ds = results.get("data_scale")
    if not ds:
        return ""
    rows = ""
    for name, info in ds.items():
        label = name.replace("_", " ").title()
        rows += (
            f'<tr><td style="padding: 10px;">{label}</td>'
            f'<td style="padding: 10px; text-align: right;">{info.get("rows", "N/A"):,}</td>'
            f'<td style="padding: 10px; text-align: right;">{info.get("columns", "N/A")}</td></tr>'
        )
    scale_str = ds.get("_scale", "")
    return f"""
    <div style="margin: 20px 0; padding: 20px; background: #1e1e2e;
                border-radius: 8px; color: #ccc;">
        <h3>Data Scale</h3>
        <table style="width: 100%; border-collapse: collapse; color: #ccc;">
            <thead>
                <tr style="border-bottom: 2px solid #4ecdc4;">
                    <th style="padding: 10px; text-align: left;">Dataset</th>
                    <th style="padding: 10px; text-align: right;">Rows</th>
                    <th style="padding: 10px; text-align: right;">Columns</th>
                </tr>
            </thead>
            <tbody>{rows}</tbody>
        </table>
        {"<p style='color: #666; font-size: 0.85em; margin-top: 10px;'>" + scale_str + "</p>" if scale_str else ""}
    </div>
    """


def generate_report(results: dict, output_path: str = "report.html") -> None:
    """Generate a self-contained HTML report with Plotly charts.

    Args:
        results: Full benchmark results dict.
        output_path: Path to write the HTML file.
    """
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Lance vs Parquet Benchmark Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #0d1117;
            color: #e6e6e6;
            max-width: 1200px;
            margin: 0 auto;
            padding: 40px 20px;
        }}
        h1 {{ color: #4ecdc4; border-bottom: 2px solid #4ecdc4; padding-bottom: 10px; }}
        h2 {{ color: #ccc; margin-top: 40px; }}
        .chart-container {{ margin: 30px 0; }}
        .timestamp {{ color: #666; font-size: 0.9em; }}
    </style>
</head>
<body>
    <h1>Lance vs Parquet Benchmark Report</h1>
    <p class="timestamp">Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>

    {_summary_card(results)}

    <h2>Access Pattern Benchmarks</h2>
    <div class="chart-container">{_access_pattern_chart(results)}</div>

    <h2>Memory Usage</h2>
    <div class="chart-container">{_memory_chart(results)}</div>

    <h2>CPU Utilization</h2>
    <div class="chart-container">{_cpu_chart(results)}</div>

    <h2>Training Step Breakdown</h2>
    <div class="chart-container">{_training_breakdown_chart(results)}</div>

    {_storage_table(results)}
    {_data_scale_section(results)}
    {_environment_section(results)}
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)


def save_results_json(results: dict, output_path: str = "results.json") -> None:
    """Save raw benchmark results as JSON.

    Args:
        results: Full benchmark results dict.
        output_path: Path to write the JSON file.
    """
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
