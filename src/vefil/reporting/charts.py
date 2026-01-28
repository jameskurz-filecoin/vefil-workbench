"""Chart generation using Plotly."""

from typing import Any, Dict, List

import plotly.graph_objects as go

from ..engine.accounting import SystemState

# Industrial Design System - sharp, professional, compact
THEME = {
    "text": "#e8eaed",
    "text_secondary": "#9aa0a6",
    "grid": "rgba(30, 33, 36, 0.8)",
    "cyan": "#00d4ff",
    "cyan_fill": "rgba(0, 212, 255, 0.12)",
    "amber": "#ffab00",
    "amber_fill": "rgba(255, 171, 0, 0.12)",
    "red": "#ff5252",
    "red_fill": "rgba(255, 82, 82, 0.12)",
    "green": "#00e676",
    "green_fill": "rgba(0, 230, 118, 0.12)",
    # Legacy aliases for compatibility
    "teal": "#00d4ff",
    "copper": "#ffab00",
    "slate": "#5f6368",
    "ember": "#ff5252",
    "glow": "#ffab00"
}


def apply_dark_layout(fig: go.Figure, title: str, x_title: str, y_title: str, showlegend: bool = True) -> None:
    """Apply industrial dark theme layout for charts - compact and professional."""
    fig.update_layout(
        title={
            "text": title,
            "x": 0,
            "xanchor": "left",
            "font": {"size": 11, "color": THEME["text_secondary"], "family": "Inter, -apple-system, sans-serif"}
        },
        xaxis_title=x_title,
        yaxis_title=y_title,
        hovermode="x unified",
        template="plotly_dark",
        height=340,
        margin=dict(l=50, r=20, t=40, b=40),
        showlegend=showlegend,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            font=dict(size=10, family="Inter, -apple-system, sans-serif"),
            bgcolor="rgba(0,0,0,0)"
        ),
        plot_bgcolor="rgba(8, 9, 10, 1)",
        paper_bgcolor="rgba(8, 9, 10, 1)",
        font={"color": THEME["text"], "family": "Inter, -apple-system, sans-serif", "size": 11},
        xaxis=dict(
            gridcolor=THEME["grid"],
            zerolinecolor=THEME["grid"],
            tickfont=dict(size=10, family="SF Mono, Consolas, monospace"),
            title_font=dict(size=10, color=THEME["text_secondary"])
        ),
        yaxis=dict(
            gridcolor=THEME["grid"],
            zerolinecolor=THEME["grid"],
            tickfont=dict(size=10, family="SF Mono, Consolas, monospace"),
            title_font=dict(size=10, color=THEME["text_secondary"])
        )
    )


def create_supply_chart(states: List[SystemState]) -> go.Figure:
    """Create supply over time chart - industrial style."""
    times = [s.t / 365.25 for s in states]  # Convert to years

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=times,
        y=[s.circulating for s in states],
        name='Circulating',
        mode='lines',
        line=dict(color=THEME["cyan"], width=2),
        fill='tozeroy',
        fillcolor=THEME["cyan_fill"]
    ))

    fig.add_trace(go.Scatter(
        x=times,
        y=[s.locked_vefil for s in states],
        name='Locked',
        mode='lines',
        line=dict(color=THEME["amber"], width=2),
        fill='tonexty',
        fillcolor=THEME["amber_fill"]
    ))

    fig.add_trace(go.Scatter(
        x=times,
        y=[s.reserve for s in states],
        name='Reserve',
        mode='lines',
        line=dict(color=THEME["text_secondary"], width=2, dash='dot')
    ))
    apply_dark_layout(fig, "Supply Distribution", "Time (years)", "FIL")

    return fig


def create_inflation_chart(metrics: List[Dict[str, Any]]) -> go.Figure:
    """Create inflation metrics chart - industrial style."""
    times = [m['t'] / 365.25 for m in metrics]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=times,
        y=[m['net_inflation_rate'] * 100 for m in metrics],
        name='Net Inflation',
        mode='lines',
        line=dict(color=THEME["amber"], width=2)
    ))

    fig.add_trace(go.Scatter(
        x=times,
        y=[m['gross_emission_rate'] * 100 for m in metrics],
        name='Gross Emission',
        mode='lines',
        line=dict(color=THEME["cyan"], width=2, dash='dot')
    ))

    fig.add_trace(go.Scatter(
        x=times,
        y=[m['effective_inflation'] * 100 for m in metrics],
        name='Effective',
        mode='lines',
        line=dict(color=THEME["red"], width=2, dash='dash')
    ))

    # Zero line for reference
    fig.add_hline(y=0, line_dash="solid", line_color=THEME["text_secondary"], line_width=1, opacity=0.5)

    apply_dark_layout(fig, "Inflation Metrics", "Time (years)", "Rate (%)")

    return fig


def create_capital_flow_chart(metrics: List[Dict[str, Any]]) -> go.Figure:
    """Create capital flow chart - industrial style."""
    times = [m['t'] / 365.25 for m in metrics]

    fig = go.Figure()

    if 'new_locks' in metrics[0]:
        fig.add_trace(go.Scatter(
            x=times,
            y=[m['new_locks'] for m in metrics],
            name='Locks',
            mode='lines',
            line=dict(color=THEME["green"], width=2),
            fill='tozeroy',
            fillcolor=THEME["green_fill"]
        ))

    if 'unlocks' in metrics[0]:
        fig.add_trace(go.Scatter(
            x=times,
            y=[m['unlocks'] for m in metrics],
            name='Unlocks',
            mode='lines',
            line=dict(color=THEME["red"], width=2),
            fill='tozeroy',
            fillcolor=THEME["red_fill"]
        ))
    apply_dark_layout(fig, "Capital Flows", "Time (years)", "FIL")

    return fig


def create_reserve_runway_chart(
    states: List[SystemState],
    emission_history: List[float],
    dt_days: float = 30.0
) -> go.Figure:
    """Create reserve runway chart - industrial style."""
    times = [s.t / 365.25 for s in states]

    # Compute runway
    runway_years = []
    for i, state in enumerate(states):
        if i == 0 or i >= len(emission_history) or emission_history[i] <= 0:
            runway = float('inf')
        else:
            annual_emission = emission_history[i] * (365.25 / dt_days) if dt_days > 0 else 0.0
            runway = state.reserve / annual_emission if annual_emission > 0 else float('inf')
        runway_years.append(min(runway, 100) if runway != float('inf') else 100)

    fig = go.Figure()

    # Add colored zones - subtle
    fig.add_hrect(y0=0, y1=5, fillcolor=THEME["red"], opacity=0.06, layer="below", line_width=0)
    fig.add_hrect(y0=5, y1=10, fillcolor=THEME["amber"], opacity=0.06, layer="below", line_width=0)
    fig.add_hrect(y0=10, y1=100, fillcolor=THEME["cyan"], opacity=0.04, layer="below", line_width=0)

    fig.add_trace(go.Scatter(
        x=times,
        y=runway_years,
        name='Runway',
        mode='lines',
        fill='tozeroy',
        line=dict(color=THEME["cyan"], width=2),
        fillcolor=THEME["cyan_fill"]
    ))

    fig.add_hline(y=5, line_dash="dot", line_color=THEME["amber"], line_width=1)
    fig.add_hline(y=2, line_dash="dot", line_color=THEME["red"], line_width=1)

    apply_dark_layout(fig, "Reserve Runway", "Time (years)", "Runway (years)", showlegend=False)
    fig.update_yaxes(range=[0, min(100, max(runway_years) * 1.1)])

    return fig


def create_apy_curve_chart(durations: List[float], apys: List[float]) -> go.Figure:
    """Create APY curve by duration - industrial style."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=durations,
        y=[apy * 100 for apy in apys],
        name='APY',
        mode='lines+markers',
        line=dict(color=THEME["cyan"], width=2),
        marker=dict(size=5, color=THEME["cyan"], line=dict(width=1, color=THEME["text"])),
        fill='tozeroy',
        fillcolor=THEME["cyan_fill"]
    ))
    apply_dark_layout(fig, "APY Curve", "Duration (years)", "APY (%)", showlegend=False)

    return fig


def create_comparison_chart(scenario_names: List[str], metric_values: Dict[str, List[float]]) -> go.Figure:
    """Create comparison chart across scenarios - industrial style."""
    fig = go.Figure()

    colors = [THEME["cyan"], THEME["amber"], THEME["green"], THEME["red"]]
    for i, (metric_name, values) in enumerate(metric_values.items()):
        fig.add_trace(go.Bar(
            x=scenario_names,
            y=values,
            name=metric_name,
            marker_color=colors[i % len(colors)]
        ))

    apply_dark_layout(fig, "Scenario Comparison", "Scenario", "Value")
    fig.update_layout(barmode="group")

    return fig


def create_locked_impact_chart(
    states: List[SystemState],
    metrics: List[Dict[str, Any]]
) -> go.Figure:
    """Create locked supply vs effective inflation overlay chart with sign flip highlights.

    This chart visualizes the relationship between locked supply (primary axis) and
    effective inflation (secondary axis), highlighting inflection points where
    inflation transitions from positive to negative (deflationary regime).
    """
    from plotly.subplots import make_subplots

    min_len = min(len(states), len(metrics))
    states = states[:min_len]
    metrics = metrics[:min_len]

    times = [s.t / 365.25 for s in states]
    locked_values = [s.locked_vefil for s in states]
    total_supply = states[0].total_supply if states else 1
    locked_share = [s.locked_vefil / total_supply * 100 for s in states]

    # Extract effective inflation (annualized percentage)
    effective_inflation = [m.get('effective_inflation', 0) * 100 for m in metrics]

    # Detect sign flips (deflation transition points)
    sign_flips = []
    for i in range(1, len(effective_inflation)):
        prev_val = effective_inflation[i - 1]
        curr_val = effective_inflation[i]
        # Sign flip: positive to negative (entering deflationary regime)
        if prev_val > 0 and curr_val <= 0:
            sign_flips.append({
                'time': times[i],
                'locked': locked_values[i],
                'locked_share': locked_share[i],
                'inflation': curr_val,
                'type': 'deflationary'
            })
        # Sign flip: negative to positive (exiting deflationary regime)
        elif prev_val <= 0 and curr_val > 0:
            sign_flips.append({
                'time': times[i],
                'locked': locked_values[i],
                'locked_share': locked_share[i],
                'inflation': curr_val,
                'type': 'inflationary'
            })

    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Locked Supply (primary y-axis) - area fill
    fig.add_trace(
        go.Scatter(
            x=times,
            y=locked_values,
            name='Locked',
            mode='lines',
            line=dict(color=THEME["amber"], width=2),
            fill='tozeroy',
            fillcolor=THEME["amber_fill"],
            hovertemplate='<b>Locked:</b> %{y:,.0f} FIL<br><b>Time:</b> %{x:.2f} years<extra></extra>'
        ),
        secondary_y=False
    )

    # Effective Inflation (secondary y-axis)
    fig.add_trace(
        go.Scatter(
            x=times,
            y=effective_inflation,
            name='Eff. Inflation',
            mode='lines',
            line=dict(color=THEME["red"], width=2, dash='dash'),
            hovertemplate='<b>Eff. Inflation:</b> %{y:.2f}%<br><b>Time:</b> %{x:.2f} years<extra></extra>'
        ),
        secondary_y=True
    )

    # Add zero line for inflation axis to highlight deflation boundary
    fig.add_hline(
        y=0,
        line_dash="solid",
        line_color=THEME["text_secondary"],
        line_width=1,
        secondary_y=True,
        opacity=0.5
    )

    # Highlight sign flip points
    for flip in sign_flips:
        marker_color = THEME["green"] if flip['type'] == 'deflationary' else THEME["red"]
        annotation_text = "DEF" if flip['type'] == 'deflationary' else "INF"

        # Add marker on locked supply line
        fig.add_trace(
            go.Scatter(
                x=[flip['time']],
                y=[flip['locked']],
                mode='markers+text',
                marker=dict(
                    size=10,
                    color=marker_color,
                    symbol='diamond',
                    line=dict(width=1, color=THEME["text"])
                ),
                text=[annotation_text],
                textposition='top center',
                textfont=dict(color=THEME["text"], size=9, family="Inter, sans-serif"),
                name=f'{annotation_text} Point',
                showlegend=False,
                hovertemplate=(
                    f'<b>{annotation_text} Transition</b><br>'
                    f'Time: {flip["time"]:.2f} years<br>'
                    f'Locked: {flip["locked"]:,.0f} FIL ({flip["locked_share"]:.1f}%)<br>'
                    f'Inflation: {flip["inflation"]:.2f}%<extra></extra>'
                )
            ),
            secondary_y=False
        )

        # Add vertical line at transition point
        fig.add_vline(
            x=flip['time'],
            line_dash="dot",
            line_color=marker_color,
            line_width=1,
            opacity=0.4
        )

    # Shade deflationary regions
    in_deflation = False
    deflation_start = None
    for i, val in enumerate(effective_inflation):
        if val <= 0 and not in_deflation:
            in_deflation = True
            deflation_start = times[i]
        elif val > 0 and in_deflation:
            in_deflation = False
            # Add shaded region
            fig.add_vrect(
                x0=deflation_start,
                x1=times[i],
                fillcolor=THEME["green"],
                opacity=0.05,
                layer="below",
                line_width=0
            )
    # Handle case where deflation continues to end
    if in_deflation and deflation_start is not None:
        fig.add_vrect(
            x0=deflation_start,
            x1=times[-1],
            fillcolor=THEME["green"],
            opacity=0.05,
            layer="below",
            line_width=0
        )

    # Apply layout - industrial style
    fig.update_layout(
        title={
            "text": "Locked Supply vs Effective Inflation",
            "x": 0,
            "xanchor": "left",
            "font": {"size": 11, "color": THEME["text_secondary"], "family": "Inter, -apple-system, sans-serif"}
        },
        hovermode="x unified",
        template="plotly_dark",
        height=340,
        margin=dict(l=50, r=50, t=40, b=70),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            font=dict(size=10, family="Inter, -apple-system, sans-serif"),
            bgcolor="rgba(0,0,0,0)"
        ),
        plot_bgcolor="rgba(8, 9, 10, 1)",
        paper_bgcolor="rgba(8, 9, 10, 1)",
        font={"color": THEME["text"], "family": "Inter, -apple-system, sans-serif", "size": 11}
    )

    # Update axes
    fig.update_xaxes(
        title_text="Time (years)",
        gridcolor=THEME["grid"],
        zerolinecolor=THEME["grid"],
        tickfont=dict(size=10, family="SF Mono, Consolas, monospace"),
        title_font=dict(size=10, color=THEME["text_secondary"])
    )
    fig.update_yaxes(
        title_text="Locked (FIL)",
        gridcolor=THEME["grid"],
        zerolinecolor=THEME["grid"],
        tickfont=dict(size=10, family="SF Mono, Consolas, monospace"),
        title_font=dict(size=10, color=THEME["text_secondary"]),
        secondary_y=False
    )
    fig.update_yaxes(
        title_text="Eff. Inflation (%)",
        gridcolor=THEME["grid"],
        zerolinecolor=THEME["grid"],
        tickfont=dict(size=10, family="SF Mono, Consolas, monospace"),
        title_font=dict(size=10, color=THEME["text_secondary"]),
        secondary_y=True
    )

    return fig
