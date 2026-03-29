"""All Plotly figure builders for the Streamlit dashboard."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils import REGIME_COLORS

# Shared dark theme for all charts
_DARK = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Consolas, Courier New, monospace", color="#dddddd", size=11),
)
_GRID = "rgba(255,255,255,0.05)"
_TICK = dict(color="#bbbbbb", size=10)


def _regime_runs(dates, states, state_stats):
    """Compute contiguous regime runs as (start_date, end_date, label) tuples."""
    runs = []
    if len(states) == 0:
        return runs
    current_state = states[0]
    start = dates[0]
    for i in range(1, len(states)):
        if states[i] != current_state:
            label = state_stats[current_state]["label"]
            runs.append((start, dates[i - 1], label))
            current_state = states[i]
            start = dates[i]
    label = state_stats[current_state]["label"]
    runs.append((start, dates[-1], label))
    return runs


def make_price_chart(result: dict) -> go.Figure:
    """Price line with regime-colored vertical bands."""
    dates = pd.to_datetime(result["dates"])
    prices = result["prices"]
    states = np.array(result["states"])
    state_stats = result["state_stats"]

    fig = go.Figure()

    # Price line -- bright white, visible
    fig.add_trace(go.Scatter(
        x=dates, y=prices, mode="lines",
        line=dict(color="#ffffff", width=2),
        name="Price",
        hovertemplate="%{x|%b %d, %Y}<br>$%{y:.2f}<extra></extra>",
        fill="tozeroy",
        fillcolor="rgba(255,255,255,0.03)",
    ))

    # Regime bands
    runs = _regime_runs(dates.tolist(), states.tolist(), state_stats)
    for start, end, label in runs:
        color = REGIME_COLORS[label]["plotly"]
        fig.add_vrect(
            x0=start, x1=end,
            fillcolor=color, opacity=0.10, line_width=0,
        )

    # Legend entries for regimes
    added_labels = set()
    for _, _, label in runs:
        if label not in added_labels:
            colors = REGIME_COLORS[label]
            fig.add_trace(go.Scatter(
                x=[None], y=[None], mode="markers",
                marker=dict(size=10, color=colors["plotly"]),
                name=label.capitalize(), showlegend=True,
            ))
            added_labels.add(label)

    fig.update_layout(
        **_DARK,
        title=dict(text=f"{result['ticker']} / Regime-Colored Price History",
                   font=dict(color="#ffffff", size=13)),
        height=520,
        margin=dict(l=10, r=10, t=40, b=50),
        hovermode="x unified",
        yaxis=dict(tickprefix="$", gridcolor=_GRID, tickfont=_TICK, zeroline=False),
        xaxis=dict(gridcolor=_GRID, tickfont=_TICK,
                   rangeslider=dict(visible=False),
                   rangeselector=dict(
                       buttons=[
                           dict(count=6, label="6M", step="month", stepmode="backward"),
                           dict(count=1, label="1Y", step="year", stepmode="backward"),
                           dict(count=2, label="2Y", step="year", stepmode="backward"),
                           dict(step="all", label="ALL"),
                       ],
                       bgcolor="#111111",
                       activecolor="#ff8c00",
                       font=dict(color="#dddddd", size=10),
                   )),
        legend=dict(
            orientation="h",
            yanchor="top", y=-0.08,
            xanchor="left", x=0,
            font=dict(color="#dddddd"),
            bgcolor="rgba(0,0,0,0)",
        ),
        hoverlabel=dict(bgcolor="#111111", font_color="#ffffff", font_size=12),
    )
    return fig


def make_regime_stats_html(result: dict) -> str:
    """HTML stat cards for each regime."""
    state_stats = result["state_stats"]
    html_parts = []
    for state_int in sorted(state_stats.keys(), key=lambda k: state_stats[k]["label"]):
        s = state_stats[state_int]
        label = s["label"]
        colors = REGIME_COLORS[label]
        ret = s["mean_return_annualized"]
        ret_color = "#00d26a" if ret >= 0 else "#ff3b3b"
        html_parts.append(f"""
        <div style="border:1px solid {colors['border']}30; border-left:3px solid {colors['border']};
                    border-radius:4px; padding:12px; margin-bottom:10px;
                    background:{colors['bg']}; font-family:Consolas,monospace">
          <div style="font-weight:600; color:{colors['text']}; margin-bottom:8px;
                      letter-spacing:1px; font-size:13px">
            {label.upper()}
          </div>
          <table style="width:100%; font-size:12px; color:#dddddd">
            <tr><td style="padding:3px 0">Ann. return</td>
                <td style="text-align:right; color:{ret_color}; font-weight:600">{ret:+.1%}</td></tr>
            <tr><td style="padding:3px 0">Volatility</td>
                <td style="text-align:right">{s['volatility_annualized']:.1%}</td></tr>
            <tr><td style="padding:3px 0">Sharpe</td>
                <td style="text-align:right">{s['sharpe']:+.2f}</td></tr>
            <tr><td style="padding:3px 0">% of time</td>
                <td style="text-align:right">{s['pct_time']:.0%}</td></tr>
            <tr><td style="padding:3px 0">Avg duration</td>
                <td style="text-align:right">{s['avg_duration_days']:.0f}d</td></tr>
          </table>
        </div>""")
    return "\n".join(html_parts)


def make_regime_timeline(result: dict) -> go.Figure:
    """Compact heatmap showing regime over time."""
    dates = pd.to_datetime(result["dates"])
    states = np.array(result["states"])
    state_stats = result["state_stats"]

    label_order = {"bear": 0, "volatile": 1, "bull": 2}
    color_vals = [label_order[state_stats[s]["label"]] for s in states]

    colorscale = [
        [0.0, REGIME_COLORS["bear"]["plotly"]],
        [0.5, REGIME_COLORS["volatile"]["plotly"]],
        [1.0, REGIME_COLORS["bull"]["plotly"]],
    ]

    fig = go.Figure(go.Heatmap(
        z=[color_vals],
        x=dates,
        y=["Regime"],
        colorscale=colorscale,
        zmin=0, zmax=2,
        showscale=False,
        hovertemplate="%{x|%b %d, %Y}<extra></extra>",
    ))
    fig.update_layout(
        **_DARK,
        title=dict(text="Regime Timeline", font=dict(color="#ffffff", size=13)),
        height=90,
        margin=dict(l=10, r=10, t=30, b=20),
        yaxis=dict(showticklabels=False),
        xaxis=dict(showgrid=False, tickfont=_TICK),
        hoverlabel=dict(bgcolor="#111111", font_color="#ffffff", font_size=12),
    )
    return fig


def make_transition_heatmap(result: dict) -> go.Figure:
    """Transition probability matrix heatmap."""
    matrix = np.array(result["transition_matrix"])
    state_stats = result["state_stats"]
    n = len(matrix)

    labels = [state_stats[i]["label"].capitalize() for i in range(n)]
    text = [[f"{matrix[i][j]:.0%}" for j in range(n)] for i in range(n)]

    fig = go.Figure(go.Heatmap(
        z=matrix,
        x=labels, y=labels,
        text=text, texttemplate="%{text}",
        textfont=dict(color="#ffffff", size=13),
        colorscale=[[0, "#0a0a0a"], [0.5, "#1a3a5c"], [1, "#ff8c00"]],
        zmin=0, zmax=1,
        showscale=False,
        hovertemplate="From %{y} to %{x}: %{z:.1%}<extra></extra>",
    ))
    fig.update_layout(
        **_DARK,
        title=dict(text="Transition Probabilities", font=dict(color="#ffffff", size=13)),
        height=380,
        margin=dict(l=70, r=10, t=40, b=60),
        xaxis_title="To state",
        yaxis_title="From state",
        xaxis=dict(tickfont=dict(color="#dddddd", size=11), title_font=dict(color="#bbbbbb")),
        yaxis=dict(autorange="reversed", tickfont=dict(color="#dddddd", size=11),
                   title_font=dict(color="#bbbbbb")),
        hoverlabel=dict(bgcolor="#111111", font_color="#ffffff", font_size=12),
    )
    return fig


def make_feature_charts(result: dict) -> go.Figure:
    """Three stacked subplots: log returns, rolling vol, momentum spread."""
    feat = result["features"]
    dates = pd.to_datetime(feat["dates"])
    log_ret = np.array(feat["log_returns"])
    vol = np.array(feat["rolling_vol"])
    mom = np.array(feat["momentum"])

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        subplot_titles=("Log Returns", "Rolling Volatility (21d)", "Momentum Spread (5/21)"),
        vertical_spacing=0.10,
    )

    # Row 1: log returns
    colors = ["#00d26a" if r >= 0 else "#ff3b3b" for r in log_ret]
    fig.add_trace(go.Bar(x=dates, y=log_ret, marker_color=colors, name="Log Returns",
                         opacity=0.85), row=1, col=1)

    # Row 2: rolling volatility
    fig.add_trace(go.Scatter(
        x=dates, y=vol, mode="lines",
        line=dict(color="#ff8c00", width=1.3), name="Volatility",
        fill="tozeroy", fillcolor="rgba(255,140,0,0.05)",
    ), row=2, col=1)

    # Row 3: momentum spread
    fig.add_trace(go.Scatter(
        x=dates, y=mom, mode="lines",
        line=dict(color="#4a9eff", width=1.3), name="Momentum",
    ), row=3, col=1)
    fig.add_hline(y=0, line_dash="dot", line_color="#333333", opacity=0.8, row=3, col=1)

    fig.update_layout(
        **_DARK,
        height=480,
        margin=dict(l=10, r=10, t=40, b=10),
        showlegend=False,
        hoverlabel=dict(bgcolor="#111111", font_color="#ffffff", font_size=12),
    )

    # Style subplot titles
    for ann in fig.layout.annotations:
        ann.font = dict(size=12, color="#ffffff")

    # Style all axes
    for i in range(1, 4):
        fig.update_yaxes(gridcolor=_GRID, tickfont=_TICK, zeroline=False, row=i, col=1)
        fig.update_xaxes(gridcolor=_GRID, tickfont=_TICK, row=i, col=1)

    return fig
