"""
================================================================================
  dashboard/charts/plotly_charts.py
  Advanced Plotly Analytics Charts for Phase 2 GIS Dashboard
  
  All charts use a consistent dark GIS-intelligence aesthetic.
  Consumes pre-processed data from Phase1DataBridge.
================================================================================
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ── Theme constants ────────────────────────────────────────────────────────────
BG       = "#111827"
BG_PLOT  = "#0d1117"
GRID     = "#1e2d40"
TEXT     = "#8b949e"
TEXT_LT  = "#e2e8f0"
MONO     = "Space Mono, monospace"
CYAN     = "#06b6d4"
BLUE     = "#3b82f6"
GREEN    = "#10b981"
AMBER    = "#f59e0b"
RED      = "#ef4444"
VIOLET   = "#8b5cf6"

_BASE_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor=BG,
    plot_bgcolor=BG_PLOT,
    font=dict(family=MONO, color=TEXT, size=10),
    margin=dict(l=12, r=12, t=36, b=12),
    legend=dict(
        bgcolor=BG, bordercolor=GRID, borderwidth=1,
        font=dict(color=TEXT, family=MONO, size=9)
    ),
    xaxis=dict(gridcolor=GRID, zeroline=False, tickfont=dict(color=TEXT, size=8)),
    yaxis=dict(gridcolor=GRID, zeroline=False, tickfont=dict(color=TEXT, size=8)),
)


def _layout(**overrides) -> dict:
    """Merge base layout with overrides."""
    base = dict(_BASE_LAYOUT)
    base.update(overrides)
    return base


# ─────────────────────────────────────────────────────────────────────────────
#   FLOOD CHARTS
# ─────────────────────────────────────────────────────────────────────────────

def flood_area_donut(metrics: dict, height: int = 320) -> go.Figure:
    """
    Donut chart showing flooded vs non-flooded pixel distribution.
    """
    flooded_pct = metrics["flooded_pct"]
    dry_pct     = round(100 - flooded_pct, 1)

    fig = go.Figure(go.Pie(
        labels=["Flooded", "Non-Flooded"],
        values=[flooded_pct, dry_pct],
        hole=0.62,
        marker=dict(
            colors=[BLUE, "#1e2d40"],
            line=dict(color=BG, width=2)
        ),
        textinfo="label+percent",
        textfont=dict(family=MONO, size=10, color=TEXT_LT),
        hovertemplate="<b>%{label}</b><br>%{value:.1f}%<extra></extra>",
    ))

    # Center annotation
    fig.add_annotation(
        text=f"<b>{flooded_pct:.1f}%</b><br><span style='font-size:9px'>FLOODED</span>",
        x=0.5, y=0.5, showarrow=False,
        font=dict(color=CYAN, family=MONO, size=14),
        align="center"
    )

    fig.update_layout(**_layout(
        title=dict(text="Flood Area Distribution", font=dict(color=TEXT, size=11, family=MONO)),
        height=height,
        showlegend=True,
    ))
    return fig


def flood_backscatter_histogram(vv_db: np.ndarray, threshold_db: float, height: int = 280) -> go.Figure:
    """
    Stacked histogram of SAR VV backscatter values split at threshold.
    """
    flat = vv_db.flatten()
    if len(flat) > 25000:
        idx  = np.random.default_rng(1).choice(len(flat), 25000, replace=False)
        flat = flat[idx]

    flooded_vals  = flat[flat < threshold_db]
    dry_vals      = flat[flat >= threshold_db]

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=flooded_vals, nbinsx=45, name="Flooded",
        marker_color=BLUE, opacity=0.85,
        hovertemplate="dB: %{x:.1f}<br>Count: %{y}<extra></extra>"
    ))
    fig.add_trace(go.Histogram(
        x=dry_vals, nbinsx=45, name="Non-Flooded",
        marker_color="#374151", opacity=0.85,
        hovertemplate="dB: %{x:.1f}<br>Count: %{y}<extra></extra>"
    ))
    fig.add_vline(
        x=threshold_db, line_width=2, line_dash="dash", line_color=AMBER,
        annotation=dict(text=f"Threshold: {threshold_db} dB", font=dict(color=AMBER, size=9),
                        bgcolor=BG, bordercolor=AMBER)
    )
    fig.update_layout(**_layout(
        title=dict(text="SAR VV Backscatter Distribution", font=dict(color=TEXT, size=11, family=MONO)),
        barmode="stack",
        xaxis_title="VV Backscatter (dB)",
        yaxis_title="Pixel Count",
        height=height,
    ))
    return fig


def flood_time_series(flood_ts: pd.DataFrame, height: int = 300) -> go.Figure:
    """
    Interactive time-series of Sentinel-1 flood extent over time.
    Annotates the flood peak.
    """
    peak_idx = flood_ts["flood_pct"].idxmax()
    peak_row = flood_ts.iloc[peak_idx]

    fig = go.Figure()
    # Area fill
    fig.add_trace(go.Scatter(
        x=flood_ts["date"], y=flood_ts["flood_pct"],
        mode="lines", name="Flood Extent (%)",
        line=dict(color=BLUE, width=2.5),
        fill="tozeroy", fillcolor="rgba(59,130,246,0.10)",
        hovertemplate="<b>%{x|%d %b %Y}</b><br>Flood: %{y:.1f}%<extra></extra>"
    ))
    # Area km²
    fig.add_trace(go.Scatter(
        x=flood_ts["date"], y=flood_ts["area_km2"],
        mode="lines", name="Area (km²)",
        line=dict(color=CYAN, width=1.5, dash="dot"),
        yaxis="y2",
        hovertemplate="<b>%{x|%d %b %Y}</b><br>Area: %{y:.0f} km²<extra></extra>"
    ))
    # Peak annotation
    fig.add_annotation(
        x=peak_row["date"], y=peak_row["flood_pct"],
        text=f"⚠ Peak: {peak_row['flood_pct']:.1f}%",
        showarrow=True, arrowhead=2, arrowcolor=RED,
        font=dict(color=RED, family=MONO, size=9),
        bgcolor=BG, bordercolor=RED, borderpad=4
    )

    fig.update_layout(**_layout(
        title=dict(text="Sentinel-1 Flood Extent — Temporal Trend", font=dict(color=TEXT, size=11, family=MONO)),
        yaxis=dict(title="Flood Area (%)", gridcolor=GRID, tickfont=dict(color=TEXT, size=8)),
        yaxis2=dict(
            title="Area (km²)", overlaying="y", side="right",
            gridcolor="transparent", tickfont=dict(color=CYAN, size=8), color=CYAN
        ),
        height=height,
        hovermode="x unified",
    ))
    return fig


def before_after_bar(metrics: dict, before_flooded_pct: float = 5.2, height: int = 260) -> go.Figure:
    """
    Before/After flood comparison bar chart.
    """
    categories = ["Flooded Area (%)", "Risk Score", "Affected Area (km²)"]
    before_vals = [
        before_flooded_pct,
        before_flooded_pct * 3.2 + (1 - 0.6) * 30,
        before_flooded_pct / 100 * 90000
    ]
    after_vals = [
        metrics["flooded_pct"],
        metrics["risk_score"],
        metrics["flooded_km2"]
    ]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Pre-Event", x=categories, y=before_vals,
        marker_color=GREEN, opacity=0.8,
        text=[f"{v:.1f}" for v in before_vals],
        textposition="outside",
        textfont=dict(color=TEXT, size=9),
    ))
    fig.add_trace(go.Bar(
        name="Post-Flood", x=categories, y=after_vals,
        marker_color=BLUE, opacity=0.85,
        text=[f"{v:.1f}" for v in after_vals],
        textposition="outside",
        textfont=dict(color=TEXT, size=9),
    ))

    fig.update_layout(**_layout(
        title=dict(text="Before vs After Flood Comparison", font=dict(color=TEXT, size=11, family=MONO)),
        barmode="group",
        height=height,
    ))
    return fig


# ─────────────────────────────────────────────────────────────────────────────
#   NDVI CHARTS
# ─────────────────────────────────────────────────────────────────────────────

def ndvi_class_donut(metrics: dict, height: int = 320) -> go.Figure:
    """
    Donut chart of vegetation class distribution.
    """
    labels = ["Healthy Vegetation", "Moderate Vegetation", "Low Vegetation"]
    values = [metrics["healthy_pct"], metrics["moderate_pct"], metrics["low_pct"]]
    colors = [GREEN, AMBER, RED]

    fig = go.Figure(go.Pie(
        labels=labels, values=values, hole=0.6,
        marker=dict(colors=colors, line=dict(color=BG, width=2)),
        textinfo="label+percent",
        textfont=dict(family=MONO, size=9, color=TEXT_LT),
        hovertemplate="<b>%{label}</b><br>%{value:.1f}%<extra></extra>",
    ))
    fig.add_annotation(
        text=f"<b>{metrics['healthy_pct']:.1f}%</b><br><span>HEALTHY</span>",
        x=0.5, y=0.5, showarrow=False,
        font=dict(color=GREEN, family=MONO, size=13)
    )
    fig.update_layout(**_layout(
        title=dict(text="Vegetation Health Distribution", font=dict(color=TEXT, size=11, family=MONO)),
        height=height,
    ))
    return fig


def ndvi_histogram(ndvi_arr: np.ndarray, lo: float, hi: float, height: int = 280) -> go.Figure:
    """
    NDVI value histogram split into 3 health classes.
    """
    flat = ndvi_arr[~np.isnan(ndvi_arr)].flatten()
    if len(flat) > 25000:
        flat = flat[np.random.default_rng(2).choice(len(flat), 25000, replace=False)]

    fig = go.Figure()
    for mask, name, color in [
        (flat < lo,                    "Low Veg.",      RED),
        ((flat >= lo) & (flat < hi),   "Moderate Veg.", AMBER),
        (flat >= hi,                   "Healthy Veg.",  GREEN),
    ]:
        fig.add_trace(go.Histogram(
            x=flat[mask], nbinsx=40, name=name,
            marker_color=color, opacity=0.85,
            hovertemplate=f"<b>{name}</b><br>NDVI: %{{x:.3f}}<br>Count: %{{y}}<extra></extra>"
        ))

    for thresh, label, color in [
        (lo, f"Low thresh ({lo})", AMBER),
        (hi, f"High thresh ({hi})", GREEN),
    ]:
        fig.add_vline(
            x=thresh, line_dash="dot", line_color=color, line_width=1.5,
            annotation=dict(text=label, font=dict(color=color, size=8), bgcolor=BG)
        )

    fig.update_layout(**_layout(
        title=dict(text="NDVI Value Distribution", font=dict(color=TEXT, size=11, family=MONO)),
        barmode="stack",
        xaxis_title="NDVI",
        yaxis_title="Pixel Count",
        height=height,
    ))
    return fig


def ndvi_time_series(ndvi_ts: pd.DataFrame, lo: float, hi: float, height: int = 300) -> go.Figure:
    """
    NDVI mean ± IQR time series over the growing season.
    """
    fig = go.Figure()

    # Confidence band
    fig.add_trace(go.Scatter(
        x=list(ndvi_ts["date"]) + list(ndvi_ts["date"][::-1]),
        y=list(ndvi_ts["ndvi_p75"]) + list(ndvi_ts["ndvi_p25"][::-1]),
        fill="toself", fillcolor="rgba(16,185,129,0.10)",
        line=dict(width=0), showlegend=False, hoverinfo="skip"
    ))
    # Mean line
    fig.add_trace(go.Scatter(
        x=ndvi_ts["date"], y=ndvi_ts["ndvi_mean"],
        mode="lines+markers", name="Mean NDVI",
        line=dict(color=GREEN, width=2.5),
        marker=dict(size=5, color=GREEN, symbol="circle"),
        hovertemplate="<b>%{x|%d %b %Y}</b><br>NDVI: %{y:.4f}<extra></extra>"
    ))

    # Threshold lines
    fig.add_hline(y=hi, line_dash="dot", line_color=GREEN, line_width=1.5,
                  annotation=dict(text=f"Healthy ≥{hi}", font=dict(color=GREEN, size=8)))
    fig.add_hline(y=lo, line_dash="dot", line_color=RED, line_width=1.5,
                  annotation=dict(text=f"Low <{lo}", font=dict(color=RED, size=8)))

    # Peak marker
    peak_idx = ndvi_ts["ndvi_mean"].idxmax()
    peak_row = ndvi_ts.iloc[peak_idx]
    fig.add_annotation(
        x=peak_row["date"], y=peak_row["ndvi_mean"],
        text=f"🌿 Peak: {peak_row['ndvi_mean']:.3f}",
        showarrow=True, arrowhead=2, arrowcolor=GREEN,
        font=dict(color=GREEN, family=MONO, size=9),
        bgcolor=BG, bordercolor=GREEN, borderpad=3
    )

    fig.update_layout(**_layout(
        title=dict(text="NDVI Time Series — Crop Growth Cycle (Sentinel-2)",
                   font=dict(color=TEXT, size=11, family=MONO)),
        yaxis=dict(title="NDVI", range=[0, 1], gridcolor=GRID, tickfont=dict(color=TEXT, size=8)),
        height=height,
        hovermode="x unified",
    ))
    return fig


def ndvi_nir_red_scatter(ndvi_result: dict, height: int = 300) -> go.Figure:
    """
    NIR vs RED scatter coloured by NDVI — band correlation analysis.
    """
    nir = ndvi_result["nir"].flatten()[::25]
    red = ndvi_result["red"].flatten()[::25]
    ndv = ndvi_result["ndvi"].flatten()[::25]
    valid = ~np.isnan(ndv)

    fig = go.Figure(go.Scattergl(
        x=red[valid], y=nir[valid],
        mode="markers",
        marker=dict(
            color=ndv[valid], colorscale="RdYlGn",
            cmin=-0.2, cmax=0.8, size=3, opacity=0.6,
            colorbar=dict(
                title="NDVI", thickness=10,
                tickfont=dict(color=TEXT, size=8),
                titlefont=dict(color=TEXT, size=9)
            )
        ),
        hovertemplate="RED: %{x:.4f}<br>NIR: %{y:.4f}<extra></extra>"
    ))

    fig.update_layout(**_layout(
        title=dict(text="NIR vs RED Band Correlation", font=dict(color=TEXT, size=11, family=MONO)),
        xaxis_title="RED Band (B4)", yaxis_title="NIR Band (B8)",
        height=height,
    ))
    return fig


# ─────────────────────────────────────────────────────────────────────────────
#   COMBINED / RISK CHARTS
# ─────────────────────────────────────────────────────────────────────────────

def risk_gauge(risk_score: float, height: int = 280) -> go.Figure:
    """
    Gauge chart showing composite environmental risk score (0–100).
    """
    if risk_score < 25:
        color, label = GREEN, "LOW RISK"
    elif risk_score < 55:
        color, label = AMBER, "MODERATE RISK"
    elif risk_score < 75:
        color, label = "#f97316", "HIGH RISK"
    else:
        color, label = RED, "CRITICAL"

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=risk_score,
        delta=dict(reference=40, increasing=dict(color=RED), decreasing=dict(color=GREEN)),
        gauge=dict(
            axis=dict(range=[0, 100], tickcolor=TEXT, tickfont=dict(color=TEXT, size=9)),
            bar=dict(color=color, thickness=0.25),
            bgcolor=BG_PLOT,
            borderwidth=1,
            bordercolor=GRID,
            steps=[
                dict(range=[0, 25],  color="rgba(16,185,129,0.08)"),
                dict(range=[25, 55], color="rgba(245,158,11,0.08)"),
                dict(range=[55, 75], color="rgba(249,115,22,0.08)"),
                dict(range=[75, 100],color="rgba(239,68,68,0.08)"),
            ],
            threshold=dict(line=dict(color=RED, width=2), thickness=0.75, value=75)
        ),
        title=dict(
            text=f"<b style='color:{color};'>{label}</b><br><span style='font-size:10px;color:{TEXT};'>Composite Risk Index</span>",
            font=dict(family=MONO, size=12)
        ),
        number=dict(font=dict(color=color, family=MONO, size=32), suffix=" / 100"),
    ))
    fig.update_layout(
        paper_bgcolor=BG, font=dict(color=TEXT, family=MONO),
        height=height, margin=dict(l=20, r=20, t=50, b=20)
    )
    return fig


def multi_hazard_timeline(flood_ts: pd.DataFrame, ndvi_ts: pd.DataFrame, height: int = 320) -> go.Figure:
    """
    Dual-axis multi-hazard risk timeline combining flood % and crop stress.
    """
    fig = make_subplots(
        specs=[[{"secondary_y": True}]],
        figure=go.Figure()
    )

    # Flood extent (primary)
    fig.add_trace(go.Scatter(
        x=flood_ts["date"], y=flood_ts["flood_pct"],
        name="Flood Extent (%)", mode="lines",
        line=dict(color=BLUE, width=2),
        fill="tozeroy", fillcolor="rgba(59,130,246,0.07)",
        hovertemplate="%{x|%b %d}<br>Flood: %{y:.1f}%<extra></extra>"
    ), secondary_y=False)

    # NDVI (secondary)
    fig.add_trace(go.Scatter(
        x=ndvi_ts["date"], y=ndvi_ts["ndvi_mean"],
        name="Mean NDVI", mode="lines+markers",
        line=dict(color=GREEN, width=2, dash="dot"),
        marker=dict(size=4),
        hovertemplate="%{x|%b %d}<br>NDVI: %{y:.3f}<extra></extra>"
    ), secondary_y=True)

    fig.update_layout(**_layout(
        title=dict(text="Multi-Hazard Environmental Timeline",
                   font=dict(color=TEXT, size=11, family=MONO)),
        hovermode="x unified",
        height=height,
        legend=dict(bgcolor=BG, bordercolor=GRID, borderwidth=1,
                    font=dict(color=TEXT, size=9))
    ))
    fig.update_yaxes(title_text="Flood Extent (%)", secondary_y=False,
                     gridcolor=GRID, tickfont=dict(color=TEXT, size=8))
    fig.update_yaxes(title_text="NDVI", secondary_y=True,
                     range=[0, 1], gridcolor="transparent",
                     tickfont=dict(color=GREEN, size=8), color=GREEN)

    return fig


def sensor_battery_bar(sensor_df: pd.DataFrame, height: int = 280) -> go.Figure:
    """
    Horizontal bar chart of IoT sensor battery levels.
    """
    df = sensor_df.sort_values("battery_pct")
    colors = [
        GREEN if b >= 70 else (AMBER if b >= 40 else RED)
        for b in df["battery_pct"]
    ]

    fig = go.Figure(go.Bar(
        x=df["battery_pct"], y=df["name"],
        orientation="h",
        marker_color=colors, opacity=0.85,
        text=df["battery_pct"].astype(str) + "%",
        textposition="outside",
        textfont=dict(color=TEXT, size=8),
        hovertemplate="<b>%{y}</b><br>Battery: %{x}%<extra></extra>"
    ))
    fig.add_vline(x=40, line_dash="dot", line_color=RED, line_width=1.2)
    fig.add_vline(x=70, line_dash="dot", line_color=GREEN, line_width=1.2)

    fig.update_layout(**_layout(
        title=dict(text="IoT Sensor Battery Status", font=dict(color=TEXT, size=11, family=MONO)),
        xaxis=dict(range=[0, 115], ticksuffix="%", gridcolor=GRID, tickfont=dict(color=TEXT, size=8)),
        yaxis=dict(tickfont=dict(color=TEXT, size=8)),
        height=height,
    ))
    return fig


def environmental_stats_table(metrics: dict, flood_threshold: float) -> go.Figure:
    """
    Summary statistics table for the analytics panel.
    """
    rows = [
        ["Satellite (Flood)", "Sentinel-1 SAR", "C-band VV, IW Mode"],
        ["Satellite (NDVI)",  "Sentinel-2 MSI", "B4 (Red) + B8 (NIR)"],
        ["SAR Threshold",     f"{flood_threshold} dB", "VV backscatter cutoff"],
        ["Flooded Area",      f"{metrics['flooded_pct']}%", f"~{metrics['flooded_km2']:.0f} km²"],
        ["Mean NDVI",         str(metrics['ndvi_mean']), "Surface vegetation index"],
        ["Healthy Vegetation",f"{metrics['healthy_pct']}%", f"NDVI ≥ {0.5}"],
        ["Moderate Vegetation",f"{metrics['moderate_pct']}%", "0.2 ≤ NDVI < 0.5"],
        ["Low Vegetation",    f"{metrics['low_pct']}%", "NDVI < 0.2"],
        ["Risk Score",        f"{metrics['risk_score']} / 100", "Composite flood+crop index"],
    ]

    fig = go.Figure(go.Table(
        header=dict(
            values=["<b>Parameter</b>", "<b>Value</b>", "<b>Notes</b>"],
            fill_color="#0d1117",
            font=dict(color=CYAN, family=MONO, size=10),
            align="left",
            height=32,
            line_color=GRID
        ),
        cells=dict(
            values=list(zip(*rows)),
            fill_color=[BG, BG_PLOT, BG],
            font=dict(color=[TEXT_LT, CYAN, TEXT], family=MONO, size=9),
            align=["left", "center", "left"],
            height=28,
            line_color=GRID
        )
    ))
    fig.update_layout(
        paper_bgcolor=BG, margin=dict(l=0, r=0, t=0, b=0),
        height=len(rows) * 30 + 50
    )
    return fig
