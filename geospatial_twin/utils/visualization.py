"""
================================================================================
  utils/visualization.py
  Reusable Visualization Functions for GeoTwin Dashboard
================================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch, Rectangle
from matplotlib.colorbar import ColorbarBase
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings("ignore")

# ── Dark background style for all matplotlib figures ─────────────────────────
DARK_BG    = "#0d1117"
DARK_PANEL = "#111827"
DARK_GRID  = "#1e2d40"
TEXT_COLOR = "#8b949e"
ACCENT_CYAN  = "#06b6d4"
ACCENT_BLUE  = "#3b82f6"
ACCENT_GREEN = "#10b981"
ACCENT_AMBER = "#f59e0b"
ACCENT_RED   = "#ef4444"


def _apply_dark_style(fig, ax_list):
    """Apply consistent dark theme to a matplotlib figure."""
    fig.patch.set_facecolor(DARK_BG)
    for ax in (ax_list if isinstance(ax_list, (list, tuple)) else [ax_list]):
        ax.set_facecolor(DARK_PANEL)
        ax.tick_params(colors=TEXT_COLOR, labelsize=8)
        ax.xaxis.label.set_color(TEXT_COLOR)
        ax.yaxis.label.set_color(TEXT_COLOR)
        ax.title.set_color(TEXT_COLOR)
        for spine in ax.spines.values():
            spine.set_edgecolor(DARK_GRID)


# ─────────────────────────────────────────────────────────────────────────────
#   FLOOD MAP
# ─────────────────────────────────────────────────────────────────────────────
def plot_flood_map(
    flood_mask: np.ndarray,
    vv_db: np.ndarray,
    large: bool = False
) -> plt.Figure:
    """
    Plot SAR backscatter + flood classification overlay.

    Parameters
    ----------
    flood_mask : bool array
        True = flooded pixels.
    vv_db : float array
        SAR VV backscatter in dB.
    large : bool
        If True, renders at higher DPI / larger size.
    """
    fig_w = 10 if large else 7
    fig_h = 4.5 if large else 3.5

    fig, axes = plt.subplots(1, 2, figsize=(fig_w, fig_h), dpi=110)
    _apply_dark_style(fig, axes)

    # ── Left: SAR backscatter ──
    ax1 = axes[0]
    im1 = ax1.imshow(vv_db, cmap="RdBu_r", vmin=-25, vmax=0, interpolation="bilinear")
    ax1.set_title("SAR VV Backscatter (dB)", fontsize=9, color=TEXT_COLOR, pad=6)
    ax1.axis("off")
    cb1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    cb1.ax.tick_params(colors=TEXT_COLOR, labelsize=7)
    cb1.set_label("dB", color=TEXT_COLOR, fontsize=8)

    # ── Right: Flood classification ──
    ax2 = axes[1]
    # Show SAR as grey background
    ax2.imshow(vv_db, cmap="gray", vmin=-25, vmax=0, alpha=0.5, interpolation="bilinear")
    # Overlay flood mask in blue
    flood_overlay = np.ma.masked_where(~flood_mask, flood_mask)
    ax2.imshow(flood_overlay, cmap=mcolors.ListedColormap([ACCENT_BLUE]),
               alpha=0.75, interpolation="nearest")

    legend_elements = [
        Patch(facecolor=ACCENT_BLUE, alpha=0.75, label="Flooded"),
        Patch(facecolor="#4b5563",   alpha=0.75, label="Non-Flooded"),
    ]
    ax2.legend(handles=legend_elements, loc="lower right",
               facecolor=DARK_BG, edgecolor=DARK_GRID,
               labelcolor=TEXT_COLOR, fontsize=7)
    ax2.set_title("Flood Classification Map", fontsize=9, color=TEXT_COLOR, pad=6)
    ax2.axis("off")

    # Add flood % annotation
    pct = flood_mask.mean() * 100
    ax2.text(0.02, 0.97, f"Flooded: {pct:.1f}%",
             transform=ax2.transAxes, fontsize=8, color=ACCENT_CYAN,
             va="top", fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.3", facecolor=DARK_BG,
                       edgecolor=ACCENT_BLUE, alpha=0.8))

    fig.tight_layout(pad=0.5)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
#   NDVI MAP
# ─────────────────────────────────────────────────────────────────────────────
def plot_ndvi_map(
    ndvi: np.ndarray,
    classification: np.ndarray,
    large: bool = False
) -> plt.Figure:
    """
    Plot NDVI continuous map + vegetation classification.

    Parameters
    ----------
    ndvi : float array
        NDVI values in [-1, 1].
    classification : int array
        0=low, 1=moderate, 2=healthy.
    large : bool
        Larger figure for detail pages.
    """
    fig_w = 10 if large else 7
    fig_h = 4.5 if large else 3.5

    fig, axes = plt.subplots(1, 2, figsize=(fig_w, fig_h), dpi=110)
    _apply_dark_style(fig, axes)

    # ── NDVI colormap (Red→Yellow→Green) ──
    ndvi_cmap = plt.cm.RdYlGn

    # ── Left: Continuous NDVI ──
    ax1 = axes[0]
    ndvi_plot = np.where(np.isnan(ndvi), -0.5, ndvi)
    im1 = ax1.imshow(ndvi_plot, cmap=ndvi_cmap, vmin=-0.2, vmax=0.8,
                     interpolation="bilinear")
    ax1.set_title("NDVI — Continuous", fontsize=9, color=TEXT_COLOR, pad=6)
    ax1.axis("off")
    cb1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    cb1.ax.tick_params(colors=TEXT_COLOR, labelsize=7)
    cb1.set_label("NDVI", color=TEXT_COLOR, fontsize=8)

    # ── Right: Classification ──
    ax2 = axes[1]
    class_colors = [ACCENT_RED, ACCENT_AMBER, ACCENT_GREEN]
    class_cmap   = mcolors.ListedColormap(class_colors)
    ax2.imshow(classification, cmap=class_cmap, vmin=0, vmax=2,
               interpolation="nearest")

    legend_elements = [
        Patch(facecolor=ACCENT_RED,   label="Low Vegetation"),
        Patch(facecolor=ACCENT_AMBER, label="Moderate Vegetation"),
        Patch(facecolor=ACCENT_GREEN, label="Healthy Vegetation"),
    ]
    ax2.legend(handles=legend_elements, loc="lower right",
               facecolor=DARK_BG, edgecolor=DARK_GRID,
               labelcolor=TEXT_COLOR, fontsize=7)
    ax2.set_title("Vegetation Health Classification", fontsize=9, color=TEXT_COLOR, pad=6)
    ax2.axis("off")

    # Annotations
    mean_ndvi = float(np.nanmean(ndvi))
    healthy_pct = float(np.mean(classification == 2)) * 100
    ax2.text(0.02, 0.97, f"Mean NDVI: {mean_ndvi:.3f}\nHealthy: {healthy_pct:.1f}%",
             transform=ax2.transAxes, fontsize=7.5, color=ACCENT_GREEN,
             va="top", fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.3", facecolor=DARK_BG,
                       edgecolor=ACCENT_GREEN, alpha=0.8))

    fig.tight_layout(pad=0.5)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
#   HISTOGRAM PLOTS (Plotly)
# ─────────────────────────────────────────────────────────────────────────────
def plot_backscatter_histogram(
    vv_db: np.ndarray,
    threshold_db: float
) -> go.Figure:
    """
    Interactive histogram of SAR backscatter values with threshold line.
    """
    flat = vv_db.flatten()
    # Subsample for performance
    if len(flat) > 20000:
        idx  = np.random.choice(len(flat), 20000, replace=False)
        flat = flat[idx]

    fig = go.Figure()
    # Flooded portion
    fig.add_trace(go.Histogram(
        x=flat[flat < threshold_db],
        nbinsx=50,
        name="Flooded",
        marker_color=ACCENT_BLUE,
        opacity=0.8
    ))
    # Non-flooded portion
    fig.add_trace(go.Histogram(
        x=flat[flat >= threshold_db],
        nbinsx=50,
        name="Non-Flooded",
        marker_color="#4b5563",
        opacity=0.8
    ))
    # Threshold line
    fig.add_vline(
        x=threshold_db,
        line_width=2, line_dash="dash", line_color=ACCENT_AMBER,
        annotation_text=f"Threshold: {threshold_db} dB",
        annotation_font_color=ACCENT_AMBER,
        annotation_font_size=10
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=DARK_PANEL, plot_bgcolor="#0d1117",
        barmode="stack",
        xaxis_title="VV Backscatter (dB)",
        yaxis_title="Pixel Count",
        legend=dict(font=dict(color=TEXT_COLOR, family="Space Mono", size=10),
                    bgcolor=DARK_PANEL),
        xaxis=dict(gridcolor=DARK_GRID, tickfont=dict(color=TEXT_COLOR, size=9)),
        yaxis=dict(gridcolor=DARK_GRID, tickfont=dict(color=TEXT_COLOR, size=9)),
        margin=dict(l=10, r=10, t=20, b=10),
        height=300
    )
    return fig


def plot_ndvi_histogram(
    ndvi: np.ndarray,
    low_thresh: float,
    high_thresh: float
) -> go.Figure:
    """
    Interactive NDVI distribution histogram with class boundaries.
    """
    flat = ndvi[~np.isnan(ndvi)].flatten()
    if len(flat) > 20000:
        flat = flat[np.random.choice(len(flat), 20000, replace=False)]

    fig = go.Figure()
    for mask, name, color in [
        (flat < low_thresh,                               "Low Veg.",      ACCENT_RED),
        ((flat >= low_thresh) & (flat < high_thresh),     "Moderate Veg.", ACCENT_AMBER),
        (flat >= high_thresh,                             "Healthy Veg.",  ACCENT_GREEN),
    ]:
        fig.add_trace(go.Histogram(
            x=flat[mask], nbinsx=40, name=name,
            marker_color=color, opacity=0.85
        ))

    for thresh, label in [(low_thresh, f"Low thresh ({low_thresh})"),
                          (high_thresh, f"High thresh ({high_thresh})")]:
        fig.add_vline(x=thresh, line_width=2, line_dash="dot",
                      line_color=TEXT_COLOR,
                      annotation_text=label,
                      annotation_font_color=TEXT_COLOR,
                      annotation_font_size=9)

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=DARK_PANEL, plot_bgcolor="#0d1117",
        barmode="stack",
        xaxis_title="NDVI Value",
        yaxis_title="Pixel Count",
        legend=dict(font=dict(color=TEXT_COLOR, family="Space Mono", size=10),
                    bgcolor=DARK_PANEL),
        xaxis=dict(gridcolor=DARK_GRID, tickfont=dict(color=TEXT_COLOR, size=9)),
        yaxis=dict(gridcolor=DARK_GRID, tickfont=dict(color=TEXT_COLOR, size=9)),
        margin=dict(l=10, r=10, t=20, b=10),
        height=300
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
#   PIE CHART
# ─────────────────────────────────────────────────────────────────────────────
def plot_class_distribution_pie(
    mask: np.ndarray,
    labels: list,
    colors: list
) -> go.Figure:
    """Generic pie chart for binary/multi-class masks."""
    if mask.dtype == bool:
        values = [int(mask.sum()), int((~mask).sum())]
    else:
        unique, counts = np.unique(mask, return_counts=True)
        values  = counts.tolist()
        labels  = [str(u) for u in unique]

    fig = go.Figure(go.Pie(
        labels=labels[:len(values)],
        values=values,
        hole=0.55,
        marker=dict(colors=colors[:len(values)],
                    line=dict(color=DARK_PANEL, width=2)),
        textfont=dict(family="Space Mono", size=10, color="#f0f6fc")
    ))
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=DARK_PANEL,
        legend=dict(font=dict(color=TEXT_COLOR, family="Space Mono", size=10)),
        margin=dict(l=10, r=10, t=20, b=10),
        height=300
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
#   TIME SERIES PLOTS
# ─────────────────────────────────────────────────────────────────────────────
def plot_flood_time_series(date_start, date_end) -> go.Figure:
    """
    Simulated Sentinel-1 SAR flood extent time series.
    In production: replace with actual GEE time series values.
    """
    np.random.seed(10)
    start = pd.Timestamp(date_start)
    end   = pd.Timestamp(date_end)
    dates = pd.date_range(start, end, freq="5D")
    n     = len(dates)

    # Simulate a flood event peaking around the middle of the period
    base    = np.linspace(5, 8, n)
    flood_event = 25 * np.exp(-((np.arange(n) - n*0.45)**2) / (2*(n*0.1)**2))
    flood_pct   = base + flood_event + np.random.normal(0, 1.5, n)
    flood_pct   = np.clip(flood_pct, 0, 50)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates, y=flood_pct, mode="lines+markers",
        name="Flood Extent (%)",
        line=dict(color=ACCENT_BLUE, width=2),
        marker=dict(size=4, color=ACCENT_BLUE),
        fill='tozeroy', fillcolor="rgba(59,130,246,0.10)"
    ))
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=DARK_PANEL, plot_bgcolor="#0d1117",
        title=dict(text="Flood Extent — Temporal Trend (Sentinel-1)", 
                   font=dict(color=TEXT_COLOR, size=11, family="Space Mono")),
        xaxis=dict(gridcolor=DARK_GRID, tickfont=dict(color=TEXT_COLOR, size=8)),
        yaxis=dict(gridcolor=DARK_GRID, tickfont=dict(color=TEXT_COLOR, size=8),
                   title="Flooded Area (%)", range=[0, 55]),
        legend=dict(font=dict(color=TEXT_COLOR, size=9)),
        margin=dict(l=10, r=10, t=40, b=10),
        height=280
    )
    return fig


def plot_ndvi_time_series(date_start, date_end) -> go.Figure:
    """
    Simulated Sentinel-2 NDVI mean time series (crop growing season).
    In production: replace with actual GEE time series.
    """
    np.random.seed(20)
    start = pd.Timestamp(date_start)
    end   = pd.Timestamp(date_end)
    dates = pd.date_range(start, end, freq="10D")
    n     = len(dates)

    # Simulate crop growth cycle: slow start → peak → slight decline
    growth = 0.25 + 0.40 * np.sin(np.linspace(0, np.pi, n))**1.5
    noise  = np.random.normal(0, 0.02, n)
    ndvi_vals = np.clip(growth + noise, 0.05, 0.95)

    # Confidence interval
    ci_upper = np.clip(ndvi_vals + 0.06, 0, 1)
    ci_lower = np.clip(ndvi_vals - 0.06, 0, 1)

    fig = go.Figure()
    # CI band
    fig.add_trace(go.Scatter(
        x=list(dates) + list(dates[::-1]),
        y=list(ci_upper) + list(ci_lower[::-1]),
        fill="toself", fillcolor="rgba(16,185,129,0.10)",
        line=dict(width=0), showlegend=False, hoverinfo="skip"
    ))
    # Mean line
    fig.add_trace(go.Scatter(
        x=dates, y=ndvi_vals,
        mode="lines+markers",
        name="Mean NDVI",
        line=dict(color=ACCENT_GREEN, width=2),
        marker=dict(size=4, color=ACCENT_GREEN),
    ))
    # Threshold lines
    fig.add_hline(y=0.5, line_dash="dot", line_color=ACCENT_AMBER, line_width=1,
                  annotation_text="Healthy threshold", annotation_font_size=9,
                  annotation_font_color=ACCENT_AMBER)
    fig.add_hline(y=0.2, line_dash="dot", line_color=ACCENT_RED, line_width=1,
                  annotation_text="Low threshold", annotation_font_size=9,
                  annotation_font_color=ACCENT_RED)

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=DARK_PANEL, plot_bgcolor="#0d1117",
        title=dict(text="NDVI Time Series — Crop Growth Cycle (Sentinel-2)",
                   font=dict(color=TEXT_COLOR, size=11, family="Space Mono")),
        xaxis=dict(gridcolor=DARK_GRID, tickfont=dict(color=TEXT_COLOR, size=8)),
        yaxis=dict(gridcolor=DARK_GRID, tickfont=dict(color=TEXT_COLOR, size=8),
                   title="NDVI", range=[0, 1]),
        legend=dict(font=dict(color=TEXT_COLOR, size=9)),
        margin=dict(l=10, r=10, t=40, b=10),
        height=280
    )
    return fig
