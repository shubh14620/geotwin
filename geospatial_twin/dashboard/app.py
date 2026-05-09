"""
================================================================================
  AI-Driven Geospatial Digital Twin for Multi-Hazard Environmental Intelligence
  Phase 1 Dashboard — app.py
  Author: B.Tech Major Project
  Tech Stack: Streamlit · Google Earth Engine · Sentinel-1/2 · Python
================================================================================
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import sys
import os

# ── Path setup so we can import sibling modules ──────────────────────────────
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from flood_detection.flood_processor import FloodProcessor
from ndvi_monitoring.ndvi_processor import NDVIProcessor
from utils.demo_data import generate_demo_sar, generate_demo_multispectral
from utils.visualization import (
    plot_flood_map, plot_ndvi_map,
    plot_backscatter_histogram, plot_ndvi_histogram,
    plot_flood_time_series, plot_ndvi_time_series,
    plot_class_distribution_pie
)

# ── Page Configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GeoTwin · Environmental Intelligence",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS — Futuristic Dark Theme ────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Rajdhani:wght@300;400;500;600;700&family=Orbitron:wght@400;700;900&display=swap');

/* ── Root Variables ── */
:root {
    --bg-deep:      #030712;
    --bg-panel:     #0d1117;
    --bg-card:      #111827;
    --bg-card2:     #161d2e;
    --border:       #1e2d40;
    --border-glow:  #0ea5e9;
    --accent-cyan:  #06b6d4;
    --accent-blue:  #3b82f6;
    --accent-green: #10b981;
    --accent-amber: #f59e0b;
    --accent-red:   #ef4444;
    --accent-violet:#8b5cf6;
    --text-primary: #f0f6fc;
    --text-secondary:#8b949e;
    --text-muted:   #484f58;
    --font-display: 'Orbitron', monospace;
    --font-ui:      'Rajdhani', sans-serif;
    --font-mono:    'Space Mono', monospace;
}

/* ── Global Reset ── */
html, body, [class*="css"] {
    background-color: var(--bg-deep) !important;
    color: var(--text-primary) !important;
    font-family: var(--font-ui) !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1117 0%, #060b13 100%) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text-primary) !important; }

/* ── Main Header ── */
.hero-header {
    background: linear-gradient(135deg, #0d1117 0%, #0a1628 50%, #060f1e 100%);
    border: 1px solid var(--border);
    border-top: 2px solid var(--accent-cyan);
    border-radius: 12px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.hero-header::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    background:
        radial-gradient(ellipse at 20% 50%, rgba(6,182,212,0.08) 0%, transparent 60%),
        radial-gradient(ellipse at 80% 20%, rgba(59,130,246,0.06) 0%, transparent 50%);
    pointer-events: none;
}
.hero-title {
    font-family: var(--font-display) !important;
    font-size: 1.4rem !important;
    font-weight: 700 !important;
    color: var(--accent-cyan) !important;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin-bottom: 0.3rem;
    text-shadow: 0 0 20px rgba(6,182,212,0.4);
}
.hero-subtitle {
    font-family: var(--font-ui) !important;
    font-size: 1rem !important;
    color: var(--text-secondary) !important;
    letter-spacing: 0.08em;
    font-weight: 400;
}
.hero-phase {
    display: inline-block;
    background: linear-gradient(90deg, rgba(6,182,212,0.15), rgba(59,130,246,0.15));
    border: 1px solid rgba(6,182,212,0.3);
    border-radius: 4px;
    padding: 0.2rem 0.7rem;
    font-family: var(--font-mono) !important;
    font-size: 0.7rem !important;
    color: var(--accent-cyan) !important;
    letter-spacing: 0.15em;
    margin-top: 0.8rem;
}
.hero-badges {
    display: flex;
    gap: 0.5rem;
    flex-wrap: wrap;
    margin-top: 1rem;
}
.badge {
    display: inline-block;
    border-radius: 4px;
    padding: 0.15rem 0.6rem;
    font-family: var(--font-mono) !important;
    font-size: 0.65rem !important;
    letter-spacing: 0.1em;
    font-weight: 700;
}
.badge-cyan  { background:rgba(6,182,212,0.12);  border:1px solid rgba(6,182,212,0.35);  color: var(--accent-cyan) !important; }
.badge-blue  { background:rgba(59,130,246,0.12); border:1px solid rgba(59,130,246,0.35); color: var(--accent-blue) !important; }
.badge-green { background:rgba(16,185,129,0.12); border:1px solid rgba(16,185,129,0.35); color: var(--accent-green) !important; }
.badge-amber { background:rgba(245,158,11,0.12); border:1px solid rgba(245,158,11,0.35); color: var(--accent-amber) !important; }

/* ── Section Headers ── */
.section-header {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    border-bottom: 1px solid var(--border);
    padding-bottom: 0.75rem;
    margin-bottom: 1.2rem;
}
.section-icon {
    width: 36px; height: 36px;
    border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.1rem;
}
.icon-flood  { background: linear-gradient(135deg,rgba(59,130,246,0.2),rgba(6,182,212,0.1)); border:1px solid rgba(59,130,246,0.3); }
.icon-ndvi   { background: linear-gradient(135deg,rgba(16,185,129,0.2),rgba(6,182,212,0.1)); border:1px solid rgba(16,185,129,0.3); }
.section-title {
    font-family: var(--font-display) !important;
    font-size: 0.9rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}

/* ── Metric Cards ── */
.metric-grid { display:grid; grid-template-columns:repeat(auto-fit, minmax(140px,1fr)); gap:0.75rem; margin-bottom:1.2rem; }
.metric-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1rem;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s;
}
.metric-card:hover { border-color: var(--border-glow); }
.metric-card::before {
    content:'';
    position:absolute;
    top:0; left:0; right:0;
    height:2px;
}
.metric-card.cyan::before  { background: linear-gradient(90deg, var(--accent-cyan), transparent); }
.metric-card.blue::before  { background: linear-gradient(90deg, var(--accent-blue), transparent); }
.metric-card.green::before { background: linear-gradient(90deg, var(--accent-green), transparent); }
.metric-card.amber::before { background: linear-gradient(90deg, var(--accent-amber), transparent); }
.metric-card.red::before   { background: linear-gradient(90deg, var(--accent-red), transparent); }
.metric-label { font-family:var(--font-mono)!important; font-size:0.62rem!important; color:var(--text-secondary)!important; letter-spacing:0.12em; text-transform:uppercase; margin-bottom:0.4rem; }
.metric-value { font-family:var(--font-display)!important; font-size:1.5rem!important; font-weight:700!important; line-height:1; }
.metric-unit  { font-family:var(--font-mono)!important; font-size:0.65rem!important; color:var(--text-secondary)!important; margin-top:0.3rem; }
.cv { color:var(--accent-cyan)!important; }
.bv { color:var(--accent-blue)!important; }
.gv { color:var(--accent-green)!important; }
.av { color:var(--accent-amber)!important; }
.rv { color:var(--accent-red)!important; }

/* ── Info/Status boxes ── */
.status-bar {
    display:flex; align-items:center; gap:0.5rem;
    background:rgba(16,185,129,0.08); border:1px solid rgba(16,185,129,0.2);
    border-radius:6px; padding:0.5rem 0.9rem; margin-bottom:1rem;
    font-family:var(--font-mono)!important; font-size:0.7rem!important;
    color:var(--accent-green)!important;
}
.pulse { width:8px; height:8px; border-radius:50%; background:var(--accent-green);
         box-shadow:0 0 6px var(--accent-green); animation:pulse 2s infinite; }
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.4} }

/* ── Plotly overrides ── */
.js-plotly-plot { border-radius:10px !important; }

/* ── Streamlit widget overrides ── */
.stButton>button {
    background: linear-gradient(135deg, rgba(6,182,212,0.1), rgba(59,130,246,0.1)) !important;
    border: 1px solid var(--accent-cyan) !important;
    color: var(--accent-cyan) !important;
    font-family: var(--font-mono) !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.1em !important;
    border-radius: 6px !important;
    transition: all 0.2s !important;
}
.stButton>button:hover {
    background: linear-gradient(135deg, rgba(6,182,212,0.2), rgba(59,130,246,0.2)) !important;
    box-shadow: 0 0 12px rgba(6,182,212,0.3) !important;
}
.stSlider>div>div>div { background: var(--accent-cyan) !important; }
.stSelectbox>div { background:var(--bg-card) !important; border-color:var(--border) !important; }
div[data-testid="stExpander"] { background:var(--bg-card) !important; border-color:var(--border) !important; }
.stTabs [data-baseweb="tab-list"] { background:var(--bg-panel) !important; gap:4px; }
.stTabs [data-baseweb="tab"] {
    background:var(--bg-card) !important;
    color:var(--text-secondary) !important;
    font-family:var(--font-mono) !important;
    font-size:0.72rem !important;
    letter-spacing:0.08em !important;
    border-radius:6px 6px 0 0 !important;
}
.stTabs [aria-selected="true"] {
    background:var(--bg-card2) !important;
    color:var(--accent-cyan) !important;
    border-bottom:2px solid var(--accent-cyan) !important;
}
hr { border-color: var(--border) !important; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#   SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:1rem 0 1.5rem;'>
      <div style='font-family:Orbitron,monospace; font-size:0.75rem; color:#06b6d4;
                  letter-spacing:0.2em; text-transform:uppercase;'>🛰️ GeoTwin</div>
      <div style='font-family:Space Mono,monospace; font-size:0.6rem; color:#484f58;
                  letter-spacing:0.1em; margin-top:0.3rem;'>v1.0.0 · PHASE 1</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**NAVIGATION**")
    page = st.radio(
        "",
        ["🏠  Overview", "🌊  Flood Detection", "🌿  NDVI Monitoring", "📊  Analytics"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("**DATA SOURCE**")
    data_mode = st.selectbox(
        "Input Mode",
        ["Demo / Synthetic Data", "Upload Custom Data"],
        help="Use synthetic data to explore the dashboard without GEE credentials."
    )

    st.markdown("**PARAMETERS**")
    with st.expander("⚙️ Flood Detection Settings"):
        flood_threshold = st.slider("SAR Backscatter Threshold (dB)", -25, -10, -16)
        speckle_filter  = st.selectbox("Speckle Filter", ["Lee Filter", "Refined Lee", "Gamma MAP", "None"])
        polarization    = st.selectbox("Polarization", ["VV", "VH", "VV+VH"])

    with st.expander("⚙️ NDVI Settings"):
        ndvi_low_thresh  = st.slider("Low Vegetation Threshold",  0.0, 0.5, 0.2, 0.05)
        ndvi_high_thresh = st.slider("Healthy Vegetation Threshold", 0.3, 0.9, 0.5, 0.05)
        cloud_mask_pct   = st.slider("Max Cloud Cover (%)", 0, 50, 20)

    st.markdown("**DATE RANGE**")
    date_start = st.date_input("Start Date", datetime(2024, 6, 1))
    date_end   = st.date_input("End Date",   datetime(2024, 8, 31))

    st.markdown("---")
    st.markdown("""
    <div style='font-family:Space Mono,monospace; font-size:0.6rem; color:#484f58;
                line-height:1.8; text-align:center;'>
      Sentinel-1 SAR · Sentinel-2 MSI<br>
      Google Earth Engine · QGIS<br>
      ─────────────────<br>
      B.Tech Major Project · 2024-25
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#   DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def load_data(mode: str, flood_thresh: float, ndvi_lo: float, ndvi_hi: float):
    """Load and process satellite data (demo or uploaded)."""
    sar_data = generate_demo_sar(seed=42)
    ms_data  = generate_demo_multispectral(seed=42)

    fp = FloodProcessor(threshold_db=flood_thresh)
    flood_result = fp.process(sar_data)

    np_proc = NDVIProcessor(low_thresh=ndvi_lo, high_thresh=ndvi_hi)
    ndvi_result = np_proc.process(ms_data)

    return sar_data, ms_data, flood_result, ndvi_result

with st.spinner("🛰️ Loading satellite data streams..."):
    sar_raw, ms_raw, flood_res, ndvi_res = load_data(
        data_mode, flood_threshold, ndvi_low_thresh, ndvi_high_thresh
    )


# ══════════════════════════════════════════════════════════════════════════════
#   HERO HEADER (always visible)
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="hero-header">
  <div class="hero-title">AI-Driven Geospatial Digital Twin</div>
  <div class="hero-subtitle">Multi-Hazard Environmental Intelligence · IoT · SAR Remote Sensing · GIS</div>
  <div class="hero-phase">◆ PHASE 1 — SOFTWARE FOUNDATION</div>
  <div class="hero-badges">
    <span class="badge badge-cyan">SENTINEL-1 SAR</span>
    <span class="badge badge-blue">SENTINEL-2 MSI</span>
    <span class="badge badge-green">GOOGLE EARTH ENGINE</span>
    <span class="badge badge-amber">QGIS COMPATIBLE</span>
    <span class="badge badge-cyan">FLOOD DETECTION</span>
    <span class="badge badge-green">NDVI MONITORING</span>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="status-bar">
  <div class="pulse"></div>
  SYSTEM ONLINE · Data Mode: {data_mode} · Date Range: {date_start} → {date_end} · Modules Active: 2/2
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#   PAGE: OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if "Overview" in page:
    # ── Top KPI row ──
    flooded_pct  = float(np.mean(flood_res["flood_mask"])) * 100
    ndvi_mean    = float(np.nanmean(ndvi_res["ndvi"]))
    healthy_pct  = float(np.mean(ndvi_res["classification"] == 2)) * 100
    risk_score   = min(100, flooded_pct * 3.5)

    st.markdown("""<div class="metric-grid">""", unsafe_allow_html=True)
    col1,col2,col3,col4,col5 = st.columns(5)
    with col1:
        st.markdown(f"""<div class="metric-card cyan">
            <div class="metric-label">Flooded Area</div>
            <div class="metric-value cv">{flooded_pct:.1f}<span style='font-size:0.8rem'>%</span></div>
            <div class="metric-unit">of AOI · SAR VV</div></div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class="metric-card blue">
            <div class="metric-label">Mean NDVI</div>
            <div class="metric-value bv">{ndvi_mean:.3f}</div>
            <div class="metric-unit">Sentinel-2 B8/B4</div></div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div class="metric-card green">
            <div class="metric-label">Healthy Veg.</div>
            <div class="metric-value gv">{healthy_pct:.1f}<span style='font-size:0.8rem'>%</span></div>
            <div class="metric-unit">NDVI ≥ {ndvi_high_thresh}</div></div>""", unsafe_allow_html=True)
    with col4:
        st.markdown(f"""<div class="metric-card amber">
            <div class="metric-label">SAR Threshold</div>
            <div class="metric-value av">{flood_threshold}</div>
            <div class="metric-unit">dB backscatter</div></div>""", unsafe_allow_html=True)
    with col5:
        color_cls = "rv" if risk_score > 60 else ("av" if risk_score > 30 else "gv")
        card_cls  = "red" if risk_score > 60 else ("amber" if risk_score > 30 else "green")
        st.markdown(f"""<div class="metric-card {card_cls}">
            <div class="metric-label">Risk Score</div>
            <div class="metric-value {color_cls}">{risk_score:.0f}</div>
            <div class="metric-unit">composite index</div></div>""", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # ── Side-by-side maps ──
    col_f, col_n = st.columns(2, gap="medium")
    with col_f:
        st.markdown("""<div class="section-header">
          <div class="section-icon icon-flood">🌊</div>
          <span class="section-title" style="color:#3b82f6">Flood Detection Map</span>
        </div>""", unsafe_allow_html=True)
        fig_flood = plot_flood_map(flood_res["flood_mask"], sar_raw["vv"])
        st.pyplot(fig_flood, use_container_width=True)
        plt.close()

    with col_n:
        st.markdown("""<div class="section-header">
          <div class="section-icon icon-ndvi">🌿</div>
          <span class="section-title" style="color:#10b981">NDVI Vegetation Map</span>
        </div>""", unsafe_allow_html=True)
        fig_ndvi = plot_ndvi_map(ndvi_res["ndvi"], ndvi_res["classification"])
        st.pyplot(fig_ndvi, use_container_width=True)
        plt.close()

    # ── Time series overview ──
    st.markdown("---")
    st.markdown("""<div class="section-header">
      <span class="section-title" style="color:#06b6d4">📈 Temporal Analysis</span>
    </div>""", unsafe_allow_html=True)
    col_ts1, col_ts2 = st.columns(2)
    with col_ts1:
        fig_fts = plot_flood_time_series(date_start, date_end)
        st.plotly_chart(fig_fts, use_container_width=True)
    with col_ts2:
        fig_nts = plot_ndvi_time_series(date_start, date_end)
        st.plotly_chart(fig_nts, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#   PAGE: FLOOD DETECTION
# ══════════════════════════════════════════════════════════════════════════════
elif "Flood" in page:
    st.markdown("""<div class="section-header">
      <div class="section-icon icon-flood">🌊</div>
      <span class="section-title" style="color:#3b82f6">Flood Detection — Sentinel-1 SAR Analysis</span>
    </div>""", unsafe_allow_html=True)

    tabs = st.tabs(["🗺️  Flood Map", "📊  Backscatter Analysis", "📋  Classification Report", "🔬  Methodology"])

    with tabs[0]:
        col_map, col_meta = st.columns([2, 1], gap="medium")
        with col_map:
            fig = plot_flood_map(flood_res["flood_mask"], sar_raw["vv"], large=True)
            st.pyplot(fig, use_container_width=True)
            plt.close()
        with col_meta:
            flooded_px = int(np.sum(flood_res["flood_mask"]))
            total_px   = flood_res["flood_mask"].size
            flooded_km2 = flooded_px * 0.01  # 100m res → 0.01 km² per pixel
            st.markdown("**DETECTION SUMMARY**")
            metrics = [
                ("Flooded Pixels",  str(flooded_px),       "cyan",  "of total AOI"),
                ("Flooded Area",    f"{flooded_km2:.1f} km²","blue",  "estimated @ 10m res"),
                ("Dry Pixels",      str(total_px-flooded_px),"green", "unaffected"),
                ("Threshold Used",  f"{flood_threshold} dB", "amber", "SAR VV backscatter"),
                ("Polarization",    polarization,            "cyan",  "SAR acquisition mode"),
                ("Speckle Filter",  speckle_filter,          "blue",  "preprocessing step"),
            ]
            for label, val, color, unit in metrics:
                st.markdown(f"""<div class="metric-card {color}" style="margin-bottom:0.5rem;">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value {color[0]}v" style="font-size:1.1rem">{val}</div>
                    <div class="metric-unit">{unit}</div></div>""", unsafe_allow_html=True)

    with tabs[1]:
        col_h, col_s = st.columns(2)
        with col_h:
            st.markdown("**SAR Backscatter Distribution**")
            fig_hist = plot_backscatter_histogram(sar_raw["vv"], flood_threshold)
            st.plotly_chart(fig_hist, use_container_width=True)
        with col_s:
            st.markdown("**Flood vs. Non-Flood Pixel Distribution**")
            fig_pie = plot_class_distribution_pie(
                flood_res["flood_mask"],
                labels=["Flooded", "Non-Flooded"],
                colors=["#3b82f6", "#0ea5e9"]
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        st.markdown("**SAR Backscatter Raw Values (VV Band)**")
        fig_heat = go.Figure(go.Heatmap(
            z=sar_raw["vv"],
            colorscale="Blues",
            colorbar=dict(title="dB", tickfont=dict(color="#8b949e")),
            zmin=-25, zmax=0
        ))
        fig_heat.update_layout(
            template="plotly_dark",
            paper_bgcolor="#111827", plot_bgcolor="#111827",
            margin=dict(l=10,r=10,t=10,b=10),
            height=300
        )
        st.plotly_chart(fig_heat, use_container_width=True)

    with tabs[2]:
        st.markdown("**Classification Results Table**")
        flood_mask = flood_res["flood_mask"]
        rows = []
        for cls, label, thresh in [
            ("Flooded",     "Water/Flooded Surface", f"VV < {flood_threshold} dB"),
            ("Non-Flooded", "Dry Land",              f"VV ≥ {flood_threshold} dB"),
        ]:
            mask = flood_mask if cls == "Flooded" else ~flood_mask
            count = int(np.sum(mask))
            rows.append({
                "Class": cls, "Description": label,
                "Pixel Count": count,
                "Area (km²)": f"{count*0.01:.2f}",
                "Coverage (%)": f"{count/flood_mask.size*100:.1f}",
                "Criterion": thresh
            })
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)

        st.markdown("**Accuracy Assessment (simulated)**")
        acc_data = {
            "Metric": ["Overall Accuracy", "Kappa Coefficient", "Producer Accuracy (Flood)",
                       "User Accuracy (Flood)", "F1 Score", "IoU (Flood)"],
            "Value": ["91.4%", "0.847", "88.2%", "93.7%", "90.8%", "0.832"],
            "Notes": ["All pixels correctly classified", "Agreement beyond chance",
                      "Recall — how many floods detected", "Precision — flood class reliability",
                      "Harmonic mean P/R", "Intersection over Union"]
        }
        st.dataframe(pd.DataFrame(acc_data), use_container_width=True, hide_index=True)

    with tabs[3]:
        st.markdown("""
        ### 🔬 SAR Flood Detection Methodology

        **1. Data Acquisition**
        - Platform: **Sentinel-1 GRD** (Ground Range Detected)
        - Polarization: VV (vertical-vertical) — optimal for flood detection
        - Resolution: 10m × 10m
        - Source: Google Earth Engine `COPERNICUS/S1_GRD`

        **2. Pre-processing Pipeline**
        ```
        Raw GRD → Apply Orbit File → Border Noise Removal
              → Thermal Noise Removal → Radiometric Calibration (σ°)
              → Speckle Filtering (Lee / Refined Lee)
              → Terrain Correction (Range-Doppler)
              → Convert to dB: σ°(dB) = 10 × log₁₀(σ°)
        ```

        **3. Flood Classification**
        - Method: **Threshold-based backscatter analysis**
        - Principle: Water bodies exhibit significantly lower backscatter than dry land
          due to specular reflection (water acts like a mirror — signal bounces away from sensor)
        - Classification rule:
          ```
          If σ°(VV) < threshold_dB  →  FLOODED
          Else                       →  NON-FLOODED
          ```

        **4. Post-processing**
        - Morphological opening (remove salt-and-pepper noise)
        - Minimum mapping unit: 3 × 3 pixel block
        - Permanent water body masking (JRC Global Surface Water)

        **References**
        - Twele et al. (2016): SAR-based flood mapping using Sentinel-1
        - Martinis et al. (2015): Towards operational near-real-time flood detection
        """)


# ══════════════════════════════════════════════════════════════════════════════
#   PAGE: NDVI MONITORING
# ══════════════════════════════════════════════════════════════════════════════
elif "NDVI" in page:
    st.markdown("""<div class="section-header">
      <div class="section-icon icon-ndvi">🌿</div>
      <span class="section-title" style="color:#10b981">NDVI Crop Health Monitoring — Sentinel-2</span>
    </div>""", unsafe_allow_html=True)

    tabs = st.tabs(["🗺️  NDVI Map", "📊  Spectral Analysis", "📋  Classification Report", "🔬  Methodology"])

    with tabs[0]:
        col_map, col_meta = st.columns([2, 1], gap="medium")
        with col_map:
            fig = plot_ndvi_map(ndvi_res["ndvi"], ndvi_res["classification"], large=True)
            st.pyplot(fig, use_container_width=True)
            plt.close()
        with col_meta:
            cls = ndvi_res["classification"]
            ndvi = ndvi_res["ndvi"]
            class_labels = {0: "Low Veg.", 1: "Moderate Veg.", 2: "Healthy Veg."}
            class_colors_m = {"Low Veg.": "red", "Moderate Veg.": "amber", "Healthy Veg.": "green"}
            st.markdown("**NDVI STATISTICS**")
            stats = [
                ("Mean NDVI",   f"{np.nanmean(ndvi):.4f}", "cyan",  "area average"),
                ("Max NDVI",    f"{np.nanmax(ndvi):.4f}",  "green", "peak vegetation"),
                ("Min NDVI",    f"{np.nanmin(ndvi):.4f}",  "blue",  "bare soil / water"),
                ("Std Dev",     f"{np.nanstd(ndvi):.4f}",  "amber", "spatial variability"),
            ]
            for label, val, color, unit in stats:
                st.markdown(f"""<div class="metric-card {color}" style="margin-bottom:0.5rem;">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value {color[0]}v" style="font-size:1.1rem">{val}</div>
                    <div class="metric-unit">{unit}</div></div>""", unsafe_allow_html=True)

            st.markdown("**CLASS BREAKDOWN**")
            for c_id, c_name in class_labels.items():
                pct = float(np.mean(cls == c_id)) * 100
                color = list(class_colors_m.values())[c_id]
                st.markdown(f"""<div class="metric-card {color}" style="margin-bottom:0.5rem;">
                    <div class="metric-label">{c_name}</div>
                    <div class="metric-value {color[0]}v" style="font-size:1.1rem">{pct:.1f}%</div>
                    <div class="metric-unit">of total pixels</div></div>""", unsafe_allow_html=True)

    with tabs[1]:
        col_h, col_p = st.columns(2)
        with col_h:
            st.markdown("**NDVI Value Distribution**")
            fig_hist = plot_ndvi_histogram(ndvi_res["ndvi"], ndvi_low_thresh, ndvi_high_thresh)
            st.plotly_chart(fig_hist, use_container_width=True)
        with col_p:
            st.markdown("**Vegetation Class Distribution**")
            cls_arr = ndvi_res["classification"]
            counts  = [int(np.sum(cls_arr == i)) for i in range(3)]
            fig_pie = go.Figure(go.Pie(
                labels=["Low Vegetation", "Moderate Vegetation", "Healthy Vegetation"],
                values=counts,
                hole=0.55,
                marker=dict(colors=["#ef4444","#f59e0b","#10b981"],
                            line=dict(color="#111827", width=2)),
                textfont=dict(family="Space Mono", size=11, color="#f0f6fc")
            ))
            fig_pie.update_layout(
                template="plotly_dark", paper_bgcolor="#111827",
                legend=dict(font=dict(color="#8b949e", family="Space Mono", size=10)),
                margin=dict(l=10,r=10,t=10,b=10), height=320
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        st.markdown("**Band Correlation: NIR vs RED**")
        nir_flat = ndvi_res["nir"].flatten()[::20]
        red_flat = ndvi_res["red"].flatten()[::20]
        fig_scatter = go.Figure(go.Scatter(
            x=red_flat, y=nir_flat, mode="markers",
            marker=dict(color=ndvi_res["ndvi"].flatten()[::20],
                        colorscale="RdYlGn", size=3, opacity=0.6,
                        colorbar=dict(title="NDVI", tickfont=dict(color="#8b949e")))
        ))
        fig_scatter.update_layout(
            template="plotly_dark", paper_bgcolor="#111827", plot_bgcolor="#0d1117",
            xaxis_title="RED Band (B4)", yaxis_title="NIR Band (B8)",
            margin=dict(l=10,r=10,t=10,b=10), height=300
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    with tabs[2]:
        st.markdown("**NDVI Classification Summary**")
        cls_arr = ndvi_res["classification"]
        ndvi_arr = ndvi_res["ndvi"]
        rows = []
        for c_id, c_name, rng, color in [
            (0, "Low Vegetation",      f"NDVI < {ndvi_low_thresh}",                    "🔴"),
            (1, "Moderate Vegetation", f"{ndvi_low_thresh} ≤ NDVI < {ndvi_high_thresh}","🟡"),
            (2, "Healthy Vegetation",  f"NDVI ≥ {ndvi_high_thresh}",                   "🟢"),
        ]:
            mask   = cls_arr == c_id
            count  = int(np.sum(mask))
            mean_v = float(np.nanmean(ndvi_arr[mask])) if count > 0 else 0
            rows.append({
                "Status": color, "Class": c_name, "NDVI Range": rng,
                "Pixel Count": count,
                "Coverage (%)": f"{count/cls_arr.size*100:.1f}",
                "Mean NDVI": f"{mean_v:.4f}"
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        st.markdown("**Crop Health Interpretation**")
        interp_data = {
            "NDVI Range": ["< 0.0", "0.0 – 0.2", "0.2 – 0.4", "0.4 – 0.6", "> 0.6"],
            "Land Cover": ["Water / Snow", "Bare Soil / Rock", "Sparse / Stressed Vegetation",
                           "Moderate Cropland", "Dense / Healthy Vegetation"],
            "Agricultural Implication": [
                "Non-agricultural area", "No crop cover detected",
                "Crop stress or early growth stage", "Developing crop — monitor closely",
                "Optimal crop health — peak growth"
            ]
        }
        st.dataframe(pd.DataFrame(interp_data), use_container_width=True, hide_index=True)

    with tabs[3]:
        st.markdown("""
        ### 🔬 NDVI Crop Health Monitoring Methodology

        **1. Data Acquisition**
        - Platform: **Sentinel-2 MSI** (MultiSpectral Instrument) Level-2A
        - Bands used: **B4 (Red, 665 nm)** and **B8 (NIR, 842 nm)**
        - Resolution: 10m × 10m
        - Source: Google Earth Engine `COPERNICUS/S2_SR_HARMONIZED`

        **2. Pre-processing**
        ```
        L2A Product → Cloud Masking (SCL band / s2cloudless)
                   → Atmospheric Correction (already applied in L2A)
                   → Scale Reflectance (÷ 10000)
                   → Clip to AOI
        ```

        **3. NDVI Calculation**
        ```
        NDVI = (NIR - RED) / (NIR + RED)
             = (B8  - B4 ) / (B8  + B4 )

        Range: -1.0 (water/bare) → +1.0 (dense vegetation)
        ```

        **4. Classification Thresholds**
        ```
        NDVI < 0.2           →  Low Vegetation (bare soil, stressed crops)
        0.2 ≤ NDVI < 0.5     →  Moderate Vegetation (developing crops)
        NDVI ≥ 0.5           →  Healthy Vegetation (peak canopy, dense crops)
        ```

        **5. Temporal Monitoring**
        - Multi-temporal composites (monthly median)
        - Change detection between seasons
        - Crop phenology tracking

        **References**
        - Rouse et al. (1974): NDVI original formulation
        - Tucker (1979): Red & photographic infrared linear combinations for vegetation monitoring
        - Drusch et al. (2012): Sentinel-2: ESA's Optical High-Resolution Mission
        """)


# ══════════════════════════════════════════════════════════════════════════════
#   PAGE: ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
elif "Analytics" in page:
    st.markdown("""<div class="section-header">
      <span class="section-title" style="color:#06b6d4">📊 Integrated Analytics Dashboard</span>
    </div>""", unsafe_allow_html=True)

    col_ts1, col_ts2 = st.columns(2)
    with col_ts1:
        fig_fts = plot_flood_time_series(date_start, date_end)
        st.plotly_chart(fig_fts, use_container_width=True)
    with col_ts2:
        fig_nts = plot_ndvi_time_series(date_start, date_end)
        st.plotly_chart(fig_nts, use_container_width=True)

    st.markdown("---")
    st.markdown("**CROSS-MODULE RISK MATRIX**")

    # Generate mock risk data
    np.random.seed(7)
    dates_r = pd.date_range(date_start.strftime('%Y-%m-%d'), date_end.strftime('%Y-%m-%d'), freq='7D')
    flood_risk_vals = np.clip(np.random.normal(35, 20, len(dates_r)), 0, 100)
    ndvi_health     = np.clip(np.random.normal(0.45, 0.15, len(dates_r)), 0, 1)
    composite_risk  = flood_risk_vals * 0.6 + (1 - ndvi_health) * 100 * 0.4

    fig_risk = go.Figure()
    fig_risk.add_trace(go.Scatter(
        x=dates_r, y=flood_risk_vals, name="Flood Risk",
        line=dict(color="#3b82f6", width=2),
        fill='tozeroy', fillcolor="rgba(59,130,246,0.08)"
    ))
    fig_risk.add_trace(go.Scatter(
        x=dates_r, y=(1-ndvi_health)*100, name="Crop Stress Index",
        line=dict(color="#ef4444", width=2),
        fill='tozeroy', fillcolor="rgba(239,68,68,0.05)"
    ))
    fig_risk.add_trace(go.Scatter(
        x=dates_r, y=composite_risk, name="Composite Risk",
        line=dict(color="#f59e0b", width=2.5, dash='dot'),
    ))
    fig_risk.update_layout(
        template="plotly_dark", paper_bgcolor="#111827", plot_bgcolor="#0d1117",
        legend=dict(font=dict(color="#8b949e", family="Space Mono", size=10), bgcolor="#111827"),
        xaxis=dict(gridcolor="#1e2d40"), yaxis=dict(gridcolor="#1e2d40", title="Risk Score (0–100)"),
        margin=dict(l=10,r=10,t=30,b=10), height=320,
        title=dict(text="Multi-Hazard Risk Index — Temporal Evolution", font=dict(color="#8b949e", size=12, family="Space Mono"))
    )
    st.plotly_chart(fig_risk, use_container_width=True)

    # Summary table
    st.markdown("**ANALYSIS SUMMARY — PHASE 1**")
    summary = {
        "Module": ["Flood Detection", "NDVI Monitoring"],
        "Satellite": ["Sentinel-1 SAR", "Sentinel-2 MSI"],
        "Band/Signal": ["VV Backscatter", "B4 (Red) + B8 (NIR)"],
        "Method": ["Threshold (dB)", "NDVI Ratio Index"],
        "Classes": ["2 (Flooded/Dry)", "3 (Low/Moderate/Healthy)"],
        "Resolution": ["10m", "10m"],
        "GEE Collection": ["COPERNICUS/S1_GRD", "COPERNICUS/S2_SR_HARMONIZED"],
        "Status": ["✅ Active", "✅ Active"]
    }
    st.dataframe(pd.DataFrame(summary), use_container_width=True, hide_index=True)
