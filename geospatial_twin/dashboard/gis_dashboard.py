"""
================================================================================
  dashboard/gis_dashboard.py
  GeoTwin Phase 2 — Main Streamlit GIS Dashboard
  
  Connects directly to Phase 1 modules:
    - flood_detection.flood_processor.FloodProcessor
    - ndvi_monitoring.ndvi_processor.NDVIProcessor
    - utils.demo_data (synthetic Sentinel-1/2 data)
  
  Run:
    cd geospatial_twin/dashboard
    streamlit run gis_dashboard.py
================================================================================
"""

import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import pandas as pd
import sys, os
from datetime import datetime, date

# ── Path resolution — works from any working directory ───────────────────────
_DASH_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT     = os.path.abspath(os.path.join(_DASH_DIR, ".."))
for p in [_ROOT, _DASH_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

# ── Phase 2 modules ───────────────────────────────────────────────────────────
from utils.data_bridge    import Phase1DataBridge
from maps.folium_maps     import build_full_gis_map, build_before_after_map, map_to_html_string
from charts.plotly_charts import (
    flood_area_donut, flood_backscatter_histogram, flood_time_series,
    before_after_bar, ndvi_class_donut, ndvi_histogram, ndvi_time_series,
    ndvi_nir_red_scatter, risk_gauge, multi_hazard_timeline,
    sensor_battery_bar, environmental_stats_table
)
from components.ui_components import (
    GLOBAL_CSS, hero_header, kpi_row, section_header,
    status_banner, map_container_open, map_container_close,
    info_card, sidebar_logo, sidebar_section_label
)
from utils.export_manager import (
    flood_map_png_bytes, ndvi_map_png_bytes,
    stats_csv_bytes, geojson_bytes,
    ndvi_dataframe_csv_bytes, full_export_zip
)

# ─────────────────────────────────────────────────────────────────────────────
#   PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GeoTwin Phase 2 · GIS Dashboard",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject global CSS
st.markdown(GLOBAL_CSS, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#   SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(sidebar_logo(), unsafe_allow_html=True)

    # ── Navigation ───────────────────────────────────────────────────────────
    st.markdown(sidebar_section_label("Navigation"), unsafe_allow_html=True)
    page = st.radio(
        "",
        ["🌐  GIS Overview",
         "🌊  Flood Monitor",
         "🌿  NDVI Monitor",
         "📊  Analytics",
         "📡  IoT Sensors",
         "📤  Export Center"],
        label_visibility="collapsed"
    )

    # ── Data Source ──────────────────────────────────────────────────────────
    st.markdown(sidebar_section_label("Data Source"), unsafe_allow_html=True)
    data_mode = st.selectbox(
        "Input Mode",
        ["Demo / Synthetic Data", "GEE Live (requires auth)"],
        help="Demo mode uses realistic synthetic Sentinel-1/2 data."
    )
    study_area = st.selectbox(
        "Study Area",
        ["Ganga Floodplain, UP/Bihar",
         "Brahmaputra, Assam",
         "Mumbai Coastal Region",
         "Kerala Backwaters",
         "Punjab Cropland"],
    )

    # ── Date Range ───────────────────────────────────────────────────────────
    st.markdown(sidebar_section_label("Date Range"), unsafe_allow_html=True)
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        date_start = st.date_input("From", date(2024, 6, 1))
    with col_d2:
        date_end   = st.date_input("To",   date(2024, 8, 31))

    # ── Processing Parameters ─────────────────────────────────────────────────
    st.markdown(sidebar_section_label("Processing Params"), unsafe_allow_html=True)
    with st.expander("⚙️ Flood Detection", expanded=False):
        flood_threshold = st.slider("SAR Threshold (dB)", -24, -10, -16, 1)
        speckle_filter  = st.selectbox("Speckle Filter",
                                       ["Lee Filter", "Refined Lee", "Gamma MAP", "None"])
        polarization    = st.selectbox("Polarization", ["VV", "VH", "VV+VH"])

    with st.expander("⚙️ NDVI Analysis", expanded=False):
        ndvi_lo = st.slider("Low Veg. Threshold",  0.0, 0.4, 0.2, 0.05)
        ndvi_hi = st.slider("Healthy Threshold",   0.3, 0.9, 0.5, 0.05)
        cloud_max = st.slider("Max Cloud Cover (%)", 0, 50, 20, 5)

    # ── Map Layers ────────────────────────────────────────────────────────────
    st.markdown(sidebar_section_label("Map Layers"), unsafe_allow_html=True)
    basemap        = st.selectbox("Base Map",
                                  ["CartoDB Dark", "Satellite (Esri)",
                                   "OpenStreetMap", "Topographic (OpenTopo)"])
    show_flood     = st.checkbox("🌊 Flood Layer",           value=True)
    show_ndvi_heat = st.checkbox("🌿 NDVI Heatmap",          value=False)
    show_ndvi_cls  = st.checkbox("🌱 Vegetation Classes",    value=False)
    show_risk      = st.checkbox("⚠️ Risk Index Layer",      value=False)
    show_sensors   = st.checkbox("📡 IoT Sensors",           value=True)
    show_aoi       = st.checkbox("🗺️ AOI Boundary",          value=True)
    zoom_level     = st.slider("Map Zoom", 5, 14, 8)

    # ── Quick Export ──────────────────────────────────────────────────────────
    st.markdown(sidebar_section_label("Quick Export"), unsafe_allow_html=True)
    run_export = st.button("⬇️  Export All Outputs", use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
#   LOAD / CACHE PHASE 1 DATA
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data(threshold: float, lo: float, hi: float, seed: int = 42) -> dict:
    bridge = Phase1DataBridge(
        flood_threshold=threshold,
        ndvi_lo=lo,
        ndvi_hi=hi,
        seed=seed
    )
    return bridge.get_all()

with st.spinner("🛰️ Connecting to Phase 1 processing engines..."):
    D = load_data(flood_threshold, ndvi_lo, ndvi_hi)

metrics   = D["metrics"]
fr        = D["flood_result"]
nr        = D["ndvi_result"]
flood_ts  = D["flood_ts"]
ndvi_ts   = D["ndvi_ts"]
sensor_df = D["sensor_points"]
bbox      = D["bbox"]


# ─────────────────────────────────────────────────────────────────────────────
#   HERO + STATUS BAR (all pages)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(hero_header(page), unsafe_allow_html=True)
st.markdown(
    status_banner(
        mode=data_mode,
        date_start=str(date_start),
        date_end=str(date_end),
        n_sensors=len(sensor_df),
        phase1_ok=True
    ),
    unsafe_allow_html=True
)

# KPI row (all pages)
st.markdown(kpi_row(metrics), unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#   QUICK EXPORT HANDLER (sidebar button)
# ─────────────────────────────────────────────────────────────────────────────
if run_export:
    with st.spinner("📦 Packaging all outputs..."):
        zip_bytes = full_export_zip(D)
    st.sidebar.download_button(
        label="📦 Download GeoTwin_Outputs.zip",
        data=zip_bytes,
        file_name=f"GeoTwin_Phase2_Outputs_{datetime.now().strftime('%Y%m%d_%H%M')}.zip",
        mime="application/zip",
        use_container_width=True
    )
    st.sidebar.success("✅ Export ready!")


# ══════════════════════════════════════════════════════════════════════════════
#   PAGE: GIS OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if "Overview" in page:

    st.markdown(
        section_header("🌐", "Interactive GIS Map",
                        "Folium · Leaflet.js · Layer Controls · Coordinate Display", "#06b6d4"),
        unsafe_allow_html=True
    )

    # ── Build Folium map ──────────────────────────────────────────────────────
    with st.spinner("🗺️ Rendering GIS map..."):
        m = build_full_gis_map(
            D,
            basemap=basemap,
            show_flood=show_flood,
            show_ndvi_heat=show_ndvi_heat,
            show_ndvi_class=show_ndvi_cls,
            show_risk=show_risk,
            show_sensors=show_sensors,
            show_aoi=show_aoi,
            zoom_start=zoom_level
        )
        map_html = map_to_html_string(m)

    st.markdown(map_container_open("GIS INTELLIGENCE MAP", study_area), unsafe_allow_html=True)
    components.html(map_html, height=580, scrolling=False)
    st.markdown(map_container_close(), unsafe_allow_html=True)

    # ── Quick summary below map ───────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            section_header("🌊", "Flood Summary", "", "#3b82f6"),
            unsafe_allow_html=True
        )
        st.plotly_chart(flood_area_donut(metrics, height=280), use_container_width=True)

    with col2:
        st.markdown(
            section_header("🌿", "Vegetation Summary", "", "#10b981"),
            unsafe_allow_html=True
        )
        st.plotly_chart(ndvi_class_donut(metrics, height=280), use_container_width=True)

    with col3:
        st.markdown(
            section_header("⚠️", "Risk Assessment", "", "#f59e0b"),
            unsafe_allow_html=True
        )
        st.plotly_chart(risk_gauge(metrics["risk_score"], height=280), use_container_width=True)

    # ── Multi-hazard timeline ─────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        section_header("📈", "Multi-Hazard Temporal Analysis", "Flood × NDVI × Risk", "#06b6d4"),
        unsafe_allow_html=True
    )
    st.plotly_chart(multi_hazard_timeline(flood_ts, ndvi_ts), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#   PAGE: FLOOD MONITOR
# ══════════════════════════════════════════════════════════════════════════════
elif "Flood" in page:

    st.markdown(
        section_header("🌊", "Flood Detection Monitor",
                        "Sentinel-1 SAR · VV Backscatter · Threshold Classification", "#3b82f6"),
        unsafe_allow_html=True
    )

    tabs = st.tabs([
        "🗺️  GIS Flood Map",
        "📊  Backscatter Analysis",
        "📅  Time Series",
        "🔄  Before / After",
        "📋  Classification Report"
    ])

    # ── Tab 1: GIS Flood Map ─────────────────────────────────────────────────
    with tabs[0]:
        col_map, col_info = st.columns([3, 1], gap="medium")

        with col_map:
            with st.spinner("Rendering flood map..."):
                m_flood = build_full_gis_map(
                    D, basemap=basemap,
                    show_flood=True, show_ndvi_heat=False,
                    show_ndvi_class=False, show_risk=show_risk,
                    show_sensors=show_sensors, show_aoi=True,
                    zoom_start=zoom_level
                )
            st.markdown(map_container_open("FLOOD CLASSIFICATION MAP — SENTINEL-1 SAR"), unsafe_allow_html=True)
            components.html(map_to_html_string(m_flood), height=500)
            st.markdown(map_container_close(), unsafe_allow_html=True)

        with col_info:
            st.markdown("**DETECTION PARAMETERS**")
            param_rows = [
                ("Satellite",    "Sentinel-1"),
                ("Sensor",       "SAR C-band"),
                ("Polarization", polarization),
                ("Mode",         "IW (10m)"),
                ("Filter",       speckle_filter),
                ("Threshold",    f"{flood_threshold} dB"),
                ("Flooded",      f"{metrics['flooded_pct']:.1f}%"),
                ("Area",         f"~{metrics['flooded_km2']:.0f} km²"),
                ("Total pixels", f"{metrics['total_pixels']:,}"),
                ("Study Area",   bbox["label"][:20] + "..."),
            ]
            for k, v in param_rows:
                st.markdown(f"""
                <div style="display:flex; justify-content:space-between;
                            padding:0.3rem 0; border-bottom:1px solid #1e2d40;">
                  <span style="font-family:Space Mono,monospace; font-size:0.65rem;
                               color:#8b949e;">{k}</span>
                  <span style="font-family:Space Mono,monospace; font-size:0.65rem;
                               color:#06b6d4;">{v}</span>
                </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Download flood map PNG
            flood_png = flood_map_png_bytes(fr["flood_mask"], fr["backscatter_db"])
            st.download_button(
                "⬇️ Flood Map (PNG)",
                data=flood_png,
                file_name="flood_classification_map.png",
                mime="image/png",
                use_container_width=True
            )
            # Download GeoJSON
            geojson_data = geojson_bytes(D["flood_geojson"])
            st.download_button(
                "⬇️ Flood GeoJSON",
                data=geojson_data,
                file_name="flood_classification.geojson",
                mime="application/json",
                use_container_width=True
            )

    # ── Tab 2: Backscatter Analysis ──────────────────────────────────────────
    with tabs[1]:
        col_hist, col_donut = st.columns(2, gap="medium")
        with col_hist:
            st.plotly_chart(
                flood_backscatter_histogram(fr["backscatter_db"], flood_threshold),
                use_container_width=True
            )
        with col_donut:
            st.plotly_chart(flood_area_donut(metrics), use_container_width=True)

        st.markdown("---")
        st.markdown("**SAR BACKSCATTER STATISTICS**")
        flood_stats = fr["stats"]
        stat_df = pd.DataFrame({
            "Metric": [
                "Mean Backscatter (all)", "Mean Backscatter (flooded)",
                "Mean Backscatter (dry)", "Std Deviation",
                "Flooded Pixels", "Dry Pixels",
                "Flooded Area (%)", "Threshold Used"
            ],
            "Value": [
                f"{flood_stats['mean_backscatter_all']:.2f} dB",
                f"{flood_stats['mean_backscatter_flood']:.2f} dB",
                f"{flood_stats['mean_backscatter_dry']:.2f} dB",
                f"{flood_stats['std_backscatter']:.2f} dB",
                f"{flood_stats['flooded_pixels']:,}",
                f"{flood_stats['dry_pixels']:,}",
                f"{flood_stats['flooded_pct']:.2f}%",
                f"{flood_stats['threshold_db']} dB"
            ]
        })
        st.dataframe(stat_df, use_container_width=True, hide_index=True)

    # ── Tab 3: Time Series ───────────────────────────────────────────────────
    with tabs[2]:
        st.plotly_chart(flood_time_series(flood_ts), use_container_width=True)
        st.markdown("**FLOOD EVENT OBSERVATIONS**")
        display_ts = flood_ts.copy()
        display_ts["date"] = display_ts["date"].dt.strftime("%d %b %Y")
        st.dataframe(
            display_ts.rename(columns={
                "date": "Date", "flood_pct": "Flood Extent (%)",
                "area_km2": "Area (km²)", "satellite": "Satellite"
            }),
            use_container_width=True, hide_index=True, height=320
        )

    # ── Tab 4: Before / After ────────────────────────────────────────────────
    with tabs[3]:
        st.markdown(
            info_card("CHANGE DETECTION",
                      "Before/After comparison of Sentinel-1 SAR backscatter. "
                      "The pre-event scene shows baseline land cover; "
                      "the post-flood scene reveals flooded areas as dark (low backscatter) patches.",
                      "#3b82f6"),
            unsafe_allow_html=True
        )
        with st.spinner("Building comparison map..."):
            m_ba = build_before_after_map(D)
        st.markdown(map_container_open("BEFORE / AFTER FLOOD — SAR COMPARISON"), unsafe_allow_html=True)
        components.html(map_to_html_string(m_ba), height=500)
        st.markdown(map_container_close(), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.plotly_chart(
            before_after_bar(metrics, before_flooded_pct=5.2),
            use_container_width=True
        )

    # ── Tab 5: Classification Report ─────────────────────────────────────────
    with tabs[4]:
        st.markdown("**FLOOD CLASSIFICATION ACCURACY ASSESSMENT** *(simulated)*")
        acc_df = pd.DataFrame({
            "Metric": [
                "Overall Accuracy", "Kappa Coefficient",
                "Producer Accuracy (Flood)", "User Accuracy (Flood)",
                "F1 Score", "IoU (Flood Class)", "False Alarm Rate"
            ],
            "Value":  ["91.4%", "0.847", "88.2%", "93.7%", "90.8%", "0.832", "6.3%"],
            "Benchmark": ["≥ 85%", "≥ 0.75", "≥ 80%", "≥ 80%", "≥ 85%", "≥ 0.75", "≤ 15%"],
            "Status": ["✅ Pass", "✅ Pass", "✅ Pass", "✅ Pass", "✅ Pass", "✅ Pass", "✅ Pass"]
        })
        st.dataframe(acc_df, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown("**PROCESSING CHAIN**")
        steps = [
            ("1", "Data Acquisition",     "Sentinel-1 GRD, IW mode, VV polarization, 10m"),
            ("2", "Orbit Correction",     "Apply precise orbit file (POD)"),
            ("3", "Noise Removal",        "Thermal + border noise removal"),
            ("4", "Radiometric Cal.",     "Calibrate to sigma-nought (σ°)"),
            ("5", "Speckle Filtering",    f"{speckle_filter} — reduces multiplicative noise"),
            ("6", "dB Conversion",        "σ°(dB) = 10 × log₁₀(σ°)"),
            ("7", "Terrain Correction",   "Range-Doppler orthorectification (SRTM DEM)"),
            ("8", "Classification",       f"σ°(VV) < {flood_threshold} dB → FLOODED"),
            ("9", "Post-processing",      "Morphological cleanup, minimum mapping unit"),
            ("10","Export",               "GeoTIFF (EPSG:4326), GeoJSON, PNG"),
        ]
        for step, title, desc in steps:
            st.markdown(f"""
            <div style="display:flex; gap:0.7rem; align-items:flex-start;
                        padding:0.4rem 0; border-bottom:1px solid #1e2d40;">
              <span style="font-family:Space Mono,monospace; font-size:0.65rem;
                           color:#3b82f6; min-width:22px;">{step}</span>
              <span style="font-family:Space Mono,monospace; font-size:0.68rem;
                           color:#06b6d4; min-width:160px;">{title}</span>
              <span style="font-family:Rajdhani,sans-serif; font-size:0.85rem;
                           color:#8b949e;">{desc}</span>
            </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#   PAGE: NDVI MONITOR
# ══════════════════════════════════════════════════════════════════════════════
elif "NDVI" in page:

    st.markdown(
        section_header("🌿", "NDVI Crop Health Monitor",
                        "Sentinel-2 MSI · Band B4 (Red) + B8 (NIR) · NDVI = (NIR−RED)/(NIR+RED)", "#10b981"),
        unsafe_allow_html=True
    )

    tabs = st.tabs([
        "🗺️  GIS NDVI Map",
        "📊  Spectral Analysis",
        "📈  Time Series",
        "🌱  Classification",
        "📋  Statistics Report"
    ])

    # ── Tab 1: GIS NDVI Map ──────────────────────────────────────────────────
    with tabs[0]:
        col_ctrl, col_map2 = st.columns([1, 3], gap="medium")
        with col_ctrl:
            st.markdown("**LAYER DISPLAY**")
            map_layer_choice = st.radio(
                "",
                ["NDVI Heatmap", "Vegetation Classes", "Both Layers"],
                label_visibility="collapsed"
            )
            show_h = map_layer_choice in ("NDVI Heatmap", "Both Layers")
            show_c = map_layer_choice in ("Vegetation Classes", "Both Layers")

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("**NDVI STATS**")
            ndvi_stats = nr["stats"]
            for label, val in [
                ("Mean NDVI",  f"{ndvi_stats['ndvi_mean']:.4f}"),
                ("Median",     f"{ndvi_stats['ndvi_median']:.4f}"),
                ("Std Dev",    f"{ndvi_stats['ndvi_std']:.4f}"),
                ("Max",        f"{ndvi_stats['ndvi_max']:.4f}"),
                ("Min",        f"{ndvi_stats['ndvi_min']:.4f}"),
            ]:
                st.markdown(f"""
                <div style="display:flex; justify-content:space-between;
                            padding:0.25rem 0; border-bottom:1px solid #1e2d40;">
                  <span style="font-family:Space Mono,monospace; font-size:0.63rem;
                               color:#8b949e;">{label}</span>
                  <span style="font-family:Space Mono,monospace; font-size:0.63rem;
                               color:#10b981;">{val}</span>
                </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            ndvi_png = ndvi_map_png_bytes(nr["ndvi"], nr["classification"])
            st.download_button(
                "⬇️ NDVI Map (PNG)",
                data=ndvi_png,
                file_name="ndvi_vegetation_map.png",
                mime="image/png",
                use_container_width=True
            )
            ndvi_csv = ndvi_dataframe_csv_bytes(D["ndvi_df"])
            st.download_button(
                "⬇️ NDVI Pixels (CSV)",
                data=ndvi_csv,
                file_name="ndvi_pixels.csv",
                mime="text/csv",
                use_container_width=True
            )

        with col_map2:
            with st.spinner("Rendering NDVI map..."):
                m_ndvi = build_full_gis_map(
                    D, basemap=basemap,
                    show_flood=False,
                    show_ndvi_heat=show_h,
                    show_ndvi_class=show_c,
                    show_risk=False,
                    show_sensors=show_sensors,
                    show_aoi=True,
                    zoom_start=zoom_level
                )
            st.markdown(map_container_open("NDVI VEGETATION MAP — SENTINEL-2"), unsafe_allow_html=True)
            components.html(map_to_html_string(m_ndvi), height=500)
            st.markdown(map_container_close(), unsafe_allow_html=True)

    # ── Tab 2: Spectral Analysis ─────────────────────────────────────────────
    with tabs[1]:
        col_hist2, col_scatter = st.columns(2, gap="medium")
        with col_hist2:
            st.plotly_chart(
                ndvi_histogram(nr["ndvi"], ndvi_lo, ndvi_hi),
                use_container_width=True
            )
        with col_scatter:
            st.plotly_chart(
                ndvi_nir_red_scatter(nr),
                use_container_width=True
            )

        # Band stats table
        st.markdown("---")
        st.markdown("**BAND STATISTICS**")
        band_df = pd.DataFrame({
            "Band": ["B4 (RED, 665nm)", "B8 (NIR, 842nm)", "NDVI Derived"],
            "Mean":  [
                f"{float(np.nanmean(nr['red'])):.4f}",
                f"{float(np.nanmean(nr['nir'])):.4f}",
                f"{float(np.nanmean(nr['ndvi'])):.4f}"
            ],
            "Std Dev": [
                f"{float(np.nanstd(nr['red'])):.4f}",
                f"{float(np.nanstd(nr['nir'])):.4f}",
                f"{float(np.nanstd(nr['ndvi'])):.4f}"
            ],
            "Min": [
                f"{float(np.nanmin(nr['red'])):.4f}",
                f"{float(np.nanmin(nr['nir'])):.4f}",
                f"{float(np.nanmin(nr['ndvi'])):.4f}"
            ],
            "Max": [
                f"{float(np.nanmax(nr['red'])):.4f}",
                f"{float(np.nanmax(nr['nir'])):.4f}",
                f"{float(np.nanmax(nr['ndvi'])):.4f}"
            ],
            "Unit": ["Reflectance [0,1]", "Reflectance [0,1]", "Dimensionless [-1,1]"]
        })
        st.dataframe(band_df, use_container_width=True, hide_index=True)

    # ── Tab 3: Time Series ───────────────────────────────────────────────────
    with tabs[2]:
        st.plotly_chart(
            ndvi_time_series(ndvi_ts, ndvi_lo, ndvi_hi),
            use_container_width=True
        )
        st.markdown("**NDVI OBSERVATIONS**")
        display_nts = ndvi_ts.copy()
        display_nts["date"] = display_nts["date"].dt.strftime("%d %b %Y")
        st.dataframe(
            display_nts.rename(columns={
                "date": "Date", "ndvi_mean": "Mean NDVI",
                "ndvi_p25": "P25", "ndvi_p75": "P75", "satellite": "Satellite"
            }),
            use_container_width=True, hide_index=True, height=320
        )

    # ── Tab 4: Classification ────────────────────────────────────────────────
    with tabs[3]:
        col_pie2, col_tbl = st.columns(2, gap="medium")
        with col_pie2:
            st.plotly_chart(ndvi_class_donut(metrics), use_container_width=True)

        with col_tbl:
            st.markdown("**VEGETATION CLASS TABLE**")
            by_cls = nr["stats"]["by_class"]
            cls_rows = []
            for cid, cname, emoji in [
                ("low", "Low Vegetation", "🔴"),
                ("moderate", "Moderate Vegetation", "🟡"),
                ("healthy", "Healthy Vegetation", "🟢"),
            ]:
                d = by_cls.get(cid, {})
                cls_rows.append({
                    "Status": emoji,
                    "Class": cname,
                    "Pixels": f"{d.get('pixel_count',0):,}",
                    "Coverage": f"{d.get('coverage_pct',0):.1f}%",
                    "Mean NDVI": f"{d.get('mean_ndvi',0):.4f}",
                    "Std Dev": f"{d.get('std_ndvi',0):.4f}",
                })
            st.dataframe(pd.DataFrame(cls_rows), use_container_width=True, hide_index=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("**NDVI CLASSIFICATION LEGEND**")
            for emoji, label, rng, desc in [
                ("🔴", "Low Vegetation",      f"NDVI < {ndvi_lo}",     "Bare soil, stressed/failed crops, fallow land"),
                ("🟡", "Moderate Vegetation", f"{ndvi_lo} – {ndvi_hi}", "Developing crops, sparse canopy cover"),
                ("🟢", "Healthy Vegetation",  f"NDVI ≥ {ndvi_hi}",     "Peak growth, dense canopy, optimal health"),
            ]:
                st.markdown(f"""
                <div style="display:flex; gap:0.5rem; padding:0.4rem 0;
                            border-bottom:1px solid #1e2d40; align-items:flex-start;">
                  <span style="font-size:1rem;">{emoji}</span>
                  <div>
                    <div style="font-family:Space Mono,monospace; font-size:0.68rem;
                                color:#e2e8f0;">{label} — <span style="color:#06b6d4;">{rng}</span></div>
                    <div style="font-family:Rajdhani,sans-serif; font-size:0.8rem;
                                color:#8b949e;">{desc}</div>
                  </div>
                </div>""", unsafe_allow_html=True)

    # ── Tab 5: Statistics Report ──────────────────────────────────────────────
    with tabs[4]:
        st.markdown("**NDVI PROCESSING CHAIN**")
        ndvi_steps = [
            ("1", "Data Acquisition",     "Sentinel-2 MSI Level-2A (Surface Reflectance)"),
            ("2", "Cloud Masking",        "SCL band: remove cloud shadow, cloud, cirrus pixels"),
            ("3", "Scale Reflectance",    "Divide DN values by 10,000 → [0.0, 1.0] range"),
            ("4", "Band Extraction",      "Select B4 (RED, 665nm) and B8 (NIR, 842nm)"),
            ("5", "NDVI Computation",     "NDVI = (B8 − B4) / (B8 + B4), clipped to [−1, 1]"),
            ("6", "Smoothing",            "Gaussian filter (σ = 0.8) for classification noise reduction"),
            ("7", "Classification",       f"3-class: Low (<{ndvi_lo}) / Moderate / Healthy (≥{ndvi_hi})"),
            ("8", "Statistics",           "Per-class pixel count, mean NDVI, coverage %"),
            ("9", "Temporal Composite",   "Monthly median composite (reduces cloud gaps)"),
            ("10","Export",               "GeoTIFF, PNG, CSV pixel table, GEE script (.js)"),
        ]
        for step, title, desc in ndvi_steps:
            st.markdown(f"""
            <div style="display:flex; gap:0.7rem; align-items:flex-start;
                        padding:0.4rem 0; border-bottom:1px solid #1e2d40;">
              <span style="font-family:Space Mono,monospace; font-size:0.65rem;
                           color:#10b981; min-width:22px;">{step}</span>
              <span style="font-family:Space Mono,monospace; font-size:0.68rem;
                           color:#06b6d4; min-width:160px;">{title}</span>
              <span style="font-family:Rajdhani,sans-serif; font-size:0.85rem;
                           color:#8b949e;">{desc}</span>
            </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#   PAGE: ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
elif "Analytics" in page:

    st.markdown(
        section_header("📊", "Environmental Analytics",
                        "Multi-hazard intelligence · Temporal trends · Risk assessment", "#8b5cf6"),
        unsafe_allow_html=True
    )

    tabs = st.tabs([
        "📈  Temporal Trends",
        "⚠️  Risk Analysis",
        "🔬  Summary Table",
        "📡  Sensor Analytics"
    ])

    with tabs[0]:
        st.plotly_chart(multi_hazard_timeline(flood_ts, ndvi_ts), use_container_width=True)
        col_fts, col_nts = st.columns(2, gap="medium")
        with col_fts:
            st.plotly_chart(flood_time_series(flood_ts, height=280), use_container_width=True)
        with col_nts:
            st.plotly_chart(ndvi_time_series(ndvi_ts, ndvi_lo, ndvi_hi, height=280), use_container_width=True)

    with tabs[1]:
        col_gauge, col_bar_ba = st.columns(2, gap="medium")
        with col_gauge:
            st.plotly_chart(risk_gauge(metrics["risk_score"]), use_container_width=True)
        with col_bar_ba:
            st.plotly_chart(before_after_bar(metrics), use_container_width=True)

        st.markdown("---")
        st.markdown(
            section_header("🌐", "Risk Layer Map", "Composite flood + crop stress index", "#f59e0b"),
            unsafe_allow_html=True
        )
        with st.spinner("Building risk map..."):
            m_risk = build_full_gis_map(
                D, basemap=basemap,
                show_flood=False, show_ndvi_heat=False,
                show_ndvi_class=False, show_risk=True,
                show_sensors=show_sensors, show_aoi=True,
                zoom_start=zoom_level
            )
        st.markdown(map_container_open("COMPOSITE RISK INDEX MAP"), unsafe_allow_html=True)
        components.html(map_to_html_string(m_risk), height=460)
        st.markdown(map_container_close(), unsafe_allow_html=True)

    with tabs[2]:
        st.plotly_chart(
            environmental_stats_table(metrics, flood_threshold),
            use_container_width=True
        )

    with tabs[3]:
        st.plotly_chart(sensor_battery_bar(sensor_df), use_container_width=True)
        st.markdown("**SENSOR REGISTRY**")
        st.dataframe(
            sensor_df.rename(columns={
                "name": "Sensor ID", "type": "Type",
                "lat": "Latitude", "lon": "Longitude",
                "battery_pct": "Battery (%)",
                "signal": "Signal", "last_ping": "Last Ping"
            }),
            use_container_width=True, hide_index=True
        )


# ══════════════════════════════════════════════════════════════════════════════
#   PAGE: IoT SENSORS
# ══════════════════════════════════════════════════════════════════════════════
elif "IoT" in page or "Sensor" in page:

    st.markdown(
        section_header("📡", "IoT Sensor Network",
                        "Phase 3 Preview — Ground Truth Integration · Weather · Soil · NDVI Cameras", "#06b6d4"),
        unsafe_allow_html=True
    )
    st.markdown(
        info_card(
            "PHASE 3 PREVIEW",
            "This panel previews the IoT sensor network planned for Phase 3. "
            "Sensor locations are shown on the GIS map. In Phase 3, live data streams "
            "(soil moisture, water level, weather parameters) will be fused with "
            "Sentinel-1/2 satellite observations for ground-truth validation and "
            "real-time alert generation.",
            "#06b6d4"
        ),
        unsafe_allow_html=True
    )

    col_map_s, col_tbl_s = st.columns([2, 1], gap="medium")
    with col_map_s:
        with st.spinner("Loading sensor map..."):
            m_sensor = build_full_gis_map(
                D, basemap=basemap,
                show_flood=show_flood, show_ndvi_heat=False,
                show_ndvi_class=False, show_risk=False,
                show_sensors=True, show_aoi=True,
                zoom_start=zoom_level
            )
        st.markdown(map_container_open("IoT SENSOR NETWORK MAP"), unsafe_allow_html=True)
        components.html(map_to_html_string(m_sensor), height=480)
        st.markdown(map_container_close(), unsafe_allow_html=True)

    with col_tbl_s:
        st.markdown("**NETWORK STATUS**")
        # Sensor type counts
        type_counts = sensor_df["type"].value_counts()
        for stype, count in type_counts.items():
            st.markdown(f"""
            <div style="display:flex; justify-content:space-between;
                        padding:0.3rem 0; border-bottom:1px solid #1e2d40;">
              <span style="font-family:Rajdhani,sans-serif; font-size:0.85rem;
                           color:#8b949e;">{stype}</span>
              <span style="font-family:Space Mono,monospace; font-size:0.68rem;
                           color:#06b6d4;">{count} units</span>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        avg_bat = sensor_df["battery_pct"].mean()
        strong_sig = (sensor_df["signal"] == "Strong").sum()
        st.markdown(f"""
        <div style="background:#111827; border:1px solid #1e2d40; border-radius:8px; padding:0.9rem;">
          <div style="font-family:Space Mono,monospace; font-size:0.65rem; color:#484f58;
                      letter-spacing:0.1em; margin-bottom:0.6rem;">NETWORK HEALTH</div>
          <div style="font-family:Orbitron,monospace; font-size:1.3rem; color:#10b981;">
            {avg_bat:.0f}%
          </div>
          <div style="font-family:Space Mono,monospace; font-size:0.6rem; color:#8b949e;">
            avg battery · {strong_sig}/{len(sensor_df)} strong signal
          </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.plotly_chart(sensor_battery_bar(sensor_df, height=300), use_container_width=True)
    st.markdown("**SENSOR REGISTRY — FULL**")
    st.dataframe(sensor_df, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
#   PAGE: EXPORT CENTER
# ══════════════════════════════════════════════════════════════════════════════
elif "Export" in page:

    st.markdown(
        section_header("📤", "Export Center",
                        "Download maps · statistics · GeoJSON · GeoTIFF · QGIS-compatible outputs", "#f59e0b"),
        unsafe_allow_html=True
    )

    col_ex1, col_ex2, col_ex3 = st.columns(3, gap="medium")

    # ── Flood Outputs ─────────────────────────────────────────────────────────
    with col_ex1:
        st.markdown(
            section_header("🌊", "Flood Outputs", "", "#3b82f6"),
            unsafe_allow_html=True
        )
        flood_png = flood_map_png_bytes(fr["flood_mask"], fr["backscatter_db"])
        st.download_button(
            "⬇️  Flood Classification Map (PNG)",
            data=flood_png,
            file_name="flood_map.png",
            mime="image/png",
            use_container_width=True
        )
        geojson_data = geojson_bytes(D["flood_geojson"])
        st.download_button(
            "⬇️  Flood Boundaries (GeoJSON)",
            data=geojson_data,
            file_name="flood_classification.geojson",
            mime="application/json",
            use_container_width=True
        )
        # Save GEE JS script
        try:
            sys.path.insert(0, _ROOT)
            from flood_detection.gee_flood import GEEFloodDetector
            det = GEEFloodDetector(
                roi_coords=[bbox["lon_min"], bbox["lat_min"],
                            bbox["lon_max"], bbox["lat_max"]],
                start_date=str(date_start), end_date=str(date_end),
                threshold_db=flood_threshold
            )
            gee_js = det.build_gee_script().encode("utf-8")
            st.download_button(
                "⬇️  GEE Flood Script (JS)",
                data=gee_js,
                file_name="gee_flood_detection.js",
                mime="text/javascript",
                use_container_width=True
            )
        except Exception as e:
            st.caption(f"GEE script unavailable: {e}")

    # ── NDVI Outputs ──────────────────────────────────────────────────────────
    with col_ex2:
        st.markdown(
            section_header("🌿", "NDVI Outputs", "", "#10b981"),
            unsafe_allow_html=True
        )
        ndvi_png = ndvi_map_png_bytes(nr["ndvi"], nr["classification"])
        st.download_button(
            "⬇️  NDVI Vegetation Map (PNG)",
            data=ndvi_png,
            file_name="ndvi_map.png",
            mime="image/png",
            use_container_width=True
        )
        ndvi_csv = ndvi_dataframe_csv_bytes(D["ndvi_df"])
        st.download_button(
            "⬇️  NDVI Pixel Table (CSV)",
            data=ndvi_csv,
            file_name="ndvi_pixels.csv",
            mime="text/csv",
            use_container_width=True
        )
        try:
            from ndvi_monitoring.gee_ndvi import GEENDVIMonitor
            mon = GEENDVIMonitor(
                roi_coords=[bbox["lon_min"], bbox["lat_min"],
                            bbox["lon_max"], bbox["lat_max"]],
                start_date=str(date_start), end_date=str(date_end),
            )
            gee_ndvi_js = mon.build_gee_script().encode("utf-8")
            st.download_button(
                "⬇️  GEE NDVI Script (JS)",
                data=gee_ndvi_js,
                file_name="gee_ndvi_analysis.js",
                mime="text/javascript",
                use_container_width=True
            )
        except Exception as e:
            st.caption(f"GEE script unavailable: {e}")

    # ── Reports & Full Bundle ─────────────────────────────────────────────────
    with col_ex3:
        st.markdown(
            section_header("📋", "Reports & Bundle", "", "#8b5cf6"),
            unsafe_allow_html=True
        )
        csv_bytes = stats_csv_bytes(metrics, flood_ts, ndvi_ts)
        st.download_button(
            "⬇️  Full Statistics (CSV)",
            data=csv_bytes,
            file_name="geotwin_statistics.csv",
            mime="text/csv",
            use_container_width=True
        )

        # Full ZIP
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            info_card("FULL EXPORT BUNDLE",
                      "Downloads all maps (PNG), GeoJSON, CSV statistics, "
                      "GEE scripts, analysis report, and GeoTIFF rasters "
                      "(if rasterio is installed) in a single ZIP file.",
                      "#8b5cf6"),
            unsafe_allow_html=True
        )
        with st.spinner("Packaging ZIP..."):
            zip_bytes = full_export_zip(D)
        ts_str = datetime.now().strftime("%Y%m%d_%H%M")
        st.download_button(
            "📦  Download ALL Outputs (ZIP)",
            data=zip_bytes,
            file_name=f"GeoTwin_Phase2_{ts_str}.zip",
            mime="application/zip",
            use_container_width=True
        )

    # ── Export contents list ──────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("**EXPORT CONTENTS**")
    export_contents = pd.DataFrame({
        "File": [
            "flood_map.png", "ndvi_map.png",
            "flood_classification.geojson", "ndvi_pixels.csv",
            "geotwin_statistics.csv", "analysis_report.json",
            "flood_mask.tif *", "ndvi_values.tif *",
            "ndvi_classification.tif *",
            "gee_flood_detection.js", "gee_ndvi_analysis.js"
        ],
        "Type": [
            "PNG Image", "PNG Image",
            "GeoJSON Vector", "CSV Table",
            "CSV Report", "JSON Metadata",
            "GeoTIFF Raster", "GeoTIFF Raster",
            "GeoTIFF Raster",
            "GEE Script", "GEE Script"
        ],
        "Description": [
            "Flood classification map (print quality)",
            "NDVI vegetation map (print quality)",
            "Flood boundaries — importable in QGIS/ArcGIS",
            "Per-pixel NDVI values with coordinates",
            "Full statistics and time series",
            "Project metadata and analysis summary",
            "Binary flood mask (EPSG:4326) — QGIS ready",
            "Continuous NDVI values (EPSG:4326) — QGIS ready",
            "3-class vegetation map (EPSG:4326) — QGIS ready",
            "Sentinel-1 flood detection for GEE Code Editor",
            "Sentinel-2 NDVI analysis for GEE Code Editor",
        ],
        "QGIS": [
            "No", "No", "✅ Yes", "No",
            "No", "No", "✅ Yes", "✅ Yes", "✅ Yes",
            "No", "No"
        ]
    })
    st.dataframe(export_contents, use_container_width=True, hide_index=True)
    st.caption("* GeoTIFF export requires: pip install rasterio")
