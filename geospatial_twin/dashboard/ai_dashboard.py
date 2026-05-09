"""
================================================================================
  dashboard/ai_dashboard.py
  GeoTwin Phase 3 — AI/ML Intelligence Dashboard
  
  Extends Phase 2 GIS dashboard with full AI classification capabilities.
  Connects directly to:
    - Phase 1: flood_processor, ndvi_processor, demo_data
    - Phase 2: Phase1DataBridge, folium_maps, plotly_charts, ui_components
    - Phase 3: AIPhase3Pipeline, all 4 ML models, evaluation module
  
  Run:
    cd geospatial_twin/dashboard
    streamlit run ai_dashboard.py
================================================================================
"""

import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import pandas as pd
import sys
import os
import io
import json
import zipfile
from datetime import datetime, date

# ── Path resolution ───────────────────────────────────────────────────────────
_DASH  = os.path.dirname(os.path.abspath(__file__))
_ROOT  = os.path.abspath(os.path.join(_DASH, ".."))
for p in [_ROOT, _DASH]:
    if p not in sys.path:
        sys.path.insert(0, p)

# ── Phase 2 imports ───────────────────────────────────────────────────────────
from utils.data_bridge   import Phase1DataBridge
from maps.folium_maps    import build_full_gis_map, map_to_html_string
from components.ui_components import (
    GLOBAL_CSS, hero_header, section_header,
    status_banner, info_card, sidebar_logo, sidebar_section_label
)
from utils.export_manager import (
    flood_map_png_bytes, ndvi_map_png_bytes, stats_csv_bytes
)

# ── Phase 3 imports ───────────────────────────────────────────────────────────
from components.ai_ui_components import (
    AI_EXTRA_CSS, ai_hero_banner, model_result_card,
    ai_kpi_row, ai_section_header, methodology_card
)
from ai_models.ai_pipeline import AIPhase3Pipeline
from ai_models.evaluation.accuracy_metrics import (
    ModelEvaluator, kappa_coefficient, compute_per_class_iou
)
from ai_models.evaluation.confusion_matrix import (
    plot_confusion_matrix, plot_metrics_comparison,
    plot_feature_importance, plot_cv_scores,
    plot_prediction_heatmap, plot_prediction_comparison,
    plot_agreement_map, plot_accuracy_radar,
    plot_per_class_metrics, plot_prediction_distribution
)


# ─────────────────────────────────────────────────────────────────────────────
#   PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GeoTwin Phase 3 · AI Intelligence",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown(GLOBAL_CSS,       unsafe_allow_html=True)
st.markdown(AI_EXTRA_CSS,     unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#   SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(sidebar_logo(), unsafe_allow_html=True)

    st.markdown(sidebar_section_label("Navigation"), unsafe_allow_html=True)
    page = st.radio(
        "",
        ["🤖  AI Overview",
         "🌊  Flood AI",
         "🌿  Vegetation AI",
         "📊  Model Comparison",
         "🔬  Feature Analysis",
         "📤  AI Export"],
        label_visibility="collapsed"
    )

    st.markdown(sidebar_section_label("Phase 1 Data"), unsafe_allow_html=True)
    flood_threshold = st.slider("SAR Threshold (dB)", -24, -10, -16, 1)
    ndvi_lo  = st.slider("NDVI Low Thresh",     0.0, 0.4, 0.2, 0.05)
    ndvi_hi  = st.slider("NDVI Healthy Thresh", 0.3, 0.9, 0.5, 0.05)

    st.markdown(sidebar_section_label("RF Hyperparameters"), unsafe_allow_html=True)
    rf_n_trees   = st.slider("RF Trees",      50, 300, 150, 25)
    rf_max_depth = st.slider("RF Max Depth",   5,  25,  12,  1)
    rf_n_samples = st.slider("RF Samples/class (×100)", 5, 50, 30, 5)

    st.markdown(sidebar_section_label("SVM Hyperparameters"), unsafe_allow_html=True)
    svm_C     = st.select_slider("SVM C",     [0.1, 1.0, 5.0, 10.0, 50.0], value=10.0)
    svm_gamma = st.selectbox("SVM Gamma",     ["scale", "auto"])
    svm_n_smp = st.slider("SVM Samples/class (×100)", 5, 30, 20, 5)

    st.markdown(sidebar_section_label("Settings"), unsafe_allow_html=True)
    run_cv      = st.checkbox("5-Fold Cross-Validation", value=True)
    basemap     = st.selectbox("Map Basemap",
                               ["CartoDB Dark", "Satellite (Esri)", "OpenStreetMap"])
    zoom_level  = st.slider("Map Zoom", 5, 14, 8)

    st.markdown("---")
    retrain_btn = st.button("🔁 Retrain All Models", use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
#   LOAD PHASE 1 DATA
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_phase1(threshold: float, lo: float, hi: float) -> dict:
    bridge = Phase1DataBridge(flood_threshold=threshold, ndvi_lo=lo, ndvi_hi=hi)
    return bridge.get_all()


# ─────────────────────────────────────────────────────────────────────────────
#   TRAIN / CACHE AI MODELS
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(
    show_spinner=False,
    hash_funcs={dict: lambda d: str(sorted(str(d).encode()))}
)
def run_ai_pipeline(
    threshold: float, lo: float, hi: float,
    n_trees: int, max_depth: int, rf_n: int,
    c: float, gamma: str, svm_n: int,
    cv: bool, _cache_bust: int = 0
) -> dict:
    """Train all 4 models and return the unified results bundle."""
    p1 = Phase1DataBridge(flood_threshold=threshold, ndvi_lo=lo, ndvi_hi=hi)
    data = p1.get_all()

    pipeline = AIPhase3Pipeline(
        run_cv=cv,
        rf_flood_params={
            "n_estimators": n_trees,
            "max_depth": max_depth,
            "n_samples_per_class": rf_n * 100
        },
        svm_flood_params={
            "C": c, "gamma": gamma,
            "n_samples_per_class": svm_n * 100
        },
        rf_ndvi_params={
            "n_estimators": n_trees,
            "max_depth": max_depth,
            "n_samples_per_class": rf_n * 100
        },
        svm_ndvi_params={
            "C": c, "gamma": gamma,
            "n_samples_per_class": svm_n * 100
        }
    )
    return pipeline.run(data)


# ── Cache-bust counter in session state (for retrain button) ─────────────────
if "cache_bust" not in st.session_state:
    st.session_state["cache_bust"] = 0

if retrain_btn:
    st.session_state["cache_bust"] += 1
    st.cache_data.clear()

# ── Load everything ───────────────────────────────────────────────────────────
with st.spinner("🛰️ Loading Phase 1 data..."):
    P1 = load_phase1(flood_threshold, ndvi_lo, ndvi_hi)

with st.spinner("🤖 Training AI models (RF + SVM × 2 tasks)..."):
    AI = run_ai_pipeline(
        flood_threshold, ndvi_lo, ndvi_hi,
        rf_n_trees, rf_max_depth, rf_n_samples,
        svm_C, svm_gamma, svm_n_smp,
        run_cv, _cache_bust=st.session_state["cache_bust"]
    )

# Convenience aliases
rf_f   = AI["rf_flood"]
svm_f  = AI["svm_flood"]
rf_n   = AI["rf_ndvi"]
svm_n_ = AI["svm_ndvi"]
summ   = AI["summary"]
kappa  = AI["kappa"]


# ─────────────────────────────────────────────────────────────────────────────
#   HEADERS (all pages)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(ai_hero_banner(), unsafe_allow_html=True)
st.markdown(
    status_banner(
        mode="AI/ML Phase 3",
        date_start="2024-06-01",
        date_end="2024-08-31",
        n_sensors=len(P1["sensor_points"]),
        phase1_ok=True
    ),
    unsafe_allow_html=True
)
st.markdown(ai_kpi_row(summ), unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#   PAGE: AI OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if "Overview" in page:

    st.markdown(
        ai_section_header("🤖", "AI Model Overview",
                          "All 4 trained models · RF + SVM · Flood + Vegetation"),
        unsafe_allow_html=True
    )

    # ── 4 model cards in a 2×2 grid ──────────────────────────────────────────
    col1, col2 = st.columns(2, gap="medium")
    with col1:
        st.markdown(model_result_card(
            "Random Forest", "FLOOD",
            accuracy=rf_f["accuracy"], f1=rf_f["f1_score"],
            precision=rf_f["precision"], recall=rf_f["recall"],
            cv_mean=rf_f.get("cv_mean"), kappa=kappa.get("RF · Flood", 0),
            n_train=rf_f["n_train"],
            is_best=(AI["best_flood_model"] == "RF · Flood"),
            extra_info=f"Trees: {rf_f['n_estimators']} · Max depth: {rf_f['max_depth']} · OOB: {rf_f['oob_score']}"
        ), unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(model_result_card(
            "Random Forest", "NDVI",
            accuracy=rf_n["accuracy"],  f1=rf_n["f1_score"],
            precision=rf_n["precision"], recall=rf_n["recall"],
            cv_mean=rf_n.get("cv_mean"), kappa=kappa.get("RF · NDVI", 0),
            n_train=rf_n["n_train"],
            is_best=(AI["best_ndvi_model"] == "RF · NDVI"),
            extra_info=f"Trees: {rf_n['n_estimators']} · Max depth: {rf_n['max_depth']} · OOB: {rf_n['oob_score']}"
        ), unsafe_allow_html=True)

    with col2:
        st.markdown(model_result_card(
            "SVM (RBF)", "FLOOD",
            accuracy=svm_f["accuracy"], f1=svm_f["f1_score"],
            precision=svm_f["precision"], recall=svm_f["recall"],
            cv_mean=svm_f.get("cv_mean"), kappa=kappa.get("SVM · Flood", 0),
            n_train=svm_f["n_train"],
            is_best=(AI["best_flood_model"] == "SVM · Flood"),
            extra_info=f"Kernel: {svm_f['kernel']} · C: {svm_f['C']} · SVs: {svm_f.get('n_support_vectors',0)}"
        ), unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(model_result_card(
            "SVM (RBF)", "NDVI",
            accuracy=svm_n_["accuracy"], f1=svm_n_["f1_score"],
            precision=svm_n_["precision"], recall=svm_n_["recall"],
            cv_mean=svm_n_.get("cv_mean"), kappa=kappa.get("SVM · NDVI", 0),
            n_train=svm_n_["n_train"],
            is_best=(AI["best_ndvi_model"] == "SVM · NDVI"),
            extra_info=f"Kernel: {svm_n_['kernel']} · C: {svm_n_['C']} · SVs: {svm_n_.get('n_support_vectors',0)}"
        ), unsafe_allow_html=True)

    st.markdown("---")

    # ── Radar charts side-by-side ─────────────────────────────────────────────
    col_r1, col_r2 = st.columns(2, gap="medium")
    with col_r1:
        st.plotly_chart(
            plot_accuracy_radar(rf_f, svm_f, "flood"),
            use_container_width=True
        )
    with col_r2:
        st.plotly_chart(
            plot_accuracy_radar(rf_n, svm_n_, "ndvi"),
            use_container_width=True
        )

    # ── Full comparison table ─────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        ai_section_header("📋", "Model Comparison Table", "All metrics across all 4 models"),
        unsafe_allow_html=True
    )
    st.dataframe(
        AI["all_comparison_df"].style.format("{:.4f}", na_rep="—")
                                    .highlight_max(color="#0d2d1a", axis=0),
        use_container_width=True
    )


# ══════════════════════════════════════════════════════════════════════════════
#   PAGE: FLOOD AI
# ══════════════════════════════════════════════════════════════════════════════
elif "Flood AI" in page:

    st.markdown(
        ai_section_header("🌊", "Flood AI Classification",
                          "SAR Sentinel-1 · Random Forest vs SVM · 10 SAR features"),
        unsafe_allow_html=True
    )

    model_choice = st.selectbox(
        "Select Model for Detail View",
        ["Random Forest", "SVM"],
        key="flood_model_sel"
    )
    active = rf_f if model_choice == "Random Forest" else svm_f

    tabs = st.tabs([
        "🗺️  AI Prediction Map",
        "🔢  Confusion Matrix",
        "📊  Metrics & CV",
        "🆚  RF vs SVM",
        "📋  Classification Report"
    ])

    # ── Tab 1: Prediction Map ─────────────────────────────────────────────────
    with tabs[0]:
        col_map, col_info = st.columns([2, 1], gap="medium")
        with col_map:
            st.plotly_chart(
                plot_prediction_heatmap(
                    active["pred_raster"],
                    title=f"{model_choice} — Flood Prediction Raster",
                    mode="flood", height=440
                ),
                use_container_width=True
            )
        with col_info:
            st.markdown("**PREDICTION STATISTICS**")
            pred_r = active["pred_raster"]
            flooded_pred_pct = float(pred_r.mean()) * 100
            phase1_pct       = float(P1["flood_result"]["flood_mask"].mean()) * 100
            agree_pct        = AI["flood_agreement_pct"]

            for label, val, color in [
                ("Model",         model_choice,               "#8b949e"),
                ("Flooded (pred)",f"{flooded_pred_pct:.1f}%", "#3b82f6"),
                ("Flooded (P1)",  f"{phase1_pct:.1f}%",       "#06b6d4"),
                ("Model Agree.",  f"{agree_pct:.1f}%",        "#10b981"),
                ("Accuracy",      f"{active['accuracy']*100:.2f}%", "#06b6d4"),
                ("F1-Score",      f"{active['f1_score']:.4f}", "#06b6d4"),
                ("AUC-ROC",       f"{active.get('roc_auc','N/A')}",  "#f59e0b"),
                ("Kappa κ",       f"{kappa.get(f'{model_choice[:2]}· Flood',0):.4f}", "#8b5cf6"),
                ("Train samples", f"{active['n_train']:,}",    "#8b949e"),
                ("Test samples",  f"{active['n_test']:,}",     "#8b949e"),
                ("Features used", f"{active['n_features']}",  "#8b949e"),
            ]:
                st.markdown(f"""
                <div style="display:flex; justify-content:space-between;
                            padding:0.28rem 0; border-bottom:1px solid #1e2d40;">
                  <span style="font-family:Space Mono,monospace; font-size:0.62rem;
                               color:#8b949e;">{label}</span>
                  <span style="font-family:Space Mono,monospace; font-size:0.62rem;
                               color:{color};">{val}</span>
                </div>""", unsafe_allow_html=True)

            # ── GIS Map with AI layer ─────────────────────────────────────────
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("**AI PREDICTION ON GIS MAP**")
            with st.spinner("Loading map..."):
                m_ai_flood = build_full_gis_map(
                    P1, basemap=basemap,
                    show_flood=True, show_ndvi_heat=False,
                    show_ndvi_class=False, show_risk=False,
                    show_sensors=False, show_aoi=True,
                    zoom_start=zoom_level
                )
            components.html(map_to_html_string(m_ai_flood), height=280)

    # ── Tab 2: Confusion Matrix ───────────────────────────────────────────────
    with tabs[1]:
        col_cm, col_iou = st.columns([2, 1], gap="medium")
        with col_cm:
            st.plotly_chart(
                plot_confusion_matrix(
                    active["confusion_matrix"],
                    active["class_names"],
                    model_name=f"{model_choice} Flood",
                    height=420
                ),
                use_container_width=True
            )
        with col_iou:
            st.markdown("**CONFUSION MATRIX BREAKDOWN**")
            cm_arr    = np.array(active["confusion_matrix"])
            total_test = cm_arr.sum()
            class_names = active["class_names"]
            for i, name in enumerate(class_names):
                tp = cm_arr[i, i]
                fp = cm_arr[:, i].sum() - tp
                fn = cm_arr[i, :].sum() - tp
                tn = total_test - tp - fp - fn
                st.markdown(f"""
                <div style="background:#111827; border:1px solid #1e2d40;
                            border-radius:7px; padding:0.7rem; margin-bottom:0.5rem;">
                  <div style="font-family:Orbitron,monospace; font-size:0.68rem;
                              color:#06b6d4; margin-bottom:0.4rem;">
                    {name.replace('Non-Flooded (Dry)','Dry').replace('Flooded (Water)','Flooded')}
                  </div>
                  <div style="display:grid; grid-template-columns:1fr 1fr;
                              gap:0.3rem; font-family:Space Mono,monospace; font-size:0.62rem;">
                    <span style="color:#10b981;">TP: {tp}</span>
                    <span style="color:#ef4444;">FP: {fp}</span>
                    <span style="color:#f59e0b;">FN: {fn}</span>
                    <span style="color:#8b949e;">TN: {tn}</span>
                  </div>
                </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            iou_df = compute_per_class_iou(active["confusion_matrix"], class_names)
            st.markdown("**IoU PER CLASS**")
            st.dataframe(iou_df, use_container_width=True, hide_index=True)

            kappa_val = kappa_coefficient(active["confusion_matrix"])
            kap_color = "#10b981" if kappa_val > 0.8 else ("#f59e0b" if kappa_val > 0.6 else "#ef4444")
            kap_label = "Near-Perfect" if kappa_val > 0.8 else ("Substantial" if kappa_val > 0.6 else "Moderate")
            st.markdown(f"""
            <div style="background:#111827; border:1px solid #1e2d40; border-radius:8px;
                        padding:0.8rem; margin-top:0.5rem; text-align:center;">
              <div style="font-family:Space Mono,monospace; font-size:0.6rem;
                          color:#8b949e; margin-bottom:0.3rem;">COHEN'S KAPPA</div>
              <div style="font-family:Orbitron,monospace; font-size:1.4rem;
                          color:{kap_color};">{kappa_val:.4f}</div>
              <div style="font-family:Space Mono,monospace; font-size:0.62rem;
                          color:{kap_color};">{kap_label} Agreement</div>
            </div>""", unsafe_allow_html=True)

    # ── Tab 3: Metrics & CV ───────────────────────────────────────────────────
    with tabs[2]:
        col_bar, col_cv = st.columns(2, gap="medium")
        with col_bar:
            st.plotly_chart(
                plot_metrics_comparison(
                    AI["flood_metrics_bar"],
                    title="Flood Models — Metrics Comparison",
                    height=320
                ),
                use_container_width=True
            )
        with col_cv:
            st.plotly_chart(
                plot_cv_scores(rf_f, svm_f, model_type="Flood"),
                use_container_width=True
            )

        st.markdown("**FULL COMPARISON TABLE — FLOOD MODELS**")
        st.dataframe(
            AI["flood_comparison_df"].style.format("{:.4f}", na_rep="—")
                                          .highlight_max(color="#0d2d1a", axis=0),
            use_container_width=True
        )

    # ── Tab 4: RF vs SVM ─────────────────────────────────────────────────────
    with tabs[3]:
        st.plotly_chart(
            plot_prediction_comparison(
                rf_f["pred_raster"], svm_f["pred_raster"],
                mode="flood", height=400
            ),
            use_container_width=True
        )
        col_agree, col_dist = st.columns(2, gap="medium")
        with col_agree:
            st.plotly_chart(
                plot_agreement_map(
                    rf_f["pred_raster"], svm_f["pred_raster"],
                    title="RF vs SVM Agreement — Flood",
                    height=360
                ),
                use_container_width=True
            )
        with col_dist:
            st.markdown("**PREDICTION DISTRIBUTIONS**")
            col_d1, col_d2 = st.columns(2)
            with col_d1:
                st.plotly_chart(
                    plot_prediction_distribution(
                        rf_f["pred_raster"], rf_f["class_names"], "RF", height=280
                    ), use_container_width=True
                )
            with col_d2:
                st.plotly_chart(
                    plot_prediction_distribution(
                        svm_f["pred_raster"], svm_f["class_names"], "SVM", height=280
                    ), use_container_width=True
                )

    # ── Tab 5: Classification Report ─────────────────────────────────────────
    with tabs[4]:
        st.markdown(f"**{model_choice.upper()} — DETAILED CLASSIFICATION REPORT**")
        report_dict = active["classification_report"]
        report_rows = []
        for cls in active["class_names"] + ["macro avg", "weighted avg"]:
            if cls in report_dict:
                d = report_dict[cls]
                report_rows.append({
                    "Class":      cls,
                    "Precision":  round(d.get("precision", 0), 4),
                    "Recall":     round(d.get("recall",    0), 4),
                    "F1-Score":   round(d.get("f1-score",  0), 4),
                    "Support":    int(d.get("support",     0))
                })
        st.dataframe(pd.DataFrame(report_rows), use_container_width=True, hide_index=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**TRAINING CONFIGURATION**")
        config_rows = []
        if model_choice == "Random Forest":
            config_rows = [
                ("Algorithm",      "Random Forest (Bagging Ensemble)"),
                ("Trees",          str(active["n_estimators"])),
                ("Max Depth",      str(active["max_depth"])),
                ("Max Features",   "sqrt(n_features) — Breiman's default"),
                ("OOB Score",      str(active.get("oob_score", "N/A"))),
                ("Class Weight",   "Balanced — handles class imbalance"),
                ("n_jobs",         "-1 (all CPU cores)"),
                ("Feature Count",  str(active["n_features"])),
            ]
        else:
            config_rows = [
                ("Algorithm",      "Support Vector Machine"),
                ("Kernel",         active.get("kernel", "rbf").upper()),
                ("C",              str(active.get("C", "—"))),
                ("Gamma",          str(active.get("gamma", "—"))),
                ("Support Vectors",str(active.get("n_support_vectors", "—"))),
                ("Class Weight",   "Balanced"),
                ("Probability",    "True (Platt scaling)"),
                ("Feature Count",  str(active["n_features"])),
            ]
        for k, v in config_rows:
            st.markdown(f"""
            <div style="display:flex; gap:1rem; padding:0.28rem 0;
                        border-bottom:1px solid #1e2d40;">
              <span style="font-family:Space Mono,monospace; font-size:0.62rem;
                           color:#8b949e; min-width:160px;">{k}</span>
              <span style="font-family:Space Mono,monospace; font-size:0.62rem;
                           color:#06b6d4;">{v}</span>
            </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#   PAGE: VEGETATION AI
# ══════════════════════════════════════════════════════════════════════════════
elif "Vegetation AI" in page:

    st.markdown(
        ai_section_header("🌿", "Vegetation AI Classification",
                          "Sentinel-2 NDVI · 3-class · RF vs SVM · 10 spectral features"),
        unsafe_allow_html=True
    )

    model_choice_n = st.selectbox(
        "Select Model",
        ["Random Forest", "SVM"],
        key="ndvi_model_sel"
    )
    active_n = rf_n if model_choice_n == "Random Forest" else svm_n_

    tabs = st.tabs([
        "🗺️  AI Vegetation Map",
        "🔢  Confusion Matrix",
        "📊  Metrics & CV",
        "🆚  RF vs SVM",
        "📋  Classification Report"
    ])

    with tabs[0]:
        col_vm, col_vi = st.columns([2, 1], gap="medium")
        with col_vm:
            st.plotly_chart(
                plot_prediction_heatmap(
                    active_n["pred_raster"],
                    title=f"{model_choice_n} — Vegetation Prediction Raster",
                    mode="ndvi", height=440
                ),
                use_container_width=True
            )
        with col_vi:
            st.markdown("**VEGETATION PREDICTION STATS**")
            pred_cls = active_n["pred_raster"]
            for cid, cname, color in [
                (0, "Low Vegetation",      "#ef4444"),
                (1, "Moderate Vegetation", "#f59e0b"),
                (2, "Healthy Vegetation",  "#10b981"),
            ]:
                pct = float(np.mean(pred_cls == cid)) * 100
                st.markdown(f"""
                <div style="display:flex; justify-content:space-between;
                            padding:0.28rem 0; border-bottom:1px solid #1e2d40;">
                  <span style="font-family:Rajdhani,sans-serif; font-size:0.82rem;
                               color:#8b949e;">{cname}</span>
                  <span style="font-family:Space Mono,monospace; font-size:0.68rem;
                               color:{color};">{pct:.1f}%</span>
                </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            for label, val, color in [
                ("Accuracy",    f"{active_n['accuracy']*100:.2f}%",  "#06b6d4"),
                ("F1-Score",    f"{active_n['f1_score']:.4f}",       "#06b6d4"),
                ("OOB Score",   f"{active_n.get('oob_score','N/A')}", "#10b981"),
                ("Kappa κ",     f"{kappa.get(f'{model_choice_n[:2]}· NDVI',kappa.get('RF · NDVI',0)):.4f}", "#8b5cf6"),
                ("Train N",     f"{active_n['n_train']:,}",           "#8b949e"),
                ("Features",    f"{active_n['n_features']}",          "#8b949e"),
            ]:
                st.markdown(f"""
                <div style="display:flex; justify-content:space-between;
                            padding:0.28rem 0; border-bottom:1px solid #1e2d40;">
                  <span style="font-family:Space Mono,monospace; font-size:0.62rem;
                               color:#8b949e;">{label}</span>
                  <span style="font-family:Space Mono,monospace; font-size:0.62rem;
                               color:{color};">{val}</span>
                </div>""", unsafe_allow_html=True)

    with tabs[1]:
        col_cm2, col_pc = st.columns([2, 1], gap="medium")
        with col_cm2:
            st.plotly_chart(
                plot_confusion_matrix(
                    active_n["confusion_matrix"],
                    active_n["class_names"],
                    model_name=f"{model_choice_n} NDVI",
                    height=420
                ),
                use_container_width=True
            )
        with col_pc:
            if active_n.get("per_class_stats"):
                st.markdown("**PER-CLASS STATS**")
                for row in active_n["per_class_stats"]:
                    color = {"Low Vegetation": "#ef4444",
                             "Moderate Vegetation": "#f59e0b",
                             "Healthy Vegetation": "#10b981"}.get(row["class"], "#8b949e")
                    st.markdown(f"""
                    <div style="background:#111827; border:1px solid #1e2d40;
                                border-left:2px solid {color}; border-radius:6px;
                                padding:0.6rem; margin-bottom:0.4rem;">
                      <div style="font-family:Orbitron,monospace; font-size:0.65rem;
                                  color:{color}; margin-bottom:0.3rem;">
                        {row['class'].replace(' Vegetation',' Veg.')}
                      </div>
                      <div style="font-family:Space Mono,monospace; font-size:0.6rem;
                                  color:#8b949e; display:flex; gap:0.5rem;">
                        <span>P:{row['precision']:.3f}</span>
                        <span>R:{row['recall']:.3f}</span>
                        <span>F1:{row['f1']:.3f}</span>
                      </div>
                    </div>""", unsafe_allow_html=True)

            iou_df_n = compute_per_class_iou(active_n["confusion_matrix"], active_n["class_names"])
            st.markdown("**IoU**")
            st.dataframe(iou_df_n, use_container_width=True, hide_index=True)

    with tabs[2]:
        col_b2, col_cv2 = st.columns(2, gap="medium")
        with col_b2:
            st.plotly_chart(
                plot_metrics_comparison(
                    AI["ndvi_metrics_bar"],
                    title="Vegetation Models — Metrics Comparison",
                    height=320
                ),
                use_container_width=True
            )
        with col_cv2:
            st.plotly_chart(
                plot_cv_scores(rf_n, svm_n_, model_type="NDVI"),
                use_container_width=True
            )

        if active_n.get("per_class_stats"):
            st.plotly_chart(
                plot_per_class_metrics(
                    active_n["per_class_stats"],
                    model_name=model_choice_n
                ),
                use_container_width=True
            )

    with tabs[3]:
        st.plotly_chart(
            plot_prediction_comparison(
                rf_n["pred_raster"], svm_n_["pred_raster"],
                mode="ndvi", height=400
            ),
            use_container_width=True
        )
        col_a2, col_d2 = st.columns(2, gap="medium")
        with col_a2:
            st.plotly_chart(
                plot_agreement_map(
                    rf_n["pred_raster"], svm_n_["pred_raster"],
                    title="RF vs SVM Agreement — Vegetation",
                    height=340
                ),
                use_container_width=True
            )
        with col_d2:
            st.plotly_chart(
                plot_prediction_distribution(
                    rf_n["pred_raster"], rf_n["class_names"], "RF NDVI", height=340
                ),
                use_container_width=True
            )

    with tabs[4]:
        report_dict_n = active_n["classification_report"]
        report_rows_n = []
        for cls in active_n["class_names"] + ["macro avg", "weighted avg"]:
            if cls in report_dict_n:
                d = report_dict_n[cls]
                report_rows_n.append({
                    "Class":     cls,
                    "Precision": round(d.get("precision", 0), 4),
                    "Recall":    round(d.get("recall",    0), 4),
                    "F1-Score":  round(d.get("f1-score",  0), 4),
                    "Support":   int(d.get("support",     0))
                })
        st.dataframe(pd.DataFrame(report_rows_n), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
#   PAGE: MODEL COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
elif "Comparison" in page:

    st.markdown(
        ai_section_header("📊", "Model Comparison",
                          "RF vs SVM · Flood vs NDVI · Accuracy · Radar · CV"),
        unsafe_allow_html=True
    )

    tabs = st.tabs([
        "📈  Metrics Bar",
        "🕸️  Radar Charts",
        "📉  CV Scores",
        "📋  Full Table"
    ])

    with tabs[0]:
        col_fb, col_nb = st.columns(2, gap="medium")
        with col_fb:
            st.plotly_chart(
                plot_metrics_comparison(AI["flood_metrics_bar"],
                                        "Flood Detection Models", height=340),
                use_container_width=True
            )
        with col_nb:
            st.plotly_chart(
                plot_metrics_comparison(AI["ndvi_metrics_bar"],
                                        "Vegetation Classification Models", height=340),
                use_container_width=True
            )

    with tabs[1]:
        col_fr, col_nr = st.columns(2, gap="medium")
        with col_fr:
            st.plotly_chart(
                plot_accuracy_radar(rf_f, svm_f, "flood"), use_container_width=True
            )
        with col_nr:
            st.plotly_chart(
                plot_accuracy_radar(rf_n, svm_n_, "ndvi"), use_container_width=True
            )

    with tabs[2]:
        col_fc, col_nc = st.columns(2, gap="medium")
        with col_fc:
            st.plotly_chart(
                plot_cv_scores(rf_f, svm_f, "Flood"), use_container_width=True
            )
        with col_nc:
            st.plotly_chart(
                plot_cv_scores(rf_n, svm_n_, "NDVI"), use_container_width=True
            )

    with tabs[3]:
        st.markdown("**ALL MODELS — FULL COMPARISON**")
        st.dataframe(
            AI["all_comparison_df"].style.format("{:.4f}", na_rep="—")
                                        .highlight_max(color="#0d2d1a", axis=0),
            use_container_width=True
        )
        st.markdown("**KAPPA COEFFICIENTS**")
        kappa_df = pd.DataFrame([
            {"Model": k, "Kappa": v,
             "Interpretation": (
                 "Near-Perfect" if v > 0.8 else
                 "Substantial"  if v > 0.6 else
                 "Moderate"     if v > 0.4 else "Fair"
             )}
            for k, v in kappa.items()
        ])
        st.dataframe(kappa_df, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
#   PAGE: FEATURE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif "Feature" in page:

    st.markdown(
        ai_section_header("🔬", "Feature Importance Analysis",
                          "Which SAR / NDVI features drive classification most?"),
        unsafe_allow_html=True
    )

    tabs = st.tabs([
        "🌊  Flood Features (RF)",
        "🌊  Flood Features (SVM)",
        "🌿  NDVI Features (RF)",
        "🌿  NDVI Features (SVM)"
    ])

    with tabs[0]:
        st.plotly_chart(
            plot_feature_importance(rf_f["feature_importance_df"],
                                    "Random Forest — Flood", top_n=10),
            use_container_width=True
        )
        st.markdown("**FEATURE DICTIONARY — SAR FLOOD**")
        feat_desc = {
            "vv_db":         "SAR VV backscatter (dB) — primary flood discriminant",
            "vh_db":         "SAR VH backscatter (dB) — cross-polarisation channel",
            "vv_vh_ratio":   "VV/VH ratio (linear) — surface roughness proxy",
            "vv_vh_diff":    "VV − VH difference (dB) — polarimetric index",
            "local_mean_3":  "3×3 local mean — spatial averaging of VV",
            "local_std_3":   "3×3 local std — texture measure (speckle level)",
            "local_mean_7":  "7×7 local mean — larger context smoothing",
            "local_std_7":   "7×7 local std — broader texture variability",
            "gradient_mag":  "Sobel gradient magnitude — edge / boundary strength",
            "entropy_local": "Coefficient of variation — local entropy proxy"
        }
        rows = [{"Feature": k, "Description": v} for k, v in feat_desc.items()]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    with tabs[1]:
        st.plotly_chart(
            plot_feature_importance(svm_f["feature_importance_df"],
                                    "SVM — Flood (support vector proxy)", top_n=10),
            use_container_width=True
        )
        st.markdown(
            info_card("SVM FEATURE IMPORTANCE NOTE",
                      "For RBF kernel SVMs, true feature importance doesn't exist. "
                      "The chart shows the mean absolute magnitude of features across "
                      "support vectors — a proxy for feature influence in the kernel space. "
                      "For true feature importance, use the Random Forest model.",
                      "#f59e0b"),
            unsafe_allow_html=True
        )

    with tabs[2]:
        st.plotly_chart(
            plot_feature_importance(rf_n["feature_importance_df"],
                                    "Random Forest — Vegetation", top_n=10),
            use_container_width=True
        )
        st.markdown("**FEATURE DICTIONARY — NDVI VEGETATION**")
        feat_desc_n = {
            "ndvi":         "NDVI = (NIR−RED)/(NIR+RED) — primary vegetation index",
            "nir":          "NIR band (B8, 842nm) — chlorophyll reflectance",
            "red":          "Red band (B4, 665nm) — chlorophyll absorption",
            "green":        "Green band (B3, 560nm) — canopy reflectance",
            "sr_index":     "Simple Ratio = NIR/RED — vegetation density",
            "local_mean_3": "3×3 NDVI local mean — neighbourhood average",
            "local_std_3":  "3×3 NDVI local std — patch heterogeneity",
            "local_mean_7": "7×7 NDVI local mean — landscape context",
            "gradient_mag": "NDVI spatial gradient — vegetation boundary edges",
            "evi":          "Enhanced Vegetation Index (EVI) — atmosphere-corrected index",
        }
        rows_n = [{"Feature": k, "Description": v} for k, v in feat_desc_n.items()]
        st.dataframe(pd.DataFrame(rows_n), use_container_width=True, hide_index=True)

    with tabs[3]:
        st.plotly_chart(
            plot_feature_importance(svm_n_["feature_importance_df"],
                                    "SVM — Vegetation (support vector proxy)", top_n=10),
            use_container_width=True
        )


# ══════════════════════════════════════════════════════════════════════════════
#   PAGE: AI EXPORT
# ══════════════════════════════════════════════════════════════════════════════
elif "Export" in page:

    st.markdown(
        ai_section_header("📤", "AI Export Center",
                          "Download model reports · prediction maps · confusion matrices · ZIP bundle"),
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3, gap="medium")

    # ── Flood AI Exports ──────────────────────────────────────────────────────
    with col1:
        st.markdown(
            section_header("🌊", "Flood AI Outputs", "", "#3b82f6"),
            unsafe_allow_html=True
        )
        # JSON report
        flood_report = {
            "task": "Flood Detection",
            "models": {
                "Random Forest": {k: v for k, v in rf_f.items()
                                  if k not in ["pred_raster", "feature_importance_df",
                                                "classification_report"]},
                "SVM": {k: v for k, v in svm_f.items()
                        if k not in ["pred_raster", "feature_importance_df",
                                      "classification_report"]}
            }
        }
        st.download_button(
            "⬇️  Flood AI Report (JSON)",
            data=json.dumps(flood_report, indent=2, default=str).encode(),
            file_name="flood_ai_report.json",
            mime="application/json",
            use_container_width=True
        )
        # Comparison CSV
        st.download_button(
            "⬇️  Flood Comparison (CSV)",
            data=AI["flood_comparison_df"].to_csv().encode(),
            file_name="flood_model_comparison.csv",
            mime="text/csv",
            use_container_width=True
        )
        # RF feature importance
        st.download_button(
            "⬇️  RF Flood Features (CSV)",
            data=rf_f["feature_importance_df"].to_csv(index=False).encode(),
            file_name="rf_flood_feature_importance.csv",
            mime="text/csv",
            use_container_width=True
        )

    # ── NDVI AI Exports ───────────────────────────────────────────────────────
    with col2:
        st.markdown(
            section_header("🌿", "Vegetation AI Outputs", "", "#10b981"),
            unsafe_allow_html=True
        )
        ndvi_report = {
            "task": "Vegetation Health Classification",
            "models": {
                "Random Forest": {k: v for k, v in rf_n.items()
                                  if k not in ["pred_raster", "feature_importance_df",
                                                "classification_report", "per_class_stats"]},
                "SVM": {k: v for k, v in svm_n_.items()
                        if k not in ["pred_raster", "feature_importance_df",
                                      "classification_report", "per_class_stats"]}
            }
        }
        st.download_button(
            "⬇️  NDVI AI Report (JSON)",
            data=json.dumps(ndvi_report, indent=2, default=str).encode(),
            file_name="ndvi_ai_report.json",
            mime="application/json",
            use_container_width=True
        )
        st.download_button(
            "⬇️  NDVI Comparison (CSV)",
            data=AI["ndvi_comparison_df"].to_csv().encode(),
            file_name="ndvi_model_comparison.csv",
            mime="text/csv",
            use_container_width=True
        )
        st.download_button(
            "⬇️  RF NDVI Features (CSV)",
            data=rf_n["feature_importance_df"].to_csv(index=False).encode(),
            file_name="rf_ndvi_feature_importance.csv",
            mime="text/csv",
            use_container_width=True
        )

    # ── Full Bundle ────────────────────────────────────────────────────────────
    with col3:
        st.markdown(
            section_header("📦", "Full AI Bundle", "", "#8b5cf6"),
            unsafe_allow_html=True
        )
        st.markdown(
            info_card("PHASE 3 FULL BUNDLE",
                      "Downloads all AI reports, model comparisons, "
                      "feature importance tables, confusion matrix data, "
                      "and the complete analysis report in a single ZIP file.",
                      "#8b5cf6"),
            unsafe_allow_html=True
        )

        # Build ZIP
        @st.cache_data(show_spinner=False)
        def build_ai_zip(_ai: dict, _cache: int) -> bytes:
            buf = io.BytesIO()
            all_report = {
                "project":      "GeoTwin Phase 3 — AI/ML",
                "generated_at": datetime.now().isoformat(),
                "summary":      _ai["summary"],
                "kappa":        _ai["kappa"],
                "best_models": {
                    "flood": _ai["best_flood_model"],
                    "ndvi":  _ai["best_ndvi_model"]
                },
                "agreement": {
                    "flood": _ai["flood_agreement_pct"],
                    "ndvi":  _ai["ndvi_agreement_pct"]
                }
            }
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
                zf.writestr("analysis_report.json",
                            json.dumps(all_report, indent=2, default=str))
                zf.writestr("flood_ai_report.json",
                            json.dumps(flood_report, indent=2, default=str))
                zf.writestr("ndvi_ai_report.json",
                            json.dumps(ndvi_report,  indent=2, default=str))
                zf.writestr("flood_model_comparison.csv",
                            _ai["flood_comparison_df"].to_csv())
                zf.writestr("ndvi_model_comparison.csv",
                            _ai["ndvi_comparison_df"].to_csv())
                zf.writestr("all_model_comparison.csv",
                            _ai["all_comparison_df"].to_csv())
                zf.writestr("rf_flood_features.csv",
                            rf_f["feature_importance_df"].to_csv(index=False))
                zf.writestr("rf_ndvi_features.csv",
                            rf_n["feature_importance_df"].to_csv(index=False))
                kappa_csv = pd.DataFrame([
                    {"Model": k, "Kappa": v} for k, v in _ai["kappa"].items()
                ]).to_csv(index=False)
                zf.writestr("kappa_coefficients.csv", kappa_csv)
            buf.seek(0)
            return buf.getvalue()

        with st.spinner("📦 Building AI export bundle..."):
            ai_zip = build_ai_zip(AI, st.session_state["cache_bust"])

        ts_str = datetime.now().strftime("%Y%m%d_%H%M")
        st.download_button(
            "📦  Download All AI Outputs (ZIP)",
            data=ai_zip,
            file_name=f"GeoTwin_Phase3_AI_{ts_str}.zip",
            mime="application/zip",
            use_container_width=True
        )

    # ── Contents table ────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("**AI EXPORT CONTENTS**")
    contents = pd.DataFrame({
        "File": [
            "analysis_report.json", "flood_ai_report.json", "ndvi_ai_report.json",
            "flood_model_comparison.csv", "ndvi_model_comparison.csv",
            "all_model_comparison.csv",
            "rf_flood_features.csv", "rf_ndvi_features.csv",
            "kappa_coefficients.csv"
        ],
        "Description": [
            "Full Phase 3 summary with best models and agreement stats",
            "Flood RF + SVM metrics, confusion matrix, config",
            "Vegetation RF + SVM metrics, confusion matrix, config",
            "RF vs SVM comparison table — Flood",
            "RF vs SVM comparison table — NDVI",
            "All 4 models in one comparison table",
            "Random Forest flood feature importance scores",
            "Random Forest NDVI feature importance scores",
            "Cohen's Kappa for all 4 models"
        ]
    })
    st.dataframe(contents, use_container_width=True, hide_index=True)
