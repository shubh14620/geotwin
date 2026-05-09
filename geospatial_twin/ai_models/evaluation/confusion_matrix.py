"""
================================================================================
  ai_models/evaluation/confusion_matrix.py
  Confusion Matrix Visualizer + All Phase 3 AI/ML Plotly Charts
================================================================================

  Provides every Plotly figure consumed by the Phase 3 Streamlit dashboard:
    - Confusion matrix heatmaps
    - Metrics bar comparisons
    - Feature importance charts
    - CV score box plots
    - Prediction overlay heatmaps
    - ROC-style accuracy radar
    - Model performance gauge
================================================================================
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ── Shared theme (matches Phase 2 palette) ───────────────────────────────────
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
ORANGE   = "#f97316"

_BASE = dict(
    template="plotly_dark",
    paper_bgcolor=BG,
    plot_bgcolor=BG_PLOT,
    font=dict(family=MONO, color=TEXT, size=10),
    margin=dict(l=12, r=12, t=40, b=12),
    legend=dict(bgcolor=BG, bordercolor=GRID, borderwidth=1,
                font=dict(color=TEXT, size=9)),
    xaxis=dict(gridcolor=GRID, zeroline=False, tickfont=dict(color=TEXT, size=8)),
    yaxis=dict(gridcolor=GRID, zeroline=False, tickfont=dict(color=TEXT, size=8)),
)

def _L(**kw):
    d = dict(_BASE); d.update(kw); return d


# ─────────────────────────────────────────────────────────────────────────────
#   CONFUSION MATRIX
# ─────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(
    cm: list,
    class_names: list,
    model_name: str = "Model",
    height: int = 380
) -> go.Figure:
    """
    Annotated confusion matrix heatmap.
    Diagonal = correct predictions (green tones).
    Off-diagonal = errors (red tones).
    Normalized percentages shown inside each cell.
    """
    cm_arr = np.array(cm, dtype=float)
    # Row-normalize (Recall perspective)
    row_sums = cm_arr.sum(axis=1, keepdims=True)
    cm_norm  = np.where(row_sums > 0, cm_arr / row_sums, 0.0)

    n = len(class_names)
    # Shortened labels for display
    short = [c.replace(" Vegetation", " Veg.").replace("Non-Flooded (Dry)", "Dry")
               .replace("Flooded (Water)", "Flooded") for c in class_names]

    # Build annotation text: count + %
    ann_text = []
    for i in range(n):
        row = []
        for j in range(n):
            row.append(f"<b>{int(cm_arr[i,j])}</b><br>{cm_norm[i,j]*100:.1f}%")
        ann_text.append(row)

    fig = go.Figure(go.Heatmap(
        z=cm_norm,
        x=short,
        y=short,
        colorscale=[
            [0.0, "#0d1117"], [0.3, "#1e3a5f"],
            [0.6, "#1d4ed8"], [1.0, "#10b981"]
        ],
        zmin=0, zmax=1,
        showscale=True,
        colorbar=dict(
            title="Recall",
            tickformat=".0%",
            tickfont=dict(color=TEXT, size=8),
            titlefont=dict(color=TEXT, size=9),
            len=0.8
        ),
        text=ann_text,
        texttemplate="%{text}",
        textfont=dict(size=11, color=TEXT_LT),
        hovertemplate=(
            "Actual: %{y}<br>Predicted: %{x}<br>"
            "Count: %{z:.0%}<extra></extra>"
        )
    ))

    fig.update_layout(**_L(
        title=dict(
            text=f"Confusion Matrix — {model_name}",
            font=dict(color=TEXT, size=11, family=MONO)
        ),
        xaxis=dict(
            title="Predicted Label",
            tickfont=dict(color=TEXT, size=9),
            gridcolor="transparent"
        ),
        yaxis=dict(
            title="True Label",
            tickfont=dict(color=TEXT, size=9),
            gridcolor="transparent",
            autorange="reversed"
        ),
        height=height
    ))
    return fig


# ─────────────────────────────────────────────────────────────────────────────
#   METRICS BAR COMPARISON
# ─────────────────────────────────────────────────────────────────────────────

def plot_metrics_comparison(
    metrics_data: pd.DataFrame,
    title: str = "Model Metrics Comparison",
    height: int = 340
) -> go.Figure:
    """
    Grouped bar chart: Accuracy / Precision / Recall / F1 for each model.

    Parameters
    ----------
    metrics_data : pd.DataFrame
        Columns: Model, Metric, Value  (long format)
    """
    models  = metrics_data["Model"].unique().tolist()
    metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
    colors  = [CYAN, BLUE, GREEN, AMBER]

    fig = go.Figure()
    for metric, color in zip(metrics, colors):
        sub = metrics_data[metrics_data["Metric"] == metric]
        fig.add_trace(go.Bar(
            name=metric,
            x=sub["Model"],
            y=sub["Value"],
            marker_color=color,
            opacity=0.88,
            text=sub["Value"].round(4).astype(str),
            textposition="outside",
            textfont=dict(color=TEXT, size=8),
            hovertemplate=f"<b>{metric}</b><br>%{{x}}: %{{y:.4f}}<extra></extra>"
        ))

    fig.update_layout(**_L(
        title=dict(text=title, font=dict(color=TEXT, size=11, family=MONO)),
        barmode="group",
        yaxis=dict(range=[0, 1.12], title="Score", gridcolor=GRID,
                   tickfont=dict(color=TEXT, size=8)),
        height=height
    ))
    return fig


# ─────────────────────────────────────────────────────────────────────────────
#   FEATURE IMPORTANCE
# ─────────────────────────────────────────────────────────────────────────────

def plot_feature_importance(
    fi_df: pd.DataFrame,
    model_name: str = "Model",
    top_n: int = 10,
    height: int = 340
) -> go.Figure:
    """
    Horizontal bar chart of feature importances (top N features).
    """
    df = fi_df.head(top_n).copy()
    # Normalize to [0,1] for display
    max_val = df["importance"].max()
    if max_val > 0:
        df["importance_norm"] = df["importance"] / max_val
    else:
        df["importance_norm"] = df["importance"]

    # Color gradient: low = dim blue, high = cyan
    colors = [
        f"rgba(6,182,212,{0.3 + 0.7 * v:.2f})"
        for v in df["importance_norm"].tolist()
    ]

    fig = go.Figure(go.Bar(
        x=df["importance"],
        y=df["feature"],
        orientation="h",
        marker=dict(color=colors, line=dict(color=GRID, width=0.5)),
        text=df["importance"].round(4).astype(str),
        textposition="outside",
        textfont=dict(color=TEXT, size=8),
        hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>"
    ))

    fig.update_layout(**_L(
        title=dict(
            text=f"Feature Importance — {model_name} (Top {top_n})",
            font=dict(color=TEXT, size=11, family=MONO)
        ),
        xaxis=dict(title="Importance Score", gridcolor=GRID,
                   tickfont=dict(color=TEXT, size=8)),
        yaxis=dict(autorange="reversed", tickfont=dict(color=TEXT, size=8)),
        height=height
    ))
    return fig


# ─────────────────────────────────────────────────────────────────────────────
#   CROSS-VALIDATION SCORES
# ─────────────────────────────────────────────────────────────────────────────

def plot_cv_scores(
    rf_results: dict,
    svm_results: dict,
    model_type: str = "flood",
    height: int = 300
) -> go.Figure:
    """
    Bar chart with error bars showing CV mean ± std for RF and SVM.
    """
    models   = []
    means    = []
    stds     = []
    cv_lists = []

    for label, r in [("Random Forest", rf_results), ("SVM", svm_results)]:
        if r and r.get("cv_scores"):
            models.append(label)
            means.append(r.get("cv_mean", 0))
            stds.append(r.get("cv_std", 0))
            cv_lists.append(r.get("cv_scores", []))

    if not models:
        fig = go.Figure()
        fig.update_layout(**_L(title=dict(text="CV Scores (no data)", font=dict(color=TEXT, size=11))))
        return fig

    colors = [CYAN, VIOLET]
    fig = go.Figure()

    for i, (model, mean, std, cv) in enumerate(zip(models, means, stds, cv_lists)):
        # Individual fold dots
        fig.add_trace(go.Scatter(
            x=[model] * len(cv),
            y=cv,
            mode="markers",
            name=f"{model} folds",
            marker=dict(color=colors[i], size=8, opacity=0.6,
                        symbol="circle"),
            showlegend=False,
            hovertemplate=f"<b>{model}</b><br>Fold F1: %{{y:.4f}}<extra></extra>"
        ))
        # Mean bar
        fig.add_trace(go.Bar(
            x=[model], y=[mean],
            name=model,
            marker_color=colors[i],
            opacity=0.5,
            error_y=dict(type="data", array=[std], color=TEXT_LT,
                         thickness=2, width=8),
            text=f"{mean:.4f} ± {std:.4f}",
            textposition="outside",
            textfont=dict(color=TEXT, size=9),
            hovertemplate=f"<b>{model}</b><br>Mean: {mean:.4f}<br>Std: {std:.4f}<extra></extra>"
        ))

    fig.update_layout(**_L(
        title=dict(
            text=f"5-Fold Cross-Validation F1 — {model_type.upper()}",
            font=dict(color=TEXT, size=11, family=MONO)
        ),
        yaxis=dict(range=[0, 1.08], title="F1-Score (Weighted)",
                   gridcolor=GRID, tickfont=dict(color=TEXT, size=8)),
        barmode="group",
        height=height
    ))
    return fig


# ─────────────────────────────────────────────────────────────────────────────
#   PREDICTION MAP HEATMAP (AI overlay)
# ─────────────────────────────────────────────────────────────────────────────

def plot_prediction_heatmap(
    pred_raster: np.ndarray,
    title: str = "AI Prediction Map",
    mode: str = "flood",
    height: int = 380
) -> go.Figure:
    """
    Imshow-style Plotly heatmap of the ML prediction raster.

    Parameters
    ----------
    pred_raster : 2D uint8 array
        0/1 for flood; 0/1/2 for vegetation classes.
    mode : str
        'flood' or 'ndvi'
    """
    if mode == "flood":
        colorscale = [
            [0.0, "#1e2d40"],  # dry
            [1.0, "#3b82f6"],  # flooded
        ]
        color_label = "0=Dry · 1=Flooded"
    else:
        colorscale = [
            [0.0,  "#ef4444"],  # low
            [0.5,  "#f59e0b"],  # moderate
            [1.0,  "#10b981"],  # healthy
        ]
        color_label = "0=Low · 1=Moderate · 2=Healthy"

    fig = go.Figure(go.Heatmap(
        z=pred_raster,
        colorscale=colorscale,
        showscale=True,
        colorbar=dict(
            title=color_label,
            tickfont=dict(color=TEXT, size=8),
            titlefont=dict(color=TEXT, size=9),
            len=0.8
        ),
        hovertemplate="Row: %{y}<br>Col: %{x}<br>Class: %{z}<extra></extra>",
    ))

    fig.update_layout(**_L(
        title=dict(text=title, font=dict(color=TEXT, size=11, family=MONO)),
        xaxis=dict(showticklabels=False, showgrid=False),
        yaxis=dict(showticklabels=False, showgrid=False, autorange="reversed"),
        height=height
    ))
    return fig


# ─────────────────────────────────────────────────────────────────────────────
#   RF vs SVM SIDE-BY-SIDE PREDICTION COMPARISON
# ─────────────────────────────────────────────────────────────────────────────

def plot_prediction_comparison(
    rf_raster: np.ndarray,
    svm_raster: np.ndarray,
    mode: str = "flood",
    height: int = 380
) -> go.Figure:
    """
    Side-by-side 1×2 subplot: RF prediction | SVM prediction.
    Highlights agreement and disagreement pixels.
    """
    colorscale = (
        [[0.0, "#1e2d40"], [1.0, "#3b82f6"]] if mode == "flood"
        else [[0.0, "#ef4444"], [0.5, "#f59e0b"], [1.0, "#10b981"]]
    )

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Random Forest Prediction", "SVM Prediction"],
        horizontal_spacing=0.04
    )
    for col_idx, (raster, name) in enumerate(
            [(rf_raster, "RF"), (svm_raster, "SVM")], start=1):
        fig.add_trace(go.Heatmap(
            z=raster,
            colorscale=colorscale,
            showscale=(col_idx == 2),
            colorbar=dict(
                tickfont=dict(color=TEXT, size=8),
                len=0.8, x=1.02
            ),
            hovertemplate=f"<b>{name}</b><br>Class: %{{z}}<extra></extra>"
        ), row=1, col=col_idx)

    fig.update_layout(
        paper_bgcolor=BG, plot_bgcolor=BG_PLOT,
        font=dict(family=MONO, color=TEXT, size=10),
        margin=dict(l=10, r=30, t=50, b=10),
        height=height
    )
    for ann in fig.layout.annotations:
        ann.font.color = TEXT
        ann.font.size  = 10

    for axis in [fig.layout.xaxis, fig.layout.xaxis2,
                 fig.layout.yaxis, fig.layout.yaxis2]:
        axis.showticklabels = False
        axis.showgrid = False

    return fig


# ─────────────────────────────────────────────────────────────────────────────
#   AGREEMENT MAP (where RF == SVM)
# ─────────────────────────────────────────────────────────────────────────────

def plot_agreement_map(
    rf_raster: np.ndarray,
    svm_raster: np.ndarray,
    title: str = "RF vs SVM Agreement Map",
    height: int = 360
) -> go.Figure:
    """
    Pixel-wise agreement: 1 = both models agree, 0 = disagree.
    High agreement = reliable prediction zones.
    """
    agreement = (rf_raster == svm_raster).astype(np.uint8)
    agree_pct  = float(agreement.mean()) * 100

    fig = go.Figure(go.Heatmap(
        z=agreement,
        colorscale=[
            [0.0, "#ef4444"],  # disagree
            [1.0, "#10b981"],  # agree
        ],
        zmin=0, zmax=1,
        showscale=True,
        colorbar=dict(
            tickvals=[0, 1], ticktext=["Disagree", "Agree"],
            tickfont=dict(color=TEXT, size=9),
            len=0.6
        ),
        hovertemplate="Agreement: %{z}<extra></extra>"
    ))

    fig.add_annotation(
        text=f"Agreement: {agree_pct:.1f}%",
        x=0.02, y=0.97,
        xref="paper", yref="paper",
        showarrow=False,
        font=dict(color=GREEN, family=MONO, size=11),
        bgcolor=BG, bordercolor=GREEN, borderpad=5
    )

    fig.update_layout(**_L(
        title=dict(text=title, font=dict(color=TEXT, size=11, family=MONO)),
        xaxis=dict(showticklabels=False, showgrid=False),
        yaxis=dict(showticklabels=False, showgrid=False, autorange="reversed"),
        height=height
    ))
    return fig


# ─────────────────────────────────────────────────────────────────────────────
#   ACCURACY RADAR CHART
# ─────────────────────────────────────────────────────────────────────────────

def plot_accuracy_radar(
    rf_results: dict,
    svm_results: dict,
    model_type: str = "flood",
    height: int = 380
) -> go.Figure:
    """
    Polar radar chart comparing RF vs SVM across 5 metrics.
    """
    categories = ["Accuracy", "Precision", "Recall", "F1-Score", "CV Mean F1"]

    def _get_vals(r):
        return [
            r.get("accuracy",  0),
            r.get("precision", 0),
            r.get("recall",    0),
            r.get("f1_score",  0),
            r.get("cv_mean",   0) or 0
        ]

    fig = go.Figure()
    for label, r, color in [
        ("Random Forest", rf_results,  CYAN),
        ("SVM",           svm_results, VIOLET)
    ]:
        if not r:
            continue
        vals = _get_vals(r)
        fig.add_trace(go.Scatterpolar(
            r=vals + [vals[0]],
            theta=categories + [categories[0]],
            name=label,
            fill="toself",
            fillcolor=color.replace("#", "rgba(") + ",0.12)" if color.startswith("#") else color,
            line=dict(color=color, width=2),
            marker=dict(size=6, color=color),
            hovertemplate="%{theta}: %{r:.4f}<extra></extra>"
        ))

    fig.update_layout(
        paper_bgcolor=BG,
        polar=dict(
            bgcolor=BG_PLOT,
            radialaxis=dict(
                visible=True, range=[0, 1],
                tickfont=dict(color=TEXT, size=8),
                gridcolor=GRID, linecolor=GRID
            ),
            angularaxis=dict(
                tickfont=dict(color=TEXT_LT, size=9),
                gridcolor=GRID, linecolor=GRID
            )
        ),
        legend=dict(bgcolor=BG, bordercolor=GRID, font=dict(color=TEXT, size=9)),
        font=dict(family=MONO, color=TEXT),
        margin=dict(l=20, r=20, t=50, b=20),
        title=dict(
            text=f"RF vs SVM — {model_type.upper()} Performance Radar",
            font=dict(color=TEXT, size=11, family=MONO)
        ),
        height=height
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
#   PER-CLASS METRICS BAR (for NDVI 3-class)
# ─────────────────────────────────────────────────────────────────────────────

def plot_per_class_metrics(
    per_class_stats: list,
    model_name: str = "Model",
    height: int = 320
) -> go.Figure:
    """
    Grouped bar chart of per-class Precision / Recall / F1.
    """
    if not per_class_stats:
        return go.Figure()

    df = pd.DataFrame(per_class_stats)
    short_cls = [c.replace(" Vegetation", " Veg.") for c in df["class"].tolist()]

    fig = go.Figure()
    for metric, color in [("precision", CYAN), ("recall", GREEN), ("f1", AMBER)]:
        if metric not in df.columns:
            continue
        fig.add_trace(go.Bar(
            name=metric.capitalize(),
            x=short_cls,
            y=df[metric],
            marker_color=color,
            opacity=0.85,
            text=df[metric].round(3).astype(str),
            textposition="outside",
            textfont=dict(color=TEXT, size=8),
            hovertemplate=f"<b>{metric.capitalize()}</b><br>%{{x}}: %{{y:.4f}}<extra></extra>"
        ))

    fig.update_layout(**_L(
        title=dict(
            text=f"Per-Class Metrics — {model_name}",
            font=dict(color=TEXT, size=11, family=MONO)
        ),
        barmode="group",
        yaxis=dict(range=[0, 1.15], title="Score",
                   gridcolor=GRID, tickfont=dict(color=TEXT, size=8)),
        height=height
    ))
    return fig


# ─────────────────────────────────────────────────────────────────────────────
#   PREDICTION STATS DONUT
# ─────────────────────────────────────────────────────────────────────────────

def plot_prediction_distribution(
    pred_raster: np.ndarray,
    class_names: list,
    model_name: str = "Model",
    height: int = 300
) -> go.Figure:
    """
    Donut chart of predicted class pixel distribution.
    """
    unique, counts = np.unique(pred_raster, return_counts=True)
    labels  = [class_names[int(u)] if int(u) < len(class_names) else str(u)
               for u in unique]
    colors_map = {
        "Flooded (Water)":    BLUE,
        "Non-Flooded (Dry)":  "#374151",
        "Dry":                "#374151",
        "Flooded":            BLUE,
        "Low Vegetation":     RED,
        "Moderate Vegetation":AMBER,
        "Healthy Vegetation": GREEN,
        "Low Veg.":           RED,
        "Moderate Veg.":      AMBER,
        "Healthy Veg.":       GREEN,
    }
    colors = [colors_map.get(l, CYAN) for l in labels]

    fig = go.Figure(go.Pie(
        labels=labels,
        values=counts,
        hole=0.55,
        marker=dict(colors=colors, line=dict(color=BG, width=2)),
        textinfo="label+percent",
        textfont=dict(family=MONO, size=9, color=TEXT_LT),
        hovertemplate="<b>%{label}</b><br>Pixels: %{value:,}<br>%{percent}<extra></extra>"
    ))
    fig.add_annotation(
        text=f"<b>{model_name}</b>",
        x=0.5, y=0.5, showarrow=False,
        font=dict(color=CYAN, family=MONO, size=11)
    )
    fig.update_layout(**_L(
        title=dict(
            text=f"Predicted Class Distribution — {model_name}",
            font=dict(color=TEXT, size=11, family=MONO)
        ),
        height=height
    ))
    return fig
