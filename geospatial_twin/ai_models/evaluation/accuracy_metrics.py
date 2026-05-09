"""
================================================================================
  ai_models/evaluation/accuracy_metrics.py
  Model Evaluation Engine — Metrics, Reports, Model Comparison
================================================================================

  Central evaluation hub that:
    1. Accepts results dicts from RF and SVM classifiers
    2. Computes additional derived metrics
    3. Builds model comparison DataFrames
    4. Generates exportable JSON/CSV reports
    5. Provides data structures consumed by Phase 3 dashboard charts

================================================================================
"""

import numpy as np
import pandas as pd
import os
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
class ModelEvaluator:
    """
    Aggregates results from multiple trained classifiers and produces
    comparison tables, ranking summaries, and export-ready reports.

    Usage:
        evaluator = ModelEvaluator()
        evaluator.add_result("RF Flood",  rf_flood_results)
        evaluator.add_result("SVM Flood", svm_flood_results)
        evaluator.add_result("RF NDVI",   rf_ndvi_results)
        evaluator.add_result("SVM NDVI",  svm_ndvi_results)

        df = evaluator.comparison_table()
        best = evaluator.best_model("flood")
    """

    def __init__(self):
        self._results: dict = {}   # key = model label, value = results dict

    def add_result(self, label: str, results: dict):
        """Register a trained model's result dict."""
        self._results[label] = results
        logger.info(f"Registered model: {label} "
                    f"(acc={results.get('accuracy')}, f1={results.get('f1_score')})")

    def get_result(self, label: str) -> dict:
        return self._results.get(label, {})

    def all_labels(self) -> list:
        return list(self._results.keys())

    # ── Comparison tables ─────────────────────────────────────────────────────
    def comparison_table(self, model_type: str = None) -> pd.DataFrame:
        """
        Build a metrics comparison DataFrame across all registered models.

        Parameters
        ----------
        model_type : str or None
            Filter by 'flood' or 'ndvi'. None = all models.
        """
        rows = []
        for label, r in self._results.items():
            if model_type and r.get("model_type") != model_type:
                continue
            row = {
                "Model":      label,
                "Accuracy":   r.get("accuracy",  0),
                "Precision":  r.get("precision", 0),
                "Recall":     r.get("recall",    0),
                "F1-Score":   r.get("f1_score",  0),
                "CV Mean F1": r.get("cv_mean",   None),
                "CV Std":     r.get("cv_std",    None),
                "OOB Score":  r.get("oob_score", None),
                "Train N":    r.get("n_train",   None),
                "Test N":     r.get("n_test",    None),
            }
            # Add ROC-AUC only for binary (flood) models
            if r.get("model_type") == "flood":
                row["ROC-AUC"] = r.get("roc_auc", None)
            rows.append(row)

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows).set_index("Model")
        # Round numeric columns
        for col in df.select_dtypes(include=float).columns:
            df[col] = df[col].round(4)
        return df

    def best_model(self, model_type: str) -> str:
        """Return the label of the best F1 model for a given type."""
        best_label = None
        best_f1    = -1.0
        for label, r in self._results.items():
            if r.get("model_type") == model_type:
                f1 = r.get("f1_score", 0)
                if f1 > best_f1:
                    best_f1    = f1
                    best_label = label
        return best_label

    def metrics_bar_data(self, model_type: str) -> pd.DataFrame:
        """
        Long-format DataFrame for grouped bar charts.
        Columns: Model, Metric, Value
        """
        rows = []
        metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
        for label, r in self._results.items():
            if model_type and r.get("model_type") != model_type:
                continue
            for metric in metrics:
                key_map = {
                    "Accuracy":  "accuracy",
                    "Precision": "precision",
                    "Recall":    "recall",
                    "F1-Score":  "f1_score"
                }
                rows.append({
                    "Model":  label,
                    "Metric": metric,
                    "Value":  r.get(key_map[metric], 0)
                })
        return pd.DataFrame(rows)

    def cv_scores_data(self, model_type: str) -> pd.DataFrame:
        """Long-format CV scores for box/violin plots."""
        rows = []
        for label, r in self._results.items():
            if r.get("model_type") != model_type:
                continue
            for fold_i, score in enumerate(r.get("cv_scores", [])):
                rows.append({"Model": label, "Fold": fold_i + 1, "F1": score})
        return pd.DataFrame(rows)

    # ── Reports ───────────────────────────────────────────────────────────────
    def generate_json_report(self, output_dir: str = None) -> dict:
        """Build a complete JSON-serialisable analysis report."""
        report = {
            "project":      "GeoTwin Phase 3 — AI/ML Classification",
            "generated_at": datetime.now().isoformat(),
            "models": {}
        }
        for label, r in self._results.items():
            report["models"][label] = {
                "type":      r.get("model_type"),
                "accuracy":  r.get("accuracy"),
                "precision": r.get("precision"),
                "recall":    r.get("recall"),
                "f1_score":  r.get("f1_score"),
                "cv_mean":   r.get("cv_mean"),
                "cv_std":    r.get("cv_std"),
                "oob_score": r.get("oob_score"),
                "n_train":   r.get("n_train"),
                "n_test":    r.get("n_test"),
                "trained_at": r.get("trained_at"),
            }

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            path = os.path.join(output_dir, "ai_model_report.json")
            with open(path, "w") as f:
                json.dump(report, f, indent=2, default=str)
            logger.info(f"Report saved: {path}")

        return report

    def generate_csv_report(self, output_dir: str = None) -> pd.DataFrame:
        """Save comparison table as CSV."""
        df = self.comparison_table()
        if output_dir and not df.empty:
            os.makedirs(output_dir, exist_ok=True)
            path = os.path.join(output_dir, "model_comparison.csv")
            df.to_csv(path)
            logger.info(f"CSV report saved: {path}")
        return df


# ─────────────────────────────────────────────────────────────────────────────
def compute_per_class_iou(cm: np.ndarray, class_names: list) -> pd.DataFrame:
    """
    Intersection over Union (IoU / Jaccard Index) per class.
    IoU = TP / (TP + FP + FN)
    Used in remote sensing to assess spatial accuracy.
    """
    rows = []
    n = len(class_names)
    cm = np.array(cm)
    for i in range(n):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        iou = tp / (tp + fp + fn + 1e-8)
        rows.append({"Class": class_names[i], "IoU": round(float(iou), 4)})
    return pd.DataFrame(rows)


def kappa_coefficient(cm: np.ndarray) -> float:
    """
    Cohen's Kappa — agreement beyond chance.
    κ = (p_o − p_e) / (1 − p_e)
    κ > 0.8 = near-perfect, 0.6–0.8 = substantial, 0.4–0.6 = moderate
    """
    cm = np.array(cm, dtype=float)
    n  = cm.sum()
    p_o = np.diag(cm).sum() / n
    p_e = (cm.sum(axis=1) * cm.sum(axis=0)).sum() / (n ** 2)
    kappa = (p_o - p_e) / (1 - p_e + 1e-10)
    return round(float(kappa), 4)


def compute_full_metrics_table(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list
) -> pd.DataFrame:
    """
    Full per-class metrics: Precision, Recall, F1, IoU, Support.
    """
    from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
    cm        = confusion_matrix(y_true, y_pred)
    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, zero_division=0
    )
    iou_df = compute_per_class_iou(cm, class_names)

    rows = []
    for i, name in enumerate(class_names):
        rows.append({
            "Class":     name,
            "Precision": round(float(prec[i]), 4),
            "Recall":    round(float(rec[i]),  4),
            "F1-Score":  round(float(f1[i]),   4),
            "IoU":       iou_df.iloc[i]["IoU"] if i < len(iou_df) else 0,
            "Support":   int(support[i])
        })
    # Add weighted averages
    rows.append({
        "Class":     "Weighted Avg",
        "Precision": round(float(np.average(prec, weights=support)), 4),
        "Recall":    round(float(np.average(rec,  weights=support)), 4),
        "F1-Score":  round(float(np.average(f1,   weights=support)), 4),
        "IoU":       round(float(np.mean(iou_df["IoU"])), 4),
        "Support":   int(np.sum(support))
    })
    return pd.DataFrame(rows)
