"""
================================================================================
  ai_models/vegetation_ai/random_forest_ndvi.py
  Random Forest Classifier for Sentinel-2 Vegetation Health Classification
================================================================================

  TASK: Multi-class classification (3 classes):
    0 → Low Vegetation      (NDVI < 0.2)
    1 → Moderate Vegetation (0.2 ≤ NDVI < 0.5)
    2 → Healthy Vegetation  (NDVI ≥ 0.5)

  FEATURES (10): ndvi, nir, red, green, sr_index,
                 local_mean_3, local_std_3, local_mean_7,
                 gradient_mag, evi

  WHY RANDOM FOREST FITS:
    - Multi-class classification is native (no one-vs-rest needed)
    - Handles non-linear spectral boundaries between classes
    - Feature importance reveals which spectral bands drive classification
    - Robust to mixed pixels at class boundaries

================================================================================
"""

import numpy as np
import pandas as pd
import os
import sys
import pickle
import logging
from datetime import datetime

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_score, recall_score, f1_score
)

_DIR  = os.path.dirname(os.path.abspath(__file__))
_TWIN = os.path.abspath(os.path.join(_DIR, "..", ".."))
if _TWIN not in sys.path:
    sys.path.insert(0, _TWIN)

from ai_models.feature_engineering import extract_ndvi_features, stratified_sample

logger = logging.getLogger(__name__)

MODEL_PATH = os.path.join(_DIR, "..", "trained_models", "rf_ndvi_model.pkl")


# ─────────────────────────────────────────────────────────────────────────────
class RandomForestNDVIClassifier:
    """
    Random Forest vegetation health classifier for Sentinel-2 NDVI data.

    Parameters
    ----------
    n_estimators : int
        Number of trees in the forest.
    max_depth : int
        Maximum tree depth to control overfitting.
    n_samples_per_class : int
        Stratified training sample per class.
    """

    CLASS_NAMES = ["Low Vegetation", "Moderate Vegetation", "Healthy Vegetation"]

    def __init__(
        self,
        n_estimators: int = 150,
        max_depth: int = 14,
        n_samples_per_class: int = 3000,
        test_size: float = 0.25,
        random_state: int = 42
    ):
        self.n_estimators         = n_estimators
        self.max_depth            = max_depth
        self.n_samples_per_class  = n_samples_per_class
        self.test_size            = test_size
        self.random_state         = random_state

        self.model: RandomForestClassifier = None
        self.feature_names: list = []
        self.results: dict = {}
        self._is_trained = False

    # ── Main API ──────────────────────────────────────────────────────────────
    def train_and_evaluate(
        self,
        ms_data: dict,
        ndvi_result: dict,
        run_cv: bool = True
    ) -> dict:
        """Full pipeline: feature extraction → split → train → evaluate."""
        logger.info("=== Random Forest Vegetation Classifier — Training ===")

        # ── 1. Extract features from Phase 1 NDVI outputs ────────────────────
        X_full, y_full, self.feature_names = extract_ndvi_features(ms_data, ndvi_result)

        # ── 2. Stratified subsample ───────────────────────────────────────────
        X_s, y_s = stratified_sample(
            X_full, y_full,
            n_per_class=self.n_samples_per_class,
            seed=self.random_state
        )

        # ── 3. Split ──────────────────────────────────────────────────────────
        X_train, X_test, y_train, y_test = train_test_split(
            X_s, y_s,
            test_size=self.test_size,
            stratify=y_s,
            random_state=self.random_state
        )

        # ── 4. Train Random Forest ────────────────────────────────────────────
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            max_features="sqrt",
            min_samples_split=8,
            min_samples_leaf=4,
            class_weight="balanced",
            n_jobs=-1,
            oob_score=True,
            random_state=self.random_state
        )
        self.model.fit(X_train, y_train)
        logger.info(f"OOB Score: {self.model.oob_score_:.4f}")

        # ── 5. Predictions ────────────────────────────────────────────────────
        y_pred      = self.model.predict(X_test)
        y_pred_full = self.model.predict(X_full)
        pred_raster = y_pred_full.reshape(ndvi_result["ndvi"].shape).astype(np.uint8)

        # ── 6. Metrics ────────────────────────────────────────────────────────
        acc    = accuracy_score(y_test, y_pred)
        prec   = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec    = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1     = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        cm     = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred,
                                        target_names=self.CLASS_NAMES,
                                        output_dict=True)

        # ── 7. Cross-validation ───────────────────────────────────────────────
        cv_scores = []
        if run_cv:
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
            cv_scores = cross_val_score(
                self.model, X_s, y_s,
                cv=cv, scoring="f1_weighted", n_jobs=-1
            )

        # ── 8. Feature importance ─────────────────────────────────────────────
        fi_df = pd.DataFrame({
            "feature":    self.feature_names,
            "importance": self.model.feature_importances_
        }).sort_values("importance", ascending=False).reset_index(drop=True)

        # ── 9. Per-class stats from confusion matrix ──────────────────────────
        per_class = _per_class_stats(cm, self.CLASS_NAMES)

        # ── 10. Compile ───────────────────────────────────────────────────────
        self.results = {
            "model_name":       "Random Forest",
            "model_type":       "ndvi",
            "trained_at":       datetime.now().isoformat(),

            "accuracy":         round(float(acc),  4),
            "precision":        round(float(prec), 4),
            "recall":           round(float(rec),  4),
            "f1_score":         round(float(f1),   4),
            "oob_score":        round(float(self.model.oob_score_), 4),

            "cv_scores":        [round(float(s), 4) for s in cv_scores],
            "cv_mean":          round(float(np.mean(cv_scores)), 4) if len(cv_scores) else None,
            "cv_std":           round(float(np.std(cv_scores)),  4) if len(cv_scores) else None,

            "confusion_matrix": cm.tolist(),
            "class_names":      self.CLASS_NAMES,
            "classification_report": report,

            "pred_raster":      pred_raster,
            "feature_importance_df": fi_df,
            "per_class_stats":  per_class,

            "n_estimators":     self.n_estimators,
            "max_depth":        self.max_depth,
            "n_train":          len(X_train),
            "n_test":           len(X_test),
            "n_features":       len(self.feature_names),
        }

        self._is_trained = True
        logger.info(f"RF NDVI — Accuracy: {acc:.4f}, F1: {f1:.4f}")
        return self.results

    def save_model(self, path: str = None):
        path = path or MODEL_PATH
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"model": self.model, "feature_names": self.feature_names}, f)

    def load_model(self, path: str = None):
        path = path or MODEL_PATH
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.model        = data["model"]
        self.feature_names = data["feature_names"]
        self._is_trained  = True


# ─────────────────────────────────────────────────────────────────────────────
def _per_class_stats(cm: np.ndarray, class_names: list) -> list:
    """Derive per-class precision/recall/F1 from confusion matrix."""
    n = len(class_names)
    rows = []
    for i in range(n):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        p  = tp / (tp + fp + 1e-8)
        r  = tp / (tp + fn + 1e-8)
        f  = 2 * p * r / (p + r + 1e-8)
        rows.append({
            "class":     class_names[i],
            "precision": round(float(p), 4),
            "recall":    round(float(r), 4),
            "f1":        round(float(f), 4),
            "support":   int(cm[i, :].sum())
        })
    return rows
