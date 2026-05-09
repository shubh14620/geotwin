"""
================================================================================
  ai_models/vegetation_ai/svm_ndvi.py
  Support Vector Machine for Vegetation Health Classification
================================================================================

  Multi-class SVM using One-vs-Rest (OvR) strategy for 3 vegetation classes.
  Pipeline: StandardScaler → SVC(kernel=rbf, probability=True)

  One-vs-Rest means:
    Binary SVM 1: Low_Veg     vs [Moderate + Healthy]
    Binary SVM 2: Moderate_Veg vs [Low + Healthy]
    Binary SVM 3: Healthy_Veg  vs [Low + Moderate]
  Final class = highest probability across all binary SVMs.

================================================================================
"""

import numpy as np
import pandas as pd
import os
import sys
import pickle
import logging
from datetime import datetime

from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_score, recall_score, f1_score
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

_DIR  = os.path.dirname(os.path.abspath(__file__))
_TWIN = os.path.abspath(os.path.join(_DIR, "..", ".."))
if _TWIN not in sys.path:
    sys.path.insert(0, _TWIN)

from ai_models.feature_engineering import extract_ndvi_features, stratified_sample
from ai_models.flood_ai.svm_flood import _batch_predict

logger = logging.getLogger(__name__)

MODEL_PATH = os.path.join(_DIR, "..", "trained_models", "svm_ndvi_model.pkl")


# ─────────────────────────────────────────────────────────────────────────────
class SVMNDVIClassifier:
    """
    SVM vegetation health classifier for Sentinel-2 NDVI outputs.

    Parameters
    ----------
    C : float
        Regularisation parameter.
    gamma : str
        RBF kernel bandwidth.
    n_samples_per_class : int
        Keep small (≤ 2000) since SVM training is O(n²).
    """

    CLASS_NAMES = ["Low Vegetation", "Moderate Vegetation", "Healthy Vegetation"]

    def __init__(
        self,
        kernel: str = "rbf",
        C: float = 8.0,
        gamma: str = "scale",
        n_samples_per_class: int = 2000,
        test_size: float = 0.25,
        random_state: int = 42
    ):
        self.kernel               = kernel
        self.C                    = C
        self.gamma                = gamma
        self.n_samples_per_class  = n_samples_per_class
        self.test_size            = test_size
        self.random_state         = random_state

        self.pipeline: Pipeline = None
        self.feature_names: list = []
        self.results: dict = {}
        self._is_trained = False

    def train_and_evaluate(
        self,
        ms_data: dict,
        ndvi_result: dict,
        run_cv: bool = True
    ) -> dict:
        logger.info("=== SVM Vegetation Classifier — Training ===")

        # ── Features ─────────────────────────────────────────────────────────
        X_full, y_full, self.feature_names = extract_ndvi_features(ms_data, ndvi_result)

        X_s, y_s = stratified_sample(
            X_full, y_full,
            n_per_class=self.n_samples_per_class,
            seed=self.random_state
        )

        X_train, X_test, y_train, y_test = train_test_split(
            X_s, y_s,
            test_size=self.test_size,
            stratify=y_s,
            random_state=self.random_state
        )

        # ── Pipeline with One-vs-Rest SVM ─────────────────────────────────────
        base_svc = SVC(
            kernel=self.kernel, C=self.C, gamma=self.gamma,
            probability=True,
            class_weight="balanced",
            random_state=self.random_state,
            cache_size=500
        )
        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("svm",    base_svc)
        ])

        logger.info("Training SVM (multi-class)...")
        self.pipeline.fit(X_train, y_train)

        # ── Predictions ───────────────────────────────────────────────────────
        y_pred      = self.pipeline.predict(X_test)
        y_pred_full = _batch_predict(self.pipeline, X_full, batch_size=10000)
        pred_raster = y_pred_full.reshape(ndvi_result["ndvi"].shape).astype(np.uint8)

        # ── Metrics ───────────────────────────────────────────────────────────
        acc    = accuracy_score(y_test, y_pred)
        prec   = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec    = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1     = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        cm     = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred,
                                        target_names=self.CLASS_NAMES,
                                        output_dict=True)

        # ── Cross-validation ──────────────────────────────────────────────────
        cv_scores = []
        if run_cv:
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
            cv_scores = cross_val_score(
                self.pipeline, X_s, y_s,
                cv=cv, scoring="f1_weighted", n_jobs=-1
            )

        # ── Feature importance proxy ──────────────────────────────────────────
        svm_est   = self.pipeline.named_steps["svm"]
        if hasattr(svm_est, "support_vectors_") and svm_est.support_vectors_ is not None:
            sv_X  = svm_est.support_vectors_
            raw   = np.abs(sv_X).mean(axis=0)
        else:
            raw = np.ones(len(self.feature_names))
        fi_df = pd.DataFrame({
            "feature":    self.feature_names,
            "importance": raw / (raw.sum() + 1e-10)
        }).sort_values("importance", ascending=False).reset_index(drop=True)

        # ── Per-class stats ───────────────────────────────────────────────────
        from ai_models.vegetation_ai.random_forest_ndvi import _per_class_stats
        per_class = _per_class_stats(cm, self.CLASS_NAMES)

        # ── Support vector count ──────────────────────────────────────────────
        n_sv = int(svm_est.n_support_.sum()) if hasattr(svm_est, "n_support_") else 0

        self.results = {
            "model_name":       "SVM",
            "model_type":       "ndvi",
            "trained_at":       datetime.now().isoformat(),

            "accuracy":         round(float(acc),  4),
            "precision":        round(float(prec), 4),
            "recall":           round(float(rec),  4),
            "f1_score":         round(float(f1),   4),
            "oob_score":        None,

            "cv_scores":        [round(float(s), 4) for s in cv_scores],
            "cv_mean":          round(float(np.mean(cv_scores)), 4) if len(cv_scores) else None,
            "cv_std":           round(float(np.std(cv_scores)),  4) if len(cv_scores) else None,

            "confusion_matrix": cm.tolist(),
            "class_names":      self.CLASS_NAMES,
            "classification_report": report,

            "pred_raster":      pred_raster,
            "feature_importance_df": fi_df,
            "per_class_stats":  per_class,

            "kernel":           self.kernel,
            "C":                self.C,
            "gamma":            str(self.gamma),
            "n_support_vectors": n_sv,
            "n_train":          len(X_train),
            "n_test":           len(X_test),
            "n_features":       len(self.feature_names),
        }

        self._is_trained = True
        logger.info(f"SVM NDVI — Accuracy: {acc:.4f}, F1: {f1:.4f}")
        return self.results

    def save_model(self, path: str = None):
        path = path or MODEL_PATH
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"pipeline": self.pipeline, "feature_names": self.feature_names}, f)

    def load_model(self, path: str = None):
        path = path or MODEL_PATH
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.pipeline      = data["pipeline"]
        self.feature_names = data["feature_names"]
        self._is_trained   = True
