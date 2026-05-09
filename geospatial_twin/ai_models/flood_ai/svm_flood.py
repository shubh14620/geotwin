"""
================================================================================
  ai_models/flood_ai/svm_flood.py
  Support Vector Machine Classifier for SAR Flood Detection
================================================================================

  THEORY:
  ───────
  SVM finds the maximum-margin hyperplane separating flooded from dry pixels
  in the 10-dimensional SAR feature space. The RBF kernel maps features into
  a higher-dimensional space where a linear separator exists even for
  non-linearly separable backscatter distributions.

  WHY SVM FOR SAR:
    - Effective in high-dimensional feature spaces
    - Works well with small training sets
    - RBF kernel captures non-linear backscatter relationships
    - C and gamma hyperparameters tunable for noise robustness
    - Probabilistic output via Platt scaling (probability=True)

  NOTE ON SCALING:
    SVM is SENSITIVE to feature scale. We use StandardScaler before fitting.
    This is a key difference vs. Random Forest.

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
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_score, recall_score, f1_score, roc_auc_score
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ── Path setup ────────────────────────────────────────────────────────────────
_DIR  = os.path.dirname(os.path.abspath(__file__))
_TWIN = os.path.abspath(os.path.join(_DIR, "..", ".."))
if _TWIN not in sys.path:
    sys.path.insert(0, _TWIN)

from ai_models.feature_engineering import extract_flood_features, stratified_sample

logger = logging.getLogger(__name__)

MODEL_PATH = os.path.join(_DIR, "..", "trained_models", "svm_flood_model.pkl")


# ─────────────────────────────────────────────────────────────────────────────
class SVMFloodClassifier:
    """
    SVM classifier for Sentinel-1 SAR flood detection.

    Uses sklearn Pipeline(StandardScaler → SVC) so scaling is encapsulated.

    Parameters
    ----------
    kernel : str
        SVM kernel. 'rbf' (default) handles non-linear backscatter distributions.
    C : float
        Regularisation strength. Larger C = tighter fit (less slack).
    gamma : str or float
        RBF kernel bandwidth. 'scale' = 1/(n_features * X.var()).
    n_samples_per_class : int
        Training samples per class. SVM scales O(n²) — keep ≤ 3000.
    """

    CLASS_NAMES = ["Non-Flooded (Dry)", "Flooded (Water)"]

    def __init__(
        self,
        kernel: str = "rbf",
        C: float = 10.0,
        gamma: str = "scale",
        n_samples_per_class: int = 2000,   # SVM is slower — smaller sample
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

    # ── Main API ──────────────────────────────────────────────────────────────
    def train_and_evaluate(
        self,
        sar_data: dict,
        flood_result: dict,
        run_cv: bool = True
    ) -> dict:
        """Full SVM pipeline: features → scale → train → evaluate."""
        logger.info("=== SVM Flood Classifier — Training ===")

        # ── 1. Feature extraction ─────────────────────────────────────────────
        X_full, y_full, self.feature_names = extract_flood_features(
            sar_data, flood_result
        )

        # ── 2. Subsample (SVM is O(n²) — critical to limit n) ─────────────────
        X_s, y_s = stratified_sample(
            X_full, y_full,
            n_per_class=self.n_samples_per_class,
            seed=self.random_state
        )

        # ── 3. Train / test split ─────────────────────────────────────────────
        X_train, X_test, y_train, y_test = train_test_split(
            X_s, y_s,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y_s
        )

        # ── 4. Build Pipeline: scaler + SVM ──────────────────────────────────
        svm_clf = SVC(
            kernel=self.kernel,
            C=self.C,
            gamma=self.gamma,
            class_weight="balanced",
            probability=True,       # enable predict_proba (Platt scaling)
            random_state=self.random_state,
            cache_size=500          # MB of kernel cache for speed
        )
        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("svm",    svm_clf)
        ])

        logger.info(f"Training SVM (kernel={self.kernel}, C={self.C}, gamma={self.gamma})...")
        self.pipeline.fit(X_train, y_train)

        # ── 5. Predict test set ───────────────────────────────────────────────
        y_pred = self.pipeline.predict(X_test)
        y_prob = self.pipeline.predict_proba(X_test)[:, 1]

        # ── 6. Full raster prediction ─────────────────────────────────────────
        # SVM is slow on large arrays — predict in batches of 10k
        y_pred_full = _batch_predict(self.pipeline, X_full, batch_size=10000)
        pred_raster = y_pred_full.reshape(flood_result["flood_mask"].shape).astype(np.uint8)

        # ── 7. Metrics ────────────────────────────────────────────────────────
        acc    = accuracy_score(y_test, y_pred)
        prec   = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec    = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1     = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        auc    = roc_auc_score(y_test, y_prob)
        cm     = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred,
                                        target_names=self.CLASS_NAMES,
                                        output_dict=True)

        # ── 8. Cross-validation ───────────────────────────────────────────────
        cv_scores = []
        if run_cv:
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
            cv_scores = cross_val_score(
                self.pipeline, X_s, y_s,
                cv=cv, scoring="f1_weighted", n_jobs=-1
            )
            logger.info(f"CV F1: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        # ── 9. SVM decision function — feature "importance" proxy ─────────────
        # For RBF SVM there's no direct feature importance.
        # We use the support vector coefficient spread as a proxy.
        svm_estimator = self.pipeline.named_steps["svm"]
        scaler        = self.pipeline.named_steps["scaler"]
        if self.kernel == "linear":
            raw_coef = np.abs(svm_estimator.coef_[0])
        else:
            # Proxy: scaled variance of each feature weighted by margin
            sv_X      = svm_estimator.support_vectors_            # support vecs (scaled)
            raw_coef  = np.abs(sv_X).mean(axis=0)                # avg magnitude per feature

        fi_df = pd.DataFrame({
            "feature":    self.feature_names,
            "importance": raw_coef / (raw_coef.sum() + 1e-10)
        }).sort_values("importance", ascending=False).reset_index(drop=True)

        # ── 10. Compile results ───────────────────────────────────────────────
        self.results = {
            "model_name":       "SVM",
            "model_type":       "flood",
            "trained_at":       datetime.now().isoformat(),

            "accuracy":         round(float(acc),  4),
            "precision":        round(float(prec), 4),
            "recall":           round(float(rec),  4),
            "f1_score":         round(float(f1),   4),
            "roc_auc":          round(float(auc),  4),
            "oob_score":        None,  # SVM has no OOB

            "cv_scores":        [round(float(s), 4) for s in cv_scores],
            "cv_mean":          round(float(np.mean(cv_scores)), 4) if len(cv_scores) else None,
            "cv_std":           round(float(np.std(cv_scores)),  4) if len(cv_scores) else None,

            "confusion_matrix": cm.tolist(),
            "class_names":      self.CLASS_NAMES,
            "classification_report": report,

            "pred_raster":      pred_raster,
            "feature_importance_df": fi_df,

            "kernel":           self.kernel,
            "C":                self.C,
            "gamma":            str(self.gamma),
            "n_support_vectors": int(svm_estimator.n_support_.sum()),
            "n_train":          len(X_train),
            "n_test":           len(X_test),
            "n_features":       len(self.feature_names),

            "flooded_precision": round(report.get("Flooded (Water)", {}).get("precision", 0), 4),
            "flooded_recall":    round(report.get("Flooded (Water)", {}).get("recall",    0), 4),
            "flooded_f1":        round(report.get("Flooded (Water)", {}).get("f1-score",  0), 4),
        }

        self._is_trained = True
        logger.info(f"SVM Flood — Accuracy: {acc:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")
        return self.results

    def save_model(self, path: str = None):
        path = path or MODEL_PATH
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"pipeline": self.pipeline, "feature_names": self.feature_names}, f)
        logger.info(f"SVM model saved: {path}")

    def load_model(self, path: str = None):
        path = path or MODEL_PATH
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.pipeline      = data["pipeline"]
        self.feature_names = data["feature_names"]
        self._is_trained   = True


# ─────────────────────────────────────────────────────────────────────────────
def _batch_predict(pipeline, X: np.ndarray, batch_size: int = 10000) -> np.ndarray:
    """Predict in batches to avoid OOM with SVM on large arrays."""
    n = len(X)
    out = np.zeros(n, dtype=np.int32)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        out[start:end] = pipeline.predict(X[start:end])
    return out
