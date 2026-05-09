"""
================================================================================
  ai_models/flood_ai/random_forest_flood.py
  Random Forest Classifier for SAR-based Flood Detection
================================================================================

  THEORY:
  ───────
  Random Forest is an ensemble of decision trees trained on random feature
  subsets (bagging). For flood detection:
    - Each tree votes: "flooded" or "dry"
    - Final prediction = majority vote
    - Feature importance shows WHICH SAR bands matter most

  WHY RANDOM FOREST FOR SAR:
    - Handles non-linear backscatter relationships
    - Robust to SAR speckle noise (ensemble averaging)
    - Built-in feature importance (interpretability)
    - No need for feature scaling
    - Fast inference on large rasters

  INPUTS  : Phase 1 SAR features (10-dimensional, from feature_engineering.py)
  OUTPUTS : Flood prediction raster + accuracy metrics

================================================================================
"""

import numpy as np
import pandas as pd
import os
import sys
import pickle
import logging
from datetime import datetime

# ── sklearn imports ───────────────────────────────────────────────────────────
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_score, recall_score, f1_score, roc_auc_score
)
from sklearn.preprocessing import label_binarize

# ── Path setup ────────────────────────────────────────────────────────────────
_DIR  = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_DIR, "..", "..", ".."))
_TWIN = os.path.abspath(os.path.join(_DIR, "..", ".."))
for p in [_ROOT, _TWIN]:
    if p not in sys.path:
        sys.path.insert(0, p)

from ai_models.feature_engineering import (
    extract_flood_features, stratified_sample, normalize_features
)

logger = logging.getLogger(__name__)

# ── Default model save path ────────────────────────────────────────────────────
MODEL_PATH = os.path.join(_DIR, "..", "trained_models", "rf_flood_model.pkl")


# ─────────────────────────────────────────────────────────────────────────────
class RandomForestFloodClassifier:
    """
    Random Forest classifier for Sentinel-1 SAR flood detection.

    Wraps sklearn RandomForestClassifier with:
      - Stratified train/test split
      - Cross-validation
      - Feature importance extraction
      - Full accuracy report generation
      - Prediction raster reconstruction
      - Model persistence (pickle)

    Parameters
    ----------
    n_estimators : int
        Number of trees. More trees = better accuracy, slower training.
        Default 150 (good balance for B.Tech project).
    max_depth : int or None
        Max tree depth. None = grow fully. Shallow trees → faster + less overfit.
    n_samples_per_class : int
        Training samples per class (stratified). Keeps training fast.
    test_size : float
        Fraction of samples held out for testing.
    random_state : int
        Seed for reproducibility.
    """

    CLASS_NAMES = ["Non-Flooded (Dry)", "Flooded (Water)"]

    def __init__(
        self,
        n_estimators: int = 150,
        max_depth: int = 12,
        n_samples_per_class: int = 3000,
        test_size: float = 0.25,
        random_state: int = 42
    ):
        self.n_estimators          = n_estimators
        self.max_depth             = max_depth
        self.n_samples_per_class   = n_samples_per_class
        self.test_size             = test_size
        self.random_state          = random_state

        self.model: RandomForestClassifier = None
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
        """
        Full pipeline: feature extraction → split → train → evaluate.

        Returns
        -------
        results dict with all metrics, predictions, confusion matrix,
        feature importances — ready for dashboard rendering.
        """
        logger.info("=== Random Forest Flood Classifier — Training ===")

        # ── 1. Feature extraction (Phase 1 bridge) ───────────────────────────
        X_full, y_full, self.feature_names = extract_flood_features(
            sar_data, flood_result
        )

        # ── 2. Stratified subsample for speed ────────────────────────────────
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
        logger.info(f"Train: {len(X_train)}, Test: {len(X_test)}")

        # ── 4. Build & train Random Forest ───────────────────────────────────
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            max_features="sqrt",          # sqrt(n_features) per split — standard
            min_samples_split=10,
            min_samples_leaf=5,
            class_weight="balanced",      # handles flood/dry class imbalance
            n_jobs=-1,                    # use all CPU cores
            random_state=self.random_state,
            oob_score=True                # out-of-bag accuracy (free cross-validation)
        )
        self.model.fit(X_train, y_train)
        logger.info(f"OOB Score: {self.model.oob_score_:.4f}")

        # ── 5. Predict on test set ────────────────────────────────────────────
        y_pred       = self.model.predict(X_test)
        y_prob       = self.model.predict_proba(X_test)[:, 1]  # P(flooded)

        # ── 6. Full prediction on entire raster ──────────────────────────────
        y_pred_full  = self.model.predict(X_full)
        pred_raster  = y_pred_full.reshape(flood_result["flood_mask"].shape).astype(np.uint8)

        # ── 7. Metrics ────────────────────────────────────────────────────────
        acc   = accuracy_score(y_test, y_pred)
        prec  = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec   = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1    = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        auc   = roc_auc_score(y_test, y_prob)
        cm    = confusion_matrix(y_test, y_pred)
        report= classification_report(y_test, y_pred,
                                       target_names=self.CLASS_NAMES, output_dict=True)

        # ── 8. Cross-validation (optional) ───────────────────────────────────
        cv_scores = []
        if run_cv:
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
            cv_scores = cross_val_score(self.model, X_s, y_s, cv=cv, scoring="f1_weighted", n_jobs=-1)
            logger.info(f"CV F1: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        # ── 9. Feature importance ─────────────────────────────────────────────
        importances = self.model.feature_importances_
        fi_df = pd.DataFrame({
            "feature":    self.feature_names,
            "importance": importances
        }).sort_values("importance", ascending=False).reset_index(drop=True)

        # ── 10. Compile results ───────────────────────────────────────────────
        self.results = {
            "model_name":       "Random Forest",
            "model_type":       "flood",
            "trained_at":       datetime.now().isoformat(),

            # Core metrics
            "accuracy":         round(float(acc),  4),
            "precision":        round(float(prec), 4),
            "recall":           round(float(rec),  4),
            "f1_score":         round(float(f1),   4),
            "roc_auc":          round(float(auc),  4),
            "oob_score":        round(float(self.model.oob_score_), 4),

            # Cross-validation
            "cv_scores":        [round(float(s), 4) for s in cv_scores],
            "cv_mean":          round(float(np.mean(cv_scores)), 4) if len(cv_scores) else None,
            "cv_std":           round(float(np.std(cv_scores)),  4) if len(cv_scores) else None,

            # Confusion matrix
            "confusion_matrix": cm.tolist(),
            "class_names":      self.CLASS_NAMES,

            # Classification report
            "classification_report": report,

            # Prediction raster (for map overlay)
            "pred_raster":      pred_raster,

            # Feature importance
            "feature_importance_df": fi_df,

            # Model config
            "n_estimators":     self.n_estimators,
            "max_depth":        self.max_depth,
            "n_train":          len(X_train),
            "n_test":           len(X_test),
            "n_features":       len(self.feature_names),

            # Per-class stats from report
            "flooded_precision":    round(report.get("Flooded (Water)", {}).get("precision", 0), 4),
            "flooded_recall":       round(report.get("Flooded (Water)", {}).get("recall",    0), 4),
            "flooded_f1":           round(report.get("Flooded (Water)", {}).get("f1-score",  0), 4),
        }

        self._is_trained = True
        logger.info(f"RF Flood — Accuracy: {acc:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")
        return self.results

    # ── Prediction only (no training) ────────────────────────────────────────
    def predict_raster(self, sar_data: dict, flood_result: dict) -> np.ndarray:
        """
        Apply trained model to produce flood prediction raster.
        Must call train_and_evaluate() first (or load_model()).
        """
        if not self._is_trained:
            raise RuntimeError("Model not trained. Call train_and_evaluate() first.")
        X_full, _, _ = extract_flood_features(sar_data, flood_result)
        y_pred = self.model.predict(X_full)
        return y_pred.reshape(flood_result["flood_mask"].shape).astype(np.uint8)

    # ── Model persistence ─────────────────────────────────────────────────────
    def save_model(self, path: str = None):
        """Pickle the trained sklearn model to disk."""
        if not self._is_trained:
            raise RuntimeError("Train the model before saving.")
        path = path or MODEL_PATH
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"model": self.model, "feature_names": self.feature_names}, f)
        logger.info(f"Model saved: {path}")

    def load_model(self, path: str = None):
        """Load a previously saved model from disk."""
        path = path or MODEL_PATH
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.model        = data["model"]
        self.feature_names = data["feature_names"]
        self._is_trained  = True
        logger.info(f"Model loaded: {path}")
