"""
================================================================================
  ai_models/ai_pipeline.py
  Phase 3 AI Pipeline Orchestrator
================================================================================

  Single entry-point that:
    1. Receives Phase 1 data from Phase1DataBridge
    2. Trains RF + SVM for both flood and NDVI
    3. Evaluates all models via ModelEvaluator
    4. Returns a unified results bundle consumed by Phase 3 dashboard page

  This keeps gis_dashboard.py clean — all ML logic lives here.

================================================================================
"""

import numpy as np
import logging
import os
import sys

_DIR  = os.path.dirname(os.path.abspath(__file__))
_TWIN = os.path.abspath(os.path.join(_DIR, ".."))
if _TWIN not in sys.path:
    sys.path.insert(0, _TWIN)

from ai_models.flood_ai.random_forest_flood import RandomForestFloodClassifier
from ai_models.flood_ai.svm_flood           import SVMFloodClassifier
from ai_models.vegetation_ai.random_forest_ndvi import RandomForestNDVIClassifier
from ai_models.vegetation_ai.svm_ndvi       import SVMNDVIClassifier
from ai_models.evaluation.accuracy_metrics  import ModelEvaluator, kappa_coefficient

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
class AIPhase3Pipeline:
    """
    Orchestrates training of all 4 models and returns a unified results bundle.

    Parameters
    ----------
    run_cv : bool
        Whether to run 5-fold cross-validation (adds ~10s per model).
    rf_flood_params / svm_flood_params / ... : dict
        Optional hyperparameter overrides for each model.
    """

    def __init__(
        self,
        run_cv: bool = True,
        rf_flood_params:  dict = None,
        svm_flood_params: dict = None,
        rf_ndvi_params:   dict = None,
        svm_ndvi_params:  dict = None,
    ):
        self.run_cv = run_cv
        self.rf_flood_params  = rf_flood_params  or {}
        self.svm_flood_params = svm_flood_params or {}
        self.rf_ndvi_params   = rf_ndvi_params   or {}
        self.svm_ndvi_params  = svm_ndvi_params  or {}

        # Model instances
        self.rf_flood  = RandomForestFloodClassifier(**self.rf_flood_params)
        self.svm_flood = SVMFloodClassifier(**self.svm_flood_params)
        self.rf_ndvi   = RandomForestNDVIClassifier(**self.rf_ndvi_params)
        self.svm_ndvi  = SVMNDVIClassifier(**self.svm_ndvi_params)

        self.evaluator = ModelEvaluator()
        self._results: dict = {}

    # ─────────────────────────────────────────────────────────────────────────
    def run(self, phase1_data: dict) -> dict:
        """
        Execute the full Phase 3 AI pipeline.

        Parameters
        ----------
        phase1_data : dict
            Output of Phase1DataBridge.get_all() — contains:
              - sar_raw, flood_result, ms_raw, ndvi_result

        Returns
        -------
        dict with all model results, comparison tables, rasters, metrics
        """
        sar_raw      = phase1_data["sar_raw"]
        flood_result = phase1_data["flood_result"]
        ms_raw       = phase1_data["ms_raw"]
        ndvi_result  = phase1_data["ndvi_result"]

        # ── 1. Flood models ──────────────────────────────────────────────────
        logger.info("--- Training RF Flood ---")
        rf_flood_res  = self.rf_flood.train_and_evaluate(
            sar_raw, flood_result, run_cv=self.run_cv
        )

        logger.info("--- Training SVM Flood ---")
        svm_flood_res = self.svm_flood.train_and_evaluate(
            sar_raw, flood_result, run_cv=self.run_cv
        )

        # ── 2. NDVI models ───────────────────────────────────────────────────
        logger.info("--- Training RF NDVI ---")
        rf_ndvi_res   = self.rf_ndvi.train_and_evaluate(
            ms_raw, ndvi_result, run_cv=self.run_cv
        )

        logger.info("--- Training SVM NDVI ---")
        svm_ndvi_res  = self.svm_ndvi.train_and_evaluate(
            ms_raw, ndvi_result, run_cv=self.run_cv
        )

        # ── 3. Register with evaluator ───────────────────────────────────────
        self.evaluator.add_result("RF · Flood",  rf_flood_res)
        self.evaluator.add_result("SVM · Flood", svm_flood_res)
        self.evaluator.add_result("RF · NDVI",   rf_ndvi_res)
        self.evaluator.add_result("SVM · NDVI",  svm_ndvi_res)

        # ── 4. Agreement maps (where RF == SVM) ──────────────────────────────
        flood_agreement_pct = float(
            (rf_flood_res["pred_raster"] == svm_flood_res["pred_raster"]).mean()
        ) * 100
        ndvi_agreement_pct  = float(
            (rf_ndvi_res["pred_raster"]  == svm_ndvi_res["pred_raster"]).mean()
        ) * 100

        # ── 5. Cohen's Kappa ─────────────────────────────────────────────────
        rf_flood_kappa  = kappa_coefficient(rf_flood_res["confusion_matrix"])
        svm_flood_kappa = kappa_coefficient(svm_flood_res["confusion_matrix"])
        rf_ndvi_kappa   = kappa_coefficient(rf_ndvi_res["confusion_matrix"])
        svm_ndvi_kappa  = kappa_coefficient(svm_ndvi_res["confusion_matrix"])

        # ── 6. Best model identification ─────────────────────────────────────
        best_flood_label = self.evaluator.best_model("flood")
        best_ndvi_label  = self.evaluator.best_model("ndvi")

        # ── 7. Compile unified bundle ─────────────────────────────────────────
        self._results = {
            # ── Individual model results ─────────────────────────────────────
            "rf_flood":   rf_flood_res,
            "svm_flood":  svm_flood_res,
            "rf_ndvi":    rf_ndvi_res,
            "svm_ndvi":   svm_ndvi_res,

            # ── Comparison tables ─────────────────────────────────────────────
            "flood_comparison_df": self.evaluator.comparison_table("flood"),
            "ndvi_comparison_df":  self.evaluator.comparison_table("ndvi"),
            "all_comparison_df":   self.evaluator.comparison_table(),

            # ── Chart data ────────────────────────────────────────────────────
            "flood_metrics_bar":  self.evaluator.metrics_bar_data("flood"),
            "ndvi_metrics_bar":   self.evaluator.metrics_bar_data("ndvi"),
            "flood_cv_data":      self.evaluator.cv_scores_data("flood"),
            "ndvi_cv_data":       self.evaluator.cv_scores_data("ndvi"),

            # ── Agreement stats ───────────────────────────────────────────────
            "flood_agreement_pct": round(flood_agreement_pct, 1),
            "ndvi_agreement_pct":  round(ndvi_agreement_pct, 1),

            # ── Kappa ─────────────────────────────────────────────────────────
            "kappa": {
                "RF · Flood":  rf_flood_kappa,
                "SVM · Flood": svm_flood_kappa,
                "RF · NDVI":   rf_ndvi_kappa,
                "SVM · NDVI":  svm_ndvi_kappa,
            },

            # ── Best models ───────────────────────────────────────────────────
            "best_flood_model":  best_flood_label,
            "best_ndvi_model":   best_ndvi_label,

            # ── Summary KPIs (for header cards) ──────────────────────────────
            "summary": {
                "rf_flood_acc":   rf_flood_res["accuracy"],
                "svm_flood_acc":  svm_flood_res["accuracy"],
                "rf_ndvi_acc":    rf_ndvi_res["accuracy"],
                "svm_ndvi_acc":   svm_ndvi_res["accuracy"],
                "rf_flood_f1":    rf_flood_res["f1_score"],
                "svm_flood_f1":   svm_flood_res["f1_score"],
                "rf_ndvi_f1":     rf_ndvi_res["f1_score"],
                "svm_ndvi_f1":    svm_ndvi_res["f1_score"],
                "best_flood":     best_flood_label,
                "best_ndvi":      best_ndvi_label,
                "flood_agreement": round(flood_agreement_pct, 1),
                "ndvi_agreement":  round(ndvi_agreement_pct, 1),
            }
        }

        logger.info("=== Phase 3 AI Pipeline Complete ===")
        logger.info(f"Best Flood: {best_flood_label}, Best NDVI: {best_ndvi_label}")
        return self._results
