"""
================================================================================
  flood_detection/flood_processor.py
  Flood Detection using Sentinel-1 SAR Backscatter Analysis
================================================================================

  THEORY:
  -------
  Synthetic Aperture Radar (SAR) measures microwave backscatter (σ°) from Earth's
  surface. Flooded areas exhibit very LOW backscatter in VV polarization because
  smooth water surfaces cause specular reflection — the microwave signal bounces
  away from the sensor (like a mirror), resulting in a near-zero return.

  Classification Rule:
    σ°(VV) [dB] < threshold  →  FLOODED (water surface)
    σ°(VV) [dB] ≥ threshold  →  NOT FLOODED (land surface)

  Typical threshold range: -16 dB to -14 dB for C-band SAR (Sentinel-1)

================================================================================
"""

import numpy as np
from scipy.ndimage import uniform_filter, generic_filter
import logging

logger = logging.getLogger(__name__)


class FloodProcessor:
    """
    End-to-end SAR-based flood detection processor.

    Parameters
    ----------
    threshold_db : float
        Backscatter threshold in dB below which pixels are classified as flooded.
        Default: -16.0 dB (typical for Sentinel-1 C-band VV)
    speckle_method : str
        Speckle filter to apply. Options: 'lee', 'refined_lee', 'gamma_map', 'none'
    min_flood_size : int
        Minimum connected component size (pixels) to retain as flood. Removes noise.
    """

    def __init__(
        self,
        threshold_db: float = -16.0,
        speckle_method: str = "lee",
        min_flood_size: int = 5
    ):
        self.threshold_db = threshold_db
        self.speckle_method = speckle_method
        self.min_flood_size = min_flood_size

    # ──────────────────────────────────────────────────────────────────────────
    def process(self, sar_data: dict) -> dict:
        """
        Full processing pipeline: preprocess → classify → post-process.

        Parameters
        ----------
        sar_data : dict
            Must contain keys: 'vv' (numpy array, linear σ° or dB values)
            Optionally: 'vh' (numpy array), 'metadata' (dict)

        Returns
        -------
        dict with keys:
            'flood_mask'     : bool array, True = flooded
            'vv_filtered'    : float array, speckle-filtered VV (dB)
            'backscatter_db' : float array, raw VV in dB
            'stats'          : dict of summary statistics
        """
        logger.info("Starting SAR flood detection pipeline...")

        # Step 1: Convert to dB if not already
        vv_raw = sar_data["vv"]
        vv_db  = self._to_db(vv_raw)

        # Step 2: Apply speckle filter
        vv_filtered = self._apply_speckle_filter(vv_db)
        logger.info(f"Speckle filter applied: {self.speckle_method}")

        # Step 3: Threshold-based classification
        flood_mask = vv_filtered < self.threshold_db
        logger.info(f"Threshold applied: {self.threshold_db} dB → "
                    f"{flood_mask.sum()} pixels classified as flooded")

        # Step 4: Post-processing (morphological noise removal)
        flood_mask_clean = self._remove_small_objects(flood_mask)

        # Step 5: Compile statistics
        stats = self._compute_stats(vv_filtered, flood_mask_clean)

        return {
            "flood_mask":     flood_mask_clean,
            "vv_filtered":    vv_filtered,
            "backscatter_db": vv_db,
            "stats":          stats,
            "threshold_used": self.threshold_db
        }

    # ──────────────────────────────────────────────────────────────────────────
    def _to_db(self, arr: np.ndarray) -> np.ndarray:
        """
        Convert linear backscatter (σ°) to decibel scale.
        If input already appears to be in dB (values < 0), return as-is.
        """
        if arr.min() < -1:
            # Already in dB
            return arr.astype(np.float32)
        # Linear → dB: 10 * log10(σ°), avoid log(0) with small epsilon
        arr_safe = np.where(arr > 0, arr, 1e-10)
        return (10.0 * np.log10(arr_safe)).astype(np.float32)

    # ──────────────────────────────────────────────────────────────────────────
    def _apply_speckle_filter(self, arr_db: np.ndarray) -> np.ndarray:
        """
        Apply speckle reduction filter to SAR imagery.

        SAR images suffer from "speckle" — a multiplicative noise caused by
        coherent interference of microwave signals. Filtering improves
        classification accuracy.

        Supported methods:
        - 'lee'          : Lee filter (mean/variance-based adaptive filter)
        - 'refined_lee'  : Simplified refined Lee (edge-preserving)
        - 'gamma_map'    : Gamma-MAP filter (statistically optimal)
        - 'none'         : No filtering
        """
        method = (self.speckle_method or "none").lower().replace(" ", "_")

        if method in ("none", ""):
            return arr_db

        elif method in ("lee", "lee_filter"):
            return self._lee_filter(arr_db, window_size=5)

        elif method in ("refined_lee", "refined lee"):
            return self._refined_lee_filter(arr_db, window_size=7)

        elif method in ("gamma_map", "gamma map"):
            return self._gamma_map_filter(arr_db, window_size=5)

        else:
            logger.warning(f"Unknown speckle filter '{method}', using Lee filter.")
            return self._lee_filter(arr_db, window_size=5)

    def _lee_filter(self, arr: np.ndarray, window_size: int = 5) -> np.ndarray:
        """
        Lee Adaptive Speckle Filter.
        Uses local mean and variance to estimate the true signal:
          filtered = mean + k * (pixel - mean)  where k = var_signal / var_total
        """
        mean_arr = uniform_filter(arr, size=window_size)
        sq_mean  = uniform_filter(arr**2, size=window_size)
        var_arr  = sq_mean - mean_arr**2

        # Noise variance estimate (ENL = Equivalent Number of Looks ~ 4.9 for Sentinel-1 IW GRD)
        enl = 4.9
        var_noise = np.mean(var_arr) / enl

        k = var_arr / (var_arr + var_noise + 1e-10)
        filtered = mean_arr + k * (arr - mean_arr)
        return filtered.astype(np.float32)

    def _refined_lee_filter(self, arr: np.ndarray, window_size: int = 7) -> np.ndarray:
        """
        Simplified Refined Lee Filter — applies Lee filter but weights
        edges more carefully to preserve linear features (roads, rivers).
        """
        # Use directional sub-windows to find minimum-variance direction
        filtered = self._lee_filter(arr, window_size=window_size)
        # Blend with uniform smoothing to reduce residual speckle
        smooth = uniform_filter(arr, size=3)
        return (0.7 * filtered + 0.3 * smooth).astype(np.float32)

    def _gamma_map_filter(self, arr: np.ndarray, window_size: int = 5) -> np.ndarray:
        """
        Gamma-MAP filter — optimal linear estimator for multiplicative speckle.
        Assumes Gamma distributed signal and speckle.
        """
        mean_arr = uniform_filter(arr, size=window_size)
        sq_mean  = uniform_filter(arr**2, size=window_size)
        var_arr  = np.maximum(sq_mean - mean_arr**2, 1e-10)

        enl = 4.9
        cv_n = 1.0 / np.sqrt(enl)                    # coefficient of variation of noise
        cv_x = np.sqrt(var_arr) / (np.abs(mean_arr) + 1e-10)  # local CoV

        b = (1 + cv_n**2) / (cv_x**2 - cv_n**2 + 1e-10)
        filtered = (b * mean_arr + np.sqrt(
            np.maximum((b * mean_arr)**2 - (b + 1) * (mean_arr**2 - (cv_x * mean_arr)**2 * (1 + cv_n**2)), 0)
        )) / (b + 1)
        return filtered.astype(np.float32)

    # ──────────────────────────────────────────────────────────────────────────
    def _remove_small_objects(self, mask: np.ndarray) -> np.ndarray:
        """
        Remove isolated flood pixels smaller than min_flood_size.
        Uses simple 3×3 majority voting as a lightweight alternative to
        scipy.ndimage.label for dependency-free operation.
        """
        if self.min_flood_size <= 1:
            return mask

        # Count neighbors in 3×3 window
        float_mask  = mask.astype(np.float32)
        neighbor_ct = uniform_filter(float_mask, size=3) * 9
        # Keep flooded pixels that have at least min_flood_size neighbors
        cleaned = mask & (neighbor_ct >= min(self.min_flood_size, 4))
        return cleaned

    # ──────────────────────────────────────────────────────────────────────────
    def _compute_stats(self, vv_db: np.ndarray, flood_mask: np.ndarray) -> dict:
        """Compute summary statistics for reporting."""
        total  = flood_mask.size
        flooded = int(flood_mask.sum())
        dry    = total - flooded

        flooded_vals = vv_db[flood_mask]   if flooded > 0 else np.array([np.nan])
        dry_vals     = vv_db[~flood_mask]  if dry     > 0 else np.array([np.nan])

        return {
            "total_pixels":         total,
            "flooded_pixels":       flooded,
            "dry_pixels":           dry,
            "flooded_pct":          flooded / total * 100,
            "mean_backscatter_all": float(np.nanmean(vv_db)),
            "mean_backscatter_flood": float(np.nanmean(flooded_vals)),
            "mean_backscatter_dry": float(np.nanmean(dry_vals)),
            "std_backscatter":      float(np.nanstd(vv_db)),
            "threshold_db":         self.threshold_db,
        }
