"""
================================================================================
  ndvi_monitoring/ndvi_processor.py
  NDVI Crop Health Monitoring using Sentinel-2 Multispectral Imagery
================================================================================

  NDVI FORMULA:
  ─────────────
                NIR − RED         B8 − B4
  NDVI = ─────────────────── = ───────────────
                NIR + RED         B8 + B4

  WHERE:
    B4  = Red band         (665 nm, 10m resolution)
    B8  = Near-Infrared    (842 nm, 10m resolution)

  NDVI RANGES (Sentinel-2 Level-2A surface reflectance):
    NDVI < 0.0   →  Water, snow, bare rock
    0.0 – 0.2    →  Bare soil, urban
    0.2 – 0.4    →  Sparse / stressed vegetation
    0.4 – 0.6    →  Moderate cropland
    > 0.6        →  Dense / healthy vegetation

================================================================================
"""

import numpy as np
from scipy.ndimage import uniform_filter, gaussian_filter
import logging

logger = logging.getLogger(__name__)


class NDVIProcessor:
    """
    Sentinel-2 NDVI computation and vegetation health classification processor.

    Parameters
    ----------
    low_thresh : float
        NDVI below this value → "Low Vegetation" (bare soil, stressed crops).
        Default: 0.2
    high_thresh : float
        NDVI above this value → "Healthy Vegetation". Between low_thresh and
        high_thresh = "Moderate Vegetation".
        Default: 0.5
    smooth_sigma : float
        Gaussian smoothing sigma applied before classification (0 = no smoothing).
        Reduces salt-and-pepper classification noise.
    """

    # Sentinel-2 band names and central wavelengths (nm)
    BAND_INFO = {
        "B2":  ("Blue",       490),
        "B3":  ("Green",      560),
        "B4":  ("Red",        665),
        "B8":  ("NIR",        842),
        "B8A": ("Red Edge",   865),
        "B11": ("SWIR-1",    1610),
        "B12": ("SWIR-2",    2190),
    }

    # Classification class IDs
    CLASS_LOW      = 0   # Low / no vegetation
    CLASS_MODERATE = 1   # Moderate vegetation
    CLASS_HEALTHY  = 2   # Healthy / dense vegetation

    def __init__(
        self,
        low_thresh: float  = 0.2,
        high_thresh: float = 0.5,
        smooth_sigma: float = 0.8
    ):
        if low_thresh >= high_thresh:
            raise ValueError(f"low_thresh ({low_thresh}) must be < high_thresh ({high_thresh})")

        self.low_thresh   = low_thresh
        self.high_thresh  = high_thresh
        self.smooth_sigma = smooth_sigma

    # ──────────────────────────────────────────────────────────────────────────
    def process(self, ms_data: dict) -> dict:
        """
        Full NDVI processing pipeline.

        Parameters
        ----------
        ms_data : dict
            Expected keys: 'nir' (Band 8 array), 'red' (Band 4 array).
            Optional keys: 'green' (B3), 'blue' (B2), 'swir1' (B11)
            All values should be float in range [0, 1] (surface reflectance).

        Returns
        -------
        dict with keys:
            'ndvi'           : float array [-1, 1], NDVI values
            'classification' : int array, 0=low, 1=moderate, 2=healthy
            'nir'            : NIR band array (passed through)
            'red'            : Red band array (passed through)
            'stats'          : dict of summary statistics
            'class_labels'   : dict mapping class_id → label string
        """
        logger.info("Starting NDVI crop health monitoring pipeline...")

        # Step 1: Extract bands
        nir = ms_data["nir"].astype(np.float32)
        red = ms_data["red"].astype(np.float32)

        # Step 2: Validate inputs
        self._validate_bands(nir, red)

        # Step 3: Calculate NDVI
        ndvi = self._compute_ndvi(nir, red)
        logger.info(f"NDVI computed. Range: [{ndvi.min():.3f}, {ndvi.max():.3f}], "
                    f"Mean: {np.nanmean(ndvi):.3f}")

        # Step 4: Optional smoothing before classification
        ndvi_smooth = self._smooth(ndvi)

        # Step 5: Classify vegetation health
        classification = self._classify(ndvi_smooth)
        logger.info("Vegetation classification complete.")

        # Step 6: Compute statistics
        stats = self._compute_stats(ndvi, classification)

        return {
            "ndvi":           ndvi,
            "ndvi_smooth":    ndvi_smooth,
            "classification": classification,
            "nir":            nir,
            "red":            red,
            "stats":          stats,
            "class_labels": {
                self.CLASS_LOW:      "Low Vegetation",
                self.CLASS_MODERATE: "Moderate Vegetation",
                self.CLASS_HEALTHY:  "Healthy Vegetation"
            },
            "thresholds": {
                "low":  self.low_thresh,
                "high": self.high_thresh
            }
        }

    # ──────────────────────────────────────────────────────────────────────────
    def _compute_ndvi(self, nir: np.ndarray, red: np.ndarray) -> np.ndarray:
        """
        Compute NDVI = (NIR - RED) / (NIR + RED).

        Uses masked array to safely handle divide-by-zero where NIR + RED ≈ 0
        (e.g., unlit pixels, saturation artifacts).
        """
        denominator = nir + red

        # Mask locations where denominator is essentially zero
        mask = denominator < 1e-6
        safe_denom = np.where(mask, 1.0, denominator)   # avoid division by zero

        ndvi = (nir - red) / safe_denom
        ndvi = np.where(mask, np.nan, ndvi)              # NaN for undefined pixels
        ndvi = np.clip(ndvi, -1.0, 1.0)                 # Enforce physical range

        return ndvi.astype(np.float32)

    # ──────────────────────────────────────────────────────────────────────────
    def _classify(self, ndvi: np.ndarray) -> np.ndarray:
        """
        Classify NDVI into 3 vegetation health categories.

        Returns integer array:
            0 → Low Vegetation      (NDVI < low_thresh)
            1 → Moderate Vegetation (low_thresh ≤ NDVI < high_thresh)
            2 → Healthy Vegetation  (NDVI ≥ high_thresh)
        """
        classification = np.full(ndvi.shape, self.CLASS_LOW, dtype=np.uint8)

        # Moderate vegetation
        moderate_mask = (ndvi >= self.low_thresh) & (ndvi < self.high_thresh)
        classification[moderate_mask] = self.CLASS_MODERATE

        # Healthy vegetation
        healthy_mask = ndvi >= self.high_thresh
        classification[healthy_mask] = self.CLASS_HEALTHY

        # Handle NaN as low vegetation (conservative approach)
        nan_mask = np.isnan(ndvi)
        classification[nan_mask] = self.CLASS_LOW

        return classification

    # ──────────────────────────────────────────────────────────────────────────
    def _smooth(self, ndvi: np.ndarray) -> np.ndarray:
        """Apply Gaussian smoothing to reduce classification speckle."""
        if self.smooth_sigma <= 0:
            return ndvi
        nan_mask  = np.isnan(ndvi)
        ndvi_fill = np.where(nan_mask, 0.0, ndvi)
        smoothed  = gaussian_filter(ndvi_fill, sigma=self.smooth_sigma)
        smoothed  = np.where(nan_mask, np.nan, smoothed)
        return smoothed.astype(np.float32)

    # ──────────────────────────────────────────────────────────────────────────
    def _validate_bands(self, nir: np.ndarray, red: np.ndarray):
        """Basic input validation."""
        if nir.shape != red.shape:
            raise ValueError(f"NIR shape {nir.shape} ≠ RED shape {red.shape}")
        if nir.ndim != 2:
            raise ValueError(f"Expected 2D arrays, got {nir.ndim}D")
        logger.debug(f"Band validation OK: shape={nir.shape}")

    # ──────────────────────────────────────────────────────────────────────────
    def _compute_stats(self, ndvi: np.ndarray, classification: np.ndarray) -> dict:
        """Compute per-class and global statistics."""
        total   = classification.size
        ndvi_v  = ndvi[~np.isnan(ndvi)]

        cls_stats = {}
        for cid, cname in [(0, "low"), (1, "moderate"), (2, "healthy")]:
            mask   = classification == cid
            count  = int(mask.sum())
            vals   = ndvi[mask & ~np.isnan(ndvi)]
            cls_stats[cname] = {
                "pixel_count": count,
                "coverage_pct": count / total * 100,
                "mean_ndvi":   float(np.nanmean(vals)) if len(vals) > 0 else 0.0,
                "std_ndvi":    float(np.nanstd(vals))  if len(vals) > 0 else 0.0,
            }

        return {
            "total_pixels": total,
            "valid_pixels": len(ndvi_v),
            "ndvi_mean":   float(np.nanmean(ndvi)),
            "ndvi_median": float(np.nanmedian(ndvi)),
            "ndvi_std":    float(np.nanstd(ndvi)),
            "ndvi_min":    float(np.nanmin(ndvi)),
            "ndvi_max":    float(np.nanmax(ndvi)),
            "by_class":    cls_stats,
            "thresholds":  {"low": self.low_thresh, "high": self.high_thresh}
        }
