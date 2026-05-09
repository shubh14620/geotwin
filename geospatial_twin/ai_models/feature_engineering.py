"""
================================================================================
  ai_models/feature_engineering.py
  Feature Engineering — Phase 1 SAR & NDVI → ML Feature Matrices
================================================================================

  Bridges Phase 1 remote sensing outputs into supervised ML training format.

  FLOOD FEATURES (per pixel):
    1.  vv_db          — SAR VV backscatter (dB) — primary discriminant
    2.  vh_db          — SAR VH backscatter (dB) — cross-polarisation
    3.  vv_vh_ratio    — VV/VH ratio (linear) — surface roughness proxy
    4.  vv_vh_diff     — VV − VH difference (dB)
    5.  local_mean_3   — Local mean in 3×3 window
    6.  local_std_3    — Local std  in 3×3 window (texture)
    7.  local_mean_7   — Local mean in 7×7 window
    8.  local_std_7    — Local std  in 7×7 window
    9.  gradient_mag   — Gradient magnitude (edge strength)
    10. entropy_local  — Local entropy proxy (coefficient of variation)

  NDVI FEATURES (per pixel):
    1.  ndvi           — Raw NDVI [-1, 1]
    2.  nir            — NIR reflectance
    3.  red            — Red reflectance
    4.  green          — Green reflectance (if available)
    5.  nir_red_ratio  — NIR/RED simple ratio (SR index)
    6.  local_mean_3   — NDVI local mean 3×3
    7.  local_std_3    — NDVI local std  3×3
    8.  local_mean_7   — NDVI local mean 7×7
    9.  gradient_mag   — NDVI spatial gradient
    10. ndvi_sq        — NDVI² (nonlinear term)

================================================================================
"""

import numpy as np
from scipy.ndimage import uniform_filter, sobel, generic_filter
import logging

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
def extract_flood_features(sar_data: dict, flood_result: dict) -> tuple:
    """
    Build flood ML feature matrix from Phase 1 SAR outputs.

    Parameters
    ----------
    sar_data    : dict from generate_demo_sar() — keys: 'vv', 'vh'
    flood_result: dict from FloodProcessor.process() — keys: 'flood_mask', 'backscatter_db'

    Returns
    -------
    X : np.ndarray shape (N, 10)  — feature matrix (N = total pixels)
    y : np.ndarray shape (N,)     — labels  0=dry, 1=flooded
    feature_names : list[str]
    """
    vv_db = flood_result["backscatter_db"].astype(np.float32)
    vh_db = sar_data.get("vh", vv_db - 6.5).astype(np.float32)
    flood_mask = flood_result["flood_mask"]

    # ── Feature 1-4: Radiometric ─────────────────────────────────────────────
    vv_lin = 10 ** (vv_db / 10.0)
    vh_lin = 10 ** (vh_db / 10.0)

    # Safe division
    vv_vh_ratio = np.where(vh_lin > 1e-10, vv_lin / vh_lin, 0.0).astype(np.float32)
    vv_vh_diff  = (vv_db - vh_db).astype(np.float32)

    # ── Feature 5-8: Texture / spatial context ───────────────────────────────
    local_mean_3 = uniform_filter(vv_db, size=3).astype(np.float32)
    local_sq_3   = uniform_filter(vv_db ** 2, size=3)
    local_std_3  = np.sqrt(np.maximum(local_sq_3 - local_mean_3 ** 2, 0)).astype(np.float32)

    local_mean_7 = uniform_filter(vv_db, size=7).astype(np.float32)
    local_sq_7   = uniform_filter(vv_db ** 2, size=7)
    local_std_7  = np.sqrt(np.maximum(local_sq_7 - local_mean_7 ** 2, 0)).astype(np.float32)

    # ── Feature 9-10: Edge & entropy ─────────────────────────────────────────
    sx = sobel(vv_db, axis=0)
    sy = sobel(vv_db, axis=1)
    gradient_mag = np.sqrt(sx ** 2 + sy ** 2).astype(np.float32)

    # Coefficient of variation as entropy proxy
    cv_safe = np.abs(local_mean_3) + 1e-8
    entropy_local = (local_std_3 / cv_safe).astype(np.float32)

    # ── Stack into matrix ────────────────────────────────────────────────────
    feature_names = [
        "vv_db", "vh_db", "vv_vh_ratio", "vv_vh_diff",
        "local_mean_3", "local_std_3",
        "local_mean_7", "local_std_7",
        "gradient_mag", "entropy_local"
    ]

    arrays = [
        vv_db, vh_db, vv_vh_ratio, vv_vh_diff,
        local_mean_3, local_std_3,
        local_mean_7, local_std_7,
        gradient_mag, entropy_local
    ]

    # Flatten to (N, n_features)
    X = np.column_stack([a.flatten() for a in arrays])
    y = flood_mask.flatten().astype(np.int32)

    # Replace NaN/Inf with 0
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    logger.info(f"Flood features: X={X.shape}, y={y.shape}, "
                f"flooded={y.sum()}, dry={(y==0).sum()}")

    return X, y, feature_names


# ─────────────────────────────────────────────────────────────────────────────
def extract_ndvi_features(ms_data: dict, ndvi_result: dict) -> tuple:
    """
    Build vegetation ML feature matrix from Phase 1 NDVI outputs.

    Parameters
    ----------
    ms_data     : dict from generate_demo_multispectral() — keys: 'nir','red','green'
    ndvi_result : dict from NDVIProcessor.process() — keys: 'ndvi','classification'

    Returns
    -------
    X : np.ndarray shape (N, 10)
    y : np.ndarray shape (N,)   — 0=low, 1=moderate, 2=healthy
    feature_names : list[str]
    """
    ndvi = np.where(np.isnan(ndvi_result["ndvi"]), 0.0,
                    ndvi_result["ndvi"]).astype(np.float32)
    nir  = ms_data["nir"].astype(np.float32)
    red  = ms_data["red"].astype(np.float32)
    grn  = ms_data.get("green", (nir * 0.4 + red * 0.3)).astype(np.float32)

    # ── Feature 1-5: Spectral indices ────────────────────────────────────────
    sr_index = np.where(red > 1e-6, nir / red, 0.0).astype(np.float32)  # Simple Ratio
    ndvi_sq  = (ndvi ** 2).astype(np.float32)

    # ── Feature 6-9: Spatial texture ─────────────────────────────────────────
    local_mean_3 = uniform_filter(ndvi, size=3).astype(np.float32)
    local_sq_3   = uniform_filter(ndvi ** 2, size=3)
    local_std_3  = np.sqrt(np.maximum(local_sq_3 - local_mean_3 ** 2, 0)).astype(np.float32)

    local_mean_7 = uniform_filter(ndvi, size=7).astype(np.float32)

    sx = sobel(ndvi, axis=0)
    sy = sobel(ndvi, axis=1)
    gradient_mag = np.sqrt(sx ** 2 + sy ** 2).astype(np.float32)

    # ── Feature 10: EVI approximation ────────────────────────────────────────
    # EVI = 2.5 * (NIR - RED) / (NIR + 6*RED - 7.5*BLUE + 1)  — simplified
    denom_evi = np.abs(nir + 6 * red + 1.0) + 1e-8
    evi = (2.5 * (nir - red) / denom_evi).astype(np.float32)
    evi = np.clip(evi, -1.0, 1.0)

    feature_names = [
        "ndvi", "nir", "red", "green", "sr_index",
        "local_mean_3", "local_std_3", "local_mean_7",
        "gradient_mag", "evi"
    ]

    arrays = [ndvi, nir, red, grn, sr_index,
              local_mean_3, local_std_3, local_mean_7,
              gradient_mag, evi]

    X = np.column_stack([a.flatten() for a in arrays])
    y = ndvi_result["classification"].flatten().astype(np.int32)

    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    logger.info(f"NDVI features: X={X.shape}, y={y.shape}, "
                f"low={( y==0).sum()}, mod={(y==1).sum()}, healthy={(y==2).sum()}")

    return X, y, feature_names


# ─────────────────────────────────────────────────────────────────────────────
def stratified_sample(
    X: np.ndarray,
    y: np.ndarray,
    n_per_class: int = 3000,
    seed: int = 42
) -> tuple:
    """
    Draw a stratified subsample for faster model training.
    Each class contributes n_per_class samples (or all if fewer exist).

    Returns
    -------
    X_s, y_s : subsampled arrays
    """
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    idx_list = []
    for c in classes:
        idx_c = np.where(y == c)[0]
        n     = min(n_per_class, len(idx_c))
        chosen = rng.choice(idx_c, size=n, replace=False)
        idx_list.append(chosen)

    idx_all = np.concatenate(idx_list)
    rng.shuffle(idx_all)

    logger.info(f"Stratified sample: {len(idx_all)} pixels "
                f"({n_per_class} per class × {len(classes)} classes)")
    return X[idx_all], y[idx_all]


# ─────────────────────────────────────────────────────────────────────────────
def normalize_features(X_train: np.ndarray, X_test: np.ndarray = None):
    """
    StandardScaler normalization. Fit on train, apply to test.
    Returns scaled arrays + (mean, std) for persistence.
    """
    mean = X_train.mean(axis=0)
    std  = np.where(X_train.std(axis=0) < 1e-8, 1.0, X_train.std(axis=0))

    X_train_s = (X_train - mean) / std
    if X_test is not None:
        X_test_s = (X_test - mean) / std
        return X_train_s, X_test_s, mean, std
    return X_train_s, mean, std
