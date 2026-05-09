"""
================================================================================
  utils/demo_data.py
  Synthetic Demo Data Generator for GeoTwin Phase 1
================================================================================

  Generates realistic synthetic SAR (Sentinel-1) and multispectral (Sentinel-2)
  data arrays for dashboard demonstration WITHOUT requiring GEE credentials.

  The synthetic data mimics real-world spatial patterns:
    - SAR: low backscatter in river/flood areas, higher on land
    - NDVI: green agricultural fields, sparse edges, waterbodies

================================================================================
"""

import numpy as np
from scipy.ndimage import gaussian_filter


# ─────────────────────────────────────────────────────────────────────────────
def generate_demo_sar(
    shape: tuple = (256, 256),
    seed: int = 42,
    flood_fraction: float = 0.22
) -> dict:
    """
    Generate a realistic synthetic Sentinel-1 SAR scene (VV polarization, dB).

    Creates a scene with:
      - River/floodplain with low backscatter (-22 to -18 dB)
      - Agricultural land with medium backscatter (-14 to -10 dB)
      - Urban areas with high backscatter (-8 to -2 dB)
      - Gaussian speckle noise

    Parameters
    ----------
    shape : tuple
        (rows, cols) spatial dimensions.
    seed : int
        Random seed for reproducibility.
    flood_fraction : float
        Approximate fraction of pixels that are "flooded" (low backscatter).

    Returns
    -------
    dict with keys 'vv' (float32 array, dB), 'vh' (float32 array, dB),
                   'metadata' (dict)
    """
    rng = np.random.default_rng(seed)
    rows, cols = shape

    # ── Base terrain map ────────────────────────────────────────────────────
    # Create smooth spatial gradient as base
    x = np.linspace(0, 2*np.pi, cols)
    y = np.linspace(0, 2*np.pi, rows)
    xx, yy = np.meshgrid(x, y)

    # Smooth terrain (slow spatial variation)
    terrain = (np.sin(xx * 0.8) * np.cos(yy * 0.6) +
               np.sin(xx * 0.3 + 0.5) * np.sin(yy * 0.4 + 1.2) * 0.5)
    terrain = gaussian_filter(terrain, sigma=15)
    terrain = (terrain - terrain.min()) / (terrain.max() - terrain.min())  # [0,1]

    # ── Create land cover zones ─────────────────────────────────────────────
    # Flooded/water area: lower portion + river meander
    flood_zone = np.zeros(shape)

    # River/channel running diagonally
    for i in range(rows):
        j_center = int(cols * 0.3 + (i / rows) * cols * 0.4 +
                       20 * np.sin(i * 0.1))
        width = int(15 + 10 * np.sin(i * 0.05))
        j_lo = max(0, j_center - width)
        j_hi = min(cols, j_center + width)
        flood_zone[i, j_lo:j_hi] = 1.0

    # Floodplain area (low-lying region)
    floodplain = (terrain < 0.35).astype(float)
    flood_zone = np.clip(flood_zone + floodplain * 0.7, 0, 1)
    flood_zone = gaussian_filter(flood_zone, sigma=5)
    flood_zone = (flood_zone > 0.45).astype(float)

    # Urban patches (high backscatter)
    urban_zone = np.zeros(shape)
    for _ in range(8):
        cr = rng.integers(20, rows - 20)
        cc = rng.integers(20, cols - 20)
        ur, uc = rng.integers(10, 30), rng.integers(10, 30)
        urban_zone[max(0,cr-ur):cr+ur, max(0,cc-uc):cc+uc] = 1.0
    urban_zone = gaussian_filter(urban_zone, sigma=3)
    urban_zone[flood_zone > 0.5] = 0  # Urban can't be flooded

    # ── Assign backscatter values (dB) ──────────────────────────────────────
    # Base: agricultural land, medium backscatter
    vv_db = rng.normal(-12.0, 2.5, shape).astype(np.float32)

    # Flooded pixels: much lower backscatter (specular reflection)
    flood_mask_bool = flood_zone > 0.5
    vv_db[flood_mask_bool] = rng.normal(-20.0, 2.0, flood_mask_bool.sum())

    # Urban pixels: higher backscatter (double-bounce)
    urban_mask_bool = urban_zone > 0.5
    vv_db[urban_mask_bool] = rng.normal(-5.0, 2.0, urban_mask_bool.sum())

    # Add SAR speckle noise (multiplicative → additive in dB)
    speckle = rng.normal(0, 1.5, shape).astype(np.float32)
    vv_db += speckle

    # VH polarization: typically ~5–8 dB lower than VV
    vh_db = vv_db - rng.normal(6.5, 1.0, shape).astype(np.float32)

    return {
        "vv": vv_db,
        "vh": vh_db,
        "metadata": {
            "satellite":    "Sentinel-1",
            "sensor":       "SAR C-band (5.6 GHz)",
            "polarization": "VV/VH",
            "mode":         "IW (Interferometric Wide)",
            "resolution_m": 10,
            "units":        "dB (σ° calibrated)",
            "synthetic":    True,
            "shape":        shape,
            "flood_fraction_actual": float(flood_mask_bool.mean())
        }
    }


# ─────────────────────────────────────────────────────────────────────────────
def generate_demo_multispectral(
    shape: tuple = (256, 256),
    seed: int = 42
) -> dict:
    """
    Generate a realistic synthetic Sentinel-2 multispectral scene.

    Creates spatial patterns for:
      - Dense cropland (high NIR, moderate RED → high NDVI)
      - Sparse vegetation / early growth (moderate NIR, moderate RED)
      - Bare soil / fallow (low NIR, higher RED → low NDVI)
      - Waterbody (low all bands)

    Parameters
    ----------
    shape : tuple
        (rows, cols) spatial dimensions.
    seed : int
        Random seed.

    Returns
    -------
    dict with keys: 'blue', 'green', 'red', 'nir', 'swir1', 'swir2',
                    'metadata' (dict)
    """
    rng = np.random.default_rng(seed)
    rows, cols = shape

    # ── Create smooth land cover base ───────────────────────────────────────
    x = np.linspace(0, 3*np.pi, cols)
    y = np.linspace(0, 3*np.pi, rows)
    xx, yy = np.meshgrid(x, y)

    # Agricultural field pattern (checkerboard-like with smooth edges)
    field_pattern = (np.sin(xx * 1.5) * np.cos(yy * 1.2) +
                     np.cos(xx * 0.7 + 1.0) * np.sin(yy * 0.9))
    field_pattern = gaussian_filter(field_pattern, sigma=8)
    field_pattern = (field_pattern - field_pattern.min()) / (field_pattern.max() - field_pattern.min())

    # ── Assign reflectance zones ─────────────────────────────────────────────
    # Zone 1: Dense healthy crops (high NDVI)
    zone_healthy = field_pattern > 0.65

    # Zone 2: Moderate crops
    zone_moderate = (field_pattern > 0.35) & (~zone_healthy)

    # Zone 3: Bare soil / fallow
    zone_bare = field_pattern <= 0.35

    # Zone 4: Waterbody (river from corner)
    water_zone = np.zeros(shape, dtype=bool)
    for i in range(rows):
        j_c = int(cols * 0.15 + (i / rows) * cols * 0.2 + 10 * np.sin(i * 0.12))
        w   = int(8 + 5 * np.sin(i * 0.07))
        water_zone[i, max(0,j_c-w):min(cols,j_c+w)] = True

    # ── Base reflectance (surface reflectance, [0, 1]) ──────────────────────
    # NIR Band (B8) — high for vegetation
    nir = np.full(shape, 0.12, dtype=np.float32)
    nir[zone_healthy]  = rng.normal(0.60, 0.06, zone_healthy.sum()).clip(0.4, 0.85)
    nir[zone_moderate] = rng.normal(0.38, 0.07, zone_moderate.sum()).clip(0.2, 0.55)
    nir[zone_bare]     = rng.normal(0.22, 0.05, zone_bare.sum()).clip(0.08, 0.35)
    nir[water_zone]    = rng.normal(0.04, 0.01, water_zone.sum()).clip(0.01, 0.08)

    # RED Band (B4) — low for healthy veg (absorbed by chlorophyll)
    red = np.full(shape, 0.15, dtype=np.float32)
    red[zone_healthy]  = rng.normal(0.07, 0.02, zone_healthy.sum()).clip(0.03, 0.15)
    red[zone_moderate] = rng.normal(0.14, 0.03, zone_moderate.sum()).clip(0.06, 0.25)
    red[zone_bare]     = rng.normal(0.22, 0.04, zone_bare.sum()).clip(0.12, 0.38)
    red[water_zone]    = rng.normal(0.05, 0.01, water_zone.sum()).clip(0.01, 0.09)

    # GREEN Band (B3)
    green = (nir * 0.35 + red * 0.4 + rng.normal(0, 0.01, shape)).clip(0.01, 0.5)

    # BLUE Band (B2)
    blue = (red * 0.7 + rng.normal(0.02, 0.01, shape)).clip(0.01, 0.3)

    # SWIR-1 Band (B11) — useful for moisture/soil
    swir1 = (1.0 - nir) * 0.4 + rng.normal(0, 0.02, shape)
    swir1 = swir1.clip(0.01, 0.6).astype(np.float32)

    # SWIR-2 Band (B12)
    swir2 = (swir1 * 0.8 + rng.normal(0, 0.01, shape)).clip(0.01, 0.5).astype(np.float32)

    # Add sensor noise
    noise_level = 0.008
    for band in [nir, red, green, blue]:
        band += rng.normal(0, noise_level, shape).astype(np.float32)
        band[:] = band.clip(0.0, 1.0)

    return {
        "blue":  blue.astype(np.float32),
        "green": green.astype(np.float32),
        "red":   red.astype(np.float32),
        "nir":   nir.astype(np.float32),
        "swir1": swir1,
        "swir2": swir2,
        "metadata": {
            "satellite":    "Sentinel-2",
            "sensor":       "MSI (MultiSpectral Instrument)",
            "level":        "L2A (Surface Reflectance)",
            "bands": {
                "blue": "B2 (490 nm)", "green": "B3 (560 nm)",
                "red":  "B4 (665 nm)", "nir":   "B8 (842 nm)",
                "swir1":"B11 (1610 nm)","swir2": "B12 (2190 nm)"
            },
            "resolution_m": 10,
            "synthetic":    True,
            "shape":        shape
        }
    }
