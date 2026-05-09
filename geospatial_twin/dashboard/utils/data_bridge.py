"""
================================================================================
  dashboard/utils/data_bridge.py
  Phase 2 → Phase 1 Data Bridge
  
  This module is the CORE CONNECTOR between Phase 1 processing engines
  (flood_processor, ndvi_processor, demo_data) and the Phase 2 GIS dashboard.
  
  It loads, caches, and prepares all data structures that the dashboard
  components consume — maps, charts, GeoDataFrames, export buffers.
================================================================================
"""

import numpy as np
import pandas as pd
import sys
import os
import io
import base64
import logging
from datetime import datetime, timedelta
from scipy.ndimage import gaussian_filter

logger = logging.getLogger(__name__)

# ── Make Phase 1 modules importable regardless of working directory ───────────
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

_TWIN = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _TWIN not in sys.path:
    sys.path.insert(0, _TWIN)

from flood_detection.flood_processor import FloodProcessor
from ndvi_monitoring.ndvi_processor import NDVIProcessor
from utils.demo_data import generate_demo_sar, generate_demo_multispectral


# ─────────────────────────────────────────────────────────────────────────────
class Phase1DataBridge:
    """
    Loads and caches Phase 1 processing results for Phase 2 dashboard.

    Usage:
        bridge = Phase1DataBridge(flood_threshold=-16, ndvi_lo=0.2, ndvi_hi=0.5)
        data   = bridge.get_all()
    """

    # Study area: Ganga floodplain bounding box (for map centering)
    DEFAULT_BBOX = {
        "lat_min": 25.0, "lat_max": 27.0,
        "lon_min": 80.0, "lon_max": 83.0,
        "center_lat": 26.0, "center_lon": 81.5,
        "label": "Ganga Floodplain, UP/Bihar, India"
    }

    def __init__(
        self,
        flood_threshold: float = -16.0,
        ndvi_lo: float = 0.2,
        ndvi_hi: float = 0.5,
        grid_shape: tuple = (256, 256),
        seed: int = 42
    ):
        self.flood_threshold = flood_threshold
        self.ndvi_lo   = ndvi_lo
        self.ndvi_hi   = ndvi_hi
        self.grid_shape = grid_shape
        self.seed = seed
        self._cache: dict = {}

    # ── Public API ──────────────────────────────────────────────────────────
    def get_all(self) -> dict:
        """Return complete processed data bundle (cached after first call)."""
        if not self._cache:
            self._cache = self._run_phase1_pipeline()
        return self._cache

    def invalidate(self):
        """Force re-run (e.g. after threshold slider change)."""
        self._cache = {}

    # ── Phase 1 Pipeline ────────────────────────────────────────────────────
    def _run_phase1_pipeline(self) -> dict:
        logger.info("Phase 2 bridge: running Phase 1 pipeline...")

        # 1. Generate synthetic demo data (Phase 1 utility)
        sar_raw = generate_demo_sar(shape=self.grid_shape, seed=self.seed)
        ms_raw  = generate_demo_multispectral(shape=self.grid_shape, seed=self.seed)

        # 2. Run Phase 1 flood processor
        fp           = FloodProcessor(threshold_db=self.flood_threshold)
        flood_result = fp.process(sar_raw)

        # 3. Run Phase 1 NDVI processor
        np_proc     = NDVIProcessor(low_thresh=self.ndvi_lo, high_thresh=self.ndvi_hi)
        ndvi_result = np_proc.process(ms_raw)

        # 4. Build Phase 2 extensions
        flood_geojson   = self._flood_mask_to_geojson(flood_result["flood_mask"])
        ndvi_df         = self._ndvi_to_dataframe(ndvi_result)
        flood_ts        = self._generate_flood_timeseries()
        ndvi_ts         = self._generate_ndvi_timeseries()
        before_sar      = self._generate_before_sar(sar_raw["vv"])
        risk_grid       = self._compute_risk_grid(flood_result["flood_mask"],
                                                   ndvi_result["ndvi"])
        sensor_points   = self._generate_sensor_locations()

        return {
            # ── Phase 1 raw outputs (pass-through) ──────────────────────────
            "sar_raw":      sar_raw,
            "ms_raw":       ms_raw,
            "flood_result": flood_result,
            "ndvi_result":  ndvi_result,

            # ── Phase 2 GIS extensions ───────────────────────────────────────
            "flood_geojson":  flood_geojson,
            "ndvi_df":        ndvi_df,
            "flood_ts":       flood_ts,
            "ndvi_ts":        ndvi_ts,
            "before_sar":     before_sar,
            "risk_grid":      risk_grid,
            "sensor_points":  sensor_points,
            "bbox":           self.DEFAULT_BBOX,

            # ── Derived metrics ──────────────────────────────────────────────
            "metrics": self._compute_dashboard_metrics(flood_result, ndvi_result),

            # ── Metadata ─────────────────────────────────────────────────────
            "generated_at":    datetime.now().isoformat(),
            "flood_threshold": self.flood_threshold,
            "ndvi_thresholds": {"low": self.ndvi_lo, "high": self.ndvi_hi},
        }

    # ── Derived Data Builders ────────────────────────────────────────────────
    def _flood_mask_to_geojson(self, flood_mask: np.ndarray) -> dict:
        """
        Convert flood binary mask to GeoJSON-like structure.
        Maps pixel grid onto the DEFAULT_BBOX lat/lon space.
        Each flooded pixel → small polygon feature.
        
        For performance: we downsample to 32×32 grid of macro-cells.
        Each macro-cell covers ~6km × ~6km.
        """
        bb   = self.DEFAULT_BBOX
        rows, cols = flood_mask.shape
        # Downsample factor
        ds = 8
        small_mask = flood_mask.reshape(rows//ds, ds, cols//ds, ds).mean(axis=(1,3)) > 0.3

        sr, sc = small_mask.shape
        lat_step = (bb["lat_max"] - bb["lat_min"]) / sr
        lon_step = (bb["lon_max"] - bb["lon_min"]) / sc

        features = []
        for r in range(sr):
            for c in range(sc):
                if not small_mask[r, c]:
                    continue
                lat0 = bb["lat_min"] + r * lat_step
                lon0 = bb["lon_min"] + c * lon_step
                lat1 = lat0 + lat_step
                lon1 = lon0 + lon_step
                features.append({
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[
                            [lon0, lat0], [lon1, lat0],
                            [lon1, lat1], [lon0, lat1], [lon0, lat0]
                        ]]
                    },
                    "properties": {
                        "class": "flooded",
                        "row": int(r), "col": int(c)
                    }
                })
        return {"type": "FeatureCollection", "features": features}

    def _ndvi_to_dataframe(self, ndvi_result: dict) -> pd.DataFrame:
        """
        Flatten NDVI grid to a DataFrame of (lat, lon, ndvi, class) rows.
        Used for heatmap layer and statistics table.
        Downsampled for Folium rendering performance.
        """
        bb   = self.DEFAULT_BBOX
        ndvi = ndvi_result["ndvi"]
        cls  = ndvi_result["classification"]
        rows, cols = ndvi.shape

        ds = 8
        ndvi_ds = ndvi.reshape(rows//ds, ds, cols//ds, ds).mean(axis=(1,3))
        cls_ds  = cls.reshape(rows//ds, ds, cols//ds, ds).mean(axis=(1,3)).round().astype(int)

        sr, sc = ndvi_ds.shape
        lat_arr = np.linspace(bb["lat_max"], bb["lat_min"], sr)
        lon_arr = np.linspace(bb["lon_min"], bb["lon_max"], sc)

        lats, lons, vals, classes = [], [], [], []
        for r in range(sr):
            for c in range(sc):
                v = float(ndvi_ds[r, c])
                if np.isnan(v):
                    continue
                lats.append(float(lat_arr[r]))
                lons.append(float(lon_arr[c]))
                vals.append(round(v, 4))
                classes.append(int(cls_ds[r, c]))

        class_labels = {0: "Low Vegetation", 1: "Moderate Vegetation", 2: "Healthy Vegetation"}
        df = pd.DataFrame({
            "lat": lats, "lon": lons,
            "ndvi": vals, "class_id": classes,
            "class_label": [class_labels.get(c, "Unknown") for c in classes]
        })
        return df

    def _generate_flood_timeseries(self, n_points: int = 40) -> pd.DataFrame:
        """Simulate multi-date Sentinel-1 flood extent observations."""
        np.random.seed(101)
        base_date = datetime(2024, 5, 1)
        dates  = [base_date + timedelta(days=6*i) for i in range(n_points)]
        trend  = np.linspace(4, 7, n_points)
        event  = 30 * np.exp(-((np.arange(n_points) - n_points*0.45)**2) / (2*(n_points*0.09)**2))
        noise  = np.random.normal(0, 1.2, n_points)
        extent = np.clip(trend + event + noise, 0, 55).round(2)
        area_km2 = (extent / 100 * 300 * 300).round(1)  # rough km²
        return pd.DataFrame({
            "date": dates,
            "flood_pct": extent,
            "area_km2":  area_km2,
            "satellite": "Sentinel-1 VV"
        })

    def _generate_ndvi_timeseries(self, n_points: int = 20) -> pd.DataFrame:
        """Simulate monthly NDVI means across a growing season."""
        np.random.seed(202)
        base_date = datetime(2024, 3, 1)
        dates = [base_date + timedelta(days=10*i) for i in range(n_points)]
        # Crop phenology: germination → growth → peak → senescence
        x = np.linspace(0, np.pi, n_points)
        ndvi_mean  = 0.18 + 0.52 * np.sin(x)**1.4 + np.random.normal(0, 0.02, n_points)
        ndvi_p25   = ndvi_mean - 0.08 + np.random.normal(0, 0.01, n_points)
        ndvi_p75   = ndvi_mean + 0.08 + np.random.normal(0, 0.01, n_points)
        return pd.DataFrame({
            "date": dates,
            "ndvi_mean": np.clip(ndvi_mean, 0, 1).round(4),
            "ndvi_p25":  np.clip(ndvi_p25,  0, 1).round(4),
            "ndvi_p75":  np.clip(ndvi_p75,  0, 1).round(4),
            "satellite": "Sentinel-2 B8/B4"
        })

    def _generate_before_sar(self, after_vv: np.ndarray) -> np.ndarray:
        """
        Simulate a 'before-flood' SAR scene by pushing low-backscatter
        pixels up (less flooding) and adding slight noise offset.
        """
        rng = np.random.default_rng(999)
        before = after_vv.copy()
        # Raise the very dark (flooded) pixels to simulate pre-flood land
        flooded_mask = before < -17
        before[flooded_mask] += rng.normal(6, 1.5, flooded_mask.sum()).astype(np.float32)
        before += rng.normal(0, 0.5, before.shape).astype(np.float32)
        return before.astype(np.float32)

    def _compute_risk_grid(self, flood_mask: np.ndarray, ndvi: np.ndarray) -> np.ndarray:
        """
        Composite risk index per pixel:
          risk = 0.6 × flood_score + 0.4 × crop_stress_score
        Flood score: 1 where flooded, 0 elsewhere (smoothed)
        Crop stress: 1 - norm(NDVI) in [0,1]
        """
        flood_score = gaussian_filter(flood_mask.astype(float), sigma=5)
        ndvi_clean  = np.where(np.isnan(ndvi), 0, ndvi)
        ndvi_norm   = (ndvi_clean - ndvi_clean.min()) / (ndvi_clean.max() - ndvi_clean.min() + 1e-8)
        stress      = 1.0 - ndvi_norm
        risk        = 0.6 * flood_score + 0.4 * stress
        risk        = (risk - risk.min()) / (risk.max() - risk.min())
        return risk.astype(np.float32)

    def _generate_sensor_locations(self) -> pd.DataFrame:
        """
        Placeholder IoT/weather sensor locations within the study AOI.
        Phase 3+ will pull live readings from these sensors.
        """
        bb = self.DEFAULT_BBOX
        np.random.seed(77)
        n = 12
        lats = np.random.uniform(bb["lat_min"] + 0.1, bb["lat_max"] - 0.1, n)
        lons = np.random.uniform(bb["lon_min"] + 0.1, bb["lon_max"] - 0.1, n)
        names = [f"Sensor-{chr(65+i)}{i+1:02d}" for i in range(n)]
        types = np.random.choice(
            ["Flood Gauge", "Weather Station", "Soil Moisture", "NDVI Camera"],
            n
        )
        battery = np.random.randint(45, 100, n)
        signal  = np.random.choice(["Strong", "Moderate", "Weak"], n,
                                   p=[0.5, 0.35, 0.15])
        return pd.DataFrame({
            "name": names, "type": types,
            "lat": lats.round(4), "lon": lons.round(4),
            "battery_pct": battery,
            "signal": signal,
            "last_ping": [
                (datetime.now() - timedelta(minutes=int(m))).strftime("%H:%M")
                for m in np.random.randint(1, 90, n)
            ]
        })

    def _compute_dashboard_metrics(
        self,
        flood_result: dict,
        ndvi_result: dict
    ) -> dict:
        """Pre-compute all KPI metrics for the dashboard header."""
        fm  = flood_result["flood_mask"]
        ndvi = ndvi_result["ndvi"]
        cls  = ndvi_result["classification"]

        flooded_pct   = float(np.mean(fm)) * 100
        flooded_km2   = flooded_pct / 100 * 300 * 300  # ~300 × 300 km AOI
        ndvi_mean     = float(np.nanmean(ndvi))
        ndvi_std      = float(np.nanstd(ndvi))
        healthy_pct   = float(np.mean(cls == 2)) * 100
        moderate_pct  = float(np.mean(cls == 1)) * 100
        low_pct       = float(np.mean(cls == 0)) * 100
        risk_score    = min(100, flooded_pct * 3.2 + (1 - ndvi_mean) * 30)

        return {
            "flooded_pct":   round(flooded_pct, 1),
            "flooded_km2":   round(flooded_km2, 0),
            "ndvi_mean":     round(ndvi_mean, 4),
            "ndvi_std":      round(ndvi_std, 4),
            "healthy_pct":   round(healthy_pct, 1),
            "moderate_pct":  round(moderate_pct, 1),
            "low_pct":       round(low_pct, 1),
            "risk_score":    round(risk_score, 1),
            "total_pixels":  int(fm.size),
            "flooded_pixels": int(fm.sum()),
        }
