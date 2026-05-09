"""
================================================================================
  dashboard/utils/export_manager.py
  Phase 2 Export Manager
  
  Generates in-memory download buffers for Streamlit's st.download_button.
  Supports: PNG maps, CSV stats, GeoJSON, GeoTIFF (if rasterio), JSON report.
================================================================================
"""

import io
import json
import csv
import zipfile
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# ── Try importing optional heavy deps ────────────────────────────────────────
try:
    import rasterio
    from rasterio.transform import from_bounds
    from rasterio.crs import CRS
    RASTERIO_OK = True
except ImportError:
    RASTERIO_OK = False

try:
    import folium
    FOLIUM_OK = True
except ImportError:
    FOLIUM_OK = False


# ─────────────────────────────────────────────────────────────────────────────

def flood_map_png_bytes(flood_mask: np.ndarray, vv_db: np.ndarray) -> bytes:
    """Render flood classification map to PNG bytes for download."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=140)
    fig.patch.set_facecolor("#0d1117")
    for ax in axes:
        ax.set_facecolor("#111827")
        for sp in ax.spines.values():
            sp.set_edgecolor("#1e2d40")

    # SAR backscatter
    im = axes[0].imshow(vv_db, cmap="RdBu_r", vmin=-25, vmax=0)
    axes[0].set_title("Sentinel-1 SAR VV Backscatter (dB)",
                       color="#8b949e", fontsize=10, pad=8)
    axes[0].axis("off")
    cb = plt.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)
    cb.ax.tick_params(colors="#8b949e", labelsize=8)
    cb.set_label("dB", color="#8b949e", fontsize=9)

    # Flood mask overlay
    axes[1].imshow(vv_db, cmap="gray", vmin=-25, vmax=0, alpha=0.4)
    overlay = np.ma.masked_where(~flood_mask, np.ones_like(flood_mask, dtype=float))
    axes[1].imshow(overlay, cmap=mcolors.ListedColormap(["#3b82f6"]), alpha=0.78)
    axes[1].set_title("Flood Classification Map — Sentinel-1 SAR",
                       color="#8b949e", fontsize=10, pad=8)
    axes[1].axis("off")
    # Stats text
    pct = flood_mask.mean() * 100
    axes[1].text(0.02, 0.97, f"Flooded: {pct:.1f}%",
                 transform=axes[1].transAxes, fontsize=9, color="#3b82f6",
                 va="top", fontfamily="monospace",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="#0d1117",
                           edgecolor="#3b82f6", alpha=0.85))
    ts = datetime.now().strftime("%Y-%m-%d %H:%M UTC")
    fig.text(0.5, -0.01, f"GeoTwin Phase 2 · Sentinel-1 SAR Flood Detection · {ts}",
             ha="center", color="#484f58", fontsize=8, fontfamily="monospace")

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=140, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def ndvi_map_png_bytes(ndvi: np.ndarray, classification: np.ndarray) -> bytes:
    """Render NDVI vegetation map to PNG bytes for download."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=140)
    fig.patch.set_facecolor("#0d1117")
    for ax in axes:
        ax.set_facecolor("#111827")
        for sp in ax.spines.values():
            sp.set_edgecolor("#1e2d40")

    ndvi_plot = np.where(np.isnan(ndvi), -0.5, ndvi)
    im = axes[0].imshow(ndvi_plot, cmap="RdYlGn", vmin=-0.2, vmax=0.8)
    axes[0].set_title("NDVI Continuous Values — Sentinel-2",
                       color="#8b949e", fontsize=10, pad=8)
    axes[0].axis("off")
    cb = plt.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)
    cb.ax.tick_params(colors="#8b949e", labelsize=8)
    cb.set_label("NDVI", color="#8b949e", fontsize=9)

    cls_cmap = mcolors.ListedColormap(["#ef4444", "#f59e0b", "#10b981"])
    axes[1].imshow(classification, cmap=cls_cmap, vmin=0, vmax=2)
    axes[1].set_title("Vegetation Health Classification — Sentinel-2",
                       color="#8b949e", fontsize=10, pad=8)
    axes[1].axis("off")
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#ef4444", label="Low Vegetation"),
        Patch(facecolor="#f59e0b", label="Moderate Vegetation"),
        Patch(facecolor="#10b981", label="Healthy Vegetation"),
    ]
    axes[1].legend(handles=legend_elements, loc="lower right",
                   facecolor="#0d1117", edgecolor="#1e2d40",
                   labelcolor="#8b949e", fontsize=8)

    mean_ndvi = float(np.nanmean(ndvi))
    axes[1].text(0.02, 0.97, f"Mean NDVI: {mean_ndvi:.4f}",
                 transform=axes[1].transAxes, fontsize=9, color="#10b981",
                 va="top", fontfamily="monospace",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="#0d1117",
                           edgecolor="#10b981", alpha=0.85))

    ts = datetime.now().strftime("%Y-%m-%d %H:%M UTC")
    fig.text(0.5, -0.01, f"GeoTwin Phase 2 · Sentinel-2 NDVI Analysis · {ts}",
             ha="center", color="#484f58", fontsize=8, fontfamily="monospace")

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=140, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def stats_csv_bytes(metrics: dict, flood_ts: pd.DataFrame, ndvi_ts: pd.DataFrame) -> bytes:
    """Generate a comprehensive CSV statistics file as bytes."""
    buf = io.StringIO()
    writer = csv.writer(buf)

    writer.writerow(["# GeoTwin Phase 2 — Analysis Statistics Report"])
    writer.writerow([f"# Generated: {datetime.now().isoformat()}"])
    writer.writerow([])

    writer.writerow(["=== KEY METRICS ==="])
    writer.writerow(["Parameter", "Value", "Unit"])
    for k, v in metrics.items():
        writer.writerow([k, v, ""])
    writer.writerow([])

    writer.writerow(["=== FLOOD TIME SERIES ==="])
    writer.writerow(["Date", "Flood_Pct", "Area_km2", "Satellite"])
    for _, row in flood_ts.iterrows():
        writer.writerow([row["date"].strftime("%Y-%m-%d"),
                         row["flood_pct"], row["area_km2"], row["satellite"]])
    writer.writerow([])

    writer.writerow(["=== NDVI TIME SERIES ==="])
    writer.writerow(["Date", "NDVI_Mean", "NDVI_P25", "NDVI_P75", "Satellite"])
    for _, row in ndvi_ts.iterrows():
        writer.writerow([row["date"].strftime("%Y-%m-%d"),
                         row["ndvi_mean"], row["ndvi_p25"],
                         row["ndvi_p75"], row["satellite"]])

    return buf.getvalue().encode("utf-8")


def geojson_bytes(flood_geojson: dict) -> bytes:
    """Serialize flood GeoJSON to bytes for download."""
    return json.dumps(flood_geojson, indent=2).encode("utf-8")


def ndvi_dataframe_csv_bytes(ndvi_df: pd.DataFrame) -> bytes:
    """Export NDVI pixel dataframe as CSV bytes."""
    return ndvi_df.to_csv(index=False).encode("utf-8")


def geotiff_bytes(
    array: np.ndarray,
    bbox: dict,
    dtype: str = "float32",
    nodata: float = -9999.0
) -> bytes:
    """
    Export array to GeoTIFF bytes (QGIS-compatible).
    Returns None if rasterio is not installed.
    """
    if not RASTERIO_OK:
        return None

    rows, cols = array.shape
    transform = from_bounds(
        bbox["lon_min"], bbox["lat_min"],
        bbox["lon_max"], bbox["lat_max"],
        cols, rows
    )

    buf = io.BytesIO()
    with rasterio.open(
        buf, "w",
        driver="GTiff",
        height=rows, width=cols,
        count=1,
        dtype=dtype,
        crs=CRS.from_epsg(4326),
        transform=transform,
        nodata=nodata,
        compress="lzw"
    ) as dst:
        dst.write(array.astype(dtype), 1)
    buf.seek(0)
    return buf.getvalue()


def full_export_zip(data: dict) -> bytes:
    """
    Bundle all export artifacts into a single ZIP download.
    
    Contents:
      - flood_map.png
      - ndvi_map.png
      - flood_classification.geojson
      - ndvi_pixels.csv
      - statistics.csv
      - analysis_report.json
      - flood_mask.tif (if rasterio available)
      - ndvi_values.tif (if rasterio available)
    """
    fr = data["flood_result"]
    nr = data["ndvi_result"]
    metrics   = data["metrics"]
    flood_ts  = data["flood_ts"]
    ndvi_ts   = data["ndvi_ts"]
    bbox      = data["bbox"]

    report = {
        "project":      "GeoTwin Phase 2 — Multi-Hazard Environmental Intelligence",
        "generated_at": datetime.now().isoformat(),
        "phase":        "Phase 2 — GIS Dashboard",
        "study_area":   bbox["label"],
        "bbox":         bbox,
        "metrics":      metrics,
        "satellites": {
            "flood": "Sentinel-1 GRD · VV polarization",
            "ndvi":  "Sentinel-2 MSI Level-2A · B4 + B8"
        }
    }

    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("flood_map.png",              flood_map_png_bytes(fr["flood_mask"], fr["backscatter_db"]))
        zf.writestr("ndvi_map.png",               ndvi_map_png_bytes(nr["ndvi"], nr["classification"]))
        zf.writestr("flood_classification.geojson", geojson_bytes(data["flood_geojson"]))
        zf.writestr("ndvi_pixels.csv",            ndvi_dataframe_csv_bytes(data["ndvi_df"]))
        zf.writestr("statistics.csv",             stats_csv_bytes(metrics, flood_ts, ndvi_ts))
        zf.writestr("analysis_report.json",       json.dumps(report, indent=2, default=str).encode())

        # GeoTIFF exports (optional)
        if RASTERIO_OK:
            flood_tif = geotiff_bytes(fr["flood_mask"].astype("uint8"), bbox, dtype="uint8", nodata=255)
            if flood_tif:
                zf.writestr("flood_mask.tif", flood_tif)
            ndvi_tif = geotiff_bytes(nr["ndvi"], bbox, dtype="float32")
            if ndvi_tif:
                zf.writestr("ndvi_values.tif", ndvi_tif)
            cls_tif = geotiff_bytes(nr["classification"].astype("uint8"), bbox, dtype="uint8", nodata=255)
            if cls_tif:
                zf.writestr("ndvi_classification.tif", cls_tif)
        else:
            zf.writestr(
                "README_GEOTIFF.txt",
                "Install rasterio to enable GeoTIFF export:\n  pip install rasterio\n"
            )

    zip_buf.seek(0)
    return zip_buf.getvalue()
