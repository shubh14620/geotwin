"""
================================================================================
  utils/export_utils.py
  Export Processed Results to QGIS-Compatible Formats
================================================================================

  Supported output formats:
    - GeoTIFF (.tif)  — raster classification maps
    - PNG              — visual maps for reports
    - GeoJSON          — vector flood boundaries
    - CSV              — statistics tables
    - QGIS project (.qgz) — pre-styled QGIS project (via pyqgis or manual)

================================================================================
"""

import numpy as np
import os
import json
import csv
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Optional imports (only needed for GeoTIFF export)
try:
    import rasterio
    from rasterio.transform import from_bounds
    from rasterio.crs import CRS
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
def export_flood_map_png(
    flood_mask: np.ndarray,
    vv_db: np.ndarray,
    output_path: str = "outputs/flood_maps/flood_map.png",
    dpi: int = 150
):
    """
    Export flood classification map as PNG.
    Compatible with any GIS software for visual inspection.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 8), dpi=dpi)
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#111827")

    ax.imshow(vv_db, cmap="gray", vmin=-25, vmax=0, alpha=0.4)
    flood_overlay = np.ma.masked_where(~flood_mask, flood_mask.astype(float))
    ax.imshow(flood_overlay, cmap=mcolors.ListedColormap(["#3b82f6"]), alpha=0.75)

    ax.set_title("Flood Classification Map — Sentinel-1 SAR",
                 color="#8b949e", fontsize=12, pad=10)
    ax.axis("off")

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M UTC")
    ax.text(0.01, 0.01, f"Generated: {timestamp} | GeoTwin Phase 1",
            transform=ax.transAxes, fontsize=7, color="#484f58", va="bottom")

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"✅ Flood map exported: {output_path}")
    return output_path


# ─────────────────────────────────────────────────────────────────────────────
def export_ndvi_map_png(
    ndvi: np.ndarray,
    classification: np.ndarray,
    output_path: str = "outputs/ndvi_maps/ndvi_map.png",
    dpi: int = 150
):
    """Export NDVI vegetation map as PNG."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=dpi)
    fig.patch.set_facecolor("#0d1117")

    for ax in axes:
        ax.set_facecolor("#111827")

    # NDVI continuous
    im = axes[0].imshow(np.where(np.isnan(ndvi), -0.5, ndvi),
                        cmap="RdYlGn", vmin=-0.2, vmax=0.8)
    axes[0].set_title("NDVI Continuous Values", color="#8b949e", fontsize=11)
    axes[0].axis("off")
    cb = plt.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)
    cb.ax.tick_params(colors="#8b949e")

    # Classification
    cmap = mcolors.ListedColormap(["#ef4444", "#f59e0b", "#10b981"])
    axes[1].imshow(classification, cmap=cmap, vmin=0, vmax=2)
    axes[1].set_title("Vegetation Health Classification", color="#8b949e", fontsize=11)
    axes[1].axis("off")

    plt.suptitle("NDVI Crop Health Monitoring — Sentinel-2",
                 color="#8b949e", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"✅ NDVI map exported: {output_path}")
    return output_path


# ─────────────────────────────────────────────────────────────────────────────
def export_geotiff(
    array: np.ndarray,
    output_path: str,
    bbox: list = None,
    crs_epsg: int = 4326,
    dtype: str = "float32",
    nodata: float = -9999.0
):
    """
    Export a numpy array as a GeoTIFF file.
    Requires rasterio: pip install rasterio

    Parameters
    ----------
    array : 2D numpy array
        Data to export.
    output_path : str
        Output .tif file path.
    bbox : list, optional
        [west, south, east, north] in CRS units. If None, uses pixel coords.
    crs_epsg : int
        EPSG code for the CRS (default 4326 = WGS84 geographic).
    dtype : str
        Output data type: 'float32', 'uint8', 'int16', etc.
    nodata : float
        Value to use for no-data pixels.
    """
    if not RASTERIO_AVAILABLE:
        print("⚠️  rasterio not installed. Install with: pip install rasterio")
        return None

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    rows, cols = array.shape
    if bbox:
        west, south, east, north = bbox
        transform = from_bounds(west, south, east, north, cols, rows)
    else:
        from rasterio.transform import from_origin
        transform = from_origin(0, rows, 1, 1)

    np_dtype = getattr(np, dtype)
    arr = array.astype(np_dtype)

    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=rows,
        width=cols,
        count=1,
        dtype=dtype,
        crs=CRS.from_epsg(crs_epsg),
        transform=transform,
        nodata=nodata,
        compress="lzw"
    ) as dst:
        dst.write(arr, 1)

    print(f"✅ GeoTIFF exported: {output_path} ({rows}×{cols}, EPSG:{crs_epsg})")
    return output_path


# ─────────────────────────────────────────────────────────────────────────────
def export_stats_csv(stats: dict, output_path: str):
    """Export analysis statistics to CSV."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    rows = []
    def flatten(d, prefix=""):
        for k, v in d.items():
            key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                flatten(v, key)
            else:
                rows.append({"Parameter": key, "Value": v})
    flatten(stats)

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Parameter", "Value"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"✅ Statistics CSV exported: {output_path}")
    return output_path


# ─────────────────────────────────────────────────────────────────────────────
def export_qgis_style_file(
    output_path: str = "outputs/flood_style.qml",
    mode: str = "flood"
):
    """
    Export a QGIS style file (.qml) for the classification rasters.
    Load in QGIS: Layer Properties → Style → Load Style.
    """
    if mode == "flood":
        qml_content = """<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis styleCategories="AllStyleCategories" version="3.22">
  <pipe>
    <provider>
      <resampling maxOversampling="2" enabled="false" zoomedInResamplingMethod="nearestNeighbour"/>
    </provider>
    <rasterrenderer opacity="1" type="paletted" alphaBand="-1" band="1">
      <colorPalette>
        <paletteEntry alpha="255" color="#1e40af" label="Flooded" value="1"/>
        <paletteEntry alpha="255" color="#d1d5db" label="Non-Flooded" value="0"/>
      </colorPalette>
    </rasterrenderer>
  </pipe>
</qgis>"""
    else:  # ndvi classification
        qml_content = """<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis styleCategories="AllStyleCategories" version="3.22">
  <pipe>
    <rasterrenderer opacity="1" type="paletted" alphaBand="-1" band="1">
      <colorPalette>
        <paletteEntry alpha="255" color="#ef4444" label="Low Vegetation" value="0"/>
        <paletteEntry alpha="255" color="#f59e0b" label="Moderate Vegetation" value="1"/>
        <paletteEntry alpha="255" color="#10b981" label="Healthy Vegetation" value="2"/>
      </colorPalette>
    </rasterrenderer>
  </pipe>
</qgis>"""

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        f.write(qml_content)
    print(f"✅ QGIS style file exported: {output_path}")
    return output_path


# ─────────────────────────────────────────────────────────────────────────────
def generate_report_json(
    flood_stats: dict,
    ndvi_stats: dict,
    output_path: str = "outputs/reports/analysis_report.json"
):
    """Generate a combined JSON report for the analysis run."""
    report = {
        "project": "GeoTwin Phase 1 — Multi-Hazard Environmental Intelligence",
        "generated_at": datetime.now().isoformat(),
        "modules": {
            "flood_detection": {
                "satellite": "Sentinel-1 SAR",
                "method":    "SAR Backscatter Thresholding",
                "stats":     flood_stats
            },
            "ndvi_monitoring": {
                "satellite": "Sentinel-2 MSI",
                "method":    "NDVI Ratio Index",
                "stats":     ndvi_stats
            }
        }
    }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"✅ JSON report saved: {output_path}")
    return output_path
