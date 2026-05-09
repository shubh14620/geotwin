# 🛰️ GeoTwin Phase 1
## AI-Driven Geospatial Digital Twin for Multi-Hazard Environmental Intelligence
### Using IoT · SAR Remote Sensing · GIS

---

> **B.Tech Major Project | Electronics & Communication Engineering | 2024–25**
> Built with Python · Streamlit · Google Earth Engine · Sentinel-1/2

---

## 📋 Project Overview

This project implements Phase 1 of an AI-driven Geospatial Digital Twin system
for multi-hazard environmental monitoring. The system integrates satellite remote
sensing, cloud-based geospatial computing (Google Earth Engine), and an
interactive web dashboard for real-time environmental intelligence.

**Phase 1 implements two core modules:**
| Module | Satellite | Method | Output |
|--------|-----------|--------|--------|
| Flood Detection | Sentinel-1 SAR | Backscatter Thresholding | Binary Flood Map |
| NDVI Crop Health | Sentinel-2 MSI | NDVI Ratio (B8-B4)/(B8+B4) | Vegetation Classification |

---

## 🗂️ Project Structure

```
geospatial_twin/
│
├── 📁 dashboard/
│   └── app.py                    # Main Streamlit dashboard application
│
├── 📁 flood_detection/
│   ├── __init__.py
│   ├── flood_processor.py        # Core SAR flood detection engine
│   └── gee_flood.py              # Google Earth Engine integration
│
├── 📁 ndvi_monitoring/
│   ├── __init__.py
│   ├── ndvi_processor.py         # NDVI computation & classification
│   └── gee_ndvi.py               # GEE Sentinel-2 integration
│
├── 📁 utils/
│   ├── __init__.py
│   ├── demo_data.py              # Synthetic data generator (no GEE needed)
│   ├── visualization.py          # Reusable plotting functions
│   └── export_utils.py           # GeoTIFF / PNG / CSV / QGIS export
│
├── 📁 config/
│   └── gee_setup.py              # GEE authentication & setup utilities
│
├── 📁 data/
│   ├── raw/                      # Raw downloaded satellite data
│   └── processed/                # Preprocessed intermediate outputs
│
├── 📁 outputs/
│   ├── flood_maps/               # Exported flood classification maps
│   ├── ndvi_maps/                # Exported NDVI vegetation maps
│   └── reports/                  # JSON / CSV analysis reports
│
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

---

## ⚡ Quick Start (5 Minutes)

### Step 1 — Clone / Download
```bash
git clone <your-repo-url>
cd geospatial_twin
```

### Step 2 — Create Virtual Environment
```bash
python -m venv venv

# Windows:
venv\Scripts\activate

# macOS/Linux:
source venv/bin/activate
```

### Step 3 — Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4 — Run Dashboard
```bash
cd dashboard
streamlit run app.py
```

The dashboard opens at `http://localhost:8501` in your browser.

> **Note:** The dashboard runs in "Demo Mode" by default using synthetic data.
> No GEE credentials needed for demonstration!

---

## 🌍 Google Earth Engine Setup (Real Satellite Data)

### Step 1 — Get GEE Access
1. Visit: https://earthengine.google.com/
2. Click "Sign Up" with your Google account
3. For academic use, approval is usually instant

### Step 2 — Authenticate
```bash
# Method 1: Command line (recommended)
earthengine authenticate

# Method 2: Python script
python config/gee_setup.py
```

### Step 3 — Initialize in Your Code
```python
import ee
ee.Initialize(project="your-gee-project-id")
```

### Step 4 — Generate GEE Scripts
```python
from flood_detection.gee_flood import GEEFloodDetector

detector = GEEFloodDetector(
    roi_coords=[80.0, 25.0, 83.0, 27.0],  # Ganga floodplain
    start_date="2024-07-01",
    end_date="2024-07-31",
    threshold_db=-16.0
)

# Save JavaScript for GEE Code Editor
detector.save_script("gee_flood_script.js")

# Run Python API (requires GEE init)
# flood_image = detector.run_python_api()
```

---

## 🔬 Scientific Methodology

### Flood Detection (Sentinel-1 SAR)

```
THEORY:
  Flooded surfaces act as specular reflectors — microwave signal bounces away
  from the sensor, returning very LOW backscatter (σ°).
  Dry land shows higher, more diffuse backscatter.

PROCESSING CHAIN:
  1. Load Sentinel-1 GRD (Ground Range Detected)
  2. Select VV polarization (optimal for open water)
  3. Convert to dB: σ°(dB) = 10 × log₁₀(σ°)
  4. Apply speckle filter (Lee / Refined Lee)
  5. Classify: σ°(VV) < threshold → FLOODED
  6. Remove small objects (morphological cleaning)
  7. Export GeoTIFF + PNG
```

### NDVI Crop Health (Sentinel-2)

```
FORMULA:
              NIR - RED        B8 - B4
  NDVI = ─────────────── = ─────────────
              NIR + RED        B8 + B4

CLASSIFICATION:
  NDVI < 0.2           → Low Vegetation (bare soil, stressed crops)
  0.2 ≤ NDVI < 0.5     → Moderate Vegetation (developing crops)
  NDVI ≥ 0.5           → Healthy Vegetation (peak canopy density)

PROCESSING CHAIN:
  1. Load Sentinel-2 SR Harmonized (Level-2A)
  2. Apply cloud mask (SCL band)
  3. Scale reflectance (÷ 10000)
  4. Compute NDVI pixel-wise
  5. Classify into 3 health categories
  6. Export GeoTIFF + PNG
```

---

## 📊 Dashboard Features

| Feature | Description |
|---------|-------------|
| 🏠 Overview | Key metrics + side-by-side flood/NDVI maps |
| 🌊 Flood Detection | Full SAR analysis, backscatter histogram, accuracy assessment |
| 🌿 NDVI Monitoring | Continuous NDVI + classification + band correlation |
| 📊 Analytics | Multi-hazard risk index, temporal time series |
| ⚙️ Sidebar Controls | Threshold sliders, date range, data source selection |

---

## 📤 Exporting Results

```python
from utils.export_utils import (
    export_flood_map_png,
    export_ndvi_map_png,
    export_geotiff,
    export_qgis_style_file,
    generate_report_json
)

# Export flood map as PNG
export_flood_map_png(flood_mask, vv_db, "outputs/flood_maps/flood_2024.png")

# Export as GeoTIFF (QGIS-compatible)
export_geotiff(flood_mask.astype("uint8"), "outputs/flood_maps/flood.tif",
               bbox=[80.0, 25.0, 83.0, 27.0])

# Export QGIS style file
export_qgis_style_file("outputs/flood_style.qml", mode="flood")

# Export JSON report
generate_report_json(flood_stats, ndvi_stats, "outputs/reports/report.json")
```

---

## 🛠️ Tech Stack

| Technology | Role |
|------------|------|
| **Python 3.10+** | Core programming language |
| **Streamlit** | Interactive web dashboard |
| **Google Earth Engine** | Cloud-based satellite image processing |
| **Sentinel-1 SAR** | Flood detection (C-band, VV polarization) |
| **Sentinel-2 MSI** | NDVI vegetation analysis (B4 + B8) |
| **NumPy / SciPy** | Array processing, speckle filtering |
| **Matplotlib** | Static maps and figures |
| **Plotly** | Interactive charts and time series |
| **rasterio** | GeoTIFF I/O for QGIS compatibility |
| **GeoPandas** | Vector geospatial data handling |

---

## 📚 References

1. Twele, A., et al. (2016). Sentinel-1 based flood mapping: a fully automated processing chain. *International Journal of Remote Sensing*.
2. Martinis, S., et al. (2015). Towards operational near-real-time flood detection using a split-based automatic thresholding procedure on high resolution TerraSAR-X data. *Remote Sensing*.
3. Rouse, J.W., et al. (1974). Monitoring vegetation systems in the Great Plains with ERTS. *NASA*.
4. Drusch, M., et al. (2012). Sentinel-2: ESA's Optical High-Resolution Mission for GMES Operational Services. *Remote Sensing of Environment*.
5. Gorelick, N., et al. (2017). Google Earth Engine: Planetary-scale geospatial analysis for everyone. *Remote Sensing of Environment*.

---

## 🚀 Future Phases

| Phase | Feature |
|-------|---------|
| Phase 2 | Drought Monitoring (SPI, VCI indices) |
| Phase 3 | Urban Heat Island (LST from Landsat-8/9) |
| Phase 4 | IoT Sensor Data Fusion (real-time ground truth) |
| Phase 5 | ML/DL Classification (Random Forest, U-Net) |
| Phase 6 | Digital Twin 3D Visualization |

---

*Built for B.Tech Major Project · ECE Department · 2024–25*
*Stack: Python · Streamlit · Google Earth Engine · Sentinel-1/2 · QGIS*
