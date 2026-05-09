"""
================================================================================
  config/gee_setup.py
  Google Earth Engine Authentication & Initialization
================================================================================

  STEP-BY-STEP SETUP GUIDE:
  ──────────────────────────
  1. Create a Google Account (if you don't have one)
  2. Go to: https://earthengine.google.com/  → click "Sign Up"
  3. Wait for GEE access approval (usually instant for academic use)
  4. Install earthengine-api: pip install earthengine-api
  5. Run: python config/gee_setup.py
     OR in terminal: earthengine authenticate
  6. Follow the URL shown → grant permission → paste auth code

================================================================================
"""

import logging

logger = logging.getLogger(__name__)


def authenticate_gee():
    """
    Authenticate with Google Earth Engine.
    Run this once to save credentials to ~/.config/earthengine/credentials
    """
    try:
        import ee
        ee.Authenticate()
        ee.Initialize(project="your-gee-project-id")  # Replace with your project ID
        print("✅ GEE Authentication successful!")
        print(f"   GEE Version: {ee.__version__}")
        return True
    except ImportError:
        print("❌ earthengine-api not installed. Run: pip install earthengine-api")
        return False
    except Exception as e:
        print(f"❌ GEE Authentication failed: {e}")
        return False


def test_gee_connection():
    """Quick test to verify GEE is working."""
    try:
        import ee
        ee.Initialize()
        # Simple test: get info about a Sentinel-1 image
        img = ee.ImageCollection("COPERNICUS/S1_GRD").first()
        info = img.getInfo()
        print(f"✅ GEE Connection OK! Test image ID: {info['id']}")
        return True
    except Exception as e:
        print(f"❌ GEE test failed: {e}")
        return False


def get_roi_geometry(coords: list):
    """
    Create a GEE Rectangle geometry from bounding box coordinates.

    Parameters
    ----------
    coords : list
        [west, south, east, north] in decimal degrees.
        Example: [72.5, 18.0, 74.0, 20.5]  (Mumbai region)

    Returns
    -------
    ee.Geometry.Rectangle
    """
    try:
        import ee
        return ee.Geometry.Rectangle(coords)
    except ImportError:
        raise ImportError("earthengine-api required. Install: pip install earthengine-api")


# ── Predefined Study Areas ────────────────────────────────────────────────────
STUDY_AREAS = {
    "Mumbai_Region":       [72.5,  18.8,  73.5,  19.5],
    "Ganga_Floodplain":    [80.0,  25.0,  83.0,  27.0],
    "Brahmaputra_Assam":   [90.0,  25.5,  93.0,  27.5],
    "Kerala_Coast":        [75.5,   8.0,  77.5,  12.0],
    "Punjab_Cropland":     [74.0,  29.0,  77.0,  32.0],
    "Odisha_Delta":        [85.0,  19.5,  87.5,  21.5],
    "Bangladesh_Meghna":   [90.0,  22.0,  91.5,  24.5],
}


if __name__ == "__main__":
    print("=" * 60)
    print("  GeoTwin — Google Earth Engine Setup")
    print("=" * 60)
    authenticate_gee()
    test_gee_connection()
