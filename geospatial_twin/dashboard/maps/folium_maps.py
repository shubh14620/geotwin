"""
================================================================================
  dashboard/maps/folium_maps.py
  Interactive GIS Map Builder using Folium
  
  Builds all Folium map objects consumed by the Phase 2 Streamlit dashboard:
    - Base map with satellite/OSM tiles
    - Flood classification layer (GeoJSON polygons)
    - NDVI heatmap layer (choropleth / circle markers)
    - Risk grid layer
    - IoT sensor layer
    - Before/After comparison map
    - Layer control (toggleable layers)
================================================================================
"""

import folium
from folium import plugins
from folium.plugins import (
    HeatMap, MiniMap, MousePosition, MeasureControl,
    Fullscreen, LocateControl
)
import numpy as np
import pandas as pd
import json
import logging

logger = logging.getLogger(__name__)


# ── Color palettes ─────────────────────────────────────────────────────────────
FLOOD_COLOR     = "#1d4ed8"
FLOOD_FILL      = "#3b82f6"
DRY_COLOR       = "#6b7280"
NDVI_COLORS     = {0: "#ef4444", 1: "#f59e0b", 2: "#10b981"}
NDVI_LABELS     = {0: "Low Vegetation", 1: "Moderate Vegetation", 2: "Healthy Vegetation"}
RISK_COLORSCALE = ["#14532d", "#16a34a", "#facc15", "#f97316", "#dc2626"]

SENSOR_ICONS = {
    "Flood Gauge":    ("tint",       "blue"),
    "Weather Station":("cloud",      "gray"),
    "Soil Moisture":  ("leaf",       "green"),
    "NDVI Camera":    ("camera",     "purple"),
}

TILE_PROVIDERS = {
    "Satellite (Esri)":
        "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    "OpenStreetMap":
        "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
    "CartoDB Dark":
        "https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
    "Topographic (OpenTopo)":
        "https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png",
}

TILE_ATTRS = {
    "Satellite (Esri)":
        "Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS",
    "OpenStreetMap":
        '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>',
    "CartoDB Dark":
        '&copy; <a href="https://carto.com/">CARTO</a>',
    "Topographic (OpenTopo)":
        "Map data: &copy; OpenStreetMap contributors, SRTM | Style: &copy; OpenTopoMap",
}


# ─────────────────────────────────────────────────────────────────────────────
def build_base_map(
    bbox: dict,
    basemap: str = "CartoDB Dark",
    zoom_start: int = 8
) -> folium.Map:
    """
    Create the base Folium map with selected tile layer and standard controls.
    
    Parameters
    ----------
    bbox : dict
        Must have keys: center_lat, center_lon
    basemap : str
        One of TILE_PROVIDERS keys.
    zoom_start : int
        Initial zoom level.
    """
    tile_url  = TILE_PROVIDERS.get(basemap, TILE_PROVIDERS["CartoDB Dark"])
    tile_attr = TILE_ATTRS.get(basemap, "")

    m = folium.Map(
        location=[bbox["center_lat"], bbox["center_lon"]],
        zoom_start=zoom_start,
        tiles=tile_url,
        attr=tile_attr,
        control_scale=True,
        prefer_canvas=True,
    )

    # ── Overlay all tile options for layer switcher ──────────────────────────
    for name, url in TILE_PROVIDERS.items():
        if name == basemap:
            continue
        folium.TileLayer(
            tiles=url,
            attr=TILE_ATTRS.get(name, ""),
            name=name,
            overlay=False,
            control=True,
        ).add_to(m)

    # ── Standard map controls ────────────────────────────────────────────────
    Fullscreen(position="topright", title="Fullscreen", title_cancel="Exit").add_to(m)
    MiniMap(toggle_display=True, tile_layer="CartoDB dark_matter").add_to(m)
    MousePosition(
        position="bottomright",
        prefix="📍",
        lat_formatter="function(num) {return L.Util.formatNum(num, 5) + '° N';}",
        lng_formatter="function(num) {return L.Util.formatNum(num, 5) + '° E';}"
    ).add_to(m)
    MeasureControl(
        position="bottomleft",
        primary_length_unit="kilometers",
        secondary_length_unit="miles",
        primary_area_unit="sqkilometers",
    ).add_to(m)

    return m


# ─────────────────────────────────────────────────────────────────────────────
def add_flood_layer(m: folium.Map, flood_geojson: dict) -> folium.Map:
    """
    Add Sentinel-1 flood classification as GeoJSON polygon layer.
    
    Flooded cells shown in blue; non-flooded cells are transparent (not added).
    Hovering a cell shows its coordinates and classification.
    """
    fg = folium.FeatureGroup(name="🌊 Flood Detection (Sentinel-1)", show=True)

    def style_flood(feature):
        return {
            "fillColor":   FLOOD_FILL,
            "color":       FLOOD_COLOR,
            "weight":      0.4,
            "fillOpacity": 0.65,
        }

    def highlight_flood(feature):
        return {
            "fillColor": "#93c5fd",
            "color":     "#60a5fa",
            "weight":    1.5,
            "fillOpacity": 0.85,
        }

    geojson_layer = folium.GeoJson(
        flood_geojson,
        name="Flood Polygons",
        style_function=style_flood,
        highlight_function=highlight_flood,
        tooltip=folium.GeoJsonTooltip(
            fields=["class"],
            aliases=["Status:"],
            localize=True,
            sticky=True,
            labels=True,
            style=(
                "background-color:#1e2d40; color:#e2e8f0; "
                "font-family:'Space Mono',monospace; font-size:11px; "
                "border:1px solid #3b82f6; border-radius:4px; padding:4px;"
            ),
        ),
    )
    geojson_layer.add_to(fg)
    fg.add_to(m)
    return m


# ─────────────────────────────────────────────────────────────────────────────
def add_ndvi_heatmap_layer(m: folium.Map, ndvi_df: pd.DataFrame) -> folium.Map:
    """
    Add Sentinel-2 NDVI as a heat-map layer.
    Intensity = NDVI value (0→1).
    """
    fg = folium.FeatureGroup(name="🌿 NDVI Heatmap (Sentinel-2)", show=False)

    # Only use valid NDVI rows
    df = ndvi_df[ndvi_df["ndvi"].notna()].copy()

    heat_data = [
        [row["lat"], row["lon"], max(0.0, float(row["ndvi"]))]
        for _, row in df.iterrows()
    ]

    HeatMap(
        heat_data,
        min_opacity=0.3,
        max_zoom=14,
        radius=18,
        blur=20,
        gradient={
            "0.0":  "#7f1d1d",   # very low → dark red
            "0.2":  "#ef4444",   # low
            "0.35": "#f59e0b",   # moderate
            "0.5":  "#84cc16",   # moderate-good
            "0.7":  "#10b981",   # healthy
            "1.0":  "#064e3b",   # dense
        },
    ).add_to(fg)

    fg.add_to(m)
    return m


# ─────────────────────────────────────────────────────────────────────────────
def add_ndvi_classification_layer(m: folium.Map, ndvi_df: pd.DataFrame) -> folium.Map:
    """
    Add NDVI classification as colored circle markers.
    Each marker = one downsampled pixel; colored by class.
    """
    fg = folium.FeatureGroup(name="🌱 Vegetation Classification", show=False)

    # Sample down to max 600 points for performance
    df = ndvi_df.copy()
    if len(df) > 600:
        df = df.sample(600, random_state=42)

    for _, row in df.iterrows():
        cid   = int(row["class_id"])
        color = NDVI_COLORS.get(cid, "#6b7280")
        label = NDVI_LABELS.get(cid, "Unknown")
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=5,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.75,
            weight=0.5,
            tooltip=folium.Tooltip(
                f"<span style='font-family:monospace; font-size:11px;'>"
                f"<b>{label}</b><br>"
                f"NDVI: {row['ndvi']:.4f}<br>"
                f"📍 {row['lat']:.4f}°N, {row['lon']:.4f}°E"
                f"</span>",
                sticky=True
            )
        ).add_to(fg)

    fg.add_to(m)
    return m


# ─────────────────────────────────────────────────────────────────────────────
def add_risk_layer(m: folium.Map, risk_grid: np.ndarray, bbox: dict) -> folium.Map:
    """
    Add composite risk index as a semi-transparent heatmap layer.
    """
    fg = folium.FeatureGroup(name="⚠️ Risk Index (Composite)", show=False)

    rows, cols = risk_grid.shape
    ds   = 8
    rg   = risk_grid.reshape(rows//ds, ds, cols//ds, ds).mean(axis=(1,3))
    sr, sc = rg.shape
    lats = np.linspace(bbox["lat_max"], bbox["lat_min"], sr)
    lons = np.linspace(bbox["lon_min"], bbox["lon_max"], sc)

    heat_data = []
    for r in range(sr):
        for c in range(sc):
            v = float(rg[r, c])
            if v > 0.05:
                heat_data.append([float(lats[r]), float(lons[c]), v])

    HeatMap(
        heat_data,
        min_opacity=0.25,
        radius=20,
        blur=22,
        gradient={
            "0.0": "#14532d",
            "0.3": "#16a34a",
            "0.5": "#facc15",
            "0.75":"#f97316",
            "1.0": "#dc2626",
        },
    ).add_to(fg)
    fg.add_to(m)
    return m


# ─────────────────────────────────────────────────────────────────────────────
def add_sensor_layer(m: folium.Map, sensor_df: pd.DataFrame) -> folium.Map:
    """
    Add IoT/weather sensor locations as interactive markers.
    Click a marker to see sensor metadata.
    """
    fg = folium.FeatureGroup(name="📡 IoT Sensors (Phase 3 Preview)", show=True)

    for _, row in sensor_df.iterrows():
        icon_name, icon_color = SENSOR_ICONS.get(row["type"], ("circle", "blue"))

        bat_color = (
            "green" if row["battery_pct"] >= 70
            else "orange" if row["battery_pct"] >= 40
            else "red"
        )
        popup_html = f"""
        <div style="font-family:'Space Mono',monospace; font-size:11px;
                    background:#111827; color:#e2e8f0; padding:10px;
                    border-radius:8px; border:1px solid #374151; min-width:200px;">
          <b style="color:#06b6d4; font-size:12px;">{row['name']}</b><br>
          <span style="color:#9ca3af;">Type:</span> {row['type']}<br>
          <span style="color:#9ca3af;">Battery:</span>
            <span style="color:{'#10b981' if row['battery_pct']>=70 else '#f59e0b' if row['battery_pct']>=40 else '#ef4444'}">
              {row['battery_pct']}%
            </span><br>
          <span style="color:#9ca3af;">Signal:</span> {row['signal']}<br>
          <span style="color:#9ca3af;">Last ping:</span> {row['last_ping']}<br>
          <span style="color:#9ca3af;">Location:</span> {row['lat']}°N, {row['lon']}°E
        </div>
        """

        folium.Marker(
            location=[row["lat"], row["lon"]],
            popup=folium.Popup(popup_html, max_width=260),
            tooltip=folium.Tooltip(
                f"<b>{row['name']}</b> — {row['type']}",
                sticky=True
            ),
            icon=folium.Icon(
                icon=icon_name,
                prefix="fa",
                color=icon_color,
                icon_color="white"
            )
        ).add_to(fg)

    fg.add_to(m)
    return m


# ─────────────────────────────────────────────────────────────────────────────
def add_aoi_boundary(m: folium.Map, bbox: dict) -> folium.Map:
    """
    Add the study area boundary rectangle as a dashed overlay.
    """
    fg = folium.FeatureGroup(name="🗺️ Study Area Boundary (AOI)", show=True)

    folium.Rectangle(
        bounds=[
            [bbox["lat_min"], bbox["lon_min"]],
            [bbox["lat_max"], bbox["lon_max"]]
        ],
        color="#06b6d4",
        weight=2,
        fill=False,
        dash_array="8,4",
        tooltip=folium.Tooltip(
            f"<b style='font-family:monospace;color:#06b6d4;'>AOI: {bbox['label']}</b><br>"
            f"Lat: {bbox['lat_min']}° – {bbox['lat_max']}°N<br>"
            f"Lon: {bbox['lon_min']}° – {bbox['lon_max']}°E",
            sticky=True
        )
    ).add_to(fg)

    fg.add_to(m)
    return m


# ─────────────────────────────────────────────────────────────────────────────
def add_map_legend(m: folium.Map) -> folium.Map:
    """
    Inject a custom HTML legend into the map (bottom-left).
    """
    legend_html = """
    <div style="
        position: fixed; bottom: 50px; left: 10px; z-index: 1000;
        background: rgba(13,17,23,0.92); border: 1px solid #1e2d40;
        border-radius: 10px; padding: 12px 16px;
        font-family: 'Space Mono', monospace; font-size: 10px; color: #e2e8f0;
        box-shadow: 0 4px 20px rgba(0,0,0,0.5); backdrop-filter: blur(8px);
        min-width: 170px;">
      <div style="font-size:11px; font-weight:bold; color:#06b6d4;
                  margin-bottom:8px; letter-spacing:0.1em;">MAP LEGEND</div>

      <div style="margin-bottom:6px; font-weight:bold; color:#8b949e;">FLOOD</div>
      <div style="display:flex; align-items:center; gap:6px; margin-bottom:4px;">
        <div style="width:14px;height:14px;border-radius:2px;background:#3b82f6;opacity:0.85;"></div>
        Flooded Area
      </div>

      <div style="margin:6px 0 4px; font-weight:bold; color:#8b949e;">VEGETATION</div>
      <div style="display:flex; align-items:center; gap:6px; margin-bottom:4px;">
        <div style="width:14px;height:14px;border-radius:50%;background:#10b981;"></div>
        Healthy Veg.
      </div>
      <div style="display:flex; align-items:center; gap:6px; margin-bottom:4px;">
        <div style="width:14px;height:14px;border-radius:50%;background:#f59e0b;"></div>
        Moderate Veg.
      </div>
      <div style="display:flex; align-items:center; gap:6px; margin-bottom:4px;">
        <div style="width:14px;height:14px;border-radius:50%;background:#ef4444;"></div>
        Low / Stressed
      </div>

      <div style="margin:6px 0 4px; font-weight:bold; color:#8b949e;">SENSORS</div>
      <div style="display:flex; align-items:center; gap:6px; margin-bottom:4px;">
        <span>📍</span> IoT / Weather
      </div>

      <div style="margin-top:8px; padding-top:6px; border-top:1px solid #1e2d40;
                  color:#4b5563; font-size:9px; letter-spacing:0.08em;">
        GeoTwin Phase 2 · Sentinel-1/2
      </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    return m


# ─────────────────────────────────────────────────────────────────────────────
def build_full_gis_map(
    data: dict,
    basemap: str = "CartoDB Dark",
    show_flood: bool = True,
    show_ndvi_heat: bool = False,
    show_ndvi_class: bool = False,
    show_risk: bool = False,
    show_sensors: bool = True,
    show_aoi: bool = True,
    zoom_start: int = 8,
) -> folium.Map:
    """
    Master function: builds the complete GIS map from Phase 1 data.
    
    Assembles all layers in order and returns a ready-to-render Folium map.
    The Streamlit component renders this via st_folium or html.
    """
    bbox = data["bbox"]
    m    = build_base_map(bbox, basemap=basemap, zoom_start=zoom_start)

    if show_aoi:
        m = add_aoi_boundary(m, bbox)
    if show_flood:
        m = add_flood_layer(m, data["flood_geojson"])
    if show_ndvi_heat:
        m = add_ndvi_heatmap_layer(m, data["ndvi_df"])
    if show_ndvi_class:
        m = add_ndvi_classification_layer(m, data["ndvi_df"])
    if show_risk:
        m = add_risk_layer(m, data["risk_grid"], bbox)
    if show_sensors:
        m = add_sensor_layer(m, data["sensor_points"])

    m = add_map_legend(m)
    folium.LayerControl(position="topright", collapsed=False).add_to(m)
    return m


# ─────────────────────────────────────────────────────────────────────────────
def build_before_after_map(data: dict) -> folium.Map:
    """
    Side-by-side before/after SAR comparison using Folium's SideBySideLayers.
    Falls back to a standard map with two layers if plugin unavailable.
    """
    bbox  = data["bbox"]
    m     = build_base_map(bbox, basemap="CartoDB Dark", zoom_start=8)

    # Add both flood states as separate feature groups
    fg_before = folium.FeatureGroup(name="📅 Before Flood (Pre-Event SAR)", show=True)
    fg_after  = folium.FeatureGroup(name="🌊 After Flood (Post-Event SAR)", show=True)

    # Before: no flood polygon (just AOI)
    folium.Rectangle(
        bounds=[[bbox["lat_min"], bbox["lon_min"]], [bbox["lat_max"], bbox["lon_max"]]],
        color="#16a34a", weight=2, fill=True, fill_color="#16a34a",
        fill_opacity=0.05,
        tooltip="Pre-flood baseline — Normal land cover"
    ).add_to(fg_before)

    # After: flood polygons
    n_features = len(data["flood_geojson"]["features"])
    for feat in data["flood_geojson"]["features"]:
        coords = feat["geometry"]["coordinates"][0]
        lats = [p[1] for p in coords]
        lons = [p[0] for p in coords]
        folium.Polygon(
            locations=list(zip(lats, lons)),
            color=FLOOD_COLOR,
            fill=True,
            fill_color=FLOOD_FILL,
            fill_opacity=0.7,
            weight=0.5,
        ).add_to(fg_after)

    fg_before.add_to(m)
    fg_after.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)

    # Annotation
    info_html = f"""
    <div style="position:fixed; top:60px; left:10px; z-index:1000;
                background:rgba(13,17,23,0.9); border:1px solid #1e2d40;
                border-radius:8px; padding:10px 14px;
                font-family:'Space Mono',monospace; font-size:10px; color:#e2e8f0;">
      <b style="color:#06b6d4;">BEFORE / AFTER COMPARISON</b><br>
      <span style="color:#10b981;">■</span> Pre-flood baseline<br>
      <span style="color:#3b82f6;">■</span> Post-flood ({n_features} flooded cells)<br>
      <span style="color:#8b949e; font-size:9px;">Sentinel-1 SAR VV · 10m resolution</span>
    </div>
    """
    m.get_root().html.add_child(folium.Element(info_html))
    return m


# ─────────────────────────────────────────────────────────────────────────────
def map_to_html_string(m: folium.Map) -> str:
    """Render a Folium map to an HTML string for st.components.v1.html()."""
    return m._repr_html_()
