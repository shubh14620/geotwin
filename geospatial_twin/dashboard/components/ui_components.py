"""
================================================================================
  dashboard/components/ui_components.py
  Reusable HTML/CSS UI Components for Phase 2 GIS Dashboard
  
  Provides pre-styled HTML blocks injected via st.markdown(unsafe_allow_html=True).
  Keeps app.py clean — all visual chrome lives here.
================================================================================
"""

from datetime import datetime


# ─────────────────────────────────────────────────────────────────────────────
#   GLOBAL CSS INJECTION
# ─────────────────────────────────────────────────────────────────────────────

GLOBAL_CSS = """
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@300;400;500;600;700&family=Space+Mono:wght@400;700&display=swap');

/* ── CSS Variables ── */
:root {
  --bg-deep:      #030712;
  --bg-main:      #0d1117;
  --bg-card:      #111827;
  --bg-card2:     #161d2e;
  --border:       #1e2d40;
  --border-glow:  #0ea5e9;
  --cyan:         #06b6d4;
  --blue:         #3b82f6;
  --green:        #10b981;
  --amber:        #f59e0b;
  --red:          #ef4444;
  --violet:       #8b5cf6;
  --text:         #f0f6fc;
  --text-muted:   #8b949e;
  --text-dim:     #484f58;
  --font-hd:      'Orbitron', monospace;
  --font-ui:      'Rajdhani', sans-serif;
  --font-mono:    'Space Mono', monospace;
}

/* ── Global reset ── */
html, body, [class*="css"] {
  background-color: var(--bg-deep) !important;
  color: var(--text) !important;
  font-family: var(--font-ui) !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #080d14 0%, #030712 100%) !important;
  border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text) !important; }
[data-testid="stSidebar"] .stRadio label,
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stCheckbox label,
[data-testid="stSidebar"] .stSlider label {
  font-family: var(--font-mono) !important;
  font-size: 0.7rem !important;
  letter-spacing: 0.08em;
  color: var(--text-muted) !important;
  text-transform: uppercase;
}

/* ── Buttons ── */
.stButton > button {
  background: linear-gradient(135deg, rgba(6,182,212,0.08), rgba(59,130,246,0.08)) !important;
  border: 1px solid var(--cyan) !important;
  color: var(--cyan) !important;
  font-family: var(--font-mono) !important;
  font-size: 0.72rem !important;
  letter-spacing: 0.12em !important;
  border-radius: 6px !important;
  transition: all 0.2s ease !important;
}
.stButton > button:hover {
  background: linear-gradient(135deg, rgba(6,182,212,0.18), rgba(59,130,246,0.15)) !important;
  box-shadow: 0 0 14px rgba(6,182,212,0.35) !important;
  transform: translateY(-1px) !important;
}
.stDownloadButton > button {
  background: linear-gradient(135deg, rgba(16,185,129,0.08), rgba(6,182,212,0.06)) !important;
  border: 1px solid var(--green) !important;
  color: var(--green) !important;
  font-family: var(--font-mono) !important;
  font-size: 0.72rem !important;
  border-radius: 6px !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
  background: var(--bg-main) !important;
  gap: 3px; border-bottom: 1px solid var(--border);
}
.stTabs [data-baseweb="tab"] {
  background: var(--bg-card) !important;
  color: var(--text-muted) !important;
  font-family: var(--font-mono) !important;
  font-size: 0.68rem !important;
  letter-spacing: 0.1em !important;
  border-radius: 6px 6px 0 0 !important;
  border: 1px solid transparent !important;
  transition: all 0.15s;
}
.stTabs [aria-selected="true"] {
  background: var(--bg-card2) !important;
  color: var(--cyan) !important;
  border: 1px solid var(--border) !important;
  border-bottom: 2px solid var(--cyan) !important;
}

/* ── Expanders ── */
div[data-testid="stExpander"] {
  background: var(--bg-card) !important;
  border: 1px solid var(--border) !important;
  border-radius: 8px !important;
}
div[data-testid="stExpander"] summary {
  font-family: var(--font-mono) !important;
  font-size: 0.72rem !important;
  color: var(--text-muted) !important;
  letter-spacing: 0.08em !important;
}

/* ── Metrics ── */
[data-testid="stMetric"] {
  background: var(--bg-card) !important;
  border: 1px solid var(--border) !important;
  border-radius: 10px !important;
  padding: 0.8rem 1rem !important;
}
[data-testid="stMetricLabel"] {
  font-family: var(--font-mono) !important;
  font-size: 0.65rem !important;
  color: var(--text-muted) !important;
  letter-spacing: 0.1em !important;
  text-transform: uppercase !important;
}
[data-testid="stMetricValue"] {
  font-family: var(--font-hd) !important;
  color: var(--cyan) !important;
}

/* ── DataFrames ── */
[data-testid="stDataFrame"] {
  border: 1px solid var(--border) !important;
  border-radius: 8px !important;
}

/* ── Selectbox / Slider / Checkbox ── */
.stSelectbox > div, .stMultiSelect > div {
  background: var(--bg-card) !important;
  border-color: var(--border) !important;
  border-radius: 6px !important;
}
.stSlider [data-baseweb="slider"] div[role="slider"] {
  background: var(--cyan) !important;
}
.stCheckbox label span {
  font-family: var(--font-mono) !important;
  font-size: 0.72rem !important;
  letter-spacing: 0.06em;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg-main); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--cyan); }

/* ── Dividers ── */
hr { border-color: var(--border) !important; }

/* ── Map iframe wrapper ── */
.map-wrapper {
  border: 1px solid var(--border);
  border-radius: 10px;
  overflow: hidden;
  box-shadow: 0 4px 24px rgba(0,0,0,0.5);
}

/* ── Pulse animation ── */
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.35} }
@keyframes scanline {
  0%{transform:translateY(-100%)} 100%{transform:translateY(100%)}
}
</style>
"""


# ─────────────────────────────────────────────────────────────────────────────
#   HERO HEADER
# ─────────────────────────────────────────────────────────────────────────────

def hero_header(active_page: str = "") -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M UTC")
    return f"""
<div style="
  background: linear-gradient(135deg, #080d14 0%, #0a1525 60%, #060f1e 100%);
  border: 1px solid #1e2d40; border-top: 2px solid #06b6d4;
  border-radius: 12px; padding: 1.5rem 2rem; margin-bottom: 1rem;
  position: relative; overflow: hidden;">

  <!-- Ambient glow orbs -->
  <div style="position:absolute; top:-40px; left:-40px; width:200px; height:200px;
              background:radial-gradient(circle, rgba(6,182,212,0.07) 0%, transparent 70%);
              pointer-events:none;"></div>
  <div style="position:absolute; bottom:-60px; right:-20px; width:250px; height:200px;
              background:radial-gradient(circle, rgba(59,130,246,0.05) 0%, transparent 70%);
              pointer-events:none;"></div>

  <div style="display:flex; justify-content:space-between; align-items:flex-start; flex-wrap:wrap; gap:1rem;">
    <div>
      <div style="font-family:'Orbitron',monospace; font-size:0.65rem; color:#484f58;
                  letter-spacing:0.25em; text-transform:uppercase; margin-bottom:0.4rem;">
        🛰️ GEOTWIN · PHASE 2
      </div>
      <div style="font-family:'Orbitron',monospace; font-size:1.1rem; font-weight:700;
                  color:#06b6d4; letter-spacing:0.04em; text-shadow:0 0 18px rgba(6,182,212,0.35);
                  line-height:1.3; margin-bottom:0.3rem;">
        AI-Driven Geospatial Digital Twin
      </div>
      <div style="font-family:'Rajdhani',sans-serif; font-size:0.85rem; color:#8b949e;
                  letter-spacing:0.06em;">
        Multi-Hazard Environmental Intelligence · IoT · SAR · GIS
      </div>
      <div style="display:flex; gap:0.4rem; flex-wrap:wrap; margin-top:0.8rem;">
        {''.join([
            f'<span style="font-family:Space Mono,monospace; font-size:0.6rem; letter-spacing:0.12em;'
            f'padding:0.15rem 0.55rem; border-radius:4px; {style}">{label}</span>'
            for label, style in [
                ("SENTINEL-1 SAR",    "background:rgba(59,130,246,0.12);border:1px solid rgba(59,130,246,0.35);color:#3b82f6;"),
                ("SENTINEL-2 MSI",    "background:rgba(16,185,129,0.12);border:1px solid rgba(16,185,129,0.35);color:#10b981;"),
                ("FOLIUM GIS",        "background:rgba(6,182,212,0.12);border:1px solid rgba(6,182,212,0.35);color:#06b6d4;"),
                ("QGIS COMPATIBLE",   "background:rgba(139,92,246,0.12);border:1px solid rgba(139,92,246,0.35);color:#8b5cf6;"),
                ("PHASE 2 DASHBOARD", "background:rgba(245,158,11,0.12);border:1px solid rgba(245,158,11,0.35);color:#f59e0b;"),
            ]
        ])}
      </div>
    </div>
    <div style="text-align:right;">
      <div style="font-family:Space Mono,monospace; font-size:0.6rem; color:#484f58;
                  letter-spacing:0.1em; line-height:2.2;">
        <div><span style="color:#10b981;">●</span> SYSTEM ONLINE</div>
        <div>GEE API: CONNECTED</div>
        <div>{now}</div>
        <div style="color:#06b6d4;">B.Tech Major Project · 2024-25</div>
      </div>
    </div>
  </div>
</div>
"""


# ─────────────────────────────────────────────────────────────────────────────
#   KPI METRIC CARDS
# ─────────────────────────────────────────────────────────────────────────────

def kpi_row(metrics: dict) -> str:
    """Render a 5-column KPI metrics row."""
    risk = metrics["risk_score"]
    risk_color = "#10b981" if risk < 25 else ("#f59e0b" if risk < 55 else "#ef4444")
    risk_label = "LOW" if risk < 25 else ("MOD." if risk < 55 else "HIGH")

    cards = [
        ("FLOODED AREA",      f"{metrics['flooded_pct']:.1f}",   "%",     "cyan",  "Sentinel-1 VV SAR"),
        ("FLOODED KM²",       f"{metrics['flooded_km2']:.0f}",   "km²",   "blue",  "Est. @ 10m res"),
        ("MEAN NDVI",         f"{metrics['ndvi_mean']:.4f}",     "",      "green", "Sentinel-2 B8/B4"),
        ("HEALTHY VEG.",      f"{metrics['healthy_pct']:.1f}",   "%",     "green", f"NDVI ≥ threshold"),
        ("RISK INDEX",        f"{risk:.0f}",                     "/ 100", risk_color.replace("#","") if len(risk_color)==7 else "amber", f"{risk_label} — composite"),
    ]
    # Map color name to CSS hex
    color_map = {
        "cyan":  ("#06b6d4", "rgba(6,182,212,0.08)",  "rgba(6,182,212,0.7)"),
        "blue":  ("#3b82f6", "rgba(59,130,246,0.08)", "rgba(59,130,246,0.7)"),
        "green": ("#10b981", "rgba(16,185,129,0.08)", "rgba(16,185,129,0.7)"),
        "amber": ("#f59e0b", "rgba(245,158,11,0.08)", "rgba(245,158,11,0.7)"),
        risk_color.replace("#","") if len(risk_color)==7 else "red":
                 (risk_color, f"rgba(239,68,68,0.08)", "rgba(239,68,68,0.7)"),
    }

    cols_html = ""
    for label, val, unit, color_key, sub in cards:
        hex_c, bg_c, bar_c = color_map.get(color_key, ("#06b6d4","rgba(6,182,212,0.08)","rgba(6,182,212,0.7)"))
        cols_html += f"""
        <div style="background:#111827; border:1px solid #1e2d40; border-radius:10px;
                    padding:0.9rem 1rem; position:relative; overflow:hidden; flex:1; min-width:130px;">
          <div style="position:absolute; top:0; left:0; right:0; height:2px;
                      background:linear-gradient(90deg,{bar_c},transparent);"></div>
          <div style="font-family:Space Mono,monospace; font-size:0.6rem; color:#8b949e;
                      letter-spacing:0.12em; text-transform:uppercase; margin-bottom:0.35rem;">
            {label}
          </div>
          <div style="font-family:Orbitron,monospace; font-size:1.45rem; font-weight:700;
                      color:{hex_c}; line-height:1.1;">
            {val}<span style="font-size:0.75rem; margin-left:2px;">{unit}</span>
          </div>
          <div style="font-family:Space Mono,monospace; font-size:0.58rem; color:#484f58;
                      margin-top:0.3rem; letter-spacing:0.05em;">
            {sub}
          </div>
        </div>"""

    return f"""
<div style="display:flex; gap:0.6rem; margin-bottom:1rem; flex-wrap:wrap;">
  {cols_html}
</div>"""


# ─────────────────────────────────────────────────────────────────────────────
#   SECTION HEADERS
# ─────────────────────────────────────────────────────────────────────────────

def section_header(icon: str, title: str, subtitle: str = "", color: str = "#06b6d4") -> str:
    sub_html = f'<div style="font-family:Space Mono,monospace; font-size:0.65rem; color:#484f58; margin-top:2px;">{subtitle}</div>' if subtitle else ""
    return f"""
<div style="display:flex; align-items:center; gap:0.8rem; border-bottom:1px solid #1e2d40;
            padding-bottom:0.7rem; margin-bottom:1rem;">
  <div style="width:38px; height:38px; border-radius:9px; display:flex; align-items:center;
              justify-content:center; font-size:1.1rem;
              background:linear-gradient(135deg,{color}18,{color}08);
              border:1px solid {color}40;">
    {icon}
  </div>
  <div>
    <div style="font-family:Orbitron,monospace; font-size:0.85rem; font-weight:700;
                color:{color}; letter-spacing:0.08em; text-transform:uppercase;">
      {title}
    </div>
    {sub_html}
  </div>
</div>"""


# ─────────────────────────────────────────────────────────────────────────────
#   STATUS BANNER
# ─────────────────────────────────────────────────────────────────────────────

def status_banner(
    mode: str,
    date_start: str,
    date_end: str,
    n_sensors: int,
    phase1_ok: bool = True
) -> str:
    dot = "●" if phase1_ok else "○"
    color = "#10b981" if phase1_ok else "#f59e0b"
    p1_label = "Phase 1 Modules: LOADED" if phase1_ok else "Phase 1 Modules: DEMO"
    return f"""
<div style="display:flex; align-items:center; gap:0.6rem; flex-wrap:wrap;
            background:rgba(16,185,129,0.06); border:1px solid rgba(16,185,129,0.18);
            border-radius:7px; padding:0.45rem 0.9rem; margin-bottom:1rem;
            font-family:Space Mono,monospace; font-size:0.68rem; color:{color};">
  <span style="animation:pulse 2s infinite;">{dot}</span>
  <span>{p1_label}</span>
  <span style="color:#1e2d40;">│</span>
  <span style="color:#484f58;">Mode: {mode}</span>
  <span style="color:#1e2d40;">│</span>
  <span style="color:#484f58;">Period: {date_start} → {date_end}</span>
  <span style="color:#1e2d40;">│</span>
  <span style="color:#484f58;">{n_sensors} IoT Sensors</span>
  <span style="color:#1e2d40;">│</span>
  <span style="color:#484f58;">GIS Engine: Folium + Plotly</span>
</div>"""


# ─────────────────────────────────────────────────────────────────────────────
#   MAP SECTION WRAPPER
# ─────────────────────────────────────────────────────────────────────────────

def map_container_open(title: str, subtitle: str = "") -> str:
    sub = f'<span style="font-size:0.65rem; color:#484f58; letter-spacing:0.05em; margin-left:0.5rem;">{subtitle}</span>' if subtitle else ""
    return f"""
<div class="map-wrapper">
  <div style="background:#080d14; padding:0.5rem 1rem; border-bottom:1px solid #1e2d40;
              display:flex; align-items:center; justify-content:space-between;">
    <span style="font-family:Space Mono,monospace; font-size:0.72rem; color:#06b6d4;
                 letter-spacing:0.12em;">{title}</span>
    {sub}
    <span style="font-family:Space Mono,monospace; font-size:0.6rem; color:#484f58;">
      🔵 LIVE · Folium/Leaflet.js
    </span>
  </div>
"""

def map_container_close() -> str:
    return "</div>"


# ─────────────────────────────────────────────────────────────────────────────
#   INFO CARDS
# ─────────────────────────────────────────────────────────────────────────────

def info_card(title: str, body: str, color: str = "#06b6d4") -> str:
    return f"""
<div style="background:#111827; border:1px solid #1e2d40; border-left:3px solid {color};
            border-radius:8px; padding:0.9rem 1.1rem; margin-bottom:0.7rem;">
  <div style="font-family:Orbitron,monospace; font-size:0.72rem; color:{color};
              letter-spacing:0.1em; margin-bottom:0.4rem; text-transform:uppercase;">
    {title}
  </div>
  <div style="font-family:Rajdhani,sans-serif; font-size:0.9rem; color:#8b949e; line-height:1.6;">
    {body}
  </div>
</div>"""


# ─────────────────────────────────────────────────────────────────────────────
#   SIDEBAR LOGO + PHASE BADGE
# ─────────────────────────────────────────────────────────────────────────────

def sidebar_logo() -> str:
    return """
<div style="text-align:center; padding:1.2rem 0 1.5rem;">
  <div style="font-family:Orbitron,monospace; font-size:0.8rem; color:#06b6d4;
              letter-spacing:0.2em; text-transform:uppercase; text-shadow:0 0 12px rgba(6,182,212,0.4);">
    🛰️ GeoTwin
  </div>
  <div style="font-family:Space Mono,monospace; font-size:0.55rem; color:#484f58;
              letter-spacing:0.12em; margin-top:0.25rem;">
    PHASE 2 · GIS DASHBOARD
  </div>
  <div style="margin-top:0.6rem; display:inline-block; padding:0.2rem 0.7rem;
              background:rgba(6,182,212,0.08); border:1px solid rgba(6,182,212,0.25);
              border-radius:4px; font-family:Space Mono,monospace; font-size:0.58rem;
              color:#06b6d4; letter-spacing:0.15em;">
    v2.0.0 · SENTINEL-1/2
  </div>
</div>"""


def sidebar_section_label(label: str) -> str:
    return f"""
<div style="font-family:Space Mono,monospace; font-size:0.6rem; color:#484f58;
            letter-spacing:0.2em; text-transform:uppercase; margin:0.8rem 0 0.4rem;
            border-bottom:1px solid #1e2d40; padding-bottom:0.3rem;">
  {label}
</div>"""
