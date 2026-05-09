"""
================================================================================
  dashboard/components/ai_ui_components.py
  Phase 3 AI/ML UI Components
  
  Extends Phase 2 ui_components.py with AI-specific visual elements:
    - AI model result cards
    - Training progress indicator
    - Model comparison banners
    - Kappa / accuracy badge rows
    - AI section headers
================================================================================
"""


# ─────────────────────────────────────────────────────────────────────────────
#   EXTRA CSS (appended on top of Phase 2 GLOBAL_CSS)
# ─────────────────────────────────────────────────────────────────────────────

AI_EXTRA_CSS = """
<style>
/* ── AI Model Cards ── */
.ai-model-card {
  background: #111827;
  border: 1px solid #1e2d40;
  border-radius: 10px;
  padding: 1rem 1.1rem;
  position: relative;
  overflow: hidden;
  transition: border-color 0.2s, box-shadow 0.2s;
}
.ai-model-card:hover {
  border-color: #06b6d4;
  box-shadow: 0 0 16px rgba(6,182,212,0.12);
}
.ai-model-card::before {
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 2px;
}
.ai-model-card.rf::before  { background: linear-gradient(90deg, #06b6d4, transparent); }
.ai-model-card.svm::before { background: linear-gradient(90deg, #8b5cf6, transparent); }
.ai-model-card.best::before{ background: linear-gradient(90deg, #10b981, transparent); }

/* ── Training status badges ── */
.train-badge {
  display: inline-flex; align-items: center; gap: 5px;
  padding: 0.2rem 0.6rem;
  border-radius: 4px;
  font-family: 'Space Mono', monospace;
  font-size: 0.62rem;
  letter-spacing: 0.1em;
}
.train-badge.done {
  background: rgba(16,185,129,0.1);
  border: 1px solid rgba(16,185,129,0.3);
  color: #10b981;
}
.train-badge.best {
  background: rgba(245,158,11,0.1);
  border: 1px solid rgba(245,158,11,0.3);
  color: #f59e0b;
}

/* ── Metric pill row ── */
.metric-pill-row {
  display: flex; gap: 0.4rem; flex-wrap: wrap; margin-top: 0.6rem;
}
.metric-pill {
  display: inline-block;
  padding: 0.15rem 0.5rem;
  border-radius: 20px;
  font-family: 'Space Mono', monospace;
  font-size: 0.6rem;
  letter-spacing: 0.06em;
}

/* ── AI Section header ── */
.ai-section-header {
  background: linear-gradient(135deg, #080d14, #0a1220);
  border: 1px solid #1e2d40;
  border-left: 3px solid #8b5cf6;
  border-radius: 8px;
  padding: 0.7rem 1rem;
  margin-bottom: 1rem;
  display: flex; align-items: center; gap: 0.7rem;
}
</style>
"""


# ─────────────────────────────────────────────────────────────────────────────
#   AI PAGE HERO
# ─────────────────────────────────────────────────────────────────────────────

def ai_hero_banner() -> str:
    return """
<div style="
  background: linear-gradient(135deg, #0a0d18 0%, #0d1428 50%, #080f1e 100%);
  border: 1px solid #1e2d40; border-top: 2px solid #8b5cf6;
  border-radius: 12px; padding: 1.4rem 2rem; margin-bottom: 1rem;
  position: relative; overflow: hidden;">

  <div style="position:absolute; top:-30px; right:-30px; width:200px; height:200px;
              background:radial-gradient(circle, rgba(139,92,246,0.07) 0%, transparent 70%);
              pointer-events:none;"></div>

  <div style="font-family:Orbitron,monospace; font-size:0.6rem; color:#484f58;
              letter-spacing:0.25em; text-transform:uppercase; margin-bottom:0.4rem;">
    🤖 GEOTWIN · PHASE 3 · AI / ML INTELLIGENCE
  </div>
  <div style="font-family:Orbitron,monospace; font-size:1.05rem; font-weight:700;
              color:#8b5cf6; letter-spacing:0.04em;
              text-shadow:0 0 16px rgba(139,92,246,0.35); margin-bottom:0.25rem;">
    AI-Driven Multi-Hazard Classification
  </div>
  <div style="font-family:Rajdhani,sans-serif; font-size:0.82rem; color:#8b949e;
              letter-spacing:0.06em;">
    Random Forest · SVM · Feature Engineering · Cross-Validation · Confusion Matrix
  </div>
  <div style="display:flex; gap:0.4rem; flex-wrap:wrap; margin-top:0.8rem;">
    <span style="font-family:Space Mono,monospace; font-size:0.6rem;
                 padding:0.15rem 0.55rem; border-radius:4px;
                 background:rgba(139,92,246,0.12); border:1px solid rgba(139,92,246,0.35);
                 color:#8b5cf6;">SCIKIT-LEARN</span>
    <span style="font-family:Space Mono,monospace; font-size:0.6rem;
                 padding:0.15rem 0.55rem; border-radius:4px;
                 background:rgba(6,182,212,0.12); border:1px solid rgba(6,182,212,0.35);
                 color:#06b6d4;">RANDOM FOREST</span>
    <span style="font-family:Space Mono,monospace; font-size:0.6rem;
                 padding:0.15rem 0.55rem; border-radius:4px;
                 background:rgba(59,130,246,0.12); border:1px solid rgba(59,130,246,0.35);
                 color:#3b82f6;">SVM · RBF KERNEL</span>
    <span style="font-family:Space Mono,monospace; font-size:0.6rem;
                 padding:0.15rem 0.55rem; border-radius:4px;
                 background:rgba(16,185,129,0.12); border:1px solid rgba(16,185,129,0.35);
                 color:#10b981;">PHASE 1 FEATURES</span>
    <span style="font-family:Space Mono,monospace; font-size:0.6rem;
                 padding:0.15rem 0.55rem; border-radius:4px;
                 background:rgba(245,158,11,0.12); border:1px solid rgba(245,158,11,0.35);
                 color:#f59e0b;">PHASE 2 DASHBOARD</span>
  </div>
</div>"""


# ─────────────────────────────────────────────────────────────────────────────
#   MODEL RESULT CARD
# ─────────────────────────────────────────────────────────────────────────────

def model_result_card(
    model_name: str,
    model_type: str,
    accuracy: float,
    f1: float,
    precision: float,
    recall: float,
    cv_mean: float,
    kappa: float,
    n_train: int,
    is_best: bool = False,
    extra_info: str = ""
) -> str:
    """
    Compact model result card showing all key metrics.
    """
    card_class = "best" if is_best else ("rf" if "Forest" in model_name else "svm")
    accent     = "#10b981" if is_best else ("#06b6d4" if "Forest" in model_name else "#8b5cf6")
    best_badge = (
        '<span class="train-badge best">⭐ BEST MODEL</span>'
        if is_best else
        '<span class="train-badge done">✓ TRAINED</span>'
    )
    model_icon = "🌲" if "Forest" in model_name else "⚙️"
    cv_str     = f"{cv_mean:.4f}" if cv_mean is not None else "N/A"

    return f"""
<div class="ai-model-card {card_class}">
  <div style="display:flex; justify-content:space-between; align-items:flex-start;
              margin-bottom:0.6rem;">
    <div>
      <div style="font-family:Orbitron,monospace; font-size:0.78rem; font-weight:700;
                  color:{accent}; letter-spacing:0.08em;">
        {model_icon} {model_name}
      </div>
      <div style="font-family:Space Mono,monospace; font-size:0.6rem; color:#484f58;
                  margin-top:2px;">
        {model_type.upper()} CLASSIFIER · {n_train:,} samples
      </div>
    </div>
    {best_badge}
  </div>

  <div style="display:grid; grid-template-columns:1fr 1fr; gap:0.4rem;">
    <div>
      <div style="font-family:Space Mono,monospace; font-size:0.58rem;
                  color:#484f58; text-transform:uppercase; letter-spacing:0.1em;">
        Accuracy
      </div>
      <div style="font-family:Orbitron,monospace; font-size:1.2rem;
                  font-weight:700; color:{accent}; line-height:1.1;">
        {accuracy*100:.1f}<span style="font-size:0.65rem;">%</span>
      </div>
    </div>
    <div>
      <div style="font-family:Space Mono,monospace; font-size:0.58rem;
                  color:#484f58; text-transform:uppercase; letter-spacing:0.1em;">
        F1-Score
      </div>
      <div style="font-family:Orbitron,monospace; font-size:1.2rem;
                  font-weight:700; color:{accent}; line-height:1.1;">
        {f1:.4f}
      </div>
    </div>
  </div>

  <div class="metric-pill-row">
    <span class="metric-pill"
          style="background:rgba(6,182,212,0.1);border:1px solid rgba(6,182,212,0.25);
                 color:#06b6d4;">
      Prec: {precision:.3f}
    </span>
    <span class="metric-pill"
          style="background:rgba(16,185,129,0.1);border:1px solid rgba(16,185,129,0.25);
                 color:#10b981;">
      Rec: {recall:.3f}
    </span>
    <span class="metric-pill"
          style="background:rgba(245,158,11,0.1);border:1px solid rgba(245,158,11,0.25);
                 color:#f59e0b;">
      CV: {cv_str}
    </span>
    <span class="metric-pill"
          style="background:rgba(139,92,246,0.1);border:1px solid rgba(139,92,246,0.25);
                 color:#8b5cf6;">
      κ: {kappa:.3f}
    </span>
  </div>
  {'<div style="font-family:Space Mono,monospace; font-size:0.6rem; color:#484f58; margin-top:0.5rem;">' + extra_info + '</div>' if extra_info else ''}
</div>"""


# ─────────────────────────────────────────────────────────────────────────────
#   AI KPI ROW
# ─────────────────────────────────────────────────────────────────────────────

def ai_kpi_row(summary: dict) -> str:
    """4-column KPI row for the AI results overview."""
    best_f = max(summary.get("rf_flood_acc", 0), summary.get("svm_flood_acc", 0))
    best_n = max(summary.get("rf_ndvi_acc",  0), summary.get("svm_ndvi_acc",  0))
    fa     = summary.get("flood_agreement", 0)
    na     = summary.get("ndvi_agreement",  0)

    cards = [
        ("FLOOD RF ACC",   f"{summary.get('rf_flood_acc',0)*100:.1f}", "%",    "#3b82f6", "Random Forest · SAR VV"),
        ("FLOOD SVM ACC",  f"{summary.get('svm_flood_acc',0)*100:.1f}","%",   "#8b5cf6", "SVM RBF · SAR features"),
        ("NDVI RF ACC",    f"{summary.get('rf_ndvi_acc',0)*100:.1f}",  "%",   "#10b981", "Random Forest · NDVI"),
        ("NDVI SVM ACC",   f"{summary.get('svm_ndvi_acc',0)*100:.1f}", "%",   "#06b6d4", "SVM RBF · spectral"),
        ("FLOOD AGREE.",   f"{fa:.1f}",    "%",   "#f59e0b", "RF ≡ SVM prediction"),
        ("NDVI AGREE.",    f"{na:.1f}",    "%",   "#f59e0b", "RF ≡ SVM prediction"),
        ("BEST FLOOD F1",  f"{summary.get('rf_flood_f1',0):.4f}", "", "#3b82f6", summary.get("best_flood", "")),
        ("BEST NDVI F1",   f"{summary.get('rf_ndvi_f1',0):.4f}",  "", "#10b981", summary.get("best_ndvi",  "")),
    ]

    cols_html = ""
    for label, val, unit, color, sub in cards:
        cols_html += f"""
        <div style="background:#111827; border:1px solid #1e2d40; border-radius:8px;
                    padding:0.7rem 0.8rem; position:relative; overflow:hidden;
                    flex:1; min-width:110px;">
          <div style="position:absolute; top:0; left:0; right:0; height:2px;
                      background:linear-gradient(90deg,{color},transparent);"></div>
          <div style="font-family:Space Mono,monospace; font-size:0.55rem; color:#8b949e;
                      letter-spacing:0.1em; text-transform:uppercase; margin-bottom:0.25rem;">
            {label}
          </div>
          <div style="font-family:Orbitron,monospace; font-size:1.1rem; font-weight:700;
                      color:{color}; line-height:1.1;">
            {val}<span style="font-size:0.65rem; margin-left:1px;">{unit}</span>
          </div>
          <div style="font-family:Space Mono,monospace; font-size:0.55rem; color:#484f58;
                      margin-top:0.2rem;">
            {sub}
          </div>
        </div>"""

    return f"""
<div style="display:flex; gap:0.5rem; margin-bottom:1rem; flex-wrap:wrap;">
  {cols_html}
</div>"""


# ─────────────────────────────────────────────────────────────────────────────
#   AI SECTION HEADER
# ─────────────────────────────────────────────────────────────────────────────

def ai_section_header(icon: str, title: str, subtitle: str = "") -> str:
    sub = (f'<div style="font-family:Space Mono,monospace; font-size:0.62rem; '
           f'color:#484f58; margin-top:2px;">{subtitle}</div>') if subtitle else ""
    return f"""
<div class="ai-section-header">
  <span style="font-size:1.1rem;">{icon}</span>
  <div>
    <div style="font-family:Orbitron,monospace; font-size:0.82rem; font-weight:700;
                color:#8b5cf6; letter-spacing:0.08em; text-transform:uppercase;">
      {title}
    </div>
    {sub}
  </div>
</div>"""


# ─────────────────────────────────────────────────────────────────────────────
#   METHODOLOGY CARD
# ─────────────────────────────────────────────────────────────────────────────

def methodology_card(model_name: str, model_type: str, details: dict) -> str:
    """
    Compact methodology reference card for the AI page sidebar.
    """
    icon   = "🌲" if "Forest" in model_name else "⚙️"
    color  = "#06b6d4" if "Forest" in model_name else "#8b5cf6"
    rows   = "".join([
        f'<div style="display:flex; justify-content:space-between; padding:0.2rem 0;'
        f'           border-bottom:1px solid #1e2d40;">'
        f'  <span style="font-family:Space Mono,monospace; font-size:0.6rem;'
        f'               color:#8b949e;">{k}</span>'
        f'  <span style="font-family:Space Mono,monospace; font-size:0.6rem;'
        f'               color:{color};">{v}</span>'
        f'</div>'
        for k, v in details.items()
    ])
    return f"""
<div style="background:#111827; border:1px solid #1e2d40; border-left:2px solid {color};
            border-radius:8px; padding:0.8rem; margin-bottom:0.6rem;">
  <div style="font-family:Orbitron,monospace; font-size:0.7rem; color:{color};
              margin-bottom:0.5rem;">{icon} {model_name}</div>
  {rows}
</div>"""
