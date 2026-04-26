"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  Living Attack Surface Mapper — Cybersecurity Intelligence Dashboard       ║
║  Author   : K Likhita Reddy — 16010423042 — TY IT A                       ║
║  Stack    : Streamlit + Plotly + Scikit-Learn + XGBoost                    ║
║  Run      : streamlit run app.py                                           ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import numpy as np
import time

from helpers import (
    load_models, load_dataset, load_timeseries, build_feature_vector,
    predict_risk, get_feature_importance, generate_insight,
    generate_explainability, plot_timeseries, plot_feature_importance,
    plot_risk_distribution, plot_anomaly_gauge,
    RISK_COLORS, RISK_BG, RISK_LABELS,
)

# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG & CUSTOM CSS
# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Living Attack Surface Mapper",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;600&display=swap');

/* ── Global ── */
html, body, [class*="css"] { font-family: 'Inter', sans-serif; color: #FFFFFF; }
.stApp { background: linear-gradient(145deg, #0B0F1A 0%, #111827 40%, #0F172A 100%); color: #FFFFFF; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0F172A 0%, #1E293B 100%);
    border-right: 1px solid rgba(99, 102, 241, 0.2);
}
section[data-testid="stSidebar"] .stSlider > div > div { background: rgba(99, 102, 241, 0.3); }

/* ── Cards ── */
div[data-testid="stMetric"] {
    background: rgba(30, 41, 59, 0.6);
    border: 1px solid rgba(99, 102, 241, 0.15);
    border-radius: 12px;
    padding: 16px 20px;
    backdrop-filter: blur(10px);
}
div[data-testid="stMetric"] label { color: #FFFFFF !important; font-size: 0.85rem !important; }
div[data-testid="stMetric"] [data-testid="stMetricValue"] { color: #FFFFFF !important; font-weight: 700 !important; }

/* ── Expander ── */
.streamlit-expanderHeader {
    background: rgba(30, 41, 59, 0.5) !important;
    border: 1px solid rgba(99, 102, 241, 0.15) !important;
    border-radius: 10px !important;
    color: #FFFFFF !important;
}

/* ── Headers ── */
h1, h2, h3 { color: #FFFFFF !important; }

/* ── Glass Card ── */
.glass-card {
    background: rgba(30, 41, 59, 0.5);
    border: 1px solid rgba(99, 102, 241, 0.15);
    border-radius: 16px;
    padding: 24px;
    backdrop-filter: blur(12px);
    margin-bottom: 16px;
}

/* ── Risk Badges ── */
.risk-badge {
    display: inline-block;
    padding: 8px 24px;
    border-radius: 50px;
    font-weight: 700;
    font-size: 1.1rem;
    letter-spacing: 1px;
    text-transform: uppercase;
    animation: pulse-glow 2s ease-in-out infinite;
}
@keyframes pulse-glow {
    0%, 100% { box-shadow: 0 0 15px rgba(99, 102, 241, 0.3); }
    50% { box-shadow: 0 0 30px rgba(99, 102, 241, 0.5); }
}

/* ── Alert Banner ── */
.alert-banner {
    background: linear-gradient(135deg, rgba(255, 23, 68, 0.15), rgba(213, 0, 249, 0.15));
    border: 1px solid rgba(255, 23, 68, 0.4);
    border-radius: 12px;
    padding: 16px 24px;
    color: #FFFFFF;
    font-weight: 600;
    animation: alert-flash 1.5s ease-in-out infinite;
}
@keyframes alert-flash {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.7; }
}

/* ── Insight Panel ── */
.insight-panel {
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.08), rgba(139, 92, 246, 0.08));
    border: 1px solid rgba(139, 92, 246, 0.25);
    border-radius: 14px;
    padding: 20px 24px;
    color: #FFFFFF;
    line-height: 1.7;
}

/* ── Section Divider ── */
.section-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(99, 102, 241, 0.3), transparent);
    margin: 24px 0;
}

/* ── Hero Title ── */
.hero-title {
    font-size: 2.2rem;
    font-weight: 800;
    background: linear-gradient(135deg, #818CF8, #C084FC, #F472B6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 4px;
}
.hero-sub {
    color: #FFFFFF;
    font-size: 0.95rem;
    font-weight: 400;
}

/* Hide default Streamlit elements */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

/* ── Button Styling ── */
.stButton > button {
    background: linear-gradient(135deg, #4F46E5, #7C3AED) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 10px 24px !important;
    font-weight: 600 !important;
    transition: all 0.3s ease !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(79, 70, 229, 0.4) !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab"] {
    color: #FFFFFF !important;
    font-weight: 500 !important;
}
.stTabs [aria-selected="true"] {
    color: #818CF8 !important;
    border-bottom-color: #818CF8 !important;
}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  LOAD RESOURCES (cached)
# ═══════════════════════════════════════════════════════════════════════════════
@st.cache_resource
def init_models():
    return load_models()

@st.cache_data
def init_dataset():
    return load_dataset()

@st.cache_data
def init_timeseries():
    return load_timeseries()

models = init_models()
dataset = init_dataset()
ts_data = init_timeseries()


# ═══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR — INPUT CONTROLS
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown('<div class="hero-title">🛡️ LASM</div>', unsafe_allow_html=True)
    st.markdown('<p class="hero-sub">Living Attack Surface Mapper</p>', unsafe_allow_html=True)
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    st.markdown("### 🎛️ Threat Parameters")

    leak_severity = st.slider("Leak Severity", 0.0, 10.0, 5.0, 0.5,
                              help="Severity of detected data leak (0=None, 10=Critical)")

    cvss_score = st.slider("CVSS Score", 0.0, 10.0, 5.0, 0.1,
                           help="Common Vulnerability Scoring System base score")

    source_type = st.selectbox("Source Type", ["GitHub", "DNS", "CVE", "Paste Site", "CT Logs"],
                               help="OSINT data source category")

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("### ⚙️ Advanced Settings")

    anomaly_threshold = st.slider("Anomaly Threshold", 0.01, 0.20, 0.05, 0.01,
                                  help="Isolation Forest contamination parameter")

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ── Attack Simulation ────────────────────────────────────────────────────
    st.markdown("### 🔴 Attack Simulation")
    attack_mode = st.button("⚡ Simulate Attack", use_container_width=True)
    if attack_mode:
        st.session_state["attack_mode"] = True
    if st.button("🔄 Reset Simulation", use_container_width=True):
        st.session_state["attack_mode"] = False

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # Model status indicators
    st.markdown("### 📡 Model Status")
    for name, key in [("Isolation Forest", "iso_forest"), ("Random Forest", "random_forest"), ("XGBoost", "xgboost")]:
        status = "🟢 Loaded" if models.get(key) else "🟡 Simulated"
        st.markdown(f"**{name}:** {status}")


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════
is_attack = st.session_state.get("attack_mode", False)

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">Cybersecurity Intelligence Dashboard</div>', unsafe_allow_html=True)
st.markdown('<p class="hero-sub">Real-time threat assessment powered by hybrid ML pipeline — Isolation Forest × Random Forest × XGBoost</p>', unsafe_allow_html=True)

# ── Attack Alert ─────────────────────────────────────────────────────────────
if is_attack:
    st.markdown("""
    <div class="alert-banner">
        🚨 ACTIVE ATTACK SIMULATION — Synthetic threat spike injected. 
        All metrics elevated to CRITICAL/EMERGENCY levels. Click "Reset Simulation" to restore normal state.
    </div>
    """, unsafe_allow_html=True)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# ── Run Prediction ───────────────────────────────────────────────────────────
with st.spinner("🔍 Analyzing threat vectors..."):
    if is_attack:
        # Override inputs for attack simulation
        atk_leak = 9.5
        atk_cvss = 9.8
        feature_vec, feats = build_feature_vector(atk_leak, atk_cvss, source_type, anomaly_threshold)
        risk_level, risk_score, anomaly_score, confidence = "Emergency", 4, 0.95, 0.92
    else:
        feature_vec, feats = build_feature_vector(leak_severity, cvss_score, source_type, anomaly_threshold)
        risk_level, risk_score, anomaly_score, confidence = predict_risk(models, feature_vec, anomaly_threshold)


# ═══════════════════════════════════════════════════════════════════════════════
#  ROW 1 — KEY METRICS
# ═══════════════════════════════════════════════════════════════════════════════
col1, col2, col3, col4 = st.columns(4)

risk_color = RISK_COLORS.get(risk_level, "#888")
risk_bg = RISK_BG.get(risk_level, "rgba(100,100,100,0.15)")

with col1:
    st.markdown(f"""
    <div class="glass-card" style="text-align:center; border-color: {risk_color}40;">
        <div style="color:#FFFFFF; font-size:0.8rem; text-transform:uppercase; letter-spacing:2px; margin-bottom:8px;">Risk Level</div>
        <div class="risk-badge" style="background:{risk_bg}; color:{risk_color}; border:2px solid {risk_color};">{risk_level}</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.metric("Risk Score", f"{risk_score} / 4", delta=f"{'⬆ Elevated' if risk_score >= 3 else '◆ Normal'}")

with col3:
    st.metric("Anomaly Score", f"{anomaly_score:.2%}", delta=f"{'⬆ Anomalous' if anomaly_score > 0.6 else '◆ Normal'}")

with col4:
    st.metric("Confidence", f"{confidence:.1%}", delta=f"{'✓ High' if confidence > 0.7 else '~ Moderate'}")


# ═══════════════════════════════════════════════════════════════════════════════
#  ROW 2 — GENAI INSIGHT + ANOMALY GAUGE
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

col_insight, col_gauge = st.columns([3, 1])

with col_insight:
    st.markdown("### 🧠 AI Threat Intelligence Report")
    insight_text = generate_insight(risk_level, risk_score, anomaly_score, confidence, feats)
    st.markdown(f'<div class="insight-panel">{insight_text}</div>', unsafe_allow_html=True)

with col_gauge:
    fig_gauge = plot_anomaly_gauge(anomaly_score)
    st.plotly_chart(fig_gauge, use_container_width=True, config={"displayModeBar": False})


# ═══════════════════════════════════════════════════════════════════════════════
#  ROW 3 — VISUALIZATIONS (Tabs)
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.markdown("### 📊 Analytics & Visualizations")

tab1, tab2, tab3 = st.tabs(["📈 Time-Series Analysis", "🏋️ Feature Importance", "🎯 Risk Distribution"])

with tab1:
    fig_ts = plot_timeseries(ts_data, attack_mode=is_attack)
    st.plotly_chart(fig_ts, use_container_width=True, config={"displayModeBar": True, "scrollZoom": True})
    if is_attack:
        st.error("⚡ Synthetic attack spike injected in the last 15 days of the timeline. Notice the sharp upward trend and triggered spike markers.")

with tab2:
    importance = get_feature_importance(models)
    fig_fi = plot_feature_importance(importance)
    st.plotly_chart(fig_fi, use_container_width=True, config={"displayModeBar": False})

with tab3:
    fig_rd = plot_risk_distribution(dataset)
    st.plotly_chart(fig_rd, use_container_width=True, config={"displayModeBar": False})


# ═══════════════════════════════════════════════════════════════════════════════
#  ROW 4 — EXPLAINABILITY
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.markdown("### 🔬 Why This Prediction?")
st.markdown("*Explainability breakdown showing top contributing features and their impact on the risk classification.*")

importance = get_feature_importance(models)
explain_data = generate_explainability(feats, importance)

if explain_data:
    # Create columns for each feature card
    cols = st.columns(min(4, len(explain_data)))
    for i, item in enumerate(explain_data):
        with cols[i % len(cols)]:
            impact_color = "#FF1744" if "High" in item["Impact"] else "#FFD600" if "Medium" in item["Impact"] else "#00E676"
            st.markdown(f"""
            <div class="glass-card" style="text-align:center; min-height:140px;">
                <div style="font-size:0.75rem; color:#FFFFFF; text-transform:uppercase; letter-spacing:1px;">{item['Feature']}</div>
                <div style="font-size:1.5rem; font-weight:700; color:#FFFFFF; margin:8px 0;">{item['Value']}</div>
                <div style="font-size:0.8rem; color:#FFFFFF;">Importance: <span style="color:{impact_color}; font-weight:600;">{item['Importance']}</span></div>
                <div style="font-size:0.85rem; margin-top:4px; color:#FFFFFF;">{item['Impact']}</div>
            </div>
            """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  ROW 5 — INPUT FEATURES SUMMARY (Expandable)
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

with st.expander("📋 Current Input Feature Vector", expanded=False):
    import pandas as pd
    feat_display = {k.replace("_", " ").title(): v for k, v in feats.items() if not k.startswith("artifact_") and not k.startswith("source_")}
    col_a, col_b = st.columns(2)
    items = list(feat_display.items())
    mid = len(items) // 2
    with col_a:
        for k, v in items[:mid]:
            st.markdown(f"**{k}:** `{v}`")
    with col_b:
        for k, v in items[mid:]:
            st.markdown(f"**{k}:** `{v}`")


# ═══════════════════════════════════════════════════════════════════════════════
#  FOOTER
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center; padding:20px; color:#FFFFFF; font-size:0.8rem;">
    <span style="background:linear-gradient(135deg, #818CF8, #C084FC); -webkit-background-clip:text; -webkit-text-fill-color:transparent; font-weight:700;">
        Living Attack Surface Mapper
    </span>
    <br>
    Hybrid ML Pipeline: Isolation Forest × Random Forest × XGBoost | Built with Streamlit & Plotly
    <br>
    K Likhita Reddy — 16010423042 — TY IT A
</div>
""", unsafe_allow_html=True)
