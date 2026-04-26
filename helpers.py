"""
Helper functions for the Living Attack Surface Mapper Dashboard.
Handles model loading, prediction, insight generation, and visualizations.
"""

import os, pickle, warnings
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

MODELS_DIR = "saved_models"
DATA_DIR = "model_results"
RISK_LABELS = {0: "Low", 1: "Medium", 2: "High", 3: "Critical", 4: "Emergency"}
RISK_COLORS = {"Low": "#00E676", "Medium": "#FFD600", "High": "#FF9100", "Critical": "#FF1744", "Emergency": "#D500F9"}
RISK_BG = {"Low": "rgba(0,230,118,0.15)", "Medium": "rgba(255,214,0,0.15)", "High": "rgba(255,145,0,0.15)", "Critical": "rgba(255,23,68,0.15)", "Emergency": "rgba(213,0,249,0.15)"}

SOURCE_MAP = {"GitHub": 0, "DNS": 1, "CVE": 2, "Paste Site": 3, "CT Logs": 4}

FEATURE_NAMES = [
    "subdomain_count", "subdomain_age_days", "domain_reputation_score",
    "cvss_base_score", "exploit_availability", "cve_age_days",
    "exposure_frequency", "has_credentials", "paste_site_mentions",
    "ct_log_anomaly", "ssl_days_remaining", "open_ports_count",
    "service_version_outdated", "dns_record_type_count", "is_wildcard_dns",
    "github_leak_count", "exposure_velocity",
    "leak_severity_encoded", "exploit_maturity_encoded",
    "source_CT_Log", "source_CVE", "source_DNS", "source_GitHub", "source_Paste",
    "artifact_API_Key", "artifact_CVE_Entry", "artifact_Cert_Anomaly",
    "artifact_Certificate", "artifact_Config", "artifact_Config_Dump",
    "artifact_Credential_Leak", "artifact_DNS_Record", "artifact_Data_Dump",
    "artifact_Exploit_Code", "artifact_PII_Leak", "artifact_Password",
    "artifact_Secret_Token", "artifact_Source_Code", "artifact_Subdomain",
    "artifact_Vulnerability", "artifact_Wildcard_Cert", "artifact_Wildcard_Entry",
    "artifact_Zone_Transfer", "anomaly_score",
]


# ── Model Loading ────────────────────────────────────────────────────────────
def load_models():
    """Load trained models from disk. Returns dict of models or None values."""
    models = {"iso_forest": None, "random_forest": None, "xgboost": None,
              "scaler_iso": None, "scaler_sup": None}
    files = {
        "iso_forest": "isolation_forest.pkl", "random_forest": "random_forest.pkl",
        "xgboost": "xgboost.pkl", "scaler_iso": "scaler_isolation.pkl",
        "scaler_sup": "scaler_supervised.pkl",
    }
    for key, fname in files.items():
        path = os.path.join(MODELS_DIR, fname)
        if os.path.exists(path):
            with open(path, "rb") as f:
                models[key] = pickle.load(f)
    return models


def load_dataset():
    """Load preprocessed dataset with anomaly scores."""
    path = "preprocessed_dataset_with_anomaly.csv"
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


def load_timeseries():
    """Load daily risk timeseries data."""
    path = os.path.join(DATA_DIR, "daily_risk_timeseries.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        df["date"] = pd.to_datetime(df["date"])
        return df
    return None


# ── Feature Construction ─────────────────────────────────────────────────────
def build_feature_vector(leak_sev, cvss, source_type, threshold):
    """Build a feature vector from sidebar inputs using realistic defaults."""
    sev_map = {"Low": 0, "Medium": 1, "High": 2, "Critical": 3}
    leak_int = int(round(leak_sev))
    if leak_int <= 2: sev_label = "Low"
    elif leak_int <= 5: sev_label = "Medium"
    elif leak_int <= 8: sev_label = "High"
    else: sev_label = "Critical"
    sev_enc = sev_map[sev_label]

    exploit_avail = 1 if cvss > 7 else 0
    mat_enc = min(3, int(cvss / 3))

    np.random.seed(int(leak_sev * 100 + cvss * 10))
    feats = {
        "subdomain_count": np.random.randint(1, 30),
        "subdomain_age_days": np.random.randint(10, 400),
        "domain_reputation_score": max(0, 1 - (cvss / 10) * 0.6 + np.random.uniform(-0.1, 0.1)),
        "cvss_base_score": cvss,
        "exploit_availability": exploit_avail,
        "cve_age_days": np.random.randint(5, 800),
        "exposure_frequency": max(1, int(leak_sev * 1.5 + np.random.randint(0, 5))),
        "has_credentials": 1 if leak_sev > 6 else 0,
        "paste_site_mentions": max(0, int(leak_sev * 2 + np.random.randint(-2, 8))),
        "ct_log_anomaly": 1 if np.random.random() < 0.3 else 0,
        "ssl_days_remaining": max(-60, int(200 - cvss * 25 + np.random.randint(-30, 30))),
        "open_ports_count": np.random.randint(0, 20),
        "service_version_outdated": 1 if cvss > 5 else 0,
        "dns_record_type_count": np.random.randint(1, 10),
        "is_wildcard_dns": 1 if np.random.random() < 0.2 else 0,
        "github_leak_count": max(0, int(leak_sev * 1.2 + np.random.randint(-1, 4))),
        "exposure_velocity": round(leak_sev * cvss / 10 + np.random.uniform(-0.5, 1.5), 2),
        "leak_severity_encoded": sev_enc,
        "exploit_maturity_encoded": mat_enc,
    }

    # One-hot source
    src_keys = ["source_CT_Log", "source_CVE", "source_DNS", "source_GitHub", "source_Paste"]
    for k in src_keys:
        feats[k] = 0
    src_map = {"GitHub": "source_GitHub", "DNS": "source_DNS", "CVE": "source_CVE",
               "Paste Site": "source_Paste", "CT Logs": "source_CT_Log"}
    feats[src_map[source_type]] = 1

    # One-hot artifact (random selection based on source)
    art_keys = ["artifact_API_Key", "artifact_CVE_Entry", "artifact_Cert_Anomaly",
                "artifact_Certificate", "artifact_Config", "artifact_Config_Dump",
                "artifact_Credential_Leak", "artifact_DNS_Record", "artifact_Data_Dump",
                "artifact_Exploit_Code", "artifact_PII_Leak", "artifact_Password",
                "artifact_Secret_Token", "artifact_Source_Code", "artifact_Subdomain",
                "artifact_Vulnerability", "artifact_Wildcard_Cert", "artifact_Wildcard_Entry",
                "artifact_Zone_Transfer"]
    for k in art_keys:
        feats[k] = 0
    art_map = {
        "GitHub": ["artifact_Password", "artifact_Secret_Token", "artifact_API_Key", "artifact_Config"],
        "CVE": ["artifact_CVE_Entry", "artifact_Exploit_Code", "artifact_Vulnerability"],
        "DNS": ["artifact_Subdomain", "artifact_DNS_Record", "artifact_Zone_Transfer", "artifact_Wildcard_Entry"],
        "Paste Site": ["artifact_Credential_Leak", "artifact_PII_Leak", "artifact_Data_Dump", "artifact_Database_Dump"],
        "CT Logs": ["artifact_Certificate", "artifact_Cert_Anomaly", "artifact_Wildcard_Cert", "artifact_Expired_Cert"],
    }
    choices = art_map.get(source_type, ["artifact_Config"])
    valid = [c for c in choices if c in art_keys]
    if valid:
        feats[valid[np.random.randint(0, len(valid))]] = 1

    vec = [feats.get(f, 0) for f in FEATURE_NAMES[:-1]]  # exclude anomaly_score
    return np.array(vec).reshape(1, -1), feats


# ── Prediction ───────────────────────────────────────────────────────────────
def predict_risk(models, feature_vec, threshold=0.05):
    """Run prediction pipeline. Returns risk_level, risk_score, anomaly_score, confidence."""
    anomaly_score = 0.5
    risk_score = 2
    confidence = 0.75

    iso = models.get("iso_forest")
    scaler_iso = models.get("scaler_iso")
    xgb = models.get("xgboost")
    rf = models.get("random_forest")

    if iso and scaler_iso:
        try:
            X_sc = scaler_iso.transform(feature_vec)
            raw = iso.decision_function(X_sc)[0]
            anomaly_score = float(1 - (raw - (-0.5)) / (0.5 - (-0.5)))
            anomaly_score = np.clip(anomaly_score, 0, 1)
        except:
            anomaly_score = simulate_anomaly(feature_vec)
    else:
        anomaly_score = simulate_anomaly(feature_vec)

    X_full = np.column_stack([feature_vec, [[anomaly_score]]])

    if xgb:
        try:
            risk_score = int(xgb.predict(X_full)[0])
            proba = xgb.predict_proba(X_full)[0]
            confidence = float(np.max(proba))
        except:
            risk_score, confidence = simulate_risk(feature_vec, anomaly_score)
    else:
        risk_score, confidence = simulate_risk(feature_vec, anomaly_score)

    risk_score = np.clip(risk_score, 0, 4)
    risk_level = RISK_LABELS[risk_score]
    return risk_level, int(risk_score), float(anomaly_score), float(confidence)


def simulate_anomaly(feature_vec):
    v = feature_vec.flatten()
    cvss = v[3] if len(v) > 3 else 5
    sev = v[17] if len(v) > 17 else 1
    return np.clip((cvss / 10) * 0.5 + (sev / 3) * 0.3 + np.random.uniform(0, 0.2), 0, 1)


def simulate_risk(feature_vec, anomaly_score):
    v = feature_vec.flatten()
    cvss = v[3] if len(v) > 3 else 5
    sev = v[17] if len(v) > 17 else 1
    combo = cvss * 0.35 + sev * 1.2 + anomaly_score * 3
    if combo > 7: risk = 4
    elif combo > 5.5: risk = 3
    elif combo > 4: risk = 2
    elif combo > 2.5: risk = 1
    else: risk = 0
    conf = 0.6 + np.random.uniform(0, 0.3)
    return risk, conf


def get_feature_importance(models):
    """Get feature importance from Random Forest model."""
    rf = models.get("random_forest")
    if rf:
        try:
            imp = rf.feature_importances_
            names = FEATURE_NAMES
            if len(imp) != len(names):
                names = [f"feature_{i}" for i in range(len(imp))]
            return dict(zip(names, imp))
        except:
            pass
    # Fallback simulated importances
    return {
        "cvss_base_score": 0.18, "anomaly_score": 0.14, "leak_severity_encoded": 0.12,
        "exposure_velocity": 0.09, "paste_site_mentions": 0.07, "github_leak_count": 0.06,
        "exploit_maturity_encoded": 0.06, "domain_reputation_score": 0.05,
        "ssl_days_remaining": 0.04, "exposure_frequency": 0.04,
        "has_credentials": 0.03, "open_ports_count": 0.03,
        "cve_age_days": 0.02, "subdomain_count": 0.02,
        "exploit_availability": 0.02, "ct_log_anomaly": 0.01,
        "dns_record_type_count": 0.01, "subdomain_age_days": 0.01,
    }


# ── GenAI Insight Generation ────────────────────────────────────────────────
def generate_insight(risk_level, risk_score, anomaly_score, confidence, feats):
    """Generate a human-readable insight paragraph based on prediction results."""
    insights = []
    cvss = feats.get("cvss_base_score", 0)
    sev = feats.get("leak_severity_encoded", 0)
    creds = feats.get("has_credentials", 0)
    paste = feats.get("paste_site_mentions", 0)
    gh = feats.get("github_leak_count", 0)
    ssl = feats.get("ssl_days_remaining", 200)
    velocity = feats.get("exposure_velocity", 0)

    # Risk headline
    if risk_score >= 4:
        insights.append(f"🚨 **EMERGENCY** threat level detected with {confidence*100:.0f}% model confidence.")
    elif risk_score == 3:
        insights.append(f"⚠️ **CRITICAL** risk identified. Immediate investigation recommended.")
    elif risk_score == 2:
        insights.append(f"🔶 **HIGH** risk detected. Active monitoring and remediation advised.")
    elif risk_score == 1:
        insights.append(f"🟡 **MEDIUM** risk level. Standard monitoring procedures apply.")
    else:
        insights.append(f"🟢 **LOW** risk. No immediate threats detected.")

    # Contributing factors
    factors = []
    if cvss >= 8: factors.append(f"critical CVSS score ({cvss:.1f}/10)")
    elif cvss >= 6: factors.append(f"elevated CVSS score ({cvss:.1f}/10)")
    if anomaly_score > 0.7: factors.append(f"high anomaly score ({anomaly_score:.2f})")
    if sev >= 3: factors.append("critical leak severity classification")
    elif sev >= 2: factors.append("high leak severity")
    if creds: factors.append("exposed credentials detected")
    if paste > 5: factors.append(f"{paste} paste site mentions indicating active data exposure")
    if gh > 3: factors.append(f"{gh} GitHub leak instances")
    if ssl < 0: factors.append(f"expired SSL certificate ({abs(ssl)} days overdue)")
    elif ssl < 30: factors.append(f"SSL certificate expiring soon ({ssl} days)")
    if velocity > 3: factors.append(f"rapid exposure velocity ({velocity:.1f})")

    if factors:
        insights.append("**Key contributing factors:** " + ", ".join(factors) + ".")

    # Recommendation
    if risk_score >= 3:
        insights.append("**Recommended actions:** Initiate incident response protocol. Rotate all exposed credentials immediately. Patch vulnerable systems and review access logs for indicators of compromise.")
    elif risk_score == 2:
        insights.append("**Recommended actions:** Schedule vulnerability remediation within 48 hours. Monitor affected endpoints for suspicious activity.")
    elif risk_score == 1:
        insights.append("**Recommended actions:** Add to regular patch cycle. Review exposure sources during next security audit.")
    else:
        insights.append("**Recommended actions:** Continue standard monitoring. No urgent action required.")

    return "\n\n".join(insights)


def generate_explainability(feats, importance_dict):
    """Generate 'Why this prediction?' explainability breakdown."""
    contrib = []
    for feat, imp in sorted(importance_dict.items(), key=lambda x: -x[1])[:8]:
        val = feats.get(feat, 0)
        if val is not None and imp > 0.01:
            contrib.append({"Feature": feat.replace("_", " ").title(), "Value": f"{val}", "Importance": f"{imp:.3f}", "Impact": "🔴 High" if imp > 0.1 else "🟡 Medium" if imp > 0.04 else "🟢 Low"})
    return contrib


# ── Visualization Functions ──────────────────────────────────────────────────
def plot_timeseries(ts_df, attack_mode=False):
    """Create time-series risk trend chart with spike detection."""
    if ts_df is None:
        return _generate_synthetic_ts(attack_mode)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3],
                        vertical_spacing=0.08, subplot_titles=("Risk Score Trend", "Event Volume"))

    df = ts_df.copy()
    if attack_mode:
        n = len(df)
        spike_start = max(0, n - 15)
        df.loc[spike_start:, "mean_risk"] = df.loc[spike_start:, "mean_risk"] * 1.8 + 1.5
        df.loc[spike_start:, "rolling_7d"] = df.loc[spike_start:, "mean_risk"].rolling(7, min_periods=1).mean()
        df.loc[spike_start:, "is_spike"] = 1

    fig.add_trace(go.Scatter(x=df["date"], y=df["mean_risk"], mode="lines", name="Daily Mean Risk",
                             line=dict(color="#60A5FA", width=1), opacity=0.4), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["date"], y=df["rolling_7d"], mode="lines", name="7-Day Rolling Avg",
                             line=dict(color="#34D399", width=2.5)), row=1, col=1)
    if "rolling_30d" in df.columns:
        fig.add_trace(go.Scatter(x=df["date"], y=df["rolling_30d"], mode="lines", name="30-Day Rolling Avg",
                                 line=dict(color="#F87171", width=2.5)), row=1, col=1)
    if "ewma_14d" in df.columns:
        fig.add_trace(go.Scatter(x=df["date"], y=df["ewma_14d"], mode="lines", name="EWMA (14d)",
                                 line=dict(color="#C084FC", width=2, dash="dash")), row=1, col=1)

    spikes = df[df["is_spike"] == 1] if "is_spike" in df.columns else pd.DataFrame()
    if len(spikes) > 0:
        fig.add_trace(go.Scatter(x=spikes["date"], y=spikes["mean_risk"], mode="markers",
                                 name="⚡ Risk Spikes", marker=dict(color="#FF1744", size=10, symbol="triangle-up")), row=1, col=1)

    if "event_count" in df.columns:
        fig.add_trace(go.Bar(x=df["date"], y=df["event_count"], name="Events",
                             marker_color="rgba(96,165,250,0.4)"), row=2, col=1)

    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      height=500, margin=dict(l=20, r=20, t=40, b=20), legend=dict(orientation="h", y=-0.15),
                      font=dict(family="Inter, sans-serif", color="#FFFFFF"))
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.05)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.05)")
    return fig


def _generate_synthetic_ts(attack_mode=False):
    dates = pd.date_range("2025-01-01", periods=180, freq="D")
    base = np.sin(np.linspace(0, 4 * np.pi, 180)) * 0.5 + 2
    noise = np.random.normal(0, 0.3, 180)
    risk = base + noise
    if attack_mode:
        risk[-15:] += np.linspace(0, 3, 15)
    df = pd.DataFrame({"date": dates, "mean_risk": risk})
    df["rolling_7d"] = df["mean_risk"].rolling(7, min_periods=1).mean()
    df["rolling_30d"] = df["mean_risk"].rolling(30, min_periods=1).mean()
    df["is_spike"] = (df["mean_risk"] > df["rolling_30d"] + 1.5).astype(int)
    df["event_count"] = np.random.randint(20, 80, 180)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3],
                        vertical_spacing=0.08, subplot_titles=("Risk Score Trend", "Event Volume"))
    fig.add_trace(go.Scatter(x=df["date"], y=df["mean_risk"], mode="lines", name="Daily Risk",
                             line=dict(color="#60A5FA", width=1), opacity=0.4), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["date"], y=df["rolling_7d"], mode="lines", name="7d Avg",
                             line=dict(color="#34D399", width=2.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["date"], y=df["rolling_30d"], mode="lines", name="30d Avg",
                             line=dict(color="#F87171", width=2.5)), row=1, col=1)
    spk = df[df["is_spike"] == 1]
    if len(spk):
        fig.add_trace(go.Scatter(x=spk["date"], y=spk["mean_risk"], mode="markers", name="Spikes",
                                 marker=dict(color="#FF1744", size=10, symbol="triangle-up")), row=1, col=1)
    fig.add_trace(go.Bar(x=df["date"], y=df["event_count"], name="Events",
                         marker_color="rgba(96,165,250,0.4)"), row=2, col=1)
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      height=500, margin=dict(l=20, r=20, t=40, b=20), legend=dict(orientation="h", y=-0.15),
                      font=dict(family="Inter, sans-serif", color="#FFFFFF"))
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.05)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.05)")
    return fig


def plot_feature_importance(importance_dict):
    """Create horizontal bar chart of feature importances."""
    sorted_items = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:15]
    names = [x[0].replace("_", " ").title() for x in sorted_items][::-1]
    vals = [x[1] for x in sorted_items][::-1]

    colors = []
    for v in vals:
        if v > 0.1: colors.append("#FF1744")
        elif v > 0.05: colors.append("#FF9100")
        elif v > 0.03: colors.append("#FFD600")
        else: colors.append("#00E676")

    fig = go.Figure(go.Bar(x=vals, y=names, orientation="h", marker_color=colors,
                           text=[f"{v:.3f}" for v in vals], textposition="outside",
                           textfont=dict(size=11, color="#FFFFFF")))
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      height=450, margin=dict(l=20, r=40, t=30, b=20), xaxis_title="Importance Score",
                      font=dict(family="Inter, sans-serif", color="#FFFFFF"))
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.05)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.05)")
    return fig


def plot_risk_distribution(dataset=None):
    """Create risk distribution donut chart."""
    if dataset is not None and "risk_score" in dataset.columns:
        counts = dataset["risk_score"].value_counts().sort_index()
        labels = [RISK_LABELS.get(i, f"Level {i}") for i in counts.index]
        values = counts.values
    else:
        labels = ["Low", "Medium", "High", "Critical", "Emergency"]
        values = [4200, 6800, 5100, 3200, 700]

    colors = [RISK_COLORS.get(l, "#888") for l in labels]

    fig = go.Figure(go.Pie(labels=labels, values=values, hole=0.55, marker=dict(colors=colors, line=dict(color="#1E293B", width=2)),
                           textinfo="label+percent", textfont=dict(size=13, color="#FFFFFF"),
                           hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Percent: %{percent}<extra></extra>"))
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      height=350, margin=dict(l=20, r=20, t=30, b=20), showlegend=False,
                      font=dict(family="Inter, sans-serif", color="#FFFFFF"),
                      annotations=[dict(text="Risk<br>Distribution", x=0.5, y=0.5, font_size=14, font_color="#FFFFFF", showarrow=False)])
    return fig


def plot_anomaly_gauge(anomaly_score):
    """Create a gauge chart for anomaly score."""
    color = "#00E676" if anomaly_score < 0.3 else "#FFD600" if anomaly_score < 0.6 else "#FF9100" if anomaly_score < 0.8 else "#FF1744"
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=anomaly_score * 100,
        title={"text": "Anomaly Score", "font": {"size": 16, "color": "#FFFFFF"}},
        number={"suffix": "%", "font": {"size": 28, "color": color}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#475569"},
            "bar": {"color": color, "thickness": 0.7},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 30], "color": "rgba(0,230,118,0.1)"},
                {"range": [30, 60], "color": "rgba(255,214,0,0.1)"},
                {"range": [60, 80], "color": "rgba(255,145,0,0.1)"},
                {"range": [80, 100], "color": "rgba(255,23,68,0.1)"},
            ],
        }
    ))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      height=220, margin=dict(l=20, r=20, t=40, b=10),
                      font=dict(family="Inter, sans-serif", color="#FFFFFF"))
    return fig
