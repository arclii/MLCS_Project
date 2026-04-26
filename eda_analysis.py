#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  Living Attack Surface Mapper — Exploratory Data Analysis (EDA)            ║
║  Input    : preprocessed_dataset_with_anomaly.csv                          ║
║  Output   : eda_plots/ (11 publication-ready visualizations)               ║
║  Author   : K Likhita Reddy — 16010423042 — TY IT A                       ║
╚══════════════════════════════════════════════════════════════════════════════╝

Generates comprehensive publication-ready visualizations for the attack
surface exposure dataset, covering risk distribution, feature analysis,
anomaly detection, and time-series patterns.
"""

import os
import warnings

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

# ── Style ────────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
RISK_LABELS = {0: "Low", 1: "Medium", 2: "High", 3: "Critical", 4: "Emergency"}
RISK_PALETTE = {
    "Low": "#2ecc71", "Medium": "#f39c12", "High": "#e67e22",
    "Critical": "#e74c3c", "Emergency": "#8e44ad"
}
RISK_ORDER = ["Low", "Medium", "High", "Critical", "Emergency"]

OUTDIR = "eda_plots"
os.makedirs(OUTDIR, exist_ok=True)

# ── Load ─────────────────────────────────────────────────────────────────────
df = pd.read_csv("preprocessed_dataset_with_anomaly.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])
df["risk_label"] = df["risk_score"].map(RISK_LABELS)

print(f"Loaded {len(df):,} samples  •  Columns: {len(df.columns)}")
print(f"\nRisk Score distribution:")
print(df["risk_label"].value_counts().reindex(RISK_ORDER).to_string())
print(f"\nDataset describe:")
print(df.describe().round(2).to_string())

# ── Key numeric features ────────────────────────────────────────────────────
CONTINUOUS = [
    "cvss_base_score", "domain_reputation_score", "exposure_frequency",
    "subdomain_count", "subdomain_age_days", "cve_age_days",
    "open_ports_count", "exposure_velocity", "anomaly_score",
]


# ═══════════════════════════════════════════════════════════════════════════════
#  1. RISK SCORE CLASS DISTRIBUTION
# ═══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 6))
counts = df["risk_label"].value_counts().reindex(RISK_ORDER)
bars = ax.bar(counts.index, counts.values,
              color=[RISK_PALETTE[l] for l in counts.index],
              edgecolor="white", linewidth=1.5)
for bar, val in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 80,
            f"{val:,}\n({val/len(df)*100:.1f}%)", ha="center", va="bottom",
            fontweight="bold", fontsize=11)
ax.set_title("Risk Score Class Distribution", fontsize=16, fontweight="bold")
ax.set_ylabel("Count")
ax.set_ylim(0, counts.max() * 1.20)
plt.tight_layout()
fig.savefig(f"{OUTDIR}/01_risk_distribution.png", dpi=150)
plt.close()
print("✅  01_risk_distribution.png")


# ═══════════════════════════════════════════════════════════════════════════════
#  2. FEATURE DISTRIBUTIONS BY RISK LEVEL
# ═══════════════════════════════════════════════════════════════════════════════
plot_features = ["cvss_base_score", "domain_reputation_score",
                  "exposure_frequency", "subdomain_count"]
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for ax, feat in zip(axes.ravel(), plot_features):
    for label in RISK_ORDER:
        subset = df.loc[df["risk_label"] == label, feat]
        if len(subset) > 0:
            ax.hist(subset, bins=40, alpha=0.4, label=label,
                    color=RISK_PALETTE[label], edgecolor="white", linewidth=0.3)
    ax.set_title(f"{feat} Distribution by Risk Level", fontsize=12, fontweight="bold")
    ax.set_xlabel(feat)
    ax.set_ylabel("Frequency")
    ax.legend(fontsize=8)
plt.suptitle("Feature Distributions by Risk Level", fontsize=16, fontweight="bold", y=1.02)
plt.tight_layout()
fig.savefig(f"{OUTDIR}/02_feature_distributions.png", dpi=150, bbox_inches="tight")
plt.close()
print("✅  02_feature_distributions.png")


# ═══════════════════════════════════════════════════════════════════════════════
#  3. CORRELATION HEATMAP
# ═══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(14, 11))
corr_features = [
    "subdomain_count", "subdomain_age_days", "domain_reputation_score",
    "cvss_base_score", "exploit_availability", "cve_age_days",
    "exposure_frequency", "has_credentials", "paste_site_mentions",
    "ct_log_anomaly", "ssl_days_remaining", "open_ports_count",
    "service_version_outdated", "github_leak_count", "exposure_velocity",
    "anomaly_score", "risk_score",
]
corr = df[corr_features].corr()
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlGn_r",
            linewidths=0.5, ax=ax, vmin=-1, vmax=1,
            cbar_kws={"shrink": 0.8, "label": "Pearson r"},
            annot_kws={"size": 8})
ax.set_title("Feature Correlation Matrix", fontsize=16, fontweight="bold")
plt.tight_layout()
fig.savefig(f"{OUTDIR}/03_correlation_heatmap.png", dpi=150)
plt.close()
print("✅  03_correlation_heatmap.png")


# ═══════════════════════════════════════════════════════════════════════════════
#  4. BOX PLOTS PER RISK CLASS
# ═══════════════════════════════════════════════════════════════════════════════
box_features = ["cvss_base_score", "exposure_frequency",
                 "domain_reputation_score", "exposure_velocity"]
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for ax, feat in zip(axes.ravel(), box_features):
    sns.boxplot(data=df, x="risk_label", y=feat, palette=RISK_PALETTE, ax=ax,
                order=RISK_ORDER, fliersize=2)
    ax.set_title(f"{feat} by Risk Level", fontsize=12, fontweight="bold")
    ax.set_xlabel("Risk Level")
plt.suptitle("Box Plots — Feature Spread per Risk Level", fontsize=16,
             fontweight="bold", y=1.02)
plt.tight_layout()
fig.savefig(f"{OUTDIR}/04_box_plots.png", dpi=150, bbox_inches="tight")
plt.close()
print("✅  04_box_plots.png")


# ═══════════════════════════════════════════════════════════════════════════════
#  5. FEATURE IMPORTANCE (from saved model results)
# ═══════════════════════════════════════════════════════════════════════════════
# Read the feature importance from the saved Random Forest model
import pickle
try:
    with open("saved_models/random_forest.pkl", "rb") as f:
        rf_model = pickle.load(f)

    feature_names = [
        "subdomain_count", "subdomain_age_days", "domain_reputation_score",
        "cvss_base_score", "exploit_availability", "cve_age_days",
        "exposure_frequency", "has_credentials", "paste_site_mentions",
        "ct_log_anomaly", "ssl_days_remaining", "open_ports_count",
        "service_version_outdated", "dns_record_type_count", "is_wildcard_dns",
        "github_leak_count", "exposure_velocity",
        "leak_severity_encoded", "exploit_maturity_encoded",
    ]
    # Add source and artifact one-hot columns
    source_cols = [c for c in df.columns if c.startswith("source_") and c != "source_type"]
    artifact_cols = [c for c in df.columns if c.startswith("artifact_") and c != "artifact_type"]
    feature_names += source_cols + artifact_cols + ["anomaly_score"]

    importances = rf_model.feature_importances_
    sorted_idx = np.argsort(importances)[-15:]  # top 15

    fig, ax = plt.subplots(figsize=(10, 8))
    colors_imp = plt.cm.viridis(np.linspace(0.3, 0.9, len(sorted_idx)))
    ax.barh([feature_names[i] for i in sorted_idx], importances[sorted_idx],
            color=colors_imp, edgecolor="white", linewidth=1)
    for i, v in enumerate(importances[sorted_idx]):
        ax.text(v + 0.002, i, f"{v:.3f}", va="center", fontsize=9)
    ax.set_title("Top 15 Feature Importances (Random Forest)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    fig.savefig(f"{OUTDIR}/05_feature_importance.png", dpi=150)
    plt.close()
    print("✅  05_feature_importance.png")
except Exception as e:
    print(f"⚠️  05_feature_importance.png skipped: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
#  6. ANOMALY DETECTION SCATTER PLOT
# ═══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Anomaly Score vs CVSS
ax1 = axes[0]
scatter1 = ax1.scatter(df["cvss_base_score"], df["anomaly_score"],
                        c=df["risk_score"], cmap="RdYlGn_r", alpha=0.3, s=8)
ax1.set_xlabel("CVSS Base Score")
ax1.set_ylabel("Anomaly Score (Isolation Forest)")
ax1.set_title("Anomaly Score vs CVSS", fontsize=13, fontweight="bold")
plt.colorbar(scatter1, ax=ax1, label="Risk Score")

# Anomaly Score vs Exposure Velocity
ax2 = axes[1]
scatter2 = ax2.scatter(df["exposure_velocity"], df["anomaly_score"],
                        c=df["risk_score"], cmap="RdYlGn_r", alpha=0.3, s=8)
ax2.set_xlabel("Exposure Velocity")
ax2.set_ylabel("Anomaly Score (Isolation Forest)")
ax2.set_title("Anomaly Score vs Exposure Velocity", fontsize=13, fontweight="bold")
plt.colorbar(scatter2, ax=ax2, label="Risk Score")

plt.suptitle("Isolation Forest Anomaly Detection", fontsize=16, fontweight="bold", y=1.02)
plt.tight_layout()
fig.savefig(f"{OUTDIR}/06_anomaly_detection.png", dpi=150, bbox_inches="tight")
plt.close()
print("✅  06_anomaly_detection.png")


# ═══════════════════════════════════════════════════════════════════════════════
#  7. TIME-SERIES RISK TREND
# ═══════════════════════════════════════════════════════════════════════════════
ts_df = df[["timestamp", "risk_score"]].copy()
ts_df["date"] = ts_df["timestamp"].dt.date
daily = ts_df.groupby("date").agg(
    mean_risk=("risk_score", "mean"),
    count=("risk_score", "count"),
).reset_index()
daily["date"] = pd.to_datetime(daily["date"])
daily["rolling_7d"] = daily["mean_risk"].rolling(window=7, min_periods=1).mean()
daily["rolling_30d"] = daily["mean_risk"].rolling(window=30, min_periods=1).mean()

fig, ax = plt.subplots(figsize=(16, 6))
ax.plot(daily["date"], daily["mean_risk"],
        alpha=0.3, color="#3498db", linewidth=0.8, label="Daily Mean Risk")
ax.plot(daily["date"], daily["rolling_7d"],
        color="#2ecc71", linewidth=2, label="7-Day Rolling Avg")
ax.plot(daily["date"], daily["rolling_30d"],
        color="#e74c3c", linewidth=2, label="30-Day Rolling Avg")
ax.set_title("Daily Risk Score Trend (12-Month Period)", fontsize=14, fontweight="bold")
ax.set_ylabel("Mean Risk Score")
ax.set_xlabel("Date")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(f"{OUTDIR}/07_timeseries_trend.png", dpi=150)
plt.close()
print("✅  07_timeseries_trend.png")


# ═══════════════════════════════════════════════════════════════════════════════
#  8. RISK SPIKE DETECTION
# ═══════════════════════════════════════════════════════════════════════════════
rolling_std = daily["mean_risk"].rolling(window=30, min_periods=7).std()
daily["spike_threshold"] = daily["rolling_30d"] + 2 * rolling_std
daily["is_spike"] = (daily["mean_risk"] > daily["spike_threshold"]).astype(int)

fig, ax = plt.subplots(figsize=(16, 6))
ax.fill_between(daily["date"], daily["rolling_30d"],
                daily["spike_threshold"], alpha=0.15, color="#e74c3c",
                label="Normal Zone")
ax.plot(daily["date"], daily["mean_risk"],
        alpha=0.5, color="#3498db", linewidth=1, label="Daily Mean Risk")
ax.plot(daily["date"], daily["spike_threshold"],
        color="#e74c3c", linewidth=1.5, linestyle="--", label="Spike Threshold (μ + 2σ)")

spikes = daily[daily["is_spike"] == 1]
ax.scatter(spikes["date"], spikes["mean_risk"],
           color="#e74c3c", s=80, zorder=5, edgecolors="white", linewidth=1,
           label=f"Risk Spikes ({len(spikes)} days)")

ax.set_title("Risk Spike Detection — Anomalous High-Risk Periods",
             fontsize=14, fontweight="bold")
ax.set_ylabel("Mean Risk Score")
ax.set_xlabel("Date")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(f"{OUTDIR}/08_risk_spikes.png", dpi=150)
plt.close()
print("✅  08_risk_spikes.png")


# ═══════════════════════════════════════════════════════════════════════════════
#  9. SOURCE-TYPE RISK BREAKDOWN
# ═══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 9a: Mean risk by source type
ax_a = axes[0]
src_risk = df.groupby("source_type")["risk_score"].mean().sort_values(ascending=True)
colors_src = ["#3498db", "#2ecc71", "#e67e22", "#9b59b6", "#e74c3c"]
bars = ax_a.barh(src_risk.index, src_risk.values,
                  color=colors_src[:len(src_risk)], edgecolor="white", linewidth=1)
for bar, val in zip(bars, src_risk.values):
    ax_a.text(val + 0.02, bar.get_y() + bar.get_height() / 2,
              f"{val:.2f}", va="center", fontsize=11)
ax_a.set_title("Mean Risk Score by Source Type", fontsize=13, fontweight="bold")
ax_a.set_xlabel("Mean Risk Score")

# 9b: Risk class distribution per source type
ax_b = axes[1]
ct = pd.crosstab(df["source_type"], df["risk_label"])
ct = ct[RISK_ORDER]
ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100
ct_pct.plot(kind="bar", stacked=True, ax=ax_b,
            color=[RISK_PALETTE[l] for l in RISK_ORDER],
            edgecolor="white", linewidth=0.5)
ax_b.set_title("Risk Class Distribution by Source Type", fontsize=13, fontweight="bold")
ax_b.set_ylabel("Percentage (%)")
ax_b.set_xticklabels(ax_b.get_xticklabels(), rotation=45, ha="right")
ax_b.legend(title="Risk Level", fontsize=8, loc="upper right")

plt.suptitle("OSINT Source Analysis", fontsize=16, fontweight="bold", y=1.02)
plt.tight_layout()
fig.savefig(f"{OUTDIR}/09_source_risk_breakdown.png", dpi=150, bbox_inches="tight")
plt.close()
print("✅  09_source_risk_breakdown.png")


# ═══════════════════════════════════════════════════════════════════════════════
#  10. PAIR PLOT (sampled for speed)
# ═══════════════════════════════════════════════════════════════════════════════
sample = df.sample(n=3000, random_state=42)
pair_features = ["cvss_base_score", "domain_reputation_score",
                  "exposure_frequency", "anomaly_score"]
g = sns.pairplot(sample, vars=pair_features, hue="risk_label",
                 hue_order=RISK_ORDER, palette=RISK_PALETTE,
                 diag_kind="kde", plot_kws={"alpha": 0.3, "s": 10},
                 diag_kws={"fill": True, "alpha": 0.4})
g.figure.suptitle("Pair Plot — Key Feature Relationships (n=3,000)",
                   fontsize=16, fontweight="bold", y=1.02)
g.savefig(f"{OUTDIR}/10_pair_plot.png", dpi=120, bbox_inches="tight")
plt.close()
print("✅  10_pair_plot.png")


# ═══════════════════════════════════════════════════════════════════════════════
#  11. VIOLIN PLOTS
# ═══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
violin_features = ["cvss_base_score", "exposure_velocity", "anomaly_score"]
for ax, feat in zip(axes, violin_features):
    sns.violinplot(data=df, x="risk_label", y=feat, palette=RISK_PALETTE,
                    ax=ax, order=RISK_ORDER, inner="quartile")
    ax.set_title(f"{feat} by Risk Level", fontsize=12, fontweight="bold")
    ax.set_xlabel("Risk Level")
plt.suptitle("Feature Density Shapes by Risk Level", fontsize=16,
             fontweight="bold", y=1.02)
plt.tight_layout()
fig.savefig(f"{OUTDIR}/11_violin_plots.png", dpi=150, bbox_inches="tight")
plt.close()
print("✅  11_violin_plots.png")


# ═══════════════════════════════════════════════════════════════════════════════
#  SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n🏁  All {11} EDA plots saved to ./{OUTDIR}/")
print("   Plot listing:")
for f in sorted(os.listdir(OUTDIR)):
    fpath = os.path.join(OUTDIR, f)
    size_kb = os.path.getsize(fpath) / 1024
    print(f"      {f:45s}  {size_kb:.0f} KB")
