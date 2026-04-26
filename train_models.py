#!/usr/bin/env python3
"""

Living Attack Surface Mapper —Hybrid ML Training & Evaluation Pipeline   
Input    : preprocessed_dataset.csv                                        
Output   : saved_models/, model_results/                                   
                       


Three-stage hybrid ML system:
    Stage 1: Unsupervised — Isolation Forest (anomaly detection)
    Stage 2: Supervised   — Random Forest + XGBoost (risk classification)
    Stage 3: Time-Series  — Rolling Average + EWMA forecasting (testing)
"""

import os
import warnings
import pickle

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", font_scale=1.05)

RESULTS_DIR = "model_results"
MODELS_DIR  = "saved_models"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

SEED = 42
RISK_LABELS = {0: "Low", 1: "Medium", 2: "High", 3: "Critical", 4: "Emergency"}
RISK_COLORS = {
    "Low": "#2ecc71", "Medium": "#f39c12", "High": "#e67e22",
    "Critical": "#e74c3c", "Emergency": "#8e44ad"
}


# ═══════════════════════════════════════════════════════════════════════════════
#  1. LOAD & PREPARE DATA
# ═══════════════════════════════════════════════════════════════════════════════
print("━" * 65)
print("  Living Attack Surface Mapper — Model Training Pipeline")
print("━" * 65)

df = pd.read_csv("preprocessed_dataset.csv")
print(f"\n📂  Loaded {len(df):,} preprocessed samples")

# Parse timestamps for time-series
df["timestamp"] = pd.to_datetime(df["timestamp"])

# ── Define feature columns ───────────────────────────────────────────────────
# Use original numeric features + encoded categoricals
FEATURES = [
    "subdomain_count", "subdomain_age_days", "domain_reputation_score",
    "cvss_base_score", "exploit_availability", "cve_age_days",
    "exposure_frequency", "has_credentials", "paste_site_mentions",
    "ct_log_anomaly", "ssl_days_remaining", "open_ports_count",
    "service_version_outdated", "dns_record_type_count", "is_wildcard_dns",
    "github_leak_count", "exposure_velocity",
    "leak_severity_encoded", "exploit_maturity_encoded",
]

# Add one-hot encoded columns (exclude the original text columns)
source_cols = [c for c in df.columns if c.startswith("source_") and c != "source_type"]
artifact_cols = [c for c in df.columns if c.startswith("artifact_") and c != "artifact_type"]
FEATURES += source_cols + artifact_cols

TARGET = "risk_score"

print(f"   Features: {len(FEATURES)} columns")
print(f"   Target: {TARGET}")

X = df[FEATURES].values
y = df[TARGET].values.astype(int)

# ── Class names ──────────────────────────────────────────────────────────────
class_names = [RISK_LABELS[i] for i in sorted(np.unique(y))]
n_classes = len(class_names)
print(f"   Classes: {class_names} ({n_classes} total)")
print(f"   Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")


# ═══════════════════════════════════════════════════════════════════════════════
#  STAGE 1: UNSUPERVISED — ISOLATION FOREST (Anomaly Detection)
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'━' * 65}")
print("  STAGE 1: Unsupervised — Isolation Forest")
print(f"{'━' * 65}")
print("""
  WHY ISOLATION FOREST?
  • Designed for anomaly detection in high-dimensional data
  • Does not require labeled data — learns normal patterns unsupervised
  • Isolates anomalies by randomly partitioning feature space
  • Anomalous points require fewer splits → lower anomaly score
  • Ideal for OSINT data where "normal" exposure patterns may shift

  Use Case: Detects unusual exposure events (e.g., sudden spike in
  GitHub leaks for one domain, anomalous certificate issuance patterns)
""")

# Scale features for Isolation Forest
scaler_iso = StandardScaler()
X_scaled = scaler_iso.fit_transform(X)

# Train Isolation Forest
iso_forest = IsolationForest(
    n_estimators=200,
    max_samples="auto",
    contamination=0.05,  # expect ~5% anomalies
    random_state=SEED,
    n_jobs=-1,
)

iso_forest.fit(X_scaled)

# Get anomaly scores (convert to 0–1 range, higher = more anomalous)
raw_scores = iso_forest.decision_function(X_scaled)
# decision_function returns negative for anomalies: normalize to [0,1]
anomaly_scores = 1 - (raw_scores - raw_scores.min()) / (raw_scores.max() - raw_scores.min())
anomaly_labels = iso_forest.predict(X_scaled)  # 1 = normal, -1 = anomaly

n_anomalies = (anomaly_labels == -1).sum()
print(f"   Anomalies detected: {n_anomalies:,} ({n_anomalies/len(df)*100:.1f}%)")
print(f"   Anomaly score range: [{anomaly_scores.min():.4f}, {anomaly_scores.max():.4f}]")
print(f"   Anomaly score mean:  {anomaly_scores.mean():.4f}")

# ── Add anomaly scores as a feature for supervised models ────────────────────
df["anomaly_score"] = anomaly_scores
X_with_anomaly = np.column_stack([X, anomaly_scores])
FEATURES_WITH_ANOMALY = FEATURES + ["anomaly_score"]

# Save Isolation Forest
with open(f"{MODELS_DIR}/isolation_forest.pkl", "wb") as f:
    pickle.dump(iso_forest, f)
with open(f"{MODELS_DIR}/scaler_isolation.pkl", "wb") as f:
    pickle.dump(scaler_iso, f)
print(f"   ✅  Isolation Forest saved to {MODELS_DIR}/")


# ═══════════════════════════════════════════════════════════════════════════════
#  STAGE 2: SUPERVISED — Random Forest + XGBoost (Risk Classification)
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'━' * 65}")
print("  STAGE 2: Supervised — Random Forest + XGBoost")
print(f"{'━' * 65}")
print("""
  WHY RANDOM FOREST?
  • Ensemble of decision trees — robust to overfitting
  • Handles mixed feature types (binary, continuous, ordinal)
  • Built-in feature importance ranking
  • Works well with tabular cybersecurity data

  WHY XGBOOST?
  • Gradient boosted trees — sequential error correction
  • State-of-the-art on structured/tabular data
  • Built-in regularization prevents overfitting
  • Handles class imbalance effectively
""")

# ── Train/Test Split (80/20, stratified) ─────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X_with_anomaly, y, test_size=0.20, random_state=SEED, stratify=y
)
print(f"\n   Train: {len(X_train):,}  |  Test: {len(X_test):,}")
print(f"   Train class distribution: {dict(zip(*np.unique(y_train, return_counts=True)))}")
print(f"   Test class distribution:  {dict(zip(*np.unique(y_test, return_counts=True)))}")

# ── Feature Scaling for consistency ──────────────────────────────────────────
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

with open(f"{MODELS_DIR}/scaler_supervised.pkl", "wb") as f:
    pickle.dump(scaler, f)

# ── Define Models ────────────────────────────────────────────────────────────
models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=SEED,
        n_jobs=-1,
        class_weight="balanced",
    ),
    "XGBoost": XGBClassifier(
        n_estimators=300,
        max_depth=10,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=SEED,
        use_label_encoder=False,
        eval_metric="mlogloss",
        verbosity=0,
    ),
}

# ── Train, Evaluate, Save ───────────────────────────────────────────────────
results = {}

for name, model in models.items():
    print(f"\n🔄  Training {name} …")

    # Tree models work fine without scaling
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    # ── Metrics ──────────────────────────────────────────────────────────
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1_w = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    # Cross-validation (5-fold)
    cv_scores = cross_val_score(model, X_train, y_train, cv=5,
                                 scoring="accuracy", n_jobs=-1)

    results[name] = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_weighted": f1_w,
        "cv_mean": cv_scores.mean(),
        "cv_std": cv_scores.std(),
        "y_pred": y_pred,
        "y_prob": y_prob,
    }

    print(f"   Accuracy:  {acc:.4f}")
    print(f"   Precision: {prec:.4f}")
    print(f"   Recall:    {rec:.4f}")
    print(f"   F1-score:  {f1_w:.4f}")
    print(f"   CV:        {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # ── Save model ───────────────────────────────────────────────────────
    safe_name = name.lower().replace(" ", "_")
    with open(f"{MODELS_DIR}/{safe_name}.pkl", "wb") as f:
        pickle.dump(model, f)

    # ── Classification Report ────────────────────────────────────────────
    report = classification_report(y_test, y_pred, target_names=class_names,
                                    zero_division=0)
    with open(f"{RESULTS_DIR}/{safe_name}_report.txt", "w") as f:
        f.write(f"{'=' * 60}\n")
        f.write(f"  Classification Report — {name}\n")
        f.write(f"{'=' * 60}\n\n")
        f.write(report)
        f.write(f"\n\nAccuracy:        {acc:.4f}\n")
        f.write(f"Precision (W):   {prec:.4f}\n")
        f.write(f"Recall (W):      {rec:.4f}\n")
        f.write(f"F1 (Weighted):   {f1_w:.4f}\n")
        f.write(f"CV Accuracy:     {cv_scores.mean():.4f} ± {cv_scores.std():.4f}\n")

    print(f"   ✅  Saved: {safe_name}.pkl, classification report")

    # ── Confusion Matrix ─────────────────────────────────────────────────
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(8, 7))
    disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title(f"Confusion Matrix — {name}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(f"{RESULTS_DIR}/{safe_name}_confusion_matrix.png", dpi=150)
    plt.close()
    print(f"   ✅  Saved: {safe_name}_confusion_matrix.png")


# ═══════════════════════════════════════════════════════════════════════════════
#  ROC CURVES (One-vs-Rest for multiclass)
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n📊  Generating ROC curves (One-vs-Rest) …")
y_test_bin = label_binarize(y_test, classes=sorted(np.unique(y)))

fig, axes = plt.subplots(1, n_classes, figsize=(5 * n_classes, 5))
colors = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6"]

for cls_idx in range(n_classes):
    ax = axes[cls_idx] if n_classes > 1 else axes
    for idx, (name, res) in enumerate(results.items()):
        # Handle case where y_prob may not have all classes
        if res["y_prob"].shape[1] > cls_idx:
            fpr, tpr, _ = roc_curve(y_test_bin[:, cls_idx], res["y_prob"][:, cls_idx])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=colors[idx % len(colors)], lw=2, alpha=0.85,
                    label=f"{name} (AUC={roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4)
    ax.set_title(f"ROC — {class_names[cls_idx]}", fontsize=13, fontweight="bold")
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])

plt.suptitle("ROC Curves (One-vs-Rest)", fontsize=16, fontweight="bold", y=1.03)
plt.tight_layout()
fig.savefig(f"{RESULTS_DIR}/roc_curves.png", dpi=150, bbox_inches="tight")
plt.close()
print("   ✅  roc_curves.png")


# ═══════════════════════════════════════════════════════════════════════════════
#  MODEL COMPARISON BAR CHART
# ═══════════════════════════════════════════════════════════════════════════════
print(f"📊  Generating model comparison chart …")
comp_df = pd.DataFrame({
    "Model": list(results.keys()),
    "Accuracy": [r["accuracy"] for r in results.values()],
    "Precision": [r["precision"] for r in results.values()],
    "Recall": [r["recall"] for r in results.values()],
    "F1 (Weighted)": [r["f1_weighted"] for r in results.values()],
    "CV Accuracy": [r["cv_mean"] for r in results.values()],
}).set_index("Model")

fig, ax = plt.subplots(figsize=(14, 6))
x = np.arange(len(comp_df))
width = 0.15
metrics = ["Accuracy", "Precision", "Recall", "F1 (Weighted)", "CV Accuracy"]
colors_bar = ["#3498db", "#2ecc71", "#e67e22", "#9b59b6", "#e74c3c"]

for i, (metric, color) in enumerate(zip(metrics, colors_bar)):
    bars = ax.bar(x + i * width, comp_df[metric], width, label=metric,
                   color=color, edgecolor="white", linewidth=1)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)

ax.set_xticks(x + width * 2)
ax.set_xticklabels(comp_df.index, fontsize=12)
ax.set_ylabel("Score")
ax.set_ylim(0, 1.15)
ax.set_title("Model Performance Comparison", fontsize=16, fontweight="bold")
ax.legend(fontsize=10, loc="upper right")
plt.tight_layout()
fig.savefig(f"{RESULTS_DIR}/model_comparison.png", dpi=150)
plt.close()
print("   ✅  model_comparison.png")

comp_df["CV Std"] = [r["cv_std"] for r in results.values()]
comp_df.to_csv(f"{RESULTS_DIR}/model_comparison.csv")


# ═══════════════════════════════════════════════════════════════════════════════
#  FEATURE IMPORTANCES (Random Forest)
# ═══════════════════════════════════════════════════════════════════════════════
print(f"📊  Generating feature importance chart …")
rf_model = models["Random Forest"]
importances = rf_model.feature_importances_
feature_names = FEATURES_WITH_ANOMALY
sorted_idx = np.argsort(importances)[-20:]  # top 20

fig, ax = plt.subplots(figsize=(10, 8))
ax.barh([feature_names[i] for i in sorted_idx], importances[sorted_idx],
        color="#3498db", edgecolor="white", linewidth=1)
for i, v in enumerate(importances[sorted_idx]):
    ax.text(v + 0.002, i, f"{v:.3f}", va="center", fontsize=9)
ax.set_title("Top 20 Feature Importances (Random Forest)", fontsize=14, fontweight="bold")
ax.set_xlabel("Importance")
plt.tight_layout()
fig.savefig(f"{RESULTS_DIR}/feature_importances.png", dpi=150)
plt.close()
print("   ✅  feature_importances.png")


# ═══════════════════════════════════════════════════════════════════════════════
#  STAGE 3: TIME-SERIES — Rolling Average + EWMA Forecasting
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'━' * 65}")
print("  STAGE 3: Time-Series — Rolling Average + EWMA")
print(f"{'━' * 65}")
print("""
  WHY TIME-SERIES ANALYSIS?
  • Exposure events are temporal — threats emerge in bursts
  • Rolling averages smooth noise, revealing underlying trends
  • Risk spikes (>2σ above rolling mean) indicate active attack campaigns
  • EWMA (Exponentially Weighted Moving Average) provides adaptive forecasting
  • Enables predictive security — anticipate risk before it peaks

  Use Case: Detect periods of abnormally high exposure (e.g., mass
  credential leak on paste sites, CVE exploit release causing spike)
""")

# ── Aggregate daily risk scores ──────────────────────────────────────────────
ts_df = df[["timestamp", "risk_score"]].copy()
ts_df["date"] = ts_df["timestamp"].dt.date
daily_risk = ts_df.groupby("date").agg(
    mean_risk=("risk_score", "mean"),
    max_risk=("risk_score", "max"),
    event_count=("risk_score", "count"),
).reset_index()
daily_risk["date"] = pd.to_datetime(daily_risk["date"])

# ── Rolling averages ────────────────────────────────────────────────────────
daily_risk["rolling_7d"]  = daily_risk["mean_risk"].rolling(window=7, min_periods=1).mean()
daily_risk["rolling_30d"] = daily_risk["mean_risk"].rolling(window=30, min_periods=1).mean()

# ── EWMA Forecasting ────────────────────────────────────────────────────────
daily_risk["ewma_14d"] = daily_risk["mean_risk"].ewm(span=14, adjust=False).mean()

# ── Risk spike detection (>2σ above rolling 30-day mean) ─────────────────────
rolling_std = daily_risk["mean_risk"].rolling(window=30, min_periods=7).std()
daily_risk["spike_threshold"] = daily_risk["rolling_30d"] + 2 * rolling_std
daily_risk["is_spike"] = (daily_risk["mean_risk"] > daily_risk["spike_threshold"]).astype(int)

n_spikes = daily_risk["is_spike"].sum()
print(f"   Daily risk data: {len(daily_risk)} days")
print(f"   Risk spikes detected: {n_spikes} days ({n_spikes/len(daily_risk)*100:.1f}%)")
print(f"   Mean daily risk: {daily_risk['mean_risk'].mean():.3f}")
print(f"   Max daily risk: {daily_risk['mean_risk'].max():.3f}")

# ── Plot: Time-Series Risk Trend ─────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(16, 14), sharex=True)

# Plot 1: Daily Risk with Rolling Averages
ax1 = axes[0]
ax1.plot(daily_risk["date"], daily_risk["mean_risk"],
         alpha=0.3, color="#3498db", linewidth=0.8, label="Daily Mean Risk")
ax1.plot(daily_risk["date"], daily_risk["rolling_7d"],
         color="#2ecc71", linewidth=2, label="7-Day Rolling Avg")
ax1.plot(daily_risk["date"], daily_risk["rolling_30d"],
         color="#e74c3c", linewidth=2, label="30-Day Rolling Avg")
ax1.plot(daily_risk["date"], daily_risk["ewma_14d"],
         color="#9b59b6", linewidth=2, linestyle="--", label="EWMA (14-day)")
ax1.set_title("Daily Mean Risk Score with Rolling Averages", fontsize=14, fontweight="bold")
ax1.set_ylabel("Mean Risk Score")
ax1.legend(loc="upper right")
ax1.grid(True, alpha=0.3)

# Plot 2: Risk Spike Detection
ax2 = axes[1]
ax2.plot(daily_risk["date"], daily_risk["mean_risk"],
         alpha=0.5, color="#3498db", linewidth=1, label="Daily Mean Risk")
ax2.plot(daily_risk["date"], daily_risk["spike_threshold"],
         color="#e74c3c", linewidth=1.5, linestyle="--", label="Spike Threshold (μ + 2σ)")
spike_dates = daily_risk[daily_risk["is_spike"] == 1]
ax2.scatter(spike_dates["date"], spike_dates["mean_risk"],
            color="#e74c3c", s=50, zorder=5, label=f"Risk Spikes ({n_spikes} days)")
ax2.set_title("Risk Spike Detection", fontsize=14, fontweight="bold")
ax2.set_ylabel("Mean Risk Score")
ax2.legend(loc="upper right")
ax2.grid(True, alpha=0.3)

# Plot 3: Event Volume
ax3 = axes[2]
ax3.bar(daily_risk["date"], daily_risk["event_count"],
        color="#3498db", alpha=0.6, width=1.0)
ax3.set_title("Daily Exposure Event Volume", fontsize=14, fontweight="bold")
ax3.set_ylabel("Number of Events")
ax3.set_xlabel("Date")
ax3.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
ax3.grid(True, alpha=0.3)

plt.suptitle("Time-Series Analysis — Living Attack Surface Mapper",
             fontsize=16, fontweight="bold", y=1.01)
plt.tight_layout()
fig.savefig(f"{RESULTS_DIR}/timeseries_analysis.png", dpi=150, bbox_inches="tight")
plt.close()
print("   ✅  timeseries_analysis.png")

# Save time-series data
daily_risk.to_csv(f"{RESULTS_DIR}/daily_risk_timeseries.csv", index=False)
print("   ✅  daily_risk_timeseries.csv")


# ═══════════════════════════════════════════════════════════════════════════════
#  HYPERPARAMETER TUNING REPORT
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'━' * 65}")
print("  Hyperparameter Configuration Summary")
print(f"{'━' * 65}")

hp_report = """
╔══════════════════════════════════════════════════════════════════╗
║                  HYPERPARAMETER CONFIGURATION                    ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  Isolation Forest                                                ║
║  ├─ n_estimators:   200                                          ║
║  ├─ max_samples:    auto                                         ║
║  ├─ contamination:  0.05 (5% expected anomaly rate)              ║
║  └─ random_state:   42                                           ║
║                                                                  ║
║  Random Forest                                                   ║
║  ├─ n_estimators:   300                                          ║
║  ├─ max_depth:      20                                           ║
║  ├─ min_samples_split: 5                                         ║
║  ├─ min_samples_leaf:  2                                         ║
║  ├─ class_weight:   balanced                                     ║
║  └─ random_state:   42                                           ║
║                                                                  ║
║  XGBoost                                                         ║
║  ├─ n_estimators:   300                                          ║
║  ├─ max_depth:      10                                           ║
║  ├─ learning_rate:  0.1                                          ║
║  ├─ subsample:      0.8                                          ║
║  ├─ colsample_bytree: 0.8                                       ║
║  ├─ reg_alpha:      0.1  (L1 regularization)                     ║
║  ├─ reg_lambda:     1.0  (L2 regularization)                     ║
║  └─ random_state:   42                                           ║
║                                                                  ║
║  Time-Series                                                     ║
║  ├─ Rolling window (short): 7 days                               ║
║  ├─ Rolling window (long):  30 days                              ║
║  ├─ EWMA span:       14 days                                     ║
║  └─ Spike threshold:  μ + 2σ (30-day rolling)                    ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
"""

print(hp_report)

with open(f"{RESULTS_DIR}/hyperparameters.txt", "w") as f:
    f.write(hp_report)


# ═══════════════════════════════════════════════════════════════════════════════
#  SAVE UPDATED DATASET (with anomaly scores)
# ═══════════════════════════════════════════════════════════════════════════════
df.to_csv("preprocessed_dataset_with_anomaly.csv", index=False)
print("   ✅  Updated dataset saved with anomaly scores")


# ═══════════════════════════════════════════════════════════════════════════════
#  FINAL SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'━' * 65}")
print("  FINAL RESULTS SUMMARY")
print(f"{'━' * 65}")

print(f"\n{'Model':<20s} {'Accuracy':>10s} {'Precision':>10s} {'Recall':>10s} "
      f"{'F1':>10s} {'CV Acc':>12s}")
print("─" * 75)

best_acc = 0
best_model = ""
for name, res in results.items():
    print(f"{name:<20s} {res['accuracy']:>10.4f} {res['precision']:>10.4f} "
          f"{res['recall']:>10.4f} {res['f1_weighted']:>10.4f} "
          f"{res['cv_mean']:>8.4f} ± {res['cv_std']:.4f}")
    if res["accuracy"] > best_acc:
        best_acc = res["accuracy"]
        best_model = name

print(f"\n🏆  Best model: {best_model}  (Accuracy: {best_acc:.4f})")
print(f"\n📁  Models saved to:  ./{MODELS_DIR}/")
print(f"📁  Results saved to: ./{RESULTS_DIR}/")
print("🏁  Done.")
