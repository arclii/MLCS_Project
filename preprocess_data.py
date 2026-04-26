#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  Living Attack Surface Mapper — Data Preprocessing Pipeline                ║
║  Input    : attack_surface_dataset.csv                                     ║
║  Output   : preprocessed_dataset.csv                                       ║
║  Author   : K Likhita Reddy — 16010423042 — TY IT A                       ║
╚══════════════════════════════════════════════════════════════════════════════╝

Complete preprocessing pipeline:
    1. Missing value handling
    2. Duplicate removal
    3. Categorical encoding (Label + One-Hot)
    4. Feature scaling (StandardScaler)
    5. Outlier detection and flagging (IQR + Z-score)

Each step includes a printed justification explaining WHY it is necessary.
"""

import os
import warnings
import pickle

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder

warnings.filterwarnings("ignore")

MODELS_DIR = "saved_models"
os.makedirs(MODELS_DIR, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  LOAD RAW DATA
# ═══════════════════════════════════════════════════════════════════════════════
print("═" * 65)
print("  Living Attack Surface Mapper — Preprocessing Pipeline")
print("═" * 65)

df = pd.read_csv("attack_surface_dataset.csv")
print(f"\n📂  Loaded {len(df):,} samples, {len(df.columns)} columns")
print(f"   Columns: {list(df.columns)}")
print(f"\n   Initial data types:")
print(df.dtypes.to_string())


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 1: HANDLE MISSING VALUES
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'─' * 65}")
print("  STEP 1: Handling Missing Values")
print(f"{'─' * 65}")
print("""
  WHY: Missing values cause errors in ML models and skew statistics.
  Real-world OSINT data has gaps due to incomplete API responses,
  unavailable feeds, or partial scan results. Proper imputation
  preserves dataset integrity without introducing bias.

  Strategy:
  • Numeric columns → Median imputation (robust to outliers)
  • Categorical columns → Mode imputation (most frequent value)
  • Timestamp → Forward-fill (maintains temporal continuity)
""")

# Report missing values before handling
missing_before = df.isnull().sum()
missing_cols = missing_before[missing_before > 0]
print(f"   Missing values BEFORE imputation:")
if len(missing_cols) > 0:
    for col, count in missing_cols.items():
        print(f"      {col:30s}  {count:>4} ({count/len(df)*100:.2f}%)")
else:
    print("      None found")

# ── Impute numeric columns with median ───────────────────────────────────────
numeric_cols = df.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    if df[col].isnull().sum() > 0:
        median_val = df[col].median()
        df[col].fillna(median_val, inplace=True)
        print(f"   ✓ {col}: filled {missing_before[col]} NaN with median = {median_val:.2f}")

# ── Impute categorical columns with mode ─────────────────────────────────────
cat_cols = df.select_dtypes(include=["object"]).columns
for col in cat_cols:
    if df[col].isnull().sum() > 0:
        mode_val = df[col].mode()[0]
        df[col].fillna(mode_val, inplace=True)
        print(f"   ✓ {col}: filled {missing_before[col]} NaN with mode = '{mode_val}'")

# ── Forward-fill timestamp gaps ──────────────────────────────────────────────
if "timestamp" in df.columns and df["timestamp"].isnull().sum() > 0:
    df["timestamp"].fillna(method="ffill", inplace=True)
    print(f"   ✓ timestamp: forward-filled {missing_before['timestamp']} NaN")

# Verify
remaining = df.isnull().sum().sum()
print(f"\n   Missing values AFTER imputation: {remaining}")
assert remaining == 0, "ERROR: Missing values still present!"
print("   ✅  All missing values handled successfully")


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 2: REMOVE DUPLICATES
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'─' * 65}")
print("  STEP 2: Removing Duplicate Records")
print(f"{'─' * 65}")
print("""
  WHY: Duplicate records inflate class frequencies and bias model
  training. In OSINT data, the same exposure event may be captured
  by multiple scans or overlapping monitoring tools. Deduplication
  ensures each unique exposure event is counted exactly once.
""")

n_before = len(df)
n_dupes = df.duplicated().sum()
df = df.drop_duplicates(keep="first").reset_index(drop=True)
n_after = len(df)

print(f"   Records before: {n_before:,}")
print(f"   Duplicates found: {n_dupes:,}")
print(f"   Records after:  {n_after:,}")
print(f"   ✅  Removed {n_before - n_after:,} duplicate rows")


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 3: ENCODING CATEGORICAL FEATURES
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'─' * 65}")
print("  STEP 3: Encoding Categorical Features")
print(f"{'─' * 65}")
print("""
  WHY: Machine learning algorithms operate on numerical data.
  Categorical features must be converted:
  • Label Encoding for ordinal features (leak_severity, exploit_maturity)
    where order matters
  • One-Hot Encoding for nominal features (source_type, artifact_type)
    where no ordinal relationship exists

  This preserves the semantic meaning of each categorical variable.
""")

# ── Label Encoding for ordinal features ──────────────────────────────────────

# leak_severity: Low < Medium < High < Critical (ordinal)
severity_order = {"Low": 0, "Medium": 1, "High": 2, "Critical": 3}
df["leak_severity_encoded"] = df["leak_severity"].map(severity_order)
print(f"   ✓ leak_severity → label-encoded: {severity_order}")

# exploit_maturity: None < POC < Functional < Weaponized (ordinal)
maturity_order = {"None": 0, "POC": 1, "Functional": 2, "Weaponized": 3}
df["exploit_maturity_encoded"] = df["exploit_maturity"].map(maturity_order)
print(f"   ✓ exploit_maturity → label-encoded: {maturity_order}")

# ── One-Hot Encoding for nominal features ────────────────────────────────────

# source_type: 5 categories, no ordinal relationship
source_dummies = pd.get_dummies(df["source_type"], prefix="source", dtype=int)
df = pd.concat([df, source_dummies], axis=1)
print(f"   ✓ source_type → one-hot encoded: {list(source_dummies.columns)}")

# artifact_type: multiple categories, no ordinal relationship
artifact_dummies = pd.get_dummies(df["artifact_type"], prefix="artifact", dtype=int)
df = pd.concat([df, artifact_dummies], axis=1)
print(f"   ✓ artifact_type → one-hot encoded: {list(artifact_dummies.columns)}")

# ── Encode domain as label (for potential use; not a primary feature) ────────
le_domain = LabelEncoder()
df["domain_encoded"] = le_domain.fit_transform(df["domain"])
print(f"   ✓ domain → label-encoded ({len(le_domain.classes_)} unique domains)")

# Save encoders
with open(f"{MODELS_DIR}/label_encoder_domain.pkl", "wb") as f:
    pickle.dump(le_domain, f)
print(f"   ✅  Encoders saved to {MODELS_DIR}/")


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 4: FEATURE SCALING
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'─' * 65}")
print("  STEP 4: Feature Scaling (Standardization)")
print(f"{'─' * 65}")
print("""
  WHY: Features have vastly different scales:
  • cvss_base_score: 0–10
  • open_ports_count: 0–65535
  • domain_reputation_score: 0–1
  • cve_age_days: 1–7300

  Without scaling, models like Isolation Forest give disproportionate
  weight to high-magnitude features. StandardScaler (z-score) centers
  each feature at mean=0 and std=1, ensuring equal contribution.
""")

# Define features to scale (all numeric except target and encoded categoricals)
scale_features = [
    "subdomain_count", "subdomain_age_days", "domain_reputation_score",
    "cvss_base_score", "cve_age_days", "exposure_frequency",
    "paste_site_mentions", "ssl_days_remaining", "open_ports_count",
    "dns_record_type_count", "github_leak_count", "exposure_velocity",
    "leak_severity_encoded", "exploit_maturity_encoded",
]

scaler = StandardScaler()
df_scaled = df.copy()
df_scaled[scale_features] = scaler.fit_transform(df[scale_features])

print(f"   Scaled {len(scale_features)} features:")
for feat in scale_features:
    print(f"      {feat:35s}  mean={df_scaled[feat].mean():7.4f}  std={df_scaled[feat].std():7.4f}")

# Save scaler
with open(f"{MODELS_DIR}/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
print(f"\n   ✅  StandardScaler saved to {MODELS_DIR}/scaler.pkl")


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 5: OUTLIER DETECTION
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'─' * 65}")
print("  STEP 5: Outlier Detection (IQR + Z-Score)")
print(f"{'─' * 65}")
print("""
  WHY: Outliers can represent:
  • Genuine anomalies (e.g., zero-day exploitation events)
  • Data errors (e.g., impossible port numbers)

  We use two methods:
  • IQR (Interquartile Range): Flags values below Q1-1.5*IQR or above Q3+1.5*IQR
  • Z-Score: Flags values with |z| > 3 (>3 standard deviations from mean)

  Outliers are FLAGGED, not removed — in cybersecurity, they may
  represent the most critical events.
""")

outlier_features = [
    "subdomain_count", "subdomain_age_days", "cvss_base_score",
    "exposure_frequency", "paste_site_mentions", "open_ports_count",
    "github_leak_count", "exposure_velocity", "cve_age_days",
]

total_iqr_outliers = 0
total_zscore_outliers = 0

for feat in outlier_features:
    # IQR method
    Q1 = df[feat].quantile(0.25)
    Q3 = df[feat].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    iqr_outliers = ((df[feat] < lower) | (df[feat] > upper)).sum()

    # Z-score method
    z_scores = np.abs((df[feat] - df[feat].mean()) / df[feat].std())
    zscore_outliers = (z_scores > 3).sum()

    total_iqr_outliers += iqr_outliers
    total_zscore_outliers += zscore_outliers

    if iqr_outliers > 0 or zscore_outliers > 0:
        print(f"   {feat:30s}  IQR={iqr_outliers:>5}  Z-score={zscore_outliers:>5}  "
              f"range=[{df[feat].min():.1f}, {df[feat].max():.1f}]")

# Add a combined outlier flag column
df["is_outlier"] = 0
for feat in outlier_features:
    Q1 = df[feat].quantile(0.25)
    Q3 = df[feat].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df.loc[(df[feat] < lower) | (df[feat] > upper), "is_outlier"] = 1

n_outlier_rows = df["is_outlier"].sum()
print(f"\n   Total rows flagged as outliers: {n_outlier_rows:,} ({n_outlier_rows/len(df)*100:.1f}%)")
print(f"   ✅  Outliers flagged (not removed — may be genuine security events)")


# ═══════════════════════════════════════════════════════════════════════════════
#  SAVE PREPROCESSED DATASET
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'─' * 65}")
print("  SAVING PREPROCESSED DATASET")
print(f"{'─' * 65}")

# Save the unscaled version with all encodings (for model training)
output_path = "preprocessed_dataset.csv"
df.to_csv(output_path, index=False)

print(f"\n   ✅  Preprocessed dataset saved to: {output_path}")
print(f"   Total samples : {len(df):,}")
print(f"   Total columns : {len(df.columns)}")
print(f"   New columns added:")
print(f"      • leak_severity_encoded")
print(f"      • exploit_maturity_encoded")
print(f"      • domain_encoded")
print(f"      • is_outlier")
print(f"      • source_* one-hot ({len(source_dummies.columns)} cols)")
print(f"      • artifact_* one-hot ({len(artifact_dummies.columns)} cols)")

print(f"\n   Column listing:")
for i, col in enumerate(df.columns):
    dtype = df[col].dtype
    print(f"      {i+1:2d}. {col:40s}  {str(dtype):>10s}")

# ── Summary statistics ───────────────────────────────────────────────────────
print(f"\n   Preprocessed Data Preview (first 5 rows):")
print(df.head().to_string(index=False, max_colwidth=15))

print(f"\n{'═' * 65}")
print("  PREPROCESSING SUMMARY")
print(f"{'═' * 65}")
print(f"""
  ┌──────────────────────────────────┬────────────────────────┐
  │ Step                             │ Result                 │
  ├──────────────────────────────────┼────────────────────────┤
  │ Missing values imputed           │ {missing_before.sum():>5} → 0              │
  │ Duplicates removed               │ {n_before - n_after:>5} rows               │
  │ Ordinal features encoded         │ 2 (severity, maturity) │
  │ Nominal features one-hot encoded │ 2 (source, artifact)   │
  │ Features scaled                  │ {len(scale_features):>5} columns           │
  │ Outlier rows flagged             │ {n_outlier_rows:>5} rows               │
  │ Final dataset size               │ {len(df):>5} × {len(df.columns):<3}           │
  └──────────────────────────────────┴────────────────────────┘
""")
print("🏁  Preprocessing complete.")
