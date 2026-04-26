#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  Living Attack Surface Mapper — Synthetic OSINT Dataset Generator          ║
║  Output   : attack_surface_dataset.csv  (20,000+ exposure events)         ║
║  Author   : K Likhita Reddy — 16010423042 — TY IT A                       ║
╚══════════════════════════════════════════════════════════════════════════════╝

Generates realistic OSINT-based exposure events from five source types:
    GitHub Leaks · DNS Updates · CVE/NVD Feeds · CT Logs · Paste Sites

25 features per row, including the target variable 'risk_score' (0–4).
All feature values are logically correlated to mirror real-world cybersecurity
exposure patterns.
"""

import random
import math
import datetime
import warnings
from collections import Counter

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ── Configuration ────────────────────────────────────────────────────────────
TOTAL_SAMPLES = 20_000
START_DATE = datetime.datetime(2025, 4, 1)
END_DATE   = datetime.datetime(2026, 3, 31)
DATE_RANGE_DAYS = (END_DATE - START_DATE).days

# ── OSINT Source Types (from PPT) ────────────────────────────────────────────
SOURCE_TYPES = ["GitHub", "DNS", "CVE", "CT_Log", "Paste"]
SOURCE_WEIGHTS = [0.25, 0.20, 0.25, 0.15, 0.15]

# ── Artifact Types per Source ────────────────────────────────────────────────
ARTIFACT_MAP = {
    "GitHub":  ["API_Key", "Password", "Config", "Source_Code", "Secret_Token"],
    "DNS":     ["Subdomain", "DNS_Record", "Zone_Transfer", "Wildcard_Entry"],
    "CVE":     ["CVE_Entry", "Vulnerability", "Exploit_Code"],
    "CT_Log":  ["Certificate", "Cert_Anomaly", "Wildcard_Cert"],
    "Paste":   ["Data_Dump", "Credential_Leak", "Config_Dump", "PII_Leak"],
}

# ── Realistic Domains ────────────────────────────────────────────────────────
TARGET_DOMAINS = [
    "acmecorp.com", "globexinc.com", "initech.io", "hooli.com",
    "piedpiper.net", "wayneenterprises.com", "starkindustries.io",
    "umbrellacorp.org", "cyberdyne.systems", "oscorptech.com",
    "lexcorp.com", "aperture-science.net", "blackmesa.org",
    "soylentgreen.co", "nakatomicorp.com", "tyrell-corp.io",
    "weyland-yutani.com", "multivac.systems", "genisys.ai",
    "skynet-defense.io", "kaijuglobal.com", "atlas-corp.org",
    "nexustech.io", "quantumleap.dev", "orbitalx.space",
]

# ── Exploit Maturity Levels ──────────────────────────────────────────────────
EXPLOIT_MATURITY_LEVELS = ["None", "POC", "Functional", "Weaponized"]

# ── Leak Severity Levels ─────────────────────────────────────────────────────
LEAK_SEVERITY_LEVELS = ["Low", "Medium", "High", "Critical"]


# ═══════════════════════════════════════════════════════════════════════════════
#  FEATURE GENERATION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def generate_timestamp() -> datetime.datetime:
    """Generate a timestamp within the 12-month range with realistic patterns.
    Adds weekly seasonality — more events on weekdays (when attackers are active)."""
    day_offset = random.randint(0, DATE_RANGE_DAYS)
    hour = int(np.random.choice(
        range(24),
        p=_hour_distribution()
    ))
    minute = random.randint(0, 59)
    second = random.randint(0, 59)
    ts = START_DATE + datetime.timedelta(days=day_offset, hours=hour,
                                          minutes=minute, seconds=second)
    return ts


def _hour_distribution():
    """More events during business hours (8–18) and late night (22–3) for attackers."""
    probs = np.array([
        0.04, 0.05, 0.05, 0.04,  # 00–03 (some late-night activity)
        0.02, 0.02, 0.02, 0.02,  # 04–07 (quiet)
        0.05, 0.06, 0.07, 0.07,  # 08–11 (business hours ramp up)
        0.06, 0.06, 0.06, 0.06,  # 12–15 (peak business)
        0.05, 0.05, 0.04, 0.03,  # 16–19 (winding down)
        0.03, 0.03, 0.04, 0.04,  # 20–23 (evening/attacker activity)
    ])
    return probs / probs.sum()


def generate_source_and_artifact():
    """Generate correlated source type and artifact type."""
    source = np.random.choice(SOURCE_TYPES, p=SOURCE_WEIGHTS)
    artifact = random.choice(ARTIFACT_MAP[source])
    return source, artifact


def generate_domain():
    """Pick a target domain with non-uniform distribution (some domains more targeted)."""
    # First 10 domains are "high-value" targets — 60% of events
    if random.random() < 0.60:
        return random.choice(TARGET_DOMAINS[:10])
    else:
        return random.choice(TARGET_DOMAINS[10:])


def generate_cvss(source: str, artifact: str) -> float:
    """Generate CVSS score correlated with source/artifact type."""
    if source == "CVE":
        # CVE entries inherently have real CVSS scores — skew higher
        base = np.random.beta(5, 3) * 10  # skewed toward 5–9
    elif artifact in ("API_Key", "Password", "Secret_Token", "Credential_Leak"):
        # Credential leaks are high severity
        base = np.random.beta(6, 2) * 10  # skewed toward 7–10
    elif artifact in ("Subdomain", "DNS_Record", "Certificate"):
        # Recon artifacts — lower base severity
        base = np.random.beta(2, 5) * 10  # skewed toward 1–4
    else:
        base = np.random.beta(3, 3) * 10  # moderate spread

    # Add small noise
    base += np.random.normal(0, 0.3)
    return round(np.clip(base, 0.0, 10.0), 1)


def generate_exploit_fields(cvss: float, source: str):
    """Generate exploit availability and maturity correlated with CVSS."""
    if source == "CVE":
        # Higher CVSS → higher exploit probability
        exploit_prob = min(0.9, cvss / 12.0 + 0.1)
        exploit_available = 1 if random.random() < exploit_prob else 0
    else:
        # Non-CVE sources — lower exploit correlation
        exploit_available = 1 if random.random() < 0.15 else 0

    if exploit_available:
        if cvss >= 9.0:
            maturity = np.random.choice(EXPLOIT_MATURITY_LEVELS[1:],
                                         p=[0.15, 0.35, 0.50])
        elif cvss >= 7.0:
            maturity = np.random.choice(EXPLOIT_MATURITY_LEVELS[1:],
                                         p=[0.30, 0.45, 0.25])
        else:
            maturity = np.random.choice(EXPLOIT_MATURITY_LEVELS[1:],
                                         p=[0.60, 0.30, 0.10])
    else:
        maturity = "None"

    return exploit_available, maturity


def generate_subdomain_fields(source: str):
    """Generate subdomain count and age."""
    if source == "DNS":
        count = int(np.random.lognormal(mean=2.5, sigma=0.8))
        count = min(count, 500)
    else:
        count = int(np.random.lognormal(mean=1.5, sigma=0.6))
        count = min(count, 200)

    # Age: older subdomains for DNS, mixed for others
    if source == "DNS":
        age = int(np.random.exponential(scale=365))
    else:
        age = int(np.random.exponential(scale=180))

    return max(1, count), max(1, min(age, 3650))


def generate_domain_reputation(source: str, cvss: float) -> float:
    """Domain reputation: 1.0 = excellent, 0.0 = terrible.
    Inversely correlated with severity."""
    if cvss >= 8.0:
        rep = np.random.beta(2, 6)  # low reputation
    elif cvss >= 5.0:
        rep = np.random.beta(4, 4)  # moderate
    else:
        rep = np.random.beta(6, 2)  # higher reputation

    # Add noise
    rep += np.random.normal(0, 0.05)
    return round(np.clip(rep, 0.0, 1.0), 3)


def generate_exposure_frequency(source: str) -> int:
    """How many times this artifact has been seen across sources."""
    if source == "Paste":
        # Paste site leaks spread rapidly
        return max(1, int(np.random.lognormal(mean=1.5, sigma=1.0)))
    elif source == "GitHub":
        return max(1, int(np.random.lognormal(mean=1.0, sigma=0.8)))
    else:
        return max(1, int(np.random.lognormal(mean=0.5, sigma=0.5)))


def generate_leak_severity(artifact: str, cvss: float) -> str:
    """Leak severity correlated with artifact type and CVSS."""
    if artifact in ("Password", "Credential_Leak", "Secret_Token"):
        weights = [0.05, 0.15, 0.35, 0.45]  # skew critical
    elif artifact in ("API_Key", "Config", "Config_Dump"):
        weights = [0.10, 0.25, 0.40, 0.25]
    elif artifact in ("Data_Dump", "PII_Leak"):
        weights = [0.05, 0.20, 0.30, 0.45]
    elif cvss >= 7.0:
        weights = [0.05, 0.20, 0.35, 0.40]
    else:
        weights = [0.30, 0.35, 0.25, 0.10]  # more Low/Medium

    return np.random.choice(LEAK_SEVERITY_LEVELS, p=weights)


def generate_has_credentials(artifact: str, source: str) -> int:
    """Whether the finding contains credentials."""
    if artifact in ("Password", "Credential_Leak", "Secret_Token"):
        return 1 if random.random() < 0.92 else 0
    elif artifact in ("API_Key", "Config", "Config_Dump"):
        return 1 if random.random() < 0.45 else 0
    elif source == "Paste":
        return 1 if random.random() < 0.35 else 0
    else:
        return 1 if random.random() < 0.08 else 0


def generate_paste_mentions(source: str, has_creds: int) -> int:
    """Number of paste site mentions."""
    if source == "Paste":
        base = int(np.random.lognormal(mean=1.5, sigma=0.8))
    elif has_creds:
        base = int(np.random.lognormal(mean=0.8, sigma=0.6))
    else:
        base = int(np.random.exponential(scale=0.5))
    return max(0, min(base, 100))


def generate_ct_fields(source: str, artifact: str):
    """CT log anomaly and SSL days remaining."""
    if source == "CT_Log":
        ct_anomaly = 1 if random.random() < 0.35 else 0
        ssl_days = int(np.random.exponential(scale=120))
    elif artifact == "Certificate":
        ct_anomaly = 1 if random.random() < 0.25 else 0
        ssl_days = int(np.random.exponential(scale=150))
    else:
        ct_anomaly = 1 if random.random() < 0.05 else 0
        ssl_days = int(np.random.normal(loc=200, scale=80))

    ssl_days = max(-30, min(ssl_days, 730))  # allow negative = expired
    return ct_anomaly, ssl_days


def generate_network_fields(source: str):
    """Open ports count and service version outdated flag."""
    if source == "DNS":
        open_ports = int(np.random.lognormal(mean=2.0, sigma=0.6))
    else:
        open_ports = int(np.random.lognormal(mean=1.2, sigma=0.5))
    open_ports = max(0, min(open_ports, 65535))

    # More open ports → more likely to have outdated services
    outdated_prob = min(0.8, open_ports / 30.0 + 0.1)
    service_outdated = 1 if random.random() < outdated_prob else 0

    return open_ports, service_outdated


def generate_dns_fields(source: str):
    """DNS record type count and wildcard DNS flag."""
    if source == "DNS":
        dns_count = random.randint(3, 15)
        wildcard = 1 if random.random() < 0.25 else 0
    else:
        dns_count = random.randint(1, 8)
        wildcard = 1 if random.random() < 0.08 else 0

    return dns_count, wildcard


def generate_github_leak_count(source: str, artifact: str) -> int:
    """Number of GitHub leaks found for this domain."""
    if source == "GitHub":
        return max(1, int(np.random.lognormal(mean=1.5, sigma=1.0)))
    elif artifact in ("API_Key", "Secret_Token", "Source_Code"):
        return max(0, int(np.random.lognormal(mean=0.8, sigma=0.6)))
    else:
        return max(0, int(np.random.exponential(scale=0.5)))


def generate_cve_age(source: str) -> int:
    """Days since CVE was published."""
    if source == "CVE":
        age = int(np.random.lognormal(mean=4.5, sigma=1.2))
    else:
        age = int(np.random.lognormal(mean=5.0, sigma=1.5))
    return max(1, min(age, 7300))  # max ~20 years


def generate_exposure_velocity(exposure_freq: int, source: str) -> float:
    """Rate of new exposures per week — derived from frequency."""
    base = exposure_freq * np.random.uniform(0.3, 1.5)
    if source in ("Paste", "GitHub"):
        base *= np.random.uniform(1.2, 2.5)  # faster spread
    noise = np.random.normal(0, 0.5)
    return round(max(0.0, base + noise), 2)


# ═══════════════════════════════════════════════════════════════════════════════
#  RISK SCORE COMPUTATION (TARGET VARIABLE)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_risk_score(row: dict) -> int:
    """
    Compute the target risk score (0–4) based on multiple weighted factors.
    This creates realistic, correlated labels.

    Risk levels:
        0 = Low       — minimal exposure, low severity
        1 = Medium    — moderate indicators
        2 = High      — significant exposure or vulnerabilities
        3 = Critical  — active exploit + high CVSS + credentials
        4 = Emergency — weaponized exploit + critical leak + high velocity

    The formula uses a weighted combination of key features:
    """
    score = 0.0

    # CVSS contribution (0–10 → 0–3.0 weight)
    score += (row["cvss_base_score"] / 10.0) * 3.0

    # Exploit availability (strong signal)
    if row["exploit_availability"] == 1:
        score += 1.5
        if row["exploit_maturity"] == "Weaponized":
            score += 2.0
        elif row["exploit_maturity"] == "Functional":
            score += 1.0
        elif row["exploit_maturity"] == "POC":
            score += 0.5

    # Credential presence (direct account takeover risk)
    if row["has_credentials"] == 1:
        score += 1.5

    # Leak severity
    severity_map = {"Low": 0.2, "Medium": 0.6, "High": 1.2, "Critical": 2.0}
    score += severity_map.get(row["leak_severity"], 0)

    # Domain reputation (inverted — low rep = higher risk)
    score += (1.0 - row["domain_reputation_score"]) * 1.5

    # Exposure frequency (more exposure = more risk)
    score += min(1.5, row["exposure_frequency"] / 10.0)

    # Exposure velocity (accelerating = danger)
    score += min(1.0, row["exposure_velocity"] / 8.0)

    # CT anomaly
    if row["ct_log_anomaly"] == 1:
        score += 0.8

    # Expired or soon-to-expire SSL
    if row["ssl_days_remaining"] <= 0:
        score += 1.0
    elif row["ssl_days_remaining"] <= 30:
        score += 0.5

    # Service outdated
    if row["service_version_outdated"] == 1:
        score += 0.6

    # Open ports (more = more surface)
    score += min(0.8, row["open_ports_count"] / 50.0)

    # GitHub leaks
    score += min(0.8, row["github_leak_count"] / 10.0)

    # Paste site mentions
    score += min(0.6, row["paste_site_mentions"] / 15.0)

    # CVE age penalty (old unpatched = negligent)
    if row["cve_age_days"] > 365:
        score += 0.5
    elif row["cve_age_days"] > 180:
        score += 0.3

    # ── Add noise (5–8%) ─────────────────────────────────────────────────────
    noise = np.random.normal(0, 0.6)
    score += noise

    # ── Map to 0–4 risk levels ───────────────────────────────────────────────
    if score < 3.5:
        return 0  # Low
    elif score < 6.0:
        return 1  # Medium
    elif score < 8.5:
        return 2  # High
    elif score < 11.0:
        return 3  # Critical
    else:
        return 4  # Emergency


# ═══════════════════════════════════════════════════════════════════════════════
#  ANOMALY INJECTION
# ═══════════════════════════════════════════════════════════════════════════════

def inject_anomalies(df: pd.DataFrame, anomaly_rate: float = 0.025) -> pd.DataFrame:
    """Inject ~2.5% anomalous records that deviate from normal patterns.
    These represent unusual events (e.g., low CVSS but weaponized exploit,
    unknown domain with massive paste exposure)."""
    n_anomalies = int(len(df) * anomaly_rate)
    anomaly_indices = np.random.choice(df.index, size=n_anomalies, replace=False)

    for idx in anomaly_indices:
        anomaly_type = random.choice(["high_port_low_cvss", "weaponized_low_cvss",
                                       "massive_paste_unknown", "expired_ssl_high_rep",
                                       "zero_day_pattern"])

        if anomaly_type == "high_port_low_cvss":
            df.at[idx, "open_ports_count"] = random.randint(100, 500)
            df.at[idx, "cvss_base_score"] = round(random.uniform(0.5, 2.0), 1)

        elif anomaly_type == "weaponized_low_cvss":
            df.at[idx, "exploit_availability"] = 1
            df.at[idx, "exploit_maturity"] = "Weaponized"
            df.at[idx, "cvss_base_score"] = round(random.uniform(1.0, 3.0), 1)

        elif anomaly_type == "massive_paste_unknown":
            df.at[idx, "paste_site_mentions"] = random.randint(50, 100)
            df.at[idx, "domain_reputation_score"] = round(random.uniform(0.0, 0.15), 3)

        elif anomaly_type == "expired_ssl_high_rep":
            df.at[idx, "ssl_days_remaining"] = random.randint(-30, -1)
            df.at[idx, "domain_reputation_score"] = round(random.uniform(0.85, 0.99), 3)

        elif anomaly_type == "zero_day_pattern":
            df.at[idx, "cvss_base_score"] = round(random.uniform(9.0, 10.0), 1)
            df.at[idx, "exploit_maturity"] = "Weaponized"
            df.at[idx, "exploit_availability"] = 1
            df.at[idx, "cve_age_days"] = random.randint(1, 7)

    print(f"   ▸ Injected {n_anomalies} anomalous records ({anomaly_rate*100:.1f}%)")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
#  MISSING VALUE INJECTION (for preprocessing demo)
# ═══════════════════════════════════════════════════════════════════════════════

def inject_missing_values(df: pd.DataFrame, missing_rate: float = 0.02) -> pd.DataFrame:
    """Inject ~2% missing values across numeric/categorical columns.
    This gives the preprocessing pipeline real missing data to handle."""
    injectable_cols = [
        "subdomain_age_days", "domain_reputation_score", "cvss_base_score",
        "cve_age_days", "ssl_days_remaining", "open_ports_count",
        "paste_site_mentions", "exposure_velocity"
    ]
    total_injected = 0
    for col in injectable_cols:
        n_missing = int(len(df) * missing_rate / len(injectable_cols))
        missing_indices = np.random.choice(df.index, size=n_missing, replace=False)
        df.loc[missing_indices, col] = np.nan
        total_injected += n_missing

    print(f"   ▸ Injected {total_injected} missing values (~{missing_rate*100:.1f}%)")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN GENERATION PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("═" * 65)
    print("  Living Attack Surface Mapper — Dataset Generator")
    print("═" * 65)
    print(f"\n🔧  Generating {TOTAL_SAMPLES:,} OSINT exposure events …\n")

    records = []

    for i in range(TOTAL_SAMPLES):
        # ── Core identifiers ─────────────────────────────────────────────
        source, artifact = generate_source_and_artifact()
        domain = generate_domain()
        timestamp = generate_timestamp()

        # ── Vulnerability features ───────────────────────────────────────
        cvss = generate_cvss(source, artifact)
        exploit_avail, exploit_mat = generate_exploit_fields(cvss, source)
        cve_age = generate_cve_age(source)

        # ── Domain/subdomain features ────────────────────────────────────
        sub_count, sub_age = generate_subdomain_fields(source)
        domain_rep = generate_domain_reputation(source, cvss)

        # ── Exposure/leak features ───────────────────────────────────────
        exposure_freq = generate_exposure_frequency(source)
        leak_sev = generate_leak_severity(artifact, cvss)
        has_creds = generate_has_credentials(artifact, source)
        paste_mentions = generate_paste_mentions(source, has_creds)

        # ── Infrastructure features ──────────────────────────────────────
        ct_anomaly, ssl_days = generate_ct_fields(source, artifact)
        open_ports, svc_outdated = generate_network_fields(source)
        dns_count, wildcard = generate_dns_fields(source)

        # ── GitHub features ──────────────────────────────────────────────
        github_leaks = generate_github_leak_count(source, artifact)

        # ── Derived features ─────────────────────────────────────────────
        exposure_vel = generate_exposure_velocity(exposure_freq, source)

        # ── Build row ────────────────────────────────────────────────────
        row = {
            "timestamp":                timestamp,
            "source_type":              source,
            "artifact_type":            artifact,
            "domain":                   domain,
            "subdomain_count":          sub_count,
            "subdomain_age_days":       sub_age,
            "domain_reputation_score":  domain_rep,
            "cvss_base_score":          cvss,
            "exploit_availability":     exploit_avail,
            "exploit_maturity":         exploit_mat,
            "cve_age_days":             cve_age,
            "exposure_frequency":       exposure_freq,
            "leak_severity":            leak_sev,
            "has_credentials":          has_creds,
            "paste_site_mentions":      paste_mentions,
            "ct_log_anomaly":           ct_anomaly,
            "ssl_days_remaining":       ssl_days,
            "open_ports_count":         open_ports,
            "service_version_outdated": svc_outdated,
            "dns_record_type_count":    dns_count,
            "is_wildcard_dns":          wildcard,
            "github_leak_count":        github_leaks,
            "anomaly_score":            0.0,  # placeholder — filled by Isolation Forest
            "exposure_velocity":        exposure_vel,
        }

        # ── Compute risk score (target) ──────────────────────────────────
        row["risk_score"] = compute_risk_score(row)
        records.append(row)

        if (i + 1) % 5000 == 0:
            print(f"   ▸ Generated {i+1:,} / {TOTAL_SAMPLES:,} records")

    # ── Build DataFrame ──────────────────────────────────────────────────────
    df = pd.DataFrame(records)

    # ── Sort by timestamp ────────────────────────────────────────────────────
    df = df.sort_values("timestamp").reset_index(drop=True)

    # ── Inject anomalies (2.5%) ──────────────────────────────────────────────
    df = inject_anomalies(df, anomaly_rate=0.025)

    # ── Inject missing values (2%) for preprocessing demo ────────────────────
    df = inject_missing_values(df, missing_rate=0.02)

    # ── Inject ~1% duplicate rows (for dedup demo) ───────────────────────────
    n_dupes = int(len(df) * 0.01)
    dupe_rows = df.sample(n=n_dupes, random_state=SEED)
    df = pd.concat([df, dupe_rows], ignore_index=True)
    print(f"   ▸ Injected {n_dupes} duplicate rows (~1%)")

    # ── Shuffle ──────────────────────────────────────────────────────────────
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

    # ── Save ─────────────────────────────────────────────────────────────────
    output_path = "attack_surface_dataset.csv"
    df.to_csv(output_path, index=False)

    # ══════════════════════════════════════════════════════════════════════════
    #  SUMMARY STATISTICS
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n✅  Dataset saved to: {output_path}")
    print(f"   Total samples : {len(df):,}")
    print(f"   Columns       : {len(df.columns)}")
    print(f"   Date range    : {df['timestamp'].min()} → {df['timestamp'].max()}")

    print(f"\n   Risk Score distribution:")
    risk_labels = {0: "Low", 1: "Medium", 2: "High", 3: "Critical", 4: "Emergency"}
    for score, count in df["risk_score"].value_counts().sort_index().items():
        label = risk_labels.get(score, "Unknown")
        print(f"      {score} ({label:10s})  {count:>6,}  ({count/len(df)*100:.1f}%)")

    print(f"\n   Source Type distribution:")
    for src, count in df["source_type"].value_counts().items():
        print(f"      {src:10s}  {count:>6,}  ({count/len(df)*100:.1f}%)")

    print(f"\n   Missing values per column:")
    missing = df.isnull().sum()
    for col in missing[missing > 0].index:
        print(f"      {col:30s}  {missing[col]:>4} missing")
    if missing.sum() == 0:
        print("      None")

    print(f"\n   Duplicate rows: {df.duplicated().sum()}")

    # ── Preview ──────────────────────────────────────────────────────────────
    print(f"\n   Dataset Preview (first 10 rows):")
    print(df.head(10).to_string(index=False, max_colwidth=20))

    print(f"\n   Feature Statistics:")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        print(f"      {col:30s}  mean={df[col].mean():8.2f}  "
              f"std={df[col].std():8.2f}  "
              f"min={df[col].min():8.1f}  max={df[col].max():8.1f}")

    print("\n🏁  Done.")


if __name__ == "__main__":
    main()
