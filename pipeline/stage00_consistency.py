"""
stage00_consistency.py — comprehensive reviewer-anticipation audit.

Pre-emptive answers to questions a reviewer might ask of:
  "Cross-Layer Characterization Framework for HPC Utilization"
  IEEE Cluster submission

Sections (each labeled with the reviewer question it answers):
   1. Corpus basic sanity            — "are headline numbers correct?"
   2. Identifier and timestamp logic — "are jobs unique, timestamps coherent?"
   3. DJC public-dump inventory      — "what's actually in the public release?"
   4. Telemetry coverage             — "are 78.1% / 25.2% / 18.4% correct?"
   5. DCGM join integrity            — "did you attribute rows correctly?"
   6. Failure taxonomy validity      — "is GPU_Idle_Timeout justified?"
   7. Taxonomy disjointness & coverage — "any double-counting? unclassified?"
   8. Darshan attachment bias        — "is the I/O-instrumented subset representative?"
   9. Temporal consistency           — "did the corpus drift?"
  10. Feature leakage audit          — "are you cheating?"
  11. ML stability                   — "is +0.198 AUC stable across seeds?"
  12. Cold-start & lookback sweep    — "what about new users? why 7 days?"
  13. User concentration             — "are results dominated by heavy users?"
  14. Calibration & robustness       — "well-calibrated or just well-ranked?"
  15. Public-data reproducibility    — "could anyone redo this from DJC alone?"
  16. Summary verdict                — JSON dump for paper writeup

Run:    python -m pipeline.stage00_consistency
Output: stdout + data/consistency_audit.json
"""

import os, sys, json, time, hashlib
import pandas as pd
import numpy as np
import warnings; warnings.filterwarnings("ignore")
from collections import OrderedDict
from scipy.stats import spearmanr, ks_2samp
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score, brier_score_loss
from sklearn.model_selection import TimeSeriesSplit
from sklearn.calibration import calibration_curve

import pyarrow.dataset as ds

# ─────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────
CFG_PATH      = "/pscratch/sd/h/hjajula/Darshan/cross_layer_hpc_tool/config/config.json"
COMBINED_CSV  = "/pscratch/sd/h/hjajula/Darshan/cross_layer_hpc_tool/data/combined_metrics_final.csv"
OUT_JSON      = "/pscratch/sd/h/hjajula/Darshan/cross_layer_hpc_tool/data/consistency_audit.json"

RNG           = 42
N_HIST        = 10
LOOKBACK_DAYS = 7
SEEDS         = [42, 7, 1337, 2024, 31415]   # for stability test
JOIN_SAMPLE_N = 35                            # join integrity sample size

WASTEFUL = {"Ghost","Scale_Waster","IO_Bottlenecked",
            "Failed_Job","Quick_Cancel","GPU_Idle_Timeout"}
TELEM_TIERS = {"Ghost","Scale_Waster","IO_Bottlenecked","Compute_Bound",
               "Moderate_Compute_No_IO","Low_Efficiency","GPU_Idle_Timeout",
               "Ideal_Compute_With_IO","Moderate_Compute_With_IO",
               "Incidental_IO_Low_GPU"}

audit = OrderedDict()  # collects key numbers for JSON dump
t0 = time.time()

def header(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)

def wilson_ci(k, n, z=1.96):
    if n == 0: return (0.0, 1.0)
    p = k/n; d = 1 + z*z/n
    c = (p + z*z/(2*n)) / d
    h = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return (max(0, c-h), min(1, c+h))

# ─────────────────────────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────────────────────────
header("LOADING DATA")
cfg = json.load(open(CFG_PATH))
djc_raw = pd.read_csv(cfg["djc_csv"], low_memory=False)
combined = pd.read_csv(COMBINED_CSV, low_memory=False)
combined["QUEUED_TIMESTAMP"] = pd.to_datetime(combined["QUEUED_TIMESTAMP"], errors="coerce")
combined["START_TIMESTAMP"]  = pd.to_datetime(combined["START_TIMESTAMP"],  errors="coerce")
combined["END_TIMESTAMP"]    = pd.to_datetime(combined["END_TIMESTAMP"],    errors="coerce")
print(f"DJC raw rows  : {len(djc_raw):,}")
print(f"Combined rows : {len(combined):,}")

# ═════════════════════════════════════════════════════════════════
# 1. CORPUS BASIC SANITY
#    Reviewer Q: "Are the headline numbers (262,634 jobs, 13 months,
#                 16.5M GPU-hr) exactly right?"
# ═════════════════════════════════════════════════════════════════
header("1. CORPUS BASIC SANITY")

n_jobs = len(combined)
total_gpu_hrs = combined["gpu_hours"].sum()
date_min = combined["QUEUED_TIMESTAMP"].min()
date_max = combined["END_TIMESTAMP"].max()
months_span = (date_max - date_min).days / 30.4

print(f"Total jobs               : {n_jobs:,}")
print(f"Total allocated GPU-hrs  : {total_gpu_hrs:,.0f}  ({total_gpu_hrs/1e6:.2f}M)")
print(f"Date range (queued→end)  : {date_min} → {date_max}")
print(f"Span (calendar months)   : {months_span:.1f}")

# Years covered
print(f"\nYear distribution:")
for y, n in combined["QUEUED_TIMESTAMP"].dt.year.value_counts().sort_index().items():
    print(f"  {int(y)}: {n:,} jobs ({n/n_jobs*100:.1f}%)")

audit["corpus"] = {
    "n_jobs": int(n_jobs),
    "total_alloc_gpu_hours": float(total_gpu_hrs),
    "date_min": str(date_min),
    "date_max": str(date_max),
    "calendar_months": float(months_span),
}

# ═════════════════════════════════════════════════════════════════
# 2. IDENTIFIER AND TIMESTAMP LOGIC
#    Reviewer Q: "Are JOB_NAMEs unique? Do timestamps make sense?
#                 Is RUNTIME ever > WALLTIME?"
# ═════════════════════════════════════════════════════════════════
header("2. IDENTIFIER AND TIMESTAMP LOGIC")

dup_job = combined["JOB_NAME"].duplicated().sum()
dup_cob = combined["COBALT_JOBID"].duplicated().sum() if "COBALT_JOBID" in combined.columns else None
print(f"Duplicate JOB_NAME rows  : {dup_job}")
print(f"Duplicate COBALT_JOBID   : {dup_cob}")

# Timestamp ordering: queued <= start <= end
mask = (combined["QUEUED_TIMESTAMP"].notna() &
        combined["START_TIMESTAMP"].notna() &
        combined["END_TIMESTAMP"].notna())
sub = combined[mask]
bad_qs = (sub["QUEUED_TIMESTAMP"] > sub["START_TIMESTAMP"]).sum()
bad_se = (sub["START_TIMESTAMP"]  > sub["END_TIMESTAMP"]).sum()
print(f"\nTimestamp violations (rows with all three timestamps: {len(sub):,}):")
print(f"  QUEUED > START         : {bad_qs}")
print(f"  START  > END           : {bad_se}")

# Runtime vs walltime
rt_overshoot = (combined["RUNTIME_SECONDS"] > combined["WALLTIME_SECONDS"] * 1.05).sum()
rt_negative  = (combined["RUNTIME_SECONDS"] < 0).sum()
print(f"\nRuntime sanity:")
print(f"  RUNTIME > WALLTIME×1.05 : {rt_overshoot}  ({rt_overshoot/n_jobs*100:.2f}%)")
print(f"  RUNTIME < 0             : {rt_negative}")
print(f"  Min runtime             : {combined['RUNTIME_SECONDS'].min():.1f}s")
print(f"  Max runtime             : {combined['RUNTIME_SECONDS'].max()/3600:.1f}h")

# Node count sanity
node_overshoot = (combined["NODES_USED"] > combined["NODES_REQUESTED"] * 1.5).sum() \
                 if "NODES_REQUESTED" in combined.columns else None
print(f"\nNode count sanity:")
print(f"  NODES_USED > REQ×1.5    : {node_overshoot}")
print(f"  Max NODES_USED          : {int(combined['NODES_USED'].max())}")

audit["identifiers"] = {
    "duplicate_job_names": int(dup_job),
    "duplicate_cobalt_ids": int(dup_cob) if dup_cob is not None else None,
    "timestamp_violations_qs": int(bad_qs),
    "timestamp_violations_se": int(bad_se),
    "runtime_overshoots_walltime": int(rt_overshoot),
}

# ═════════════════════════════════════════════════════════════════
# 3. DJC PUBLIC-DUMP INVENTORY
#    Reviewer Q: "What's actually in the public release? Could
#                 anyone reproduce this with just DJC?"
# ═════════════════════════════════════════════════════════════════
header("3. DJC PUBLIC-DUMP INVENTORY")

print(f"DJC total columns: {len(djc_raw.columns)}")

GPU_FIELDS_CLAIMED = [
    "GPUS_REQUESTED", "GPUS_USED",
    "GPU_UTILIZATION", "GPU_MEMORY_ALLOCATED_AVERAGE",
    "GPU_MEMORY_UTILIZATION_AVERAGE", "GPU_POWER_USAGE_AVERAGE",
    "GPU_TEMPERATURE_AVERAGE", "GPUS_USED_PER_NODE_AVERAGE",
]
print(f"\n[GPU fields presence in public DJC]")
present_gpu = []
for c in GPU_FIELDS_CLAIMED:
    is_present = c in djc_raw.columns
    nn = djc_raw[c].notna().sum() if is_present else 0
    flag = "yes" if is_present else "NO"
    print(f"  {c:<35} {flag:>4} {nn:>9,}")
    if is_present and nn > 0:
        present_gpu.append(c)

# Anonymization sanity: USERNAME_GENID and PROJECT_NAME_GENID look hashed
if "USERNAME_GENID" in djc_raw.columns:
    sample_user = str(djc_raw["USERNAME_GENID"].dropna().iloc[0])
    is_anon = sample_user.isdigit() and len(sample_user) > 8
    print(f"\nAnonymization sanity:")
    print(f"  Sample USERNAME_GENID   : {sample_user}  (looks hashed: {is_anon})")
    print(f"  Unique users in corpus  : {djc_raw['USERNAME_GENID'].nunique():,}")
    print(f"  Unique projects         : {djc_raw['PROJECT_NAME_GENID'].nunique():,}")

audit["djc_inventory"] = {
    "total_columns": int(len(djc_raw.columns)),
    "gpu_fields_present": present_gpu,
    "gpu_fields_absent": [c for c in GPU_FIELDS_CLAIMED if c not in present_gpu],
}

# ═════════════════════════════════════════════════════════════════
# 4. TELEMETRY COVERAGE
#    Reviewer Q: "Are coverage claims exact?"
# ═════════════════════════════════════════════════════════════════
header("4. TELEMETRY COVERAGE")

n_dcgm    = combined["has_gpu"].sum()
n_dar     = combined["darshan_present"].sum()
n_dar_io  = combined["io_detected"].sum()
n_both    = (combined["has_gpu"] & combined["darshan_present"]).sum()
n_neither = (~combined["has_gpu"] & ~combined["darshan_present"]).sum()

for label, n in [("DCGM telemetry      ", n_dcgm),
                 ("Darshan attached    ", n_dar),
                 ("Darshan I/O detected", n_dar_io),
                 ("Both DCGM & Darshan ", n_both),
                 ("Neither             ", n_neither)]:
    pct = n / n_jobs * 100
    lo, hi = wilson_ci(n, n_jobs)
    print(f"  {label}: {n:>8,} ({pct:5.1f}%, 95% CI [{lo*100:5.2f}, {hi*100:5.2f}])")

# DCGM rows-per-job distribution: detect short / sparse jobs
if "gpu_telemetry_rows" in combined.columns:
    print(f"\nDCGM rows-per-job distribution (jobs with telemetry):")
    s = combined.loc[combined["has_gpu"], "gpu_telemetry_rows"].dropna()
    for q in [0.05, 0.25, 0.50, 0.75, 0.95, 0.99]:
        print(f"  P{int(q*100):<3}: {s.quantile(q):>8,.0f}")
    print(f"  Min : {s.min():.0f}   Max : {s.max():,.0f}")

audit["coverage"] = {
    "n_dcgm": int(n_dcgm),  "pct_dcgm": float(n_dcgm/n_jobs*100),
    "n_darshan": int(n_dar),"pct_darshan": float(n_dar/n_jobs*100),
    "n_both": int(n_both),  "pct_both": float(n_both/n_jobs*100),
}

# ═════════════════════════════════════════════════════════════════
# 5. DCGM JOIN INTEGRITY
#    Reviewer Q: "Did you correctly attribute telemetry rows to jobs?
#                 Could nodes/timestamps be mismatched?"
# ═════════════════════════════════════════════════════════════════
header("5. DCGM JOIN INTEGRITY")

dataset = ds.dataset(cfg["parquet_out_dir"], partitioning="hive")
sample_tiers = ["Ghost","IO_Bottlenecked","Compute_Bound","GPU_Idle_Timeout",
                "Scale_Waster","Failed_Job","Moderate_Compute_No_IO"]
samples = []
for t in sample_tiers:
    sub = combined[(combined["crosslayer_tier"] == t) & combined["has_gpu"]]
    if len(sub) == 0: continue
    samples.append(sub.sample(min(JOIN_SAMPLE_N // len(sample_tiers), len(sub)),
                              random_state=RNG))
sample_df = pd.concat(samples, ignore_index=True)
print(f"Auditing {len(sample_df)} jobs across {len(sample_tiers)} tiers")

results = []
for _, job in sample_df.iterrows():
    start = job["START_TIMESTAMP"]; end = job["END_TIMESTAMP"]
    job_nodes = set(str(job["LOCATION"]).split(","))
    try:
        gpu_df = dataset.to_table(
            filter=(ds.field("date") >= start.date().isoformat()) &
                   (ds.field("date") <= end.date().isoformat())
        ).to_pandas()
    except Exception:
        continue
    gpu_df["TIMESTAMP"] = pd.to_datetime(gpu_df["TIMESTAMP"])
    in_window = gpu_df["TIMESTAMP"].between(start, end)
    on_nodes  = gpu_df["HOST"].isin(job_nodes)
    matched   = gpu_df[in_window & on_nodes]
    foreign_nodes = (~on_nodes & in_window).sum()

    rederived = matched["GPU_UTILIZATION_AVG"].mean() if len(matched) > 0 else np.nan
    stored    = job.get("util_mean", np.nan)
    drift = abs(rederived - stored) if pd.notna(rederived) and pd.notna(stored) else np.nan

    coverage = 0.0
    if len(matched) > 0:
        rt = (end - start).total_seconds()
        span = (matched["TIMESTAMP"].max() - matched["TIMESTAMP"].min()).total_seconds()
        coverage = span / rt if rt > 0 else 0

    results.append({"tier": job["crosslayer_tier"], "n_nodes": len(job_nodes),
                    "rows": len(matched), "foreign_partition_rows": foreign_nodes,
                    "stored": stored, "rederived": rederived,
                    "drift": drift, "coverage": coverage})

a = pd.DataFrame(results).dropna(subset=["drift"])
print(f"\n[Re-derivation drift summary]")
print(f"  Sampled jobs              : {len(a)}")
print(f"  Mean drift                : {a['drift'].mean():.6f}")
print(f"  Max drift                 : {a['drift'].max():.6f}")
print(f"  Drift < 0.001 (machine eps): {(a['drift']<0.001).sum()}/{len(a)}")
print(f"  Mean coverage             : {a['coverage'].mean():.3f}")
print(f"  Mean foreign-partition rows correctly excluded: {a['foreign_partition_rows'].mean():,.0f}")

audit["join_integrity"] = {
    "sampled": int(len(a)),
    "mean_drift": float(a["drift"].mean()),
    "max_drift": float(a["drift"].max()),
    "mean_coverage": float(a["coverage"].mean()),
}

# ═════════════════════════════════════════════════════════════════
# 6. FAILURE TAXONOMY VALIDITY
#    Reviewer Q: "Why is GPU_Idle_Timeout a separate tier?
#                 Is exit -29 actually walltime timeout?"
# ═════════════════════════════════════════════════════════════════
header("6. FAILURE TAXONOMY VALIDITY")

fail = combined[combined["EXIT_STATUS"].fillna(0) != 0]
print(f"Failed jobs (any non-zero exit) : {len(fail):,}")
print(f"\nTop 10 EXIT_STATUS values among failures:")
top_codes = fail["EXIT_STATUS"].value_counts().head(10)
for code, n in top_codes.items():
    print(f"  exit={int(code):>5}  {n:>7,} jobs ({n/len(fail)*100:5.1f}%)")

# GIT criteria: exit=-29 AND walltime fraction > 80% AND gpu < 5% AND >=4 GPUs
neg29 = combined[combined["EXIT_STATUS"] == -29].copy()
print(f"\n[exit=-29 deep dive]")
print(f"  Total exit=-29 jobs : {len(neg29):,}")
if len(neg29) > 0:
    neg29["wt_frac"] = neg29["RUNTIME_SECONDS"] / neg29["WALLTIME_SECONDS"].replace(0, np.nan)
    n_high_wt = (neg29["wt_frac"] > 0.8).sum()
    print(f"    walltime fraction > 80%      : {n_high_wt:,} ({n_high_wt/len(neg29)*100:.1f}%)")
    n_low_gpu = (neg29["util_mean"].fillna(100) < 5).sum()
    print(f"    GPU util < 5%                 : {n_low_gpu:,} ({n_low_gpu/len(neg29)*100:.1f}%)")
    n_classified_git = (neg29["crosslayer_tier"] == "GPU_Idle_Timeout").sum()
    print(f"    Classified as GPU_Idle_Timeout: {n_classified_git:,}")

# Other GIT-eligible exits (not -29) — should be zero by construction
other_git = combined[(combined["crosslayer_tier"] == "GPU_Idle_Timeout") &
                     (combined["EXIT_STATUS"] != -29)]
print(f"\nGIT classified with exit != -29 : {len(other_git)}  (should be 0)")

audit["failure_taxonomy"] = {
    "total_failed": int(len(fail)),
    "exit_neg29": int(len(neg29)),
    "git_classified": int((combined["crosslayer_tier"]=="GPU_Idle_Timeout").sum()),
    "git_with_wrong_exit": int(len(other_git)),
}

# ═════════════════════════════════════════════════════════════════
# 7. TAXONOMY DISJOINTNESS AND COVERAGE
#    Reviewer Q: "Are tiers really disjoint? Any double-counting
#                 or unclassified jobs?"
# ═════════════════════════════════════════════════════════════════
header("7. TAXONOMY DISJOINTNESS AND COVERAGE")

assert combined["crosslayer_tier"].notna().all(), "Some jobs are unclassified!"
print(f"Coverage: {combined['crosslayer_tier'].notna().sum():,}/{n_jobs:,} = 100.0%  ✓")

tier_counts = combined["crosslayer_tier"].value_counts()
print(f"\nTier sizes (n, pct of corpus, Wilson 95% CI on prop):")
for t, n in tier_counts.items():
    pct = n / n_jobs * 100
    lo, hi = wilson_ci(n, n_jobs)
    print(f"  {t:<28} {n:>8,}  {pct:5.2f}%  [{lo*100:5.3f}, {hi*100:5.3f}]")

# Disjointness via diagnostic_tier
overlap = (combined["crosslayer_tier"] == "Ghost") & \
          (combined["diagnostic_tier"] == "Scale_Waster")
print(f"\nGhost ∩ Scale_Waster overlap: {overlap.sum()}  (should be 0)")

# Sanity: every wasteful tier classification has expected GPU range
print(f"\nGPU util ranges by tier (sanity vs spec):")
for t, expected in [("Ghost", "<5%"), ("Scale_Waster","[5,10)%"),
                    ("Compute_Bound",">=70%")]:
    sub = combined[combined["crosslayer_tier"] == t]
    if len(sub) > 0:
        u = sub["util_mean"].dropna()
        print(f"  {t:<22} expected {expected:<10}  observed [{u.min():.2f}, {u.max():.2f}]")

audit["taxonomy"] = {
    "coverage_pct": 100.0,
    "ghost_scale_waster_overlap": int(overlap.sum()),
    "tier_sizes": {str(k): int(v) for k, v in tier_counts.items()},
}

# ═════════════════════════════════════════════════════════════════
# 8. DARSHAN ATTACHMENT BIAS
#    Reviewer Q: "Is the Darshan-attached subset a biased sample?
#                 Does I/O analysis generalize to the full corpus?"
# ═════════════════════════════════════════════════════════════════
header("8. DARSHAN ATTACHMENT BIAS")

with_d   = combined[combined["darshan_present"]]
without_d = combined[~combined["darshan_present"]]
print(f"With Darshan    : {len(with_d):,}")
print(f"Without Darshan : {len(without_d):,}")

# Compare distributions on key variables
print(f"\n[Distribution comparisons (with vs without Darshan)]")
print(f"  {'Variable':<22} {'with mean':>11} {'without':>11} {'KS p':>10}")
for c in ["RUNTIME_SECONDS", "NODES_REQUESTED", "WALLTIME_SECONDS"]:
    if c in combined.columns:
        a, b = with_d[c].dropna(), without_d[c].dropna()
        if len(a) > 50 and len(b) > 50:
            ks_stat, ks_p = ks_2samp(a.sample(min(5000,len(a)),random_state=RNG),
                                     b.sample(min(5000,len(b)),random_state=RNG))
            print(f"  {c:<22} {a.mean():>11.1f} {b.mean():>11.1f} {ks_p:>10.2e}")

# Categorical comparison: queue, science field
print(f"\n[Top 5 queues — distribution shift]")
print(f"  {'Queue':<22} {'with %':>8} {'without %':>10}")
for q in combined["QUEUE_NAME"].value_counts().head(5).index:
    p_w  = (with_d["QUEUE_NAME"]    == q).mean() * 100
    p_wo = (without_d["QUEUE_NAME"] == q).mean() * 100
    print(f"  {q:<22} {p_w:>7.1f}% {p_wo:>9.1f}%")

# Failure rate comparison
fail_w  = (with_d["EXIT_STATUS"]    != 0).mean() * 100
fail_wo = (without_d["EXIT_STATUS"] != 0).mean() * 100
print(f"\n[Exit-failure rate]")
print(f"  With Darshan    : {fail_w:.1f}%")
print(f"  Without Darshan : {fail_wo:.1f}%")

audit["darshan_bias"] = {
    "n_with_darshan": int(len(with_d)),
    "n_without_darshan": int(len(without_d)),
    "fail_rate_with_pct": float(fail_w),
    "fail_rate_without_pct": float(fail_wo),
}

# ═════════════════════════════════════════════════════════════════
# 9. TEMPORAL CONSISTENCY
#    Reviewer Q: "Did the corpus drift over 13 months? Are early-
#                 month results comparable to late-month?"
# ═════════════════════════════════════════════════════════════════
header("9. TEMPORAL CONSISTENCY")

combined["ym"] = combined["QUEUED_TIMESTAMP"].dt.to_period("M")
print(f"\n[Monthly trends]")
print(f"  {'Month':<10} {'Jobs':>8} {'DCGM%':>7} {'Dar%':>6} {'Wast%':>7} {'Mean util':>10}")
for ym, grp in combined.groupby("ym", observed=True):
    if len(grp) < 100: continue
    dcgm_pct = grp["has_gpu"].mean() * 100
    dar_pct  = grp["darshan_present"].mean() * 100
    wast_pct = grp["is_wasteful"].mean() * 100
    util_m   = grp.loc[grp["has_gpu"],"util_mean"].mean()
    print(f"  {str(ym):<10} {len(grp):>8,} {dcgm_pct:>6.1f}% {dar_pct:>5.1f}% "
          f"{wast_pct:>6.1f}% {util_m:>9.1f}")

# Variance check: is monthly wasteful rate stable enough?
monthly_wpct = combined.groupby("ym", observed=True)["is_wasteful"].mean() * 100
print(f"\n  Monthly wasteful% — mean: {monthly_wpct.mean():.1f}, "
      f"std: {monthly_wpct.std():.1f}, range: [{monthly_wpct.min():.1f}, {monthly_wpct.max():.1f}]")

audit["temporal"] = {
    "monthly_wasteful_mean": float(monthly_wpct.mean()),
    "monthly_wasteful_std":  float(monthly_wpct.std()),
}

# ═════════════════════════════════════════════════════════════════
# 10. FEATURE LEAKAGE AUDIT
#    Reviewer Q: "Are you secretly using current-job telemetry
#                 in the predictive features?"
# ═════════════════════════════════════════════════════════════════
header("10. FEATURE LEAKAGE AUDIT")

# Build the feature lists exactly as framework.py does
groupA = ["NODES_REQUESTED","WALLTIME_SECONDS","CORES_REQUESTED",
          "submit_hour","submit_dow","submit_month",
          "queue_freq","SCIENCE_FIELD_enc","executable_freq"]
hist_cols = ["util_mean","idle_frac","zero_util_frac","power_efficiency",
             "io_time_frac","bytes_per_gpu_hour"]
groupB = ["user_job_count","user_mean_runtime","user_walltime_efficiency",
          "user_fail_rate","user_quick_cancel_rate",
          "user_mean_nodes","user_mean_walltime"]
groupC = [f"hist_{c}" for c in hist_cols]
M3_feats = groupA + groupB + groupC

# Forbidden current-job columns that must NOT appear in any feature list
forbidden = {"util_mean","util_max","util_std","mem_util_mean","power_mean",
             "temp_mean","io_time_frac","BWio_MB","bytes_read","bytes_written",
             "idle_frac","zero_util_frac","gpu_hours","posix_reads",
             "metadata_ops_per_gb","crosslayer_tier","is_wasteful",
             "diagnostic_tier","RUNTIME_SECONDS","EXIT_STATUS"}

print(f"[Forbidden-column overlap check]")
for name, feats in [("M1 (groupA)", groupA), ("M3 (full)", M3_feats)]:
    overlap = set(feats) & forbidden
    print(f"  {name:<15}: {len(overlap)} overlap  {overlap if overlap else '✓ clean'}")

# Confirm every "hist_*" feature has its historical prefix
hist_in_M3 = [f for f in M3_feats if f.startswith("hist_") or f.startswith("user_")]
print(f"\n[Historical-feature prefix check]")
print(f"  M3 total features              : {len(M3_feats)}")
print(f"  Historical features (hist_/user_): {len(hist_in_M3)}")
print(f"  Scheduler-current features      : {len(M3_feats) - len(hist_in_M3)}")

# Train/test temporal disjointness (re-run with fresh load)
combined_sorted = combined.sort_values("QUEUED_TIMESTAMP").reset_index(drop=True)
telem = combined_sorted[combined_sorted["crosslayer_tier"].isin(TELEM_TIERS)].reset_index(drop=True)
split_t = int(len(telem) * 0.80)
cutoff = telem["QUEUED_TIMESTAMP"].iloc[split_t]
train_max_end = telem["END_TIMESTAMP"].iloc[:split_t].max()
test_min_q   = telem["QUEUED_TIMESTAMP"].iloc[split_t:].min()
print(f"\n[Train/test temporal disjointness]")
print(f"  Cutoff (queued)        : {cutoff}")
print(f"  Max train END_TIMESTAMP: {train_max_end}")
print(f"  Min test QUEUED        : {test_min_q}")
print(f"  Test queued before any train ended? "
      f"{(test_min_q < train_max_end)} (overlap is allowed since queued != end)")

audit["leakage"] = {
    "M1_forbidden_overlap": list(set(groupA) & forbidden),
    "M3_forbidden_overlap": list(set(M3_feats) & forbidden),
    "M3_total_features": len(M3_feats),
    "M3_historical_features": len(hist_in_M3),
}

# ═════════════════════════════════════════════════════════════════
# 11. ML STABILITY (multi-seed)
#    Reviewer Q: "Did you cherry-pick seed=42? How variable is AUC
#                 across random seeds?"
# ═════════════════════════════════════════════════════════════════
header("11. ML STABILITY (MULTI-SEED)")

# Light-weight rebuild of telem_df features for stability test
def to_X(df, feats):
    avail = [f for f in feats if f in df.columns]
    return np.nan_to_num(df[avail].values, nan=-1, posinf=1e9, neginf=-1e9), avail

# Rebuild submit_* features
combined_sorted["submit_hour"]  = combined_sorted["QUEUED_TIMESTAMP"].dt.hour
combined_sorted["submit_dow"]   = combined_sorted["QUEUED_TIMESTAMP"].dt.dayofweek
combined_sorted["submit_month"] = combined_sorted["QUEUED_TIMESTAMP"].dt.month
telem = combined_sorted[combined_sorted["crosslayer_tier"].isin(TELEM_TIERS)].copy().reset_index(drop=True)
split_t = int(len(telem) * 0.80)

queue_freq = telem["QUEUE_NAME"].iloc[:split_t].value_counts()
exe_freq   = (telem["executable"].iloc[:split_t].value_counts()
              if "executable" in telem.columns else pd.Series(dtype=int))
telem["queue_freq"]      = telem["QUEUE_NAME"].map(queue_freq).fillna(0)
telem["executable_freq"] = (telem["executable"].map(exe_freq).fillna(0)
                            if "executable" in telem.columns else 0)
le = LabelEncoder()
le.fit(telem["SCIENCE_FIELD"].iloc[:split_t].astype(str))
known = set(le.classes_)
telem["SCIENCE_FIELD_enc"] = telem["SCIENCE_FIELD"].astype(str).apply(
    lambda x: le.transform([x])[0] if x in known else -1)

# Use existing hist_* and user_* if already present from framework.py merge
have_hist = all(f in telem.columns for f in groupC + groupB)
print(f"Historical features present in saved CSV: {have_hist}")
if not have_hist:
    print("  -> framework.py must be run first to populate hist_* features.")
    print("  -> Skipping multi-seed stability test.")
    audit["stability"] = {"skipped": True}
else:
    y_tr = telem["is_wasteful"].iloc[:split_t].values
    y_te = telem["is_wasteful"].iloc[split_t:].values
    X_tr_M1, _ = to_X(telem.iloc[:split_t], groupA)
    X_te_M1, _ = to_X(telem.iloc[split_t:], groupA)
    X_tr_M3, M3_avail = to_X(telem.iloc[:split_t], M3_feats)
    X_te_M3, _        = to_X(telem.iloc[split_t:], M3_feats)

    print(f"\nRunning {len(SEEDS)} seeds for M1 and M3...")
    m1_aucs, m3_aucs = [], []
    for s in SEEDS:
        rf1 = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=s)
        rf1.fit(X_tr_M1, y_tr)
        m1_aucs.append(roc_auc_score(y_te, rf1.predict_proba(X_te_M1)[:,1]))
        rf3 = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=s)
        rf3.fit(X_tr_M3, y_tr)
        m3_aucs.append(roc_auc_score(y_te, rf3.predict_proba(X_te_M3)[:,1]))
    print(f"\n[Across {len(SEEDS)} seeds]")
    print(f"  M1 AUC: {np.mean(m1_aucs):.4f} ± {np.std(m1_aucs):.4f}  "
          f"[{min(m1_aucs):.4f}, {max(m1_aucs):.4f}]")
    print(f"  M3 AUC: {np.mean(m3_aucs):.4f} ± {np.std(m3_aucs):.4f}  "
          f"[{min(m3_aucs):.4f}, {max(m3_aucs):.4f}]")
    print(f"  M3 lift: {np.mean(m3_aucs)-np.mean(m1_aucs):+.4f} ± "
          f"{np.std(np.array(m3_aucs)-np.array(m1_aucs)):.4f}")

    # Feature SHA — proves identical preprocessing across runs
    sha = hashlib.sha256(telem[M3_avail].fillna(-999).values.tobytes()).hexdigest()[:16]
    print(f"\n  Feature matrix SHA256[:16]: {sha}")

    audit["stability"] = {
        "seeds": SEEDS,
        "M1_AUC_mean": float(np.mean(m1_aucs)),
        "M1_AUC_std":  float(np.std(m1_aucs)),
        "M3_AUC_mean": float(np.mean(m3_aucs)),
        "M3_AUC_std":  float(np.std(m3_aucs)),
        "feature_sha16": sha,
    }

    # ═════════════════════════════════════════════════════════════════
    # 12. COLD-START & LOOKBACK SENSITIVITY
    #    Reviewer Q: "What about new users with no history?
    #                 Why is 7 days the right lookback?"
    # ═════════════════════════════════════════════════════════════════
    header("12. COLD-START AND HISTORY-DEPTH SENSITIVITY")

    test = telem.iloc[split_t:].copy().reset_index(drop=True)
    rf3_main = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=RNG)
    rf3_main.fit(X_tr_M3, y_tr)
    test["pred_proba"] = rf3_main.predict_proba(X_te_M3)[:,1]

    print(f"\n[M3 AUC stratified by user_job_count at submission]")
    bins = [(-0.5, 0.5,  "0 prior jobs (cold start)"),
            (0.5,  3.5,  "1-3 prior jobs"),
            (3.5,  9.5,  "4-9 prior jobs"),
            (9.5,  np.inf, "10 prior jobs (max)")]
    print(f"  {'Bucket':<28} {'n':>7} {'pos%':>6} {'AUC':>8}")
    for lo, hi, label in bins:
        sub = test[(test["user_job_count"] > lo) & (test["user_job_count"] <= hi)]
        if len(sub) < 50 or sub["is_wasteful"].nunique() < 2:
            print(f"  {label:<28} {len(sub):>7,} (insufficient or single-class)")
            continue
        auc = roc_auc_score(sub["is_wasteful"], sub["pred_proba"])
        print(f"  {label:<28} {len(sub):>7,} {sub['is_wasteful'].mean()*100:>5.1f}% "
              f"{auc:>8.4f}")

    audit["cold_start"] = {
        "note": "Per-bucket AUCs printed above; significant drop in 0-prior bucket "
                "would suggest cold-start handling needed.",
    }

# ═════════════════════════════════════════════════════════════════
# 13. USER CONCENTRATION
#    Reviewer Q: "Are results dominated by a few heavy users? Would
#                 dropping the top user collapse the corpus?"
# ═════════════════════════════════════════════════════════════════
header("13. USER CONCENTRATION")

user_counts = combined.groupby("USERNAME_GENID").size().sort_values(ascending=False)
user_gpu    = combined.groupby("USERNAME_GENID")["gpu_hours"].sum().sort_values(ascending=False)
n_users = len(user_counts)
print(f"Unique users         : {n_users:,}")
print(f"\n[Top-10 user share]")
top10_jobs = user_counts.head(10).sum()
top10_gpu  = user_gpu.head(10).sum()
print(f"  Top-10 jobs share    : {top10_jobs/n_jobs*100:5.1f}%")
print(f"  Top-10 GPU-hr share  : {top10_gpu/total_gpu_hrs*100:5.1f}%")

# Lorenz / Gini-style: cumulative share at top-1%, top-5%
for p in [0.01, 0.05, 0.10, 0.50]:
    k = max(1, int(n_users * p))
    cum_jobs = user_counts.head(k).sum() / n_jobs * 100
    cum_gpu  = user_gpu.head(k).sum() / total_gpu_hrs * 100
    print(f"  Top-{int(p*100):>2}% users: {cum_jobs:5.1f}% of jobs, {cum_gpu:5.1f}% of GPU-hrs")

audit["user_concentration"] = {
    "n_unique_users": int(n_users),
    "top10_pct_jobs": float(top10_jobs/n_jobs*100),
    "top10_pct_gpu_hrs": float(top10_gpu/total_gpu_hrs*100),
}

# ═════════════════════════════════════════════════════════════════
# 14. CALIBRATION & ROBUSTNESS
#    Reviewer Q: "Is M3 well-calibrated, or just well-ranked?"
# ═════════════════════════════════════════════════════════════════
header("14. CALIBRATION AND ROBUSTNESS")

if audit.get("stability", {}).get("skipped"):
    print("Skipped (depends on stability section's classifier).")
else:
    brier = brier_score_loss(y_te, test["pred_proba"])
    print(f"M3 Brier score : {brier:.4f}  (lower is better; 0.25 = chance)")
    frac_pos, mean_pred = calibration_curve(y_te, test["pred_proba"],
                                            n_bins=10, strategy="quantile")
    print(f"\n[Reliability — predicted vs actual positive rate]")
    print(f"  {'pred':>8} {'actual':>8}")
    for p, a in zip(mean_pred, frac_pos):
        print(f"  {p:>8.3f} {a:>8.3f}")
    audit["calibration"] = {"brier_score": float(brier)}

# ═════════════════════════════════════════════════════════════════
# 15. PUBLIC-DATA REPRODUCIBILITY
#    Reviewer Q: "Could anyone redo this from the public DJC dump?"
# ═════════════════════════════════════════════════════════════════
header("15. PUBLIC-DATA REPRODUCIBILITY")

print("Public ALCF Polaris release contains:")
print(f"  - DJC scheduler metadata: {len(djc_raw.columns)} columns, "
      f"{len(djc_raw):,} jobs")
print(f"  - Per-job Darshan logs (anonymized; job_id preserved in filename)")
print(f"  - Raw DCGM telemetry stream (timestamped, node-keyed)")
print(f"\nWhat the public DJC alone CANNOT reconstruct:")
absent = audit["djc_inventory"]["gpu_fields_absent"]
for c in absent:
    print(f"  - {c}")
print(f"\n=> The DCGM-to-scheduler join in this paper's Aggregation Layer is the")
print(f"   only way to obtain per-job GPU aggregates from the public release.")

audit["reproducibility"] = {
    "public_release_columns": len(djc_raw.columns),
    "gpu_fields_only_via_dcgm_join": absent,
}

# ═════════════════════════════════════════════════════════════════
# 16. SUMMARY VERDICT — JSON for paper writeup
# ═════════════════════════════════════════════════════════════════
header("16. SUMMARY VERDICT")

audit["meta"] = {
    "runtime_minutes": round((time.time() - t0) / 60, 2),
    "script": "stage00_consistency.py",
}

print("\nAll checks complete. Headline numbers:")
print(f"  Corpus: {audit['corpus']['n_jobs']:,} jobs, "
      f"{audit['corpus']['total_alloc_gpu_hours']/1e6:.2f}M GPU-hrs, "
      f"{audit['corpus']['calendar_months']:.1f} months")
print(f"  Coverage: DCGM {audit['coverage']['pct_dcgm']:.1f}%, "
      f"Darshan {audit['coverage']['pct_darshan']:.1f}%, "
      f"both {audit['coverage']['pct_both']:.1f}%")
print(f"  Join integrity: max drift {audit['join_integrity']['max_drift']:.6f} "
      f"across {audit['join_integrity']['sampled']} jobs")
print(f"  Disjointness:   Ghost ∩ Scale_Waster = "
      f"{audit['taxonomy']['ghost_scale_waster_overlap']}")
print(f"  Leakage:        M3 forbidden overlap = "
      f"{len(audit['leakage']['M3_forbidden_overlap'])}")
if not audit.get("stability", {}).get("skipped"):
    print(f"  Stability:      M3 AUC = {audit['stability']['M3_AUC_mean']:.4f} "
          f"± {audit['stability']['M3_AUC_std']:.4f} across "
          f"{len(audit['stability']['seeds'])} seeds")

os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
with open(OUT_JSON, "w") as f:
    json.dump(audit, f, indent=2, default=str)
print(f"\nFull audit saved to: {OUT_JSON}")
print(f"Total runtime: {(time.time()-t0)/60:.1f} min")




# ═════════════════════════════════════════════════════════════════
# 17. KEY USAGE AUDIT (the COBALT_JOBID question)
#     Reviewer Q: "You reported 262,633 duplicate COBALT_JOBIDs.
#                  Did you actually use that as a key?"
# ═════════════════════════════════════════════════════════════════
header("17. KEY USAGE AUDIT")

# JOB_NAME is the primary key throughout the pipeline.
# COBALT_JOBID appears in the public DJC dump but is unused.
n_jobname_unique = combined["JOB_NAME"].nunique()
n_jobname_null   = combined["JOB_NAME"].isna().sum()

if "COBALT_JOBID" in combined.columns:
    n_cobalt_nonnull = combined["COBALT_JOBID"].notna().sum()
    n_cobalt_null    = combined["COBALT_JOBID"].isna().sum()
    n_cobalt_unique  = combined["COBALT_JOBID"].dropna().nunique()
else:
    n_cobalt_nonnull = n_cobalt_null = n_cobalt_unique = 0

print(f"Primary key (used everywhere)  : JOB_NAME")
print(f"  Unique JOB_NAME values       : {n_jobname_unique:,}")
print(f"  Null  JOB_NAME values        : {n_jobname_null:,}")
print(f"  Coverage as key              : "
      f"{n_jobname_unique/n_jobs*100:.4f}%  "
      f"{'✓' if n_jobname_unique == n_jobs and n_jobname_null == 0 else '✗'}")

print(f"\nUnused column (in DJC dump)    : COBALT_JOBID")
print(f"  Non-null values              : {n_cobalt_nonnull:,}")
print(f"  Null     values              : {n_cobalt_null:,}")
print(f"  Unique non-null values       : {n_cobalt_unique:,}")
if n_cobalt_null == n_jobs - 1 or n_cobalt_nonnull == 0:
    print(f"  Verdict                      : Field scrubbed in public release.")
    print(f"                                 Apparent duplicates are NaN==NaN; not used as key.")

audit["key_usage"] = {
    "primary_key": "JOB_NAME",
    "jobname_unique": int(n_jobname_unique),
    "jobname_nulls": int(n_jobname_null),
    "cobalt_jobid_nonnull": int(n_cobalt_nonnull),
    "cobalt_jobid_used_as_key": False,
}

# ═════════════════════════════════════════════════════════════════
# 18. SEPTEMBER 2025 ANOMALY INVESTIGATION
#     Reviewer Q: "Your monthly trend shows mean GPU util = 0.0% in
#                  Sept 2025 with 98% wasteful rate. What happened?"
# ═════════════════════════════════════════════════════════════════
header("18. SEPTEMBER 2025 ANOMALY INVESTIGATION")

sep_mask = combined["QUEUED_TIMESTAMP"].dt.to_period("M") == "2025-09"
sep = combined[sep_mask]
print(f"September 2025 jobs            : {len(sep):,}")
print(f"  with DCGM telemetry          : {sep['has_gpu'].sum():,} "
      f"({sep['has_gpu'].mean()*100:.1f}%)")
print(f"  with Darshan attached        : {sep['darshan_present'].sum():,} "
      f"({sep['darshan_present'].mean()*100:.1f}%)")

print(f"\n[Sept 2025 GPU util distribution (jobs with telemetry)]")
sep_with = sep[sep["has_gpu"]]
if len(sep_with) > 0:
    s = sep_with["util_mean"]
    print(f"  N           : {len(s):,}")
    print(f"  Mean        : {s.mean():.4f}")
    print(f"  Std         : {s.std():.4f}")
    print(f"  Max         : {s.max():.2f}")
    print(f"  Min         : {s.min():.2f}")
    for q in [0.50, 0.75, 0.90, 0.95, 0.99, 1.00]:
        print(f"  P{int(q*100):<10}: {s.quantile(q):.4f}")
    n_zero = (s == 0).sum()
    n_below_001 = (s < 0.01).sum()
    print(f"  ==0         : {n_zero:,} ({n_zero/len(s)*100:.1f}%)")
    print(f"   <0.01      : {n_below_001:,} ({n_below_001/len(s)*100:.1f}%)")

print(f"\n[Sept 2025 tier distribution]")
sep_tiers = sep["crosslayer_tier"].value_counts()
for t, n in sep_tiers.head(10).items():
    print(f"  {t:<28} {n:>8,} ({n/len(sep)*100:5.1f}%)")

# Is this a DCGM-side issue or a real facility event?
# Compare GPU util distribution Sept vs neighbors (Aug, Oct)
print(f"\n[Aug vs Sept vs Oct GPU util comparison (jobs with DCGM)]")
for ym in ["2025-08", "2025-09", "2025-10"]:
    m = combined["QUEUED_TIMESTAMP"].dt.to_period("M").astype(str) == ym
    sub = combined[m & combined["has_gpu"]]
    if len(sub) > 0:
        u = sub["util_mean"]
        print(f"  {ym}: N={len(sub):>6,}  mean={u.mean():>5.2f}  "
              f"median={u.median():>5.2f}  P95={u.quantile(0.95):>6.2f}")

# Check if power telemetry shows the same pattern (tells us if it's
# DCGM-wide or just utilization channel)
print(f"\n[Sept 2025: power telemetry sanity]")
if "power_mean" in sep_with.columns:
    p = sep_with["power_mean"].dropna()
    if len(p) > 0:
        print(f"  N with power data    : {len(p):,}")
        print(f"  Mean power (W)       : {p.mean():.1f}")
        print(f"  Median power (W)     : {p.median():.1f}")
        print(f"  Power == 0           : {(p == 0).sum():,}")
        if p.mean() > 50 and sep_with["util_mean"].mean() < 1:
            print(f"  -> Power high, util zero: smells like DCGM util-channel anomaly,")
            print(f"     not a facility-wide power-down.")
        elif p.mean() < 10:
            print(f"  -> Both util and power near zero: consistent with facility outage.")

audit["september_anomaly"] = {
    "n_jobs": int(len(sep)),
    "n_with_dcgm": int(sep["has_gpu"].sum()),
    "mean_util_sept": float(sep_with["util_mean"].mean()) if len(sep_with) > 0 else None,
    "mean_util_aug":  float(combined[(combined["QUEUED_TIMESTAMP"].dt.to_period("M").astype(str) == "2025-08")
                                     & combined["has_gpu"]]["util_mean"].mean()),
    "mean_util_oct":  float(combined[(combined["QUEUED_TIMESTAMP"].dt.to_period("M").astype(str) == "2025-10")
                                     & combined["has_gpu"]]["util_mean"].mean()),
}

# ═════════════════════════════════════════════════════════════════
# 19. STABILITY TEST WITH IN-SCRIPT HISTORICAL FEATURES
#     Reviewer Q: "Did you cherry-pick seed=42? How variable is AUC
#                  across random seeds?"
#
#     This rebuilds hist_* and user_* features from scratch using the
#     same logic as framework.py, so the audit is self-contained.
# ═════════════════════════════════════════════════════════════════
header("19. ML STABILITY (SELF-CONTAINED, MULTI-SEED)")

# Rebuild scheduler features
combined_sorted = combined.sort_values("QUEUED_TIMESTAMP").reset_index(drop=True)
combined_sorted["submit_hour"]  = combined_sorted["QUEUED_TIMESTAMP"].dt.hour
combined_sorted["submit_dow"]   = combined_sorted["QUEUED_TIMESTAMP"].dt.dayofweek
combined_sorted["submit_month"] = combined_sorted["QUEUED_TIMESTAMP"].dt.month

# Need 'executable' column — fallback to NaN if absent
if "executable" not in combined_sorted.columns:
    combined_sorted["executable"] = np.nan

train_df = combined_sorted[combined_sorted["use_for_training"] == True].copy().reset_index(drop=True)
telem = train_df[train_df["crosslayer_tier"].isin(TELEM_TIERS)].copy().reset_index(drop=True)
split_t_s = int(len(telem) * 0.80)
print(f"Trainable jobs : {len(train_df):,}")
print(f"Telem subset   : {len(telem):,}  (split at {split_t_s:,})")

# Recreate categorical encodings using only the training prefix
queue_freq = telem["QUEUE_NAME"].iloc[:split_t_s].value_counts()
exe_freq   = telem["executable"].iloc[:split_t_s].value_counts()
telem["queue_freq"]      = telem["QUEUE_NAME"].map(queue_freq).fillna(0)
telem["executable_freq"] = telem["executable"].map(exe_freq).fillna(0)
le2 = LabelEncoder()
le2.fit(telem["SCIENCE_FIELD"].iloc[:split_t_s].astype(str))
known2 = set(le2.classes_)
telem["SCIENCE_FIELD_enc"] = telem["SCIENCE_FIELD"].astype(str).apply(
    lambda x: le2.transform([x])[0] if x in known2 else -1)

# Build historical features (same logic as framework.py build_hist)
def build_hist_audit(train_df, target_jobs, cols, lookback_days=7, n_history=10):
    rows = []
    target_set = set(target_jobs)
    lb_ns = np.timedelta64(lookback_days, "D")
    for user, grp in train_df.groupby("USERNAME_GENID", sort=False):
        grp = grp.sort_values("QUEUED_TIMESTAMP").reset_index(drop=True)
        qts = grp["QUEUED_TIMESTAMP"].values
        ets = grp["END_TIMESTAMP"].values
        for i in range(len(grp)):
            jn = grp.loc[i, "JOB_NAME"]
            if jn not in target_set: continue
            mask = (ets < qts[i]) & (ets >= qts[i] - lb_ns)
            past_idx = np.where(mask)[0][-n_history:]
            row = {"JOB_NAME": jn}
            if len(past_idx) > 0:
                p = grp.iloc[past_idx]
                row["user_job_count"]           = int(mask.sum())
                row["user_mean_runtime"]        = p["RUNTIME_SECONDS"].mean()
                row["user_walltime_efficiency"] = (p["RUNTIME_SECONDS"] /
                    p["WALLTIME_SECONDS"].replace(0, np.nan)).mean()
                row["user_fail_rate"]           = (p["EXIT_STATUS"] != 0).mean()
                row["user_quick_cancel_rate"]   = (p["RUNTIME_SECONDS"] < 60).mean()
                row["user_mean_nodes"]          = p["NODES_REQUESTED"].mean()
                row["user_mean_walltime"]       = p["WALLTIME_SECONDS"].mean()
                for c in cols:
                    v = p[c].dropna() if c in p.columns else pd.Series(dtype=float)
                    row[f"hist_{c}"] = v.mean() if len(v) else -1
            else:
                cur = grp.iloc[i]
                row.update({"user_job_count": 0,
                            "user_mean_runtime": cur["WALLTIME_SECONDS"],
                            "user_walltime_efficiency": 0.5,
                            "user_fail_rate": 0.0,
                            "user_quick_cancel_rate": 0.0,
                            "user_mean_nodes": cur["NODES_REQUESTED"],
                            "user_mean_walltime": cur["WALLTIME_SECONDS"]})
                for c in cols: row[f"hist_{c}"] = -1
            rows.append(row)
    return pd.DataFrame(rows)

hist_cols = ["util_mean","idle_frac","zero_util_frac","power_efficiency",
             "io_time_frac","bytes_per_gpu_hour"]
print(f"\nBuilding historical features ({LOOKBACK_DAYS}-day lookback, "
      f"last {N_HIST} jobs)...")
t_h = time.time()
hist_df = build_hist_audit(train_df, telem["JOB_NAME"].tolist(),
                            hist_cols, LOOKBACK_DAYS, N_HIST)
telem = telem.merge(hist_df, on="JOB_NAME", how="left")
print(f"  {len(hist_df):,} hist rows | {time.time()-t_h:.0f}s elapsed")

groupA_s = ["NODES_REQUESTED","WALLTIME_SECONDS","CORES_REQUESTED",
            "submit_hour","submit_dow","submit_month",
            "queue_freq","SCIENCE_FIELD_enc","executable_freq"]
groupB_s = ["user_job_count","user_mean_runtime","user_walltime_efficiency",
            "user_fail_rate","user_quick_cancel_rate",
            "user_mean_nodes","user_mean_walltime"]
groupC_s = [f"hist_{c}" for c in hist_cols]
M3_feats_s = groupA_s + groupB_s + groupC_s

def to_X_s(df, feats):
    avail = [f for f in feats if f in df.columns]
    return np.nan_to_num(df[avail].values, nan=-1, posinf=1e9, neginf=-1e9), avail

y_tr = telem["is_wasteful"].iloc[:split_t_s].values
y_te = telem["is_wasteful"].iloc[split_t_s:].values
X_tr_M1, _    = to_X_s(telem.iloc[:split_t_s], groupA_s)
X_te_M1, _    = to_X_s(telem.iloc[split_t_s:], groupA_s)
X_tr_M3, M3a  = to_X_s(telem.iloc[:split_t_s], M3_feats_s)
X_te_M3, _    = to_X_s(telem.iloc[split_t_s:], M3_feats_s)

print(f"\nRunning {len(SEEDS)} seeds for M1 and M3...")
m1_aucs, m3_aucs, m1_f1s, m3_f1s = [], [], [], []
for s in SEEDS:
    rf1 = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=s).fit(X_tr_M1, y_tr)
    p1 = rf1.predict_proba(X_te_M1)[:, 1]
    m1_aucs.append(roc_auc_score(y_te, p1))
    m1_f1s.append(f1_score(y_te, rf1.predict(X_te_M1), average="macro"))

    rf3 = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=s).fit(X_tr_M3, y_tr)
    p3 = rf3.predict_proba(X_te_M3)[:, 1]
    m3_aucs.append(roc_auc_score(y_te, p3))
    m3_f1s.append(f1_score(y_te, rf3.predict(X_te_M3), average="macro"))

print(f"\n[Across {len(SEEDS)} seeds]")
print(f"  M1 AUC: {np.mean(m1_aucs):.4f} ± {np.std(m1_aucs):.4f}  "
      f"[{min(m1_aucs):.4f}, {max(m1_aucs):.4f}]")
print(f"  M3 AUC: {np.mean(m3_aucs):.4f} ± {np.std(m3_aucs):.4f}  "
      f"[{min(m3_aucs):.4f}, {max(m3_aucs):.4f}]")
lifts = np.array(m3_aucs) - np.array(m1_aucs)
print(f"  Lift  : {lifts.mean():+.4f} ± {lifts.std():.4f}")
print(f"  M1 F1 : {np.mean(m1_f1s):.4f} ± {np.std(m1_f1s):.4f}")
print(f"  M3 F1 : {np.mean(m3_f1s):.4f} ± {np.std(m3_f1s):.4f}")

# Calibration on the seed=42 M3 (matches the paper's reported model)
rf3_main = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=RNG).fit(X_tr_M3, y_tr)
p3_main = rf3_main.predict_proba(X_te_M3)[:, 1]
brier = brier_score_loss(y_te, p3_main)
frac_pos, mean_pred = calibration_curve(y_te, p3_main, n_bins=10, strategy="quantile")
print(f"\n[Calibration of seed=42 M3]")
print(f"  Brier score : {brier:.4f}  (lower is better; 0.25 = chance)")
print(f"  {'pred bin':>10} {'actual':>8}")
for pb, ab in zip(mean_pred, frac_pos):
    print(f"  {pb:>10.3f} {ab:>8.3f}")

# Feature SHA — proves identical preprocessing
sha = hashlib.sha256(telem[M3a].fillna(-999).values.tobytes()).hexdigest()[:16]
print(f"\nFeature matrix SHA256[:16]: {sha}")

audit["stability"] = {
    "seeds": SEEDS,
    "M1_AUC_mean": float(np.mean(m1_aucs)),
    "M1_AUC_std":  float(np.std(m1_aucs)),
    "M3_AUC_mean": float(np.mean(m3_aucs)),
    "M3_AUC_std":  float(np.std(m3_aucs)),
    "lift_mean":   float(lifts.mean()),
    "lift_std":    float(lifts.std()),
    "M1_F1_mean":  float(np.mean(m1_f1s)),
    "M3_F1_mean":  float(np.mean(m3_f1s)),
    "feature_sha16": sha,
}
audit["calibration"] = {
    "brier_score": float(brier),
    "n_bins": 10,
}