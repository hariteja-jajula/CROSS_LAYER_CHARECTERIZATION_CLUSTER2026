from pathlib import Path


import os
import sys
import json
import time
import hashlib
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from scipy.stats import spearmanr

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, f1_score, classification_report

import importlib
import utils.combined as cu
importlib.reload(cu)
from utils.combined import classify_crosslayer

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIG
# =============================================================================

RNG = 42
N_HIST = 10
LOOKBACK_DAYS = 7

ROOT = Path("/pscratch/sd/h/hjajula/Darshan/cross_layer_hpc_tool")
CFG_PATH = ROOT / "config/config.json"
GPU_PATH = ROOT / "data/gpu_metrics.csv"
DM_PATH = ROOT / "data/darshan_metrics_final.csv"
OLD_EXE_PATH = ROOT / "data/darshan_metrics_old.csv"
OUT_COMBINED = ROOT / "data/combined_metrics_final_paperstats.csv"

# Set true only if you need the slower sensitivity models.
RUN_BOOSTING = True
RUN_MULTICLASS = True
RUN_EXPANDED_HISTORY = True

# =============================================================================
# HELPERS
# =============================================================================

t0 = time.time()

def box(title: str) -> None:
    print("\n" + "=" * 78, flush=True)
    print(title, flush=True)
    print("=" * 78, flush=True)

def pct(num, den) -> float:
    return 100.0 * num / den if den else np.nan

def safe_div(num, den) -> float:
    return num / den if den else np.nan

def check(label: str, computed, claimed, tol=0.005) -> None:
    """Numerical sanity check. For percentages, pass fractions, not 0-100 values."""
    try:
        ok = abs(float(computed) - float(claimed)) <= tol
    except Exception:
        ok = False
    flag = "OK" if ok else f"CHECK claimed={claimed}"
    print(f"  {label:<58} {computed:>12.4f}  {flag}", flush=True)

def paper_name(tier: str) -> str:
    return {
        "Ghost": "Idle",
        "Scale_Waster": "Scale_Inefficient",
        "Idle_Hidden_Activity": "Idle_Hidden_Activity",
    }.get(tier, tier)

def latex_texttt(s: str) -> str:
    return r"\texttt{" + str(s).replace("_", r"\_") + "}"

def to_X(df: pd.DataFrame, feats):
    avail = [f for f in feats if f in df.columns]
    X = df[avail].apply(pd.to_numeric, errors="coerce").values
    return np.nan_to_num(X, nan=-1, posinf=1e9, neginf=-1e9)

def unique_keep_order(items):
    seen, out = set(), []
    for x in items:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out

# =============================================================================
# 1. LOAD + MERGE
# =============================================================================

box("1. LOAD + MERGE")

cfg = json.load(open(CFG_PATH))
job_df = pd.read_csv(cfg["djc_csv"], low_memory=False)
gm = pd.read_csv(GPU_PATH, low_memory=False)
dm = pd.read_csv(DM_PATH, low_memory=False)

dm["job_id"] = dm["job_id"].astype(str).str.replace(r"\.0$", "", regex=True)
job_df["job_id"] = job_df["JOB_NAME"].astype(str).str.split(".", n=1).str[0]

print(f"Scheduler jobs: {len(job_df):,}")
print(f"GPU rows/jobs:  {len(gm):,}")
print(f"Darshan rows:   {len(dm):,}")
print(f"Darshan unique jobs: {dm['job_id'].nunique():,}")

agg_spec = {k: v for k, v in {
    # Bytes and operation counts
    "bytes_read": "sum",
    "bytes_written": "sum",
    "posix_reads": "sum",
    "posix_writes": "sum",
    "posix_opens": "sum",
    "posix_stats": "sum",
    "mpiio_bytes_read": "sum",
    "mpiio_bytes_written": "sum",
    "mpiio_coll_reads": "sum",
    "mpiio_coll_writes": "sum",
    "mpiio_indep_reads": "sum",
    "mpiio_indep_writes": "sum",
    "stdio_bytes_read": "sum",
    "stdio_bytes_written": "sum",

    # Job/runtime metadata from Darshan
    "runtime": "max",
    "nprocs": "max",

    # Rank timing and imbalance
    "slowest_rank_time": "max",
    "fastest_rank_time": "max",
    "variance_rank_time": "max",
    "rank_imbalance": "max",
    "rank_time_imbalance": "max",
    "rank_time_gap": "max",

    # Heatmap-derived timing
    "io_time_frac": "max",
    "io_density": "max",
    "io_active_bins": "sum",
    "io_phase_start_frac": "min",
    "io_phase_end_frac": "max",
    "io_read_time_frac": "max",
    "io_write_time_frac": "max",
    "io_read_density": "max",
    "io_write_density": "max",
    "io_read_active_bins": "sum",
    "io_write_active_bins": "sum",
    "io_read_phase_start_frac": "min",
    "io_read_phase_end_frac": "max",
    "io_write_phase_start_frac": "min",
    "io_write_phase_end_frac": "max",
    "io_rw_overlap_frac": "max",
    "io_max_gap_bins": "max",
    "io_mean_gap_bins": "mean",
    "io_n_io_bursts": "max",

    # I/O structure
    "seq_read_ratio": "mean",
    "seq_write_ratio": "mean",
    "small_read_ratio": "mean",
    "large_read_ratio": "mean",
    "small_write_ratio": "mean",
    "cons_write_ratio": "mean",
    "write_dominance": "mean",
    "mpiio_coll_ratio": "mean",
    "io_read_front_heavy": "max",
    "io_write_back_heavy": "max",

    # POSIX time breakdown
    "posix_meta_time": "max",
    "posix_read_time": "max",
    "posix_write_time": "max",
    "meta_time_frac": "mean",
    "read_time_frac": "mean",
    "write_time_frac": "mean",

    # Module/status indicators
    "has_posix": "max",
    "has_mpiio": "max",
    "has_heatmap": "max",
    "has_lustre": "max",
    "fs_lustre_grand": "max",
    "fs_lustre_eagle": "max",

    # Metadata/configuration features
    "cb_nodes": "max",
    "unique_files": "sum",
    "metadata_ops_per_gb": "mean",
    "mem_not_aligned_ratio": "mean",
    "file_not_aligned_ratio": "mean",
}.items() if k in dm.columns}

dm_agg = dm.groupby("job_id").agg(agg_spec).reset_index()
dm_agg["darshan_file_count"] = dm.groupby("job_id").size().reindex(dm_agg["job_id"]).values

combined = (
    job_df
    .merge(gm, on="JOB_NAME", how="left")
    .merge(dm_agg, on="job_id", how="left")
)

# Executable is recovered from Darshan in this public dataset. In the paper's
# predictive experiment, executable frequency is treated as workload-context
# available at launch/early runtime, not as scheduler-only metadata.
# It is NOT a current-job performance counter.
if OLD_EXE_PATH.exists():
    old_exe = pd.read_csv(OLD_EXE_PATH, usecols=["fname", "executable"], low_memory=False)
    old_exe["job_id"] = old_exe["fname"].astype(str).str.extract(r"^(\d+)", expand=False)
    old_exe["exec_key"] = (
        old_exe["executable"]
        .astype("string")
        .str.strip()
        .replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "0": pd.NA})
        .str.replace(r".*/", "", regex=True)
    )
    exec_by_job = (
        old_exe.dropna(subset=["job_id", "exec_key"])
        .groupby("job_id")["exec_key"]
        .agg(lambda x: "|".join(sorted(set(map(str, x)))))
        .reset_index()
    )
    combined = combined.merge(exec_by_job, on="job_id", how="left")
else:
    combined["exec_key"] = pd.NA

combined["gpu_util_mean"] = combined.get("util_mean", np.nan)
for c in ["io_read_front_heavy", "io_write_back_heavy"]:
    if c in combined.columns:
        combined[c] = combined[c].fillna(0).astype(bool)

combined["QUEUED_TIMESTAMP"] = pd.to_datetime(combined["QUEUED_TIMESTAMP"], errors="coerce")
combined["END_TIMESTAMP"] = pd.to_datetime(combined["END_TIMESTAMP"], errors="coerce")

print(f"Combined rows: {len(combined):,}")
print(f"Executable coverage overall: {combined['exec_key'].notna().mean()*100:.1f}%")

# =============================================================================
# 2. TAXONOMY + FINAL PAPER RELABELS
# =============================================================================

box("2. TAXONOMY + FINAL PAPER RELABELS")

_stdout = sys.stdout
try:
    sys.stdout = open(os.devnull, "w")
    combined = classify_crosslayer(combined)
finally:
    try:
        sys.stdout.close()
    except Exception:
        pass
    sys.stdout = _stdout

# Purify IO_Bottlenecked: io_time_frac <= 5% is incidental, not substantive.
m_io = combined["crosslayer_tier"].eq("IO_Bottlenecked")
m_inc = combined["io_time_frac"].fillna(0) <= 0.05
combined.loc[m_io & m_inc, ["crosslayer_tier", "diagnostic_tier"]] = "Incidental_IO_Low_GPU"

# Split old Balanced into paper-facing Ideal and Moderate.
m_bal = combined["crosslayer_tier"].eq("Balanced")
m_ideal = combined["gpu_util_mean"].fillna(0) >= 70.0
combined.loc[m_bal & m_ideal, ["crosslayer_tier", "diagnostic_tier"]] = "Ideal_Compute_With_IO"
combined.loc[m_bal & ~m_ideal, ["crosslayer_tier", "diagnostic_tier"]] = "Moderate_Compute_With_IO"

# Split Ghost into low-power Idle (internal Ghost) vs power-elevated hidden activity.
m_ghost = combined["crosslayer_tier"].eq("Ghost")
m_hidden = combined["power_mean"].fillna(0) >= 50
combined.loc[m_ghost & m_hidden, ["crosslayer_tier", "diagnostic_tier"]] = "Idle_Hidden_Activity"

# Final wasteful target. Idle_Hidden_Activity is informational, not wasteful.
WASTEFUL = {
    "Ghost",
    "Scale_Waster",
    "IO_Bottlenecked",
    "Failed_Job",
    "Quick_Cancel",
    "GPU_Idle_Timeout",
}
combined["is_wasteful"] = combined["crosslayer_tier"].isin(WASTEFUL).astype(int)

combined.to_csv(OUT_COMBINED, index=False)
print(f"Wrote {OUT_COMBINED}")

total_jobs = len(combined)
total_gpu = combined["gpu_hours"].sum()
print(f"Total jobs: {total_jobs:,}")
print(f"Total allocated GPU-hours: {total_gpu:,.0f}")

# =============================================================================
# 3. COVERAGE + TAXONOMY TABLE
# =============================================================================

box("3. COVERAGE + TAXONOMY TABLE")

n_dcgm = int(combined["has_gpu"].sum())
n_dar = int(combined["darshan_present"].sum())
n_dar_io = int(combined["io_detected"].sum())
n_both = int((combined["has_gpu"] & combined["darshan_present"]).sum())

print(f"DCGM telemetry:      {n_dcgm:,} ({pct(n_dcgm, total_jobs):.1f}%)")
print(f"Darshan attached:    {n_dar:,} ({pct(n_dar, total_jobs):.1f}%)")
print(f"Darshan I/O detected:{n_dar_io:,} ({pct(n_dar_io, total_jobs):.1f}%)")
print(f"Both DCGM+Darshan:   {n_both:,} ({pct(n_both, total_jobs):.1f}%)")

TIER_GROUPS = [
    ("Excluded from predictive modeling", [
        "No_GPU_Telemetry", "No_GPU_With_Darshan",
        "Short_No_GPU", "Short_No_GPU_With_IO",
    ]),
    ("Under-utilized", [
        "Quick_Cancel", "Failed_Job", "GPU_Idle_Timeout",
        "Ghost", "Scale_Waster", "IO_Bottlenecked",
    ]),
    ("Productive", [
        "Ideal_Compute_With_IO", "Compute_Bound",
    ]),
    ("Informational", [
        "Moderate_Compute_No_IO", "Moderate_Compute_With_IO",
        "Incidental_IO_Low_GPU", "Low_Efficiency", "Idle_Hidden_Activity",
    ]),
]

print(f"\n{'Tier':<34} {'Jobs':>9} {'GPU-hrs':>14} {'% GPU-hrs':>10}")
print("-" * 72)
grand_jobs = 0
grand_gpu = 0.0

for group, tiers in TIER_GROUPS:
    print(f"\n[{group}]")
    g_jobs = 0
    g_gpu = 0.0
    for tier in tiers:
        sub = combined[combined["crosslayer_tier"].eq(tier)]
        n = len(sub)
        gh = sub["gpu_hours"].sum()
        g_jobs += n
        g_gpu += gh
        grand_jobs += n
        grand_gpu += gh
        print(f"{paper_name(tier):<34} {n:>9,} {gh:>14,.0f} {pct(gh, total_gpu):>9.1f}%")
    print(f"{'Subtotal':<34} {g_jobs:>9,} {g_gpu:>14,.0f} {pct(g_gpu, total_gpu):>9.1f}%")

print("-" * 72)
print(f"{'TOTAL':<34} {grand_jobs:>9,} {grand_gpu:>14,.0f} {pct(grand_gpu, total_gpu):>9.1f}%")

under_gpu = combined.loc[combined["crosslayer_tier"].isin(WASTEFUL), "gpu_hours"].sum()
print(f"\nUnsuccessful or under-utilized GPU-hours: {under_gpu:,.0f} ({pct(under_gpu, total_gpu):.1f}%)")

# =============================================================================
# 4. CROSS-LAYER INTERPRETATION: PHASE TABLE + CORRELATION + SCALE POWER
# =============================================================================

box("4. CROSS-LAYER INTERPRETATION")

sig_df = combined[["util_mean", "io_time_frac", "BWio_MB"]].dropna()
r_io, _ = spearmanr(sig_df["util_mean"], sig_df["io_time_frac"])
r_bw, _ = spearmanr(sig_df["util_mean"], sig_df["BWio_MB"])
print(f"Spearman r(GPU_util, io_time_frac) = {r_io:.3f} (N={len(sig_df):,})")
print(f"Spearman r(GPU_util, BWio_MB)      = {r_bw:.3f}")

phase_df = combined.dropna(subset=["util_phase1", "util_phase2", "util_phase3"]).copy()
phase_df["io_end_display"] = (
    pd.to_numeric(phase_df["io_phase_end_frac"], errors="coerce")
    .fillna(0)
    .clip(0, 1)
)

phase_groups = {
    "Idle-like/Timeout": phase_df["crosslayer_tier"].isin([
        "Ghost", "Idle_Hidden_Activity", "GPU_Idle_Timeout"
    ]),
    "IO_Bottlenecked": phase_df["crosslayer_tier"].eq("IO_Bottlenecked"),
    "Scale_Inefficient": phase_df["crosslayer_tier"].eq("Scale_Waster"),
    "Compute_Bound": phase_df["crosslayer_tier"].eq("Compute_Bound"),
}

print(f"\n{'Tier/group':<26} {'n':>8} {'P1':>6} {'P2':>6} {'P3':>6} {'io_end':>8}")
print("-" * 68)
phase_rows = {}
for label, mask in phase_groups.items():
    sub = phase_df[mask]
    row = {
        "n": len(sub),
        "P1": sub["util_phase1"].mean(),
        "P2": sub["util_phase2"].mean(),
        "P3": sub["util_phase3"].mean(),
        "io_end": sub["io_end_display"].median(),
    }
    phase_rows[label] = row
    print(f"{label:<26} {row['n']:>8,} {row['P1']:>6.1f} {row['P2']:>6.1f} "
          f"{row['P3']:>6.1f} {row['io_end']:>8.3f}")

sw = combined[combined["crosslayer_tier"].eq("Scale_Waster")].copy()
if len(sw):
    sw_pwr = sw["power_mean"].notna()
    sw_high = (sw.loc[sw_pwr, "power_mean"] >= 50).sum()
    sw_low = (sw.loc[sw_pwr, "power_mean"] < 50).sum()
    print(f"\nScale_Inefficient power validation:")
    print(f"  total={len(sw):,}, power>=50W={sw_high:,} ({pct(sw_high, sw_pwr.sum()):.1f}%), "
          f"power<50W={sw_low:,} ({pct(sw_low, sw_pwr.sum()):.1f}%)")

# =============================================================================
# 5. I/O-MEDIATED UNDER-UTILIZATION DECOMPOSITION
# =============================================================================

box("5. I/O-MEDIATED UNDER-UTILIZATION DECOMPOSITION")

io_b = combined[combined["crosslayer_tier"].eq("IO_Bottlenecked")].copy()
print(f"IO_Bottlenecked jobs: {len(io_b):,}")

if len(io_b):
    m_bw = io_b["BWio_MB"].fillna(np.inf) < 1000
    m_meta = io_b["metadata_ops_per_gb"].fillna(0) > 1000
    m_imb = io_b["rank_time_imbalance"].fillna(0) > 2.0
    flags = pd.DataFrame({"Bandwidth-bound": m_bw, "Metadata-bound": m_meta, "Rank-imbalance": m_imb})
    n_flags = flags.sum(axis=1)
    mode_counts = {
        "Bandwidth-bound": int((m_bw & (n_flags == 1)).sum()),
        "Metadata-bound": int((m_meta & (n_flags == 1)).sum()),
        "Rank-imbalance": int((m_imb & (n_flags == 1)).sum()),
        "Mixed": int((n_flags > 1).sum()),
    }
    print("\nBottleneck signature")
    for label, n in mode_counts.items():
        print(f"  {label:<20} {n:>6,} ({pct(n, len(io_b)):>5.1f}%)")

    if {"io_read_time_frac", "io_write_time_frac"}.issubset(io_b.columns):
        rt = io_b["io_read_time_frac"].fillna(0)
        wt = io_b["io_write_time_frac"].fillna(0)
        n_read = int((rt > 1.5 * wt).sum())
        n_write = int((wt > 1.5 * rt).sum())
        n_bal = len(io_b) - n_read - n_write
        print("\nRead/write temporal dominance")
        print(f"  Read-dominated          {n_read:>6,} ({pct(n_read, len(io_b)):>5.1f}%)")
        print(f"  Write-dominated         {n_write:>6,} ({pct(n_write, len(io_b)):>5.1f}%)")
        print(f"  Balanced/interleaved    {n_bal:>6,} ({pct(n_bal, len(io_b)):>5.1f}%)")

    if "io_n_io_bursts" in io_b.columns:
        bursts = io_b["io_n_io_bursts"].fillna(1)
        n_sust = int((bursts <= 2).sum())
        n_bursty = int((bursts >= 5).sum())
        n_mod = len(io_b) - n_sust - n_bursty
        overlap = io_b.get("io_rw_overlap_frac", pd.Series(0, index=io_b.index)).fillna(0)
        n_conc = int((overlap > 0.1).sum())
        print("\nBurstiness")
        print(f"  Sustained <=2 bursts    {n_sust:>6,} ({pct(n_sust, len(io_b)):>5.1f}%)")
        print(f"  Moderate 3-4 bursts     {n_mod:>6,} ({pct(n_mod, len(io_b)):>5.1f}%)")
        print(f"  Bursty >=5 bursts       {n_bursty:>6,} ({pct(n_bursty, len(io_b)):>5.1f}%)")
        print(f"  Concurrent R/W >10%     {n_conc:>6,} ({pct(n_conc, len(io_b)):>5.1f}%)")

    if "meta_time_frac" in io_b.columns:
        time_meta = io_b["meta_time_frac"].fillna(0) > 0.5
        op_meta = io_b["metadata_ops_per_gb"].fillna(0) > 1000
        print("\nIndependent metadata-time validation")
        print(f"  meta_time_frac > 0.5    {int(time_meta.sum()):>6,} ({pct(time_meta.sum(), len(io_b)):>5.1f}%)")
        print(f"  op+time both metadata   {int((op_meta & time_meta).sum()):>6,}")

if "cb_nodes" in dm.columns:
    mpi_jobs = dm_agg[dm_agg.get("has_mpiio", 0).fillna(0).astype(bool)]
    if len(mpi_jobs):
        default_cb4 = int((mpi_jobs["cb_nodes"].fillna(-1).astype(int) == 4).sum())
        print(f"\nMPI-IO jobs: {len(mpi_jobs):,}; cb_nodes=4: {default_cb4:,} ({pct(default_cb4, len(mpi_jobs)):.1f}%)")

# =============================================================================
# 6. USER CONCENTRATION + PERSISTENCE
# =============================================================================

box("6. USER CONCENTRATION + PERSISTENCE")

n_users = combined["USERNAME_GENID"].nunique()
n_projects = combined["PROJECT_NAME_GENID"].nunique()
wdf = combined[combined["is_wasteful"].eq(1)].copy()
waste_gpu = wdf["gpu_hours"].sum()
user_waste_gpu = wdf.groupby("USERNAME_GENID")["gpu_hours"].sum().sort_values(ascending=False)

top10_share = user_waste_gpu.head(10).sum() / waste_gpu
k_5pct = max(1, int(np.ceil(n_users * 0.05)))
top5pct_share = user_waste_gpu.head(k_5pct).sum() / waste_gpu

print(f"Unique users: {n_users:,}")
print(f"Unique projects: {n_projects:,}")
print(f"Top 10 users share of under-utilized GPU-hours: {top10_share*100:.1f}%")
print(f"Top {k_5pct} users (5%) share of under-utilized GPU-hours: {top5pct_share*100:.1f}%")

cs = combined.sort_values(["USERNAME_GENID", "QUEUED_TIMESTAMP"]).reset_index(drop=True)
cs["prev_w"] = cs.groupby("USERNAME_GENID")["is_wasteful"].shift(1).fillna(0)
cs["in_burst"] = ((cs["is_wasteful"].eq(1)) & (cs["prev_w"].eq(1))).astype(int)

SUBST = {"Ghost", "IO_Bottlenecked", "Scale_Waster", "Failed_Job", "GPU_Idle_Timeout"}
sub_w = cs[cs["crosslayer_tier"].isin(SUBST)]
sub_b = cs[cs["crosslayer_tier"].isin(SUBST) & cs["in_burst"].eq(1)]
all_w = cs[cs["is_wasteful"].eq(1)]
all_b = cs[cs["in_burst"].eq(1)]

print(f"\nSubstantive under-utilized jobs in burst: {len(sub_b):,}/{len(sub_w):,} ({pct(len(sub_b), len(sub_w)):.1f}%)")
print(f"Including Quick_Cancel in burst:          {len(all_b):,}/{len(all_w):,} ({pct(len(all_b), len(all_w)):.1f}%)")

# =============================================================================
# 7. I/O ACTIONABLE CONCENTRATION + T_IO-ONLY REPEATABILITY
# =============================================================================

box("7. I/O ACTIONABLE CONCENTRATION + T_IO-ONLY REPEATABILITY")

def bin01(s, scale=10):
    return (
        pd.to_numeric(s, errors="coerce")
        .fillna(0)
        .clip(0, 1)
        .mul(scale)
        .round()
        .astype(int)
    )

# Job-level Darshan table joined to user/project.
job_user = job_df[["job_id", "USERNAME_GENID", "PROJECT_NAME_GENID"]].drop_duplicates("job_id")
job_dm = dm_agg.merge(job_user, on="job_id", how="left").copy()

job_dm["total_bytes"] = job_dm.get("bytes_read", 0).fillna(0) + job_dm.get("bytes_written", 0).fillna(0)
job_dm["total_volume_gib"] = job_dm["total_bytes"] / (1024 ** 3)
job_dm["runtime_s"] = pd.to_numeric(job_dm.get("runtime", np.nan), errors="coerce")

# Use Darshan heatmap I/O-active window when available.
job_dm["t_io_s"] = job_dm["runtime_s"] * pd.to_numeric(job_dm.get("io_time_frac", np.nan), errors="coerce")
valid_tio = job_dm["t_io_s"].notna() & (job_dm["t_io_s"] > 0)
job_dm["bw_source"] = np.where(valid_tio, "T_io", "T_wall")
denom = np.where(valid_tio, job_dm["t_io_s"], job_dm["runtime_s"])
job_dm["bw_io_mbs"] = job_dm["total_bytes"] / np.maximum(denom, 1) / 1e6

# Actionable = top decile high-volume I/O, matching the REX-IO style.
p90_gib = job_dm["total_volume_gib"].quantile(0.90)
job_dm["volume_tier"] = np.where(job_dm["total_volume_gib"] >= p90_gib, "Actionable", "Non_Actionable")

act = job_dm[job_dm["volume_tier"].eq("Actionable")].copy()
act["diagnostic_class"] = np.select(
    [
        act["bw_io_mbs"] < 300,
        (act["bw_io_mbs"] >= 300) & (act["bw_io_mbs"] < 6000),
    ],
    ["App_Limited", "Parallelism_Bounded"],
    default="Other"
)

app = act[act["diagnostic_class"].eq("App_Limited")]
par = act[act["diagnostic_class"].eq("Parallelism_Bounded")]
both_users = set(app["USERNAME_GENID"].dropna()) & set(par["USERNAME_GENID"].dropna())

print(f"Actionable high-volume I/O jobs: {len(act):,} ({pct(len(act), len(job_dm)):.1f}% of Darshan-covered jobs)")
print(f"Actionable users: {act['USERNAME_GENID'].nunique():,}")
for label, sub in [("Application-Limited", app), ("Parallelism-Bounded", par)]:
    user_vol = sub.groupby("USERNAME_GENID")["total_volume_gib"].sum().sort_values(ascending=False)
    top5 = user_vol.head(5).sum() / user_vol.sum() if len(user_vol) else np.nan
    print(f"{label}: jobs={len(sub):,}, users={sub['USERNAME_GENID'].nunique():,}, top5 volume share={top5*100:.1f}%")
print(f"Users in both diagnostic classes: {len(both_users):,}")

# Repeatability: validated T_io only, inefficient classes only, executable attribution required.
ineff_tio = act[
    act["diagnostic_class"].isin(["App_Limited", "Parallelism_Bounded"])
    & act["bw_source"].eq("T_io")
].copy()

exec_map = combined[["job_id", "exec_key"]].drop_duplicates("job_id")
ineff_tio = ineff_tio.merge(exec_map, on="job_id", how="left")
ineff_tio = ineff_tio.dropna(subset=["exec_key", "USERNAME_GENID"]).copy()

# Coarse signature.
ineff_tio["user_id"] = ineff_tio["USERNAME_GENID"]
ineff_tio["project_id"] = ineff_tio["PROJECT_NAME_GENID"]
ineff_tio["vol_bin"] = np.floor(np.log2(ineff_tio["total_volume_gib"].clip(lower=1))).astype(int)
ineff_tio["bw_bin"] = np.floor(np.log10(ineff_tio["bw_io_mbs"].clip(lower=1))).astype(int)

for c in [
    "small_write_ratio", "seq_write_ratio", "cons_write_ratio",
    "meta_time_frac", "mpiio_coll_ratio", "file_not_aligned_ratio",
]:
    ineff_tio[c + "_bin"] = bin01(ineff_tio[c]) if c in ineff_tio.columns else 0

for c in ["fs_lustre_eagle", "fs_lustre_grand"]:
    if c not in ineff_tio.columns:
        ineff_tio[c] = 0

sig_keys = [
    "user_id", "project_id", "exec_key", "diagnostic_class",
    "vol_bin", "bw_bin",
    "small_write_ratio_bin", "seq_write_ratio_bin", "cons_write_ratio_bin",
    "meta_time_frac_bin", "mpiio_coll_ratio_bin", "file_not_aligned_ratio_bin",
    "fs_lustre_eagle", "fs_lustre_grand",
]
ineff_tio["sig_id"] = ineff_tio.groupby(sig_keys, dropna=False).ngroup()

sig = (
    ineff_tio.groupby("sig_id")
    .agg(
        jobs=("job_id", "nunique"),
        users=("user_id", "nunique"),
        total_gib=("total_volume_gib", "sum"),
        median_bw=("bw_io_mbs", "median"),
        diagnostic_class=("diagnostic_class", "first"),
    )
    .reset_index()
)
repeat_sig = sig[sig["jobs"] >= 2]
repeat_jobs = ineff_tio[ineff_tio["sig_id"].isin(repeat_sig["sig_id"])].copy()

print("\nT_io-only repeatability")
print(f"T_io-only inefficient users:        {ineff_tio['user_id'].nunique():,}")
print(f"Users with repeated signatures:     {repeat_jobs['user_id'].nunique():,} ({pct(repeat_jobs['user_id'].nunique(), ineff_tio['user_id'].nunique()):.1f}%)")
print(f"T_io-only inefficient jobs:         {ineff_tio['job_id'].nunique():,}")
print(f"Jobs in repeated signatures:        {repeat_jobs['job_id'].nunique():,} ({pct(repeat_jobs['job_id'].nunique(), ineff_tio['job_id'].nunique()):.1f}%)")
print(f"T_io-only inefficient volume:       {ineff_tio['total_volume_gib'].sum():,.0f} GiB")
print(f"Repeated-signature volume:          {repeat_jobs['total_volume_gib'].sum():,.0f} GiB ({pct(repeat_jobs['total_volume_gib'].sum(), ineff_tio['total_volume_gib'].sum()):.1f}%)")

bw = repeat_jobs["bw_io_mbs"].replace([np.inf, -np.inf], np.nan).dropna()
print(f"Repeated jobs median BW_IO:         {bw.median():,.1f} MB/s")
print("\nRepeated jobs by diagnostic class")
if len(repeat_jobs):
    print(
        repeat_jobs.groupby("diagnostic_class")
        .agg(
            users=("user_id", "nunique"),
            jobs=("job_id", "nunique"),
            total_gib=("total_volume_gib", "sum"),
            med_bw=("bw_io_mbs", "median"),
        )
        .to_string(formatters={
            "total_gib": "{:,.0f}".format,
            "med_bw": "{:,.1f}".format,
        })
    )

# =============================================================================
# 8. FEATURE ENGINEERING + CORE ML
# =============================================================================
def build_entity_hist(train_df, target_df, key_cols, cols, prefix,
                      lookback_days=7, n_history=10):
    """
    Leakage-safe history for entity keys:
      key_cols=["exec_key"] -> same-executable history
      key_cols=["USERNAME_GENID", "exec_key"] -> same-user+same-executable history

    Current executable identity is allowed, but history uses only prior jobs:
      prior END_TIMESTAMP < current QUEUED_TIMESTAMP
    """
    rows = []
    cols = [c for c in cols if c in train_df.columns]
    lb_ns = np.timedelta64(lookback_days, "D")

    # Default rows for all targets, including missing exec_key.
    for _, cur in target_df[["JOB_NAME"] + key_cols].iterrows():
        row = {"JOB_NAME": cur["JOB_NAME"]}
        row[f"{prefix}_job_count"] = 0
        row[f"{prefix}_waste_rate"] = 0.0
        row[f"{prefix}_mean_runtime"] = -1
        row[f"{prefix}_mean_nodes"] = -1
        row[f"{prefix}_mean_walltime"] = -1
        for c in cols:
            row[f"{prefix}_hist_{c}"] = -1
        rows.append(row)

    out = pd.DataFrame(rows).set_index("JOB_NAME")

    valid_targets = target_df.dropna(subset=key_cols).copy()
    if len(valid_targets) == 0:
        return out.reset_index()

    target_keys = valid_targets[key_cols].drop_duplicates()

    work = train_df.dropna(subset=key_cols).copy()
    work = work.merge(target_keys, on=key_cols, how="inner")

    for _, grp in work.groupby(key_cols, sort=False):
        grp = grp.sort_values("QUEUED_TIMESTAMP").reset_index(drop=True)

        qts = grp["QUEUED_TIMESTAMP"].values
        ets = grp["END_TIMESTAMP"].values
        job_names = grp["JOB_NAME"].values
        hist_mat = grp[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)

        for i, job in enumerate(job_names):
            if job not in out.index:
                continue

            mask = (ets < qts[i]) & (ets >= qts[i] - lb_ns)
            past_idx = np.where(mask)[0][-n_history:]

            if len(past_idx) == 0:
                continue

            p = grp.iloc[past_idx]

            out.loc[job, f"{prefix}_job_count"] = int(mask.sum())
            out.loc[job, f"{prefix}_waste_rate"] = p["is_wasteful"].mean()
            out.loc[job, f"{prefix}_mean_runtime"] = p["RUNTIME_SECONDS"].mean()
            out.loc[job, f"{prefix}_mean_nodes"] = p["NODES_REQUESTED"].mean()
            out.loc[job, f"{prefix}_mean_walltime"] = p["WALLTIME_SECONDS"].mean()

            with np.errstate(all="ignore"):
                means = np.nanmean(hist_mat[past_idx, :], axis=0)

            means = np.where(np.isnan(means), -1, means)

            for c, v in zip(cols, means):
                out.loc[job, f"{prefix}_hist_{c}"] = v

    return out.reset_index()
box("8. FEATURE ENGINEERING + CORE ML")

combined = combined.sort_values("QUEUED_TIMESTAMP").reset_index(drop=True)
combined["submit_hour"] = combined["QUEUED_TIMESTAMP"].dt.hour
combined["submit_dow"] = combined["QUEUED_TIMESTAMP"].dt.dayofweek
combined["submit_month"] = combined["QUEUED_TIMESTAMP"].dt.month

train_df = combined[combined["use_for_training"]].copy().reset_index(drop=True)

TELEM_TIERS = {
    "Ghost", "Idle_Hidden_Activity", "Scale_Waster", "IO_Bottlenecked",
    "Compute_Bound", "Moderate_Compute_No_IO", "Low_Efficiency",
    "GPU_Idle_Timeout", "Ideal_Compute_With_IO", "Moderate_Compute_With_IO",
    "Incidental_IO_Low_GPU",
}
telem_df = train_df[train_df["crosslayer_tier"].isin(TELEM_TIERS)].copy().reset_index(drop=True)
split_t = int(len(telem_df) * 0.80)

# Train-prefix encodings.
queue_freq = telem_df["QUEUE_NAME"].iloc[:split_t].value_counts()
telem_df["queue_freq"] = telem_df["QUEUE_NAME"].map(queue_freq).fillna(0)

# Executable/workload context.
# Important: counts are built only from the training prefix.
# Missing executable IDs map to 0; we do not create a giant "missing" bucket.
exe_freq = telem_df["exec_key"].iloc[:split_t].dropna().value_counts()
telem_df["executable_freq"] = telem_df["exec_key"].map(exe_freq).fillna(0)

le = LabelEncoder()
le.fit(telem_df["SCIENCE_FIELD"].iloc[:split_t].astype(str))
known_fields = set(le.classes_)
telem_df["SCIENCE_FIELD_enc"] = telem_df["SCIENCE_FIELD"].astype(str).apply(
    lambda x: le.transform([x])[0] if x in known_fields else -1
)

# M1 uses scheduler-visible metadata plus executable/workload context.
# executable_freq is not a performance counter; it is an executable identity
# frequency computed from the training prefix.
groupA = [
    "NODES_REQUESTED", "WALLTIME_SECONDS", "CORES_REQUESTED",
    "submit_hour", "submit_dow", "submit_month",
    "queue_freq", "SCIENCE_FIELD_enc", "executable_freq",
]

def build_hist(train_df, target_jobs, cols, lookback_days=7, n_history=10):
    rows = []
    target_set = set(target_jobs)
    lb_ns = np.timedelta64(lookback_days, "D")
    cols = [c for c in cols if c in train_df.columns]

    for user, grp in train_df.groupby("USERNAME_GENID", sort=False):
        grp = grp.sort_values("QUEUED_TIMESTAMP").reset_index(drop=True)
        qts = grp["QUEUED_TIMESTAMP"].values
        ets = grp["END_TIMESTAMP"].values
        job_names = grp["JOB_NAME"].values
        hist_mat = grp[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)

        for i, name in enumerate(job_names):
            if name not in target_set:
                continue
            mask = (ets < qts[i]) & (ets >= qts[i] - lb_ns)
            past_idx = np.where(mask)[0][-n_history:]
            row = {"JOB_NAME": name}

            if len(past_idx):
                p = grp.iloc[past_idx]
                row.update({
                    "user_job_count": int(mask.sum()),
                    "user_mean_runtime": p["RUNTIME_SECONDS"].mean(),
                    "user_walltime_efficiency": (
                        p["RUNTIME_SECONDS"] / p["WALLTIME_SECONDS"].replace(0, np.nan)
                    ).mean(),
                    "user_fail_rate": (p["EXIT_STATUS"] != 0).mean(),
                    "user_quick_cancel_rate": (p["RUNTIME_SECONDS"] < 60).mean(),
                    "user_mean_nodes": p["NODES_REQUESTED"].mean(),
                    "user_mean_walltime": p["WALLTIME_SECONDS"].mean(),
                })
                with np.errstate(all="ignore"):
                    means = np.nanmean(hist_mat[past_idx, :], axis=0)
                means = np.where(np.isnan(means), -1, means)
                for c, v in zip(cols, means):
                    row[f"hist_{c}"] = v
            else:
                cur = grp.iloc[i]
                row.update({
                    "user_job_count": 0,
                    "user_mean_runtime": cur["WALLTIME_SECONDS"],
                    "user_walltime_efficiency": 0.5,
                    "user_fail_rate": 0.0,
                    "user_quick_cancel_rate": 0.0,
                    "user_mean_nodes": cur["NODES_REQUESTED"],
                    "user_mean_walltime": cur["WALLTIME_SECONDS"],
                })
                for c in cols:
                    row[f"hist_{c}"] = -1
            rows.append(row)

    return pd.DataFrame(rows)

hist_cols = [
    "util_mean", "idle_frac", "zero_util_frac",
    "power_efficiency", "io_time_frac", "bytes_per_gpu_hour",
]
print(f"Building compact history ({LOOKBACK_DAYS}-day lookback, last {N_HIST} jobs)...", flush=True)
hist_df = build_hist(train_df, telem_df["JOB_NAME"].tolist(), hist_cols, LOOKBACK_DAYS, N_HIST)
telem_df = telem_df.merge(hist_df, on="JOB_NAME", how="left")

groupB = [
    "user_job_count", "user_mean_runtime", "user_walltime_efficiency",
    "user_fail_rate", "user_quick_cancel_rate", "user_mean_nodes", "user_mean_walltime",
]
groupC = [f"hist_{c}" for c in hist_cols if f"hist_{c}" in telem_df.columns]

# -------------------------------------------------------------------------
# Same-executable and same-user+same-executable history for compact M3.
# -------------------------------------------------------------------------
box("8b. BUILD EXECUTABLE HISTORY FOR COMPACT MODELS")

compact_exec_hist_cols = hist_cols

exec_hist_df = build_entity_hist(
    train_df=train_df,
    target_df=telem_df[["JOB_NAME", "exec_key"]],
    key_cols=["exec_key"],
    cols=compact_exec_hist_cols,
    prefix="exec",
    lookback_days=LOOKBACK_DAYS,
    n_history=N_HIST,
)

user_exec_hist_df = build_entity_hist(
    train_df=train_df,
    target_df=telem_df[["JOB_NAME", "USERNAME_GENID", "exec_key"]],
    key_cols=["USERNAME_GENID", "exec_key"],
    cols=compact_exec_hist_cols,
    prefix="user_exec",
    lookback_days=LOOKBACK_DAYS,
    n_history=N_HIST,
)

telem_df = telem_df.merge(exec_hist_df, on="JOB_NAME", how="left")
telem_df = telem_df.merge(user_exec_hist_df, on="JOB_NAME", how="left")

groupE = [
    "exec_job_count",
    "exec_waste_rate",
    "exec_mean_runtime",
    "exec_mean_nodes",
    "exec_mean_walltime",
] + [f"exec_hist_{c}" for c in compact_exec_hist_cols if f"exec_hist_{c}" in telem_df.columns]

groupF = [
    "user_exec_job_count",
    "user_exec_waste_rate",
    "user_exec_mean_runtime",
    "user_exec_mean_nodes",
    "user_exec_mean_walltime",
] + [f"user_exec_hist_{c}" for c in compact_exec_hist_cols if f"user_exec_hist_{c}" in telem_df.columns]

print(f"User-history features:             {len(groupB) + len(groupC)}")
print(f"Executable-history features:       {len(groupE)}")
print(f"User+executable-history features:  {len(groupF)}")

# Final compact feature sets.
groupA0 = [
    "NODES_REQUESTED", "WALLTIME_SECONDS", "CORES_REQUESTED",
    "submit_hour", "submit_dow", "submit_month",
    "queue_freq", "SCIENCE_FIELD_enc",
]
groupA1 = groupA0 + ["executable_freq"]

M0_feats = groupA0
M1_feats = groupA1
M2_feats = groupB + groupC
M3_feats = groupA1 + groupB + groupC + groupE + groupF

print(
    f"M0 features: {len([f for f in M0_feats if f in telem_df.columns])}; "
    f"M1 features: {len([f for f in M1_feats if f in telem_df.columns])}; "
    f"M2 features: {len([f for f in M2_feats if f in telem_df.columns])}; "
    f"M3 features: {len([f for f in M3_feats if f in telem_df.columns])}"
)

y_train = telem_df["is_wasteful"].iloc[:split_t].values
y_test = telem_df["is_wasteful"].iloc[split_t:].values

print(f"Train jobs: {split_t:,} ({telem_df['QUEUED_TIMESTAMP'].iloc[0].date()} -> {telem_df['QUEUED_TIMESTAMP'].iloc[split_t-1].date()})")
print(f"Test jobs:  {len(telem_df)-split_t:,} ({telem_df['QUEUED_TIMESTAMP'].iloc[split_t].date()} -> {telem_df['QUEUED_TIMESTAMP'].iloc[-1].date()})")
print(
    f"M0 features: {len([f for f in M0_feats if f in telem_df.columns])}; "
    f"M1 features: {len([f for f in M1_feats if f in telem_df.columns])}; "
    f"M2 features: {len([f for f in M2_feats if f in telem_df.columns])}; "
    f"M3 features: {len([f for f in M3_feats if f in telem_df.columns])}"
)

def fit_eval(feats, name, cw=None):
    avail = [f for f in feats if f in telem_df.columns]
    clf = RandomForestClassifier(n_estimators=200, class_weight=cw, n_jobs=-1, random_state=RNG)
    clf.fit(to_X(telem_df.iloc[:split_t], avail), y_train)
    prob = clf.predict_proba(to_X(telem_df.iloc[split_t:], avail))[:, 1]
    pred = clf.predict(to_X(telem_df.iloc[split_t:], avail))
    return {
        "name": name,
        "clf": clf,
        "prob": prob,
        "pred": pred,
        "auc": roc_auc_score(y_test, prob),
        "f1": f1_score(y_test, pred, average="macro"),
        "avail": avail,
    }

M0 = fit_eval(M0_feats, "M0 scheduler only")
M1 = fit_eval(M1_feats, "M1 scheduler + executable context")
M2 = fit_eval(M2_feats, "M2 user history only")
M3 = fit_eval(M3_feats, "M3 cross-layer + executable history")

# Baselines.
dst = DummyClassifier(strategy="stratified", random_state=RNG).fit(np.zeros((split_t, 1)), y_train)
dst_prob = dst.predict_proba(np.zeros((len(y_test), 1)))[:, 1]
dst_pred = dst.predict(np.zeros((len(y_test), 1)))
auc_st = roc_auc_score(y_test, dst_prob)
f1_st = f1_score(y_test, dst_pred, average="macro")

lr_wt = LogisticRegression(max_iter=1000, class_weight="balanced").fit(
    telem_df[["WALLTIME_SECONDS"]].iloc[:split_t].values, y_train
)
wt_prob = lr_wt.predict_proba(telem_df[["WALLTIME_SECONDS"]].iloc[split_t:].values)[:, 1]
wt_pred = lr_wt.predict(telem_df[["WALLTIME_SECONDS"]].iloc[split_t:].values)
auc_wt = roc_auc_score(y_test, wt_prob)
f1_wt = f1_score(y_test, wt_pred, average="macro")

print("\nModel comparison")
print(f"{'Model':<24} {'AUC':>8} {'macro-F1':>10}")
print("-" * 46)
for label, auc, f1 in [
    ("Stratified baseline", auc_st, f1_st),
    ("Walltime-only LR", auc_wt, f1_wt),
    ("M0 scheduler only", M0["auc"], M0["f1"]),
    ("M1 scheduler+exe", M1["auc"], M1["f1"]),
    ("M2 user history", M2["auc"], M2["f1"]),
    ("M3 + exec history", M3["auc"], M3["f1"]),
]:
    print(f"{label:<24} {auc:>8.4f} {f1:>10.3f}")

# Bootstrap CI for M3.
rng = np.random.RandomState(RNG)
boots = []
for _ in range(1000):
    idx = rng.randint(0, len(y_test), len(y_test))
    boots.append(roc_auc_score(y_test[idx], M3["prob"][idx]))
ci_lo, ci_hi = np.percentile(boots, [2.5, 97.5])
print(f"\nM3 bootstrap 95% CI: [{ci_lo:.4f}, {ci_hi:.4f}]")
print(f"M3 lift over M1: {M3['auc'] - M1['auc']:+.4f} AUC")
print(f"M2 lift over M1: {M2['auc'] - M1['auc']:+.4f} AUC")

# Temporal CV.
X_m1_all = to_X(telem_df, M1["avail"])
X_m3_all = to_X(telem_df, M3["avail"])
y_all = telem_df["is_wasteful"].values

m1_cv, m3_cv = [], []
for tr, te in TimeSeriesSplit(n_splits=5).split(X_m3_all):
    rf1 = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=RNG).fit(X_m1_all[tr], y_all[tr])
    rf3 = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=RNG).fit(X_m3_all[tr], y_all[tr])
    m1_cv.append(roc_auc_score(y_all[te], rf1.predict_proba(X_m1_all[te])[:, 1]))
    m3_cv.append(roc_auc_score(y_all[te], rf3.predict_proba(X_m3_all[te])[:, 1]))

print(f"\n5-fold temporal CV:")
print(f"  M1: {np.mean(m1_cv):.3f} ± {np.std(m1_cv):.3f}")
print(f"  M3: {np.mean(m3_cv):.3f} ± {np.std(m3_cv):.3f}")
print(f"  M3 > M1 in {sum(m3 > m1 for m1, m3 in zip(m1_cv, m3_cv))}/5 folds")

# Multi-seed stability.
box("9. MULTI-SEED STABILITY")

SEEDS = [42, 7, 1337, 2024, 31415]
m1_aucs, m2_aucs, m3_aucs = [], [], []
m1_f1s, m2_f1s, m3_f1s = [], [], []

_X_tr_m1 = to_X(telem_df.iloc[:split_t], M1["avail"])
_X_te_m1 = to_X(telem_df.iloc[split_t:], M1["avail"])
_X_tr_m2 = to_X(telem_df.iloc[:split_t], M2["avail"])
_X_te_m2 = to_X(telem_df.iloc[split_t:], M2["avail"])
_X_tr_m3 = to_X(telem_df.iloc[:split_t], M3["avail"])
_X_te_m3 = to_X(telem_df.iloc[split_t:], M3["avail"])

for seed in SEEDS:
    rf1 = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=seed).fit(_X_tr_m1, y_train)
    rf2 = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=seed).fit(_X_tr_m2, y_train)
    rf3 = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=seed).fit(_X_tr_m3, y_train)

    for model, Xte, aucs, f1s in [
        (rf1, _X_te_m1, m1_aucs, m1_f1s),
        (rf2, _X_te_m2, m2_aucs, m2_f1s),
        (rf3, _X_te_m3, m3_aucs, m3_f1s),
    ]:
        prob = model.predict_proba(Xte)[:, 1]
        pred = model.predict(Xte)
        aucs.append(roc_auc_score(y_test, prob))
        f1s.append(f1_score(y_test, pred, average="macro"))

print(f"M1 AUC: {np.mean(m1_aucs):.4f} ± {np.std(m1_aucs):.4f}; F1: {np.mean(m1_f1s):.4f} ± {np.std(m1_f1s):.4f}")
print(f"M2 AUC: {np.mean(m2_aucs):.4f} ± {np.std(m2_aucs):.4f}; F1: {np.mean(m2_f1s):.4f} ± {np.std(m2_f1s):.4f}")
print(f"M3 AUC: {np.mean(m3_aucs):.4f} ± {np.std(m3_aucs):.4f}; F1: {np.mean(m3_f1s):.4f} ± {np.std(m3_f1s):.4f}")
print(f"M3-M1 lift: {(np.array(m3_aucs) - np.array(m1_aucs)).mean():+.4f} ± {(np.array(m3_aucs) - np.array(m1_aucs)).std():.4f}")

# OvR per-tier AUC.
box("10. OVR PER-TIER AUC: M1 vs M2 vs M3")

tier_name = {
    "Ghost": "Idle-like",
    "IO_Bottlenecked": "IO_Bottlenecked",
    "Scale_Waster": "Scale_Inefficient",
    "GPU_Idle_Timeout": "GPU_Idle_Timeout",
}
tiers = ["Ghost", "IO_Bottlenecked", "Scale_Waster", "GPU_Idle_Timeout"]
feature_sets = {"M1": M1["avail"], "M2": M2["avail"], "M3": M3["avail"]}

ovr_rows = []
print(f"{'Tier':<22} {'support':>8} {'M1':>8} {'M2':>8} {'M3':>8} {'Δ M3-M1':>10}")
print("-" * 74)

for tier in tiers:
    y_tr = (telem_df["crosslayer_tier"].iloc[:split_t] == tier).astype(int).values
    y_te = (telem_df["crosslayer_tier"].iloc[split_t:] == tier).astype(int).values
    support = int(y_te.sum())

    if support < 10:
        print(f"{tier_name[tier]:<22} {support:>8,} insufficient positives")
        continue

    aucs = {}
    for model_name, feats in feature_sets.items():
        rf = RandomForestClassifier(n_estimators=200, class_weight="balanced", n_jobs=-1, random_state=RNG)
        rf.fit(to_X(telem_df.iloc[:split_t], feats), y_tr)
        prob = rf.predict_proba(to_X(telem_df.iloc[split_t:], feats))[:, 1]
        aucs[model_name] = roc_auc_score(y_te, prob)

    delta = aucs["M3"] - aucs["M1"]
    print(f"{tier_name[tier]:<22} {support:>8,} {aucs['M1']:>8.4f} {aucs['M2']:>8.4f} {aucs['M3']:>8.4f} {delta:>10.4f}")
    ovr_rows.append({
        "internal_tier": tier,
        "paper_tier": tier_name[tier],
        "support": support,
        "M1": aucs["M1"],
        "M2": aucs["M2"],
        "M3": aucs["M3"],
        "delta": delta,
    })

ovr_df = pd.DataFrame(ovr_rows)
print("\nLaTeX rows:")
for _, r in ovr_df.iterrows():
    tier_latex = r["paper_tier"].replace("_", r"\_")
    print(rf"\texttt{{{tier_latex}}} & {r['M1']:.3f} & {r['M2']:.3f} & {r['M3']:.3f} & ${r['delta']:+.3f}$ \\")

# Cold-start.
box("11. COLD-START STRATIFIED AUC")
test_df = telem_df.iloc[split_t:].copy().reset_index(drop=True)
test_df["pred_proba"] = M3["prob"]

print(f"{'User history bucket':<30} {'n':>8} {'pos%':>7} {'AUC':>8}")
print("-" * 58)
for lo, hi, label in [
    (-0.5, 0.5, "0 prior jobs (cold start)"),
    (0.5, 3.5, "1-3 prior jobs"),
    (3.5, 9.5, "4-9 prior jobs"),
    (9.5, 999, "10 prior jobs (max history)"),
]:
    sub = test_df[(test_df["user_job_count"] > lo) & (test_df["user_job_count"] <= hi)]
    if len(sub) < 50 or sub["is_wasteful"].nunique() < 2:
        print(f"{label:<30} {len(sub):>8,} insufficient")
        continue
    auc = roc_auc_score(sub["is_wasteful"], sub["pred_proba"])
    pos = sub["is_wasteful"].mean() * 100
    print(f"{label:<30} {len(sub):>8,} {pos:>6.1f}% {auc:>8.4f}")

# Optional boosting model comparison.
if RUN_BOOSTING:
    box("12. MODEL COMPARISON: RF vs BOOSTING")
    results = {"RF (M3 baseline)": {"auc": M3["auc"], "f1": M3["f1"]}}

    X_tr_full = to_X(telem_df.iloc[:split_t], M3["avail"])
    X_te_full = to_X(telem_df.iloc[split_t:], M3["avail"])

    hgb = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, max_depth=6, min_samples_leaf=20, random_state=RNG
    )
    hgb.fit(X_tr_full, y_train)
    hgb_prob = hgb.predict_proba(X_te_full)[:, 1]
    hgb_pred = hgb.predict(X_te_full)
    results["HistGradientBoosting"] = {
        "auc": roc_auc_score(y_test, hgb_prob),
        "f1": f1_score(y_test, hgb_pred, average="macro"),
    }

    try:
        import xgboost as xgb
        n_neg = (y_train == 0).sum()
        n_pos = (y_train == 1).sum()
        xgb_model = xgb.XGBClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=n_neg / max(n_pos, 1),
            eval_metric="auc", random_state=RNG, n_jobs=-1, verbosity=0,
        )
        xgb_model.fit(X_tr_full, y_train)
        xgb_prob = xgb_model.predict_proba(X_te_full)[:, 1]
        xgb_pred = xgb_model.predict(X_te_full)
        results["XGBoost"] = {
            "auc": roc_auc_score(y_test, xgb_prob),
            "f1": f1_score(y_test, xgb_pred, average="macro"),
        }
    except Exception as e:
        print(f"Skipping XGBoost: {e}")

    try:
        import lightgbm as lgb
        n_neg = (y_train == 0).sum()
        n_pos = (y_train == 1).sum()
        lgb_model = lgb.LGBMClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=6,
            num_leaves=63, subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=n_neg / max(n_pos, 1),
            random_state=RNG, n_jobs=-1, verbose=-1,
        )
        lgb_model.fit(X_tr_full, y_train)
        lgb_prob = lgb_model.predict_proba(X_te_full)[:, 1]
        lgb_pred = lgb_model.predict(X_te_full)
        results["LightGBM"] = {
            "auc": roc_auc_score(y_test, lgb_prob),
            "f1": f1_score(y_test, lgb_pred, average="macro"),
        }
    except Exception as e:
        print(f"Skipping LightGBM: {e}")

    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr_full)
    X_te_scaled = scaler.transform(X_te_full)
    lr_full = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RNG)
    lr_full.fit(X_tr_scaled, y_train)
    lr_prob = lr_full.predict_proba(X_te_scaled)[:, 1]
    lr_pred = lr_full.predict(X_te_scaled)
    results["Logistic Regression"] = {
        "auc": roc_auc_score(y_test, lr_prob),
        "f1": f1_score(y_test, lr_pred, average="macro"),
    }

    print(f"{'Model':<26} {'AUC':>8} {'macro-F1':>10}")
    print("-" * 48)
    for name, r in results.items():
        print(f"{name:<26} {r['auc']:>8.4f} {r['f1']:>10.4f}")

# Optional multiclass.
if RUN_MULTICLASS:
    box("13. MULTICLASS TIER PREDICTION")

    def make_mc_label(tier):
        if tier in {"Ghost", "Scale_Waster", "IO_Bottlenecked", "GPU_Idle_Timeout"}:
            return tier
        return "Productive"

    telem_df["mc_label"] = telem_df["crosslayer_tier"].apply(make_mc_label)
    mc_le = LabelEncoder()
    mc_le.fit(telem_df["mc_label"].iloc[:split_t])

    y_mc_train = mc_le.transform(telem_df["mc_label"].iloc[:split_t])
    test_labels = telem_df["mc_label"].iloc[split_t:].apply(lambda x: x if x in set(mc_le.classes_) else "Productive")
    y_mc_test = mc_le.transform(test_labels)

    mc_results = {}

    for name, feats in [("RF M1", M1["avail"]), ("RF M3", M3["avail"])]:
        clf = RandomForestClassifier(n_estimators=200, class_weight="balanced", n_jobs=-1, random_state=RNG)
        clf.fit(to_X(telem_df.iloc[:split_t], feats), y_mc_train)
        pred = clf.predict(to_X(telem_df.iloc[split_t:], feats))
        mc_results[name] = f1_score(y_mc_test, pred, average="macro", zero_division=0)
        print(f"\n[{name}]")
        print(classification_report(y_mc_test, pred, target_names=mc_le.classes_, digits=3, zero_division=0))

    if RUN_BOOSTING:
        try:
            import lightgbm as lgb
            lgb_mc = lgb.LGBMClassifier(
                n_estimators=300, learning_rate=0.05, max_depth=6, num_leaves=63,
                subsample=0.8, colsample_bytree=0.8, class_weight="balanced",
                random_state=RNG, n_jobs=-1, verbose=-1,
            )
            lgb_mc.fit(to_X(telem_df.iloc[:split_t], M3["avail"]), y_mc_train)
            pred = lgb_mc.predict(to_X(telem_df.iloc[split_t:], M3["avail"]))
            mc_results["LightGBM M3"] = f1_score(y_mc_test, pred, average="macro", zero_division=0)
            print("\n[LightGBM M3]")
            print(classification_report(y_mc_test, pred, target_names=mc_le.classes_, digits=3, zero_division=0))
        except Exception as e:
            print(f"Skipping LightGBM multiclass: {e}")

    print("\nMulticlass macro-F1 summary")
    for name, val in sorted(mc_results.items(), key=lambda x: -x[1]):
        print(f"  {name:<18} {val:.4f}")

# Optional M4/M5 compact expansion.
if RUN_EXPANDED_HISTORY:
    box("14. M4/M5 EXPANDED HISTORY")

    compact_hist_cols = [
        "util_mean", "idle_frac", "zero_util_frac", "power_efficiency", "io_time_frac", "bytes_per_gpu_hour",
    ]

    expanded_gpu_hist_cols = [
        "util_mean", "util_max", "util_std", "util_p25", "util_p50", "util_p75", "util_p95",
        "zero_util_frac", "idle_frac", "active_phase_frac", "max_consecutive_idle_readings",
        "mem_util_mean", "mem_util_max", "mem_pressure_frac", "mem_bound_frac",
        "power_mean", "power_max", "power_std", "power_p95", "power_efficiency",
        "high_power_low_util_frac", "near_power_cap_frac",
        "temp_mean", "temp_max", "temp_p95", "thermal_throttle_frac",
        "util_phase1", "util_phase2", "util_phase3", "phase_drop",
        "gpu_imbalance_mean", "gpu_imbalance_max", "node_util_imbalance_std", "node_util_imbalance_max",
        "telemetry_coverage_frac", "telemetry_gap_detected",
    ]

    expanded_io_hist_cols = [
        "io_time_frac", "bytes_per_gpu_hour", "BWio_MB", "total_bytes", "bytes_read", "bytes_written",
        "io_density", "posix_reads", "posix_writes", "posix_opens", "posix_stats",
        "mpiio_bytes_read", "mpiio_bytes_written", "mpiio_coll_reads", "mpiio_coll_writes",
        "mpiio_indep_reads", "mpiio_indep_writes", "mpiio_coll_ratio",
        "stdio_bytes_read", "stdio_bytes_written",
        "seq_read_ratio", "seq_write_ratio", "small_read_ratio", "large_read_ratio", "write_dominance",
        "metadata_ops_per_gb", "unique_files", "has_posix", "has_mpiio", "has_heatmap", "cb_nodes",
        "rank_imbalance", "rank_time_imbalance", "rank_time_gap", "variance_rank_time",
        "slowest_rank_time", "fastest_rank_time",
        "io_phase_start_frac", "io_phase_end_frac", "io_read_front_heavy", "io_write_back_heavy",
        "mem_not_aligned_ratio", "file_not_aligned_ratio",
    ]

    available_compact = [c for c in compact_hist_cols if c in train_df.columns]
    available_gpu = [c for c in expanded_gpu_hist_cols if c in train_df.columns]
    available_io = [c for c in expanded_io_hist_cols if c in train_df.columns]

    m3_hist_cols = unique_keep_order(available_compact)
    m4_hist_cols = unique_keep_order(available_compact + available_gpu)
    m5_hist_cols = unique_keep_order(m4_hist_cols + available_io)
    all_hist_cols = unique_keep_order(m5_hist_cols)

    print(f"Available compact cols: {len(available_compact)}")
    print(f"Available GPU cols:     {len(available_gpu)}")
    print(f"Available I/O cols:     {len(available_io)}")

    telem_m = train_df[train_df["crosslayer_tier"].isin(TELEM_TIERS)].copy().reset_index(drop=True)
    split_m = int(len(telem_m) * 0.80)

    queue_freq_m = telem_m["QUEUE_NAME"].iloc[:split_m].value_counts()
    telem_m["queue_freq"] = telem_m["QUEUE_NAME"].map(queue_freq_m).fillna(0)
    
    exe_freq_m = telem_m["exec_key"].iloc[:split_m].dropna().value_counts()
    telem_m["executable_freq"] = telem_m["exec_key"].map(exe_freq_m).fillna(0)
    
    le_m = LabelEncoder()
    le_m.fit(telem_m["SCIENCE_FIELD"].iloc[:split_m].astype(str))
    known_m = set(le_m.classes_)
    telem_m["SCIENCE_FIELD_enc"] = telem_m["SCIENCE_FIELD"].astype(str).apply(
        lambda x: le_m.transform([x])[0] if x in known_m else -1
    )

    print("Building expanded user history once...")
    hist_m = build_hist(train_df, telem_m["JOB_NAME"].tolist(), all_hist_cols, LOOKBACK_DAYS, N_HIST)
    telem_m = telem_m.merge(hist_m, on="JOB_NAME", how="left")
    
    print("Building expanded executable history once...")
    exec_hist_m = build_entity_hist(
        train_df=train_df,
        target_df=telem_m[["JOB_NAME", "exec_key"]],
        key_cols=["exec_key"],
        cols=all_hist_cols,
        prefix="exec",
        lookback_days=LOOKBACK_DAYS,
        n_history=N_HIST,
    )
    
    user_exec_hist_m = build_entity_hist(
        train_df=train_df,
        target_df=telem_m[["JOB_NAME", "USERNAME_GENID", "exec_key"]],
        key_cols=["USERNAME_GENID", "exec_key"],
        cols=all_hist_cols,
        prefix="user_exec",
        lookback_days=LOOKBACK_DAYS,
        n_history=N_HIST,
    )
    
    telem_m = telem_m.merge(exec_hist_m, on="JOB_NAME", how="left")
    telem_m = telem_m.merge(user_exec_hist_m, on="JOB_NAME", how="left")
    
    groupA_m = groupA1
    groupB_m = groupB
    
    exec_summary_m = [
        "exec_job_count", "exec_waste_rate", "exec_mean_runtime",
        "exec_mean_nodes", "exec_mean_walltime",
        "user_exec_job_count", "user_exec_waste_rate", "user_exec_mean_runtime",
        "user_exec_mean_nodes", "user_exec_mean_walltime",
    ]
    
    m3_user_hist = [f"hist_{c}" for c in m3_hist_cols if f"hist_{c}" in telem_m.columns]
    m4_user_hist = [f"hist_{c}" for c in m4_hist_cols if f"hist_{c}" in telem_m.columns]
    m5_user_hist = [f"hist_{c}" for c in m5_hist_cols if f"hist_{c}" in telem_m.columns]
    
    m3_exec_hist = (
        [f"exec_hist_{c}" for c in m3_hist_cols if f"exec_hist_{c}" in telem_m.columns] +
        [f"user_exec_hist_{c}" for c in m3_hist_cols if f"user_exec_hist_{c}" in telem_m.columns]
    )
    
    m4_exec_hist = (
        [f"exec_hist_{c}" for c in m4_hist_cols if f"exec_hist_{c}" in telem_m.columns] +
        [f"user_exec_hist_{c}" for c in m4_hist_cols if f"user_exec_hist_{c}" in telem_m.columns]
    )
    
    m5_exec_hist = (
        [f"exec_hist_{c}" for c in m5_hist_cols if f"exec_hist_{c}" in telem_m.columns] +
        [f"user_exec_hist_{c}" for c in m5_hist_cols if f"user_exec_hist_{c}" in telem_m.columns]
    )
    
    m3_features = groupA_m + groupB_m + m3_user_hist + exec_summary_m + m3_exec_hist
    m4_features = groupA_m + groupB_m + m4_user_hist + exec_summary_m + m4_exec_hist
    m5_features = groupA_m + groupB_m + m5_user_hist + exec_summary_m + m5_exec_hist

    y_train_m = telem_m["is_wasteful"].iloc[:split_m].values
    y_test_m = telem_m["is_wasteful"].iloc[split_m:].values

    def fit_eval_m(feats, name):
        avail = [f for f in feats if f in telem_m.columns]
        clf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=RNG)
        clf.fit(to_X(telem_m.iloc[:split_m], avail), y_train_m)
        prob = clf.predict_proba(to_X(telem_m.iloc[split_m:], avail))[:, 1]
        pred = clf.predict(to_X(telem_m.iloc[split_m:], avail))
        return {
            "name": name,
            "features": avail,
            "auc": roc_auc_score(y_test_m, prob),
            "f1": f1_score(y_test_m, pred, average="macro"),
        }

    M1_m = fit_eval_m(groupA_m, "M1 scheduler + executable context")
    M3_m = fit_eval_m(m3_features, "M3 compact history")
    M4_m = fit_eval_m(m4_features, "M4 expanded GPU history")
    M5_m = fit_eval_m(m5_features, "M5 expanded GPU/I/O history")

    print(f"{'Model':<32} {'#feat':>7} {'AUC':>8} {'macro-F1':>10}")
    print("-" * 65)
    for r in [M1_m, M3_m, M4_m, M5_m]:
        print(f"{r['name']:<32} {len(r['features']):>7} {r['auc']:>8.4f} {r['f1']:>10.4f}")

    # Temporal CV M3/M4/M5.
    X_m3_all = to_X(telem_m, M3_m["features"])
    X_m4_all = to_X(telem_m, M4_m["features"])
    X_m5_all = to_X(telem_m, M5_m["features"])
    y_all_m = telem_m["is_wasteful"].values

    m3_cv2, m4_cv2, m5_cv2 = [], [], []
    for tr, te in TimeSeriesSplit(n_splits=5).split(X_m5_all):
        rf3 = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=RNG).fit(X_m3_all[tr], y_all_m[tr])
        rf4 = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=RNG).fit(X_m4_all[tr], y_all_m[tr])
        rf5 = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=RNG).fit(X_m5_all[tr], y_all_m[tr])
        m3_cv2.append(roc_auc_score(y_all_m[te], rf3.predict_proba(X_m3_all[te])[:, 1]))
        m4_cv2.append(roc_auc_score(y_all_m[te], rf4.predict_proba(X_m4_all[te])[:, 1]))
        m5_cv2.append(roc_auc_score(y_all_m[te], rf5.predict_proba(X_m5_all[te])[:, 1]))

    print("\n5-fold temporal CV: M3 vs M4 vs M5")
    print(f"  M3 compact:         {np.mean(m3_cv2):.4f} ± {np.std(m3_cv2):.4f}")
    print(f"  M4 expanded GPU:    {np.mean(m4_cv2):.4f} ± {np.std(m4_cv2):.4f}")
    print(f"  M5 expanded GPU/IO: {np.mean(m5_cv2):.4f} ± {np.std(m5_cv2):.4f}")



# =============================================================================
# OVR PER-TIER AUC FOR EXPANDED MODELS: M1 vs M3 vs M4 vs M5
# Run after M1_m, M3_m, M4_m, M5_m are trained.
# =============================================================================

box("OVR PER-TIER AUC: M1 vs M3 vs M4 vs M5")

tier_name = {
    "Ghost": "Idle",
    "IO_Bottlenecked": "IO_Bottlenecked",
    "Scale_Waster": "Scale_Inefficient",
    "GPU_Idle_Timeout": "GPU_Idle_Timeout",
}

tiers = ["Ghost", "IO_Bottlenecked", "Scale_Waster", "GPU_Idle_Timeout"]

feature_sets_ovr_exp = {
    "M1": M1_m["features"],
    "M3": M3_m["features"],
    "M4": M4_m["features"],
    "M5": M5_m["features"],
}

ovr_exp_rows = []

print(
    f"{'Tier':<22} {'support':>8} "
    f"{'M1':>8} {'M3':>8} {'M4':>8} {'M5':>8} "
    f"{'Δ M5-M1':>10}"
)
print("-" * 86)

for tier in tiers:
    y_tr = (telem_m["crosslayer_tier"].iloc[:split_m] == tier).astype(int).values
    y_te = (telem_m["crosslayer_tier"].iloc[split_m:] == tier).astype(int).values
    support = int(y_te.sum())

    if support < 10:
        print(f"{tier_name[tier]:<22} {support:>8,} insufficient positives")
        continue

    aucs = {}

    for model_name, feats in feature_sets_ovr_exp.items():
        rf = RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            n_jobs=-1,
            random_state=RNG,
        )
        rf.fit(to_X(telem_m.iloc[:split_m], feats), y_tr)

        prob = rf.predict_proba(
            to_X(telem_m.iloc[split_m:], feats)
        )[:, 1]

        aucs[model_name] = roc_auc_score(y_te, prob)

    delta = aucs["M5"] - aucs["M1"]

    print(
        f"{tier_name[tier]:<22} {support:>8,} "
        f"{aucs['M1']:>8.4f} {aucs['M3']:>8.4f} "
        f"{aucs['M4']:>8.4f} {aucs['M5']:>8.4f} "
        f"{delta:>10.4f}"
    )

    ovr_exp_rows.append({
        "internal_tier": tier,
        "paper_tier": tier_name[tier],
        "support": support,
        "M1": aucs["M1"],
        "M3": aucs["M3"],
        "M4": aucs["M4"],
        "M5": aucs["M5"],
        "delta": delta,
    })

ovr_exp_df = pd.DataFrame(ovr_exp_rows)

print("\nLaTeX rows:")
for _, r in ovr_exp_df.iterrows():
    tier_latex = r["paper_tier"].replace("_", r"\_")
    print(
        rf"\texttt{{{tier_latex}}} & {int(r['support']):,} "
        rf"& {r['M1']:.3f} & {r['M3']:.3f} & {r['M4']:.3f} "
        rf"& {r['M5']:.3f} & ${r['delta']:+.3f}$ \\"
    )
# =============================================================================
# 15. SANITY CHECKS + FINAL PASTE-READY SUMMARY
# =============================================================================

box("15. SANITY CHECKS + FINAL SUMMARY")

issues = []

if not telem_df["QUEUED_TIMESTAMP"].is_monotonic_increasing:
    issues.append("Temporal ordering failed")
print(f"Temporal ordering: {'OK' if not issues else 'CHECK'}")

# The only runtime telemetry allowed in M3 is historical hist_* features.
current_telemetry_cols = {
    "util_mean", "idle_frac", "zero_util_frac", "power_efficiency",
    "io_time_frac", "bytes_per_gpu_hour", "gpu_hours", "BWio_MB",
    "power_mean", "temp_mean",
}
leaked = [f for f in M3["avail"] if f in current_telemetry_cols]
print(f"No current-job performance telemetry leakage: {'OK' if not leaked else 'CHECK ' + str(leaked)}")
print("Executable frequency is included as workload-context, not as performance telemetry.")
if leaked:
    issues.append(f"Leakage: {leaked}")

dup_jobs = combined["JOB_NAME"].duplicated().sum()
print(f"No duplicate JOB_NAME: {'OK' if dup_jobs == 0 else 'CHECK ' + str(dup_jobs)}")
if dup_jobs:
    issues.append(f"Duplicate jobs: {dup_jobs}")

overlap = set(telem_df["JOB_NAME"].iloc[:split_t]) & set(telem_df["JOB_NAME"].iloc[split_t:])
print(f"No train/test overlap: {'OK' if not overlap else 'CHECK ' + str(len(overlap))}")
if overlap:
    issues.append(f"Train/test overlap: {len(overlap)}")

sha = hashlib.sha256(telem_df[M3["avail"]].fillna(-999).values.tobytes()).hexdigest()[:16]
print(f"Feature matrix SHA256[:16]: {sha}")

print("\nPASTE-READY SUMMARY")
print(f"""
CORPUS
  total jobs:                  {total_jobs:,}
  allocated GPU-hours:         {total_gpu:,.0f} ({total_gpu/1e6:.2f}M)
  unique users/projects:       {n_users:,} / {n_projects:,}

COVERAGE
  DCGM:                        {n_dcgm:,} ({pct(n_dcgm, total_jobs):.1f}%)
  Darshan attached:            {n_dar:,} ({pct(n_dar, total_jobs):.1f}%)
  joint DCGM+Darshan:          {n_both:,} ({pct(n_both, total_jobs):.1f}%)

TAXONOMY
  under-utilized GPU-hours:    {under_gpu:,.0f} ({pct(under_gpu, total_gpu):.1f}%)

USER CONCENTRATION
  top 10 user share:           {top10_share*100:.1f}% of under-utilized GPU-hours
  top 5% user share:           {top5pct_share*100:.1f}% of under-utilized GPU-hours

ML
  train/test jobs:             {split_t:,} / {len(telem_df)-split_t:,}
  M1 AUC/F1:                   {M1['auc']:.4f} / {M1['f1']:.4f}
  M2 AUC/F1:                   {M2['auc']:.4f} / {M2['f1']:.4f}
  M3 AUC/F1:                   {M3['auc']:.4f} / {M3['f1']:.4f}
  M3 bootstrap CI:             [{ci_lo:.4f}, {ci_hi:.4f}]
  M3-M1 lift:                  {M3['auc']-M1['auc']:+.4f}
  M3 multi-seed AUC/F1:        {np.mean(m3_aucs):.4f} ± {np.std(m3_aucs):.4f} / {np.mean(m3_f1s):.4f} ± {np.std(m3_f1s):.4f}
  M3-M0 lift: {M3['auc']-M0['auc']:+.4f} 
  M3-M1 lift: {M3['auc']-M1['auc']:+.4f}
""")

print(f"Total runtime: {(time.time() - t0)/60:.1f} min")
print("ALL SANITY CHECKS PASSED" if not issues else f"ISSUES: {issues}")



# =============================================================================
# 16. M5 BOOTSTRAP CI + FEATURE-FAMILY IMPORTANCE
# =============================================================================
box("16. M5 BOOTSTRAP CI + FEATURE-FAMILY IMPORTANCE")

# -----------------------------------------------------------------------------
# 16a. Refit M3/M4/M5 keeping the fitted estimator + held-out probabilities,
#     so we can bootstrap and read feature importances without re-running §14.
# -----------------------------------------------------------------------------
def fit_eval_m_with_clf(feats, name):
    avail = [f for f in feats if f in telem_m.columns]
    X_tr = to_X(telem_m.iloc[:split_m], avail)
    X_te = to_X(telem_m.iloc[split_m:], avail)
    clf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=RNG)
    clf.fit(X_tr, y_train_m)
    prob = clf.predict_proba(X_te)[:, 1]
    pred = clf.predict(X_te)
    return {
        "name": name,
        "features": avail,
        "clf": clf,
        "prob": prob,
        "pred": pred,
        "auc": roc_auc_score(y_test_m, prob),
        "f1": f1_score(y_test_m, pred, average="macro"),
    }

print("Refitting M3/M4/M5 to retain estimator + probabilities...")
M3_m_full = fit_eval_m_with_clf(m3_features, "M3 compact + exec history")
M4_m_full = fit_eval_m_with_clf(m4_features, "M4 expanded GPU history")
M5_m_full = fit_eval_m_with_clf(m5_features, "M5 expanded GPU/I/O history")

print(f"  M3 AUC = {M3_m_full['auc']:.4f}, M4 AUC = {M4_m_full['auc']:.4f}, "
      f"M5 AUC = {M5_m_full['auc']:.4f}")

# -----------------------------------------------------------------------------
# 16b. Bootstrap 95% CI for M5 (and M4 for completeness)
# -----------------------------------------------------------------------------
def bootstrap_ci(prob, y, n_boot=1000, seed=RNG):
    rng = np.random.RandomState(seed)
    boots = []
    for _ in range(n_boot):
        idx = rng.randint(0, len(y), len(y))
        boots.append(roc_auc_score(y[idx], prob[idx]))
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))

m4_lo, m4_hi = bootstrap_ci(M4_m_full["prob"], y_test_m)
m5_lo, m5_hi = bootstrap_ci(M5_m_full["prob"], y_test_m)

print(f"\nBootstrap 95% CI (1,000 resamples on held-out test):")
print(f"  M4 expanded GPU history    AUC = {M4_m_full['auc']:.4f}  CI = [{m4_lo:.4f}, {m4_hi:.4f}]")
print(f"  M5 expanded GPU/IO history AUC = {M5_m_full['auc']:.4f}  CI = [{m5_lo:.4f}, {m5_hi:.4f}]")

# -----------------------------------------------------------------------------
# 16c. Feature-family importance breakdown for M5
# -----------------------------------------------------------------------------
# Resolve membership using the variables already built in §14:
#   groupA_m / groupA1     -> scheduler + executable freq
#   groupB_m / groupB      -> user-level scheduler history
#   exec_summary_m         -> exec-level scalar summaries
#   exec_hist_*  /  user_exec_hist_*  -> historical telemetry per executable
#   hist_*                 -> historical telemetry per user
#   available_gpu / available_io -> raw column names that determine GPU vs I/O
#                                  for the user-history hist_* features
gpu_set = set(available_gpu)
io_set  = set(available_io)

def feat_family(f):
    # Scheduler-visible at submission/launch
    if f in groupA_m:
        return "scheduler+exec_context"
    if f in groupB_m:
        return "user_scheduler_history"
    if f in exec_summary_m:
        return "exec_history_summary"

    # User+executable historical telemetry
    if f.startswith("user_exec_hist_"):
        raw = f[len("user_exec_hist_"):]
        if raw in io_set:  return "user_exec_io_history"
        if raw in gpu_set: return "user_exec_gpu_power_history"
        return "user_exec_other_history"

    # Same-executable historical telemetry
    if f.startswith("exec_hist_"):
        raw = f[len("exec_hist_"):]
        if raw in io_set:  return "exec_io_history"
        if raw in gpu_set: return "exec_gpu_power_history"
        return "exec_other_history"

    # User-level historical telemetry (non-exec)
    if f.startswith("hist_"):
        raw = f[len("hist_"):]
        if raw in io_set:  return "user_io_history"
        if raw in gpu_set: return "user_gpu_power_history"
        return "user_other_history"

    return "uncategorized"

fi = pd.Series(M5_m_full["clf"].feature_importances_,
               index=M5_m_full["features"])
fi_family = (fi.groupby(fi.index.map(feat_family))
               .sum()
               .sort_values(ascending=False))
fi_total = fi.sum()
fi_family_pct = (fi_family / fi_total * 100).round(2)

print(f"\nM5 feature-family importance share (% of total importance mass):")
print(f"  {'Family':<32} {'Sum':>10} {'Share %':>10} {'#Feat':>8}")
print("  " + "-" * 64)
for fam, val in fi_family.items():
    n_feat = int(fi.index.map(feat_family).to_series().eq(fam).sum())
    print(f"  {fam:<32} {val:>10.4f} {fi_family_pct[fam]:>10.2f} {n_feat:>8d}")

# Aggregate to coarser buckets that map directly to a paper sentence
def coarse_family(fam):
    if fam.endswith("_io_history"):           return "I/O history (any source)"
    if fam.endswith("_gpu_power_history"):    return "GPU/power history (any source)"
    if fam.endswith("_other_history"):        return "Other history (any source)"
    if fam == "exec_history_summary":         return "Executable scalar summaries"
    if fam == "user_scheduler_history":       return "User scheduler history"
    if fam == "scheduler+exec_context":       return "Scheduler + exec context (current)"
    return fam

coarse = (fi_family
          .groupby(lambda k: coarse_family(k))
          .sum()
          .sort_values(ascending=False))
coarse_pct = (coarse / coarse.sum() * 100).round(2)

print(f"\nM5 importance grouped for paper:")
print(f"  {'Group':<40} {'Share %':>10}")
print("  " + "-" * 52)
for fam, share in coarse_pct.items():
    print(f"  {fam:<40} {share:>10.2f}")

# -----------------------------------------------------------------------------
# 16d. Top-15 individual M5 features (sanity check that nothing weird leads)
# -----------------------------------------------------------------------------
print(f"\nM5 top 15 individual features:")
for f, v in fi.sort_values(ascending=False).head(15).items():
    print(f"  {feat_family(f):<32} {f:<48} {v:.4f}")

# -----------------------------------------------------------------------------
# 16e. Paste-ready paper numbers
# -----------------------------------------------------------------------------
print("\n" + "-" * 78)
print("PASTE-READY (Section VIII + Conclusion):")
print("-" * 78)
print(f"  M5 held-out AUC: {M5_m_full['auc']:.3f} (95% bootstrap CI [{m5_lo:.3f}, {m5_hi:.3f}])")
print(f"  M5 macro-F1:     {M5_m_full['f1']:.3f}")
print(f"  M4 held-out AUC: {M4_m_full['auc']:.3f} (95% bootstrap CI [{m4_lo:.3f}, {m4_hi:.3f}])")
print(f"  Importance attributable to GPU/power historical features: "
      f"{coarse_pct.get('GPU/power history (any source)', 0):.1f}%")
print(f"  Importance attributable to I/O historical features:       "
      f"{coarse_pct.get('I/O history (any source)', 0):.1f}%")
print(f"  Importance attributable to user scheduler history:        "
      f"{coarse_pct.get('User scheduler history', 0):.1f}%")
print(f"  Importance attributable to current scheduler/exec context:"
      f" {coarse_pct.get('Scheduler + exec context (current)', 0):.1f}%")

print(len(M5_m["features"]))
print(len(set(M5_m["features"])))