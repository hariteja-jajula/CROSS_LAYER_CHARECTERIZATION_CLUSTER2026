"""
framework.py — Cross-layer HPC paper results.
Single run produces all numbers needed for paper body + sanity checks.
"""
print("BOOT: script entered", flush=True)

import os, sys, json, time, hashlib
print("BOOT: base imports done", flush=True)

import pandas as pd, numpy as np
print("BOOT: pandas/numpy imported", flush=True)

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, f1_score
print("BOOT: sklearn imported", flush=True)

from scipy.stats import spearmanr
print("BOOT: scipy imported", flush=True)

import importlib, utils.combined as cu
print("BOOT: utils.combined imported", flush=True)

importlib.reload(cu)
print("BOOT: utils.combined reloaded", flush=True)

from utils.combined import classify_crosslayer
print("BOOT: classify_crosslayer imported", flush=True)
import os, sys, json, time, hashlib
import pandas as pd, numpy as np
import warnings; warnings.filterwarnings("ignore")
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, f1_score
from scipy.stats import spearmanr
import importlib, utils.combined as cu; importlib.reload(cu)
from utils.combined import classify_crosslayer

t0 = time.time()
RNG = 42
N_HIST, LOOKBACK_DAYS = 10, 7
print("Started")
# ════════════════════════════════════════════════════════════════
# 1. LOAD + MERGE
# ════════════════════════════════════════════════════════════════
cfg    = json.load(open("/pscratch/sd/h/hjajula/Darshan/cross_layer_hpc_tool/config/config.json"))
job_df = pd.read_csv(cfg["djc_csv"], low_memory=False)
gm     = pd.read_csv("/pscratch/sd/h/hjajula/Darshan/cross_layer_hpc_tool/data/gpu_metrics.csv", low_memory=False)
dm     = pd.read_csv("/pscratch/sd/h/hjajula/Darshan/cross_layer_hpc_tool/data/darshan_metrics_final.csv", low_memory=False)

dm["job_id"]     = dm["job_id"].astype(str)
job_df["job_id"] = job_df["JOB_NAME"].str.split(".").str[0]
print("data loaded")
# agg_spec = {k:v for k,v in {
#     "bytes_read":"sum","bytes_written":"sum","posix_reads":"sum","posix_writes":"sum",
#     "posix_opens":"sum","posix_stats":"sum","mpiio_bytes_read":"sum",
#     "mpiio_bytes_written":"sum","mpiio_coll_reads":"sum","mpiio_coll_writes":"sum",
#     "mpiio_indep_reads":"sum","mpiio_indep_writes":"sum","stdio_bytes_read":"sum",
#     "stdio_bytes_written":"sum","runtime":"max","nprocs":"max",
#     "slowest_rank_time":"max","fastest_rank_time":"max","variance_rank_time":"max",
#     "io_time_frac":"max","io_density":"max","seq_read_ratio":"mean",
#     "seq_write_ratio":"mean","small_read_ratio":"mean","large_read_ratio":"mean",
#     "rank_imbalance":"max","rank_time_imbalance":"max","rank_time_gap":"max",
#     "write_dominance":"mean","mpiio_coll_ratio":"mean","io_phase_start_frac":"min",
#     "io_phase_end_frac":"max","io_read_front_heavy":"max","io_write_back_heavy":"max",
#     "has_posix":"max","has_mpiio":"max","has_heatmap":"max","cb_nodes":"max",
#     "unique_files":"sum","metadata_ops_per_gb":"mean","mem_not_aligned_ratio":"mean",
#     "file_not_aligned_ratio":"mean",
# }.items() if k in dm.columns}

# dm_agg   = dm.groupby("job_id").agg(agg_spec).reset_index()

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

    # Combined heatmap-derived I/O timing
    "io_time_frac": "max",
    "io_density": "max",
    "io_active_bins": "sum",
    "io_phase_start_frac": "min",
    "io_phase_end_frac": "max",

    # Read/write split heatmap-derived I/O timing
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

    # R/W overlap and burstiness
    "io_rw_overlap_frac": "max",
    "io_max_gap_bins": "max",
    "io_mean_gap_bins": "mean",
    "io_n_io_bursts": "max",

    # I/O structure and access pattern features
    "seq_read_ratio": "mean",
    "seq_write_ratio": "mean",
    "small_read_ratio": "mean",
    "large_read_ratio": "mean",
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

    # Metadata/configuration features
    "cb_nodes": "max",
    "unique_files": "sum",
    "metadata_ops_per_gb": "mean",
    "mem_not_aligned_ratio": "mean",
    "file_not_aligned_ratio": "mean",
}.items() if k in dm.columns}

dm_agg = dm.groupby("job_id").agg(agg_spec).reset_index()

dm_agg = dm.groupby("job_id").agg(agg_spec).reset_index()

combined = job_df.merge(gm, on="JOB_NAME", how="left").merge(dm_agg, on="job_id", how="left")

old_exe  = pd.read_csv("/pscratch/sd/h/hjajula/Darshan/cross_layer_hpc_tool/data/darshan_metrics_old.csv", usecols=["fname","executable"], low_memory=False)
old_exe["job_id"] = old_exe["fname"].str.split("-").str[0]
combined = combined.merge(old_exe.groupby("job_id")["executable"].first().reset_index(), on="job_id", how="left")
combined["gpu_util_mean"]       = combined["util_mean"]
combined["io_read_front_heavy"] = combined["io_read_front_heavy"].fillna(0).astype(bool)
combined["io_write_back_heavy"] = combined["io_write_back_heavy"].fillna(0).astype(bool)

# ════════════════════════════════════════════════════════════════
# 2. TAXONOMY
# ════════════════════════════════════════════════════════════════
sys.stdout = open(os.devnull, 'w')
combined = classify_crosslayer(combined)
sys.stdout = sys.__stdout__

# Purify IO_Bottlenecked: io_time_frac <= 5% = incidental, not bottlenecked
m_io  = combined["crosslayer_tier"] == "IO_Bottlenecked"
m_inc = combined["io_time_frac"].fillna(0) <= 0.05
combined.loc[m_io & m_inc, ["crosslayer_tier","diagnostic_tier"]] = "Incidental_IO_Low_GPU"

# Split Balanced into Ideal (gpu>=70%) and Moderate
m_bal   = combined["crosslayer_tier"] == "Balanced"
m_ideal = combined["gpu_util_mean"].fillna(0) >= 70.0
combined.loc[m_bal &  m_ideal, ["crosslayer_tier","diagnostic_tier"]] = "Ideal_Compute_With_IO"
combined.loc[m_bal & ~m_ideal, ["crosslayer_tier","diagnostic_tier"]] = "Moderate_Compute_With_IO"

# is_wasteful set AFTER all relabeling — Incidental_IO_Low_GPU is NOT wasteful
WASTEFUL = {"Ghost","Scale_Waster","IO_Bottlenecked","Failed_Job","Quick_Cancel","GPU_Idle_Timeout"}
combined["is_wasteful"] = combined["crosslayer_tier"].isin(WASTEFUL).astype(int)
combined.to_csv("/pscratch/sd/h/hjajula/Darshan/cross_layer_hpc_tool/data/combined_metrics_final.csv", index=False)

total_gpu = combined["gpu_hours"].sum()
total_jobs = len(combined)

# ════════════════════════════════════════════════════════════════
# 3. RESULTS — COVERAGE & TAXONOMY
# ════════════════════════════════════════════════════════════════
print("=" * 60)
print("COVERAGE")
print("=" * 60)
gpu_cov     = combined["has_gpu"].sum()
dar_present = combined["darshan_present"].sum()
io_detected = combined["io_detected"].sum()
print(f"Total jobs            : {total_jobs:,}")
print(f"GPU telemetry (DCGM)  : {gpu_cov:,} ({gpu_cov/total_jobs*100:.1f}%)")
print(f"Darshan attached      : {dar_present:,} ({dar_present/total_jobs*100:.1f}%)")
print(f"Darshan I/O detected  : {io_detected:,} ({io_detected/total_jobs*100:.1f}%)")
print(f"Both GPU + Darshan    : {(combined['has_gpu']&combined['darshan_present']).sum():,}")
print(f"Total allocated GPU-hrs: {total_gpu:,.0f}")

print("\n" + "=" * 60)
print("TAXONOMY TABLE", flush=True)
print("=" * 60)
tier_order = ["Quick_Cancel","Failed_Job","GPU_Idle_Timeout","Ghost","Scale_Waster",
              "IO_Bottlenecked","Incidental_IO_Low_GPU",
              "Moderate_Compute_No_IO","Moderate_Compute_With_IO",
              "Compute_Bound","Ideal_Compute_With_IO","Low_Efficiency"]
print(f"{'Tier':<28} {'Jobs':>8} {'GPU-hrs':>12} {'% GPU-hrs':>10}")
print("-" * 62)
for tier in tier_order:
    s = combined[combined["crosslayer_tier"] == tier]
    if len(s) == 0: continue
    gh = s["gpu_hours"].sum()
    print(f"{tier:<28} {len(s):>8,} {gh:>12,.0f} {gh/total_gpu*100:>9.1f}%")

# Wasteful summary
wasteful_gpu = combined.loc[combined["is_wasteful"]==1,"gpu_hours"].sum()
failed_git   = combined.loc[combined["crosslayer_tier"].isin({"Failed_Job","GPU_Idle_Timeout"}),"gpu_hours"].sum()
print(f"\nTotal wasteful GPU-hrs : {wasteful_gpu:,.0f} ({wasteful_gpu/total_gpu*100:.1f}%)")
print(f"Failed+GIT GPU-hrs     : {failed_git:,.0f} ({failed_git/total_gpu*100:.1f}%)")

print("\n[IO_Bottlenecked sub-tiers]")
io_b = combined[combined["crosslayer_tier"] == "IO_Bottlenecked"]
if len(io_b) > 0:
    m_meta  = io_b["metadata_ops_per_gb"].fillna(0) > 1000
    m_bw    = io_b["BWio_MB"].fillna(10000) < 1000
    m_strag = io_b["rank_time_imbalance"].fillna(0) > 2.0
    n_meta  = m_meta.sum()
    n_bw    = (m_bw & ~m_meta).sum()
    n_strag = (m_strag & ~m_meta & ~m_bw).sum()
    n_other = len(io_b) - n_meta - n_bw - n_strag
    for label, n in [("Metadata-bound (>1000 ops/GB)",n_meta),
                     ("Bandwidth-bound (<1000 MB/s)", n_bw),
                     ("Rank-imbalance (>2x)",         n_strag),
                     ("Other",                        n_other)]:
        print(f"  {label:<35}: {n:>5,} ({n/len(io_b)*100:.1f}%)")

print("\n[Scale_Waster by allocation size]")
sw = combined[combined["crosslayer_tier"] == "Scale_Waster"]
if len(sw) > 0:
    for label, mask in [
        ("Medium  (2-7 nodes,   8-31 GPUs)",  (sw["gpus"]>=8)  & (sw["gpus"]<32)),
        ("Large   (8-31 nodes, 32-127 GPUs)", (sw["gpus"]>=32) & (sw["gpus"]<128)),
        ("Extreme (32+ nodes,  128+ GPUs)",    sw["gpus"]>=128),
    ]:
        n = mask.sum()
        print(f"  {label}: {n:>4,} jobs | {sw.loc[mask,'gpu_hours'].sum():>8,.0f} GPU-hrs")
# ════════════════════════════════════════════════════════════════
# 3b. CROSS-SECTIONAL ANALYSIS
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("CROSS-SECTIONAL ANALYSIS")
print("=" * 60)

# Convert timestamps here since feature engineering hasn't run yet
combined["QUEUED_TIMESTAMP"] = pd.to_datetime(combined["QUEUED_TIMESTAMP"], errors="coerce")
combined["END_TIMESTAMP"]    = pd.to_datetime(combined["END_TIMESTAMP"], errors="coerce")

# --- By queue ---
print("\n[Tier distribution by queue (top 5 queues)]")
top_queues = combined["QUEUE_NAME"].value_counts().head(5).index
queue_tier = combined[combined["QUEUE_NAME"].isin(top_queues)].groupby(
    ["QUEUE_NAME","crosslayer_tier"]).size().unstack(fill_value=0)
wasteful_cols = [c for c in WASTEFUL if c in queue_tier.columns]
queue_tier["wasteful_pct"] = (queue_tier[wasteful_cols].sum(axis=1) /
                               queue_tier.sum(axis=1) * 100)
print(f"  {'Queue':<20} {'Total':>8} {'Wasteful%':>10}")
for q in top_queues:
    if q not in queue_tier.index: continue
    total = queue_tier.loc[q].sum() - queue_tier.loc[q,"wasteful_pct"]
    print(f"  {q:<20} {int(combined[combined['QUEUE_NAME']==q].shape[0]):>8,} "
          f"{queue_tier.loc[q,'wasteful_pct']:>9.1f}%")

# --- By science field ---
print("\n[Wasteful job rate by science field (top 8 fields)]")
top_fields = combined["SCIENCE_FIELD"].value_counts().head(8).index
for field in top_fields:
    sub = combined[combined["SCIENCE_FIELD"] == field]
    w_pct = sub["is_wasteful"].mean() * 100
    gpu_wasted = sub.loc[sub["is_wasteful"]==1,"gpu_hours"].sum()
    print(f"  {str(field):<30} {len(sub):>6,} jobs  {w_pct:>5.1f}% wasteful  "
          f"{gpu_wasted:>10,.0f} GPU-hrs wasted")

# --- By allocation size ---
print("\n[Wasteful rate by node count (allocation size)]")
combined["node_bin"] = pd.cut(combined["NODES_USED"],
    bins=[0,1,2,4,8,16,32,64,128,9999],
    labels=["1","2","3-4","5-8","9-16","17-32","33-64","65-128","128+"])
for nb, grp in combined.groupby("node_bin", observed=True):
    if len(grp) < 50: continue
    w_pct    = grp["is_wasteful"].mean() * 100
    gpu_mean = grp.loc[grp["has_gpu"],"util_mean"].mean()
    print(f"  {str(nb):<8} nodes: {len(grp):>7,} jobs  {w_pct:>5.1f}% wasteful  "
          f"mean GPU util {gpu_mean:>5.1f}%")

# --- Temporal trend (monthly) ---
print("\n[Monthly wasteful job rate (temporal trend)]")
combined["month_year"] = combined["QUEUED_TIMESTAMP"].dt.to_period("M")
for period, grp in combined.groupby("month_year", observed=True):
    total    = len(grp)
    wasteful = grp["is_wasteful"].sum()
    gpu_w    = grp.loc[grp["is_wasteful"]==1, "gpu_hours"].sum()
    print(f"  {str(period):<10}  {total:>6,} jobs  "
          f"{wasteful/total*100:>5.1f}% wasteful  {gpu_w:>10,.0f} GPU-hrs wasted")

# --- User concentration ---
print("\n[Waste concentration by user (top 10 users by GPU-hrs wasted)]")
user_waste = combined[combined["is_wasteful"]==1].groupby("USERNAME_GENID").agg(
    jobs=("JOB_NAME","count"),
    gpu_hrs=("gpu_hours","sum")
).sort_values("gpu_hrs", ascending=False).head(10)
total_wasteful_gpu = combined.loc[combined["is_wasteful"]==1,"gpu_hours"].sum()
cumulative = 0
for i, (user, row) in enumerate(user_waste.iterrows(), 1):
    cumulative += row["gpu_hrs"]
    print(f"  #{i:<3} {str(user):<20} {int(row['jobs']):>6,} jobs  "
          f"{row['gpu_hrs']:>10,.0f} GPU-hrs  "
          f"cumulative {cumulative/total_wasteful_gpu*100:.1f}%")

# --- Ghost power gap cross-section ---
print("\n[Ghost jobs: power-elevated vs truly-idle by queue]")
ghost_df = combined[combined["crosslayer_tier"] == "Ghost"].copy()
ghost_df["power_elevated"] = ghost_df["power_mean"].fillna(0) > 50
for q, grp in ghost_df.groupby("QUEUE_NAME"):
    if len(grp) < 20: continue
    elev_pct = grp["power_elevated"].mean() * 100
    print(f"  {q:<20} {len(grp):>6,} ghost jobs  {elev_pct:>5.1f}% power-elevated")

# --- IO_Bottlenecked cross-section: user concentration ---
print("\n[IO_Bottlenecked: GPU-hrs by user (top 5)]")
io_users = combined[combined["crosslayer_tier"]=="IO_Bottlenecked"].groupby(
    "USERNAME_GENID")["gpu_hours"].sum().sort_values(ascending=False).head(5)
total_io_gpu = combined.loc[combined["crosslayer_tier"]=="IO_Bottlenecked","gpu_hours"].sum()
for user, gpu_hrs in io_users.items():
    print(f"  {str(user):<20} {gpu_hrs:>10,.0f} GPU-hrs ({gpu_hrs/total_io_gpu*100:.1f}%)")


print("\n[Scale_Waster by allocation size]")
sw = combined[combined["crosslayer_tier"] == "Scale_Waster"]
if len(sw) > 0:
    for label, mask in [
        ("Medium  (2-7 nodes,   8-31 GPUs)",  (sw["gpus"]>=8)  & (sw["gpus"]<32)),
        ("Large   (8-31 nodes, 32-127 GPUs)", (sw["gpus"]>=32) & (sw["gpus"]<128)),
        ("Extreme (32+ nodes,  128+ GPUs)",    sw["gpus"]>=128),
    ]:
        n = mask.sum()
        print(f"  {label}: {n:>4,} jobs | {sw.loc[mask,'gpu_hours'].sum():>8,.0f} GPU-hrs")

# After the existing Scale_Waster by allocation size block in 3b:

print("\n[Scale_Waster: power validation (shared-memory vs true waste)]")
sw = combined[combined["crosslayer_tier"] == "Scale_Waster"].copy()
if len(sw) > 0:
    has_pwr = sw["power_mean"].notna()
    low_pwr  = (sw.loc[has_pwr, "power_mean"] < 50).sum()
    high_pwr = (sw.loc[has_pwr, "power_mean"] >= 50).sum()
    no_pwr   = (~has_pwr).sum()
    print(f"  Total Scale_Waster jobs     : {len(sw):,}")
    print(f"  With power data             : {has_pwr.sum():,}")
    print(f"  power < 50W (true idle)     : {low_pwr:,} ({low_pwr/has_pwr.sum()*100:.1f}%)")
    print(f"  power >= 50W (possible smem): {high_pwr:,} ({high_pwr/has_pwr.sum()*100:.1f}%)")
    print(f"  No power data               : {no_pwr:,}")
    # breakdown by alloc size for power-confirmed waste
    sw["alloc_size"] = pd.cut(sw["gpus"],
        bins=[0,31,127,99999],
        labels=["Medium (8-31 GPUs)","Large (32-127 GPUs)","Extreme (128+ GPUs)"])
    print(f"\n  Power-confirmed waste (power<50W) by allocation size:")
    sw_low = sw[sw["power_mean"].fillna(0) < 50]
    for label, grp in sw_low.groupby("alloc_size", observed=True):
        print(f"    {str(label):<25}: {len(grp):>4,} jobs | {grp['gpu_hours'].sum():>8,.0f} GPU-hrs")

print("\n[Workload concentration: per-tier breakdown for top users]")
wasteful_df = combined[combined["is_wasteful"] == 1]
tier_conc = wasteful_df.groupby(["USERNAME_GENID","crosslayer_tier"])["gpu_hours"].sum().reset_index()
top10_users = (wasteful_df.groupby("USERNAME_GENID")["gpu_hours"]
               .sum().sort_values(ascending=False).head(10).index)
print(f"  Top 10 users by wasteful GPU-hrs — tier breakdown:")
for user in top10_users:
    sub = tier_conc[tier_conc["USERNAME_GENID"] == user].sort_values("gpu_hours", ascending=False)
    tiers_str = ", ".join(f"{r['crosslayer_tier']}={r['gpu_hours']:,.0f}" for _, r in sub.iterrows())
    print(f"  {str(user):<22}: {tiers_str}")

print("\n[GPU-hrs wasted by exact tier — for concentration reporting]")
for tier in ["Ghost","Scale_Waster","IO_Bottlenecked","Failed_Job","Quick_Cancel","GPU_Idle_Timeout"]:
    sub = combined[combined["crosslayer_tier"] == tier]
    if len(sub) == 0: continue
    gpu_w = sub["gpu_hours"].sum()
    print(f"  {tier:<22}: {len(sub):>6,} jobs | {gpu_w:>10,.0f} GPU-hrs ({gpu_w/total_gpu*100:.1f}% of total)")      
# ════════════════════════════════════════════════════════════════
# 4. SINGLE-LAYER BLINDNESS
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("SINGLE-LAYER BLINDNESS")
print("=" * 60)

# Signal anti-correlation
sig_df = combined[combined["util_mean"].notna() & combined["io_time_frac"].notna()][
    ["util_mean","io_time_frac","BWio_MB","rank_time_imbalance"]].dropna()
r_io, _  = spearmanr(sig_df["util_mean"], sig_df["io_time_frac"])
r_bw, _  = spearmanr(sig_df["util_mean"], sig_df["BWio_MB"])
print(f"Spearman r(GPU_util, io_time_frac) = {r_io:.3f}  (N={len(sig_df):,})")
print(f"Spearman r(GPU_util, BWio_MB)      = {r_bw:.3f}")

# Phase table — Ghost vs IO_Bottlenecked indistinguishability
print("\n[GPU phase trajectory by tier]")
phase_df = combined[["util_phase1","util_phase2","util_phase3",
                      "io_phase_end_frac","crosslayer_tier"]].dropna(
    subset=["util_phase1","util_phase2","util_phase3"])
print(f"  {'Tier':<22} {'n':>6} {'P1':>5} {'P2':>5} {'P3':>5} {'io_end':>7}")
for tier in ["Ghost","IO_Bottlenecked","Scale_Waster","Compute_Bound"]:
    sub = phase_df[phase_df["crosslayer_tier"]==tier]
    if len(sub) < 30: continue
    print(f"  {tier:<22} {len(sub):>6,} {sub['util_phase1'].mean():>5.1f} "
          f"{sub['util_phase2'].mean():>5.1f} {sub['util_phase3'].mean():>5.1f} "
          f"{sub['io_phase_end_frac'].median():>7.3f}")
print("  KEY: Ghost and IO_Bottlenecked are observationally identical")
print("       under GPU-only monitoring. Only io_phase_end_frac (Layer 3)")# Phase table — low-GPU no-I/O vs IO_Bottlenecked indistinguishability
print("\n[GPU phase trajectory by tier/group]")

phase_df = combined[[
    "util_phase1", "util_phase2", "util_phase3",
    "io_phase_end_frac", "crosslayer_tier"
]].dropna(subset=["util_phase1", "util_phase2", "util_phase3"]).copy()

# For jobs without Darshan/heatmap, no observed I/O phase end.
phase_df["io_end_display"] = (
    pd.to_numeric(phase_df["io_phase_end_frac"], errors="coerce")
    .fillna(0)
    .clip(0, 1)
)

groups = {
    "Idle_like_or_timeout": phase_df["crosslayer_tier"].isin(["Ghost", "GPU_Idle_Timeout"]),
    "IO_Bottlenecked": phase_df["crosslayer_tier"].eq("IO_Bottlenecked"),
    "Scale_Inefficient": phase_df["crosslayer_tier"].eq("Scale_Waster"),
    "Compute_Bound": phase_df["crosslayer_tier"].eq("Compute_Bound"),
}

print(f"  {'Tier/group':<24} {'n':>6} {'P1':>5} {'P2':>5} {'P3':>5} {'io_end':>7}")

for label, mask in groups.items():
    sub = phase_df[mask]
    if len(sub) < 30:
        continue
    print(
        f"  {label:<24} {len(sub):>6,} "
        f"{sub['util_phase1'].mean():>5.1f} "
        f"{sub['util_phase2'].mean():>5.1f} "
        f"{sub['util_phase3'].mean():>5.1f} "
        f"{sub['io_end_display'].median():>7.3f}"
    )

print("  KEY: Idle-like/timeout jobs and IO_Bottlenecked jobs are nearly")
print("       indistinguishable under GPU-only monitoring. Layer-3 I/O timing")
print("       separates them: IO_Bottlenecked jobs show I/O activity persisting")
print("       close to job termination, while idle-like/timeout jobs do not.")

# Ghost power gap
ghost = combined[combined["crosslayer_tier"]=="Ghost"]
pwr   = ghost["power_mean"].notna()
n_elevated = (ghost.loc[pwr,"power_mean"] > 50).sum()
print(f"\n[Ghost DCGM intra-layer gap]")
print(f"  Ghost total: {len(ghost):,}")
print(f"  power>50W AND util<5% : {n_elevated:,} ({n_elevated/pwr.sum()*100:.1f}% of those with power data)")
print(f"  Queue distribution of power-elevated Ghost:")
if "QUEUE_NAME" in ghost.columns:
    for q, n in ghost.loc[ghost["power_mean"].fillna(0)>50,"QUEUE_NAME"].value_counts().head(3).items():
        print(f"    {q:<20}: {n:,} ({n/n_elevated*100:.1f}%)")

# ════════════════════════════════════════════════════════════════
# 5. FEATURE ENGINEERING & ML
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("FEATURE ENGINEERING + ML",flush=True)
print("=" * 60)
combined["QUEUED_TIMESTAMP"] = pd.to_datetime(combined["QUEUED_TIMESTAMP"])
combined["END_TIMESTAMP"]    = pd.to_datetime(combined["END_TIMESTAMP"])
combined = combined.sort_values("QUEUED_TIMESTAMP").reset_index(drop=True)
combined["submit_hour"]  = combined["QUEUED_TIMESTAMP"].dt.hour
combined["submit_dow"]   = combined["QUEUED_TIMESTAMP"].dt.dayofweek
combined["submit_month"] = combined["QUEUED_TIMESTAMP"].dt.month

train_df = combined[combined["use_for_training"]].copy().reset_index(drop=True)
TELEM_TIERS = {"Ghost","Scale_Waster","IO_Bottlenecked","Compute_Bound",
               "Moderate_Compute_No_IO","Low_Efficiency","GPU_Idle_Timeout",
               "Ideal_Compute_With_IO","Moderate_Compute_With_IO","Incidental_IO_Low_GPU"}
telem_df = train_df[train_df["crosslayer_tier"].isin(TELEM_TIERS)].copy().reset_index(drop=True)
split_t  = int(len(telem_df) * 0.80)

queue_freq = telem_df["QUEUE_NAME"].iloc[:split_t].value_counts()
exe_freq   = telem_df["executable"].iloc[:split_t].value_counts()
telem_df["queue_freq"]      = telem_df["QUEUE_NAME"].map(queue_freq).fillna(0)
telem_df["executable_freq"] = telem_df["executable"].map(exe_freq).fillna(0)
le = LabelEncoder()
le.fit(telem_df["SCIENCE_FIELD"].iloc[:split_t].astype(str))
known = set(le.classes_)
telem_df["SCIENCE_FIELD_enc"] = telem_df["SCIENCE_FIELD"].astype(str).apply(
    lambda x: le.transform([x])[0] if x in known else -1)

groupA = ["NODES_REQUESTED","WALLTIME_SECONDS","CORES_REQUESTED",
          "submit_hour","submit_dow","submit_month",
          "queue_freq","SCIENCE_FIELD_enc","executable_freq"]

def build_hist(train_df, target_jobs, cols, lookback_days=7, n_history=10):
    rows, lb_ns = [], np.timedelta64(lookback_days,"D")
    target_set  = set(target_jobs)
    for user, grp in train_df.groupby("USERNAME_GENID", sort=False):
        grp = grp.sort_values("QUEUED_TIMESTAMP").reset_index(drop=True)
        qts, ets = grp["QUEUED_TIMESTAMP"].values, grp["END_TIMESTAMP"].values
        for i in range(len(grp)):
            if grp.loc[i,"JOB_NAME"] not in target_set: continue
            mask     = (ets < qts[i]) & (ets >= qts[i] - lb_ns)
            past_idx = np.where(mask)[0][-n_history:]
            row = {"JOB_NAME": grp.loc[i,"JOB_NAME"]}
            if len(past_idx) > 0:
                p = grp.iloc[past_idx]
                row.update({
                    "user_job_count": int(mask.sum()),
                    "user_mean_runtime": p["RUNTIME_SECONDS"].mean(),
                    "user_walltime_efficiency": (p["RUNTIME_SECONDS"]/p["WALLTIME_SECONDS"].replace(0,np.nan)).mean(),
                    "user_fail_rate": (p["EXIT_STATUS"]!=0).mean(),
                    "user_quick_cancel_rate": (p["RUNTIME_SECONDS"]<60).mean(),
                    "user_mean_nodes": p["NODES_REQUESTED"].mean(),
                    "user_mean_walltime": p["WALLTIME_SECONDS"].mean(),
                })
                for c in cols:
                    v = p[c].dropna() if c in p.columns else pd.Series(dtype=float)
                    row[f"hist_{c}"] = v.mean() if len(v) else -1
            else:
                cur = grp.iloc[i]
                row.update({"user_job_count":0,"user_mean_runtime":cur["WALLTIME_SECONDS"],
                            "user_walltime_efficiency":0.5,"user_fail_rate":0.0,
                            "user_quick_cancel_rate":0.0,"user_mean_nodes":cur["NODES_REQUESTED"],
                            "user_mean_walltime":cur["WALLTIME_SECONDS"]})
                for c in cols: row[f"hist_{c}"] = -1
            rows.append(row)
    return pd.DataFrame(rows)

hist_cols = ["util_mean","idle_frac","zero_util_frac","power_efficiency","io_time_frac","bytes_per_gpu_hour"]
print(f"Building hist features ({LOOKBACK_DAYS}-day lookback, last {N_HIST} jobs)...")
hist_df  = build_hist(train_df, telem_df["JOB_NAME"].tolist(), hist_cols)
telem_df = telem_df.merge(hist_df, on="JOB_NAME", how="left")
print(f"  {len(hist_df):,} rows | {time.time()-t0:.0f}s elapsed")

groupB = ["user_job_count","user_mean_runtime","user_walltime_efficiency",
          "user_fail_rate","user_quick_cancel_rate","user_mean_nodes","user_mean_walltime"]
groupC = [f"hist_{c}" for c in hist_cols]
allABC = groupA + groupB + groupC

y_train = telem_df["is_wasteful"].iloc[:split_t].values
y_test  = telem_df["is_wasteful"].iloc[split_t:].values
print(f"  Train: {split_t:,} ({telem_df['QUEUED_TIMESTAMP'].iloc[0].date()} → "
      f"{telem_df['QUEUED_TIMESTAMP'].iloc[split_t-1].date()})")
print(f"  Test : {len(telem_df)-split_t:,} ({telem_df['QUEUED_TIMESTAMP'].iloc[split_t].date()} → "
      f"{telem_df['QUEUED_TIMESTAMP'].iloc[-1].date()})")

def to_X(df, feats):
    return np.nan_to_num(df[[f for f in feats if f in df.columns]].values, nan=-1, posinf=1e9, neginf=-1e9)

def fit_eval(feats, name, cw=None):
    avail = [f for f in feats if f in telem_df.columns]
    rf = RandomForestClassifier(n_estimators=200, class_weight=cw, n_jobs=-1, random_state=RNG)
    rf.fit(to_X(telem_df.iloc[:split_t], avail), y_train)
    prob = rf.predict_proba(to_X(telem_df.iloc[split_t:], avail))[:,1]
    pred = rf.predict(to_X(telem_df.iloc[split_t:], avail))
    return {"clf":rf,"prob":prob,"pred":pred,"auc":roc_auc_score(y_test,prob),
            "f1":f1_score(y_test,pred,average="macro"),"avail":avail}

M1   = fit_eval(groupA, "M1")
M3   = fit_eval(allABC, "M3")
ab_C = fit_eval(groupC, "C-only")
X_tr = to_X(telem_df.iloc[:split_t], M3["avail"])
X_te = to_X(telem_df.iloc[split_t:], M3["avail"])

# ════════════════════════════════════════════════════════════════
# 6. ML RESULTS
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("ML VALIDATION")
print("=" * 60)

# ----------------------------
# Baselines
# ----------------------------
dst = DummyClassifier(strategy="stratified", random_state=RNG)
dst.fit(np.zeros((split_t, 1)), y_train)

dst_prob = dst.predict_proba(np.zeros((len(y_test), 1)))[:, 1]
dst_pred = dst.predict(np.zeros((len(y_test), 1)))

auc_st = roc_auc_score(y_test, dst_prob)
f1_st  = f1_score(y_test, dst_pred, average="macro")


lr_wt = LogisticRegression(max_iter=1000, class_weight="balanced")
lr_wt.fit(telem_df[["WALLTIME_SECONDS"]].iloc[:split_t].values, y_train)

wt_prob = lr_wt.predict_proba(
    telem_df[["WALLTIME_SECONDS"]].iloc[split_t:].values
)[:, 1]
wt_pred = lr_wt.predict(
    telem_df[["WALLTIME_SECONDS"]].iloc[split_t:].values
)

auc_wt = roc_auc_score(y_test, wt_prob)
f1_wt  = f1_score(y_test, wt_pred, average="macro")


# ----------------------------
# Model comparison table
# ----------------------------
print(f"\n[Model comparison]")
print(f"  {'Model':<25} {'AUC':>8} {'macro-F1':>10}")
print(f"  {'-'*25} {'-'*8} {'-'*10}")

for label, auc, f1 in [
    ("Stratified baseline",  auc_st,     f1_st),
    ("Walltime-only LR",     auc_wt,     f1_wt),
    ("M1 (scheduler only)",  M1["auc"],  M1["f1"]),
    ("M2 (history only)",    ab_C["auc"], ab_C["f1"]),
    ("M3 (cross-layer)",     M3["auc"],  M3["f1"]),
]:
    print(f"  {label:<25} {auc:>8.4f} {f1:>10.3f}")
# Bootstrap CI
rng   = np.random.RandomState(RNG)
boots = [roc_auc_score(y_test[idx:=rng.randint(0,len(y_test),len(y_test))], M3["prob"][idx])
         for _ in range(1000)]
ci_lo, ci_hi = np.percentile(boots, [2.5, 97.5])
print(f"\nM3 bootstrap 95% CI : [{ci_lo:.4f}, {ci_hi:.4f}]")
print(f"M3 lift over M1     : +{M3['auc']-M1['auc']:.4f} AUC")
print(f"M2 lift over M1     : +{ab_C['auc']-M1['auc']:.4f} AUC")

# Temporal CV
X_all = to_X(telem_df, M3["avail"])
X_a   = to_X(telem_df, M1["avail"])
y_all = telem_df["is_wasteful"].values
tscv  = TimeSeriesSplit(n_splits=5)
m1_cv, m3_cv = [], []
for tr, te in tscv.split(X_all):
    rf3 = RandomForestClassifier(n_estimators=200,n_jobs=-1,random_state=RNG).fit(X_all[tr],y_all[tr])
    rf1 = RandomForestClassifier(n_estimators=200,n_jobs=-1,random_state=RNG).fit(X_a[tr],y_all[tr])
    m3_cv.append(roc_auc_score(y_all[te], rf3.predict_proba(X_all[te])[:,1]))
    m1_cv.append(roc_auc_score(y_all[te], rf1.predict_proba(X_a[te])[:,1]))
print(f"\n5-fold temporal CV:")
print(f"  M1: {np.mean(m1_cv):.3f} ± {np.std(m1_cv):.3f}  M3: {np.mean(m3_cv):.3f} ± {np.std(m3_cv):.3f}")
print(f"  M3 > M1 in {sum(m>m1 for m,m1 in zip(m3_cv,m1_cv))}/5 folds")

# OvR per-tier necessity
print(f"\n[OvR per-tier necessity]")
print(f"  {'Tier':<22} {'Sched':>7} {'Cross-layer':>12} {'Δ':>7}  verdict")
for tier in ["Ghost","IO_Bottlenecked","Scale_Waster","GPU_Idle_Timeout"]:
    y_tr = (telem_df["crosslayer_tier"].iloc[:split_t]==tier).astype(int).values
    y_te = (telem_df["crosslayer_tier"].iloc[split_t:]==tier).astype(int).values
    if y_te.sum() < 10: continue
    aucs = {}
    for gname, feats in [("A",groupA),("M3",allABC)]:
        avail = [f for f in feats if f in telem_df.columns]
        rf = RandomForestClassifier(n_estimators=200,class_weight="balanced",n_jobs=-1,random_state=RNG)
        rf.fit(to_X(telem_df.iloc[:split_t],avail), y_tr)
        aucs[gname] = roc_auc_score(y_te, rf.predict_proba(to_X(telem_df.iloc[split_t:],avail))[:,1])
    d = aucs["M3"]-aucs["A"]
    verdict = "essential (Δ>0.10)" if d>0.10 else "helpful (Δ>0.04)" if d>0.04 else "marginal"
    print(f"  {tier:<22} {aucs['A']:>7.4f} {aucs['M3']:>12.4f} {d:>7.4f}  {verdict}")

# Per-tier recall
print(f"\n[Per-tier recall on test period]")
def wilson_ci(k,n,z=1.96):
    if n==0: return (0.,1.)
    p=k/n;d=1+z*z/n;c=(p+z*z/(2*n))/d;h=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return (max(0,c-h),min(1,c+h))

test_sl = telem_df.iloc[split_t:].copy().reset_index(drop=True)
test_sl["pred"] = M3["pred"]
for tier, sub in test_sl.groupby("crosslayer_tier"):
    if len(sub) < 30: continue
    n_pos = (sub["is_wasteful"]==1).sum()
    if n_pos == 0:
        flagged = (sub["pred"]==1).sum()
        lo,hi = wilson_ci(flagged,len(sub))
        print(f"  {tier:<28} n={len(sub):>5,}  FPR={flagged/len(sub):.3f} [{lo:.2f},{hi:.2f}]")
    else:
        flagged = ((sub["is_wasteful"]==1)&(sub["pred"]==1)).sum()
        lo,hi = wilson_ci(flagged,n_pos)
        print(f"  {tier:<28} n={len(sub):>5,}  rec={flagged/n_pos:.3f} [{lo:.2f},{hi:.2f}]")

# ════════════════════════════════════════════════════════════════
# 7. WASTE BURST STRUCTURE
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("WASTE BURST STRUCTURE")
print("=" * 60)
cs = combined.sort_values(["USERNAME_GENID","QUEUED_TIMESTAMP"]).reset_index(drop=True)
cs["prev_w"]   = cs.groupby("USERNAME_GENID")["is_wasteful"].shift(1).fillna(0)
cs["in_burst"] = ((cs["is_wasteful"]==1)&(cs["prev_w"]==1)).astype(int)

SUBST  = {"Ghost","IO_Bottlenecked","Scale_Waster","Failed_Job","GPU_Idle_Timeout"}
sub_w  = cs[cs["crosslayer_tier"].isin(SUBST)]
sub_b  = cs[cs["crosslayer_tier"].isin(SUBST) & (cs["in_burst"]==1)]
all_w  = cs[cs["is_wasteful"]==1]
all_b  = cs[cs["in_burst"]==1]
print(f"Including Quick_Cancel  : {len(all_b):,}/{len(all_w):,} ({len(all_b)/len(all_w)*100:.1f}%)")
print(f"Substantive tiers only  : {len(sub_b):,}/{len(sub_w):,} ({len(sub_b)/max(len(sub_w),1)*100:.1f}%)")

tm = telem_df[["JOB_NAME","hist_idle_frac","is_wasteful"]].merge(
    cs[["JOB_NAME","in_burst"]], on="JOB_NAME", how="left").fillna(0)
valid = tm[["hist_idle_frac","in_burst"]].dropna()
r_burst, _ = spearmanr(valid["hist_idle_frac"], valid["in_burst"])
burst_hi   = (tm[tm["in_burst"]==1]["hist_idle_frac"].fillna(0) > 0.5).mean() * 100
print(f"Spearman r(hist_idle_frac, in_burst) = {r_burst:.3f} (N={len(valid):,})")
print(f"{burst_hi:.1f}% of in-burst jobs have hist_idle_frac > 0.5 at submission")

# ════════════════════════════════════════════════════════════════
# 8. THRESHOLD SENSITIVITY
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("THRESHOLD SENSITIVITY")
print("=" * 60)
drop_cols = ["crosslayer_tier","diagnostic_tier","use_for_training","_scale_high",
             "gpu_waste_score","io_waste_score","scale_factor","cross_layer_waste",
             "bytes_per_gpu_hour","gpu_hours","BWio_MB","io_time_seconds",
             "darshan_present","total_bytes","io_detected","has_gpu","gpus","bytes_out","exit_failed"]
print(f"  {'Ghost<':>8} {'Scale<':>8} {'Ghost':>8} {'Scale_Waster':>13}")
for g in [3.0,5.0,7.0]:
    for s in [8.0,10.0,12.0]:
        if g >= s: continue
        cu.GPU_GHOST_UTIL, cu.GPU_UTIL_LOW = g, s
        sys.stdout = open(os.devnull,'w')
        tmp = cu.classify_crosslayer(combined.drop(columns=drop_cols, errors="ignore"))
        sys.stdout = sys.__stdout__
        marker = " <- base" if (g==5.0 and s==10.0) else ""
        print(f"  {g:>6.0f}%   {s:>6.0f}%   {(tmp['crosslayer_tier']=='Ghost').sum():>8,}   "
              f"{(tmp['crosslayer_tier']=='Scale_Waster').sum():>8,}{marker}")
cu.GPU_GHOST_UTIL, cu.GPU_UTIL_LOW = 5.0, 10.0

# ════════════════════════════════════════════════════════════════
# 9. DISCUSSION NUMBERS
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("DISCUSSION / CONTEXT NUMBERS")
print("=" * 60)
has_gpu_jobs = combined[combined["has_gpu"]]
zero_util = (has_gpu_jobs["util_mean"].fillna(0) == 0).sum()
half_idle = (has_gpu_jobs["idle_frac"].fillna(0) > 0.5).sum() if "idle_frac" in has_gpu_jobs else 0
mean_util = has_gpu_jobs["util_mean"].mean()
print(f"Zero GPU utilization (of DCGM jobs) : {zero_util:,} ({zero_util/len(has_gpu_jobs)*100:.1f}%)")
print(f"Jobs >50% idle frac                 : {half_idle:,} ({half_idle/len(has_gpu_jobs)*100:.1f}%)")
print(f"Mean GPU utilization (DCGM jobs)    : {mean_util:.1f}%")

# ════════════════════════════════════════════════════════════════
# 10. SANITY CHECKS
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("SANITY CHECKS")
print("=" * 60)
issues = []
assert telem_df["QUEUED_TIMESTAMP"].is_monotonic_increasing, "Temporal ordering failed"
print("[1] Temporal ordering ✓")
leaky = ["util_mean","idle_frac","zero_util_frac","power_efficiency","io_time_frac","bytes_per_gpu_hour","gpu_hours"]
leaked = [f for f in M3["avail"] if f in leaky]
print(f"[2] No telemetry leak: {'✓' if not leaked else '✗ '+str(leaked)}")
if leaked: issues.append(f"LEAK:{leaked}")
assert combined["JOB_NAME"].duplicated().sum() == 0
print("[3] No duplicate JOB_NAME ✓")
n_inf = np.isinf(X_tr).sum() + np.isinf(X_te).sum()
n_nan = np.isnan(X_tr).sum() + np.isnan(X_te).sum()
print(f"[4] Feature matrix clean: {'✓' if (n_inf+n_nan)==0 else f'✗ inf={n_inf} nan={n_nan}'}")
overlap = set(telem_df["JOB_NAME"].iloc[:split_t]) & set(telem_df["JOB_NAME"].iloc[split_t:])
print(f"[5] No train/test overlap: {'✓' if not overlap else f'✗ {len(overlap)}'}")
sha = hashlib.sha256(telem_df[M3["avail"]].fillna(-999).values.tobytes()).hexdigest()[:16]
print(f"[6] Feature matrix SHA256[:16]: {sha}")
print(f"\nTotal runtime: {(time.time()-t0)/60:.1f} min")
print("All checks passed ✓" if not issues else f"Issues: {issues}")

# ════════════════════════════════════════════════════════════════
# MULTICLASS TIER PREDICTION (actionable waste tiers)
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("MULTICLASS TIER PREDICTION")
print("=" * 60)

from sklearn.metrics import classification_report

ACTION_TIERS = ["Ghost","Scale_Waster","IO_Bottlenecked",
                "Failed_Job","GPU_Idle_Timeout","Not_Wasteful"]

# Build multiclass target
def make_mc_label(tier, is_wasteful):
    if tier in {"Ghost","Scale_Waster","IO_Bottlenecked",
                "GPU_Idle_Timeout"} :
        return tier
    if is_wasteful:
        return "Failed_Job"  # QC+Failed combined
    return "Not_Wasteful"

telem_df["mc_label"] = [
    make_mc_label(t, w) for t, w in
    zip(telem_df["crosslayer_tier"], telem_df["is_wasteful"])
]

from sklearn.preprocessing import LabelEncoder
mc_le = LabelEncoder()
mc_le.fit(telem_df["mc_label"].iloc[:split_t])
y_mc_train = mc_le.transform(telem_df["mc_label"].iloc[:split_t])
y_mc_test  = mc_le.transform(telem_df["mc_label"].iloc[split_t:])

# M1 multiclass
rf_mc_m1 = RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                   n_jobs=-1, random_state=RNG)
rf_mc_m1.fit(to_X(telem_df.iloc[:split_t], groupA), y_mc_train)
pred_mc_m1 = rf_mc_m1.predict(to_X(telem_df.iloc[split_t:], groupA))

# M3 multiclass
avail_abc = [f for f in allABC if f in telem_df.columns]
rf_mc_m3 = RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                   n_jobs=-1, random_state=RNG)
rf_mc_m3.fit(to_X(telem_df.iloc[:split_t], avail_abc), y_mc_train)
pred_mc_m3 = rf_mc_m3.predict(to_X(telem_df.iloc[split_t:], avail_abc))

print("\n[M1 — scheduler only]")
print(classification_report(y_mc_test, pred_mc_m1,
      target_names=mc_le.classes_, digits=3, zero_division=0))

print("\n[M3 — cross-layer]")
print(classification_report(y_mc_test, pred_mc_m3,
      target_names=mc_le.classes_, digits=3, zero_division=0))

# Macro F1 comparison
from sklearn.metrics import f1_score as f1s
print(f"M1 macro-F1 (multiclass): {f1s(y_mc_test,pred_mc_m1,average='macro',zero_division=0):.4f}")
print(f"M3 macro-F1 (multiclass): {f1s(y_mc_test,pred_mc_m3,average='macro',zero_division=0):.4f}")

# ════════════════════════════════════════════════════════════════
# MODEL COMPARISON: RF vs BOOSTING ENSEMBLE
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("MODEL COMPARISON: RF vs BOOSTING")
print("=" * 60)

from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("xgboost not installed, skipping")

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    print("lightgbm not installed, skipping")

avail_abc = [f for f in allABC if f in telem_df.columns]
X_tr_full = to_X(telem_df.iloc[:split_t], avail_abc)
X_te_full = to_X(telem_df.iloc[split_t:], avail_abc)

# Compute class weight ratio for imbalanced classifiers
n_neg = (y_train == 0).sum()
n_pos = (y_train == 1).sum()
scale_pos = n_neg / n_pos
print(f"Class ratio (neg/pos): {scale_pos:.2f}")

results = {}

# 1. Random Forest (already trained as M3, just record)
results["RF (M3 baseline)"] = {"auc": M3["auc"], "f1": M3["f1"]}

# 2. HistGradientBoosting (sklearn, no extra install needed)
hgb = HistGradientBoostingClassifier(
    max_iter=300, learning_rate=0.05, max_depth=6,
    min_samples_leaf=20, random_state=RNG)
hgb.fit(X_tr_full, y_train)
hgb_prob = hgb.predict_proba(X_te_full)[:,1]
hgb_pred = hgb.predict(X_te_full)
results["HistGradBoost (sklearn)"] = {
    "auc": roc_auc_score(y_test, hgb_prob),
    "f1":  f1_score(y_test, hgb_pred, average="macro")
}

# 3. XGBoost
if HAS_XGB:
    xgb_model = xgb.XGBClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=scale_pos,
        eval_metric="auc", random_state=RNG,
        n_jobs=-1, verbosity=0)
    xgb_model.fit(X_tr_full, y_train)
    xgb_prob = xgb_model.predict_proba(X_te_full)[:,1]
    xgb_pred = xgb_model.predict(X_te_full)
    results["XGBoost"] = {
        "auc": roc_auc_score(y_test, xgb_prob),
        "f1":  f1_score(y_test, xgb_pred, average="macro")
    }

# 4. LightGBM
if HAS_LGB:
    lgb_model = lgb.LGBMClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=6,
        num_leaves=63, subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=scale_pos,
        random_state=RNG, n_jobs=-1, verbose=-1)
    lgb_model.fit(X_tr_full, y_train)
    lgb_prob = lgb_model.predict_proba(X_te_full)[:,1]
    lgb_pred = lgb_model.predict(X_te_full)
    results["LightGBM"] = {
        "auc": roc_auc_score(y_test, lgb_prob),
        "f1":  f1_score(y_test, lgb_pred, average="macro")
    }

# 5. Logistic Regression with full features (linear baseline)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_tr_scaled = scaler.fit_transform(X_tr_full)
X_te_scaled = scaler.transform(X_te_full)
lr_full = LogisticRegression(max_iter=1000, class_weight="balanced",
                              C=1.0, random_state=RNG)
lr_full.fit(X_tr_scaled, y_train)
lr_prob = lr_full.predict_proba(X_te_scaled)[:,1]
lr_pred = lr_full.predict(X_te_scaled)
results["Logistic Regression (full)"] = {
    "auc": roc_auc_score(y_test, lr_prob),
    "f1":  f1_score(y_test, lr_pred, average="macro")
}

# Print comparison table
print(f"\n{'Model':<30} {'AUC':>8} {'Macro-F1':>10}")
print("-" * 50)
for name, r in results.items():
    print(f"{name:<30} {r['auc']:>8.4f} {r['f1']:>10.4f}")

# Bootstrap CI on best model
best_name = max(results, key=lambda k: results[k]["auc"])
print(f"\nBest model: {best_name} (AUC={results[best_name]['auc']:.4f})")

# CV on best non-RF model if it's not RF
if best_name != "RF (M3 baseline)":
    print(f"\n5-fold temporal CV on {best_name}:")
    if "XGBoost" in best_name and HAS_XGB:
        best_clf = xgb.XGBClassifier(n_estimators=300,learning_rate=0.05,
            max_depth=6,subsample=0.8,colsample_bytree=0.8,
            scale_pos_weight=scale_pos,random_state=RNG,n_jobs=-1,verbosity=0)
    elif "LightGBM" in best_name and HAS_LGB:
        best_clf = lgb.LGBMClassifier(n_estimators=300,learning_rate=0.05,
            max_depth=6,num_leaves=63,subsample=0.8,colsample_bytree=0.8,
            scale_pos_weight=scale_pos,random_state=RNG,n_jobs=-1,verbose=-1)
    elif "HistGrad" in best_name:
        best_clf = HistGradientBoostingClassifier(
            max_iter=300,learning_rate=0.05,max_depth=6,
            min_samples_leaf=20,random_state=RNG)
    else:
        best_clf = None

    if best_clf is not None:
        X_all_full = to_X(telem_df, avail_abc)
        y_all = telem_df["is_wasteful"].values
        cv_aucs = []
        for tr, te in TimeSeriesSplit(n_splits=5).split(X_all_full):
            best_clf.fit(X_all_full[tr], y_all[tr])
            cv_aucs.append(roc_auc_score(
                y_all[te],
                best_clf.predict_proba(X_all_full[te])[:,1]))
        print(f"  AUC: {np.mean(cv_aucs):.3f} ± {np.std(cv_aucs):.3f}")


# ════════════════════════════════════════════════════════════════
# MULTICLASS TIER PREDICTION WITH BEST MODEL
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("MULTICLASS TIER PREDICTION (best model)")
print("=" * 60)

from sklearn.metrics import classification_report, f1_score as f1s
from sklearn.preprocessing import LabelEncoder as LE2

# Build multiclass target — 6 actionable tiers
def make_mc_label(tier):
    if tier in {"Ghost","Scale_Waster","IO_Bottlenecked","GPU_Idle_Timeout"}:
        return tier
    if tier in {"Failed_Job","Quick_Cancel"}:
        return "Failed_or_Cancel"
    return "Productive"

telem_df["mc_label"] = telem_df["crosslayer_tier"].apply(make_mc_label)

mc_le = LE2()
mc_le.fit(telem_df["mc_label"].iloc[:split_t])
# Handle unseen labels in test gracefully
known_mc = set(mc_le.classes_)
test_labels = telem_df["mc_label"].iloc[split_t:].apply(
    lambda x: x if x in known_mc else "Productive")
y_mc_train = mc_le.transform(telem_df["mc_label"].iloc[:split_t])
y_mc_test  = mc_le.transform(test_labels)

avail_abc = [f for f in allABC if f in telem_df.columns]
X_tr_full = to_X(telem_df.iloc[:split_t], avail_abc)
X_te_full = to_X(telem_df.iloc[split_t:], avail_abc)
X_tr_m1   = to_X(telem_df.iloc[:split_t], groupA)
X_te_m1   = to_X(telem_df.iloc[split_t:], groupA)

# Train RF multiclass (M1 vs M3 for comparison)
rf_mc_m1 = RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                   n_jobs=-1, random_state=RNG)
rf_mc_m1.fit(X_tr_m1, y_mc_train)
pred_mc_m1 = rf_mc_m1.predict(X_te_m1)

rf_mc_m3 = RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                   n_jobs=-1, random_state=RNG)
rf_mc_m3.fit(X_tr_full, y_mc_train)
pred_mc_m3 = rf_mc_m3.predict(X_te_full)

print("\n[RF M1 — scheduler only]")
print(classification_report(y_mc_test, pred_mc_m1,
      target_names=mc_le.classes_, digits=3, zero_division=0))

print("\n[RF M3 — cross-layer]")
print(classification_report(y_mc_test, pred_mc_m3,
      target_names=mc_le.classes_, digits=3, zero_division=0))

# Train best boosting model multiclass
best_mc_results = {
    "RF M3": {"pred": pred_mc_m3,
              "f1": f1s(y_mc_test, pred_mc_m3, average="macro", zero_division=0)}
}

if HAS_XGB:
    xgb_mc = xgb.XGBClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8,
        random_state=RNG, n_jobs=-1, verbosity=0,
        objective="multi:softprob",
        num_class=len(mc_le.classes_))
    xgb_mc.fit(X_tr_full, y_mc_train)
    pred_xgb_mc = xgb_mc.predict(X_te_full)
    best_mc_results["XGBoost M3"] = {
        "pred": pred_xgb_mc,
        "f1": f1s(y_mc_test, pred_xgb_mc, average="macro", zero_division=0)
    }
    print("\n[XGBoost M3 — cross-layer]")
    print(classification_report(y_mc_test, pred_xgb_mc,
          target_names=mc_le.classes_, digits=3, zero_division=0))

if HAS_LGB:
    lgb_mc = lgb.LGBMClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=6,
        num_leaves=63, subsample=0.8, colsample_bytree=0.8,
        class_weight="balanced",
        random_state=RNG, n_jobs=-1, verbose=-1)
    lgb_mc.fit(X_tr_full, y_mc_train)
    pred_lgb_mc = lgb_mc.predict(X_te_full)
    best_mc_results["LightGBM M3"] = {
        "pred": pred_lgb_mc,
        "f1": f1s(y_mc_test, pred_lgb_mc, average="macro", zero_division=0)
    }
    print("\n[LightGBM M3 — cross-layer]")
    print(classification_report(y_mc_test, pred_lgb_mc,
          target_names=mc_le.classes_, digits=3, zero_division=0))

# Summary
print("\n[Multiclass macro-F1 summary]",flush=True)
print(f"{'Model':<20} {'Macro-F1':>10}")
print("-" * 32)
for name, r in sorted(best_mc_results.items(), key=lambda x: -x[1]["f1"]):
    print(f"{name:<20} {r['f1']:>10.4f}")

best_mc_name = max(best_mc_results, key=lambda k: best_mc_results[k]["f1"])
print(f"\nBest multiclass model: {best_mc_name}")

# 5-fold temporal CV on best multiclass model
print(f"\n5-fold temporal CV (multiclass, {best_mc_name}):")
X_all_full = to_X(telem_df, avail_abc)
y_mc_all   = mc_le.transform(telem_df["mc_label"].apply(
    lambda x: x if x in known_mc else "Productive"))
tscv = TimeSeriesSplit(n_splits=5)
cv_f1s = []

for tr, te in tscv.split(X_all_full):
    if "XGBoost" in best_mc_name and HAS_XGB:
        clf = xgb.XGBClassifier(n_estimators=300,learning_rate=0.05,
            max_depth=6,subsample=0.8,colsample_bytree=0.8,
            random_state=RNG,n_jobs=-1,verbosity=0,
            objective="multi:softprob",num_class=len(mc_le.classes_))
    elif "LightGBM" in best_mc_name and HAS_LGB:
        clf = lgb.LGBMClassifier(n_estimators=300,learning_rate=0.05,
            max_depth=6,num_leaves=63,subsample=0.8,colsample_bytree=0.8,
            class_weight="balanced",random_state=RNG,n_jobs=-1,verbose=-1)
    else:
        clf = RandomForestClassifier(n_estimators=200,class_weight="balanced",
                                     n_jobs=-1,random_state=RNG)
    clf.fit(X_all_full[tr], y_mc_all[tr])
    pred_cv = clf.predict(X_all_full[te])
    cv_f1s.append(f1s(y_mc_all[te], pred_cv, average="macro", zero_division=0))

print(f"  Macro-F1: {np.mean(cv_f1s):.3f} ± {np.std(cv_f1s):.3f}")
print(f"  Per-fold: {[f'{x:.3f}' for x in cv_f1s]}")
# ════════════════════════════════════════════════════════════════
# EXPANDED HISTORICAL FEATURE TEST (M4)
# Scheduler + user history + expanded historical GPU/I/O behavior
# Leakage-safe: only prior completed jobs are used.
# ════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("EXPANDED HISTORICAL FEATURE TEST (M4)")
print("=" * 60)

# Candidate completed-job telemetry features.
# These are NOT used from the current job. Only hist_* aggregates from prior jobs
# whose END_TIMESTAMP < current QUEUED_TIMESTAMP are used.
expanded_hist_candidates = [
    # GPU utilization distribution
    "util_mean", "util_max", "util_std", "util_p25", "util_p50",
    "util_p75", "util_p95", "zero_util_frac", "idle_frac",
    "active_phase_frac", "max_consecutive_idle_readings",

    # GPU memory behavior
    "mem_util_mean", "mem_util_max", "mem_pressure_frac", "mem_bound_frac",
    "gpu_mem_alloc_mean_kb", "gpu_mem_alloc_max_kb", "gpu_mem_alloc_std_kb",

    # Power / thermal behavior
    "power_mean", "power_max", "power_std", "power_p95", "power_efficiency",
    "high_power_low_util_frac", "power_cap_proximity_mean", "near_power_cap_frac",
    "temp_mean", "temp_max", "temp_p95", "thermal_throttle_frac",
    "sustained_throttle_frac",

    # Cross-signal GPU behavior
    "high_util_low_mem_frac", "high_mem_low_util_frac",

    # Temporal GPU phases
    "util_warmup_mean", "util_cooldown_mean", "util_first_half",
    "util_second_half", "phase_drop", "util_phase1", "util_phase2",
    "util_phase3",

    # Cross-GPU / cross-node imbalance
    "gpu_imbalance_mean", "gpu_imbalance_max",
    "gpu_mem_imbalance_mean", "gpu_power_imbalance_mean",
    "gpu_temp_imbalance_mean", "node_util_imbalance_std",
    "node_util_imbalance_max", "node_count_observed",

    # Telemetry quality
    "telemetry_coverage_frac", "telemetry_gap_detected",

    # I/O behavior
    "io_time_frac", "bytes_per_gpu_hour", "BWio_MB", "total_bytes",
    "bytes_read", "bytes_written", "posix_reads", "posix_writes",
    "posix_opens", "posix_stats", "mpiio_bytes_read",
    "mpiio_bytes_written", "mpiio_coll_reads", "mpiio_coll_writes",
    "mpiio_indep_reads", "mpiio_indep_writes",
    "stdio_bytes_read", "stdio_bytes_written", "metadata_ops_per_gb",
    "unique_files", "seq_read_ratio", "seq_write_ratio",
    "small_read_ratio", "large_read_ratio", "write_dominance",
    "mpiio_coll_ratio", "rank_imbalance", "rank_time_imbalance",
    "rank_time_gap", "variance_rank_time", "slowest_rank_time",
    "fastest_rank_time", "io_phase_start_frac", "io_phase_end_frac",
    "io_read_front_heavy", "io_write_back_heavy",
    "mem_not_aligned_ratio", "file_not_aligned_ratio",
]

# Do not include labels or target-derived features as historical predictors.
blocked_hist_cols = {
    "crosslayer_tier", "diagnostic_tier", "is_wasteful", "use_for_training",
    "gpu_waste_score", "io_waste_score", "cross_layer_waste",
    "_scale_high", "scale_factor",
}

expanded_hist_cols = [
    c for c in expanded_hist_candidates
    if c in train_df.columns and c not in blocked_hist_cols
]

print(f"Expanded historical columns available: {len(expanded_hist_cols)}")
print(expanded_hist_cols)

def build_hist_fast(train_df, target_jobs, cols, lookback_days=7, n_history=10):
    """
    Faster leakage-safe historical feature builder.

    For each target job, use only prior jobs from the same user where:
        prior END_TIMESTAMP < current QUEUED_TIMESTAMP
        prior END_TIMESTAMP >= current QUEUED_TIMESTAMP - lookback_days

    Then aggregate the most recent n_history prior jobs.
    """
    rows = []
    target_set = set(target_jobs)
    lb_ns = np.timedelta64(lookback_days, "D")

    # Ensure requested columns are numeric/boolean-compatible.
    cols = [c for c in cols if c in train_df.columns]

    for user, grp in train_df.groupby("USERNAME_GENID", sort=False):
        grp = grp.sort_values("QUEUED_TIMESTAMP").reset_index(drop=True)

        qts = grp["QUEUED_TIMESTAMP"].values
        ets = grp["END_TIMESTAMP"].values
        job_names = grp["JOB_NAME"].values

        hist_mat = grp[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)

        target_indices = np.where(np.isin(job_names, list(target_set)))[0]

        for i in target_indices:
            mask = (ets < qts[i]) & (ets >= qts[i] - lb_ns)
            past_idx = np.where(mask)[0][-n_history:]

            row = {"JOB_NAME": job_names[i]}

            if len(past_idx) > 0:
                p = grp.iloc[past_idx]

                row.update({
                    "user_job_count": int(mask.sum()),
                    "user_mean_runtime": p["RUNTIME_SECONDS"].mean(),
                    "user_walltime_efficiency": (
                        p["RUNTIME_SECONDS"] /
                        p["WALLTIME_SECONDS"].replace(0, np.nan)
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


# Rebuild a clean trainable telemetry frame so M3 and M4 are compared fairly.
telem_exp = train_df[
    train_df["crosslayer_tier"].isin(TELEM_TIERS)
].copy().reset_index(drop=True)

split_exp = int(len(telem_exp) * 0.80)

# Recreate scheduler-visible encodings using train prefix only.
queue_freq_exp = telem_exp["QUEUE_NAME"].iloc[:split_exp].value_counts()
exe_freq_exp   = telem_exp["executable"].iloc[:split_exp].value_counts()

telem_exp["queue_freq"] = telem_exp["QUEUE_NAME"].map(queue_freq_exp).fillna(0)
telem_exp["executable_freq"] = telem_exp["executable"].map(exe_freq_exp).fillna(0)

le_exp = LabelEncoder()
le_exp.fit(telem_exp["SCIENCE_FIELD"].iloc[:split_exp].astype(str))
known_exp = set(le_exp.classes_)
telem_exp["SCIENCE_FIELD_enc"] = telem_exp["SCIENCE_FIELD"].astype(str).apply(
    lambda x: le_exp.transform([x])[0] if x in known_exp else -1
)

groupA_exp = [
    "NODES_REQUESTED", "WALLTIME_SECONDS", "CORES_REQUESTED",
    "submit_hour", "submit_dow", "submit_month",
    "queue_freq", "SCIENCE_FIELD_enc", "executable_freq"
]

compact_hist_cols = [
    "util_mean", "idle_frac", "zero_util_frac",
    "power_efficiency", "io_time_frac", "bytes_per_gpu_hour"
]
compact_hist_cols = [c for c in compact_hist_cols if c in train_df.columns]

# Build expanded histories once. Compact M3 is a subset of the same frame.
print(
    f"Building expanded historical features "
    f"({LOOKBACK_DAYS}-day lookback, last {N_HIST} jobs)..."
)
t_m4 = time.time()

hist_exp_df = build_hist_fast(
    train_df=train_df,
    target_jobs=telem_exp["JOB_NAME"].tolist(),
    cols=expanded_hist_cols,
    lookback_days=LOOKBACK_DAYS,
    n_history=N_HIST,
)

telem_exp = telem_exp.merge(hist_exp_df, on="JOB_NAME", how="left")

print(f"  {len(hist_exp_df):,} rows | {time.time() - t_m4:.1f}s elapsed")

groupB_exp = [
    "user_job_count", "user_mean_runtime", "user_walltime_efficiency",
    "user_fail_rate", "user_quick_cancel_rate",
    "user_mean_nodes", "user_mean_walltime"
]

groupC_compact_exp = [f"hist_{c}" for c in compact_hist_cols if f"hist_{c}" in telem_exp.columns]
groupD_expanded    = [f"hist_{c}" for c in expanded_hist_cols if f"hist_{c}" in telem_exp.columns]

M1_feats_exp = groupA_exp
M3_feats_exp = groupA_exp + groupB_exp + groupC_compact_exp
M4_feats_exp = groupA_exp + groupB_exp + groupD_expanded

y_train_exp = telem_exp["is_wasteful"].iloc[:split_exp].values
y_test_exp  = telem_exp["is_wasteful"].iloc[split_exp:].values

def to_X_exp(df, feats):
    avail = [f for f in feats if f in df.columns]
    X = df[avail].apply(pd.to_numeric, errors="coerce").values
    return np.nan_to_num(X, nan=-1, posinf=1e9, neginf=-1e9), avail

def fit_eval_exp(feats, name):
    X_tr, avail = to_X_exp(telem_exp.iloc[:split_exp], feats)
    X_te, _     = to_X_exp(telem_exp.iloc[split_exp:], avail)

    clf = RandomForestClassifier(
        n_estimators=200,
        class_weight=None,
        n_jobs=-1,
        random_state=RNG,
    )
    clf.fit(X_tr, y_train_exp)

    prob = clf.predict_proba(X_te)[:, 1]
    pred = clf.predict(X_te)

    out = {
        "name": name,
        "clf": clf,
        "features": avail,
        "prob": prob,
        "pred": pred,
        "auc": roc_auc_score(y_test_exp, prob),
        "f1": f1_score(y_test_exp, pred, average="macro"),
    }
    return out

M1_exp = fit_eval_exp(M1_feats_exp, "M1 scheduler only")
M3_exp = fit_eval_exp(M3_feats_exp, "M3 compact history")
M4_exp = fit_eval_exp(M4_feats_exp, "M4 expanded history")

print("\n[Expanded historical feature comparison]")
print(f"{'Model':<25} {'#feat':>7} {'AUC':>8} {'Macro-F1':>10}")
print("-" * 55)
for r in [M1_exp, M3_exp, M4_exp]:
    print(f"{r['name']:<25} {len(r['features']):>7} {r['auc']:>8.4f} {r['f1']:>10.4f}")

print(f"\nM4 lift over M3: +{M4_exp['auc'] - M3_exp['auc']:.4f} AUC")
print(f"M4 lift over M1: +{M4_exp['auc'] - M1_exp['auc']:.4f} AUC")

# Feature importance for M4
importances = pd.Series(
    M4_exp["clf"].feature_importances_,
    index=M4_exp["features"]
).sort_values(ascending=False)

print("\n[M4 top 30 feature importances]")
for feat, val in importances.head(30).items():
    print(f"  {feat:<40} {val:.5f}")

# Optional temporal CV for the expanded model.
print("\n[5-fold temporal CV: M3 compact vs M4 expanded]")
X_m3_all, m3_avail = to_X_exp(telem_exp, M3_feats_exp)
X_m4_all, m4_avail = to_X_exp(telem_exp, M4_feats_exp)
y_all_exp = telem_exp["is_wasteful"].values

m3_cv, m4_cv = [], []
for tr, te in TimeSeriesSplit(n_splits=5).split(X_m4_all):
    rf3 = RandomForestClassifier(
        n_estimators=200, n_jobs=-1, random_state=RNG
    ).fit(X_m3_all[tr], y_all_exp[tr])

    rf4 = RandomForestClassifier(
        n_estimators=200, n_jobs=-1, random_state=RNG
    ).fit(X_m4_all[tr], y_all_exp[tr])

    m3_cv.append(roc_auc_score(y_all_exp[te], rf3.predict_proba(X_m3_all[te])[:, 1]))
    m4_cv.append(roc_auc_score(y_all_exp[te], rf4.predict_proba(X_m4_all[te])[:, 1]))

print(f"  M3 compact : {np.mean(m3_cv):.4f} ± {np.std(m3_cv):.4f}")
print(f"  M4 expanded: {np.mean(m4_cv):.4f} ± {np.std(m4_cv):.4f}")
print(f"  M4 > M3 in {sum(m4 > m3 for m3, m4 in zip(m3_cv, m4_cv))}/5 folds")


# ════════════════════════════════════════════════════════════════
# M4/M5 HISTORICAL FEATURE EXPANSION
# M1 = scheduler only
# M3 = compact historical cross-layer proxies
# M4 = M3 + expanded historical GPU/power/phase/imbalance features
# M5 = M4 + expanded historical Darshan I/O features
#
# Leakage-safe: all hist_* features are computed only from prior jobs where
# prior END_TIMESTAMP < current QUEUED_TIMESTAMP.
# ════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("M4/M5 HISTORICAL FEATURE EXPANSION")
print("=" * 60)

def unique_keep_order(items):
    seen = set()
    out = []
    for x in items:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


# ----------------------------
# Compact historical features: current M3 design
# ----------------------------
compact_hist_cols = [
    "util_mean",
    "idle_frac",
    "zero_util_frac",
    "power_efficiency",
    "io_time_frac",
    "bytes_per_gpu_hour",
]


# ----------------------------
# M4: expanded historical GPU / power / thermal / phase / imbalance behavior
# ----------------------------
expanded_gpu_hist_cols = [
    # Utilization distribution
    "util_mean", "util_max", "util_std",
    "util_p25", "util_p50", "util_p75", "util_p95",
    "zero_util_frac", "idle_frac", "active_phase_frac",
    "max_consecutive_idle_readings",

    # Memory behavior
    "mem_util_mean", "mem_util_max",
    "mem_pressure_frac", "mem_bound_frac",
    "gpu_mem_alloc_mean_kb", "gpu_mem_alloc_max_kb", "gpu_mem_alloc_std_kb",

    # Power behavior
    "power_mean", "power_max", "power_std", "power_p95",
    "power_efficiency", "high_power_low_util_frac",
    "power_cap_proximity_mean", "near_power_cap_frac",

    # Thermal behavior
    "temp_mean", "temp_max", "temp_p95",
    "temp_headroom_mean",
    "thermal_throttle_frac", "sustained_throttle_frac",

    # Cross-signal GPU behavior
    "high_util_low_mem_frac",
    "high_mem_low_util_frac",

    # Temporal phase behavior
    "util_warmup_mean", "util_cooldown_mean",
    "util_first_half", "util_second_half",
    "phase_drop",
    "util_phase1", "util_phase2", "util_phase3",

    # GPU / node imbalance
    "gpu_imbalance_mean", "gpu_imbalance_max",
    "gpu_mem_imbalance_mean",
    "gpu_power_imbalance_mean",
    "gpu_temp_imbalance_mean",
    "node_util_imbalance_std",
    "node_util_imbalance_max",
    "node_count_observed",

    # Telemetry quality
    "telemetry_coverage_frac",
    "telemetry_gap_detected",
]


# ----------------------------
# M5: expanded historical Darshan I/O behavior
# ----------------------------
expanded_io_hist_cols = [
    # Basic I/O volume and intensity
    "io_time_frac",
    "bytes_per_gpu_hour",
    "BWio_MB",
    "total_bytes",
    "bytes_read",
    "bytes_written",
    "io_density",

    # POSIX behavior
    "posix_reads",
    "posix_writes",
    "posix_opens",
    "posix_stats",

    # MPI-IO behavior
    "mpiio_bytes_read",
    "mpiio_bytes_written",
    "mpiio_coll_reads",
    "mpiio_coll_writes",
    "mpiio_indep_reads",
    "mpiio_indep_writes",
    "mpiio_coll_ratio",

    # STDIO behavior
    "stdio_bytes_read",
    "stdio_bytes_written",

    # Access pattern behavior
    "seq_read_ratio",
    "seq_write_ratio",
    "small_read_ratio",
    "large_read_ratio",
    "write_dominance",

    # Metadata / file behavior
    "metadata_ops_per_gb",
    "unique_files",
    "has_posix",
    "has_mpiio",
    "has_heatmap",
    "cb_nodes",

    # Rank imbalance / timing
    "rank_imbalance",
    "rank_time_imbalance",
    "rank_time_gap",
    "variance_rank_time",
    "slowest_rank_time",
    "fastest_rank_time",

    # I/O phase behavior
    "io_phase_start_frac",
    "io_phase_end_frac",
    "io_read_front_heavy",
    "io_write_back_heavy",

    # Alignment behavior
    "mem_not_aligned_ratio",
    "file_not_aligned_ratio",
]


# Keep only columns that exist in the current dataframe.
available_compact_hist = [c for c in compact_hist_cols if c in train_df.columns]
available_gpu_hist = [c for c in expanded_gpu_hist_cols if c in train_df.columns]
available_io_hist = [c for c in expanded_io_hist_cols if c in train_df.columns]

# M3 = compact history
m3_hist_cols = unique_keep_order(available_compact_hist)

# M4 = compact history + expanded GPU history
m4_hist_cols = unique_keep_order(available_compact_hist + available_gpu_hist)

# M5 = M4 + expanded I/O history
m5_hist_cols = unique_keep_order(m4_hist_cols + available_io_hist)

all_hist_cols_for_build = unique_keep_order(m5_hist_cols)

print(f"Available compact historical columns : {len(available_compact_hist)}")
print(f"Available expanded GPU columns       : {len(available_gpu_hist)}")
print(f"Available expanded I/O columns       : {len(available_io_hist)}")
print(f"Total M5 historical columns          : {len(m5_hist_cols)}")


def build_hist_expanded(train_df, target_jobs, cols, lookback_days=7, n_history=10):
    """
    Leakage-safe historical feature builder.

    For each target job, use only prior jobs from the same user where:
        prior END_TIMESTAMP < current QUEUED_TIMESTAMP
        prior END_TIMESTAMP >= current QUEUED_TIMESTAMP - lookback_days

    Then aggregate the most recent n_history prior jobs.
    """
    rows = []
    target_set = set(target_jobs)
    lb_ns = np.timedelta64(lookback_days, "D")

    cols = [c for c in cols if c in train_df.columns]

    for user, grp in train_df.groupby("USERNAME_GENID", sort=False):
        grp = grp.sort_values("QUEUED_TIMESTAMP").reset_index(drop=True)

        qts = grp["QUEUED_TIMESTAMP"].values
        ets = grp["END_TIMESTAMP"].values
        job_names = grp["JOB_NAME"].values

        # Convert historical telemetry features to numeric once per user group.
        hist_mat = grp[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)

        target_indices = [i for i, name in enumerate(job_names) if name in target_set]

        for i in target_indices:
            mask = (ets < qts[i]) & (ets >= qts[i] - lb_ns)
            past_idx = np.where(mask)[0][-n_history:]

            row = {"JOB_NAME": job_names[i]}

            if len(past_idx) > 0:
                p = grp.iloc[past_idx]

                row.update({
                    "user_job_count": int(mask.sum()),
                    "user_mean_runtime": p["RUNTIME_SECONDS"].mean(),
                    "user_walltime_efficiency": (
                        p["RUNTIME_SECONDS"] /
                        p["WALLTIME_SECONDS"].replace(0, np.nan)
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


# Build a fresh trainable telemetry frame so M1/M3/M4/M5 are compared fairly.
telem_m = train_df[
    train_df["crosslayer_tier"].isin(TELEM_TIERS)
].copy().reset_index(drop=True)

split_m = int(len(telem_m) * 0.80)

# Recreate scheduler-visible encodings using train prefix only.
queue_freq_m = telem_m["QUEUE_NAME"].iloc[:split_m].value_counts()
exe_freq_m = telem_m["executable"].iloc[:split_m].value_counts()

telem_m["queue_freq"] = telem_m["QUEUE_NAME"].map(queue_freq_m).fillna(0)
telem_m["executable_freq"] = telem_m["executable"].map(exe_freq_m).fillna(0)

le_m = LabelEncoder()
le_m.fit(telem_m["SCIENCE_FIELD"].iloc[:split_m].astype(str))
known_m = set(le_m.classes_)

telem_m["SCIENCE_FIELD_enc"] = telem_m["SCIENCE_FIELD"].astype(str).apply(
    lambda x: le_m.transform([x])[0] if x in known_m else -1
)

groupA_m = [
    "NODES_REQUESTED",
    "WALLTIME_SECONDS",
    "CORES_REQUESTED",
    "submit_hour",
    "submit_dow",
    "submit_month",
    "queue_freq",
    "SCIENCE_FIELD_enc",
    "executable_freq",
]

groupB_m = [
    "user_job_count",
    "user_mean_runtime",
    "user_walltime_efficiency",
    "user_fail_rate",
    "user_quick_cancel_rate",
    "user_mean_nodes",
    "user_mean_walltime",
]

print(
    f"Building M5 historical features "
    f"({LOOKBACK_DAYS}-day lookback, last {N_HIST} jobs)..."
)
t0_m5 = time.time()

hist_m = build_hist_expanded(
    train_df=train_df,
    target_jobs=telem_m["JOB_NAME"].tolist(),
    cols=all_hist_cols_for_build,
    lookback_days=LOOKBACK_DAYS,
    n_history=N_HIST,
)

telem_m = telem_m.merge(hist_m, on="JOB_NAME", how="left")

print(f"  {len(hist_m):,} rows | {time.time() - t0_m5:.1f}s elapsed")

m3_features = groupA_m + groupB_m + [f"hist_{c}" for c in m3_hist_cols if f"hist_{c}" in telem_m.columns]
m4_features = groupA_m + groupB_m + [f"hist_{c}" for c in m4_hist_cols if f"hist_{c}" in telem_m.columns]
m5_features = groupA_m + groupB_m + [f"hist_{c}" for c in m5_hist_cols if f"hist_{c}" in telem_m.columns]

y_train_m = telem_m["is_wasteful"].iloc[:split_m].values
y_test_m = telem_m["is_wasteful"].iloc[split_m:].values


def to_X_m(df, feats):
    avail = [f for f in feats if f in df.columns]
    X = df[avail].apply(pd.to_numeric, errors="coerce").values
    X = np.nan_to_num(X, nan=-1, posinf=1e9, neginf=-1e9)
    return X, avail


def fit_eval_m(feats, name):
    X_tr, avail = to_X_m(telem_m.iloc[:split_m], feats)
    X_te, _ = to_X_m(telem_m.iloc[split_m:], avail)

    clf = RandomForestClassifier(
        n_estimators=200,
        n_jobs=-1,
        random_state=RNG,
    )

    clf.fit(X_tr, y_train_m)

    prob = clf.predict_proba(X_te)[:, 1]
    pred = clf.predict(X_te)

    return {
        "name": name,
        "clf": clf,
        "features": avail,
        "prob": prob,
        "pred": pred,
        "auc": roc_auc_score(y_test_m, prob),
        "f1": f1_score(y_test_m, pred, average="macro"),
    }


M1_m = fit_eval_m(groupA_m, "M1 scheduler only")
M3_m = fit_eval_m(m3_features, "M3 compact history")
M4_m = fit_eval_m(m4_features, "M4 + expanded GPU history")
M5_m = fit_eval_m(m5_features, "M5 + expanded GPU/I/O history")

print("\n[M1/M3/M4/M5 comparison]")
print(f"{'Model':<32} {'#feat':>7} {'AUC':>8} {'Macro-F1':>10}")
print("-" * 65)

for r in [M1_m, M3_m, M4_m, M5_m]:
    print(f"{r['name']:<32} {len(r['features']):>7} {r['auc']:>8.4f} {r['f1']:>10.4f}")

print(f"\nM3 lift over M1: +{M3_m['auc'] - M1_m['auc']:.4f} AUC")
print(f"M4 lift over M3: +{M4_m['auc'] - M3_m['auc']:.4f} AUC")
print(f"M5 lift over M4: +{M5_m['auc'] - M4_m['auc']:.4f} AUC")
print(f"M5 lift over M3: +{M5_m['auc'] - M3_m['auc']:.4f} AUC")


# ----------------------------
# Feature importance summary for M5
# ----------------------------
importances_m5 = pd.Series(
    M5_m["clf"].feature_importances_,
    index=M5_m["features"]
).sort_values(ascending=False)

print("\n[M5 top 30 feature importances]")
for feat, val in importances_m5.head(30).items():
    print(f"  {feat:<45} {val:.5f}")


def feature_family(feat):
    if feat in groupA_m:
        return "scheduler_current"
    if feat in groupB_m:
        return "user_history_scheduler"
    if feat.startswith("hist_"):
        raw = feat.replace("hist_", "", 1)
        if raw in available_io_hist:
            return "historical_io"
        if raw in available_gpu_hist:
            return "historical_gpu_power"
        return "historical_other"
    return "other"


family_importance = (
    importances_m5
    .groupby(importances_m5.index.map(feature_family))
    .sum()
    .sort_values(ascending=False)
)

print("\n[M5 feature importance by family]")
for fam, val in family_importance.items():
    print(f"  {fam:<28} {val:.5f}")


# ----------------------------
# Temporal CV comparison
# ----------------------------
print("\n[5-fold temporal CV: M3 vs M4 vs M5]")

X_m3_all, _ = to_X_m(telem_m, m3_features)
X_m4_all, _ = to_X_m(telem_m, m4_features)
X_m5_all, _ = to_X_m(telem_m, m5_features)
y_all_m = telem_m["is_wasteful"].values

m3_cv, m4_cv, m5_cv = [], [], []

for tr, te in TimeSeriesSplit(n_splits=5).split(X_m5_all):
    rf3 = RandomForestClassifier(
        n_estimators=200, n_jobs=-1, random_state=RNG
    ).fit(X_m3_all[tr], y_all_m[tr])

    rf4 = RandomForestClassifier(
        n_estimators=200, n_jobs=-1, random_state=RNG
    ).fit(X_m4_all[tr], y_all_m[tr])

    rf5 = RandomForestClassifier(
        n_estimators=200, n_jobs=-1, random_state=RNG
    ).fit(X_m5_all[tr], y_all_m[tr])

    m3_cv.append(roc_auc_score(y_all_m[te], rf3.predict_proba(X_m3_all[te])[:, 1]))
    m4_cv.append(roc_auc_score(y_all_m[te], rf4.predict_proba(X_m4_all[te])[:, 1]))
    m5_cv.append(roc_auc_score(y_all_m[te], rf5.predict_proba(X_m5_all[te])[:, 1]))

print(f"  M3 compact         : {np.mean(m3_cv):.4f} ± {np.std(m3_cv):.4f}")
print(f"  M4 expanded GPU    : {np.mean(m4_cv):.4f} ± {np.std(m4_cv):.4f}")
print(f"  M5 expanded GPU/IO : {np.mean(m5_cv):.4f} ± {np.std(m5_cv):.4f}")
print(f"  M4 > M3 in {sum(m4 > m3 for m3, m4 in zip(m3_cv, m4_cv))}/5 folds")
print(f"  M5 > M4 in {sum(m5 > m4 for m4, m5 in zip(m4_cv, m5_cv))}/5 folds")
print(f"  M5 > M3 in {sum(m5 > m3 for m3, m5 in zip(m3_cv, m5_cv))}/5 folds")




# ════════════════════════════════════════════════════════════════
# PAPER TABLE AND NUMBER GENERATOR
# Paste at end of framework.py — runs after M3 is trained.
# Generates every table and claimed number in the paper.
# ════════════════════════════════════════════════════════════════
import textwrap

print("\n" + "=" * 70)
print("PAPER TABLE AND NUMBER GENERATOR")
print("=" * 70)

# ── Helpers ───────────────────────────────────────────────────
def box(title):
    print(f"\n{'─'*70}")
    print(f"  {title}")
    print(f"{'─'*70}")

def check(label, computed, claimed, tol=0.005):
    ok = abs(computed - claimed) <= tol
    flag = "✓" if ok else f"✗  CLAIMED={claimed}"
    print(f"  {label:<55} {computed:>10.4f}  {flag}")

# ── Reload combined if needed ──────────────────────────────────
if "combined" not in dir():
    combined = pd.read_csv(
        "/pscratch/sd/h/hjajula/Darshan/cross_layer_hpc_tool/data/combined_metrics_final.csv",
        low_memory=False)
    combined["QUEUED_TIMESTAMP"] = pd.to_datetime(
        combined["QUEUED_TIMESTAMP"], errors="coerce")

total_jobs    = len(combined)
total_gpu_hrs = combined["gpu_hours"].sum()

# ════════════════════════════════════════════════════════════════
# TABLE I — Full 16-tier taxonomy
# ════════════════════════════════════════════════════════════════
box("TABLE I — Cross-layer workload taxonomy (all 16 tiers)")

TIER_GROUPS = {
    "Wasteful": [
        "Quick_Cancel", "Failed_Job", "GPU_Idle_Timeout",
        "Ghost", "Scale_Waster", "IO_Bottlenecked",
    ],
    "Productive": [
        "Ideal_Compute_With_IO", "Compute_Bound",
    ],
    "Informational": [
        "Moderate_Compute_No_IO", "Moderate_Compute_With_IO",
        "Incidental_IO_Low_GPU", "Low_Efficiency",
    ],
    "Excluded from ML (no DCGM)": [
        "No_GPU_Telemetry", "No_GPU_With_Darshan",
        "Short_No_GPU", "Short_No_GPU_With_IO",
    ],
}

print(f"\n  {'Tier':<32} {'Jobs':>8} {'GPU-hrs':>12} {'%GPU-hrs':>9}")
print(f"  {'─'*64}")
grand_jobs, grand_gpu = 0, 0
for group, tiers in TIER_GROUPS.items():
    print(f"\n  [{group}]")
    g_jobs, g_gpu = 0, 0
    for t in tiers:
        sub = combined[combined["crosslayer_tier"] == t]
        n = len(sub); g = sub["gpu_hours"].sum()
        g_jobs += n; g_gpu += g; grand_jobs += n; grand_gpu += g
        print(f"  {t:<32} {n:>8,} {g:>12,.0f} {g/total_gpu_hrs*100:>8.1f}%")
    print(f"  {'Subtotal':<32} {g_jobs:>8,} {g_gpu:>12,.0f} {g_gpu/total_gpu_hrs*100:>8.1f}%")

print(f"\n  {'─'*64}")
print(f"  {'TOTAL (all tiers)':<32} {grand_jobs:>8,} {grand_gpu:>12,.0f} {grand_gpu/total_gpu_hrs*100:>8.1f}%")
print(f"\n  Corpus total: {total_jobs:,} jobs, {total_gpu_hrs:,.0f} GPU-hrs ({total_gpu_hrs/1e6:.2f}M)")

# Reconciliation
tier_sum = combined["crosslayer_tier"].value_counts().sum()
gpu_sum  = combined.groupby("crosslayer_tier")["gpu_hours"].sum().sum()
print(f"\n[Reconciliation]")
print(f"  Tier counts sum to corpus total: "
      f"{tier_sum:,} == {total_jobs:,}  {'✓' if tier_sum==total_jobs else '✗'}")
print(f"  GPU-hrs sum to corpus total:     "
      f"{gpu_sum:,.0f} == {total_gpu_hrs:,.0f}  "
      f"{'✓' if abs(gpu_sum-total_gpu_hrs)<1 else '✗'}")

# Key claims
wasteful_tiers = ["Quick_Cancel","Failed_Job","GPU_Idle_Timeout",
                  "Ghost","Scale_Waster","IO_Bottlenecked"]
wasteful_gpu = combined.loc[combined["crosslayer_tier"].isin(wasteful_tiers),
                            "gpu_hours"].sum()
print(f"\n[Key claims]")
print(f"  Wasteful GPU-hrs: {wasteful_gpu:,.0f} ({wasteful_gpu/total_gpu_hrs*100:.1f}%)")
check("  -> claimed 63.5%", wasteful_gpu/total_gpu_hrs, 0.635, tol=0.005)

failed_git_gpu = combined.loc[
    combined["crosslayer_tier"].isin(["Failed_Job","GPU_Idle_Timeout"]),
    "gpu_hours"].sum()
print(f"  Failed+GIT GPU-hrs: {failed_git_gpu:,.0f} "
      f"({failed_git_gpu/total_gpu_hrs*100:.1f}%)")
check("  -> claimed 56.9%", failed_git_gpu/total_gpu_hrs, 0.569, tol=0.005)

qc_pct_jobs = (combined["crosslayer_tier"]=="Quick_Cancel").mean()
print(f"  Quick_Cancel share: {qc_pct_jobs*100:.1f}% of submissions")
check("  -> claimed 36.0%", qc_pct_jobs, 0.360, tol=0.005)

# ════════════════════════════════════════════════════════════════
# CORPUS COVERAGE NUMBERS
# ════════════════════════════════════════════════════════════════
box("CORPUS COVERAGE NUMBERS")

n_dcgm   = combined["has_gpu"].sum()
n_dar    = combined["darshan_present"].sum()
n_dar_io = combined["io_detected"].sum()
n_both   = (combined["has_gpu"] & combined["darshan_present"]).sum()

for label, n, claimed in [
    ("DCGM coverage",              n_dcgm,   0.781),
    ("Darshan attached",           n_dar,    0.252),
    ("Darshan I/O detected",       n_dar_io, 0.072),
    ("Both DCGM + Darshan",        n_both,   0.184),
]:
    pct = n / total_jobs
    print(f"  {label:<30} {n:>8,}  ({pct*100:5.1f}%)", end="")
    ok = abs(pct - claimed) <= 0.002
    print(f"  {'✓' if ok else f'✗ claimed={claimed*100:.1f}%'}")

# ════════════════════════════════════════════════════════════════
# IO_BOTTLENECKED SUB-TIERS
# ════════════════════════════════════════════════════════════════
box("IO_BOTTLENECKED SUB-TIERS")

io_b = combined[combined["crosslayer_tier"] == "IO_Bottlenecked"].copy()
print(f"  IO_Bottlenecked total: {len(io_b):,}")

m_meta  = io_b["metadata_ops_per_gb"].fillna(0) > 1000
m_bw    = io_b["BWio_MB"].fillna(10000) < 1000
m_imb   = io_b["rank_time_imbalance"].fillna(0) > 2.0

n_meta  = m_meta.sum()
n_bw    = (m_bw & ~m_meta).sum()
n_imb   = (m_imb & ~m_meta & ~m_bw).sum()
n_other = len(io_b) - n_meta - n_bw - n_imb

for label, n, claimed in [
    ("Bandwidth-bound  (<1000 MB/s)",    n_bw,    0.518),
    ("Metadata-bound   (>1000 ops/GB)",  n_meta,  0.430),
    ("Rank-imbalance   (>2x)",           n_imb,   0.012),
    ("Other",                            n_other, 0.040),
]:
    pct = n / len(io_b)
    print(f"  {label:<38} {n:>6,}  ({pct*100:5.1f}%)", end="")
    check("", pct, claimed, tol=0.005)

# ════════════════════════════════════════════════════════════════
# SCALE_WASTER BY ALLOCATION
# ════════════════════════════════════════════════════════════════
box("SCALE_WASTER BY ALLOCATION SIZE")

sw = combined[combined["crosslayer_tier"] == "Scale_Waster"].copy()
print(f"  Scale_Waster total: {len(sw):,}")
for label, mask, j_claimed, g_claimed in [
    ("Medium  (2-7 nodes,  8-31 GPUs)",   (sw["gpus"]>=8)   & (sw["gpus"]<32),   241, 1643),
    ("Large   (8-31 nodes, 32-127 GPUs)", (sw["gpus"]>=32)  & (sw["gpus"]<128),  671, 24287),
    ("Extreme (32+ nodes,  128+ GPUs)",    sw["gpus"]>=128,                        83, 82248),
]:
    n = mask.sum(); g = sw.loc[mask, "gpu_hours"].sum()
    ok_j = abs(n - j_claimed) <= 2
    ok_g = abs(g - g_claimed) / max(g_claimed, 1) <= 0.01
    print(f"  {label:<38} {n:>4,} jobs  {g:>8,.0f} GPU-hrs  "
          f"{'✓' if ok_j and ok_g else f'✗ claimed={j_claimed}/{g_claimed}'}")

extreme = sw[sw["gpus"] >= 128]
extreme_pct = extreme["gpu_hours"].sum() / sw["gpu_hours"].sum()
print(f"\n  Extreme-scale share of Scale_Waster GPU-hrs: {extreme_pct*100:.0f}%")
check("  -> claimed 76%", extreme_pct, 0.76, tol=0.03)

# ════════════════════════════════════════════════════════════════
# TABLE II — GPU PHASE TRAJECTORY
# ════════════════════════════════════════════════════════════════
box("TABLE II — GPU phase trajectory by tier")

phase_cols = ["util_phase1","util_phase2","util_phase3","io_phase_end_frac"]
phase_df = combined.dropna(subset=["util_phase1","util_phase2","util_phase3"])

print(f"\n  {'Tier':<26} {'n':>7} {'P1':>6} {'P2':>6} {'P3':>6} {'io_end':>8}")
print(f"  {'─'*62}")

paper_vals = {
    "Ghost":          (0.3, 0.3, 0.4, 0.000),
    "IO_Bottlenecked":(0.4, 0.7, 0.7, 0.993),
    "Scale_Waster":   (6.5, 7.6, 7.6, 0.000),
    "Compute_Bound":  (83.0,90.9,87.1,0.000),
}
for tier in ["Ghost","IO_Bottlenecked","Scale_Waster","Compute_Bound"]:
    sub = phase_df[phase_df["crosslayer_tier"]==tier]
    if len(sub) < 30: continue
    p1 = sub["util_phase1"].mean()
    p2 = sub["util_phase2"].mean()
    p3 = sub["util_phase3"].mean()
    ie = sub["io_phase_end_frac"].median()
    pv = paper_vals[tier]
    ok = all(abs(c-p)<=0.15 for c,p in zip([p1,p2,p3,ie],pv))
    flag = "✓" if ok else f"✗ claimed={pv}"
    print(f"  {tier:<26} {len(sub):>7,} {p1:>6.1f} {p2:>6.1f} {p3:>6.1f} {ie:>8.3f}  {flag}")

# ════════════════════════════════════════════════════════════════
# SIGNAL CORRELATION
# ════════════════════════════════════════════════════════════════
box("SIGNAL CORRELATION")

from scipy.stats import spearmanr
sig_df = combined[["util_mean","io_time_frac","BWio_MB"]].dropna()
r_io, _ = spearmanr(sig_df["util_mean"], sig_df["io_time_frac"])
r_bw, _ = spearmanr(sig_df["util_mean"], sig_df["BWio_MB"])
print(f"  Spearman r(GPU_util, io_time_frac) = {r_io:.3f}  (N={len(sig_df):,})")
check("  -> claimed -0.411", r_io, -0.411, tol=0.005)
print(f"  Spearman r(GPU_util, BWio_MB)      = {r_bw:.3f}")
check("  -> claimed -0.272", r_bw, -0.272, tol=0.005)

# ════════════════════════════════════════════════════════════════
# GHOST POWER GAP
# ════════════════════════════════════════════════════════════════
box("GHOST POWER GAP")

ghost = combined[combined["crosslayer_tier"]=="Ghost"]
pwr   = ghost["power_mean"].notna()
n_elevated = (ghost.loc[pwr,"power_mean"] > 50).sum()
n_pwr_total = pwr.sum()
print(f"  Ghost total:              {len(ghost):,}")
print(f"  With power data:          {n_pwr_total:,}")
print(f"  Power > 50W AND util < 5%:{n_elevated:,} ({n_elevated/n_pwr_total*100:.1f}%)")
check("  -> claimed 69.2%", n_elevated/n_pwr_total, 0.692, tol=0.005)
check("  -> claimed n=22,111", n_elevated, 22111, tol=50)

# ════════════════════════════════════════════════════════════════
# USER CONCENTRATION
# ════════════════════════════════════════════════════════════════
box("USER CONCENTRATION")

n_users    = combined["USERNAME_GENID"].nunique()
user_gpu   = combined.groupby("USERNAME_GENID")["gpu_hours"].sum().sort_values(ascending=False)
top10_gpu  = user_gpu.head(10).sum()
top10_jobs = combined.groupby("USERNAME_GENID").size().sort_values(ascending=False).head(10).sum()

print(f"  Unique users: {n_users:,}",flush=True)
check("  -> claimed 1,008", n_users, 1008, tol=1)
print(f"  Top-10 GPU-hrs share: {top10_gpu/total_gpu_hrs*100:.1f}%")
check("  -> claimed 45.3%", top10_gpu/total_gpu_hrs, 0.453, tol=0.005)
print(f"  Top-10 jobs share:    {top10_jobs/total_jobs*100:.1f}%")
check("  -> claimed 35.4%", top10_jobs/total_jobs, 0.354, tol=0.005)

for p, claimed_jobs, claimed_gpu in [
    (0.05, 0.605, 0.812),
    (0.10, 0.723, 0.909),
]:
    k = max(1, int(n_users * p))
    cj = combined.groupby("USERNAME_GENID").size().sort_values(ascending=False).head(k).sum() / total_jobs
    cg = user_gpu.head(k).sum() / total_gpu_hrs
    print(f"  Top-{int(p*100):>2}% users: {cj*100:.1f}% jobs, {cg*100:.1f}% GPU-hrs")
    check(f"    jobs -> claimed {claimed_jobs*100:.1f}%", cj, claimed_jobs, tol=0.005)
    check(f"    GPU  -> claimed {claimed_gpu*100:.1f}%", cg, claimed_gpu, tol=0.005)

# ════════════════════════════════════════════════════════════════
# WASTE PERSISTENCE
# ════════════════════════════════════════════════════════════════
box("WASTE PERSISTENCE")

cs = combined.sort_values(["USERNAME_GENID","QUEUED_TIMESTAMP"]).reset_index(drop=True)
cs["prev_w"] = cs.groupby("USERNAME_GENID")["is_wasteful"].shift(1).fillna(0)
cs["in_burst"] = ((cs["is_wasteful"]==1) & (cs["prev_w"]==1)).astype(int)

SUBST = {"Ghost","IO_Bottlenecked","Scale_Waster","Failed_Job","GPU_Idle_Timeout"}
sub_w = cs[cs["crosslayer_tier"].isin(SUBST)]
sub_b = cs[cs["crosslayer_tier"].isin(SUBST) & (cs["in_burst"]==1)]
all_w = cs[cs["is_wasteful"]==1]
all_b = cs[cs["in_burst"]==1]

burst_subst = len(sub_b)/len(sub_w)
burst_all   = len(all_b)/len(all_w)
print(f"  Substantive tiers in burst: {len(sub_b):,}/{len(sub_w):,} ({burst_subst*100:.1f}%)")
check("  -> claimed 92.9%", burst_subst, 0.929, tol=0.005)
print(f"  Including Quick_Cancel:     {len(all_b):,}/{len(all_w):,} ({burst_all*100:.1f}%)")
check("  -> claimed 93.8%", burst_all, 0.938, tol=0.005)

# hist_idle_frac correlation with burst
# (requires hist_idle_frac in combined — populated by build_hist in framework.py)
if "hist_idle_frac" in telem_df.columns:
    tm = telem_df[["JOB_NAME","hist_idle_frac","is_wasteful"]].merge(
        cs[["JOB_NAME","in_burst"]], on="JOB_NAME", how="left").fillna(0)
    valid = tm[["hist_idle_frac","in_burst"]].dropna()
    r_burst, _ = spearmanr(valid["hist_idle_frac"], valid["in_burst"])
    burst_hi = (tm[tm["in_burst"]==1]["hist_idle_frac"].fillna(0) > 0.5).mean()*100
    print(f"\n  Spearman r(hist_idle_frac, in_burst) = {r_burst:.3f}")
    check("  -> claimed 0.617", r_burst, 0.617, tol=0.005)
    print(f"  In-burst jobs with hist_idle_frac>0.5: {burst_hi:.1f}%")
    check("  -> claimed 91.4%", burst_hi/100, 0.914, tol=0.005)
else:
    print("  (hist_idle_frac not in combined — run build_hist first)")

# ════════════════════════════════════════════════════════════════
# FAILURE TAXONOMY VALIDATION
# ════════════════════════════════════════════════════════════════
box("FAILURE TAXONOMY VALIDATION")

neg29 = combined[combined["EXIT_STATUS"] == -29].copy()
neg29["wt_frac"] = neg29["RUNTIME_SECONDS"] / neg29["WALLTIME_SECONDS"].replace(0, np.nan)
n_git_classified = (neg29["crosslayer_tier"] == "GPU_Idle_Timeout").sum()
n_neg29_total    = len(neg29)
n_productive_wt  = n_neg29_total - n_git_classified

print(f"  exit=-29 total:             {n_neg29_total:,}")
print(f"  -> classified GPU_Idle_Timeout: {n_git_classified:,} "
      f"({n_git_classified/n_neg29_total*100:.1f}%)")
check("  -> claimed 52.3%", n_git_classified/n_neg29_total, 0.523, tol=0.005)
print(f"  -> productive walltime-exhausted: {n_productive_wt:,} "
      f"({n_productive_wt/n_neg29_total*100:.1f}%)")
check("  -> claimed 47.7%", n_productive_wt/n_neg29_total, 0.477, tol=0.005)

# ════════════════════════════════════════════════════════════════
# TABLE III — MODEL COMPARISON (multi-seed, from section 19)
# ════════════════════════════════════════════════════════════════
box("TABLE III — Model comparison (requires telem_df + M1/M3 to be in scope)")

if "M1" in dir() and "M3" in dir() and "y_test" in dir():
    from sklearn.dummy import DummyClassifier
    from sklearn.linear_model import LogisticRegression

    dst = DummyClassifier(strategy="stratified", random_state=RNG).fit(
        np.zeros((split_t,1)), y_train)
    auc_st = roc_auc_score(y_test, dst.predict_proba(np.zeros((len(y_test),1)))[:,1])

    lr_wt = LogisticRegression(max_iter=1000, class_weight="balanced").fit(
        telem_df[["WALLTIME_SECONDS"]].iloc[:split_t].values, y_train)
    auc_wt = roc_auc_score(y_test,
        lr_wt.predict_proba(telem_df[["WALLTIME_SECONDS"]].iloc[split_t:].values)[:,1])

    print(f"\n  {'Model':<35} {'AUC':>8} {'F1':>8}")
    print(f"  {'─'*54}")
    for label, auc, f1 in [
        ("Stratified random",       auc_st,    None),
        ("Walltime-only LR",        auc_wt,    None),
        ("M1 (scheduler, 9 feats)", M1["auc"], M1["f1"]),
        ("C only (13 feats)",       ab_C["auc"], None),
        ("M3 (cross-layer, 22)",    M3["auc"], M3["f1"]),
    ]:
        f1s = f"{f1:.3f}" if f1 else "---"
        print(f"  {label:<35} {auc:>8.4f} {f1s:>8}")

    # Multi-seed numbers (from section 19)
    if "m1_aucs" in dir() and "m3_aucs" in dir():
        print(f"\n  Multi-seed (5 seeds):")
        print(f"  M1: {np.mean(m1_aucs):.4f} ± {np.std(m1_aucs):.4f}")
        print(f"  M3: {np.mean(m3_aucs):.4f} ± {np.std(m3_aucs):.4f}")
        lift = np.array(m3_aucs) - np.array(m1_aucs)
        print(f"  Lift: {lift.mean():+.4f} ± {lift.std():.4f}")
        check("  M1 AUC -> claimed 0.726", np.mean(m1_aucs), 0.726, tol=0.005)
        check("  M3 AUC -> claimed 0.919", np.mean(m3_aucs), 0.919, tol=0.005)
        check("  Lift   -> claimed 0.193", lift.mean(), 0.193, tol=0.005)
else:
    print("  (M1/M3/y_test not in scope — run after ML section)")
# ════════════════════════════════════════════════════════════════
# OvR PER-TIER NECESSITY: M1 vs M2 vs M3
# ════════════════════════════════════════════════════════════════

try:
    box("OvR PER-TIER AUC: M1 vs M2 vs M3")
except NameError:
    print("\n" + "=" * 60)
    print("OvR PER-TIER AUC: M1 vs M2 vs M3")
    print("=" * 60)

# Internal tier names in the dataframe -> paper-facing names
tier_name = {
    "Ghost": "Idle-like",
    "IO_Bottlenecked": "IO_Bottlenecked",
    "Scale_Waster": "Scale_Inefficient",
    "GPU_Idle_Timeout": "GPU_Idle_Timeout",
}

# Existing paper values for sanity checking M1/M3 only
claimed_ovr = {
    "Ghost":            {"M1": 0.547, "M3": 0.815},
    "IO_Bottlenecked":  {"M1": 0.844, "M3": 0.977},
    "Scale_Waster":     {"M1": 0.732, "M3": 0.906},
    "GPU_Idle_Timeout": {"M1": 0.719, "M3": 0.900},
}

tiers = ["Ghost", "IO_Bottlenecked", "Scale_Waster", "GPU_Idle_Timeout"]

# Feature sets
feature_sets = {
    "M1": groupA,                  # scheduler-only
    "M2": ab_C["avail"],           # history-only / compact historical telemetry
    "M3": M3["avail"],             # scheduler + history
}

ovr_rows = []

print(
    f"\n  {'Tier':<22} {'support':>8} "
    f"{'M1':>8} {'M2':>8} {'M3':>8} {'Δ M3-M1':>10} {'check':>12}"
)
print("  " + "─" * 86)

for tier in tiers:
    y_tr = (telem_df["crosslayer_tier"].iloc[:split_t] == tier).astype(int).values
    y_te = (telem_df["crosslayer_tier"].iloc[split_t:] == tier).astype(int).values
    support = int(y_te.sum())

    if support < 10:
        print(f"  {tier_name[tier]:<22} {support:>8,}  insufficient positives")
        continue

    aucs = {}

    for model_name, feats in feature_sets.items():
        avail = [f for f in feats if f in telem_df.columns]

        rf = RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            n_jobs=-1,
            random_state=RNG
        )

        rf.fit(to_X(telem_df.iloc[:split_t], avail), y_tr)

        prob = rf.predict_proba(
            to_X(telem_df.iloc[split_t:], avail)
        )[:, 1]

        aucs[model_name] = roc_auc_score(y_te, prob)

    delta = aucs["M3"] - aucs["M1"]

    # sanity check against old paper values, within ±0.01
    c_m1 = claimed_ovr[tier]["M1"]
    c_m3 = claimed_ovr[tier]["M3"]
    ok_m1 = abs(aucs["M1"] - c_m1) <= 0.01
    ok_m3 = abs(aucs["M3"] - c_m3) <= 0.01
    check = "ok" if (ok_m1 and ok_m3) else f"was {c_m1:.3f}/{c_m3:.3f}"

    print(
        f"  {tier_name[tier]:<22} {support:>8,} "
        f"{aucs['M1']:>8.4f} {aucs['M2']:>8.4f} {aucs['M3']:>8.4f} "
        f"{delta:>10.4f} {check:>12}"
    )

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
    print(
        rf"\texttt{{{r['paper_tier'].replace('_', r'\_')}}} "
        rf"& {r['M1']:.3f} & {r['M2']:.3f} & {r['M3']:.3f} "
        rf"& ${r['delta']:+.3f}$ \\"
    )
# # ════════════════════════════════════════════════════════════════
# # OVR PER-TIER NECESSITY + M1 vs M2 vs M3
# # ════════════════════════════════════════════════════════════════
# box("OvR PER-TIER AUC: M1 vs M2 vs M3")

# paper_ovr = {
#     "Ghost":            (0.547, 0.815, 0.268),
#     "IO_Bottlenecked":  (0.844, 0.977, 0.133),
#     "Scale_Waster":     (0.732, 0.906, 0.174),
#     "GPU_Idle_Timeout": (0.719, 0.900, 0.181),
# }

# if "telem_df" in dir() and "M3" in dir():
#     print(f"\n  {'Tier':<22} {'M1':>8} {'M3':>8} {'Δ':>8} {'claimed Δ':>10}  verdict")
#     print(f"  {'─'*72}")
#     avail_m3 = M3["avail"]
#     for tier, (cm1, cm3, cdelta) in paper_ovr.items():
#         y_tr = (telem_df["crosslayer_tier"].iloc[:split_t]==tier).astype(int).values
#         y_te = (telem_df["crosslayer_tier"].iloc[split_t:]==tier).astype(int).values
#         if y_te.sum() < 10:
#             print(f"  {tier:<22} (insufficient positives in test set)")
#             continue
#         aucs = {}
#         for gname, feats in [("M1", groupA), ("M3", avail_m3)]:
#             avail = [f for f in feats if f in telem_df.columns]
#             rf = RandomForestClassifier(n_estimators=200, class_weight="balanced",
#                                         n_jobs=-1, random_state=RNG)
#             rf.fit(to_X(telem_df.iloc[:split_t], avail), y_tr)
#             aucs[gname] = roc_auc_score(
#                 y_te, rf.predict_proba(to_X(telem_df.iloc[split_t:], avail))[:,1])
#         delta = aucs["M3"] - aucs["M1"]
#         ok_m1 = abs(aucs["M1"] - cm1) <= 0.01
#         ok_m3 = abs(aucs["M3"] - cm3) <= 0.01
#         ok_d  = abs(delta - cdelta) <= 0.01
#         flag  = "✓" if ok_m1 and ok_m3 and ok_d else f"✗ claimed={cm1:.3f}→{cm3:.3f}"
#         verdict = "essential (Δ>0.10)" if delta > 0.10 else "marginal"
#         print(f"  {tier:<22} {aucs['M1']:>8.4f} {aucs['M3']:>8.4f} "
#               f"{delta:>8.4f} {f'+{cdelta:.3f}':>10}  {flag}  {verdict}")
# else:
#     print("  (telem_df not in scope)")

# ════════════════════════════════════════════════════════════════
# COLD-START STRATIFIED AUC (NEW — addresses Reviewer 2)
# ════════════════════════════════════════════════════════════════
box("COLD-START STRATIFIED AUC")

if "telem_df" in dir() and "M3" in dir():
    test_df = telem_df.iloc[split_t:].copy().reset_index(drop=True)
    test_df["pred_proba"] = M3["prob"]

    print(f"\n  {'User history bucket':<28} {'n':>7} {'pos%':>6} {'AUC':>8}")
    print(f"  {'─'*55}")
    buckets = [
        (-0.5,  0.5, "0 prior jobs (cold start)"),
        ( 0.5,  3.5, "1-3 prior jobs"),
        ( 3.5,  9.5, "4-9 prior jobs"),
        ( 9.5, 999,  "10 prior jobs (max history)"),
    ]
    cold_start_auc = None
    for lo, hi, label in buckets:
        sub = test_df[(test_df["user_job_count"]>lo) & (test_df["user_job_count"]<=hi)]
        if len(sub) < 50 or sub["is_wasteful"].nunique() < 2:
            print(f"  {label:<28} {len(sub):>7,} (insufficient)")
            continue
        auc = roc_auc_score(sub["is_wasteful"], sub["pred_proba"])
        pos = sub["is_wasteful"].mean()*100
        print(f"  {label:<28} {len(sub):>7,} {pos:>5.1f}% {auc:>8.4f}")
        if "cold" in label:
            cold_start_auc = auc
    if cold_start_auc:
        print(f"\n  Cold-start AUC ({cold_start_auc:.4f}) vs M1 ({M1['auc']:.4f}) vs "
              f"M3 full ({M3['auc']:.4f})")
        print(f"  -> Suggests model relies on history: "
              f"{'YES' if cold_start_auc < M1['auc']+0.05 else 'partially'}")
else:
    print("  (requires telem_df + M3 in scope)")

# ════════════════════════════════════════════════════════════════
# M2 HIERARCHY (DJC-only baseline)
# ════════════════════════════════════════════════════════════════
box("M2 HIERARCHY: M1 → M2 → M3")

if "M2" in dir():
    for label, auc, claimed in [
        ("M1 scheduler only",    M1["auc"], 0.726),
        ("M2 DJC GPU history",   M2["auc"], 0.769),
        ("M3 cross-layer",       M3["auc"], 0.918),
    ]:
        ok = abs(auc - claimed) <= 0.005
        print(f"  {label:<28} AUC={auc:.4f}  "
              f"{'✓' if ok else f'✗ claimed={claimed}'}")
    print(f"\n  DJC-only lift (M2-M1):  +{M2['auc']-M1['auc']:.4f}")
    print(f"  DCGM marginal (M3-M2):  +{M3['auc']-M2['auc']:.4f}")
    print(f"  Full lift (M3-M1):      +{M3['auc']-M1['auc']:.4f}")
else:
    print("  (M2 not in scope — run M2 block first)")

# ════════════════════════════════════════════════════════════════
# DISCUSSION CONTEXT NUMBERS
# ════════════════════════════════════════════════════════════════
box("DISCUSSION CONTEXT NUMBERS")

has_gpu = combined[combined["has_gpu"]]
zero_util  = (has_gpu["util_mean"].fillna(0) == 0).sum()
half_idle  = (has_gpu["idle_frac"].fillna(0) > 0.5).sum() if "idle_frac" in has_gpu else 0
mean_util  = has_gpu["util_mean"].mean()

print(f"  Zero GPU utilization (DCGM jobs): {zero_util:,} ({zero_util/len(has_gpu)*100:.1f}%)")
check("  -> claimed 54.5%", zero_util/len(has_gpu), 0.545, tol=0.005)
print(f"  Jobs >50% idle fraction:          {half_idle:,} ({half_idle/len(has_gpu)*100:.1f}%)")
check("  -> claimed 74.5%", half_idle/len(has_gpu), 0.745, tol=0.005)
print(f"  Mean GPU utilization (DCGM jobs): {mean_util:.1f}%")
check("  -> claimed 13.3%", mean_util, 13.3, tol=0.3)

# ════════════════════════════════════════════════════════════════
# ML TARGET DEFINITION (for paper §V clarity)
# ════════════════════════════════════════════════════════════════
box("ML TARGET DEFINITION (for §V / Reviewer 2)")

ml_df = combined[combined["use_for_training"]]
print(f"  Total jobs in ML subset (use_for_training=True): {len(ml_df):,}")
print(f"  is_wasteful=1: {ml_df['is_wasteful'].sum():,} ({ml_df['is_wasteful'].mean()*100:.1f}%)")
print(f"  is_wasteful=0: {(ml_df['is_wasteful']==0).sum():,}")
print(f"\n  Tiers included in ML (TELEM_TIERS):")
for t, n in ml_df["crosslayer_tier"].value_counts().items():
    w = "wasteful" if t in WASTEFUL else "not wasteful"
    print(f"    {t:<32} {n:>7,}  ({w})")
print(f"\n  Tiers excluded from ML:")
for t, n in combined[~combined["use_for_training"]]["crosslayer_tier"].value_counts().items():
    print(f"    {t:<32} {n:>7,}")

# ════════════════════════════════════════════════════════════════
# FINAL PAPER NUMBER SUMMARY
# ════════════════════════════════════════════════════════════════
box("FINAL PAPER NUMBER SUMMARY (paste-ready for proofreading)")

print(f"""
  CORPUS
    Jobs total           : {total_jobs:,}
    GPU-hrs total        : {total_gpu_hrs:,.0f} ({total_gpu_hrs/1e6:.2f}M)
    Date span            : 13 months
    Unique users         : {n_users:,}
    Unique projects      : {combined['PROJECT_NAME_GENID'].nunique():,}

  COVERAGE
    DCGM                 : {n_dcgm:,} ({n_dcgm/total_jobs*100:.1f}%)
    Darshan attached     : {n_dar:,} ({n_dar/total_jobs*100:.1f}%)
    Darshan I/O detected : {n_dar_io:,} ({n_dar_io/total_jobs*100:.1f}%)
    Both DCGM+Darshan    : {n_both:,} ({n_both/total_jobs*100:.1f}%)

  TAXONOMY
    Wasteful GPU-hrs     : {wasteful_gpu:,.0f} ({wasteful_gpu/total_gpu_hrs*100:.1f}%)
    Failed+GIT GPU-hrs   : {failed_git_gpu:,.0f} ({failed_git_gpu/total_gpu_hrs*100:.1f}%)

  PREDICTION
    Train jobs           : {split_t:,}
    Test jobs            : {len(telem_df)-split_t:,}
    Train cutoff         : {telem_df['QUEUED_TIMESTAMP'].iloc[split_t].date() if 'telem_df' in dir() else 'N/A'}
""")

print("\n" + "=" * 70)
print("PAPER NUMBER GENERATOR COMPLETE")
print("All ✗ markers indicate mismatches between paper claims and computed values.")
print("=" * 70)



# ════════════════════════════════════════════════════════════════
# MULTI-SEED STABILITY — produces m1_aucs, m3_aucs for paper Table III
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("MULTI-SEED STABILITY")
print("=" * 60)

SEEDS = [42, 7, 1337, 2024, 31415]
m1_aucs, m3_aucs, m1_f1s, m3_f1s = [], [], [], []

_X_tr_m1 = to_X(telem_df.iloc[:split_t], M1["avail"])
_X_te_m1 = to_X(telem_df.iloc[split_t:], M1["avail"])
_X_tr_m3 = to_X(telem_df.iloc[:split_t], M3["avail"])
_X_te_m3 = to_X(telem_df.iloc[split_t:], M3["avail"])

print(f"Running {len(SEEDS)} seeds...")
for s in SEEDS:
    rf1 = RandomForestClassifier(n_estimators=200, n_jobs=-1,
                                  random_state=s).fit(_X_tr_m1, y_train)
    rf3 = RandomForestClassifier(n_estimators=200, n_jobs=-1,
                                  random_state=s).fit(_X_tr_m3, y_train)
    p1 = rf1.predict_proba(_X_te_m1)[:, 1]
    p3 = rf3.predict_proba(_X_te_m3)[:, 1]
    m1_aucs.append(roc_auc_score(y_test, p1))
    m3_aucs.append(roc_auc_score(y_test, p3))
    m1_f1s.append(f1_score(y_test, rf1.predict(_X_te_m1), average="macro"))
    m3_f1s.append(f1_score(y_test, rf3.predict(_X_te_m3), average="macro"))

lifts = np.array(m3_aucs) - np.array(m1_aucs)
print(f"  M1: {np.mean(m1_aucs):.4f} ± {np.std(m1_aucs):.4f}  "
      f"[{min(m1_aucs):.4f}, {max(m1_aucs):.4f}]")
print(f"  M3: {np.mean(m3_aucs):.4f} ± {np.std(m3_aucs):.4f}  "
      f"[{min(m3_aucs):.4f}, {max(m3_aucs):.4f}]")
print(f"  Lift: {lifts.mean():+.4f} ± {lifts.std():.4f}")
print(f"  M1 F1: {np.mean(m1_f1s):.4f} ± {np.std(m1_f1s):.4f}")
print(f"  M3 F1: {np.mean(m3_f1s):.4f} ± {np.std(m3_f1s):.4f}")




# ════════════════════════════════════════════════════════════════
# REVIEWER-PROOFING EXTENSIONS
# Run after the main framework.py pipeline. Produces every number
# needed to address common reviewer objections plus the new
# read/write/burst/time-breakdown analyses from the rerun.
# ════════════════════════════════════════════════════════════════

print("\n" + "=" * 68 )
print("REVIEWER-PROOFING EXTENSIONS")
print("=" * 68)

# ── A. Ghost split: structural implementation (replaces manual approximation)
print("\n" + "=" * 68)
print("A. GHOST / GHOST_HIDDEN_ACTIVITY SPLIT (structural)")
print("=" )

m_ghost_all = combined["crosslayer_tier"] == "Ghost"
n_ghost_total = m_ghost_all.sum()
ghost_total_gpu = combined.loc[m_ghost_all, "gpu_hours"].sum()

# Power split (50W threshold, same as Discussion)
power_avail = combined["power_mean"].notna()
m_pwr_low  = combined["power_mean"].fillna(0) < 50
m_pwr_high = combined["power_mean"].fillna(0) >= 50

# Apply split
combined.loc[m_ghost_all & m_pwr_high, ["crosslayer_tier", "diagnostic_tier"]] = "Ghost_Hidden_Activity"
# Ghost_Hidden_Activity is NOT in WASTEFUL — it's informational
combined["is_wasteful"] = combined["crosslayer_tier"].isin(WASTEFUL).astype(int)

# Recompute headline numbers
total_gpu = combined["gpu_hours"].sum()
ghost_jobs = (combined["crosslayer_tier"] == "Ghost").sum()
ghost_gpu  = combined.loc[combined["crosslayer_tier"] == "Ghost", "gpu_hours"].sum()
ghost_hidden_jobs = (combined["crosslayer_tier"] == "Ghost_Hidden_Activity").sum()
ghost_hidden_gpu  = combined.loc[combined["crosslayer_tier"] == "Ghost_Hidden_Activity", "gpu_hours"].sum()

print(f"  Original Ghost (combined):    {n_ghost_total:,} jobs | {ghost_total_gpu:>10,.0f} GPU-hrs")
print(f"  Split:")
print(f"    Ghost (power<50W):          {ghost_jobs:>6,} jobs | {ghost_gpu:>10,.0f} GPU-hrs ({ghost_gpu/total_gpu*100:.2f}%)")
print(f"    Ghost_Hidden_Activity:      {ghost_hidden_jobs:>6,} jobs | {ghost_hidden_gpu:>10,.0f} GPU-hrs ({ghost_hidden_gpu/total_gpu*100:.2f}%)")
print(f"  Reconciliation: {ghost_jobs + ghost_hidden_jobs:,} == {n_ghost_total:,}  "
      f"{'✓' if ghost_jobs + ghost_hidden_jobs == n_ghost_total else '✗'}")

# Updated wasteful headline
new_wasteful_gpu = combined.loc[combined["is_wasteful"]==1, "gpu_hours"].sum()
print(f"\n  Updated wasteful GPU-hrs: {new_wasteful_gpu:,.0f} ({new_wasteful_gpu/total_gpu*100:.2f}%)")
print(f"    Down from 63.5% (original Ghost in wasteful)")

# ── B. Scale_Inefficient power asymmetry (no split, but report)
print("\n" + "=" * 68)
print("B. SCALE_INEFFICIENT POWER ASYMMETRY (reported, not split)")
print("=" * 68)

sw = combined[combined["crosslayer_tier"] == "Scale_Inefficient"].copy()
if len(sw) == 0:
    sw = combined[combined["crosslayer_tier"] == "Scale_Waster"].copy()
sw_pwr = sw["power_mean"].notna()
sw_low_pwr  = (sw.loc[sw_pwr, "power_mean"] < 50).sum()
sw_high_pwr = (sw.loc[sw_pwr, "power_mean"] >= 50).sum()
print(f"  Scale_Inefficient total:        {len(sw):,}")
print(f"    With power data:              {sw_pwr.sum():,}")
print(f"    power < 50W (true low-util):  {sw_low_pwr:,} ({sw_low_pwr/sw_pwr.sum()*100:.1f}%)")
print(f"    power >= 50W (hidden activ.): {sw_high_pwr:,} ({sw_high_pwr/sw_pwr.sum()*100:.1f}%)")
print(f"  Reported as caveat in §V-C, not subdivided due to small sample (n={len(sw)})")
# Print exact low-power Scale_Inefficient jobs
low_sw = sw[sw["power_mean"].notna() & (sw["power_mean"] < 50)].copy()

low_sw["scale_bin"] = np.select(
    [
        (low_sw["gpus"] >= 8) & (low_sw["gpus"] < 32),
        (low_sw["gpus"] >= 32) & (low_sw["gpus"] < 128),
        (low_sw["gpus"] >= 128),
    ],
    ["Medium", "Large", "Extreme"],
    default="Other"
)

print("\n  Low-power Scale_Inefficient jobs:")
print(low_sw[[
    "JOB_NAME", "scale_bin", "NODES_USED", "gpus",
    "gpu_hours", "util_mean", "power_mean"
]].to_string(index=False))

# Scale_Inefficient phase/imbalance audit
print("\n[Scale_Inefficient phase and imbalance audit]")
sw = combined[combined["crosslayer_tier"].isin(["Scale_Inefficient", "Scale_Waster"])].copy()

sw["scale_bin"] = np.select(
    [
        (sw["gpus"] >= 8) & (sw["gpus"] < 32),
        (sw["gpus"] >= 32) & (sw["gpus"] < 128),
        (sw["gpus"] >= 128),
    ],
    ["Medium", "Large", "Extreme"],
    default="Other"
)

cols = [
    "util_mean", "util_phase1", "util_phase2", "util_phase3",
    "util_p95", "idle_frac", "active_phase_frac",
    "gpu_imbalance_mean", "node_util_imbalance_std",
    "power_mean"
]
cols = [c for c in cols if c in sw.columns]

print(
    sw.groupby("scale_bin")[cols]
      .median(numeric_only=True)
      .round(2)
      .to_string()
)
# ── C. R/W split analysis on IO_Bottlenecked (NEW)
print("\n" + "=" * 68)
print("C. IO_BOTTLENECKED: READ/WRITE TEMPORAL DOMINANCE (NEW)")
print("=" * 68)

io_b = combined[combined["crosslayer_tier"] == "IO_Bottlenecked"].copy()
if "io_read_time_frac" in io_b.columns and "io_write_time_frac" in io_b.columns:
    rt = io_b["io_read_time_frac"].fillna(0)
    wt = io_b["io_write_time_frac"].fillna(0)
    
    n_read_dom  = (rt > 1.5 * wt).sum()
    n_write_dom = (wt > 1.5 * rt).sum()
    n_balanced  = len(io_b) - n_read_dom - n_write_dom
    
    print(f"  IO_Bottlenecked total: {len(io_b):,}")
    print(f"  Read-dominated  (read_time > 1.5x write_time): {n_read_dom:>4,} ({n_read_dom/len(io_b)*100:.1f}%)")
    print(f"  Write-dominated (write_time > 1.5x read_time): {n_write_dom:>4,} ({n_write_dom/len(io_b)*100:.1f}%)")
    print(f"  Balanced/interleaved:                          {n_balanced:>4,} ({n_balanced/len(io_b)*100:.1f}%)")
    
    # Phase position of dominant direction
    if n_read_dom > 0:
        rd = io_b[rt > 1.5 * wt]
        print(f"\n  Read-dominated phase position (median io_read_phase_start_frac):")
        print(f"    median start: {rd['io_read_phase_start_frac'].median():.3f}")
        print(f"    median end:   {rd['io_read_phase_end_frac'].median():.3f}")
    if n_write_dom > 0:
        wd = io_b[wt > 1.5 * rt]
        print(f"  Write-dominated phase position (median io_write_phase_start_frac):")
        print(f"    median start: {wd['io_write_phase_start_frac'].median():.3f}")
        print(f"    median end:   {wd['io_write_phase_end_frac'].median():.3f}")
else:
    print("  R/W split columns not available — rerun parse_darshan with patched _parse_heatmap")

# ── D. Burstiness analysis on IO_Bottlenecked (NEW)
print("\n" + "=" * 68)
print("D. IO_BOTTLENECKED: BURSTINESS STRUCTURE (NEW)")
print("=" * 68)

if "io_n_io_bursts" in io_b.columns and "io_max_gap_bins" in io_b.columns:
    bursts = io_b["io_n_io_bursts"].fillna(1)
    overlap = io_b["io_rw_overlap_frac"].fillna(0)
    
    n_sustained = (bursts <= 2).sum()
    n_bursty    = (bursts >= 5).sum()
    n_moderate  = len(io_b) - n_sustained - n_bursty
    
    n_concurrent = (overlap > 0.1).sum()
    n_sequential = len(io_b) - n_concurrent
    
    print(f"  Burstiness ({len(io_b):,} jobs):")
    print(f"    Sustained (<=2 bursts):  {n_sustained:>4,} ({n_sustained/len(io_b)*100:.1f}%)")
    print(f"    Moderate  (3-4 bursts):  {n_moderate:>4,} ({n_moderate/len(io_b)*100:.1f}%)")
    print(f"    Bursty    (>=5 bursts):  {n_bursty:>4,} ({n_bursty/len(io_b)*100:.1f}%)")
    print(f"\n  R/W concurrency:")
    print(f"    Concurrent (>10% overlap bins): {n_concurrent:>4,} ({n_concurrent/len(io_b)*100:.1f}%)")
    print(f"    Sequential (<10% overlap):      {n_sequential:>4,} ({n_sequential/len(io_b)*100:.1f}%)")
else:
    print("  Burstiness columns not available — rerun parse_darshan")

# ── E. POSIX time breakdown validation (NEW — independent meta-bound check)
print("\n" + "=" * 68)
print("E. POSIX TIME BREAKDOWN: INDEPENDENT METADATA-BOUND VALIDATION (NEW)")
print("=" * 68)

if "meta_time_frac" in io_b.columns:
    mtf = io_b["meta_time_frac"].fillna(0)
    rtf = io_b["read_time_frac"].fillna(0)
    wtf = io_b["write_time_frac"].fillna(0)
    
    # Time-based metadata-bound (independent of metadata_ops_per_gb)
    n_meta_time = (mtf > 0.5).sum()
    print(f"  Time-based attribution (POSIX_F_*_TIME):")
    print(f"    meta_time > 50% of POSIX I/O time:  {n_meta_time:>4,} ({n_meta_time/len(io_b)*100:.1f}%)")
    
    # Cross-check: agreement between operation-based and time-based metadata-bound
    op_meta = io_b["metadata_ops_per_gb"].fillna(0) > 1000
    time_meta = mtf > 0.5
    both_agree = (op_meta & time_meta).sum()
    only_op   = (op_meta & ~time_meta).sum()
    only_time = (~op_meta & time_meta).sum()
    print(f"\n  Cross-check (operation-based vs time-based metadata-bound):")
    print(f"    Both signals agree (meta-bound):     {both_agree:>4,}")
    print(f"    Only ops/GB > 1000 (high meta ops):  {only_op:>4,}")
    print(f"    Only time-frac > 0.5 (slow meta):    {only_time:>4,}")
    print(f"    Agreement strengthens metadata-bound claim in §V-C")
else:
    print("  POSIX time columns not available")

# ── F. Multi-Darshan-log audit (NEW — addresses aggregation question)
print("\n" + "=" * 68)
print("F. MULTI-DARSHAN-LOG PER JOB AUDIT")
print("=" * 68)

if "darshan_file_count" in combined.columns:
    dfc = combined.loc[combined["darshan_present"], "darshan_file_count"].fillna(0)
    n_one  = (dfc == 1).sum()
    n_two  = (dfc == 2).sum()
    n_more = (dfc >= 3).sum()
    print(f"  Jobs with Darshan instrumentation:    {len(dfc):,}")
    print(f"    Single Darshan log per job:         {n_one:>6,} ({n_one/len(dfc)*100:.1f}%)")
    print(f"    Two Darshan logs:                   {n_two:>6,} ({n_two/len(dfc)*100:.1f}%)")
    print(f"    Three or more (multi-rank/restart): {n_more:>6,} ({n_more/len(dfc)*100:.1f}%)")
else:
    # Compute from raw dm
    dm_per_job = dm.groupby("job_id").size()
    n_one  = (dm_per_job == 1).sum()
    n_more = (dm_per_job >= 2).sum()
    print(f"  From raw darshan_metrics:")
    print(f"    Single log:        {n_one:,} ({n_one/len(dm_per_job)*100:.1f}%)")
    print(f"    Multiple logs:     {n_more:,} ({n_more/len(dm_per_job)*100:.1f}%)")

# ── G. Darshan instrumentation bias check
print("\n" + "=" * 68)
print("G. DARSHAN INSTRUMENTATION BIAS CHECK")
print("=" * 68)

with_dar = combined[combined["darshan_present"] & combined["has_gpu"]]
without_dar = combined[~combined["darshan_present"] & combined["has_gpu"]]

print(f"  Jobs with DCGM:")
print(f"    With Darshan (n={len(with_dar):,}):    "
      f"mean runtime={with_dar['RUNTIME_SECONDS'].mean():>7.0f}s, "
      f"fail rate={(with_dar['EXIT_STATUS']!=0).mean()*100:>4.1f}%")
print(f"    Without Darshan (n={len(without_dar):,}): "
      f"mean runtime={without_dar['RUNTIME_SECONDS'].mean():>7.0f}s, "
      f"fail rate={(without_dar['EXIT_STATUS']!=0).mean()*100:>4.1f}%")

from scipy.stats import ks_2samp
ks_stat, ks_p = ks_2samp(with_dar["RUNTIME_SECONDS"].dropna(), 
                          without_dar["RUNTIME_SECONDS"].dropna())
print(f"  KS test (runtime distribution): D={ks_stat:.3f}, p={ks_p:.2e}")
print(f"  -> Darshan-instrumented jobs are systematically different from corpus")

# ── H. Threshold sensitivity: I/O substantive threshold (NEW sweep)
print("\n" + "=" * 68)
print("H. THRESHOLD SENSITIVITY: I/O SUBSTANTIVE BOUNDARY")
print("=" * 68)

print(f"  {'I/O thresh':>10} {'IO_Bottlenecked':>16} {'Incidental_IO_Low_GPU':>22}")
for io_thresh in [0.02, 0.03, 0.05, 0.07, 0.10]:
    # Reapply: jobs with util<10%, exit=0, has_io, and io_time_frac > thresh -> IO_Bottlenecked
    m_low_gpu = combined["has_gpu"] & (combined["util_mean"].fillna(-1) < 10) & ~combined["exit_failed"]
    m_io_det  = combined["io_detected"]
    m_substantive = combined["io_time_frac"].fillna(0) > io_thresh
    
    n_iob = (m_low_gpu & m_io_det & m_substantive).sum()
    n_inc = (m_low_gpu & m_io_det & ~m_substantive).sum()
    marker = " <- base" if io_thresh == 0.05 else ""
    print(f"  {io_thresh:>10.2f}   {n_iob:>15,}   {n_inc:>21,}{marker}")

# ── I. Threshold sensitivity: GPU_Idle_Timeout walltime fraction
print("\n" + "=" * 68)
print("I. THRESHOLD SENSITIVITY: GPU_IDLE_TIMEOUT WALLTIME FRACTION")
print("=" * 68)

neg29 = combined[combined["EXIT_STATUS"] == -29].copy()
wt_frac = neg29["RUNTIME_SECONDS"] / neg29["WALLTIME_SECONDS"].replace(0, np.nan)
util = neg29["util_mean"].fillna(-1)
gpus = neg29["NODES_USED"] * 4
no_io = ~neg29["io_detected"]

print(f"  {'Walltime thresh':>16} {'GPU_Idle_Timeout count':>24}")
for wt_thresh in [0.70, 0.75, 0.80, 0.85, 0.90]:
    n_git = ((wt_frac > wt_thresh) & (util < 5) & (gpus >= 4) & no_io).sum()
    marker = " <- base" if wt_thresh == 0.80 else ""
    print(f"  {wt_thresh:>16.2f}   {n_git:>23,}{marker}")

# ── J. Calibration / Brier score for M3 (NEW)
print("\n" + "=" * 68)
print("J. M3 CALIBRATION (Brier score + reliability bins)")
print("=" * 68)

if "M3" in dir():
    from sklearn.metrics import brier_score_loss
    brier = brier_score_loss(y_test, M3["prob"])
    print(f"  Brier score: {brier:.4f}  (lower = better-calibrated; perfect=0)")
    
    # Reliability diagram bins
    print(f"\n  Reliability bins (predicted vs actual wasteful rate):")
    print(f"  {'Pred bin':>12} {'n':>8} {'Mean pred':>11} {'Actual rate':>13}")
    bins = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
    for lo, hi in bins:
        mask = (M3["prob"] >= lo) & (M3["prob"] < hi)
        if mask.sum() < 30: continue
        mean_pred = M3["prob"][mask].mean()
        actual = y_test[mask].mean()
        print(f"  [{lo:.1f}, {hi:.1f})   {mask.sum():>8,} {mean_pred:>11.3f} {actual:>13.3f}")
    print(f"  -> If miscalibrated, Platt or isotonic recalibration is recommended")

# ── K. Cold-start performance (MOVE TO PAPER)
print("\n" + "=" * 68)
print("K. COLD-START PERFORMANCE (ELEVATE TO PAPER)")
print("=" * 68)

if "M3" in dir() and "telem_df" in dir():
    test_df = telem_df.iloc[split_t:].copy().reset_index(drop=True)
    test_df["pred_proba"] = M3["prob"]
    
    print(f"  {'Bucket':<32} {'n':>7} {'pos%':>6} {'AUC':>8}")
    for lo, hi, label in [(-0.5, 0.5, "0 prior jobs (cold start)"),
                           (0.5, 3.5, "1-3 prior jobs"),
                           (3.5, 9.5, "4-9 prior jobs"),
                           (9.5, 999, "10 prior jobs (max history)")]:
        sub = test_df[(test_df["user_job_count"] > lo) & (test_df["user_job_count"] <= hi)]
        if len(sub) < 50 or sub["is_wasteful"].nunique() < 2:
            print(f"  {label:<32} {len(sub):>7,}  (insufficient)")
            continue
        auc = roc_auc_score(sub["is_wasteful"], sub["pred_proba"])
        pos = sub["is_wasteful"].mean() * 100
        print(f"  {label:<32} {len(sub):>7,} {pos:>5.1f}% {auc:>8.4f}")
    print(f"  -> Cold-start AUC vs M1 ({M1['auc']:.3f}): adds caveat for new users")

# ── L. Scale_Inefficient recall investigation
print("\n" + "=" * 68)
print("L. SCALE_INEFFICIENT 70% RECALL INVESTIGATION")
print("=" * 68)

if "M3" in dir():
    test_df = telem_df.iloc[split_t:].copy().reset_index(drop=True)
    test_df["pred_proba"] = M3["prob"]
    test_df["pred"] = M3["pred"]
    
    sw_test = test_df[test_df["crosslayer_tier"].isin(["Scale_Inefficient", "Scale_Waster"])]
    if len(sw_test) > 0:
        print(f"  Scale_Inefficient test set: {len(sw_test):,} jobs")
        print(f"  Recall: {(sw_test['pred']==1).mean():.3f}")
        print(f"\n  Missed Scale_Inefficient jobs (pred=0):")
        missed = sw_test[sw_test["pred"] == 0]
        if len(missed) > 0:
            print(f"    n={len(missed)}")
            print(f"    Mean util_mean (their actual util): {missed['util_mean'].mean():.2f}%")
            print(f"    Mean pred probability: {missed['pred_proba'].mean():.3f}")
            print(f"    -> These jobs sit at the productive boundary (util close to 10%)")
            close_to_boundary = (missed["util_mean"] >= 8.0).sum()
            print(f"    {close_to_boundary}/{len(missed)} have util >= 8%, near the 10% threshold")

# ── M. Burst structure consolidated (from §VII)
print("\n" + "=" * 68)
print("M. BURST STRUCTURE CONSOLIDATED")
print("=" * 68)

cs = combined.sort_values(["USERNAME_GENID", "QUEUED_TIMESTAMP"]).reset_index(drop=True)
cs["prev_w"] = cs.groupby("USERNAME_GENID")["is_wasteful"].shift(1).fillna(0)
cs["in_burst"] = ((cs["is_wasteful"] == 1) & (cs["prev_w"] == 1)).astype(int)

SUBST = {"Ghost", "IO_Bottlenecked", "Scale_Inefficient", "Scale_Waster", "Failed_Job", "GPU_Idle_Timeout"}
sub_w = cs[cs["crosslayer_tier"].isin(SUBST)]
sub_b = cs[cs["crosslayer_tier"].isin(SUBST) & (cs["in_burst"] == 1)]
all_w = cs[cs["is_wasteful"] == 1]
all_b = cs[cs["in_burst"] == 1]

print(f"  Substantive tiers in burst: {len(sub_b):,}/{len(sub_w):,} ({len(sub_b)/max(len(sub_w),1)*100:.1f}%)")
print(f"  Including Quick_Cancel:     {len(all_b):,}/{len(all_w):,} ({len(all_b)/len(all_w)*100:.1f}%)")

# ── N. UPDATED PAPER NUMBER SUMMARY
print("\n" + "=" * 68)
print("UPDATED PAPER NUMBERS — paste-ready")
print("=" * 68)

new_wasteful_jobs = (combined["is_wasteful"] == 1).sum()
new_wasteful_gpu  = combined.loc[combined["is_wasteful"] == 1, "gpu_hours"].sum()

print(f"""
  CORPUS (unchanged)
    Jobs total           : {total_jobs:,}
    GPU-hrs total        : {total_gpu:,.0f} ({total_gpu/1e6:.2f}M)

  TAXONOMY (UPDATED — Ghost split structurally)
    Ghost                  : {ghost_jobs:>6,} jobs | {ghost_gpu:>10,.0f} GPU-hrs ({ghost_gpu/total_gpu*100:.2f}%)
    Ghost_Hidden_Activity  : {ghost_hidden_jobs:>6,} jobs | {ghost_hidden_gpu:>10,.0f} GPU-hrs ({ghost_hidden_gpu/total_gpu*100:.2f}%)
    Wasteful (under-util)  : {new_wasteful_jobs:>6,} jobs | {new_wasteful_gpu:>10,.0f} GPU-hrs ({new_wasteful_gpu/total_gpu*100:.2f}%)

  HEADLINE CHANGE
    Old "wasteful" claim:  63.5%
    New "wasteful" claim:  {new_wasteful_gpu/total_gpu*100:.1f}%
""")

print("=" * 68)
print("EXTENSIONS COMPLETE")
print( "=" * 68)




# ════════════════════════════════════════════════════════════════
# PUBLICATION FIGURES
# IEEE-style, vector PDF, single-column ~3.5in width unless noted
# Each plot is independent — comment out any block you skip
# ════════════════════════════════════════════════════════════════

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
from pathlib import Path

FIG_DIR = Path("/pscratch/sd/h/hjajula/Darshan/cross_layer_hpc_tool/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

# IEEE-friendly defaults
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 9,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.linewidth": 0.6,
    "grid.linewidth": 0.4,
    "lines.linewidth": 1.2,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# Tol bright palette — colorblind-safe
TIER_COLORS = {
    "Ghost":            "#4477AA",   # blue
    "Idle":             "#4477AA",
    "IO_Bottlenecked":  "#EE6677",   # red
    "Scale_Waster":     "#228833",   # green
    "Scale_Inefficient":"#228833",
    "Compute_Bound":    "#CCBB44",   # yellow
    "GPU_Idle_Timeout": "#AA3377",   # purple
    "Failed_Job":       "#888888",   # gray
    "Other":            "#BBBBBB",
}

# ════════════════════════════════════════════════════════════════
# PUBLICATION FIGURES — TARGETED
# Each figure earns its space by replacing reader effort
# ════════════════════════════════════════════════════════════════

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

FIG_DIR = Path("/pscratch/sd/h/hjajula/Darshan/cross_layer_hpc_tool/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8.5,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.linewidth": 0.6,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42,
})

GROUP_COLORS = {
    "Wasteful":      "#C44E52",
    "Productive":    "#55A868",
    "Informational": "#DDB45D",
    "Coverage":      "#999999",
}

print("\n" + "=" * 68)
print("GENERATING TARGETED PUBLICATION FIGURES")
print("=" * 68)

# ════════════════════════════════════════════════════════════════
# FIG A — TAXONOMY TREEMAP
# Replaces the proportional reading work of Table I
# Shows: 16 tiers nested under 4 groups, area = GPU-hours
# ════════════════════════════════════════════════════════════════
try:
    import squarify
    HAS_SQ = True
except ImportError:
    HAS_SQ = False
    print("  squarify not installed (pip install squarify) — skipping treemap")

if HAS_SQ:
    GROUPS = [
        ("Wasteful", [
            ("Failed_Job",          "Failed"),
            ("GPU_Idle_Timeout",    "GPU Idle\nTimeout"),
            ("Ghost",               "Idle"),
            ("Scale_Waster",        "Scale\nIneff."),
            ("IO_Bottlenecked",     "IO\nBottle"),
            ("Quick_Cancel",        "Quick\nCancel"),
        ]),
        ("Informational", [
            ("Moderate_Compute_No_IO",  "Moderate\nCompute\nNo IO"),
            ("Idle_Hidden_Activity",    "Idle\nHidden\nActivity"),
            ("Moderate_Compute_With_IO","Moderate\nCompute\nWith IO"),
            ("Incidental_IO_Low_GPU",   "Incid.\nIO"),
            ("Low_Efficiency",          "Low\nEff."),
        ]),
        ("Productive", [
            ("Compute_Bound",        "Compute\nBound"),
            ("Ideal_Compute_With_IO","Ideal\nCompute"),
        ]),
        ("Coverage", [
            ("No_GPU_Telemetry",    "No GPU\nTelemetry"),
            ("No_GPU_With_Darshan", "No GPU\n+ Darshan"),
            ("Short_No_GPU",        "Short"),
            ("Short_No_GPU_With_IO","Short\n+ IO"),
        ]),
    ]

    # Collect sizes
    tile_sizes, tile_labels, tile_colors, tile_groups = [], [], [], []
    for grp, tiers in GROUPS:
        for tier_name, disp in tiers:
            sub = combined[combined["crosslayer_tier"] == tier_name]
            if len(sub) == 0:
                continue
            gpu = sub["gpu_hours"].sum()
            tile_sizes.append(gpu)
            tile_labels.append(disp)
            tile_colors.append(GROUP_COLORS[grp])
            tile_groups.append(grp)

    total_gpu = sum(tile_sizes)
    pcts = [s/total_gpu*100 for s in tile_sizes]

    # Squarify layout
    fig, ax = plt.subplots(figsize=(7.0, 3.6))  # double-column width
    norm_sizes = squarify.normalize_sizes(tile_sizes, 100, 60)
    rects = squarify.squarify(norm_sizes, 0, 0, 100, 60)

    for r, lbl, c, pct, gpu in zip(rects, tile_labels, tile_colors, pcts, tile_sizes):
        rect = mpatches.Rectangle(
            (r["x"], r["y"]), r["dx"], r["dy"],
            facecolor=c, edgecolor="white", linewidth=1.4)
        ax.add_patch(rect)
        # Adaptive label sizing: only show full text if tile is big enough
        cx = r["x"] + r["dx"]/2
        cy = r["y"] + r["dy"]/2
        if r["dx"] * r["dy"] > 80:  # large tile
            ax.text(cx, cy + r["dy"]*0.10, lbl,
                    ha="center", va="center", fontsize=7.5, color="white",
                    fontweight="bold")
            ax.text(cx, cy - r["dy"]*0.18, f"{pct:.1f}\\%",
                    ha="center", va="center", fontsize=7, color="white")
        elif r["dx"] * r["dy"] > 18:  # medium
            ax.text(cx, cy, f"{lbl}\n{pct:.1f}\\%",
                    ha="center", va="center", fontsize=6, color="white")
        elif r["dx"] * r["dy"] > 5:  # small
            ax.text(cx, cy, f"{pct:.1f}\\%",
                    ha="center", va="center", fontsize=5.5, color="white")
        # tiny tiles get no label

    ax.set_xlim(0, 100); ax.set_ylim(0, 60)
    ax.set_aspect("equal")
    ax.axis("off")

    # Group legend
    legend_handles = [mpatches.Patch(facecolor=GROUP_COLORS[g], edgecolor="white",
                                      label=g) for g in ["Wasteful","Productive",
                                                          "Informational","Coverage"]]
    fig.legend(handles=legend_handles, loc="lower center",
               ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.02),
               fontsize=8)
    fig.savefig(FIG_DIR / "fig_taxonomy_treemap.pdf")
    plt.close(fig)
    print(f"  ✓ fig_taxonomy_treemap.pdf  (replaces visual work of Table I)")

# ════════════════════════════════════════════════════════════════
# FIG B — SINGLE-LAYER vs CROSS-LAYER SEPARABILITY
# Visual proof of Table II's Ghost-vs-IO_Bottlenecked claim
# Two panels: GPU-only (left, indistinguishable)
#             GPU + io_end (right, separates cleanly)
# ════════════════════════════════════════════════════════════════
focus_tiers = ["Ghost", "IO_Bottlenecked", "Compute_Bound"]
sep_df = combined[combined["crosslayer_tier"].isin(focus_tiers)].copy()
sep_df = sep_df.dropna(subset=["util_mean", "io_phase_end_frac"])

# subsample for legibility
samples = []
for t in focus_tiers:
    s = sep_df[sep_df["crosslayer_tier"] == t]
    if len(s) > 1500:
        s = s.sample(1500, random_state=42)
    samples.append(s)
sep_plot = pd.concat(samples)

tier_color = {
    "Ghost":           "#4477AA",
    "IO_Bottlenecked": "#EE6677",
    "Compute_Bound":   "#228833",
}

fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6),
                          gridspec_kw={"wspace": 0.3})

# Left: GPU utilization only (1D strip)
ax = axes[0]
np.random.seed(42)
for i, t in enumerate(focus_tiers):
    s = sep_plot[sep_plot["crosslayer_tier"] == t]
    jitter = np.random.uniform(-0.3, 0.3, len(s))
    ax.scatter(s["util_mean"], np.full(len(s), i) + jitter,
               s=4, alpha=0.35, c=tier_color[t],
               edgecolors="none", rasterized=True)
ax.set_yticks(range(len(focus_tiers)))
ax.set_yticklabels(focus_tiers)
ax.set_xlabel("Mean GPU utilization (\\%)")
ax.set_xlim(-2, 100)
ax.set_title("(a) GPU utilization only:\nGhost and IO\\_Bottlenecked overlap",
             fontsize=8)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Right: GPU util × io_phase_end_frac
ax = axes[1]
for t in focus_tiers:
    s = sep_plot[sep_plot["crosslayer_tier"] == t]
    ax.scatter(s["util_mean"], s["io_phase_end_frac"],
               s=4, alpha=0.4, c=tier_color[t],
               label=f"{t} (n={len(sep_df[sep_df['crosslayer_tier']==t]):,})",
               edgecolors="none", rasterized=True)
ax.set_xlabel("Mean GPU utilization (\\%)")
ax.set_ylabel("I/O end-fraction (Layer 3)")
ax.set_xlim(-2, 100)
ax.set_ylim(-0.04, 1.04)
ax.set_title("(b) Adding I/O end-fraction:\ntiers separate cleanly", fontsize=8)
ax.legend(loc="center right", frameon=False, markerscale=2.5,
          handletextpad=0.3, labelspacing=0.2, fontsize=6.5)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.savefig(FIG_DIR / "fig_single_vs_crosslayer.pdf", dpi=300)
plt.close(fig)
print(f"  ✓ fig_single_vs_crosslayer.pdf  (cross-layer thesis in one image)")

# ════════════════════════════════════════════════════════════════
# FIG C — USER CONCENTRATION
# Replaces the buried prose: top-10=45.3%, top-5%=81.2%, top-10%=90.9%
# ════════════════════════════════════════════════════════════════
user_gpu = combined.groupby("USERNAME_GENID")["gpu_hours"].sum().sort_values(ascending=False)
total_gpu = user_gpu.sum()
n_users = len(user_gpu)
cum_gpu  = np.cumsum(user_gpu.values) / total_gpu * 100
cum_user = (np.arange(1, n_users + 1) / n_users) * 100

fig, ax = plt.subplots(figsize=(3.5, 2.4))
ax.fill_between(cum_user, 0, cum_gpu, alpha=0.18, color="#4477AA")
ax.plot(cum_user, cum_gpu, "-", c="#4477AA", lw=1.5)
ax.plot([0, 100], [0, 100], "k--", lw=0.6, alpha=0.4,
        label="equality")

# Annotate the three key points
key_points = [
    (10/n_users*100, "top-10 users"),     # 10 users
    (5,              "top-5\\% of users"), # 5%
    (10,             "top-10\\% of users"),# 10%
]
for pct_x, lbl in key_points:
    idx = max(0, int(n_users * pct_x / 100) - 1)
    cy = cum_gpu[idx]
    ax.plot([pct_x, pct_x], [0, cy], "-", c="#EE6677", lw=0.6, alpha=0.7)
    ax.plot(pct_x, cy, "o", c="#EE6677", markersize=4)
    ax.annotate(f"{lbl}\n→ {cy:.0f}\\% of GPU-hrs",
                xy=(pct_x, cy), xytext=(pct_x + 12, cy - 8),
                fontsize=6.5, color="#AA3377",
                arrowprops=dict(arrowstyle="-", color="#AA3377",
                                lw=0.4, alpha=0.7))

ax.set_xlabel("Cumulative \\% of users (sorted desc by GPU-hrs)")
ax.set_ylabel("Cumulative \\% of GPU-hours")
ax.set_xlim(0, 100); ax.set_ylim(0, 102)
ax.legend(loc="lower right", frameon=False)
ax.grid(alpha=0.25)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.savefig(FIG_DIR / "fig_user_concentration.pdf")
plt.close(fig)
print(f"  ✓ fig_user_concentration.pdf  (replaces §VII-D prose paragraph)")

# ════════════════════════════════════════════════════════════════
# FIG D — OvR SLOPE CHART: M1 → M3 per tier
# Replaces Table V's two columns + delta column with a slope visual
# ════════════════════════════════════════════════════════════════
ovr = {
    "IO_Bottlenecked":   (0.844, 0.977),
    "Scale_Inefficient": (0.732, 0.906),
    "GPU_Idle_Timeout":  (0.719, 0.900),
    "Ghost / Idle":      (0.547, 0.815),
}

fig, ax = plt.subplots(figsize=(3.5, 2.6))
y_pos = list(range(len(ovr)))
tier_names = list(ovr.keys())
tier_colors_list = ["#EE6677", "#228833", "#AA3377", "#4477AA"]

for i, (tier, (m1, m3)) in enumerate(ovr.items()):
    c = tier_colors_list[i]
    ax.plot([m1, m3], [i, i], "-", c=c, lw=2, alpha=0.9)
    ax.plot(m1, i, "o", c="white", markeredgecolor=c, markersize=8,
            markeredgewidth=1.5)
    ax.plot(m3, i, "o", c=c, markersize=8, markeredgecolor=c)
    # Numerical labels
    ax.annotate(f"{m1:.2f}", (m1, i), xytext=(-7, 0),
                textcoords="offset points", ha="right", va="center",
                fontsize=6.5, color=c)
    ax.annotate(f"{m3:.2f}", (m3, i), xytext=(7, 0),
                textcoords="offset points", ha="left", va="center",
                fontsize=6.5, color=c, fontweight="bold")
    # Delta in middle
    ax.text((m1+m3)/2, i + 0.18, f"+{m3-m1:.3f}",
            ha="center", fontsize=6, color=c, style="italic")

ax.set_yticks(y_pos)
ax.set_yticklabels(tier_names)
ax.set_xlabel("One-vs-rest AUC")
ax.set_xlim(0.45, 1.05)
ax.invert_yaxis()
ax.axvline(0.5, color="k", lw=0.4, ls=":", alpha=0.4)

# Custom legend for M1 vs M3 markers
m1_marker = plt.Line2D([],[], marker="o", color="w",
                        markerfacecolor="white", markeredgecolor="#666",
                        markersize=8, markeredgewidth=1.5,
                        label="M1 (scheduler)", linestyle="None")
m3_marker = plt.Line2D([],[], marker="o", color="w",
                        markerfacecolor="#666", markersize=8,
                        label="M3 (cross-layer)", linestyle="None")
ax.legend(handles=[m1_marker, m3_marker], loc="lower right",
          frameon=False, handletextpad=0.3, fontsize=7)
ax.grid(axis="x", alpha=0.25)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.savefig(FIG_DIR / "fig_ovr_slope.pdf")
plt.close(fig)
print(f"  ✓ fig_ovr_slope.pdf  (replaces Table V visual scan)")

print("\n  All figures →", FIG_DIR)
print("=" * 68)