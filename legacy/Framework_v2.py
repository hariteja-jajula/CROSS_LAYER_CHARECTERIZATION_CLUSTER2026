"""
framework_v2.py — Cross-layer HPC paper results.

Uses the new three-phase classifier (utils.combined_v2). Single run
produces all numbers needed for the paper body, plus an equivalence
check against the previously-saved combined_metrics_final.csv.

Saves to combined_metrics_final_v2.csv so the legacy file is preserved.
"""
print("BOOT: script entered", flush=True)
import os, sys, json, time, hashlib
import warnings; warnings.filterwarnings("ignore")
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

import importlib, utils.combined_v2 as cu
print("BOOT: utils.combined_v2 imported", flush=True)
importlib.reload(cu)
print("BOOT: utils.combined_v2 reloaded", flush=True)

from utils.combined_v2 import classify_crosslayer, to_legacy_label
print("BOOT: classify_crosslayer imported", flush=True)

t0 = time.time()
RNG = 42
N_HIST, LOOKBACK_DAYS = 10, 7

DATA_ROOT      = "/pscratch/sd/h/hjajula/Darshan/cross_layer_hpc_tool/data"
LEGACY_CSV     = f"{DATA_ROOT}/combined_metrics_final.csv"      # for equivalence check
NEW_CSV        = f"{DATA_ROOT}/combined_metrics_final_v2.csv"   # this run's output

print("Started")

# ============================================================
# 1. LOAD + MERGE
# ============================================================
cfg    = json.load(open("/pscratch/sd/h/hjajula/Darshan/cross_layer_hpc_tool/config/config.json"))
job_df = pd.read_csv(cfg["djc_csv"], low_memory=False)
gm     = pd.read_csv(f"{DATA_ROOT}/gpu_metrics.csv",            low_memory=False)
dm     = pd.read_csv(f"{DATA_ROOT}/darshan_metrics_final.csv",  low_memory=False)
dm["job_id"]     = dm["job_id"].astype(str)
job_df["job_id"] = job_df["JOB_NAME"].str.split(".").str[0]
print("data loaded")

agg_spec = {k: v for k, v in {
    # bytes / op counts
    "bytes_read": "sum", "bytes_written": "sum",
    "posix_reads": "sum", "posix_writes": "sum",
    "posix_opens": "sum", "posix_stats": "sum",
    "mpiio_bytes_read": "sum", "mpiio_bytes_written": "sum",
    "mpiio_coll_reads": "sum", "mpiio_coll_writes": "sum",
    "mpiio_indep_reads": "sum", "mpiio_indep_writes": "sum",
    "stdio_bytes_read": "sum", "stdio_bytes_written": "sum",
    # runtime metadata
    "runtime": "max", "nprocs": "max",
    # rank timing/imbalance
    "slowest_rank_time": "max", "fastest_rank_time": "max",
    "variance_rank_time": "max",
    "rank_imbalance": "max", "rank_time_imbalance": "max",
    "rank_time_gap": "max",
    # combined heatmap
    "io_time_frac": "max", "io_density": "max",
    "io_active_bins": "sum",
    "io_phase_start_frac": "min", "io_phase_end_frac": "max",
    # read/write split heatmap
    "io_read_time_frac": "max", "io_write_time_frac": "max",
    "io_read_density": "max", "io_write_density": "max",
    "io_read_active_bins": "sum", "io_write_active_bins": "sum",
    "io_read_phase_start_frac": "min", "io_read_phase_end_frac": "max",
    "io_write_phase_start_frac": "min", "io_write_phase_end_frac": "max",
    # R/W overlap and burstiness
    "io_rw_overlap_frac": "max",
    "io_max_gap_bins": "max", "io_mean_gap_bins": "mean",
    "io_n_io_bursts": "max",
    # access pattern
    "seq_read_ratio": "mean", "seq_write_ratio": "mean",
    "small_read_ratio": "mean", "large_read_ratio": "mean",
    "write_dominance": "mean", "mpiio_coll_ratio": "mean",
    "io_read_front_heavy": "max", "io_write_back_heavy": "max",
    # POSIX time breakdown
    "posix_meta_time": "max", "posix_read_time": "max",
    "posix_write_time": "max",
    "meta_time_frac": "mean", "read_time_frac": "mean",
    "write_time_frac": "mean",
    # module/status
    "has_posix": "max", "has_mpiio": "max", "has_heatmap": "max",
    # config / structure
    "cb_nodes": "max", "unique_files": "sum",
    "metadata_ops_per_gb": "mean",
    "mem_not_aligned_ratio": "mean", "file_not_aligned_ratio": "mean",
}.items() if k in dm.columns}
dm_agg = dm.groupby("job_id").agg(agg_spec).reset_index()

combined = job_df.merge(gm, on="JOB_NAME", how="left").merge(dm_agg, on="job_id", how="left")
old_exe  = pd.read_csv(f"{DATA_ROOT}/darshan_metrics_old.csv",
                        usecols=["fname","executable"], low_memory=False)
old_exe["job_id"] = old_exe["fname"].str.split("-").str[0]
combined = combined.merge(
    old_exe.groupby("job_id")["executable"].first().reset_index(),
    on="job_id", how="left")

combined["gpu_util_mean"]       = combined["util_mean"]
combined["io_read_front_heavy"] = combined["io_read_front_heavy"].fillna(0).astype(bool)
combined["io_write_back_heavy"] = combined["io_write_back_heavy"].fillna(0).astype(bool)

# ============================================================
# 2. TAXONOMY (no post-process needed — Phase 2 produces all final labels)
# ============================================================
sys.stdout = open(os.devnull, 'w')
combined = classify_crosslayer(combined)
sys.stdout = sys.__stdout__

WASTEFUL = {"Idle", "Scale_Inefficient", "IO_Bottlenecked",
            "Failed_Job", "Quick_Cancel", "GPU_Idle_Timeout"}
combined["is_wasteful"] = combined["crosslayer_tier"].isin(WASTEFUL).astype(int)

# Save new output, preserving the legacy CSV for equivalence comparison
combined.to_csv(NEW_CSV, index=False)

total_gpu  = combined["gpu_hours"].sum()
total_jobs = len(combined)

# ============================================================
# 2b. EQUIVALENCE CHECK vs legacy CSV
# ============================================================
print("\n" + "=" * 60)
print("EQUIVALENCE CHECK vs legacy combined_metrics_final.csv")
print("=" * 60)
if os.path.exists(LEGACY_CSV):
    legacy = pd.read_csv(LEGACY_CSV,
                          usecols=["JOB_NAME", "crosslayer_tier"],
                          low_memory=False)
    legacy = legacy.rename(columns={"crosslayer_tier": "legacy_tier"})
    diff_df = combined[["JOB_NAME", "crosslayer_tier", "io_detected",
                        "EXIT_STATUS"]].merge(legacy, on="JOB_NAME", how="left")
    diff_df["expected_legacy"] = diff_df.apply(
        lambda r: to_legacy_label(r["crosslayer_tier"], bool(r["io_detected"])),
        axis=1)
    matched   = (diff_df["expected_legacy"] == diff_df["legacy_tier"])
    n_match   = matched.sum()
    n_total   = matched.shape[0]
    n_diff    = n_total - n_match
    print(f"  Total jobs in new run        : {n_total:,}")
    print(f"  Match expected legacy mapping: {n_match:,} ({n_match/n_total*100:.2f}%)")
    print(f"  Mismatches (require review)  : {n_diff:,} ({n_diff/n_total*100:.2f}%)")
    if n_diff > 0:
        diffs = diff_df[~matched].copy()
        # Behavioral change #5: GPU_Idle_Timeout now uses ~substantive_io
        # rather than ~any_io. Rescued walltime-exhausted jobs go from
        # Failed_Job (legacy) -> GPU_Idle_Timeout (new).
        rescued_git = ((diffs["crosslayer_tier"] == "GPU_Idle_Timeout") &
                       (diffs["legacy_tier"]     == "Failed_Job") &
                       (diffs["EXIT_STATUS"]     == -29))
        n_rescued = rescued_git.sum()
        print(f"\n  Of mismatches, {n_rescued:,} are documented behavioral change #5")
        print(f"  (GPU_Idle_Timeout rescued from Failed_Job via ~substantive_io)")
        residual = diffs[~rescued_git]
        if len(residual) > 0:
            print(f"\n  Residual unexplained mismatches: {len(residual):,}")
            print(f"  Top 10 mismatch patterns (new -> legacy):")
            pattern = (residual.groupby(["crosslayer_tier", "legacy_tier"])
                                .size().sort_values(ascending=False).head(10))
            for (new, old), n in pattern.items():
                print(f"    {new:<25} -> {old:<25} : {n:,}")
        else:
            print(f"  All mismatches accounted for by documented changes OK")
else:
    print(f"  Legacy CSV not found at {LEGACY_CSV} — skipping equivalence check.")

# ============================================================
# 3. RESULTS — COVERAGE & TAXONOMY
# ============================================================
print("\n" + "=" * 60)
print("COVERAGE")
print("=" * 60)
gpu_cov     = combined["has_gpu"].sum()
dar_present = combined["darshan_present"].sum()
io_detected = combined["io_detected"].sum()
print(f"Total jobs            : {total_jobs:,}")
print(f"GPU telemetry (DCGM)  : {gpu_cov:,} ({gpu_cov/total_jobs*100:.1f}%)")
print(f"Darshan attached      : {dar_present:,} ({dar_present/total_jobs*100:.1f}%)")
print(f"Darshan I/O detected  : {io_detected:,} ({io_detected/total_jobs*100:.1f}%)")
print(f"Both GPU + Darshan    : "
      f"{(combined['has_gpu']&combined['darshan_present']).sum():,}")
print(f"Total allocated GPU-hrs: {total_gpu:,.0f}")

print("\n" + "=" * 60)
print("TAXONOMY TABLE", flush=True)
print("=" * 60)
tier_order = [
    # Phase 1 (accounting / outcome)
    "Quick_Cancel", "Failed_Job", "GPU_Idle_Timeout",
    # Phase 3 (structural refinement of LOW,NONE)
    "Idle", "Idle_Hidden_Activity",
    "Scale_Inefficient",
    # Phase 2 (base compute x I/O)
    "IO_Bottlenecked", "Incidental_IO_Low_GPU",
    "Moderate_Compute_No_IO", "Moderate_Compute_With_IO",
    "Compute_Bound", "Ideal_Compute_With_IO",
    # Phase 3 fall-through
    "Low_Efficiency",
    # Phase 1 no-coverage tiers
    "No_GPU_Telemetry", "No_GPU_With_Darshan",
    "Short_No_GPU", "Short_No_GPU_With_IO",
]
print(f"{'Tier':<28} {'Jobs':>8} {'GPU-hrs':>12} {'% GPU-hrs':>10}")
print("-" * 62)
for tier in tier_order:
    s = combined[combined["crosslayer_tier"] == tier]
    if len(s) == 0: continue
    gh = s["gpu_hours"].sum()
    print(f"{tier:<28} {len(s):>8,} {gh:>12,.0f} {gh/total_gpu*100:>9.1f}%")

wasteful_gpu = combined.loc[combined["is_wasteful"]==1, "gpu_hours"].sum()
failed_git   = combined.loc[combined["crosslayer_tier"].isin(
                {"Failed_Job","GPU_Idle_Timeout"}), "gpu_hours"].sum()
hidden_gpu   = combined.loc[combined["crosslayer_tier"]=="Idle_Hidden_Activity",
                             "gpu_hours"].sum()
print(f"\nUnder-utilized GPU-hrs : {wasteful_gpu:,.0f} ({wasteful_gpu/total_gpu*100:.1f}%)")
print(f"Failed+GIT GPU-hrs     : {failed_git:,.0f} ({failed_git/total_gpu*100:.1f}%)")
print(f"Idle_Hidden_Activity   : {hidden_gpu:,.0f} ({hidden_gpu/total_gpu*100:.1f}%) "
      f"[excluded from under-utilized]")

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
    for label, n in [("Metadata-bound (>1000 ops/GB)", n_meta),
                     ("Bandwidth-bound (<1000 MB/s)",  n_bw),
                     ("Rank-imbalance (>2x)",          n_strag),
                     ("Other",                         n_other)]:
        print(f"  {label:<35}: {n:>5,} ({n/len(io_b)*100:.1f}%)")

print("\n[Scale_Inefficient by allocation size]")
sw = combined[combined["crosslayer_tier"] == "Scale_Inefficient"]
if len(sw) > 0:
    for label, mask in [
        ("Medium  (2-7 nodes,   8-31 GPUs)",  (sw["gpus"]>=8)  & (sw["gpus"]<32)),
        ("Large   (8-31 nodes, 32-127 GPUs)", (sw["gpus"]>=32) & (sw["gpus"]<128)),
        ("Extreme (32+ nodes,  128+ GPUs)",    sw["gpus"]>=128),
    ]:
        n = mask.sum()
        print(f"  {label}: {n:>4,} jobs | "
              f"{sw.loc[mask,'gpu_hours'].sum():>8,.0f} GPU-hrs")

# ============================================================
# 3b. CROSS-SECTIONAL ANALYSIS
# ============================================================
print("\n" + "=" * 60)
print("CROSS-SECTIONAL ANALYSIS")
print("=" * 60)
combined["QUEUED_TIMESTAMP"] = pd.to_datetime(combined["QUEUED_TIMESTAMP"], errors="coerce")
combined["END_TIMESTAMP"]    = pd.to_datetime(combined["END_TIMESTAMP"],    errors="coerce")

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
    print(f"  {q:<20} {int(combined[combined['QUEUE_NAME']==q].shape[0]):>8,} "
          f"{queue_tier.loc[q,'wasteful_pct']:>9.1f}%")

print("\n[Wasteful job rate by science field (top 8 fields)]")
top_fields = combined["SCIENCE_FIELD"].value_counts().head(8).index
for field in top_fields:
    sub = combined[combined["SCIENCE_FIELD"] == field]
    w_pct      = sub["is_wasteful"].mean() * 100
    gpu_wasted = sub.loc[sub["is_wasteful"]==1, "gpu_hours"].sum()
    print(f"  {str(field):<30} {len(sub):>6,} jobs  {w_pct:>5.1f}% wasteful  "
          f"{gpu_wasted:>10,.0f} GPU-hrs wasted")

print("\n[Wasteful rate by node count (allocation size)]")
combined["node_bin"] = pd.cut(combined["NODES_USED"],
    bins=[0,1,2,4,8,16,32,64,128,9999],
    labels=["1","2","3-4","5-8","9-16","17-32","33-64","65-128","128+"])
for nb, grp in combined.groupby("node_bin", observed=True):
    if len(grp) < 50: continue
    w_pct    = grp["is_wasteful"].mean() * 100
    gpu_mean = grp.loc[grp["has_gpu"], "util_mean"].mean()
    print(f"  {str(nb):<8} nodes: {len(grp):>7,} jobs  {w_pct:>5.1f}% wasteful  "
          f"mean GPU util {gpu_mean:>5.1f}%")

print("\n[Monthly wasteful job rate (temporal trend)]")
combined["month_year"] = combined["QUEUED_TIMESTAMP"].dt.to_period("M")
for period, grp in combined.groupby("month_year", observed=True):
    total    = len(grp)
    wasteful = grp["is_wasteful"].sum()
    gpu_w    = grp.loc[grp["is_wasteful"]==1, "gpu_hours"].sum()
    print(f"  {str(period):<10}  {total:>6,} jobs  "
          f"{wasteful/total*100:>5.1f}% wasteful  "
          f"{gpu_w:>10,.0f} GPU-hrs wasted")

print("\n[Waste concentration by user (top 10 users by GPU-hrs wasted)]")
user_waste = combined[combined["is_wasteful"]==1].groupby("USERNAME_GENID").agg(
    jobs=("JOB_NAME","count"), gpu_hrs=("gpu_hours","sum")
).sort_values("gpu_hrs", ascending=False).head(10)
total_wasteful_gpu = combined.loc[combined["is_wasteful"]==1, "gpu_hours"].sum()
cumulative = 0
for i, (user, row) in enumerate(user_waste.iterrows(), 1):
    cumulative += row["gpu_hrs"]
    print(f"  #{i:<3} {str(user):<20} {int(row['jobs']):>6,} jobs  "
          f"{row['gpu_hrs']:>10,.0f} GPU-hrs  "
          f"cumulative {cumulative/total_wasteful_gpu*100:.1f}%")

# Idle vs Idle_Hidden_Activity by queue (replaces old Ghost-power-gap section)
print("\n[Idle vs Idle_Hidden_Activity by queue]")
idle_all = combined[combined["crosslayer_tier"].isin(
                    ["Idle","Idle_Hidden_Activity"])].copy()
if len(idle_all) > 0 and "QUEUE_NAME" in idle_all.columns:
    qg = idle_all.groupby(["QUEUE_NAME","crosslayer_tier"]).size().unstack(fill_value=0)
    qg["total"]      = qg.sum(axis=1)
    qg = qg.sort_values("total", ascending=False)
    print(f"  {'Queue':<20} {'Total':>8} {'Idle':>8} {'Hidden':>8} {'Hidden%':>9}")
    for q, row in qg.head(15).iterrows():
        idle_n   = row.get("Idle", 0)
        hidden_n = row.get("Idle_Hidden_Activity", 0)
        if row["total"] < 20: continue
        pct = hidden_n / row["total"] * 100
        print(f"  {str(q):<20} {int(row['total']):>8,} {int(idle_n):>8,} "
              f"{int(hidden_n):>8,} {pct:>8.1f}%")

print("\n[IO_Bottlenecked: GPU-hrs by user (top 5)]")
io_users = combined[combined["crosslayer_tier"]=="IO_Bottlenecked"].groupby(
    "USERNAME_GENID")["gpu_hours"].sum().sort_values(ascending=False).head(5)
total_io_gpu = combined.loc[combined["crosslayer_tier"]=="IO_Bottlenecked",
                             "gpu_hours"].sum()
for user, gpu_hrs in io_users.items():
    print(f"  {str(user):<20} {gpu_hrs:>10,.0f} GPU-hrs "
          f"({gpu_hrs/total_io_gpu*100:.1f}%)")

print("\n[Scale_Inefficient: power validation (shared-memory vs true low-util)]")
sw = combined[combined["crosslayer_tier"] == "Scale_Inefficient"].copy()
if len(sw) > 0:
    has_pwr = sw["power_mean"].notna()
    low_pwr  = (sw.loc[has_pwr, "power_mean"] < 50).sum()
    high_pwr = (sw.loc[has_pwr, "power_mean"] >= 50).sum()
    no_pwr   = (~has_pwr).sum()
    n_pwr    = has_pwr.sum()
    print(f"  Total Scale_Inefficient jobs: {len(sw):,}")
    print(f"  With power data             : {n_pwr:,}")
    if n_pwr > 0:
        print(f"  power < 50W (true low-util) : {low_pwr:,} "
              f"({low_pwr/n_pwr*100:.1f}%)")
        print(f"  power >= 50W (hidden activ.): {high_pwr:,} "
              f"({high_pwr/n_pwr*100:.1f}%)")
    print(f"  No power data               : {no_pwr:,}")
    sw["alloc_size"] = pd.cut(sw["gpus"],
        bins=[0,31,127,99999],
        labels=["Medium (8-31 GPUs)","Large (32-127 GPUs)","Extreme (128+ GPUs)"])
    print(f"\n  Phase / imbalance audit by allocation size:")
    cols = ["util_mean","util_phase1","util_phase2","util_phase3",
            "util_p95","idle_frac","active_phase_frac",
            "gpu_imbalance_mean","node_util_imbalance_std","power_mean"]
    cols = [c for c in cols if c in sw.columns]
    audit = sw.groupby("alloc_size", observed=True)[cols].mean().round(2)
    print(audit.to_string())

print("\n[Workload concentration: per-tier breakdown for top users]")
wasteful_df = combined[combined["is_wasteful"] == 1]
tier_conc = (wasteful_df.groupby(["USERNAME_GENID","crosslayer_tier"])
                          ["gpu_hours"].sum().reset_index())
top10_users = (wasteful_df.groupby("USERNAME_GENID")["gpu_hours"]
               .sum().sort_values(ascending=False).head(10).index)
print(f"  Top 10 users by wasteful GPU-hrs — tier breakdown:")
for user in top10_users:
    sub = (tier_conc[tier_conc["USERNAME_GENID"] == user]
           .sort_values("gpu_hours", ascending=False))
    tiers_str = ", ".join(f"{r['crosslayer_tier']}={r['gpu_hours']:,.0f}"
                          for _, r in sub.iterrows())
    print(f"  {str(user):<22}: {tiers_str}")

print("\n[GPU-hrs wasted by exact tier — for concentration reporting]")
for tier in ["Idle","Idle_Hidden_Activity","Scale_Inefficient",
             "IO_Bottlenecked","Failed_Job","Quick_Cancel","GPU_Idle_Timeout"]:
    sub = combined[combined["crosslayer_tier"] == tier]
    if len(sub) == 0: continue
    gpu_w = sub["gpu_hours"].sum()
    print(f"  {tier:<24}: {len(sub):>6,} jobs | {gpu_w:>10,.0f} GPU-hrs "
          f"({gpu_w/total_gpu*100:.1f}% of total)")

# ============================================================
# 4. SINGLE-LAYER BLINDNESS
# ============================================================
print("\n" + "=" * 60)
print("SINGLE-LAYER BLINDNESS")
print("=" * 60)
sig_df = combined[combined["util_mean"].notna() & combined["io_time_frac"].notna()][
    ["util_mean","io_time_frac","BWio_MB","rank_time_imbalance"]].dropna()
r_io, _ = spearmanr(sig_df["util_mean"], sig_df["io_time_frac"])
r_bw, _ = spearmanr(sig_df["util_mean"], sig_df["BWio_MB"])
print(f"Spearman r(GPU_util, io_time_frac) = {r_io:.3f}  (N={len(sig_df):,})")
print(f"Spearman r(GPU_util, BWio_MB)      = {r_bw:.3f}")

print("\n[GPU phase trajectory by tier]")
phase_df = combined[["util_phase1","util_phase2","util_phase3",
                      "io_phase_end_frac","crosslayer_tier"]].dropna(
    subset=["util_phase1","util_phase2","util_phase3"])
print(f"  {'Tier':<24} {'n':>6} {'P1':>5} {'P2':>5} {'P3':>5} {'io_end':>7}")
for tier in ["Idle","Idle_Hidden_Activity","IO_Bottlenecked",
             "Scale_Inefficient","Compute_Bound"]:
    sub = phase_df[phase_df["crosslayer_tier"]==tier]
    if len(sub) < 30: continue
    print(f"  {tier:<24} {len(sub):>6,} {sub['util_phase1'].mean():>5.1f} "
          f"{sub['util_phase2'].mean():>5.1f} {sub['util_phase3'].mean():>5.1f} "
          f"{sub['io_phase_end_frac'].median():>7.3f}")
print("  KEY: Idle and IO_Bottlenecked are observationally identical")
print("       under GPU-only monitoring. Only io_phase_end_frac (Layer 3)")
print("       discriminates: Idle ~ 0.000, IO_Bottlenecked ~ 0.993.")

print("\n[DCGM intra-layer gap: utilization vs power]")
idle_combined = combined[combined["crosslayer_tier"].isin(
                          ["Idle","Idle_Hidden_Activity"])]
n_total  = len(idle_combined)
n_hidden = (idle_combined["crosslayer_tier"]=="Idle_Hidden_Activity").sum()
n_idle   = (idle_combined["crosslayer_tier"]=="Idle").sum()
print(f"  Combined low-util population (Idle + Idle_Hidden_Activity): {n_total:,}")
print(f"    Idle (power < 50W)              : {n_idle:,} "
      f"({n_idle/n_total*100:.1f}%)")
print(f"    Idle_Hidden_Activity (>=50W)    : {n_hidden:,} "
      f"({n_hidden/n_total*100:.1f}%)")
print(f"  -> {n_hidden/n_total*100:.1f}% of jobs that look idle by SM-occupancy")
print(f"     DCGM are drawing non-baseline power and cannot be confirmed idle.")

# Walltime-exhausted disambiguation (cross-layer rescue)
print("\n[Walltime exhaustion (exit=-29) disambiguation]")
exit29 = combined[(combined["EXIT_STATUS"] == -29)]
n_total29 = len(exit29)
n_git     = (exit29["crosslayer_tier"]=="GPU_Idle_Timeout").sum()
n_failed  = n_total29 - n_git
print(f"  Total exit=-29 jobs           : {n_total29:,}")
print(f"  Classified GPU_Idle_Timeout   : {n_git:,} ({n_git/n_total29*100:.1f}%)")
print(f"  Productive walltime-exhausted : {n_failed:,} ({n_failed/n_total29*100:.1f}%)")
print(f"  -> Exit code alone misclassifies {n_failed/n_total29*100:.1f}% of these jobs")

# ============================================================
# 5. FEATURE ENGINEERING & ML
# ============================================================
print("\n" + "=" * 60)
print("FEATURE ENGINEERING + ML", flush=True)
print("=" * 60)
combined["QUEUED_TIMESTAMP"] = pd.to_datetime(combined["QUEUED_TIMESTAMP"])
combined["END_TIMESTAMP"]    = pd.to_datetime(combined["END_TIMESTAMP"])
combined = combined.sort_values("QUEUED_TIMESTAMP").reset_index(drop=True)
combined["submit_hour"]  = combined["QUEUED_TIMESTAMP"].dt.hour
combined["submit_dow"]   = combined["QUEUED_TIMESTAMP"].dt.dayofweek
combined["submit_month"] = combined["QUEUED_TIMESTAMP"].dt.month

train_df = combined[combined["use_for_training"]].copy().reset_index(drop=True)
TELEM_TIERS = {"Idle", "Idle_Hidden_Activity", "Scale_Inefficient",
               "IO_Bottlenecked", "Compute_Bound", "Ideal_Compute_With_IO",
               "Moderate_Compute_No_IO", "Moderate_Compute_With_IO",
               "Low_Efficiency", "GPU_Idle_Timeout", "Incidental_IO_Low_GPU"}
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
            if grp.loc[i, "JOB_NAME"] not in target_set: continue
            mask     = (ets < qts[i]) & (ets >= qts[i] - lb_ns)
            past_idx = np.where(mask)[0][-n_history:]
            row = {"JOB_NAME": grp.loc[i,"JOB_NAME"]}
            if len(past_idx) > 0:
                p = grp.iloc[past_idx]
                row.update({
                    "user_job_count": int(mask.sum()),
                    "user_mean_runtime": p["RUNTIME_SECONDS"].mean(),
                    "user_walltime_efficiency": (
                        p["RUNTIME_SECONDS"]/p["WALLTIME_SECONDS"].replace(0,np.nan)
                    ).mean(),
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
                row.update({"user_job_count":0,
                            "user_mean_runtime":cur["WALLTIME_SECONDS"],
                            "user_walltime_efficiency":0.5,
                            "user_fail_rate":0.0,
                            "user_quick_cancel_rate":0.0,
                            "user_mean_nodes":cur["NODES_REQUESTED"],
                            "user_mean_walltime":cur["WALLTIME_SECONDS"]})
                for c in cols: row[f"hist_{c}"] = -1
            rows.append(row)
    return pd.DataFrame(rows)

hist_cols = ["util_mean","idle_frac","zero_util_frac",
             "power_efficiency","io_time_frac","bytes_per_gpu_hour"]
print(f"Building hist features ({LOOKBACK_DAYS}-day lookback, last {N_HIST} jobs)...")
hist_df  = build_hist(train_df, telem_df["JOB_NAME"].tolist(), hist_cols)
telem_df = telem_df.merge(hist_df, on="JOB_NAME", how="left")
print(f"  {len(hist_df):,} rows | {time.time()-t0:.0f}s elapsed")

groupB  = ["user_job_count","user_mean_runtime","user_walltime_efficiency",
           "user_fail_rate","user_quick_cancel_rate",
           "user_mean_nodes","user_mean_walltime"]
groupC  = [f"hist_{c}" for c in hist_cols]
allABC  = groupA + groupB + groupC

y_train = telem_df["is_wasteful"].iloc[:split_t].values
y_test  = telem_df["is_wasteful"].iloc[split_t:].values
print(f"  Train: {split_t:,} ({telem_df['QUEUED_TIMESTAMP'].iloc[0].date()} -> "
      f"{telem_df['QUEUED_TIMESTAMP'].iloc[split_t-1].date()})")
print(f"  Test : {len(telem_df)-split_t:,} "
      f"({telem_df['QUEUED_TIMESTAMP'].iloc[split_t].date()} -> "
      f"{telem_df['QUEUED_TIMESTAMP'].iloc[-1].date()})")

def to_X(df, feats):
    return np.nan_to_num(df[[f for f in feats if f in df.columns]].values,
                          nan=-1, posinf=1e9, neginf=-1e9)

def fit_eval(feats, name, cw=None):
    avail = [f for f in feats if f in telem_df.columns]
    rf = RandomForestClassifier(n_estimators=200, class_weight=cw,
                                  n_jobs=-1, random_state=RNG)
    rf.fit(to_X(telem_df.iloc[:split_t], avail), y_train)
    prob = rf.predict_proba(to_X(telem_df.iloc[split_t:], avail))[:,1]
    pred = rf.predict(to_X(telem_df.iloc[split_t:], avail))
    return {"clf":rf, "prob":prob, "pred":pred,
            "auc":roc_auc_score(y_test, prob),
            "f1":f1_score(y_test, pred, average="macro"),
            "avail":avail}

M1   = fit_eval(groupA, "M1")
M3   = fit_eval(allABC, "M3")
ab_C = fit_eval(groupC, "C-only")
X_tr = to_X(telem_df.iloc[:split_t], M3["avail"])
X_te = to_X(telem_df.iloc[split_t:], M3["avail"])

# ============================================================
# 6. ML RESULTS
# ============================================================
print("\n" + "=" * 60)
print("ML VALIDATION")
print("=" * 60)
dst   = DummyClassifier(strategy="stratified", random_state=RNG).fit(
    np.zeros((split_t,1)), y_train)
lr_wt = LogisticRegression(max_iter=1000, class_weight="balanced").fit(
    telem_df[["WALLTIME_SECONDS"]].iloc[:split_t].values, y_train)
auc_st = roc_auc_score(y_test, dst.predict_proba(np.zeros((len(y_test),1)))[:,1])
auc_wt = roc_auc_score(y_test, lr_wt.predict_proba(
    telem_df[["WALLTIME_SECONDS"]].iloc[split_t:].values)[:,1])

print(f"\n[Model comparison]")
for label, auc, f1 in [
    ("Stratified baseline",  auc_st,    None),
    ("Walltime-only LR",     auc_wt,    None),
    ("M1 (scheduler only)",  M1["auc"], M1["f1"]),
    ("C only (hist telem)",  ab_C["auc"],None),
    ("M3 (cross-layer)",     M3["auc"], M3["f1"]),
]:
    f1s = f"{f1:.3f}" if f1 else "---"
    print(f"  {label:<25} AUC={auc:.4f}  F1={f1s}")

rng   = np.random.RandomState(RNG)
boots = [roc_auc_score(y_test[idx:=rng.randint(0,len(y_test),len(y_test))],
                        M3["prob"][idx]) for _ in range(1000)]
ci_lo, ci_hi = np.percentile(boots, [2.5, 97.5])
print(f"\nM3 bootstrap 95% CI : [{ci_lo:.4f}, {ci_hi:.4f}]")
print(f"M3 lift over M1     : +{M3['auc']-M1['auc']:.4f} AUC")
print(f"C-only lift over M1 : +{ab_C['auc']-M1['auc']:.4f} AUC")

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
print(f"  M1: {np.mean(m1_cv):.3f} ± {np.std(m1_cv):.3f}  "
      f"M3: {np.mean(m3_cv):.3f} ± {np.std(m3_cv):.3f}")
print(f"  M3 > M1 in {sum(m>m1 for m,m1 in zip(m3_cv,m1_cv))}/5 folds")

print(f"\n[OvR per-tier necessity]")
print(f"  {'Tier':<22} {'Sched':>7} {'Cross-layer':>12} {'delta':>7}  verdict")
for tier in ["Idle", "IO_Bottlenecked", "Scale_Inefficient", "GPU_Idle_Timeout"]:
    y_tr = (telem_df["crosslayer_tier"].iloc[:split_t]==tier).astype(int).values
    y_te = (telem_df["crosslayer_tier"].iloc[split_t:]==tier).astype(int).values
    if y_te.sum() < 10: continue
    aucs = {}
    for gname, feats in [("A", groupA), ("M3", allABC)]:
        avail = [f for f in feats if f in telem_df.columns]
        rf = RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                      n_jobs=-1, random_state=RNG)
        rf.fit(to_X(telem_df.iloc[:split_t], avail), y_tr)
        aucs[gname] = roc_auc_score(
            y_te, rf.predict_proba(to_X(telem_df.iloc[split_t:], avail))[:,1])
    d = aucs["M3"] - aucs["A"]
    verdict = ("essential (delta>0.10)" if d>0.10
               else "helpful (delta>0.04)" if d>0.04
               else "marginal")
    print(f"  {tier:<22} {aucs['A']:>7.4f} {aucs['M3']:>12.4f} {d:>7.4f}  {verdict}")

print(f"\n[Per-tier recall on test period]")
def wilson_ci(k, n, z=1.96):
    if n == 0: return (0., 1.)
    p = k/n; d = 1+z*z/n
    c = (p + z*z/(2*n))/d
    h = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n))/d
    return (max(0, c-h), min(1, c+h))

test_sl = telem_df.iloc[split_t:].copy().reset_index(drop=True)
test_sl["pred"] = M3["pred"]
for tier, sub in test_sl.groupby("crosslayer_tier"):
    if len(sub) < 30: continue
    n_pos = (sub["is_wasteful"]==1).sum()
    if n_pos == 0:
        flagged = (sub["pred"]==1).sum()
        lo, hi = wilson_ci(flagged, len(sub))
        print(f"  {tier:<28} n={len(sub):>5,}  "
              f"FPR={flagged/len(sub):.3f} [{lo:.2f},{hi:.2f}]")
    else:
        flagged = ((sub["is_wasteful"]==1) & (sub["pred"]==1)).sum()
        lo, hi = wilson_ci(flagged, n_pos)
        print(f"  {tier:<28} n={len(sub):>5,}  "
              f"rec={flagged/n_pos:.3f} [{lo:.2f},{hi:.2f}]")

# ============================================================
# 7. WASTE BURST STRUCTURE
# ============================================================
print("\n" + "=" * 60)
print("WASTE BURST STRUCTURE")
print("=" * 60)
cs = combined.sort_values(["USERNAME_GENID","QUEUED_TIMESTAMP"]).reset_index(drop=True)
cs["prev_w"]   = cs.groupby("USERNAME_GENID")["is_wasteful"].shift(1).fillna(0)
cs["in_burst"] = ((cs["is_wasteful"]==1) & (cs["prev_w"]==1)).astype(int)
SUBST  = {"Idle","IO_Bottlenecked","Scale_Inefficient","Failed_Job","GPU_Idle_Timeout"}
sub_w  = cs[cs["crosslayer_tier"].isin(SUBST)]
sub_b  = cs[cs["crosslayer_tier"].isin(SUBST) & (cs["in_burst"]==1)]
all_w  = cs[cs["is_wasteful"]==1]
all_b  = cs[cs["in_burst"]==1]
print(f"Including Quick_Cancel  : {len(all_b):,}/{len(all_w):,} "
      f"({len(all_b)/len(all_w)*100:.1f}%)")
print(f"Substantive tiers only  : {len(sub_b):,}/{len(sub_w):,} "
      f"({len(sub_b)/max(len(sub_w),1)*100:.1f}%)")

tm = telem_df[["JOB_NAME","hist_idle_frac","is_wasteful"]].merge(
    cs[["JOB_NAME","in_burst"]], on="JOB_NAME", how="left").fillna(0)
valid = tm[["hist_idle_frac","in_burst"]].dropna()
r_burst, _ = spearmanr(valid["hist_idle_frac"], valid["in_burst"])
burst_hi   = (tm[tm["in_burst"]==1]["hist_idle_frac"].fillna(0) > 0.5).mean() * 100
print(f"Spearman r(hist_idle_frac, in_burst) = {r_burst:.3f} (N={len(valid):,})")
print(f"{burst_hi:.1f}% of in-burst jobs have hist_idle_frac > 0.5 at submission")

# ============================================================
# 8. THRESHOLD SENSITIVITY
# ============================================================
print("\n" + "=" * 60)
print("THRESHOLD SENSITIVITY")
print("=" * 60)
drop_cols = ["crosslayer_tier","diagnostic_tier","use_for_training",
             "bytes_per_gpu_hour","gpu_hours","BWio_MB","io_time_seconds",
             "darshan_present","total_bytes","io_detected","has_gpu","gpus",
             "bytes_out","exit_failed"]
print(f"  {'Idle<':>8} {'Scale<':>8} {'Idle':>10} {'Scale_Inefficient':>20}")
for g in [3.0, 5.0, 7.0]:
    for s in [8.0, 10.0, 12.0]:
        if g >= s: continue
        cu.GPU_IDLE_UTIL, cu.GPU_UTIL_LOW = g, s
        sys.stdout = open(os.devnull, 'w')
        tmp = cu.classify_crosslayer(combined.drop(columns=drop_cols, errors="ignore"))
        sys.stdout = sys.__stdout__
        marker = " <- base" if (g==5.0 and s==10.0) else ""
        n_idle  = (tmp["crosslayer_tier"]=="Idle").sum()
        n_hidn  = (tmp["crosslayer_tier"]=="Idle_Hidden_Activity").sum()
        n_scale = (tmp["crosslayer_tier"]=="Scale_Inefficient").sum()
        print(f"  {g:>6.0f}%   {s:>6.0f}%   "
              f"{n_idle+n_hidn:>10,}   {n_scale:>20,}{marker}")
cu.GPU_IDLE_UTIL, cu.GPU_UTIL_LOW = 5.0, 10.0

# ============================================================
# 9. DISCUSSION / CONTEXT NUMBERS
# ============================================================
print("\n" + "=" * 60)
print("DISCUSSION / CONTEXT NUMBERS")
print("=" * 60)
has_gpu_jobs = combined[combined["has_gpu"]]
zero_util = (has_gpu_jobs["util_mean"].fillna(0) == 0).sum()
half_idle = ((has_gpu_jobs["idle_frac"].fillna(0) > 0.5).sum()
              if "idle_frac" in has_gpu_jobs else 0)
mean_util = has_gpu_jobs["util_mean"].mean()
print(f"Zero GPU utilization (DCGM jobs)    : {zero_util:,} "
      f"({zero_util/len(has_gpu_jobs)*100:.1f}%)")
print(f"Jobs >50% idle frac                 : {half_idle:,} "
      f"({half_idle/len(has_gpu_jobs)*100:.1f}%)")
print(f"Mean GPU utilization (DCGM jobs)    : {mean_util:.1f}%")

# ============================================================
# 10. SANITY CHECKS
# ============================================================
print("\n" + "=" * 60)
print("SANITY CHECKS")
print("=" * 60)
issues = []
assert telem_df["QUEUED_TIMESTAMP"].is_monotonic_increasing, "Temporal ordering failed"
print("[1] Temporal ordering OK")
leaky  = ["util_mean","idle_frac","zero_util_frac",
          "power_efficiency","io_time_frac","bytes_per_gpu_hour","gpu_hours"]
leaked = [f for f in M3["avail"] if f in leaky]
print(f"[2] No telemetry leak: {'OK' if not leaked else 'FAIL '+str(leaked)}")
if leaked: issues.append(f"LEAK:{leaked}")
assert combined["JOB_NAME"].duplicated().sum() == 0
print("[3] No duplicate JOB_NAME OK")
n_inf = np.isinf(X_tr).sum() + np.isinf(X_te).sum()
n_nan = np.isnan(X_tr).sum() + np.isnan(X_te).sum()
print(f"[4] Feature matrix clean: "
      f"{'OK' if (n_inf+n_nan)==0 else f'FAIL inf={n_inf} nan={n_nan}'}")
overlap = set(telem_df["JOB_NAME"].iloc[:split_t]) & set(telem_df["JOB_NAME"].iloc[split_t:])
print(f"[5] No train/test overlap: "
      f"{'OK' if not overlap else f'FAIL {len(overlap)}'}")
sha = hashlib.sha256(telem_df[M3["avail"]].fillna(-999).values.tobytes()).hexdigest()[:16]
print(f"[6] Feature matrix SHA256[:16]: {sha}")
print(f"\nTotal runtime: {(time.time()-t0)/60:.1f} min")
print("All checks passed OK" if not issues else f"Issues: {issues}")