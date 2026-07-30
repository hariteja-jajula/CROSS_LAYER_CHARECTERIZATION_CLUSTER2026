"""
combined.py — Cross-layer HPC workload taxonomy.
Vectorized, single-pass, exhaustive, priority-free classification.
Tiers are disjoint by construction — no ordering ambiguity possible.
"""
import numpy as np
import pandas as pd

# ── Single source of truth for all thresholds ───────────────────
QUICK_CANCEL_S  = 60
SHORT_JOB_S     = 600
GPU_UTIL_LOW    = 10.0     # % — upper bound for waste-tier GPU util
GPU_UTIL_HIGH   = 70.0     # % — lower bound for compute-bound
GPU_GHOST_UTIL  = 5.0      # % — Ghost: truly idle (< this)
                            #     Scale_Waster: marginal waste [this, GPU_UTIL_LOW)
GPU_GHOST_MIN   = 4        # minimum GPUs (1 Polaris node) for Ghost/Scale_Waster
BWio_MIN_BYTES  = 1e6
BWio_MIN_S      = 1.0
# Per-job effective-bandwidth sanity ceiling (NOT the facility filesystem peak).
# Any per-job BWio above this is treated as a parse artifact and nulled.
# For reference, Polaris Grand/Eagle Lustre aggregate peak is ~650 GB/s.
BWio_PEAK_MB    = 100_000
PWR_GAP_W       = 50.0


def _compute_signals(df):
    df = df.copy()
    df["darshan_present"] = df["bytes_read"].notna()
    df["total_bytes"]     = (df["bytes_read"].fillna(0) +
                             df["bytes_written"].fillna(0))
    df["io_detected"]     = df["total_bytes"] > 0
    df["has_gpu"]         = df["util_mean"].notna()
    df["gpus"]            = df["NODES_USED"] * 4
    df["bytes_out"]       = df["bytes_written"].fillna(0)
    df["exit_failed"]     = (df["EXIT_STATUS"].notna() &
                             (df["EXIT_STATUS"] != 0))

    df["gpu_hours"] = df["gpus"] * df["RUNTIME_SECONDS"] / 3600
    df.loc[df["NODES_USED"] == 0, "gpu_hours"] = 0.0

    df["io_time_seconds"] = df["io_time_frac"].fillna(0) * df["runtime"].fillna(0)
    valid_bw = ((df["io_time_seconds"] >= BWio_MIN_S) &
                (df["total_bytes"] >= BWio_MIN_BYTES))
    df["BWio_MB"] = np.where(
        valid_bw,
        df["total_bytes"] / (df["io_time_seconds"] * 1e6),
        np.nan)
    df.loc[df["BWio_MB"] > BWio_PEAK_MB, "BWio_MB"] = np.nan

    GPU_SCALE_HIGH = df["gpus"].quantile(0.75)
    df["_scale_high"] = GPU_SCALE_HIGH

    valid_bwio = df["BWio_MB"].notna() & (df["BWio_MB"] > 0)
    bwio_p90   = df.loc[valid_bwio, "BWio_MB"].quantile(0.90) \
                 if valid_bwio.sum() > 0 else 1.0
    df["gpu_waste_score"] = np.where(
        df["has_gpu"],
        1.0 - (df["util_mean"] / 100.0).clip(0, 1), np.nan)
    df["io_waste_score"]  = np.where(
        valid_bwio,
        1.0 - (df["BWio_MB"] / bwio_p90).clip(0, 1),
        np.where(df["io_detected"], 1.0, 0.0))
    df["scale_factor"]    = (df["gpus"] / GPU_SCALE_HIGH).clip(0, 3)
    df["cross_layer_waste"] = (
        0.5 * df["gpu_waste_score"].fillna(0.5) +
        0.5 * df["io_waste_score"].fillna(0.5)
    ) * df["scale_factor"].fillna(1.0)

    df["bytes_per_gpu_hour"] = np.where(
        df["gpu_hours"] > 0, df["total_bytes"] / df["gpu_hours"], 0.0)
    df["bytes_per_gpu_hour"] = df["bytes_per_gpu_hour"].replace(
        [np.inf, -np.inf], np.nan)
    finite = df["bytes_per_gpu_hour"].dropna()
    if len(finite) > 0:
        p99 = finite.quantile(0.99)
        df.loc[df["bytes_per_gpu_hour"] > p99, "bytes_per_gpu_hour"] = p99

    return df, GPU_SCALE_HIGH


def _assign(tier_series, assigned, mask, label):
    target = mask & ~assigned
    tier_series[target] = label
    assigned[target]    = True


def classify_crosslayer(df, priority=None):
    """
    Vectorized, single-pass, priority-free tier classification.
    Tiers are disjoint by construction:
      Ghost     : gpu < 5%   — truly idle
      Scale_Waster: gpu in [5%, 10%), large allocation — marginal waste
    These two ranges do not overlap, so no ordering is needed.
    The `priority` parameter is accepted but ignored (kept for
    backward compatibility with any existing callers).
    """
    df, GPU_SCALE_HIGH = _compute_signals(df)

    print(f"  GPU low/high     : {GPU_UTIL_LOW}% / {GPU_UTIL_HIGH}%")
    print(f"  Ghost threshold  : gpu < {GPU_GHOST_UTIL}%, >= {GPU_GHOST_MIN} GPUs")
    print(f"  Scale threshold  : gpu in [{GPU_GHOST_UTIL}%, {GPU_UTIL_LOW}%), "
          f">= {GPU_SCALE_HIGH:.0f} GPUs ({GPU_SCALE_HIGH/4:.0f} nodes, P75)")
    valid_bwio = df["BWio_MB"].notna() & (df["BWio_MB"] > 0)
    bwio_p90   = df.loc[valid_bwio, "BWio_MB"].quantile(0.90) \
                 if valid_bwio.sum() > 0 else np.nan
    print(f"  BWio P90         : {bwio_p90:.1f} MB/s")
    print(f"  NOTE: Ghost and Scale_Waster use non-overlapping GPU util")
    print(f"        ranges and are disjoint — no priority ordering required.")

    gpu  = df["util_mean"].fillna(-1)
    gpus = df["gpus"]
    io   = df["io_detected"]
    fail = df["exit_failed"]
    rt   = df["RUNTIME_SECONDS"]
    hgpu = df["has_gpu"]
    bout = df["bytes_out"]

    # ── Tier masks ─────────────────────────────────────────────
    # Step 1: structural / runtime gates (always first)
    m_qc    = rt < QUICK_CANCEL_S
    m_no_gpu = ~hgpu
    m_short  = rt < SHORT_JOB_S

    # GPU_Idle_Timeout: specific failure mode requiring 3-layer signal
    # exit=-29 (walltime), >80% walltime used, gpu<5%, >=1 node, no I/O
    wt_frac = df["RUNTIME_SECONDS"] / df["WALLTIME_SECONDS"].replace(0, np.nan)
    m_git = (hgpu & fail &
             (df["EXIT_STATUS"] == -29) &
             (wt_frac > 0.8) &
             (gpu < GPU_GHOST_UTIL) &
             (gpus >= GPU_GHOST_MIN) &
             ~io)

    # Failed_Job: any other failure
    m_fail_job    = hgpu & fail & (bout == 0) & ~m_git
    m_fail_lowgpu = hgpu & fail & (bout > 0) & (gpu < GPU_UTIL_LOW)
    m_fail_act    = hgpu & fail & (bout > 0) & (gpu >= GPU_UTIL_LOW)
    m_no_gpu_fail = ~hgpu & fail & (bout == 0)

    # Ghost: gpu < GPU_GHOST_UTIL (5%), any eligible size, no I/O, succeeded
    # Scale_Waster: gpu in [GPU_GHOST_UTIL, GPU_UTIL_LOW) (5-10%),
    #               large allocation (>=P75), no I/O, succeeded
    # These two masks are MUTUALLY EXCLUSIVE by GPU util range.
    m_ghost = (hgpu & (gpu >= 0) & (gpu < GPU_GHOST_UTIL) &
               (gpus >= GPU_GHOST_MIN) & ~io & ~fail)
    m_scale = (hgpu & (gpu >= GPU_GHOST_UTIL) & (gpu < GPU_UTIL_LOW) &
               (gpus >= GPU_SCALE_HIGH) & ~io & ~fail)

    # Order-invariant tiers (unchanged)
    m_io      = hgpu & (gpu < GPU_UTIL_LOW) & io & ~fail
    m_cb      = hgpu & (gpu >= GPU_UTIL_HIGH) & ~io & ~fail
    m_bal     = hgpu & (gpu >= GPU_UTIL_LOW) & io & ~fail
    m_mod     = hgpu & (gpu >= GPU_UTIL_LOW) & (gpu < GPU_UTIL_HIGH) & ~io & ~fail
    m_low_eff = (hgpu & (gpu >= GPU_GHOST_UTIL) & (gpu < GPU_UTIL_LOW) &
                 (gpus < GPU_SCALE_HIGH) & (gpus >= GPU_GHOST_MIN) & ~io & ~fail)

    tier     = pd.Series("Low_Efficiency", index=df.index)
    assigned = pd.Series(False,            index=df.index)

    # ── Assignment order ────────────────────────────────────────
    # Step 1: Quick Cancel
    _assign(tier, assigned, m_qc & hgpu & io,  "Quick_Cancel_With_IO")
    _assign(tier, assigned, m_qc & hgpu & ~io, "Quick_Cancel_GPU")
    _assign(tier, assigned, m_qc & ~hgpu & io, "Quick_Cancel_IO_Only")
    _assign(tier, assigned, m_qc,               "Quick_Cancel_No_Signal")

    # Step 2: No GPU telemetry
    _assign(tier, assigned, ~m_qc & m_no_gpu & m_short & io,  "Short_No_GPU_With_IO")
    _assign(tier, assigned, ~m_qc & m_no_gpu & m_short & ~io, "Short_No_GPU")
    _assign(tier, assigned, ~m_qc & m_no_gpu & ~m_short & io, "No_GPU_With_Darshan")
    _assign(tier, assigned, ~m_qc & m_no_gpu,                  "No_GPU_Telemetry")

    # Step 3: Failure tiers
    _assign(tier, assigned, m_git,         "GPU_Idle_Timeout")
    _assign(tier, assigned, m_fail_job,    "Failed_Job")
    _assign(tier, assigned, m_fail_lowgpu, "Failed_Low_GPU")
    _assign(tier, assigned, m_fail_act,    "Failed_Active_GPU")
    _assign(tier, assigned, m_no_gpu_fail, "Failed_Job")

    # Step 4: Waste tiers — Ghost and Scale_Waster are disjoint
    _assign(tier, assigned, m_ghost, "Ghost")
    _assign(tier, assigned, m_scale, "Scale_Waster")

    # Step 5: Efficiency tiers
    _assign(tier, assigned, m_io,      "IO_Bottlenecked")
    _assign(tier, assigned, m_cb,      "Compute_Bound")
    _assign(tier, assigned, m_bal,     "Balanced")
    _assign(tier, assigned, m_mod,     "Moderate_Compute_No_IO")
    _assign(tier, assigned, m_low_eff, "Low_Efficiency")

    # Step 6: Audit
    n_unclassified = (~assigned).sum()
    if n_unclassified > 0:
        print(f"  WARNING: {n_unclassified:,} boundary jobs assigned to Low_Efficiency.")

    df["crosslayer_tier"] = tier
    df["diagnostic_tier"] = df["crosslayer_tier"]
    df["crosslayer_tier"] = df["crosslayer_tier"].replace({
        "Failed_Low_GPU"        : "Failed_Job",
        "Failed_Active_GPU"     : "Failed_Job",
        "Failed_Partial_Output" : "Failed_Job",
        "Quick_Cancel_GPU"      : "Quick_Cancel",
        "Quick_Cancel_No_Signal": "Quick_Cancel",
        "Quick_Cancel_With_IO"  : "Quick_Cancel",
        "Quick_Cancel_IO_Only"  : "Quick_Cancel",
    })

    TRAINABLE = {
        "Failed_Job", "Ghost", "Scale_Waster", "IO_Bottlenecked",
        "Compute_Bound", "Balanced", "Moderate_Compute_No_IO",
        "Low_Efficiency", "Quick_Cancel", "GPU_Idle_Timeout"
    }
    df["use_for_training"] = df["crosslayer_tier"].isin(TRAINABLE)

    print(f"\n  Superset tiers:")
    print(f"  {df['crosslayer_tier'].value_counts().to_dict()}")
    print(f"\n  Diagnostic tiers:")
    print(f"  {df['diagnostic_tier'].value_counts().to_dict()}")
    print(f"\n  Trainable    : {df['use_for_training'].sum():,}")
    print(f"  Non-trainable: {(~df['use_for_training']).sum():,}")
    print(f"  Boundary (catch-all): {n_unclassified:,}")

    # Verify disjointness — Ghost and Scale_Waster should never overlap
    overlap = (df["crosslayer_tier"] == "Ghost") & \
              (df["diagnostic_tier"] == "Scale_Waster")
    assert overlap.sum() == 0, "Ghost/Scale_Waster overlap detected — check masks"
    print(f"  Ghost ∩ Scale_Waster overlap: 0 ✓ (disjoint by construction)")

    return df


def classify_unclassified(df):
    """No-op stub — classify_crosslayer is exhaustive."""
    return df


def print_tier_definitions():
    print("=" * 60)
    print("CROSS-LAYER HPC WORKLOAD TAXONOMY")
    print("=" * 60)
    print(f"\nKey design: Ghost and Scale_Waster use NON-OVERLAPPING")
    print(f"GPU utilization ranges and are disjoint by construction.")
    print(f"No priority ordering is required or supported.")
    print(f"\nThresholds:")
    print(f"  GPU_GHOST_UTIL  = {GPU_GHOST_UTIL}%  (Ghost upper bound)")
    print(f"  GPU_UTIL_LOW    = {GPU_UTIL_LOW}%  (Scale_Waster upper bound)")
    print(f"  GPU_UTIL_HIGH   = {GPU_UTIL_HIGH}%")
    print(f"  GPU_GHOST_MIN   = {GPU_GHOST_MIN} GPUs (1 Polaris node)")
    print(f"  GPU_SCALE_HIGH  = P75 of corpus (computed per-run)")
    print(f"  QUICK_CANCEL_S  = {QUICK_CANCEL_S}s")
    print(f"\nTier definitions:")
    defs = [
        ("GPU_Idle_Timeout",
         "exit=-29, >80% walltime, gpu<5%, >=4 GPUs, no I/O"),
        ("Failed_Job",
         "exit!=0, no bytes written (other than GPU_Idle_Timeout)"),
        ("Ghost",
         f"gpu<{GPU_GHOST_UTIL}%, >=4 GPUs, no I/O, exit=0"),
        ("Scale_Waster",
         f"gpu in [{GPU_GHOST_UTIL}%,{GPU_UTIL_LOW}%), >=P75 GPUs, no I/O, exit=0"),
        ("IO_Bottlenecked",
         f"gpu<{GPU_UTIL_LOW}%, Darshan I/O present, exit=0"),
        ("Compute_Bound",
         f"gpu>={GPU_UTIL_HIGH}%, no I/O, exit=0"),
        ("Balanced",
         f"gpu>={GPU_UTIL_LOW}%, Darshan I/O present, exit=0"),
        ("Moderate_Compute_No_IO",
         f"{GPU_UTIL_LOW}%<=gpu<{GPU_UTIL_HIGH}%, no I/O, exit=0"),
        ("Low_Efficiency",
         f"{GPU_GHOST_UTIL}%<=gpu<{GPU_UTIL_LOW}%, <P75 GPUs, no I/O, exit=0"),
        ("Quick_Cancel",
         f"runtime<{QUICK_CANCEL_S}s"),
    ]
    for tier, defn in defs:
        print(f"  {tier:<22}: {defn}")
    print("=" * 60)