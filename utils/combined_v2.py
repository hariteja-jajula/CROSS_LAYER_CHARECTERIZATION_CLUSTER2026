"""
combined_v2.py — Cross-layer HPC workload taxonomy (three-phase pipeline).

Phase 1 — Accounting and outcome gates (scheduler signals only):
    Quick_Cancel, No_GPU_* tiers, Failed_Job, GPU_Idle_Timeout.

Phase 2 — Base classification on the (compute, I/O) cell — 3x3 lookup:
    One label per cell of util_band x io_state. The (LOW, NONE) cell
    is left for Phase 3.

Phase 3 — Structural refinement of (LOW, NONE) using power and
    allocation size:
    Idle, Idle_Hidden_Activity, Scale_Inefficient, Low_Efficiency.

Each phase uses signals from one family. Tiers are produced exactly once.

================================================================
DOCUMENTED CHANGES from the legacy single-pass classifier
================================================================
  1. Old "Ghost" is split structurally into:
       - "Idle"               : util<5%, >=4 GPUs, power < 50W
       - "Idle_Hidden_Activity": util<5%, >=4 GPUs, power >= 50W (or no
                                  power data; conservative default)
     Idle_Hidden_Activity is EXCLUDED from is_wasteful — DCGM cannot
     confirm idleness when the device is drawing power.

  2. "Scale_Waster" renamed to "Scale_Inefficient" (rename only).

  3. "Balanced" intermediate label is removed. Phase 2 produces
     Ideal_Compute_With_IO / Moderate_Compute_With_IO directly.

  4. At HIGH and MOD utilization, "incidental I/O" (io_detected but
     io_time_frac <= 5%) collapses to the "no I/O" label rather than
     the "with I/O" label. Justification: incidental I/O at high
     utilization is genuinely incidental; calling it "with I/O"
     overweights a marginal signal.

  5. GPU_Idle_Timeout uses ~substantive_io (io_time_frac > 5%) rather
     than ~any_io (io_detected). Walltime-exhausted jobs with only
     incidental I/O are now correctly classified as GPU_Idle_Timeout
     instead of being demoted to Failed_Job.
"""
import numpy as np
import pandas as pd

# ============================================================
# Thresholds: single source of truth
# ============================================================
QUICK_CANCEL_S      = 60
SHORT_JOB_S         = 600

GPU_UTIL_LOW        = 10.0   # % — boundary between LOW and MOD bands
GPU_UTIL_HIGH       = 70.0   # % — boundary between MOD and HIGH bands
GPU_IDLE_UTIL       = 5.0    # % — Idle requires util < this within LOW

IO_TIME_SUBSTANTIVE = 0.05   # io_time_frac > 5% -> SUB; else INC/NONE

GPU_MIN_FOR_IDLE    = 4      # >= 1 Polaris node to call something Idle
PWR_GAP_W           = 50.0   # power threshold for Idle vs Hidden_Activity

WT_EXHAUST_FRAC     = 0.80   # GPU_Idle_Timeout walltime fraction
EXIT_WALLTIME       = -29

# Effective bandwidth (used by downstream IO_Bottlenecked sub-tier analysis)
BWio_MIN_BYTES      = 1e6
BWio_MIN_S          = 1.0
# Per-job effective-bandwidth sanity ceiling (NOT the facility filesystem peak).
# Any per-job BWio above this is treated as a parse artifact and nulled.
# For reference, Polaris Grand/Eagle Lustre aggregate peak is ~650 GB/s.
BWio_PEAK_MB        = 100_000


# ============================================================
# Signal derivation
# ============================================================
def _compute_signals(df):
    """Derive booleans, byte totals, gpu-hours, and bandwidth used by all phases."""
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
    df["gpu_hours"]       = df["gpus"] * df["RUNTIME_SECONDS"] / 3600
    df.loc[df["NODES_USED"] == 0, "gpu_hours"] = 0.0

    # Effective I/O bandwidth — only for jobs with substantial I/O time
    df["io_time_seconds"] = df["io_time_frac"].fillna(0) * df["runtime"].fillna(0)
    valid_bw = ((df["io_time_seconds"] >= BWio_MIN_S) &
                (df["total_bytes"] >= BWio_MIN_BYTES))
    df["BWio_MB"] = np.where(
        valid_bw,
        df["total_bytes"] / (df["io_time_seconds"] * 1e6),
        np.nan)
    df.loc[df["BWio_MB"] > BWio_PEAK_MB, "BWio_MB"] = np.nan

    df["bytes_per_gpu_hour"] = np.where(
        df["gpu_hours"] > 0, df["total_bytes"] / df["gpu_hours"], 0.0)
    df["bytes_per_gpu_hour"] = df["bytes_per_gpu_hour"].replace(
        [np.inf, -np.inf], np.nan)
    finite = df["bytes_per_gpu_hour"].dropna()
    if len(finite) > 0:
        p99 = finite.quantile(0.99)
        df.loc[df["bytes_per_gpu_hour"] > p99, "bytes_per_gpu_hour"] = p99

    return df


# ============================================================
# Three-phase classifier
# ============================================================
def classify_crosslayer(df, priority=None, verbose=True):
    """
    Three-phase cross-layer classification.

    Parameters
    ----------
    df : DataFrame
        Joined scheduler/DCGM/Darshan job records.
    priority : ignored
        Kept for backward compatibility with legacy callers.
    verbose : bool
        Print thresholds and tier counts when True.

    Returns
    -------
    df : DataFrame with added columns:
        crosslayer_tier, diagnostic_tier, use_for_training,
        plus signal columns from _compute_signals().
    """
    df = _compute_signals(df)

    rt    = df["RUNTIME_SECONDS"]
    hgpu  = df["has_gpu"]
    fail  = df["exit_failed"]
    gpus  = df["gpus"]
    util  = df["util_mean"].fillna(-1)
    power = (df["power_mean"]
             if "power_mean" in df.columns
             else pd.Series(np.nan, index=df.index))

    P75_GPUS = gpus[hgpu & ~fail].quantile(0.75)

    # --- Three-state I/O axis ---
    io_sub  = df["io_detected"] & (df["io_time_frac"].fillna(0) > IO_TIME_SUBSTANTIVE)
    io_inc  = df["io_detected"] & ~io_sub
    io_none = ~df["io_detected"]

    # --- Three-state compute axis ---
    c_high = util >= GPU_UTIL_HIGH
    c_mod  = (util >= GPU_UTIL_LOW) & ~c_high
    c_low  = (util >= 0) & (util < GPU_UTIL_LOW)

    if verbose:
        print(f"  Compute axis  : LOW < {GPU_UTIL_LOW}%  |  "
              f"MOD in [{GPU_UTIL_LOW}%, {GPU_UTIL_HIGH}%)  |  "
              f"HIGH >= {GPU_UTIL_HIGH}%")
        print(f"  I/O axis      : SUB > {IO_TIME_SUBSTANTIVE*100:.0f}% runtime  |  "
              f"INC = detected <= {IO_TIME_SUBSTANTIVE*100:.0f}%  |  "
              f"NONE = not detected")
        print(f"  Idle threshold: util < {GPU_IDLE_UTIL}%, "
              f">= {GPU_MIN_FOR_IDLE} GPUs, power tiebreaker at {PWR_GAP_W:.0f}W")
        print(f"  P75 GPUs      : {P75_GPUS:.0f}")

    tier = pd.Series("__UNSET__", index=df.index, dtype=object)

    # ============================================================
    # Phase 1: Accounting and outcome gates (scheduler-driven)
    # ============================================================
    # Quick_Cancel — runtime gate, applies regardless of any other signal
    m_qc = rt < QUICK_CANCEL_S
    tier[m_qc] = "Quick_Cancel"

    # No DCGM coverage — cannot do behavioral classification
    no_gpu = ~hgpu & ~m_qc
    tier[no_gpu & (rt <  SHORT_JOB_S) &  df["io_detected"]] = "Short_No_GPU_With_IO"
    tier[no_gpu & (rt <  SHORT_JOB_S) & ~df["io_detected"]] = "Short_No_GPU"
    tier[no_gpu & (rt >= SHORT_JOB_S) &  df["io_detected"]] = "No_GPU_With_Darshan"
    tier[no_gpu & (rt >= SHORT_JOB_S) & ~df["io_detected"]] = "No_GPU_Telemetry"

    # Failure: GPU_Idle_Timeout (specific pattern) vs Failed_Job (everything else)
    wt_frac = rt / df["WALLTIME_SECONDS"].replace(0, np.nan)
    m_git = (
        hgpu & fail & ~m_qc &
        (df["EXIT_STATUS"] == EXIT_WALLTIME) &
        (wt_frac > WT_EXHAUST_FRAC) &
        (util < GPU_IDLE_UTIL) &
        (gpus >= GPU_MIN_FOR_IDLE) &
        ~io_sub
    )
    tier[m_git] = "GPU_Idle_Timeout"
    tier[hgpu & fail & ~m_qc & (tier == "__UNSET__")] = "Failed_Job"

    phase1_done = tier != "__UNSET__"

    # ============================================================
    # Phase 2: Base classification on (compute, I/O) — 3x3 lookup
    # ============================================================
    # Lookup table:
    #              SUB                        INC                    NONE
    # HIGH         Ideal_Compute_With_IO      Compute_Bound          Compute_Bound
    # MOD          Moderate_Compute_With_IO   Moderate_Compute_No_IO Moderate_Compute_No_IO
    # LOW          IO_Bottlenecked            Incidental_IO_Low_GPU  -> Phase 3
    elig = ~phase1_done & hgpu & ~fail

    tier[elig & c_high & io_sub]              = "Ideal_Compute_With_IO"
    tier[elig & c_high & (io_inc | io_none)]  = "Compute_Bound"

    tier[elig & c_mod  & io_sub]              = "Moderate_Compute_With_IO"
    tier[elig & c_mod  & (io_inc | io_none)]  = "Moderate_Compute_No_IO"

    tier[elig & c_low  & io_sub]              = "IO_Bottlenecked"
    tier[elig & c_low  & io_inc]              = "Incidental_IO_Low_GPU"
    # (LOW, NONE) cell falls through to Phase 3.

    # ============================================================
    # Phase 3: Structural refinement of (LOW, NONE)
    # ============================================================
    refine = elig & c_low & io_none

    truly_low_util = (util >= 0) & (util < GPU_IDLE_UTIL)            # < 5%
    marginal_util  = (util >= GPU_IDLE_UTIL) & (util < GPU_UTIL_LOW) # [5%, 10%)
    big            = gpus >= GPU_MIN_FOR_IDLE
    big_alloc      = gpus >= P75_GPUS

    # Power-based split for Idle. NaN power -> conservative: Hidden_Activity
    # (we cannot confirm idleness without power evidence).
    pwr_low_confirmed = power < PWR_GAP_W   # NaN -> False

    m_idle        = refine & truly_low_util & big &  pwr_low_confirmed
    m_idle_hidden = refine & truly_low_util & big & ~pwr_low_confirmed
    m_scale       = refine & marginal_util  & big_alloc
    m_low_eff     = refine & ~m_idle & ~m_idle_hidden & ~m_scale

    tier[m_idle]        = "Idle"
    tier[m_idle_hidden] = "Idle_Hidden_Activity"
    tier[m_scale]       = "Scale_Inefficient"
    tier[m_low_eff]     = "Low_Efficiency"

    # ============================================================
    # Audit
    # ============================================================
    n_unset = (tier == "__UNSET__").sum()
    if n_unset > 0:
        if verbose:
            print(f"  WARNING: {n_unset:,} jobs unclassified -> Low_Efficiency")
        tier[tier == "__UNSET__"] = "Low_Efficiency"

    df["crosslayer_tier"] = tier
    df["diagnostic_tier"] = tier  # back-compat; same as crosslayer_tier

    TRAINABLE = {
        "Quick_Cancel", "Failed_Job", "GPU_Idle_Timeout",
        "Idle", "Idle_Hidden_Activity",
        "Scale_Inefficient",
        "IO_Bottlenecked", "Incidental_IO_Low_GPU",
        "Compute_Bound", "Ideal_Compute_With_IO",
        "Moderate_Compute_No_IO", "Moderate_Compute_With_IO",
        "Low_Efficiency",
    }
    df["use_for_training"] = df["crosslayer_tier"].isin(TRAINABLE)

    # Sanity: Idle and Idle_Hidden_Activity are disjoint by construction
    mask_idle = (tier == "Idle")
    if mask_idle.any():
        viol = (df.loc[mask_idle, "power_mean"].fillna(np.inf) >= PWR_GAP_W).sum()
        assert viol == 0, "Idle/Idle_Hidden_Activity overlap detected"

    if verbose:
        print(f"  Idle and Idle_Hidden_Activity disjoint by power threshold OK")
        print(f"\n  Tier counts:")
        for t, n in tier.value_counts().items():
            print(f"    {t:<28} {n:>8,}")
        print(f"\n  Trainable    : {df['use_for_training'].sum():,}")
        print(f"  Non-trainable: {(~df['use_for_training']).sum():,}")

    return df


def classify_unclassified(df):
    """No-op stub. classify_crosslayer is exhaustive."""
    return df


# ============================================================
# Migration helper: map a NEW tier label back to its OLD equivalent
# ============================================================
def to_legacy_label(new_tier, io_detected):
    """
    Map a new-pipeline tier label back to what the legacy classifier
    would have produced for the same job.

    Used by the equivalence check in framework_v2.py.

    Documented changes:
      Idle, Idle_Hidden_Activity                -> Ghost
      Scale_Inefficient                         -> Scale_Waster
      Compute_Bound          AND io_detected    -> Ideal_Compute_With_IO
      Moderate_Compute_No_IO AND io_detected    -> Moderate_Compute_With_IO

    Note: behavioral change #5 (GPU_Idle_Timeout using ~substantive_io
    instead of ~any_io) means a small set of jobs labeled GPU_Idle_Timeout
    in the new pipeline will have been Failed_Job in the old one. That
    diff is accounted for explicitly in the equivalence check rather
    than via this mapping.
    """
    if new_tier in ("Idle", "Idle_Hidden_Activity"):
        return "Ghost"
    if new_tier == "Scale_Inefficient":
        return "Scale_Waster"
    if new_tier == "Compute_Bound" and io_detected:
        return "Ideal_Compute_With_IO"
    if new_tier == "Moderate_Compute_No_IO" and io_detected:
        return "Moderate_Compute_With_IO"
    return new_tier


def print_tier_definitions():
    print("=" * 64)
    print("CROSS-LAYER HPC WORKLOAD TAXONOMY (THREE-PHASE PIPELINE)")
    print("=" * 64)

    print("\nPhase 1 -- Accounting and outcome gates")
    print("-" * 64)
    print(f"  Quick_Cancel           : runtime < {QUICK_CANCEL_S}s")
    print(f"  Short_No_GPU(_With_IO) : runtime < {SHORT_JOB_S}s, no DCGM")
    print(f"  No_GPU_Telemetry / No_GPU_With_Darshan : no DCGM, longer runtime")
    print(f"  GPU_Idle_Timeout       : exit={EXIT_WALLTIME}, runtime > "
          f"{WT_EXHAUST_FRAC*100:.0f}% walltime,")
    print(f"                            util < {GPU_IDLE_UTIL}%, "
          f">= {GPU_MIN_FOR_IDLE} GPUs, no substantive I/O")
    print(f"  Failed_Job             : exit != 0 (and not GPU_Idle_Timeout)")

    print("\nPhase 2 -- Base classification on (compute x I/O)")
    print("-" * 64)
    print(f"               SUB (>{IO_TIME_SUBSTANTIVE*100:.0f}%)        "
          f"INC (detected <={IO_TIME_SUBSTANTIVE*100:.0f}%)   NONE")
    print(f"  HIGH >={GPU_UTIL_HIGH}%   Ideal_Compute_With_IO   "
          f"Compute_Bound             Compute_Bound")
    print(f"  MOD  >={GPU_UTIL_LOW}%    Moderate_With_IO        "
          f"Moderate_No_IO            Moderate_No_IO")
    print(f"  LOW  <{GPU_UTIL_LOW}%     IO_Bottlenecked         "
          f"Incidental_IO_Low_GPU     -> Phase 3")

    print("\nPhase 3 -- Structural refinement of (LOW, NONE) cell")
    print("-" * 64)
    print(f"  Idle                   : util < {GPU_IDLE_UTIL}%, "
          f">= {GPU_MIN_FOR_IDLE} GPUs, power < {PWR_GAP_W:.0f}W")
    print(f"  Idle_Hidden_Activity   : util < {GPU_IDLE_UTIL}%, "
          f">= {GPU_MIN_FOR_IDLE} GPUs, power >= {PWR_GAP_W:.0f}W")
    print(f"  Scale_Inefficient      : util in [{GPU_IDLE_UTIL}%, "
          f"{GPU_UTIL_LOW}%), GPUs >= P75 of corpus")
    print(f"  Low_Efficiency         : remainder of (LOW, NONE) cell")
    print("=" * 64)