"""
stage00_validate.py — Comprehensive data validation for cross_layer_hpc_tool
Run AFTER each pipeline stage to verify data integrity.

Usage:
  python -m pipeline.stage00_validate --config config/config.json --stage gpu
  python -m pipeline.stage00_validate --config config/config.json --stage darshan
  python -m pipeline.stage00_validate --config config/config.json --stage combined
  python -m pipeline.stage00_validate --config config/config.json --stage all

Checks for:
  - Sentinel values (-1, 0, NaN) that silently corrupt downstream metrics
  - Join quality (duplication, orphan records, key mismatches)
  - Distribution anomalies (unexpected zeros, impossible values)
  - Temporal coverage gaps (missing months, sudden dropoffs)
  - Known failure modes from past pipeline bugs
  - Cross-source consistency (GPU telemetry vs DJC timestamps)

Exit codes:
  0 = all checks passed
  1 = warnings only (non-blocking issues)
  2 = critical failures (stop pipeline, investigate)
"""

import pandas as pd
import numpy as np
import json, argparse, sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True)
parser.add_argument("--stage", default="all",
                    choices=["gpu", "darshan", "combined", "all"])
args = parser.parse_args()

def load_config(path):
    with open(path) as f:
        return json.load(f)

# ── Tracking ─────────────────────────────────────────────────────────────────
warnings_count = 0
failures_count = 0

def sep(title):
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")

def PASS(msg):
    print(f"  ✅ PASS: {msg}")

def WARN(msg):
    global warnings_count
    warnings_count += 1
    print(f"  ⚠️  WARN: {msg}")

def FAIL(msg):
    global failures_count
    failures_count += 1
    print(f"  ❌ FAIL: {msg}")

def INFO(msg):
    print(f"  ℹ️  INFO: {msg}")

def check(condition, pass_msg, fail_msg, is_critical=True):
    if condition:
        PASS(pass_msg)
    elif is_critical:
        FAIL(fail_msg)
    else:
        WARN(fail_msg)


# ═════════════════════════════════════════════════════════════════════════════
# STAGE 1: DJC (scheduler metadata) validation
# ═════════════════════════════════════════════════════════════════════════════

def validate_djc(cfg):
    sep("VALIDATING DJC (scheduler metadata)")

    djc = pd.read_csv(cfg["djc_csv"], low_memory=False)
    INFO(f"Loaded {len(djc):,} rows, {len(djc.columns)} columns")

    # ── Row count ────────────────────────────────────────────────────────
    check(len(djc) > 0,
          f"{len(djc):,} jobs loaded",
          "DJC is empty")

    # ── Duplicate job IDs ────────────────────────────────────────────────
    djc['job_id'] = djc['JOB_NAME'].str.split('.').str[0]
    n_unique = djc['job_id'].nunique()
    n_total = len(djc)
    dup_rate = 1 - n_unique / n_total
    check(dup_rate < 0.01,
          f"Job ID uniqueness: {n_unique:,}/{n_total:,} ({dup_rate*100:.2f}% duplicates)",
          f"High duplicate rate: {dup_rate*100:.1f}% — check JOB_NAME parsing",
          is_critical=dup_rate > 0.05)

    if dup_rate > 0:
        dups = djc[djc.duplicated(subset='job_id', keep=False)]
        INFO(f"  Duplicate job_ids: {dups['job_id'].nunique():,} unique IDs with duplicates")
        INFO(f"  Example: {dups['job_id'].iloc[0]}")

    # ── Required columns ─────────────────────────────────────────────────
    required = ['JOB_NAME', 'START_TIMESTAMP', 'END_TIMESTAMP', 'RUNTIME_SECONDS',
                'WALLTIME_SECONDS', 'NODES_USED', 'GPUS_REQUESTED', 'EXIT_STATUS',
                'USERNAME_GENID', 'PROJECT_NAME_GENID', 'SCIENCE_FIELD_SHORT',
                'QUEUED_TIMESTAMP', 'LOCATION']
    missing = [c for c in required if c not in djc.columns]
    check(len(missing) == 0,
          f"All {len(required)} required columns present",
          f"Missing columns: {missing}")

    # ── GPUS_REQUESTED sentinel values ───────────────────────────────────
    neg1_count = (djc['GPUS_REQUESTED'] == -1).sum()
    neg1_pct = neg1_count / len(djc) * 100
    INFO(f"GPUS_REQUESTED = -1: {neg1_count:,} jobs ({neg1_pct:.1f}%)")

    if neg1_count > 0:
        # check which months have -1
        djc['month'] = pd.to_datetime(djc['START_TIMESTAMP'], errors='coerce').dt.month
        neg1_months = djc[djc['GPUS_REQUESTED'] == -1]['month'].value_counts().sort_index()
        INFO(f"  -1 by month: {dict(neg1_months)}")

        # critical: if -1 jobs have NODES_USED > 0, gpu_hours derivation is needed
        neg1_with_nodes = djc[(djc['GPUS_REQUESTED'] == -1) & (djc['NODES_USED'] > 0)]
        if len(neg1_with_nodes) > 0:
            WARN(f"  {len(neg1_with_nodes):,} jobs have GPUS_REQUESTED=-1 but NODES_USED>0")
            WARN(f"  → gpu_hours must use NODES_USED*4, not GPUS_REQUESTED")

    # ── NODES_USED sanity ────────────────────────────────────────────────
    zero_nodes = (djc['NODES_USED'] == 0).sum()
    neg_nodes = (djc['NODES_USED'] < 0).sum()
    max_nodes = djc['NODES_USED'].max()

    check(neg_nodes == 0,
          f"No negative NODES_USED",
          f"{neg_nodes:,} jobs with NODES_USED < 0")
    check(max_nodes <= 560,
          f"Max NODES_USED = {max_nodes} (Polaris has 560 nodes)",
          f"Max NODES_USED = {max_nodes} — exceeds Polaris capacity (560)",
          is_critical=False)
    INFO(f"NODES_USED = 0: {zero_nodes:,} jobs")

    # ── RUNTIME sanity ───────────────────────────────────────────────────
    neg_runtime = (djc['RUNTIME_SECONDS'] < 0).sum()
    zero_runtime = (djc['RUNTIME_SECONDS'] == 0).sum()
    max_runtime_hrs = djc['RUNTIME_SECONDS'].max() / 3600

    check(neg_runtime == 0,
          f"No negative RUNTIME_SECONDS",
          f"{neg_runtime:,} jobs with negative runtime")
    INFO(f"Zero runtime: {zero_runtime:,} jobs")
    INFO(f"Max runtime: {max_runtime_hrs:.1f} hours")
    check(max_runtime_hrs < 200,
          f"Max runtime reasonable ({max_runtime_hrs:.1f}h)",
          f"Max runtime {max_runtime_hrs:.1f}h — exceeds typical walltime limits",
          is_critical=False)

    # ── WALLTIME > RUNTIME ───────────────────────────────────────────────
    wt_lt_rt = (djc['WALLTIME_SECONDS'] < djc['RUNTIME_SECONDS']).sum()
    check(wt_lt_rt / len(djc) < 0.05,
          f"Walltime >= runtime for {(1 - wt_lt_rt/len(djc))*100:.1f}% of jobs",
          f"{wt_lt_rt:,} jobs ({wt_lt_rt/len(djc)*100:.1f}%) have WALLTIME < RUNTIME",
          is_critical=False)

    # ── Timestamp parsing ────────────────────────────────────────────────
    start_ts = pd.to_datetime(djc['START_TIMESTAMP'], errors='coerce')
    ts_null = start_ts.isna().sum()
    check(ts_null / len(djc) < 0.01,
          f"START_TIMESTAMP parseable: {(1-ts_null/len(djc))*100:.1f}%",
          f"{ts_null:,} jobs ({ts_null/len(djc)*100:.1f}%) have unparseable timestamps")

    # ── Temporal coverage ────────────────────────────────────────────────
    if ts_null < len(djc):
        monthly = start_ts.dropna().dt.to_period('M').value_counts().sort_index()
        INFO(f"Monthly job counts:")
        for month, count in monthly.items():
            print(f"    {month}: {count:>6,}")

        # check for missing months
        all_months = pd.period_range(monthly.index.min(), monthly.index.max(), freq='M')
        missing_months = set(all_months) - set(monthly.index)
        check(len(missing_months) == 0,
              f"All months covered ({len(monthly)} months)",
              f"Missing months: {sorted(missing_months)}")

        # check for sudden drops (>50% decline month-over-month)
        counts = monthly.values
        for i in range(1, len(counts)):
            if counts[i-1] > 100 and counts[i] < counts[i-1] * 0.5:
                WARN(f"  Sudden drop: {monthly.index[i-1]} ({counts[i-1]:,}) → "
                     f"{monthly.index[i]} ({counts[i]:,}) = {counts[i]/counts[i-1]*100:.0f}%")

    # ── LOCATION field (needed for GPU join) ─────────────────────────────
    loc_null = djc['LOCATION'].isna().sum()
    loc_empty = (djc['LOCATION'] == '').sum()
    INFO(f"LOCATION null: {loc_null:,}, empty: {loc_empty:,}")
    check((loc_null + loc_empty) / len(djc) < 0.3,
          f"LOCATION populated for {(1-(loc_null+loc_empty)/len(djc))*100:.1f}% of jobs",
          f"LOCATION missing for {(loc_null+loc_empty)/len(djc)*100:.1f}% — GPU join will have gaps",
          is_critical=False)

    return djc


# ═════════════════════════════════════════════════════════════════════════════
# STAGE 2: GPU telemetry validation (after stage01_parse_gpu)
# ═════════════════════════════════════════════════════════════════════════════

def validate_gpu(cfg):
    sep("VALIDATING GPU METRICS (post stage01_parse_gpu)")

    path = cfg.get("gpu_parsed_out")
    if not path or not Path(path).exists():
        WARN(f"GPU parsed output not found: {path}")
        return None

    gpu = pd.read_csv(path)
    gpu['job_id'] = gpu['JOB_NAME'].str.split('.').str[0]
    INFO(f"Loaded {len(gpu):,} rows, {len(gpu.columns)} columns")

    # ── Duplicate job IDs ────────────────────────────────────────────────
    n_unique = gpu['job_id'].nunique()
    dup_rate = 1 - n_unique / len(gpu)
    check(dup_rate < 0.01,
          f"Job ID uniqueness: {n_unique:,}/{len(gpu):,}",
          f"Duplicate rate: {dup_rate*100:.1f}%")

    # ── Telemetry row counts ─────────────────────────────────────────────
    has_rows = (gpu['gpu_telemetry_rows'] > 0).sum()
    zero_rows = (gpu['gpu_telemetry_rows'] == 0).sum()
    INFO(f"Jobs with GPU data: {has_rows:,} ({has_rows/len(gpu)*100:.1f}%)")
    INFO(f"Jobs with zero rows: {zero_rows:,}")

    # ── GPU utilization range ────────────────────────────────────────────
    has_util = gpu['gpu_util_mean'].notna()
    if has_util.sum() > 0:
        util = gpu.loc[has_util, 'gpu_util_mean']
        check(util.min() >= 0,
              f"GPU util min = {util.min():.2f}% (non-negative)",
              f"GPU util min = {util.min():.2f}% — NEGATIVE VALUES")
        check(util.max() <= 100,
              f"GPU util max = {util.max():.2f}% (≤100)",
              f"GPU util max = {util.max():.2f}% — EXCEEDS 100%")

        INFO(f"GPU util distribution: mean={util.mean():.1f}%, "
             f"median={util.median():.1f}%, P90={util.quantile(0.9):.1f}%")

        # suspicious: all zeros
        all_zero_pct = (util == 0).mean() * 100
        if all_zero_pct > 80:
            WARN(f"  {all_zero_pct:.1f}% of GPU util values are exactly 0 — verify DCGM collection")

    # ── Temperature range ────────────────────────────────────────────────
    if 'gpu_temp_mean' in gpu.columns:
        temp = gpu['gpu_temp_mean'].dropna()
        if len(temp) > 0:
            check(temp.min() >= -10,
                  f"GPU temp min = {temp.min():.1f}°C (reasonable)",
                  f"GPU temp min = {temp.min():.1f}°C — impossibly low")
            check(temp.max() <= 100,
                  f"GPU temp max = {temp.max():.1f}°C (under throttle limit)",
                  f"GPU temp max = {temp.max():.1f}°C — potential sensor error",
                  is_critical=False)

    # ── Power range ──────────────────────────────────────────────────────
    if 'gpu_power_mean' in gpu.columns:
        power = gpu['gpu_power_mean'].dropna()
        if len(power) > 0:
            check(power.min() >= 0,
                  f"GPU power min = {power.min():.1f}W (non-negative)",
                  f"GPU power min = {power.min():.1f}W — negative power")
            check(power.max() <= 500,
                  f"GPU power max = {power.max():.1f}W (under A100 TDP)",
                  f"GPU power max = {power.max():.1f}W — exceeds A100 TDP (400W)",
                  is_critical=False)

    # ── Memory allocation ────────────────────────────────────────────────
    if 'gpu_mem_alloc_mean_kb' in gpu.columns:
        mem = gpu['gpu_mem_alloc_mean_kb'].dropna()
        if len(mem) > 0:
            max_gb = mem.max() / 1e6
            # Polaris A100s are the 40 GB SXM4 model (160 GB HBM/node / 4 GPUs).
            # 45 GB gives a little headroom over the 40 GB card capacity.
            check(max_gb <= 45,
                  f"GPU mem max = {max_gb:.1f} GB (within A100 40GB HBM on Polaris)",
                  f"GPU mem max = {max_gb:.1f} GB — exceeds A100 40GB capacity",
                  is_critical=False)

    # ── Phase consistency ────────────────────────────────────────────────
    if 'gpu_util_phase1' in gpu.columns:
        p1 = gpu['gpu_util_phase1'].dropna()
        p2 = gpu['gpu_util_phase2'].dropna()
        p3 = gpu['gpu_util_phase3'].dropna()
        if len(p1) > 0:
            for name, vals in [('Phase1', p1), ('Phase2', p2), ('Phase3', p3)]:
                check(vals.min() >= 0 and vals.max() <= 100,
                      f"{name} range: [{vals.min():.1f}%, {vals.max():.1f}%]",
                      f"{name} out of range: [{vals.min():.1f}%, {vals.max():.1f}%]",
                      is_critical=False)

    return gpu


# ═════════════════════════════════════════════════════════════════════════════
# STAGE 3: DARSHAN validation (after stage02_parse_darshan)
# ═════════════════════════════════════════════════════════════════════════════

def validate_darshan(cfg):
    sep("VALIDATING DARSHAN METRICS (post stage02_parse_darshan)")

    path = cfg.get("darshan_parsed_out")
    if not path or not Path(path).exists():
        WARN(f"Darshan parsed output not found: {path}")
        return None

    darshan = pd.read_csv(path)
    darshan['job_id'] = darshan['job_id'].astype(str)
    INFO(f"Loaded {len(darshan):,} file records, {len(darshan.columns)} columns")

    n_jobs = darshan['job_id'].nunique()
    INFO(f"Unique jobs: {n_jobs:,}")
    INFO(f"Mean files per job: {len(darshan)/n_jobs:.1f}")

    # ── Negative byte counts ─────────────────────────────────────────────
    for col in ['bytes_read', 'bytes_written']:
        if col in darshan.columns:
            neg = (darshan[col] < 0).sum()
            check(neg == 0,
                  f"No negative {col}",
                  f"{neg:,} negative {col} values")

    # ── Timestamp ordering ───────────────────────────────────────────────
    if 'io_read_start' in darshan.columns and 'io_read_end' in darshan.columns:
        inverted = darshan[
            (darshan['io_read_end'] > 0) &
            (darshan['io_read_start'] > 0) &
            (darshan['io_read_end'] < darshan['io_read_start'])
        ]
        check(len(inverted) / len(darshan) < 0.05,
              f"Read timestamps ordered for {(1-len(inverted)/len(darshan))*100:.1f}% of records",
              f"{len(inverted):,} records have io_read_end < io_read_start",
              is_critical=False)

    # ── cb_nodes values ──────────────────────────────────────────────────
    if 'cb_nodes' in darshan.columns:
        cb_dist = darshan['cb_nodes'].value_counts().head(5)
        INFO(f"cb_nodes distribution: {dict(cb_dist)}")

    # ── I/O interface flags ──────────────────────────────────────────────
    for col in ['has_posix', 'has_mpiio', 'has_stdio']:
        if col in darshan.columns:
            pct = darshan[col].mean() * 100
            INFO(f"{col}: {pct:.1f}% of records")

    # ── fname uniqueness ─────────────────────────────────────────────────
    if 'fname' in darshan.columns:
        fname_unique = darshan['fname'].nunique()
        fname_dup = len(darshan) - fname_unique
        check(fname_dup / len(darshan) < 0.01,
              f"fname uniqueness: {fname_unique:,}/{len(darshan):,}",
              f"{fname_dup:,} duplicate fnames — possible re-parsing",
              is_critical=False)

    # ── Ratio sanity ─────────────────────────────────────────────────────
    for col in ['seq_read_ratio', 'seq_write_ratio', 'small_read_ratio',
                'small_write_ratio', 'write_dominance']:
        if col in darshan.columns:
            vals = darshan[col].dropna()
            if len(vals) > 0:
                check(vals.min() >= 0 and vals.max() <= 1.0001,
                      f"{col} in [0,1] range",
                      f"{col} out of range: [{vals.min():.4f}, {vals.max():.4f}]",
                      is_critical=False)

    return darshan


# ═════════════════════════════════════════════════════════════════════════════
# STAGE 4: COMBINED metrics validation (after stage03_build_combined)
# ═════════════════════════════════════════════════════════════════════════════

def validate_combined(cfg):
    sep("VALIDATING COMBINED METRICS (post stage03_build_combined)")

    path = cfg.get("combined_out")
    if not path or not Path(path).exists():
        FAIL(f"Combined output not found: {path}")
        return None

    df = pd.read_csv(path, low_memory=False)
    df['job_id'] = df['JOB_NAME'].str.split('.').str[0]
    INFO(f"Loaded {len(df):,} rows, {len(df.columns)} columns")

    # ── Row count matches DJC ────────────────────────────────────────────
    try:
        djc = pd.read_csv(cfg["djc_csv"], low_memory=False)
        check(len(df) == len(djc),
              f"Row count matches DJC: {len(df):,} = {len(djc):,}",
              f"Row count mismatch: combined={len(df):,} vs DJC={len(djc):,}")
    except Exception as e:
        WARN(f"Could not load DJC for row count check: {e}")

    # ── Join quality ─────────────────────────────────────────────────────
    has_gpu = df['gpu_util_mean'].notna().sum()
    has_darshan = df['darshan_file_count'].notna().sum()
    has_both = ((df['gpu_util_mean'].notna()) & (df['darshan_file_count'].notna())).sum()
    has_neither = ((df['gpu_util_mean'].isna()) & (df['darshan_file_count'].isna())).sum()

    INFO(f"Join coverage:")
    INFO(f"  GPU only:     {has_gpu - has_both:,}")
    INFO(f"  Darshan only: {has_darshan - has_both:,}")
    INFO(f"  Both:         {has_both:,}")
    INFO(f"  Neither:      {has_neither:,}")

    check(has_gpu / len(df) > 0.5,
          f"GPU coverage: {has_gpu/len(df)*100:.1f}% (>50%)",
          f"GPU coverage only {has_gpu/len(df)*100:.1f}% — check GPU join",
          is_critical=False)

    # ── gpu_hours validation ─────────────────────────────────────────────
    sep("gpu_hours VALIDATION (known failure point)")

    neg_gpu_hours = (df['gpu_hours'] < 0).sum()
    check(neg_gpu_hours == 0,
          f"No negative gpu_hours",
          f"{neg_gpu_hours:,} negative gpu_hours values")

    zero_gpu_hours = (df['gpu_hours'] == 0).sum()
    INFO(f"Zero gpu_hours: {zero_gpu_hours:,} ({zero_gpu_hours/len(df)*100:.1f}%)")

    # ── CRITICAL: check for the GPUS_REQUESTED=-1 → gpu_hours=0 bug ────
    df['month'] = pd.to_datetime(df['START_TIMESTAMP'], errors='coerce').dt.month
    monthly_hours = df.groupby('month')['gpu_hours'].sum()
    INFO(f"Monthly gpu_hours:")
    for month, hours in monthly_hours.items():
        flag = " ← CHECK" if hours == 0 else ""
        print(f"    Month {month:>2}: {hours:>12,.0f}{flag}")

    zero_months = (monthly_hours == 0).sum()
    check(zero_months == 0,
          f"All months have non-zero gpu_hours",
          f"{zero_months} months have ZERO gpu_hours — likely GPUS_REQUESTED=-1 bug")

    # check gpu_hours derivation method
    sample = df[df['NODES_USED'] > 0].head(100)
    expected = sample['NODES_USED'] * 4 * sample['RUNTIME_SECONDS'] / 3600
    actual = sample['gpu_hours']
    mismatch = (abs(expected - actual) > 1).sum()
    if mismatch > 0:
        WARN(f"gpu_hours derivation mismatch in {mismatch}/100 sampled rows")
        WARN(f"  Expected: NODES_USED * 4 * RUNTIME_SECONDS / 3600")
        INFO(f"  Sample expected: {expected.head(3).values}")
        INFO(f"  Sample actual:   {actual.head(3).values}")

    # ── BWio validation ──────────────────────────────────────────────────
    sep("BWio VALIDATION")

    bwio = df['BWio_MB'].dropna()
    if len(bwio) > 0:
        neg_bwio = (bwio < 0).sum()
        check(neg_bwio == 0,
              f"No negative BWio",
              f"{neg_bwio:,} negative BWio values")

        extreme_bwio = (bwio > 500000).sum()  # >500 GB/s — impossible
        check(extreme_bwio == 0,
              f"No impossible BWio values (>500 GB/s)",
              f"{extreme_bwio:,} jobs with BWio > 500 GB/s — check T_IO computation",
              is_critical=False)

        INFO(f"BWio: median={bwio.median():.1f} MB/s, P90={bwio.quantile(0.9):.1f} MB/s, "
             f"max={bwio.max():.1f} MB/s")

    # ── T_IO vs T_wall consistency ───────────────────────────────────────
    if 'T_IO' in df.columns:
        t_io = df['T_IO'].dropna()
        t_wall = df.loc[t_io.index, 'RUNTIME_SECONDS']
        tio_gt_twall = (t_io > 1.05 * t_wall).sum()
        INFO(f"T_IO > 1.05*T_wall: {tio_gt_twall:,} jobs (fallback to T_wall used)")

    # ── total_bytes validation ───────────────────────────────────────────
    if 'total_bytes' in df.columns:
        neg_bytes = (df['total_bytes'] < 0).sum()
        check(neg_bytes == 0,
              f"No negative total_bytes",
              f"{neg_bytes:,} negative total_bytes")

    # ── Tier distribution ────────────────────────────────────────────────
    sep("TIER CLASSIFICATION VALIDATION")

    tier_counts = df['crosslayer_tier'].value_counts()
    INFO(f"Tiers ({len(tier_counts)} unique):")
    for tier, count in tier_counts.items():
        print(f"    {tier:<25s} {count:>8,} ({count/len(df)*100:.1f}%)")

    unclassified = tier_counts.get('Unclassified', 0)
    check(unclassified / len(df) < 0.01,
          f"Unclassified jobs: {unclassified:,} ({unclassified/len(df)*100:.2f}%)",
          f"{unclassified:,} unclassified jobs ({unclassified/len(df)*100:.1f}%) — check classify logic",
          is_critical=False)

    # ── Ghost validation ─────────────────────────────────────────────────
    ghost = df[df['crosslayer_tier'] == 'Ghost']
    if len(ghost) > 0:
        ghost_with_io = ghost[ghost['total_bytes'].notna() & (ghost['total_bytes'] > 0)]
        check(len(ghost_with_io) == 0,
              f"All Ghost jobs have zero I/O (by definition)",
              f"{len(ghost_with_io):,} Ghost jobs have non-zero total_bytes — classification bug")

        ghost_high_gpu = ghost[ghost['gpu_util_mean'].notna() & (ghost['gpu_util_mean'] >= 5)]
        check(len(ghost_high_gpu) == 0,
              f"All Ghost jobs have <5% GPU util (by definition)",
              f"{len(ghost_high_gpu):,} Ghost jobs have ≥5% GPU util — classification bug")

    # ── IO_Bottlenecked validation ───────────────────────────────────────
    iobot = df[df['crosslayer_tier'] == 'IO_Bottlenecked']
    if len(iobot) > 0:
        iobot_no_io = iobot[iobot['total_bytes'].isna() | (iobot['total_bytes'] == 0)]
        check(len(iobot_no_io) == 0,
              f"All IO_Bottlenecked jobs have I/O data (by definition)",
              f"{len(iobot_no_io):,} IO_Bottlenecked jobs have zero I/O — classification bug")

        iobot_high_gpu = iobot[iobot['gpu_util_mean'].notna() & (iobot['gpu_util_mean'] >= 10)]
        check(len(iobot_high_gpu) == 0,
              f"All IO_Bottlenecked jobs have <10% GPU util",
              f"{len(iobot_high_gpu):,} IO_Bottlenecked jobs have ≥10% GPU util — check threshold")

    # ── Compute_Bound validation ─────────────────────────────────────────
    cb = df[df['crosslayer_tier'] == 'Compute_Bound']
    if len(cb) > 0:
        cb_low_gpu = cb[cb['gpu_util_mean'].notna() & (cb['gpu_util_mean'] < 70)]
        check(len(cb_low_gpu) == 0,
              f"All Compute_Bound jobs have ≥70% GPU util",
              f"{len(cb_low_gpu):,} Compute_Bound jobs have <70% GPU util — check threshold")

    # ── Cross-source consistency ─────────────────────────────────────────
    sep("CROSS-SOURCE CONSISTENCY")

    # GPU util vs power correlation
    has_both_gp = df[df['gpu_util_mean'].notna() & df['gpu_power_mean'].notna()]
    if len(has_both_gp) > 100:
        corr = has_both_gp['gpu_util_mean'].corr(has_both_gp['gpu_power_mean'])
        check(corr > 0.3,
              f"GPU util-power correlation: {corr:.3f} (positive, as expected)",
              f"GPU util-power correlation: {corr:.3f} — unusually low, check DCGM data",
              is_critical=False)

    # Darshan bytes vs BWio consistency
    has_bwio = df[df['BWio_MB'].notna() & (df['BWio_MB'] > 0) & (df['total_bytes'] > 0)]
    if len(has_bwio) > 0:
        # BWio = total_bytes / time, so total_bytes / BWio should give reasonable time
        implied_time = has_bwio['total_bytes'] / (has_bwio['BWio_MB'] * 1e6)
        neg_time = (implied_time < 0).sum()
        check(neg_time == 0,
              f"BWio implies non-negative I/O time",
              f"{neg_time:,} jobs with implied negative I/O time")

    # ── Science field distribution ───────────────────────────────────────
    fields = df['SCIENCE_FIELD_SHORT'].value_counts()
    null_field = df['SCIENCE_FIELD_SHORT'].isna().sum()
    INFO(f"Science fields: {len(fields)} unique, {null_field:,} null")

    return df


# ═════════════════════════════════════════════════════════════════════════════
# KNOWN BUG REGRESSION TESTS
# ═════════════════════════════════════════════════════════════════════════════

def regression_tests(df):
    sep("REGRESSION TESTS (known past bugs)")

    # BUG 1: GPUS_REQUESTED=-1 zeroing gpu_hours for Sep-Dec
    df['month'] = pd.to_datetime(df['START_TIMESTAMP'], errors='coerce').dt.month
    sep_dec_hours = df[df['month'] >= 9]['gpu_hours'].sum()
    jan_aug_hours = df[df['month'] <= 8]['gpu_hours'].sum()
    if jan_aug_hours > 0:
        ratio = sep_dec_hours / (jan_aug_hours / 8 * 4)  # expected proportional
        check(ratio > 0.3,
              f"Sep-Dec gpu_hours proportional to Jan-Aug (ratio={ratio:.2f})",
              f"Sep-Dec gpu_hours suspiciously low (ratio={ratio:.2f}) — GPUS_REQUESTED=-1 bug?")

    # BUG 2: Ghost io_phase_end_frac inflated by zero-I/O jobs
    ghost = df[df['crosslayer_tier'] == 'Ghost']
    if len(ghost) > 0:
        ghost_with_darshan = ghost[ghost['darshan_file_count'].notna()]
        ghost_without = ghost[ghost['darshan_file_count'].isna()]
        INFO(f"Ghost with Darshan: {len(ghost_with_darshan):,} ({len(ghost_with_darshan)/len(ghost)*100:.1f}%)")
        INFO(f"Ghost without Darshan: {len(ghost_without):,} ({len(ghost_without)/len(ghost)*100:.1f}%)")

        if len(ghost_with_darshan) > 0:
            io_end_all = ghost['io_phase_end_frac'].mean()
            io_end_darshan = ghost_with_darshan['io_phase_end_frac'].mean()
            if abs(io_end_all - io_end_darshan) > 0.2:
                WARN(f"Ghost io_phase_end_frac: all={io_end_all:.3f} vs darshan-only={io_end_darshan:.3f}")
                WARN(f"  → Use darshan-only value in paper, not all-Ghost average")

    # BUG 3: classify_crosslayer using GPUS_REQUESTED instead of NODES_USED*4
    sep_ghost = df[(df['month'] >= 9) & (df['crosslayer_tier'] == 'Ghost')]
    if len(sep_ghost) > 0:
        neg1_ghost = sep_ghost[sep_ghost['GPUS_REQUESTED'] == -1]
        INFO(f"Sep-Dec Ghost with GPUS_REQUESTED=-1: {len(neg1_ghost):,}/{len(sep_ghost):,}")
        # these should exist if the fix is in place
        if len(neg1_ghost) > 0:
            PASS(f"Ghost classification works for GPUS_REQUESTED=-1 jobs (fix confirmed)")
        elif len(df[(df['month'] >= 9) & (df['GPUS_REQUESTED'] == -1)]) > 0:
            WARN(f"Sep-Dec has GPUS_REQUESTED=-1 jobs but none classified as Ghost — check gate")

    # BUG 4: cpu_no_io getting nonzero gpu_hours
    cpu_no_io = df[df['crosslayer_tier'] == 'CPU_No_IO']
    if len(cpu_no_io) > 0:
        cpu_with_hours = cpu_no_io[cpu_no_io['gpu_hours'] > 0]
        if len(cpu_with_hours) > 0:
            total_cpu_hours = cpu_with_hours['gpu_hours'].sum()
            WARN(f"CPU_No_IO has {total_cpu_hours:,.0f} gpu_hours across {len(cpu_with_hours):,} jobs")
            WARN(f"  → These are GPUS_REQUESTED=-1 jobs; gpu_hours from NODES_USED*4 is cosmetically wrong")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    cfg = load_config(args.config)

    print(f"Cross-Layer HPC Tool — Data Validation")
    print(f"Stage: {args.stage}")
    print(f"Config: {args.config}")

    if args.stage in ("all", "djc"):
        validate_djc(cfg)

    if args.stage in ("all", "gpu"):
        validate_gpu(cfg)

    if args.stage in ("all", "darshan"):
        validate_darshan(cfg)

    if args.stage in ("all", "combined"):
        df = validate_combined(cfg)
        if df is not None:
            regression_tests(df)

    # ── Final summary ────────────────────────────────────────────────────
    sep("VALIDATION SUMMARY")
    print(f"  Checks passed:  (see ✅ above)")
    print(f"  Warnings:       {warnings_count}")
    print(f"  Failures:       {failures_count}")

    if failures_count > 0:
        print(f"\n  ❌ {failures_count} CRITICAL FAILURES — investigate before proceeding")
        sys.exit(2)
    elif warnings_count > 0:
        print(f"\n  ⚠️  {warnings_count} warnings — review but pipeline can proceed")
        sys.exit(1)
    else:
        print(f"\n  ✅ All checks passed")
        sys.exit(0)
