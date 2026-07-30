"""
stage09_security_audit.py — Ghost job anomaly analysis for security-relevant patterns
Run after stage03_build_combined.

Examines Ghost jobs for signals inconsistent with benign misconfiguration:
  - GPU memory allocation without compute (something loaded on GPU)
  - Elevated power draw at zero utilization (workload DCGM isn't capturing)
  - Multi-node jobs with zero I/O (network-only communication)
  - Regular submission intervals (automated/scripted submission)
  - Off-hours clustering (avoiding operator observation)
  - Thermal anomalies at zero utilization

Run: python -m pipeline.stage09_security_audit --config config/config.json
"""

import pandas as pd
import numpy as np
import json, argparse
from pathlib import Path
from collections import Counter

parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True)
args = parser.parse_args()

def load_config(path):
    with open(path) as f:
        return json.load(f)

def sep(title):
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")


# ─────────────────────────────────────────────────────────────────────────────
# 1. GPU MEMORY WITHOUT COMPUTE
# ─────────────────────────────────────────────────────────────────────────────
def memory_without_compute(ghost):
    sep("SIGNAL 1: GPU MEMORY ALLOCATION WITHOUT COMPUTE")

    print(f"Ghost jobs with GPU telemetry: {ghost['gpu_util_mean'].notna().sum():,}")
    print(f"Ghost GPU util mean:           {ghost['gpu_util_mean'].mean():.2f}%")

    # jobs where GPU memory is allocated but compute is zero
    has_mem = ghost['gpu_mem_alloc_mean_kb'].notna() & (ghost['gpu_mem_alloc_mean_kb'] > 0)
    has_zero_util = ghost['gpu_util_mean'].notna() & (ghost['gpu_util_mean'] < 1.0)

    mem_no_compute = ghost[has_mem & has_zero_util].copy()
    print(f"\nGhost jobs with >0 GPU memory AND <1% compute: {len(mem_no_compute):,}")

    if len(mem_no_compute) == 0:
        print("  No anomalous memory-without-compute pattern found")
        return pd.DataFrame()

    # memory allocation distribution among these jobs
    mem_kb = mem_no_compute['gpu_mem_alloc_mean_kb']
    print(f"\nGPU memory allocation (KB) among anomalous jobs:")
    print(f"  Median:    {mem_kb.median():,.0f} KB ({mem_kb.median()/1e6:.2f} GB)")
    print(f"  Mean:      {mem_kb.mean():,.0f} KB ({mem_kb.mean()/1e6:.2f} GB)")
    print(f"  P75:       {mem_kb.quantile(0.75):,.0f} KB ({mem_kb.quantile(0.75)/1e6:.2f} GB)")
    print(f"  P90:       {mem_kb.quantile(0.90):,.0f} KB ({mem_kb.quantile(0.90)/1e6:.2f} GB)")
    print(f"  Max:       {mem_kb.max():,.0f} KB ({mem_kb.max()/1e6:.2f} GB)")

    # A100 has 40GB or 80GB HBM. Significant allocation = >1GB
    significant_mem = mem_no_compute[mem_no_compute['gpu_mem_alloc_mean_kb'] > 1e6]  # >1GB
    print(f"\n  With >1 GB GPU memory allocated: {len(significant_mem):,}")
    very_high_mem = mem_no_compute[mem_no_compute['gpu_mem_alloc_mean_kb'] > 10e6]  # >10GB
    print(f"  With >10 GB GPU memory allocated: {len(very_high_mem):,}")
    extreme_mem = mem_no_compute[mem_no_compute['gpu_mem_alloc_mean_kb'] > 30e6]  # >30GB
    print(f"  With >30 GB GPU memory allocated: {len(extreme_mem):,}")

    if len(significant_mem) > 0:
        print(f"\n  ⚠ {len(significant_mem):,} Ghost jobs loaded >1GB onto GPU memory with zero compute")
        print(f"    This is inconsistent with idle/misconfigured jobs")
        print(f"    Possible explanations: model loading without inference, memory-bound")
        print(f"    kernels between DCGM samples, or unauthorized GPU workload")

        # breakdown by science field
        print(f"\n  Science field breakdown (>1GB mem, zero compute Ghost):")
        fields = significant_mem['SCIENCE_FIELD_SHORT'].value_counts()
        for f, c in fields.head(5).items():
            print(f"    {f:<30s} {c:>5,}")

    # also check high_mem_low_util_frac for Ghost specifically
    hmlu = ghost[ghost['high_mem_low_util_frac'].notna() & (ghost['high_mem_low_util_frac'] > 0.5)]
    print(f"\n  Ghost jobs with high_mem_low_util_frac > 50%: {len(hmlu):,}")

    return mem_no_compute


# ─────────────────────────────────────────────────────────────────────────────
# 2. POWER DRAW ANOMALY
# ─────────────────────────────────────────────────────────────────────────────
def power_anomaly(ghost):
    sep("SIGNAL 2: ELEVATED POWER DRAW AT ZERO UTILIZATION")

    has_power = ghost['gpu_power_mean'].notna()
    has_zero_util = ghost['gpu_util_mean'].notna() & (ghost['gpu_util_mean'] < 1.0)

    idle_with_power = ghost[has_power & has_zero_util].copy()
    print(f"Ghost jobs with power data and <1% util: {len(idle_with_power):,}")

    if len(idle_with_power) == 0:
        print("  No power data available")
        return pd.DataFrame()

    power = idle_with_power['gpu_power_mean']
    print(f"\nPower draw distribution (Watts) at <1% utilization:")
    print(f"  Median:    {power.median():.1f} W")
    print(f"  Mean:      {power.mean():.1f} W")
    print(f"  P75:       {power.quantile(0.75):.1f} W")
    print(f"  P90:       {power.quantile(0.90):.1f} W")
    print(f"  P95:       {power.quantile(0.95):.1f} W")
    print(f"  Max:       {power.max():.1f} W")

    # A100 idle ~60-75W. >150W at zero util is anomalous
    IDLE_POWER = 75.0
    ANOMALY_THRESHOLD = 150.0

    elevated = idle_with_power[idle_with_power['gpu_power_mean'] > ANOMALY_THRESHOLD]
    print(f"\n  Ghost jobs with mean power > {ANOMALY_THRESHOLD}W at <1% util: {len(elevated):,}")

    if len(elevated) > 0:
        print(f"  ⚠ {len(elevated):,} Ghost jobs drawing >150W with reported 0% GPU utilization")
        print(f"    A100 idle power is ~60-75W. Elevated power suggests active workload")
        print(f"    that DCGM utilization counters are not capturing.")
        print(f"\n  Power distribution of anomalous jobs:")
        print(f"    Median: {elevated['gpu_power_mean'].median():.1f} W")
        print(f"    Mean:   {elevated['gpu_power_mean'].mean():.1f} W")
        print(f"    Max:    {elevated['gpu_power_mean'].max():.1f} W")

        # cross with memory
        elevated_with_mem = elevated[elevated['gpu_mem_alloc_mean_kb'] > 1e6]
        print(f"\n  Of these, also >1GB GPU memory: {len(elevated_with_mem):,}")
        if len(elevated_with_mem) > 0:
            print(f"  ⚠⚠ HIGH ANOMALY: {len(elevated_with_mem):,} jobs with elevated power + GPU memory + zero util")

        # science field
        print(f"\n  Science fields (elevated power Ghost):")
        for f, c in elevated['SCIENCE_FIELD_SHORT'].value_counts().head(5).items():
            print(f"    {f:<30s} {c:>5,}")

        # runtime
        print(f"\n  Runtime distribution:")
        print(f"    Median: {elevated['RUNTIME_SECONDS'].median():.0f}s ({elevated['RUNTIME_SECONDS'].median()/3600:.2f}h)")
        print(f"    Mean:   {elevated['RUNTIME_SECONDS'].mean():.0f}s ({elevated['RUNTIME_SECONDS'].mean()/3600:.2f}h)")

    # also check max power spikes
    max_power_spike = idle_with_power[idle_with_power['gpu_power_max'] > 250]
    print(f"\n  Ghost jobs with max power spike >250W at <1% mean util: {len(max_power_spike):,}")

    return elevated


# ─────────────────────────────────────────────────────────────────────────────
# 3. THERMAL ANOMALY
# ─────────────────────────────────────────────────────────────────────────────
def thermal_anomaly(ghost):
    sep("SIGNAL 3: THERMAL ANOMALY AT ZERO UTILIZATION")

    has_temp = ghost['gpu_temp_mean'].notna()
    has_zero_util = ghost['gpu_util_mean'].notna() & (ghost['gpu_util_mean'] < 1.0)

    idle_with_temp = ghost[has_temp & has_zero_util].copy()
    print(f"Ghost jobs with temp data and <1% util: {len(idle_with_temp):,}")

    if len(idle_with_temp) == 0:
        return pd.DataFrame()

    temp = idle_with_temp['gpu_temp_mean']
    print(f"\nTemperature distribution (°C) at <1% utilization:")
    print(f"  Median:    {temp.median():.1f}°C")
    print(f"  Mean:      {temp.mean():.1f}°C")
    print(f"  P90:       {temp.quantile(0.90):.1f}°C")
    print(f"  P95:       {temp.quantile(0.95):.1f}°C")
    print(f"  Max:       {temp.max():.1f}°C")

    # A100 idle should be ~25-35°C in a well-cooled datacenter
    # >50°C at zero util suggests active workload
    TEMP_ANOMALY = 50.0
    hot_idle = idle_with_temp[idle_with_temp['gpu_temp_mean'] > TEMP_ANOMALY]
    print(f"\n  Ghost jobs with mean temp > {TEMP_ANOMALY}°C at <1% util: {len(hot_idle):,}")

    if len(hot_idle) > 0:
        print(f"  ⚠ Elevated temperature at zero utilization suggests hidden GPU workload")
        # cross-check with power
        hot_with_power = hot_idle[hot_idle['gpu_power_mean'] > 150]
        print(f"  Also elevated power (>150W): {len(hot_with_power):,}")

    # thermal throttle risk at zero util — very suspicious
    throttle_idle = idle_with_temp[idle_with_temp['gpu_thermal_throttle_frac'] > 0]
    print(f"  Ghost jobs with thermal throttle events at <1% util: {len(throttle_idle):,}")
    if len(throttle_idle) > 0:
        print(f"  ⚠⚠ CRITICAL: GPU throttling at zero reported utilization — strong anomaly signal")

    return hot_idle


# ─────────────────────────────────────────────────────────────────────────────
# 4. MULTI-NODE GHOST JOBS (network without storage)
# ─────────────────────────────────────────────────────────────────────────────
def multinode_ghost(ghost):
    sep("SIGNAL 4: MULTI-NODE GHOST JOBS (NETWORK WITHOUT STORAGE)")

    # Ghost jobs on many nodes with zero I/O = using network but not filesystem
    large_ghost = ghost[ghost['NODES_USED'] >= 10].copy()
    print(f"Ghost jobs on ≥10 nodes: {len(large_ghost):,}")

    if len(large_ghost) == 0:
        print("  None found")
        return pd.DataFrame()

    print(f"\nNode distribution:")
    for p in [50, 75, 90, 95, 99]:
        val = large_ghost['NODES_USED'].quantile(p/100)
        print(f"  P{p:02d}: {val:.0f} nodes ({val*4:.0f} GPUs)")

    print(f"\n  Max nodes: {large_ghost['NODES_USED'].max():.0f} ({large_ghost['NODES_USED'].max()*4:.0f} GPUs)")
    print(f"  GPU hours consumed: {large_ghost['gpu_hours'].clip(lower=0).sum():,.0f}")

    # zero I/O + many nodes = what are they communicating?
    zero_io = large_ghost[large_ghost['total_bytes'].fillna(0) == 0]
    print(f"\n  With zero I/O: {len(zero_io):,} ({len(zero_io)/len(large_ghost)*100:.1f}%)")
    print(f"  → These jobs allocated {zero_io['NODES_USED'].sum()*4:,.0f} GPUs, transferred zero bytes")
    print(f"  → Either pure MPI communication, failed launches, or non-filesystem activity")

    # runtime distribution — very short multi-node jobs are suspicious (probe/test patterns)
    # very long ones are suspicious too (persistent background activity)
    short = large_ghost[large_ghost['RUNTIME_SECONDS'] < 300]
    long_jobs = large_ghost[large_ghost['RUNTIME_SECONDS'] > 7200]
    print(f"\n  Short (<5min): {len(short):,} jobs — possible probing/testing")
    print(f"  Long (>2hr):   {len(long_jobs):,} jobs — possible persistent workload")
    if len(long_jobs) > 0:
        print(f"    Max runtime:   {long_jobs['RUNTIME_SECONDS'].max()/3600:.1f} hours")
        print(f"    GPU hours:     {long_jobs['gpu_hours'].clip(lower=0).sum():,.0f}")
        print(f"    Unique users:  {long_jobs['USERNAME_GENID'].nunique()}")

    # exit codes
    print(f"\n  Exit codes (multi-node Ghost):")
    exits = large_ghost['EXIT_STATUS'].value_counts().head(5)
    for e, c in exits.items():
        print(f"    Exit {e}: {c:>5,} ({c/len(large_ghost)*100:.1f}%)")

    return large_ghost


# ─────────────────────────────────────────────────────────────────────────────
# 5. TEMPORAL PATTERN ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
def temporal_patterns(ghost):
    sep("SIGNAL 5: TEMPORAL SUBMISSION PATTERNS")

    ghost_ts = ghost.copy()
    ghost_ts['submit_time'] = pd.to_datetime(ghost_ts['QUEUED_TIMESTAMP'], errors='coerce')
    ghost_ts = ghost_ts.dropna(subset=['submit_time'])
    ghost_ts['hour'] = ghost_ts['submit_time'].dt.hour
    ghost_ts['day_of_week'] = ghost_ts['submit_time'].dt.dayofweek  # 0=Mon, 6=Sun
    ghost_ts['is_weekend'] = ghost_ts['day_of_week'] >= 5
    ghost_ts['is_offhours'] = (ghost_ts['hour'] < 6) | (ghost_ts['hour'] >= 22)
    ghost_ts['is_night_weekend'] = ghost_ts['is_weekend'] | ghost_ts['is_offhours']

    total = len(ghost_ts)
    print(f"Ghost jobs with timestamps: {total:,}")

    # hour-of-day distribution
    print(f"\nSubmission by hour (CT):")
    hour_dist = ghost_ts['hour'].value_counts().sort_index()
    max_hour = hour_dist.max()
    for hour, count in hour_dist.items():
        bar = '█' * int(count / max_hour * 30)
        print(f"  {hour:02d}:00  {count:>6,} ({count/total*100:>4.1f}%) {bar}")

    # off-hours concentration
    offhours = ghost_ts[ghost_ts['is_offhours']]
    weekend = ghost_ts[ghost_ts['is_weekend']]
    night_weekend = ghost_ts[ghost_ts['is_night_weekend']]
    print(f"\nOff-hours (10PM-6AM):      {len(offhours):,} ({len(offhours)/total*100:.1f}%)")
    print(f"Weekend:                   {len(weekend):,} ({len(weekend)/total*100:.1f}%)")
    print(f"Night or weekend:          {len(night_weekend):,} ({len(night_weekend)/total*100:.1f}%)")

    # expected if uniform: 8/24 = 33% off-hours, 2/7 = 28.6% weekend
    expected_offhours = total * (8/24)
    expected_weekend = total * (2/7)
    print(f"\nExpected (uniform):        off-hours={expected_offhours/total*100:.1f}%, weekend={expected_weekend/total*100:.1f}%")
    print(f"Observed:                  off-hours={len(offhours)/total*100:.1f}%, weekend={len(weekend)/total*100:.1f}%")

    offhours_ratio = (len(offhours)/total) / (8/24)
    weekend_ratio = (len(weekend)/total) / (2/7)
    print(f"Off-hours overrepresentation: {offhours_ratio:.2f}x")
    print(f"Weekend overrepresentation:   {weekend_ratio:.2f}x")

    # per-user submission cadence — look for very regular intervals
    print(f"\nPer-user submission cadence (top Ghost users by job count):")
    top_ghost_users = ghost_ts['USERNAME_GENID'].value_counts().head(15).index
    regular_users = []

    print(f"{'User':<15s} {'Jobs':>6s} {'MedInterval':>13s} {'CV':>8s} {'OffHrs%':>8s} {'Pattern':>15s}")
    print("-" * 75)

    for user in top_ghost_users:
        u = ghost_ts[ghost_ts['USERNAME_GENID'] == user].sort_values('submit_time')
        intervals = u['submit_time'].diff().dt.total_seconds().dropna()
        if len(intervals) < 5:
            continue

        med_interval = intervals.median()
        cv = intervals.std() / intervals.mean() if intervals.mean() > 0 else 0
        offhours_pct = u['is_offhours'].mean() * 100
        uid = str(int(user))[-6:]

        # classify pattern
        if cv < 0.3 and med_interval < 7200:
            pattern = "⚠ AUTOMATED"
            regular_users.append(user)
        elif cv < 0.5 and med_interval < 3600:
            pattern = "⚠ SEMI-REGULAR"
            regular_users.append(user)
        elif med_interval < 600:
            pattern = "BURST"
        else:
            pattern = "irregular"

        interval_str = f"{med_interval/3600:.1f}h" if med_interval > 3600 else f"{med_interval/60:.0f}min"

        print(f"···{uid:<11s} {len(u):>6,} {interval_str:>13s} {cv:>8.2f} {offhours_pct:>7.1f}% {pattern:>15s}")

    if regular_users:
        print(f"\n  ⚠ {len(regular_users)} users show automated/semi-regular Ghost submission patterns")
        total_hours = ghost_ts[ghost_ts['USERNAME_GENID'].isin(regular_users)]['gpu_hours'].clip(lower=0).sum()
        print(f"    Combined GPU hours: {total_hours:,.0f}")

    return ghost_ts


# ─────────────────────────────────────────────────────────────────────────────
# 6. GPU IMBALANCE AT ZERO UTILIZATION
# ─────────────────────────────────────────────────────────────────────────────
def gpu_imbalance_anomaly(ghost):
    sep("SIGNAL 6: GPU IMBALANCE AT ZERO UTILIZATION")

    # If mean util is 0% but imbalance is high, some GPUs are active while others idle
    # This could indicate selective GPU use (e.g., mining on GPU 0, leaving others idle)
    has_data = ghost['gpu_imbalance_mean'].notna() & ghost['gpu_util_mean'].notna()
    idle_ghost = ghost[has_data & (ghost['gpu_util_mean'] < 1.0)].copy()

    print(f"Ghost jobs with imbalance data and <1% mean util: {len(idle_ghost):,}")

    if len(idle_ghost) == 0:
        return pd.DataFrame()

    imb = idle_ghost['gpu_imbalance_mean']
    print(f"\nGPU imbalance distribution at <1% mean utilization:")
    print(f"  Median:    {imb.median():.2f}")
    print(f"  Mean:      {imb.mean():.2f}")
    print(f"  P90:       {imb.quantile(0.90):.2f}")
    print(f"  P95:       {imb.quantile(0.95):.2f}")
    print(f"  Max:       {imb.max():.2f}")

    # high imbalance at zero mean = some GPUs doing something
    high_imb = idle_ghost[idle_ghost['gpu_imbalance_mean'] > 5.0]
    print(f"\n  Ghost jobs with imbalance >5 at <1% mean util: {len(high_imb):,}")
    if len(high_imb) > 0:
        print(f"  ⚠ Some GPUs are differentially active despite near-zero mean utilization")
        print(f"    Max imbalance: {high_imb['gpu_imbalance_max'].max():.1f}")
        print(f"    These jobs may have selective GPU activity on a subset of devices")

    # max util vs mean util — if max is high but mean is low, brief bursts
    burst = idle_ghost[
        (idle_ghost['gpu_util_mean'] < 1.0) &
        (idle_ghost['gpu_util_max'] > 50.0)
    ]
    print(f"\n  Ghost jobs with <1% mean but >50% max util: {len(burst):,}")
    if len(burst) > 0:
        print(f"  ⚠ Brief high-utilization bursts within otherwise idle jobs")
        print(f"    This pattern is consistent with intermittent GPU workloads")
        print(f"    (batch processing, periodic inference, or evasive mining)")

    return high_imb


# ─────────────────────────────────────────────────────────────────────────────
# 7. COMPOSITE ANOMALY SCORING
# ─────────────────────────────────────────────────────────────────────────────
def composite_anomaly(ghost):
    sep("COMPOSITE ANOMALY SCORING")

    g = ghost.copy()
    g['anomaly_score'] = 0.0

    # signal 1: GPU memory without compute
    has_mem = g['gpu_mem_alloc_mean_kb'].notna() & (g['gpu_mem_alloc_mean_kb'] > 1e6)  # >1GB
    has_zero_util = g['gpu_util_mean'].notna() & (g['gpu_util_mean'] < 1.0)
    g.loc[has_mem & has_zero_util, 'anomaly_score'] += 2.0

    # signal 2: elevated power
    high_power = g['gpu_power_mean'].notna() & (g['gpu_power_mean'] > 150)
    g.loc[high_power & has_zero_util, 'anomaly_score'] += 2.0

    # signal 3: elevated temperature
    high_temp = g['gpu_temp_mean'].notna() & (g['gpu_temp_mean'] > 50)
    g.loc[high_temp & has_zero_util, 'anomaly_score'] += 1.0

    # signal 4: multi-node
    g.loc[g['NODES_USED'] >= 10, 'anomaly_score'] += 1.0
    g.loc[g['NODES_USED'] >= 50, 'anomaly_score'] += 1.0

    # signal 5: long runtime (>2 hours) — persistent presence
    g.loc[g['RUNTIME_SECONDS'] > 7200, 'anomaly_score'] += 1.0

    # signal 6: GPU imbalance
    high_imb = g['gpu_imbalance_mean'].notna() & (g['gpu_imbalance_mean'] > 5.0)
    g.loc[high_imb & has_zero_util, 'anomaly_score'] += 1.0

    # signal 7: burst utilization (low mean, high max)
    burst = (g['gpu_util_mean'] < 1.0) & (g['gpu_util_max'].notna()) & (g['gpu_util_max'] > 50)
    g.loc[burst, 'anomaly_score'] += 2.0

    # signal 8: exit code 0 + long runtime + zero everything (successful nothing)
    successful_nothing = (
        (g['EXIT_STATUS'] == 0) &
        (g['RUNTIME_SECONDS'] > 3600) &
        has_zero_util &
        (g['total_bytes'].fillna(0) == 0)
    )
    g.loc[successful_nothing, 'anomaly_score'] += 1.0

    # distribution
    print(f"Anomaly score distribution across {len(g):,} Ghost jobs:")
    for threshold in [0, 1, 2, 3, 4, 5, 6, 7, 8]:
        n = (g['anomaly_score'] >= threshold).sum()
        hours = g[g['anomaly_score'] >= threshold]['gpu_hours'].clip(lower=0).sum()
        print(f"  Score ≥{threshold}: {n:>7,} jobs ({n/len(g)*100:>5.1f}%) | {hours:>12,.0f} GPU-hrs")

    # top anomalous jobs
    top = g.nlargest(20, 'anomaly_score')
    print(f"\nTop 20 anomalous Ghost jobs:")
    print(f"{'JobID':<15s} {'Score':>6s} {'GPUs':>5s} {'Runtime':>8s} {'Power':>7s} {'MemGB':>7s} {'Util%':>6s} {'MaxUtil':>7s} {'Field':<20s}")
    print("-" * 100)
    for _, r in top.iterrows():
        job = str(r.get('job_id', ''))[:12]
        gpus = r['NODES_USED'] * 4
        rt = f"{r['RUNTIME_SECONDS']/3600:.1f}h"
        power = f"{r['gpu_power_mean']:.0f}W" if pd.notna(r.get('gpu_power_mean')) else 'N/A'
        mem = f"{r['gpu_mem_alloc_mean_kb']/1e6:.1f}" if pd.notna(r.get('gpu_mem_alloc_mean_kb')) else 'N/A'
        util = f"{r['gpu_util_mean']:.1f}" if pd.notna(r.get('gpu_util_mean')) else 'N/A'
        maxu = f"{r['gpu_util_max']:.1f}" if pd.notna(r.get('gpu_util_max')) else 'N/A'
        field = str(r.get('SCIENCE_FIELD_SHORT', ''))[:18]
        print(f"{job:<15s} {r['anomaly_score']:>6.1f} {gpus:>5.0f} {rt:>8s} {power:>7s} {mem:>7s} {util:>6s} {maxu:>7s} {field:<20s}")

    # per-user anomaly concentration
    high_anomaly = g[g['anomaly_score'] >= 3]
    if len(high_anomaly) > 0:
        print(f"\nUsers with ≥3 anomaly score Ghost jobs:")
        user_anom = high_anomaly.groupby('USERNAME_GENID').agg(
            jobs=('job_id', 'count'),
            gpu_hours=('gpu_hours', lambda x: x.clip(lower=0).sum()),
            mean_score=('anomaly_score', 'mean'),
            max_score=('anomaly_score', 'max'),
        ).sort_values('gpu_hours', ascending=False).head(15)

        print(f"{'User':<15s} {'Jobs':>6s} {'GPU-hrs':>10s} {'MeanScore':>10s} {'MaxScore':>9s}")
        print("-" * 55)
        for uid, r in user_anom.iterrows():
            u = str(int(uid))[-6:]
            print(f"···{u:<11s} {r['jobs']:>6,} {r['gpu_hours']:>10,.0f} {r['mean_score']:>10.1f} {r['max_score']:>9.1f}")

    return g


# ─────────────────────────────────────────────────────────────────────────────
# 8. PAPER-READY SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
def paper_summary(ghost, scored):
    sep("PAPER-READY SECURITY SUMMARY")

    total = len(ghost)
    has_zero_util = ghost['gpu_util_mean'].notna() & (ghost['gpu_util_mean'] < 1.0)
    zero_util = ghost[has_zero_util]

    # count each signal
    mem_anomaly = (
        zero_util['gpu_mem_alloc_mean_kb'].notna() &
        (zero_util['gpu_mem_alloc_mean_kb'] > 1e6)
    ).sum()

    power_anomaly_n = (
        zero_util['gpu_power_mean'].notna() &
        (zero_util['gpu_power_mean'] > 150)
    ).sum()

    temp_anomaly = (
        zero_util['gpu_temp_mean'].notna() &
        (zero_util['gpu_temp_mean'] > 50)
    ).sum()

    burst_anomaly = (
        (zero_util['gpu_util_max'].notna()) &
        (zero_util['gpu_util_max'] > 50)
    ).sum()

    multinode = (ghost['NODES_USED'] >= 10).sum()

    composite_high = (scored['anomaly_score'] >= 3).sum() if scored is not None else 0
    composite_hours = scored[scored['anomaly_score'] >= 3]['gpu_hours'].clip(lower=0).sum() if scored is not None else 0

    print(f"Ghost jobs analyzed:           {total:,}")
    print(f"With <1% GPU utilization:      {len(zero_util):,}")
    print(f"\nAnomaly signals detected:")
    print(f"  GPU memory >1GB at 0% util:  {mem_anomaly:,}")
    print(f"  Power >150W at 0% util:      {power_anomaly_n:,}")
    print(f"  Temp >50°C at 0% util:       {temp_anomaly:,}")
    print(f"  Burst util (max>50%, mean<1%):{burst_anomaly:,}")
    print(f"  Multi-node (≥10 nodes):      {multinode:,}")
    print(f"\nComposite score ≥3:            {composite_high:,} jobs, {composite_hours:,.0f} GPU-hrs")

    print(f"\n--- PAPER PARAGRAPH (for §6.3 Discussion) ---")
    print(f"Beyond efficiency, the Ghost population presents an operational awareness")
    print(f"concern. Of {total:,} Ghost jobs, {len(zero_util):,} achieve <1% GPU utilization.")
    print(f"Among these, {mem_anomaly:,} allocate >1 GB of GPU memory despite zero compute,")
    print(f"{power_anomaly_n:,} draw >150W (exceeding A100 idle power of ~75W), and")
    print(f"{burst_anomaly:,} exhibit brief high-utilization bursts (>50% peak) within")
    print(f"otherwise idle jobs. A composite anomaly score combining these signals with")
    print(f"temporal regularity and multi-node allocation identifies {composite_high:,} jobs")
    print(f"({composite_hours:,.0f} GPU hours) warranting operator review. We do not claim")
    print(f"these jobs are malicious; however, current single-layer monitoring cannot")
    print(f"distinguish between misconfigured legitimate workflows and unauthorized")
    print(f"resource consumption. Cross-layer correlation provides the minimum")
    print(f"observability needed to surface these patterns for review.")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    cfg = load_config(args.config)

    print("Loading combined metrics...", flush=True)
    df = pd.read_csv(cfg["combined_out"], low_memory=False)
    df["job_id"] = df["JOB_NAME"].str.split(".").str[0]
    print(f"  {len(df):,} jobs loaded")

    ghost = df[df['crosslayer_tier'] == 'Ghost'].copy()
    print(f"  Ghost jobs: {len(ghost):,}")

    # run all signal analyses
    mem_anomalous = memory_without_compute(ghost)
    power_elevated = power_anomaly(ghost)
    hot_idle = thermal_anomaly(ghost)
    large_ghost_df = multinode_ghost(ghost)
    temporal_df = temporal_patterns(ghost)
    imbalance_df = gpu_imbalance_anomaly(ghost)
    scored = composite_anomaly(ghost)

    # summary
    paper_summary(ghost, scored)

    # save scored results
    out_path = cfg["combined_out"].replace("combined_metrics.csv", "ghost_anomaly_scores.csv")
    cols_to_save = ['job_id', 'USERNAME_GENID', 'SCIENCE_FIELD_SHORT',
                    'NODES_USED', 'RUNTIME_SECONDS', 'gpu_util_mean', 'gpu_util_max',
                    'gpu_mem_alloc_mean_kb', 'gpu_power_mean', 'gpu_power_max',
                    'gpu_temp_mean', 'gpu_imbalance_mean', 'total_bytes',
                    'EXIT_STATUS', 'anomaly_score']
    save_cols = [c for c in cols_to_save if c in scored.columns]
    scored[save_cols].to_csv(out_path, index=False)
    print(f"\nAnomaly scores → {out_path}")

    print(f"\n{'='*80}")
    print(f"  DONE — pipe output to a file:")
    print(f"  python -m pipeline.stage09_security_audit --config config/config.json > security_audit.txt")
    print(f"{'='*80}")
