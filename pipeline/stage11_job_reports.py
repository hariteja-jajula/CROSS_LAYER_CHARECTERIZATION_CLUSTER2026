"""
stage11_job_reports.py — Automated per-job diagnostic report generator
Run after stage03_build_combined.

Generates plain-English, cross-layer diagnostic reports for completed jobs.
This is the operational output of the pipeline — what a facility would
actually deploy as automated post-job feedback to users.

For the paper: generates example reports for each tier using real job data,
demonstrating what closed-loop feedback looks like in practice.

Run: python -m pipeline.stage11_job_reports --config config/config.json
"""

import pandas as pd
import numpy as np
import json, argparse
from pathlib import Path

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
# REPORT GENERATION ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def generate_job_report(job, user_history=None):
    """
    Generate a plain-English diagnostic report for a single job.
    Returns a dict with structured report fields.
    """
    tier = job.get('crosslayer_tier', 'Unknown')
    job_id = str(job.get('job_id', 'Unknown'))
    nodes = job.get('NODES_USED', 0)
    gpus = int(nodes * 4) if nodes > 0 else 0
    runtime_s = job.get('RUNTIME_SECONDS', 0)
    walltime_s = job.get('WALLTIME_SECONDS', 0)
    runtime_h = runtime_s / 3600
    gpu_util = job.get('gpu_util_mean', None)
    gpu_hours = job.get('gpu_hours', 0)
    total_bytes = job.get('total_bytes', 0) or 0
    bwio = job.get('BWio_MB', None)
    has_mpiio = job.get('has_mpiio', False)
    has_posix = job.get('has_posix', False)
    cb_nodes = job.get('cb_nodes', 0)
    small_write = job.get('small_write_ratio', 0)
    small_read = job.get('small_read_ratio', 0)
    write_dom = job.get('write_dominance', 0)
    exit_code = job.get('EXIT_STATUS', 0)
    field = job.get('SCIENCE_FIELD_SHORT', 'Unknown')
    rank_imbalance = job.get('rank_imbalance', 0)
    gpu_phase1 = job.get('gpu_util_phase1', None)
    gpu_phase2 = job.get('gpu_util_phase2', None)
    gpu_phase3 = job.get('gpu_util_phase3', None)
    mem_alloc_kb = job.get('gpu_mem_alloc_mean_kb', 0) or 0
    power = job.get('gpu_power_mean', None)
    walltime_util = runtime_s / walltime_s if walltime_s > 0 else 0

    report = {
        'job_id': job_id,
        'tier': tier,
        'severity': 'info',
        'headline': '',
        'summary': '',
        'findings': [],
        'recommendations': [],
        'metrics': {},
    }

    # ── METRICS (always included) ────────────────────────────────────────
    report['metrics'] = {
        'nodes': nodes,
        'gpus': gpus,
        'runtime': f"{runtime_h:.2f}h",
        'gpu_util': f"{gpu_util:.1f}%" if pd.notna(gpu_util) else "N/A",
        'data_moved': format_bytes(total_bytes),
        'bandwidth': f"{bwio:.1f} MB/s" if pd.notna(bwio) and bwio > 0 else "N/A",
        'gpu_hours': f"{gpu_hours:.1f}",
        'walltime_util': f"{walltime_util*100:.1f}%",
        'exit_code': int(exit_code) if pd.notna(exit_code) else 'N/A',
    }

    # ── TIER-SPECIFIC DIAGNOSIS ──────────────────────────────────────────

    if tier == 'Ghost':
        report['severity'] = 'critical'
        report['headline'] = 'Ghost Job — GPU resources allocated but unused'
        report['summary'] = (
            f"Your job ran for {runtime_h:.2f} hours on {nodes:.0f} nodes "
            f"({gpus} GPUs) with {gpu_util:.1f}% GPU utilization and transferred "
            f"zero bytes of data. The GPUs were effectively idle for the entire "
            f"job duration, consuming {gpu_hours:.0f} GPU-hours."
        )

        report['findings'].append(
            f"GPU utilization was {gpu_util:.1f}% across all job phases "
            f"(Phase 1: {gpu_phase1:.1f}%, Phase 2: {gpu_phase2:.1f}%, "
            f"Phase 3: {gpu_phase3:.1f}%). No I/O activity was detected."
            if pd.notna(gpu_phase1) else
            f"GPU utilization was near zero. No I/O activity was detected."
        )

        if walltime_util < 0.1:
            report['findings'].append(
                f"Walltime utilization was {walltime_util*100:.1f}% — your job used "
                f"only {runtime_h:.2f}h of the {walltime_s/3600:.1f}h requested, "
                f"holding the scheduler slot for {1/walltime_util:.0f}x longer than needed."
            )

        if gpus > 4 and total_bytes == 0:
            report['findings'].append(
                f"Multi-node allocation ({nodes:.0f} nodes, {gpus} GPUs) with zero "
                f"compute and zero I/O suggests this workload may not require GPU resources."
            )

        report['recommendations'].append(
            "Verify that your application is compiled with GPU support and that "
            "GPU kernels are being launched. Check for missing CUDA/HIP initialization."
        )
        report['recommendations'].append(
            "If this workload does not use GPUs, submit to a CPU queue to free "
            "GPU resources for compute-intensive jobs."
        )
        if walltime_util < 0.1:
            report['recommendations'].append(
                f"Reduce your requested walltime to match actual runtime. "
                f"A walltime of {max(runtime_s * 1.5 / 3600, 0.5):.1f}h would provide "
                f"50% buffer while freeing scheduler capacity."
            )
        if mem_alloc_kb > 1e6:
            report['findings'].append(
                f"GPU memory allocation detected ({mem_alloc_kb/1e6:.1f} GB) despite "
                f"zero compute utilization. Something was loaded onto the GPU but "
                f"no compute kernels executed."
            )

    elif tier == 'IO_Bottlenecked':
        report['severity'] = 'warning'
        report['headline'] = 'I/O Bottlenecked — GPUs idle during sustained I/O'
        report['summary'] = (
            f"Your job ran for {runtime_h:.2f} hours on {nodes:.0f} nodes "
            f"({gpus} GPUs) and transferred {format_bytes(total_bytes)} "
            f"at {bwio:.1f} MB/s, but GPU utilization was only {gpu_util:.1f}%. "
            f"The GPUs were idle while your application performed I/O."
            if pd.notna(bwio) and bwio > 0 else
            f"Your job ran for {runtime_h:.2f} hours on {nodes:.0f} nodes "
            f"({gpus} GPUs) with I/O activity but near-zero GPU utilization."
        )

        if pd.notna(gpu_phase1):
            report['findings'].append(
                f"GPU was idle across all job phases (P1: {gpu_phase1:.1f}%, "
                f"P2: {gpu_phase2:.1f}%, P3: {gpu_phase3:.1f}%), indicating "
                f"the GPU was never engaged — not even briefly."
            )

        if has_mpiio and cb_nodes == 4 and nodes > 4:
            report['findings'].append(
                f"Your MPI-IO job uses cb_nodes=4 (facility default) on "
                f"{nodes:.0f} nodes. Collective buffering is serialized through "
                f"4 aggregator nodes regardless of your job's scale."
            )
            report['recommendations'].append(
                f"Set cb_nodes to match your node count or use a higher value. "
                f"Add to your job script: export ROMIO_HINTS with cb_nodes={int(min(nodes, 64))}. "
                f"This alone could improve I/O throughput significantly."
            )

        if small_write > 0.5:
            pct = small_write * 100
            report['findings'].append(
                f"{pct:.0f}% of write operations are below 10 KB. This metadata-heavy "
                f"pattern generates excessive filesystem overhead on Lustre."
            )
            report['recommendations'].append(
                "Buffer small writes into larger I/O operations. Consider using "
                "node-local storage (/dev/shm or local NVMe) for intermediate files "
                "and writing aggregated output to the parallel filesystem."
            )

        if small_read > 0.5:
            report['findings'].append(
                f"{small_read*100:.0f}% of read operations are below 10 KB. "
                f"Consider preloading data into memory or using a data loader "
                f"that aggregates small reads."
            )

        if rank_imbalance > 10:
            report['findings'].append(
                f"Rank I/O imbalance ratio is {rank_imbalance:.1f}x — the slowest "
                f"rank transfers {rank_imbalance:.0f}x more data than the fastest. "
                f"This serialization bottleneck limits parallel I/O throughput."
            )
            report['recommendations'].append(
                "Redistribute I/O across ranks more evenly. If using a single-writer "
                "pattern, switch to collective I/O with MPI_File_write_all."
            )

        if pd.notna(bwio) and bwio < 100 and total_bytes > 1e9:
            report['recommendations'].append(
                f"Your effective bandwidth ({bwio:.1f} MB/s) is well below the "
                f"filesystem capability. Consider profiling with Darshan's DXT "
                f"module for a detailed trace of your I/O pattern."
            )

    elif tier == 'Compute_Bound':
        report['severity'] = 'good'
        report['headline'] = 'Compute Bound — healthy GPU utilization'
        report['summary'] = (
            f"Your job ran for {runtime_h:.2f} hours on {nodes:.0f} nodes "
            f"({gpus} GPUs) with {gpu_util:.1f}% GPU utilization. "
            f"This is an efficiently compute-bound workload."
        )

        if pd.notna(gpu_phase1) and gpu_phase1 < gpu_phase2 * 0.7:
            report['findings'].append(
                f"GPU ramp-up detected: Phase 1 ({gpu_phase1:.1f}%) → "
                f"Phase 2 ({gpu_phase2:.1f}%). Data loading or initialization "
                f"may account for {(1 - gpu_phase1/gpu_phase2)*100:.0f}% of "
                f"Phase 1 underutilization."
            )
            report['recommendations'].append(
                "Consider overlapping data loading with compute using "
                "asynchronous data pipelines or prefetching."
            )

        if walltime_util > 0.95:
            report['findings'].append(
                f"Walltime utilization is {walltime_util*100:.1f}% — your job "
                f"nearly exhausted its allocation. Risk of timeout on longer runs."
            )
            report['recommendations'].append(
                "Add 20-30% walltime buffer to prevent job termination, "
                "or implement checkpointing to resume from the last saved state."
            )

    elif tier == 'Balanced':
        report['severity'] = 'good'
        report['headline'] = 'Balanced — both GPU and I/O active'
        report['summary'] = (
            f"Your job ran for {runtime_h:.2f} hours on {nodes:.0f} nodes "
            f"({gpus} GPUs) with {gpu_util:.1f}% GPU utilization and "
            f"transferred {format_bytes(total_bytes)}. Both compute and I/O "
            f"resources are being utilized."
        )

        if has_mpiio and cb_nodes == 4 and nodes > 4:
            report['findings'].append(
                f"MPI-IO with cb_nodes=4 on {nodes:.0f} nodes. I/O throughput "
                f"could improve with tuned collective buffering."
            )
            report['recommendations'].append(
                f"Consider increasing cb_nodes to {int(min(nodes, 64))} for "
                f"better I/O parallelism."
            )

        if pd.notna(bwio) and bwio < 100 and total_bytes > 10e9:
            report['recommendations'].append(
                f"I/O bandwidth ({bwio:.1f} MB/s) is low relative to data volume "
                f"({format_bytes(total_bytes)}). Optimizing I/O could reduce "
                f"overall runtime."
            )

    elif tier == 'Scale_Waster':
        report['severity'] = 'critical'
        report['headline'] = 'Scale Waster — large allocation with minimal GPU use'
        report['summary'] = (
            f"Your job ran for {runtime_h:.2f} hours on {nodes:.0f} nodes "
            f"({gpus} GPUs) with only {gpu_util:.1f}% GPU utilization and "
            f"no significant I/O. This large allocation was underutilized."
        )

        report['findings'].append(
            f"At {gpu_util:.1f}% utilization across {gpus} GPUs, your job "
            f"consumed {gpu_hours:.0f} GPU-hours but performed minimal computation."
        )

        report['recommendations'].append(
            "Run a smaller-scale test first to verify GPU utilization before "
            "submitting large allocations."
        )
        report['recommendations'].append(
            f"If your workload scales to only a fraction of {gpus} GPUs, "
            f"reduce the node count to match actual parallelism."
        )

    elif tier == 'Failed_Job':
        report['severity'] = 'warning'
        report['headline'] = f'Job Failed — exit code {int(exit_code) if pd.notna(exit_code) else "unknown"}'
        report['summary'] = (
            f"Your job ran for {runtime_h:.2f} hours on {nodes:.0f} nodes "
            f"({gpus} GPUs) before failing with exit code "
            f"{int(exit_code) if pd.notna(exit_code) else 'unknown'}. "
            f"{gpu_hours:.0f} GPU-hours were consumed before failure."
        )

        if pd.notna(gpu_util) and gpu_util > 50:
            report['findings'].append(
                f"GPU utilization was {gpu_util:.1f}% before failure — "
                f"the job was doing useful work before it crashed."
            )
            report['recommendations'].append(
                "Implement checkpointing to avoid losing compute progress "
                "on future failures."
            )
        elif pd.notna(gpu_util) and gpu_util < 5:
            report['findings'].append(
                f"GPU utilization was only {gpu_util:.1f}% before failure — "
                f"the job may have failed during initialization."
            )
            report['recommendations'].append(
                "Test your job with a short walltime and minimal resources "
                "to identify the failure point before using full allocation."
            )

        if pd.notna(exit_code):
            ec = int(exit_code)
            if ec == -29 or ec == 271:
                report['recommendations'].append(
                    "Exit code suggests job was killed by the scheduler (timeout or "
                    "resource limit). Increase walltime or reduce resource requirements."
                )
            elif ec == 137 or ec == 143:
                report['recommendations'].append(
                    "Exit code suggests OOM kill or SIGTERM. Check memory usage "
                    "and consider reducing batch size or per-rank memory footprint."
                )

    elif tier == 'Short_Job':
        report['severity'] = 'info'
        report['headline'] = 'Short Job — runtime under 60 seconds'
        report['summary'] = (
            f"Your job ran for {runtime_s:.0f} seconds on {nodes:.0f} nodes "
            f"({gpus} GPUs). Short jobs incur scheduling overhead relative to "
            f"useful work performed."
        )
        if gpus > 4:
            report['recommendations'].append(
                "Consider batching multiple short runs into a single job "
                "to amortize scheduling and initialization overhead."
            )

    else:
        report['severity'] = 'info'
        report['headline'] = f'Job classified as {tier}'
        report['summary'] = (
            f"Your job ran for {runtime_h:.2f} hours on {nodes:.0f} nodes "
            f"({gpus} GPUs)."
        )

    # ── USER HISTORY CONTEXT (if available) ──────────────────────────────
    if user_history is not None and len(user_history) > 0:
        same_tier = (user_history['crosslayer_tier'] == tier).sum()
        total_hist = len(user_history)
        if same_tier / total_hist > 0.5 and same_tier > 5:
            report['findings'].append(
                f"Pattern alert: {same_tier} of your last {total_hist} jobs "
                f"({same_tier/total_hist*100:.0f}%) were also classified as {tier}. "
                f"This is a recurring workflow pattern, not an isolated event."
            )

    return report


def format_bytes(b):
    if b is None or pd.isna(b) or b == 0:
        return "0 bytes"
    if b < 1e6:
        return f"{b/1e3:.1f} KB"
    if b < 1e9:
        return f"{b/1e6:.1f} MB"
    if b < 1e12:
        return f"{b/1e9:.1f} GB"
    return f"{b/1e12:.2f} TB"


def format_report_text(report):
    """Format a report dict into plain text for display."""
    severity_icons = {
        'critical': '🔴',
        'warning': '🟡',
        'good': '🟢',
        'info': '⚪',
    }
    icon = severity_icons.get(report['severity'], '⚪')

    lines = []
    lines.append(f"┌─────────────────────────────────────────────────────────────────┐")
    lines.append(f"│ {icon} JOB REPORT: {report['job_id']:<49s}│")
    lines.append(f"│ {report['headline']:<63s}│")
    lines.append(f"├─────────────────────────────────────────────────────────────────┤")

    # metrics bar
    m = report['metrics']
    lines.append(f"│ Nodes: {m['nodes']:<5} GPUs: {m['gpus']:<5} Runtime: {m['runtime']:<8s}"
                 f" GPU: {m['gpu_util']:<7s} Exit: {m['exit_code']}  │")
    lines.append(f"│ Data: {m['data_moved']:<10s} BW: {m['bandwidth']:<12s}"
                 f" GPU-hrs: {m['gpu_hours']:<8s} WT: {m['walltime_util']}│")
    lines.append(f"├─────────────────────────────────────────────────────────────────┤")

    # summary
    wrapped = wrap_text(report['summary'], 63)
    for line in wrapped:
        lines.append(f"│ {line:<63s}│")

    # findings
    if report['findings']:
        lines.append(f"│                                                                 │")
        lines.append(f"│ FINDINGS:                                                       │")
        for i, finding in enumerate(report['findings'], 1):
            wrapped = wrap_text(f"  {i}. {finding}", 63)
            for line in wrapped:
                lines.append(f"│ {line:<63s}│")

    # recommendations
    if report['recommendations']:
        lines.append(f"│                                                                 │")
        lines.append(f"│ RECOMMENDATIONS:                                                │")
        for i, rec in enumerate(report['recommendations'], 1):
            wrapped = wrap_text(f"  {i}. {rec}", 63)
            for line in wrapped:
                lines.append(f"│ {line:<63s}│")

    lines.append(f"└─────────────────────────────────────────────────────────────────┘")

    return '\n'.join(lines)


def wrap_text(text, width):
    """Simple word-wrap."""
    words = text.split()
    lines = []
    current = ""
    for word in words:
        if len(current) + len(word) + 1 <= width:
            current = current + " " + word if current else word
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


# ─────────────────────────────────────────────────────────────────────────────
# GENERATE EXAMPLE REPORTS FOR EACH TIER
# ─────────────────────────────────────────────────────────────────────────────

def generate_tier_examples(df):
    """Generate one representative report per tier using real job data."""
    sep("EXAMPLE REPORTS BY TIER")

    tiers_to_report = [
        'Ghost', 'IO_Bottlenecked', 'Compute_Bound', 'Balanced',
        'Scale_Waster', 'Failed_Job', 'Short_Job',
    ]

    for tier in tiers_to_report:
        tier_df = df[df['crosslayer_tier'] == tier]
        if len(tier_df) == 0:
            continue

        # pick a representative job — median gpu_hours for the tier
        tier_sorted = tier_df.sort_values('gpu_hours')
        median_idx = len(tier_sorted) // 2
        job = tier_sorted.iloc[median_idx]

        # get user history
        user_history = df[
            (df['USERNAME_GENID'] == job['USERNAME_GENID']) &
            (df['START_TIMESTAMP'] < job['START_TIMESTAMP'])
        ].tail(20)

        report = generate_job_report(job, user_history)
        print(format_report_text(report))
        print()


# ─────────────────────────────────────────────────────────────────────────────
# GENERATE REPORTS FOR WORST OFFENDERS
# ─────────────────────────────────────────────────────────────────────────────

def generate_worst_offender_reports(df):
    """Generate reports for the highest-impact waste jobs."""
    sep("WORST OFFENDER REPORTS (highest GPU-hours waste jobs)")

    waste_tiers = ['Ghost', 'Scale_Waster', 'IO_Bottlenecked']
    waste = df[df['crosslayer_tier'].isin(waste_tiers)].copy()
    worst = waste.nlargest(5, 'gpu_hours')

    for _, job in worst.iterrows():
        user_history = df[
            (df['USERNAME_GENID'] == job['USERNAME_GENID']) &
            (df['START_TIMESTAMP'] < job['START_TIMESTAMP'])
        ].tail(20)

        report = generate_job_report(job, user_history)
        print(format_report_text(report))
        print()


# ─────────────────────────────────────────────────────────────────────────────
# AGGREGATE STATISTICS — WHAT WOULD AUTOMATED DEPLOYMENT LOOK LIKE
# ─────────────────────────────────────────────────────────────────────────────

def deployment_statistics(df):
    """Compute statistics about what automated deployment would produce."""
    sep("DEPLOYMENT IMPACT ANALYSIS")

    total = len(df)

    # classify severity
    severity_map = {
        'Ghost': 'critical',
        'Scale_Waster': 'critical',
        'IO_Bottlenecked': 'warning',
        'Failed_Job': 'warning',
        'Compute_Bound': 'good',
        'Balanced': 'good',
        'Short_Job': 'info',
    }
    df2 = df.copy()
    df2['severity'] = df2['crosslayer_tier'].map(severity_map).fillna('info')

    print(f"If deployed on all {total:,} jobs:\n")

    # severity distribution
    sev_counts = df2['severity'].value_counts()
    for sev in ['critical', 'warning', 'good', 'info']:
        n = sev_counts.get(sev, 0)
        hours = df2[df2['severity'] == sev]['gpu_hours'].clip(lower=0).sum()
        print(f"  {sev:<12s} {n:>8,} jobs ({n/total*100:>5.1f}%) | {hours:>12,.0f} GPU-hrs")

    # actionable reports (critical + warning with specific recommendations)
    actionable_tiers = ['Ghost', 'Scale_Waster', 'IO_Bottlenecked', 'Failed_Job']
    actionable = df2[df2['crosslayer_tier'].isin(actionable_tiers)]
    print(f"\n  Actionable reports generated: {len(actionable):,} ({len(actionable)/total*100:.1f}%)")
    print(f"  GPU hours covered:           {actionable['gpu_hours'].clip(lower=0).sum():,.0f}")

    # cb_nodes recommendations
    mpiio_fixable = df2[(df2['has_mpiio'] == True) & (df2['cb_nodes'] == 4) & (df2['NODES_USED'] > 4)]
    print(f"\n  cb_nodes tuning recommendations: {len(mpiio_fixable):,} jobs")
    print(f"    → Automated: no user action needed, facility can set per-allocation defaults")

    # CPU queue redirect recommendations
    ghost_cpu = df2[(df2['crosslayer_tier'] == 'Ghost') & (df2['total_bytes'].fillna(0) == 0)]
    print(f"\n  CPU queue redirect recommendations: {len(ghost_cpu):,} Ghost jobs")
    print(f"    → GPU hours recoverable: {ghost_cpu['gpu_hours'].clip(lower=0).sum():,.0f}")

    # walltime recommendations (jobs using <10% of walltime)
    df2['wt_util'] = df2['RUNTIME_SECONDS'] / df2['WALLTIME_SECONDS'].replace(0, np.nan)
    low_wt = df2[df2['wt_util'] < 0.1]
    print(f"\n  Walltime reduction recommendations: {len(low_wt):,} jobs (using <10% of requested)")

    # user notification summary
    print(f"\n  Unique users who would receive at least one critical report:")
    critical_users = df2[df2['severity'] == 'critical']['USERNAME_GENID'].nunique()
    warning_users = df2[df2['severity'] == 'warning']['USERNAME_GENID'].nunique()
    print(f"    Critical: {critical_users:,} users")
    print(f"    Warning:  {warning_users:,} users")

    # repeat offenders — users who would receive >10 critical reports
    user_critical = df2[df2['severity'] == 'critical'].groupby('USERNAME_GENID').size()
    repeat = (user_critical > 10).sum()
    print(f"\n  Repeat offenders (>10 critical reports): {repeat:,} users")
    if repeat > 0:
        print(f"    Top 5:")
        for uid, count in user_critical.nlargest(5).items():
            u = str(int(uid))[-6:]
            print(f"      ···{u}: {count:,} critical reports")


# ─────────────────────────────────────────────────────────────────────────────
# RECOMMENDATION DISTRIBUTION
# ─────────────────────────────────────────────────────────────────────────────

def recommendation_distribution(df):
    """Analyze what types of recommendations the system would generate."""
    sep("RECOMMENDATION DISTRIBUTION")

    # sample 1000 jobs and generate reports to count recommendation types
    sample = df.sample(n=min(1000, len(df)), random_state=42)

    rec_counts = {
        'cpu_queue_redirect': 0,
        'cb_nodes_tuning': 0,
        'buffer_small_writes': 0,
        'reduce_walltime': 0,
        'gpu_verification': 0,
        'checkpointing': 0,
        'reduce_scale': 0,
        'batch_short_jobs': 0,
        'io_profiling': 0,
        'rank_rebalance': 0,
    }

    for _, job in sample.iterrows():
        tier = job.get('crosslayer_tier', '')
        total_bytes = job.get('total_bytes', 0) or 0
        nodes = job.get('NODES_USED', 0)
        gpu_util = job.get('gpu_util_mean', None)
        has_mpiio = job.get('has_mpiio', False)
        cb_nodes = job.get('cb_nodes', 0)
        small_write = job.get('small_write_ratio', 0)
        walltime_s = job.get('WALLTIME_SECONDS', 0)
        runtime_s = job.get('RUNTIME_SECONDS', 0)
        rank_imbalance = job.get('rank_imbalance', 0)
        bwio = job.get('BWio_MB', None)
        gpus = nodes * 4

        wt_util = runtime_s / walltime_s if walltime_s > 0 else 0

        if tier == 'Ghost' and total_bytes == 0:
            rec_counts['cpu_queue_redirect'] += 1
            rec_counts['gpu_verification'] += 1
        if tier in ['Ghost', 'Scale_Waster'] and wt_util < 0.1:
            rec_counts['reduce_walltime'] += 1
        if tier == 'Scale_Waster':
            rec_counts['reduce_scale'] += 1
        if has_mpiio and cb_nodes == 4 and nodes > 4:
            rec_counts['cb_nodes_tuning'] += 1
        if small_write > 0.5 and total_bytes > 0:
            rec_counts['buffer_small_writes'] += 1
        if tier == 'Failed_Job' and pd.notna(gpu_util) and gpu_util > 50:
            rec_counts['checkpointing'] += 1
        if tier == 'Short_Job' and gpus > 4:
            rec_counts['batch_short_jobs'] += 1
        if pd.notna(bwio) and bwio < 100 and total_bytes > 1e9:
            rec_counts['io_profiling'] += 1
        if rank_imbalance > 10 and total_bytes > 0:
            rec_counts['rank_rebalance'] += 1

    # scale to full dataset
    scale = len(df) / len(sample)
    print(f"Estimated recommendation distribution (scaled from {len(sample)} sample):\n")
    total_recs = 0
    for rec, count in sorted(rec_counts.items(), key=lambda x: -x[1]):
        scaled = int(count * scale)
        total_recs += scaled
        bar = '█' * int(count / max(rec_counts.values()) * 30) if max(rec_counts.values()) > 0 else ''
        print(f"  {rec:<25s} {scaled:>8,} jobs {bar}")

    print(f"\n  Total recommendations:   {total_recs:,}")
    print(f"  Unique recommendation types: {sum(1 for v in rec_counts.values() if v > 0)}")


# ─────────────────────────────────────────────────────────────────────────────
# PAPER CONTENT
# ─────────────────────────────────────────────────────────────────────────────

def paper_content(df):
    sep("PAPER CONTENT — AUTOMATED FEEDBACK LOOP")

    total = len(df)
    waste_tiers = ['Ghost', 'Scale_Waster', 'IO_Bottlenecked']
    waste = df[df['crosslayer_tier'].isin(waste_tiers)]
    ghost_cpu = df[(df['crosslayer_tier'] == 'Ghost') & (df['total_bytes'].fillna(0) == 0)]
    mpiio_fix = df[(df['has_mpiio'] == True) & (df['cb_nodes'] == 4) & (df['NODES_USED'] > 4)]

    print(f"--- PAPER PARAGRAPH (§6 Discussion — Operational Feedback Loop) ---\n")
    print(f"To demonstrate the operational utility of the cross-layer pipeline,")
    print(f"we implement an automated per-job diagnostic report generator that")
    print(f"translates cross-layer classification into plain-English feedback")
    print(f"with specific, actionable recommendations. Applied retrospectively")
    print(f"to the full Polaris dataset, the system would generate actionable")
    print(f"reports for {len(waste):,} waste-tier jobs ({len(waste)/total*100:.1f}% of submissions),")
    print(f"including {len(ghost_cpu):,} CPU-queue redirect recommendations for Ghost")
    print(f"jobs with zero I/O, and {len(mpiio_fix):,} cb\\_nodes tuning recommendations")
    print(f"for multi-node MPI-IO jobs using the facility default of 4 aggregator")
    print(f"nodes. The report system requires no manual analysis — classification,")
    print(f"diagnosis, and recommendation are fully automated from the three")
    print(f"telemetry sources, enabling a closed-loop operational workflow from")
    print(f"job completion to user notification.")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cfg = load_config(args.config)

    print("Loading combined metrics...", flush=True)
    df = pd.read_csv(cfg["combined_out"], low_memory=False)
    df["job_id"] = df["JOB_NAME"].str.split(".").str[0]
    print(f"  {len(df):,} jobs loaded")

    # generate example reports for each tier
    generate_tier_examples(df)

    # worst offender reports
    generate_worst_offender_reports(df)

    # deployment impact analysis
    deployment_statistics(df)

    # recommendation distribution
    recommendation_distribution(df)

    # paper content
    paper_content(df)

    print(f"\n{'='*80}")
    print(f"  DONE — pipe output to a file:")
    print(f"  python -m pipeline.stage11_job_reports --config config/config.json > job_reports.txt")
    print(f"{'='*80}")
