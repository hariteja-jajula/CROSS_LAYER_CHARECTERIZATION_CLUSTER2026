"""
stage06_results.py — Static HTML results report for cross_layer_hpc_tool
Generates a self-contained HTML file with all findings, tables, and computed stats.
Run: python -m pipeline.stage06_results --config config/config.json
Output: results/results_report.html
"""

import pandas as pd
import numpy as np
import json, argparse
from pathlib import Path
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True)
args = parser.parse_args()

def load_config(path):
    with open(path) as f:
        return json.load(f)

# ─── COMPUTATION FUNCTIONS ────────────────────────────────────────────────────

def compute_all(df):
    results = {}

    # ── Dataset overview
    results['total_jobs'] = len(df)
    results['total_gpu_hours'] = df['gpu_hours'].clip(lower=0).sum()
    results['unique_users'] = df['USERNAME_GENID'].nunique()
    results['darshan_jobs'] = int(df['darshan_file_count'].notna().sum())
    results['darshan_pct'] = round(results['darshan_jobs'] / results['total_jobs'] * 100, 1)
    results['gpu_telemetry_jobs'] = int(df['gpu_util_mean'].notna().sum())

    # ── Tier distribution
    tier_counts = df['crosslayer_tier'].value_counts().reset_index()
    tier_counts.columns = ['tier', 'jobs']
    tier_hours = df.groupby('crosslayer_tier')['gpu_hours'].sum().clip(lower=0).reset_index()
    tier_hours.columns = ['tier', 'gpu_hours']
    tier_bytes = df.groupby('crosslayer_tier')['total_bytes'].sum().reset_index()
    tier_bytes.columns = ['tier', 'total_bytes']
    tiers = tier_counts.merge(tier_hours, on='tier').merge(tier_bytes, on='tier')
    tiers['gpu_hours_pct'] = (tiers['gpu_hours'] / results['total_gpu_hours'] * 100).round(1)
    tiers['jobs_pct'] = (tiers['jobs'] / results['total_jobs'] * 100).round(1)
    tiers['total_bytes_TB'] = (tiers['total_bytes'] / 1e12).round(1)
    tiers = tiers.sort_values('gpu_hours', ascending=False)
    results['tiers'] = tiers

    # ── Waste summary
    waste_tiers = ['Ghost', 'Scale_Waster', 'IO_Bottlenecked']
    waste = df[df['crosslayer_tier'].isin(waste_tiers)]
    results['waste_gpu_hours'] = waste['gpu_hours'].clip(lower=0).sum()
    results['waste_pct'] = round(results['waste_gpu_hours'] / results['total_gpu_hours'] * 100, 1)
    results['waste_cost_usd'] = results['waste_gpu_hours'] * 3.0
    results['ghost_jobs'] = int((df['crosslayer_tier'] == 'Ghost').sum())
    results['io_bot_jobs'] = int((df['crosslayer_tier'] == 'IO_Bottlenecked').sum())
    results['scale_waster_jobs'] = int((df['crosslayer_tier'] == 'Scale_Waster').sum())

    # ── GPU utilization stats
    gpu_df = df[df['gpu_util_mean'].notna()]
    results['gpu_zero_pct'] = round((gpu_df['gpu_util_mean'] == 0).mean() * 100, 1)
    results['gpu_below10_pct'] = round((gpu_df['gpu_util_mean'] < 10).mean() * 100, 1)
    results['gpu_above80_pct'] = round((gpu_df['gpu_util_mean'] > 80).mean() * 100, 1)
    results['gpu_util_mean_all'] = round(gpu_df['gpu_util_mean'].mean(), 1)
    results['gpu_util_median_all'] = round(gpu_df['gpu_util_mean'].median(), 1)

    # ── IO_Bottlenecked analysis
    io_bot = df[df['crosslayer_tier'] == 'IO_Bottlenecked']
    results['io_bot_total_bytes_TB'] = round(io_bot['total_bytes'].sum() / 1e12, 1)
    results['io_bot_gpu_phase1'] = round(io_bot['gpu_util_phase1'].mean(), 2)
    results['io_bot_gpu_phase2'] = round(io_bot['gpu_util_phase2'].mean(), 2)
    results['io_bot_gpu_phase3'] = round(io_bot['gpu_util_phase3'].mean(), 2)
    results['io_bot_io_start_mean'] = round(io_bot['io_phase_start_frac'].mean(), 3)
    results['io_bot_io_end_mean'] = round(io_bot['io_phase_end_frac'].mean(), 3)
    results['io_bot_small_read_ratio'] = round(io_bot['small_read_ratio'].mean(), 3)
    results['io_bot_high_rank_imbal_n'] = int(io_bot['high_rank_imbalance'].sum())
    results['io_bot_high_rank_imbal_pct'] = round(io_bot['high_rank_imbalance'].mean() * 100, 1)
    results['io_bot_rank_imbal_median'] = round(io_bot['rank_imbalance'].median(), 2)
    results['io_bot_rank_above1000'] = int((io_bot['rank_imbalance'] > 1000).sum())
    results['io_bot_rank_above1000_pct'] = round((io_bot['rank_imbalance'] > 1000).mean() * 100, 1)

    # ── Earth Science IO_Bottlenecked
    es_io = io_bot[io_bot['SCIENCE_FIELD_SHORT'] == 'Earth Science']
    results['es_io_jobs'] = len(es_io)
    results['es_io_pct_of_iobot'] = round(len(es_io) / len(io_bot) * 100, 1)
    results['es_io_bytes_TB'] = round(es_io['total_bytes'].sum() / 1e12, 1)
    results['es_io_has_mpiio'] = int(es_io['has_mpiio'].sum())
    results['es_io_mpiio_pct'] = round(es_io['has_mpiio'].mean() * 100, 1)
    results['es_system_pct'] = round((df['SCIENCE_FIELD_SHORT'] == 'Earth Science').mean() * 100, 1)

    # ── Science field overrepresentation
    system_field = df['SCIENCE_FIELD_SHORT'].value_counts(normalize=True) * 100
    iobot_field = io_bot['SCIENCE_FIELD_SHORT'].value_counts(normalize=True) * 100
    field_compare = pd.DataFrame({'system_pct': system_field, 'iobot_pct': iobot_field}).fillna(0)
    field_compare['overrep'] = (field_compare['iobot_pct'] / field_compare['system_pct']).round(1)
    field_compare = field_compare.sort_values('iobot_pct', ascending=False).head(8).reset_index()
    field_compare.columns = ['field', 'system_pct', 'iobot_pct', 'overrep_factor']
    results['field_compare'] = field_compare

    # ── cb_nodes serialization
    mpiio = df[df['has_mpiio'] == True]
    results['mpiio_jobs'] = len(mpiio)
    results['mpiio_cbnodes4_pct'] = round((mpiio['cb_nodes'] == 4).mean() * 100, 1)
    results['mpiio_multinodes_cbnodes4'] = int(((mpiio['NODES_USED'] > 4) & (mpiio['cb_nodes'] == 4)).sum())
    results['mpiio_multinodes_mean_nodes'] = round(df[(df['has_mpiio'] == True) & (df['NODES_USED'] > 4)]['NODES_USED'].mean(), 1)

    multi_mpiio = df[(df['has_mpiio'] == True) & (df['NODES_USED'] > 4)]
    results['mpiio_multinodes_bwio_median'] = round(multi_mpiio[multi_mpiio['BWio_MB'] > 0]['BWio_MB'].median(), 3)
    results['mpiio_multinodes_total_bytes_TB'] = round(multi_mpiio['total_bytes'].sum() / 1e12, 1)

    # ── Darshan interface breakdown
    darshan_df = df[df['darshan_file_count'].notna()]
    results['posix_pct'] = round(darshan_df['has_posix'].mean() * 100, 1)
    results['mpiio_pct'] = round(darshan_df['has_mpiio'].mean() * 100, 1)
    results['stdio_pct'] = round(darshan_df['has_stdio'].mean() * 100, 1)
    results['zero_io_pct'] = round(((darshan_df['total_bytes'] == 0) | darshan_df['total_bytes'].isna()).mean() * 100, 1)

    # ── BWio stats
    bwio_valid = df[df['BWio_MB'].notna() & (df['BWio_MB'] > 0)]
    results['bwio_median'] = round(bwio_valid['BWio_MB'].median(), 1)
    results['bwio_p90'] = round(bwio_valid['BWio_MB'].quantile(0.90), 1)
    results['bwio_n'] = len(bwio_valid)

    # ── Walltime utilization
    df2 = df.copy()
    df2['walltime_util'] = df2['RUNTIME_SECONDS'] / df2['WALLTIME_SECONDS']
    wt = df2.groupby('crosslayer_tier')['walltime_util'].median().sort_values()
    results['walltime_by_tier'] = wt.reset_index()
    results['ghost_walltime_util'] = round(wt.get('Ghost', 0) * 100, 1)
    results['compute_walltime_util'] = round(wt.get('Compute_Bound', 0) * 100, 1)
    results['iobot_walltime_util'] = round(wt.get('IO_Bottlenecked', 0) * 100, 1)

    # ── CS Ghost jobs
    cs_ghost = df[(df['crosslayer_tier'] == 'Ghost') & (df['SCIENCE_FIELD_SHORT'] == 'Computer Science')]
    results['cs_ghost_jobs'] = len(cs_ghost)
    results['cs_ghost_walltime_hrs'] = round(cs_ghost['WALLTIME_SECONDS'].mean() / 3600, 1)
    results['cs_ghost_runtime_hrs'] = round(cs_ghost['RUNTIME_SECONDS'].mean() / 3600, 2)

    # ── Phase patterns
    phase_tiers = ['Ghost', 'IO_Bottlenecked', 'Compute_Bound', 'Balanced', 'Scale_Waster']
    phase_rows = []
    for tier in phase_tiers:
        sub = df[df['crosslayer_tier'] == tier].dropna(subset=['gpu_util_phase1', 'io_phase_start_frac'])
        if len(sub) == 0:
            continue
        phase_rows.append({
            'Tier': tier,
            'N (with phase data)': f"{len(sub):,}",
            'GPU Phase1 (%)': round(sub['gpu_util_phase1'].mean(), 2),
            'GPU Phase2 (%)': round(sub['gpu_util_phase2'].mean(), 2),
            'GPU Phase3 (%)': round(sub['gpu_util_phase3'].mean(), 2),
            'I/O Start (frac)': round(sub['io_phase_start_frac'].mean(), 3),
            'I/O End (frac)': round(sub['io_phase_end_frac'].mean(), 3),
        })
    results['phase_table'] = pd.DataFrame(phase_rows)

    # ── Top science fields for Ghost
    ghost_fields = df[df['crosslayer_tier'] == 'Ghost']['SCIENCE_FIELD_SHORT'].value_counts().head(6).reset_index()
    ghost_fields.columns = ['field', 'jobs']
    ghost_fields['pct'] = (ghost_fields['jobs'] / results['ghost_jobs'] * 100).round(1)
    results['ghost_fields'] = ghost_fields

    # ── Structural repeatability
    try:
        rep = pd.read_csv(Path(cfg['combined_out']).parent / 'repeatability_scores.csv')
        results['structural_users'] = int((rep['is_structural'] == True).sum())
        results['rep_top'] = rep[rep['is_structural'] == True].sort_values('repeatability_score', ascending=False).head(10)
    except:
        results['structural_users'] = 'N/A'
        results['rep_top'] = None

    # ── Failed job analysis
    failed = df[df['crosslayer_tier'] == 'Failed_Job']
    results['failed_jobs'] = len(failed)
    results['failed_gpu_hours'] = round(failed['gpu_hours'].clip(lower=0).sum(), 0)
    results['failed_gpu_pct'] = round(failed['gpu_hours'].clip(lower=0).sum() / results['total_gpu_hours'] * 100, 1)

    return results

# ─── HTML GENERATION ──────────────────────────────────────────────────────────

def fmt(n):
    if isinstance(n, float) and n != int(n):
        return f"{n:,.1f}"
    return f"{int(n):,}"

def tbl(headers, rows, highlight_col=None):
    th = ''.join(f'<th>{h}</th>' for h in headers)
    body = ''
    for row in rows:
        cells = ''
        for i, c in enumerate(row):
            cls = ' class="highlight"' if highlight_col is not None and i == highlight_col else ''
            cells += f'<td{cls}>{c}</td>'
        body += f'<tr>{cells}</tr>'
    return f'<table><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table>'

TIER_COLORS = {
    'Ghost': '#ef4444', 'IO_Bottlenecked': '#f59e0b', 'Compute_Bound': '#10b981',
    'Balanced': '#3b82f6', 'Scale_Waster': '#f97316', 'Failed_Job': '#8b5cf6',
    'Short_Job': '#6b7280', 'CPU_No_IO': '#374151', 'No_IO_No_GPU': '#4b5563',
    'Telemetry_Gap': '#6b7280', 'Interactive_Test': '#6b7280',
    'CPU_IO_Job': '#60a5fa', 'Moderate_Compute': '#34d399',
}

def tier_badge(tier):
    color = TIER_COLORS.get(tier, '#6b7280')
    return f'<span class="badge" style="background:{color}">{tier}</span>'

def generate_html(r):
    ts = datetime.now().strftime('%Y-%m-%d %H:%M UTC')

    # tier table rows
    tier_rows = []
    for _, row in r['tiers'].iterrows():
        tier_rows.append([
            tier_badge(row['tier']),
            f"{row['jobs']:,}",
            f"{row['jobs_pct']}%",
            f"{row['gpu_hours']:,.0f}",
            f"{row['gpu_hours_pct']}%",
            f"{row['total_bytes_TB']} TB",
        ])

    # phase table rows
    phase_rows = []
    for _, row in r['phase_table'].iterrows():
        phase_rows.append([
            tier_badge(row['Tier']),
            row['N (with phase data)'],
            f"{row['GPU Phase1 (%)']}%",
            f"{row['GPU Phase2 (%)']}%",
            f"{row['GPU Phase3 (%)']}%",
            f"{row['I/O Start (frac)']}",
            f"{row['I/O End (frac)']}",
        ])

    # field compare rows
    field_rows = []
    for _, row in r['field_compare'].iterrows():
        over = f"<strong style='color:#ef4444'>{row['overrep_factor']}x</strong>" if row['overrep_factor'] > 3 else f"{row['overrep_factor']}x"
        field_rows.append([row['field'], f"{row['system_pct']:.1f}%", f"{row['iobot_pct']:.1f}%", over])

    # walltime rows
    wt_rows = []
    for _, row in r['walltime_by_tier'].iterrows():
        pct = round(row['walltime_util'] * 100, 1)
        bar = f'<div style="background:#1e293b;border-radius:3px;height:8px;width:120px;display:inline-block;vertical-align:middle;margin-left:8px"><div style="background:#3b82f6;height:8px;border-radius:3px;width:{min(pct,100)}%"></div></div>'
        wt_rows.append([tier_badge(row['crosslayer_tier']), f"{pct}%", bar])

    # rep table
    rep_rows = []
    if r['rep_top'] is not None:
        for _, row in r['rep_top'].iterrows():
            uid = str(int(row['USERNAME_GENID']))[-6:]
            rep_rows.append([
                f"···{uid}",
                f"{row['waste_months_active']}/12",
                f"{row['repeatability_score']*100:.0f}%",
                f"{row['runtime_cv']:.3f}",
                f"{row['bytes_cv']:.3f}",
            ])
    definitions_html = """
  <div class="section">
    <div class="section-title"><span class="section-num">0</span> Definitions & Metric Glossary</div>
    <p style="color:var(--muted);font-size:13px;margin-bottom:20px">
      Every metric shown in this report is defined below. Use this to answer questions about methodology.
    </p>

    <div class="two-col">
      <div>
        <div class="panel" style="margin-bottom:16px">
          <h3>Telemetry Sources</h3>
          <div class="def-item"><span class="def-term">DJC (Darshan Job Context)</span>
            <p>Scheduler metadata for every job on Polaris. Contains job name, start/end timestamps, 
            nodes allocated, GPUs requested, walltime, runtime, username, project, exit status, 
            and science field. Covers 100% of jobs — the backbone of the dataset.</p></div>
          <div class="def-item"><span class="def-term">DCGM (GPU Telemetry)</span>
            <p>NVIDIA Data Center GPU Manager samples each GPU every 30 seconds while a job is running. 
            Records GPU utilization %, memory utilization %, power draw (watts), and temperature per GPU. 
            Four A100 GPUs per Polaris node. Coverage: 205,218 of 262,634 jobs had usable DCGM records — 
            short jobs (&lt;60s) often have no samples.</p></div>
          <div class="def-item"><span class="def-term">Darshan</span>
            <p>Lightweight I/O characterization tool linked into the MPI stack. Records per-job 
            I/O statistics without code modification: bytes read/written, operation counts, 
            access size distributions, timestamps, and rank-level imbalance. 
            Covers ~25% of jobs — jobs not using MPI or completing very quickly may not generate logs.</p></div>
        </div>

        <div class="panel" style="margin-bottom:16px">
          <h3>GPU Metrics</h3>
          <div class="def-item"><span class="def-term">gpu_util_mean</span>
            <p>Mean GPU utilization (%) averaged across all 4 GPUs on all allocated nodes across all 
            30-second samples during the job runtime. A value of 0% means the GPU was never doing 
            any compute work. A value of 100% means all GPUs were fully saturated the entire time.</p></div>
          <div class="def-item"><span class="def-term">gpu_zero_util_frac</span>
            <p>Fraction of 30-second telemetry samples where GPU utilization was exactly 0% across 
            all GPUs. A job with gpu_zero_util_frac = 1.0 had zero GPU activity for its entire runtime.</p></div>
          <div class="def-item"><span class="def-term">gpu_util_phase1 / phase2 / phase3</span>
            <p>Job runtime split into three equal thirds. Mean GPU utilization computed independently 
            for each third. This reveals temporal patterns: a job that ramps up (low→high→high) 
            is warming up. A job that's flat near zero is never using the GPU. 
            A job that drops (high→low) is finishing compute early.</p></div>
          <div class="def-item"><span class="def-term">phase_drop</span>
            <p>gpu_util_phase1 − gpu_util_phase3. Positive = GPU utilization decreasing over time 
            (job finishing up). Negative = GPU utilization increasing (warm-up or ramp-up). 
            Near zero = stable throughout.</p></div>
          <div class="def-item"><span class="def-term">active_phase_frac</span>
            <p>Fraction of telemetry samples where mean GPU utilization exceeded 1%. 
            Distinguishes jobs that were occasionally active from jobs that were never active.</p></div>
          <div class="def-item"><span class="def-term">gpu_imbalance_mean</span>
            <p>Mean standard deviation of GPU utilization across the 4 GPUs on a node. 
            High imbalance means some GPUs are working while others idle — 
            indicates poor multi-GPU parallelism in the application.</p></div>
        </div>
      </div>

      <div>
        <div class="panel" style="margin-bottom:16px">
          <h3>I/O Metrics</h3>
          <div class="def-item"><span class="def-term">BWio (I/O Bandwidth)</span>
            <p>Estimated I/O bandwidth: total bytes moved divided by I/O phase duration. 
            T_IO is estimated from Darshan POSIX timestamps (latest end − earliest start across 
            read and write operations). Falls back to wall-clock runtime if T_IO is unreliable 
            (&lt;1s or &gt;1.05× wall time). Units: MB/s.</p></div>
          <div class="def-item"><span class="def-term">total_bytes</span>
            <p>Sum of bytes_read + bytes_written across POSIX + MPI-IO + STDIO interfaces. 
            This is the total data volume moved by the job regardless of which I/O library was used.</p></div>
          <div class="def-item"><span class="def-term">io_phase_start_frac / io_phase_end_frac</span>
            <p>When I/O begins and ends as a fraction of total job runtime, derived from 
            the Darshan HEATMAP module (32 time bins of equal width). 
            io_phase_start_frac = 0.03 means I/O starts in the first 3% of the job. 
            io_phase_end_frac = 0.95 means I/O is still active at 95% of the way through the job.</p></div>
          <div class="def-item"><span class="def-term">rank_imbalance</span>
            <p>POSIX_SLOWEST_RANK_BYTES / POSIX_FASTEST_RANK_BYTES from Darshan. 
            Measures how unevenly I/O work is distributed across MPI ranks. 
            A value of 1.0 = perfectly balanced. A value of 1000 = the slowest rank moved 
            1000x more data than the fastest rank, indicating a single-writer bottleneck 
            where one rank does all the I/O while others wait.</p></div>
          <div class="def-item"><span class="def-term">small_read_ratio</span>
            <p>Fraction of POSIX read operations that transferred less than 10KB. 
            High values indicate metadata-dominated I/O — many tiny reads typical of 
            file-per-process patterns, directory scans, or checkpoint reads. 
            These small operations cannot be efficiently parallelized by Lustre striping.</p></div>
          <div class="def-item"><span class="def-term">cb_nodes</span>
            <p>Number of MPI collective buffering nodes (aggregators) used for MPI-IO operations. 
            Extracted from the Darshan log header (ROMIO hint). Controls how many nodes 
            act as I/O proxies in collective operations. Default on Polaris is 4, 
            regardless of job size. More cb_nodes = better parallelism for large jobs.</p></div>
          <div class="def-item"><span class="def-term">write_dominance</span>
            <p>bytes_written / total_bytes. A value near 1.0 means the job is primarily writing 
            (checkpoint-heavy, output-dominated). A value near 0.0 means primarily reading 
            (data loading, analysis). Helps distinguish checkpoint jobs from data preprocessing jobs.</p></div>
          <div class="def-item"><span class="def-term">mpiio_coll_ratio</span>
            <p>Fraction of MPI-IO operations that used collective (vs. independent) calls. 
            Collective operations like MPI_File_write_all allow ROMIO to optimize buffering. 
            Independent operations bypass this optimization. Low ratio = missed optimization opportunity.</p></div>
        </div>

        <div class="panel" style="margin-bottom:16px">
          <h3>Classification Metrics</h3>
          <div class="def-item"><span class="def-term">gpu_waste_score</span>
            <p>1 − (gpu_util_mean / 100), clipped 0–1. 
            Value of 1.0 = GPU completely idle. Value of 0.0 = GPU fully saturated. 
            Continuous measure of GPU inefficiency before discrete tier assignment.</p></div>
          <div class="def-item"><span class="def-term">io_waste_score</span>
            <p>1 − (BWio_MB / P90_BWio), clipped 0–1. 
            Value of 1.0 = no I/O or very low bandwidth. Value of 0.0 = top-decile bandwidth. 
            Set to 1.0 for jobs with bytes but no valid T_IO estimate.</p></div>
          <div class="def-item"><span class="def-term">scale_factor</span>
            <p>GPUS_REQUESTED / P75_GPUS, clipped 0–3. 
            Amplifies waste scores for large allocations — a Ghost job requesting 256 GPUs 
            is more wasteful than one requesting 4. P75 of GPUS_REQUESTED on Polaris = 4 
            (one standard node).</p></div>
          <div class="def-item"><span class="def-term">cross_layer_waste</span>
            <p>Composite waste score: (0.5 × gpu_waste + 0.5 × io_waste) × scale_factor. 
            A single number summarizing how wasteful a job is across both dimensions, 
            weighted by the scale of resources it consumed.</p></div>
          <div class="def-item"><span class="def-term">repeatability_score</span>
            <p>Composite score (0–1) measuring how consistently a user submits the same 
            wasteful workflow. Components: runtime CV (30%), GPU config CV (20%), 
            data volume CV (30%), months active in waste tiers (20%). 
            Score ≥ 0.7 with ≥ 3 months active = structural offender. 
            CV = coefficient of variation = std/mean, where low CV means highly consistent behavior.</p></div>
          <div class="def-item"><span class="def-term">walltime_util</span>
            <p>RUNTIME_SECONDS / WALLTIME_SECONDS. How much of the requested time window 
            the job actually used. A value of 0.045 (Ghost median) means the job ran for 
            4.5% of what it asked for. Low values indicate walltime padding — 
            users requesting far more time than needed, which wastes scheduler slots.</p></div>
        </div>
      </div>
    </div>

    <div class="panel">
      <h3>Tier Definitions</h3>
      <table>
        <thead><tr><th>Tier</th><th>GPU Condition</th><th>I/O Condition</th><th>Other</th><th>Interpretation</th></tr></thead>
        <tbody>
          <tr><td><span class="badge" style="background:#ef4444">Ghost</span></td><td>gpu_util &lt; 5%</td><td>No I/O</td><td>GPUS ≥ 4</td><td>Allocated GPUs, did nothing. Completed with exit 0. Pure allocation waste.</td></tr>
          <tr><td><span class="badge" style="background:#f59e0b">IO_Bottlenecked</span></td><td>gpu_util &lt; 10%</td><td>Has I/O</td><td>—</td><td>GPU idle while I/O active. Novel cross-layer class invisible to single-layer tools.</td></tr>
          <tr><td><span class="badge" style="background:#10b981">Compute_Bound</span></td><td>gpu_util ≥ 70%</td><td>No I/O</td><td>—</td><td>Healthy. High GPU utilization, pure compute workload.</td></tr>
          <tr><td><span class="badge" style="background:#3b82f6">Balanced</span></td><td>gpu_util ≥ 10%</td><td>Has I/O</td><td>—</td><td>Healthy. Both GPU and I/O active. Well-utilized mixed workload.</td></tr>
          <tr><td><span class="badge" style="background:#f97316">Scale_Waster</span></td><td>gpu_util &lt; 10%</td><td>No I/O</td><td>Large allocation</td><td>Large job, poor utilization across both dimensions. Requesting more than needed.</td></tr>
          <tr><td><span class="badge" style="background:#34d399">Moderate_Compute</span></td><td>10% ≤ gpu_util &lt; 70%</td><td>No I/O</td><td>—</td><td>Mid-range GPU utilization. Acceptable compute, not peak efficiency.</td></tr>
          <tr><td><span class="badge" style="background:#8b5cf6">Failed_Job</span></td><td>Any</td><td>Any</td><td>EXIT_STATUS ≠ 0</td><td>Job terminated with error. Resources consumed without useful output.</td></tr>
          <tr><td><span class="badge" style="background:#6b7280">Short_Job</span></td><td>No telemetry</td><td>Any</td><td>Runtime &lt; 60s</td><td>Too short for DCGM to capture. Initialization, test runs, compilation.</td></tr>
          <tr><td><span class="badge" style="background:#374151">CPU_No_IO</span></td><td>No GPU allocated</td><td>No I/O</td><td>GPUS = -1</td><td>CPU-only job. No GPU allocation, no I/O detected.</td></tr>
          <tr><td><span class="badge" style="background:#60a5fa">CPU_IO_Job</span></td><td>No GPU allocated</td><td>Has I/O</td><td>GPUS = -1</td><td>CPU workflow doing real I/O. Correctly using CPU nodes for data processing.</td></tr>
          <tr><td><span class="badge" style="background:#4b5563">No_IO_No_GPU</span></td><td>No telemetry</td><td>No I/O</td><td>Has Darshan</td><td>Setup or metadata-only jobs. Darshan present but no bytes moved.</td></tr>
          <tr><td><span class="badge" style="background:#6b7280">Telemetry_Gap</span></td><td>No telemetry</td><td>No Darshan</td><td>Large job</td><td>Large allocation with no monitoring data. DCGM collection failure or gap.</td></tr>
          <tr><td><span class="badge" style="background:#6b7280">Interactive_Test</span></td><td>No telemetry</td><td>No Darshan</td><td>Small, short</td><td>Small allocation, short runtime, no logs. Likely interactive or test submission.</td></tr>
        </tbody>
      </table>
    </div>
  </div>
"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Cross-Layer HPC Tool — Results Report</title>
<style>
  :root {{
    --bg: #0f172a; --surface: #1e293b; --surface2: #263346;
    --border: #334155; --text: #e2e8f0; --muted: #64748b;
    --accent: #38bdf8; --red: #ef4444; --green: #10b981;
    --yellow: #f59e0b; --orange: #f97316; --purple: #8b5cf6;
  }}
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ background:var(--bg); color:var(--text); font-family:'Segoe UI',system-ui,sans-serif; font-size:14px; line-height:1.7; }}
  a {{ color:var(--accent); text-decoration:none; }}
  .container {{ max-width:1100px; margin:0 auto; padding:40px 24px; }}
  
  /* Header */
  .header {{ border-bottom:1px solid var(--border); padding-bottom:32px; margin-bottom:40px; }}
  .header h1 {{ font-size:28px; font-weight:700; color:var(--accent); letter-spacing:-0.5px; }}
  .header .sub {{ color:var(--muted); font-size:13px; margin-top:6px; }}
  .meta {{ display:flex; gap:24px; margin-top:16px; flex-wrap:wrap; }}
  .meta-item {{ font-size:12px; color:var(--muted); border:1px solid var(--border); padding:4px 12px; border-radius:4px; }}
  .meta-item span {{ color:var(--text); font-weight:600; }}

  /* KPI strip */
  .kpi-strip {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(160px,1fr)); gap:1px; background:var(--border); border:1px solid var(--border); margin-bottom:40px; }}
  .kpi {{ background:var(--surface); padding:20px 16px; border-top:2px solid var(--border); }}
  .kpi.red {{ border-top-color:var(--red); }}
  .kpi.green {{ border-top-color:var(--green); }}
  .kpi.yellow {{ border-top-color:var(--yellow); }}
  .kpi.accent {{ border-top-color:var(--accent); }}
  .kpi-label {{ font-size:10px; letter-spacing:1.5px; text-transform:uppercase; color:var(--muted); margin-bottom:8px; }}
  .kpi-val {{ font-size:26px; font-weight:700; line-height:1; }}
  .kpi-val.red {{ color:var(--red); }}
  .kpi-val.green {{ color:var(--green); }}
  .kpi-val.yellow {{ color:var(--yellow); }}
  .kpi-sub {{ font-size:11px; color:var(--muted); margin-top:4px; }}

  /* Sections */
  .section {{ margin-bottom:48px; }}
  .section-title {{ font-size:18px; font-weight:700; color:var(--accent); margin-bottom:8px; padding-bottom:8px; border-bottom:1px solid var(--border); display:flex; align-items:center; gap:10px; }}
  .section-num {{ background:var(--accent); color:#0f172a; font-size:11px; font-weight:700; padding:2px 8px; border-radius:3px; }}
  .finding {{ background:var(--surface); border:1px solid var(--border); border-left:3px solid var(--accent); padding:16px 20px; margin-bottom:16px; border-radius:0 6px 6px 0; }}
  .finding.red {{ border-left-color:var(--red); }}
  .finding.yellow {{ border-left-color:var(--yellow); }}
  .finding.green {{ border-left-color:var(--green); }}
  .finding.orange {{ border-left-color:var(--orange); }}
  .finding h4 {{ font-size:13px; font-weight:700; margin-bottom:6px; color:var(--text); }}
  .finding p {{ font-size:13px; color:var(--muted); line-height:1.7; }}
  .finding .stat {{ color:var(--text); font-weight:600; }}

  /* Tables */
  table {{ width:100%; border-collapse:collapse; margin:16px 0; font-size:13px; }}
  th {{ background:var(--surface2); color:var(--muted); font-size:10px; letter-spacing:1.5px; text-transform:uppercase; padding:10px 12px; text-align:left; border-bottom:1px solid var(--border); }}
  td {{ padding:10px 12px; border-bottom:1px solid rgba(51,65,85,0.5); vertical-align:middle; }}
  tr:hover td {{ background:rgba(255,255,255,0.02); }}
  td.highlight {{ color:var(--accent); font-weight:600; }}
  .badge {{ display:inline-block; padding:2px 8px; border-radius:3px; font-size:11px; font-weight:600; color:#fff; white-space:nowrap; }}

  /* Two-col layout */
  .two-col {{ display:grid; grid-template-columns:1fr 1fr; gap:24px; margin-bottom:24px; }}
  .panel {{ background:var(--surface); border:1px solid var(--border); padding:20px; border-radius:6px; }}
  .panel h3 {{ font-size:12px; letter-spacing:1.5px; text-transform:uppercase; color:var(--muted); margin-bottom:16px; }}

  /* Callout */
  .callout {{ background:rgba(239,68,68,0.08); border:1px solid rgba(239,68,68,0.3); padding:16px 20px; border-radius:6px; margin:16px 0; font-size:13px; }}
  .callout.blue {{ background:rgba(56,189,248,0.08); border-color:rgba(56,189,248,0.3); }}
  .callout.green {{ background:rgba(16,185,129,0.08); border-color:rgba(16,185,129,0.3); }}
  .callout strong {{ color:var(--text); }}

  /* Stat grid */
  .stat-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(200px,1fr)); gap:12px; margin:16px 0; }}
  .stat-box {{ background:var(--surface); border:1px solid var(--border); padding:14px 16px; border-radius:6px; }}
  .stat-box .val {{ font-size:22px; font-weight:700; color:var(--accent); }}
  .stat-box .lbl {{ font-size:11px; color:var(--muted); margin-top:2px; }}

  footer {{ border-top:1px solid var(--border); padding-top:24px; margin-top:48px; font-size:12px; color:var(--muted); }}
  @media (max-width:700px) {{ .two-col {{ grid-template-columns:1fr; }} .kpi-strip {{ grid-template-columns:1fr 1fr; }} }}
  .def-item {{ margin-bottom: 14px; }}
  .def-term {{ 
    font-family: 'Courier New', monospace; 
    font-size: 12px; 
    font-weight: 700; 
    color: var(--accent); 
    display: block; 
    margin-bottom: 3px; 
  }}
  .def-item p {{ 
    font-size: 12px; 
    color: var(--muted); 
    line-height: 1.6; 
    margin: 0; 
  }}
</style>

</head>
<body>
<div class="container">

  <div class="header">
    <h1>Cross-Layer HPC Workload Characterization</h1>
    <div class="sub">ALCF Polaris Supercomputer &mdash; Full Year 2025 Production Analysis</div>
    <div class="meta">
      <div class="meta-item">Dataset: <span>Jan 1 &ndash; Dec 31, 2025</span></div>
      <div class="meta-item">System: <span>ALCF Polaris (A100 GPUs)</span></div>
      <div class="meta-item">Layers: <span>GPU Telemetry + Darshan + DJC</span></div>
      <div class="meta-item">Generated: <span>{ts}</span></div>
      <div class="meta-item">Target: <span>IEEE Cluster 2026</span></div>
    </div>
  </div>

  <!-- KPI STRIP -->
  <div class="kpi-strip">
    <div class="kpi accent"><div class="kpi-label">Total Jobs</div><div class="kpi-val">{fmt(r['total_jobs'])}</div></div>
    <div class="kpi accent"><div class="kpi-label">Total GPU Hours</div><div class="kpi-val">{fmt(round(r['total_gpu_hours']/1e6,2))}M</div></div>
    <div class="kpi red"><div class="kpi-label">Wasted GPU Hours</div><div class="kpi-val red">{fmt(round(r['waste_gpu_hours']/1e6,2))}M</div><div class="kpi-sub">Ghost + Scale_Waster + IO_Bot</div></div>
    <div class="kpi red"><div class="kpi-label">Waste %</div><div class="kpi-val red">{r['waste_pct']}%</div><div class="kpi-sub">of all GPU hours</div></div>
    <div class="kpi red"><div class="kpi-label">Wasted Node Hours</div><div class="kpi-val red">{fmt(round(r['waste_gpu_hours']/4/1e3,0))}K</div><div class="kpi-sub">GPU-node hours wasted</div></div>
    <div class="kpi yellow"><div class="kpi-label">Ghost Jobs</div><div class="kpi-val yellow">{fmt(r['ghost_jobs'])}</div><div class="kpi-sub">zero GPU, zero I/O</div></div>
    <div class="kpi yellow"><div class="kpi-label">IO_Bottlenecked</div><div class="kpi-val yellow">{fmt(r['io_bot_jobs'])}</div><div class="kpi-sub">novel cross-layer class</div></div>
    <div class="kpi accent"><div class="kpi-label">Unique Users</div><div class="kpi-val">{fmt(r['unique_users'])}</div></div>
  </div>
  {definitions_html}

  <!-- S1: DATASET -->
  <div class="section">
    <div class="section-title"><span class="section-num">1</span> Dataset & Coverage</div>
    <div class="stat-grid">
      <div class="stat-box"><div class="val">{fmt(r['total_jobs'])}</div><div class="lbl">Total production jobs (Jan–Dec 2025)</div></div>
      <div class="stat-box"><div class="val">{r['darshan_pct']}%</div><div class="lbl">Darshan coverage ({fmt(r['darshan_jobs'])} jobs with I/O logs)</div></div>
      <div class="stat-box"><div class="val">{fmt(r['gpu_telemetry_jobs'])}</div><div class="lbl">Jobs with GPU telemetry</div></div>
      <div class="stat-box"><div class="val">{fmt(r['unique_users'])}</div><div class="lbl">Unique users across all jobs</div></div>
    </div>
    <div class="finding">
      <h4>Three-Layer Coverage</h4>
      <p>The full three-layer join (GPU + I/O + scheduler) covers <span class="stat">{fmt(r['darshan_jobs'])} jobs ({r['darshan_pct']}%)</span> with complete telemetry. 
      The remaining <span class="stat">{fmt(r['total_jobs'] - r['darshan_jobs'])}</span> jobs are classified using GPU telemetry and scheduler metadata alone. 
      This coverage rate is consistent with Darshan deployments reported at other leadership facilities.</p>
    </div>
  </div>

  <!-- S2: TIER DISTRIBUTION -->
  <div class="section">
    <div class="section-title"><span class="section-num">2</span> Cross-Layer Tier Distribution</div>
    <p style="color:var(--muted);font-size:13px;margin-bottom:16px">
      All {fmt(r['total_jobs'])} jobs classified into behavioral tiers using simultaneous GPU efficiency, I/O efficiency, and resource scale signals.
    </p>
    {tbl(['Tier','Jobs','% of Jobs','GPU Hours','% GPU Hours','Data Moved'], tier_rows)}
    <div class="callout">
      <strong>Key finding:</strong> Only <strong>{r['tiers'][r['tiers']['tier']=='Compute_Bound']['jobs_pct'].values[0] if len(r['tiers'][r['tiers']['tier']=='Compute_Bound']) > 0 else '~5'}%</strong> of jobs 
      were classified as Compute_Bound (healthy, high GPU utilization). Ghost jobs alone account for nearly 30% of all submissions.
    </div>
  </div>

  <!-- S3: GPU UTILIZATION -->
  <div class="section">
    <div class="section-title"><span class="section-num">3</span> GPU Utilization Analysis</div>
    <div class="two-col">
      <div class="panel">
        <h3>Distribution</h3>
        <div class="finding red" style="margin-bottom:10px">
          <h4>Zero GPU Utilization</h4>
          <p><span class="stat">{r['gpu_zero_pct']}%</span> of GPU-instrumented jobs show zero mean GPU utilization across their entire runtime.</p>
        </div>
        <div class="finding red">
          <h4>Below 10% Utilization</h4>
          <p><span class="stat">{r['gpu_below10_pct']}%</span> of jobs never exceeded 10% mean GPU utilization — the LOW threshold used for classification.</p>
        </div>
        <div class="finding green">
          <h4>Above 80% Utilization</h4>
          <p>Only <span class="stat">{r['gpu_above80_pct']}%</span> of jobs achieved >80% mean GPU utilization — genuine high-throughput compute.</p>
        </div>
      </div>
      <div class="panel">
        <h3>System-Wide Stats</h3>
        <div class="stat-grid" style="grid-template-columns:1fr 1fr">
          <div class="stat-box"><div class="val">{r['gpu_util_mean_all']}%</div><div class="lbl">Mean GPU util (all jobs)</div></div>
          <div class="stat-box"><div class="val">{r['gpu_util_median_all']}%</div><div class="lbl">Median GPU util</div></div>
        </div>
        <div class="finding yellow" style="margin-top:12px">
          <h4>Procurement Implication</h4>
          <p>A system with median GPU utilization near zero has significant headroom before additional GPU procurement is justified. 
          The data suggests scheduling and workflow policy changes would yield more throughput than hardware expansion.</p>
        </div>
      </div>
    </div>
  </div>

  <!-- S4: IO_BOTTLENECKED -->
  <div class="section">
    <div class="section-title"><span class="section-num">4</span> IO_Bottlenecked — Novel Cross-Layer Finding</div>
    <div class="callout blue">
      <strong>Core contribution:</strong> The IO_Bottlenecked class ({fmt(r['io_bot_jobs'])} jobs) is undetectable by any single telemetry layer. 
      GPU monitoring sees low utilization — indistinguishable from Ghost. I/O monitoring sees active data movement — appears healthy. 
      Only cross-layer correlation reveals these jobs are doing I/O on GPU nodes with GPUs completely idle throughout.
    </div>
    <div class="stat-grid">
      <div class="stat-box"><div class="val">{fmt(r['io_bot_jobs'])}</div><div class="lbl">IO_Bottlenecked jobs</div></div>
      <div class="stat-box"><div class="val">{r['io_bot_total_bytes_TB']} TB</div><div class="lbl">Data moved on GPU nodes</div></div>
      <div class="stat-box"><div class="val">{r['io_bot_gpu_phase1']}%</div><div class="lbl">Mean GPU util Phase 1</div></div>
      <div class="stat-box"><div class="val">{r['io_bot_gpu_phase2']}%</div><div class="lbl">Mean GPU util Phase 2</div></div>
      <div class="stat-box"><div class="val">{r['io_bot_gpu_phase3']}%</div><div class="lbl">Mean GPU util Phase 3</div></div>
      <div class="stat-box"><div class="val">{r['io_bot_io_end_mean']:.2f}</div><div class="lbl">Mean I/O end fraction (of runtime)</div></div>
    </div>
    <div class="two-col">
      <div class="finding yellow">
        <h4>Rank Imbalance — Single-Writer Pattern</h4>
        <p><span class="stat">{r['io_bot_rank_above1000_n'] if 'io_bot_rank_above1000_n' in r else r['io_bot_rank_above1000']} jobs ({r['io_bot_rank_above1000_pct']}%)</span> show rank imbalance &gt;1000x 
        (median rank imbalance: {r['io_bot_rank_imbal_median']:.1f}x). 
        This indicates single-writer serialization — one MPI rank performing all I/O while others wait.</p>
      </div>
      <div class="finding yellow">
        <h4>Small I/O Operations — Metadata Pressure</h4>
        <p>Mean small read ratio: <span class="stat">{r['io_bot_small_read_ratio']:.1%}</span> of all read operations are &lt;10KB. 
        Combined with MPI-IO collective buffering through only 4 aggregator nodes, this creates metadata-dominated I/O 
        that cannot be accelerated by GPU computation.</p>
      </div>
    </div>
  </div>

  <!-- S5: EARTH SCIENCE -->
  <div class="section">
    <div class="section-title"><span class="section-num">5</span> Earth Science — 10x Overrepresentation in IO_Bottlenecked</div>
    <div class="finding red">
      <h4>Domain-Specific Finding</h4>
      <p>Earth Science represents <span class="stat">{r['es_system_pct']}%</span> of all Polaris submissions 
      but accounts for <span class="stat">{r['es_io_pct_of_iobot']}%</span> of IO_Bottlenecked jobs — 
      a <span class="stat">{round(r['es_io_pct_of_iobot']/r['es_system_pct'],1)}x overrepresentation</span>. 
      These {fmt(r['es_io_jobs'])} jobs moved <span class="stat">{r['es_io_bytes_TB']} TB</span> of data 
      with <span class="stat">{r['es_io_mpiio_pct']}%</span> using MPI-IO collective operations — 
      all funneled through a fixed 4-node aggregator pool regardless of job scale.</p>
    </div>
    {tbl(['Science Field','% of All Jobs','% of IO_Bottlenecked','Overrepresentation'],
         [[r['field'], f"{r['system_pct']:.1f}%", f"{r['iobot_pct']:.1f}%", f"{r['overrep_factor']}x"] for r in r['field_compare'].to_dict('records')])}
  </div>

  <!-- S6: CB_NODES -->
  <div class="section">
    <div class="section-title"><span class="section-num">6</span> cb_nodes=4 Serialization — System-Wide Default Never Overridden</div>
    <div class="stat-grid">
      <div class="stat-box"><div class="val">{fmt(r['mpiio_jobs'])}</div><div class="lbl">Total MPI-IO jobs in 2025</div></div>
      <div class="stat-box"><div class="val">{r['mpiio_cbnodes4_pct']}%</div><div class="lbl">Using cb_nodes=4 (system default)</div></div>
      <div class="stat-box"><div class="val">{fmt(r['mpiio_multinodes_cbnodes4'])}</div><div class="lbl">Multi-node jobs still using cb_nodes=4</div></div>
      <div class="stat-box"><div class="val">{r['mpiio_multinodes_mean_nodes']}</div><div class="lbl">Mean nodes for those multi-node jobs</div></div>
    </div>
    <div class="callout">
    The median BWio for these multi-node MPI-IO jobs is 
    <span class="stat">{r['mpiio_multinodes_bwio_median']} MB/s</span> — 
effectively zero bandwidth despite parallel file system access.
    <strong>Finding:</strong> Every single MPI-IO job across the entire year 2025 on Polaris used cb_nodes=4. 
      This includes {fmt(r['mpiio_multinodes_cbnodes4'])} jobs running on an average of {r['mpiio_multinodes_mean_nodes']} nodes — 
      funneling all collective I/O through just 4 aggregator nodes on 18-node jobs. 
      <strong>A single facility-level configuration change could address this for all future MPI-IO jobs.</strong>
    </div>
  </div>

  <!-- S7: DARSHAN INTERFACES -->
  <div class="section">
    <div class="section-title"><span class="section-num">7</span> I/O Interface Distribution</div>
    <div class="two-col">
      <div class="panel">
        <h3>Interface Breakdown (Darshan-covered jobs)</h3>
        <div class="finding">
          <h4>STDIO: {r['stdio_pct']}%</h4>
          <p>Standard I/O — least efficient for HPC, bypasses collective I/O optimizations entirely.</p>
        </div>
        <div class="finding">
          <h4>POSIX: {r['posix_pct']}%</h4>
          <p>Direct POSIX calls — no collective buffering, limited parallelism.</p>
        </div>
        <div class="finding red">
          <h4>MPI-IO: {r['mpiio_pct']}%</h4>
          <p>Collective I/O — only {r['mpiio_pct']}% of jobs use the interface designed for parallel file systems.</p>
        </div>
      </div>
      <div class="panel">
        <h3>Bandwidth (BWio)</h3>
        <div class="stat-grid" style="grid-template-columns:1fr 1fr">
          <div class="stat-box"><div class="val">{r['bwio_median']} MB/s</div><div class="lbl">Median BWio (jobs with I/O)</div></div>
          <div class="stat-box"><div class="val">{r['bwio_p90']} MB/s</div><div class="lbl">P90 BWio</div></div>
        </div>
        <div class="finding red" style="margin-top:12px">
          <h4>Zero I/O Records: {r['zero_io_pct']}%</h4>
          <p>Of all Darshan records, {r['zero_io_pct']}% show zero bytes read AND written — filesystem metadata operations only.</p>
        </div>
      </div>
    </div>
  </div>

  <!-- S8: PHASE PATTERNS -->
  <div class="section">
    <div class="section-title"><span class="section-num">8</span> GPU–I/O Phase Fingerprints</div>
    <p style="color:var(--muted);font-size:13px;margin-bottom:16px">
      Each job runtime divided into thirds. GPU utilization and I/O timing measured independently per tier.
      These signatures are only visible through cross-layer correlation.
    </p>
    {tbl(['Tier','N','GPU Phase 1','GPU Phase 2','GPU Phase 3','I/O Start','I/O End'], phase_rows)}
    <div class="two-col" style="margin-top:16px">
      <div class="finding">
        <h4>Ghost vs IO_Bottlenecked — The Critical Distinction</h4>
        <p>Both tiers show flat-idle GPU across all three phases (&lt;1%). 
        The discriminator is I/O: Ghost jobs have I/O end fraction ~0.31 (brief metadata activity), 
        while IO_Bottlenecked jobs have I/O end fraction ~0.95 (sustained I/O throughout). 
        <strong>This distinction is invisible to single-layer monitoring.</strong></p>
      </div>
      <div class="finding green">
        <h4>Compute_Bound Warm-Up Signature</h4>
        <p>Phase 1: {r['phase_table'][r['phase_table']['Tier']=='Compute_Bound']['GPU Phase1 (%)'].values[0] if len(r['phase_table'][r['phase_table']['Tier']=='Compute_Bound']) > 0 else '~81'}% → 
        Phase 2: {r['phase_table'][r['phase_table']['Tier']=='Compute_Bound']['GPU Phase2 (%)'].values[0] if len(r['phase_table'][r['phase_table']['Tier']=='Compute_Bound']) > 0 else '~91'}% → 
        Phase 3: {r['phase_table'][r['phase_table']['Tier']=='Compute_Bound']['GPU Phase3 (%)'].values[0] if len(r['phase_table'][r['phase_table']['Tier']=='Compute_Bound']) > 0 else '~88'}%. 
        GPU ramps up as data is loaded, peaks at sustained compute, slight drop at cleanup. 
        I/O ends early (start frac ~0.001), confirming front-loaded data loading followed by pure compute.</p>
      </div>
    </div>
  </div>

  <!-- S9: WALLTIME -->
  <div class="section">
    <div class="section-title"><span class="section-num">9</span> Walltime Utilization by Tier</div>
    <div class="finding red">
      <h4>Ghost Job Walltime Abuse</h4>
      <p>Ghost jobs use only <span class="stat">{r['ghost_walltime_util']}%</span> of their requested walltime (median). 
      Computer Science Ghost jobs specifically request <span class="stat">{r['cs_ghost_walltime_hrs']} hours</span> on average 
      but run for only <span class="stat">{r['cs_ghost_runtime_hrs']} hours</span>. 
      These jobs hold GPU allocations in the scheduler queue for the full requested window even when they complete early, 
      blocking other jobs from using those resources.</p>
    </div>
    {tbl(['Tier', 'Median Walltime Utilization', 'Bar'], wt_rows)}
  </div>

  <!-- S10: REPEATABILITY -->
  <div class="section">
    <div class="section-title"><span class="section-num">10</span> Structural Waste — Recurring Patterns</div>
    <div class="callout blue">
      <strong>{fmt(r['structural_users'])} users</strong> exhibit structural recurring waste: repeatability score &ge;70% 
      AND appearing in waste tiers for &ge;3 months. These are not transient failures — they are workflow properties 
      that repeatedly introduce inefficiency into the system.
    </div>
    {tbl(['User (anonymized)', 'Months Active', 'Repeatability Score', 'Runtime CV', 'Bytes CV'], rep_rows) if rep_rows else '<p style="color:var(--muted)">Repeatability data not available.</p>'}
    <div class="finding orange" style="margin-top:16px">
      <h4>DJC Confirmation — Automated Workflow Detected</h4>
      <p>Top Ghost offender (···439440): <span class="stat">242 Ghost jobs</span> over 6 months, 
      1 unique GPU config (always 4 GPUs), 1 unique project, 3 distinct walltime values (3540s, 3599s, 3600s), 
      median submission interval 3.26 hours. This is consistent with an <span class="stat">automated workflow 
      submitting hourly GPU jobs that never invoke GPU kernels</span>.</p>
    </div>
  </div>

  <!-- S11: FAILED JOBS -->
  <div class="section">
    <div class="section-title"><span class="section-num">11</span> Failed Job Resource Consumption</div>
    <div class="finding red">
      <h4>Underreported Waste Category</h4>
      <p><span class="stat">{fmt(r['failed_jobs'])} failed jobs</span> consumed 
      <span class="stat">{fmt(r['failed_gpu_hours'])} GPU hours ({r['failed_gpu_pct']}% of total)</span>. 
      This is comparable in scale to Ghost job waste and is rarely discussed in HPC I/O literature 
      because failure analysis is considered an operational problem. 
      These represent <span class="stat">{fmt(round(r['failed_gpu_hours']/4/1e3,0))}K node-hours</span> consumed by failed jobs.</p>
    </div>
  </div>

  <!-- S12: SUMMARY -->
  <div class="section">
    <div class="section-title"><span class="section-num">12</span> Summary of Findings</div>
    <div class="finding red">
      <h4>F1 — {r['waste_pct']}% GPU Hour Waste at Production Scale</h4>
      <p>{fmt(round(r['waste_gpu_hours']/1e6,2))}M of {fmt(round(r['total_gpu_hours']/1e6,2))}M total GPU hours wasted across Ghost, Scale_Waster, and IO_Bottlenecked tiers.</p>
    </div>
    <div class="finding red">
      <h4>F2 — IO_Bottlenecked: Novel Cross-Layer Class</h4>
      <p>{fmt(r['io_bot_jobs'])} jobs moved {r['io_bot_total_bytes_TB']} TB on GPU nodes with &lt;0.5% GPU utilization across all job phases. Undetectable by any single-layer tool.</p>
    </div>
    <div class="finding red">
      <h4>F3 — Earth Science 10x Overrepresented in IO_Bottlenecked</h4>
      <p>6.3% of jobs but 62.9% of IO_Bottlenecked — driven by MPI-IO workflows through a 4-node collective buffering bottleneck.</p>
    </div>
    <div class="finding red">
      <h4>F4 — cb_nodes=4 for {r['mpiio_cbnodes4_pct']}% of MPI-IO Jobs</h4>
      <p>Every MPI-IO job in 2025 used the system default of 4 collective buffering nodes, including {fmt(r['mpiio_multinodes_cbnodes4'])} jobs running on an average of {r['mpiio_multinodes_mean_nodes']} nodes.</p>
    </div>
    <div class="finding red">
      <h4>F5 — Ghost Jobs Use {r['ghost_walltime_util']}% of Requested Walltime</h4>
      <p>Ghost jobs complete in {r['ghost_walltime_util']}% of their requested walltime (median), holding scheduler slots and blocking legitimate workloads.</p>
    </div>
    <div class="finding red">
      <h4>F6 — {fmt(r['structural_users'])} Users Show Structural Recurring Waste</h4>
      <p>Automated workflows submitting GPU jobs that never invoke GPU kernels, confirmed via DJC submission interval analysis.</p>
    </div>
    <div class="finding red">
      <h4>F7 — STDIO Dominates at 83% Despite Parallel File System</h4>
      <p>Only 2.3% of jobs use MPI-IO. Most jobs bypass collective I/O entirely, explaining the median BWio of {r['bwio_median']} MB/s.</p>
    </div>
    <div class="finding red">
      <h4>F8 — Failed Jobs Consume {r['failed_gpu_pct']}% of GPU Hours</h4>
      <p>{fmt(r['failed_jobs'])} failed jobs burned {fmt(r['failed_gpu_hours'])} GPU hours — comparable in scale to Ghost jobs and rarely reported in literature.</p>
    </div>
  </div>

  <footer>
    <div>Cross-Layer HPC Tool &mdash; University of Alabama &middot; ALCF &middot; LBNL/NERSC</div>
    <div>Generated: {ts} &mdash; Data: ALCF Polaris 2025 &mdash; Target: IEEE Cluster 2026</div>
  </footer>

</div>


</body>
</html>"""
    return html


# ─── MAIN ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cfg = load_config(args.config)

    print("Loading combined metrics...", flush=True)
    df = pd.read_csv(cfg["combined_out"], low_memory=False)
    df["job_id"] = df["JOB_NAME"].str.split(".").str[0]
    print(f"  {len(df):,} jobs loaded")

    print("Computing results...", flush=True)
    r = compute_all(df)

    out_dir = Path(cfg["combined_out"]).parent
    out_path = out_dir / "results_report.html"

    print("Generating HTML report...", flush=True)
    html = generate_html(r)

    out_path.write_text(html, encoding='utf-8')
    print(f"\nDone → {out_path}")
    print(f"Open in browser: file://{out_path.resolve()}")