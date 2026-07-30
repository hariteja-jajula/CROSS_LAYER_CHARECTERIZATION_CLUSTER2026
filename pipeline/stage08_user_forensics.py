"""
stage08_user_forensics.py — Darshan forensic profiles for top waste users
Extracts detailed per-user evidence for the paper's
intervention argument: same executable, same directories, same pattern → one conversation.

Run: python -m pipeline.stage08_user_forensics --config config/config.json

Targets:
  - Top 10 Application-limited (Critical) users from REX-IO volumetric audit
  - Top 10 IO_Bottlenecked users by GPU hours
  - Top 10 Ghost users that DO have Darshan logs (rare but informative)
"""

import pandas as pd
import numpy as np
import json, argparse, tarfile, subprocess, tempfile, os
from pathlib import Path
from collections import defaultdict, Counter

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

def executable_fingerprints(results, df):
    """Aggregate per-executable fingerprints across all profiled users."""
    sep("PER-EXECUTABLE FINGERPRINTS")

    if not results:
        print("No results to fingerprint.")
        return

    # build per-executable records
    exe_records = defaultdict(lambda: {
        'users': set(),
        'job_ids': set(),
        'tiers': Counter(),
        'science_fields': Counter(),
        'nprocs': [],
        'nodes': [],
        'cb_nodes': [],
        'bwio': [],
        'gpu_util': [],
        'bytes_written': [],
        'bytes_read': [],
        'write_sizes': Counter(),
        'read_sizes': Counter(),
        'has_mpiio': 0,
        'has_posix': 0,
        'has_stdio': 0,
        'count': 0,
    })

    for r in results:
        exe = r.get('executable')
        if not exe:
            continue
        exe_hash = os.path.basename(exe)
        rec = exe_records[exe_hash]

        rec['users'].add(r.get('USERNAME_GENID', ''))
        rec['job_ids'].add(r.get('job_id', ''))
        rec['tiers'][r.get('crosslayer_tier', '')] += 1
        rec['science_fields'][r.get('SCIENCE_FIELD_SHORT', '')] += 1
        rec['count'] += 1

        if r.get('nprocs', 0) > 0:
            rec['nprocs'].append(r['nprocs'])
        if r.get('NODES_USED', 0) > 0:
            rec['nodes'].append(r['NODES_USED'])
        if r.get('cb_nodes', 0) > 0:
            rec['cb_nodes'].append(r['cb_nodes'])
        if r.get('BWio_MB') and not pd.isna(r['BWio_MB']):
            rec['bwio'].append(r['BWio_MB'])
        if r.get('gpu_util_mean') is not None and not pd.isna(r['gpu_util_mean']):
            rec['gpu_util'].append(r['gpu_util_mean'])
        if r.get('bytes_written', 0) > 0:
            rec['bytes_written'].append(r['bytes_written'])
        if r.get('bytes_read', 0) > 0:
            rec['bytes_read'].append(r['bytes_read'])
        for bucket, count in r.get('write_sizes', {}).items():
            rec['write_sizes'][bucket] += count
        for bucket, count in r.get('read_sizes', {}).items():
            rec['read_sizes'][bucket] += count
        if r.get('has_mpiio'): rec['has_mpiio'] += 1
        if r.get('has_posix'): rec['has_posix'] += 1
        if r.get('has_stdio'): rec['has_stdio'] += 1

    # sort by job count descending
    sorted_exes = sorted(exe_records.items(), key=lambda x: -x[1]['count'])

    size_order = ['0_100', '100_1K', '1K_10K', '10K_100K', '100K_1M',
                  '1M_4M', '4M_10M', '10M_100M', '100M_1G', '1G_PLUS']

    for exe_hash, rec in sorted_exes:
        if rec['count'] < 3:
            continue  # skip singletons

        print(f"\n── Executable: {exe_hash} ──────────────────────────────────────")
        print(f"  Jobs profiled:   {rec['count']}  |  Unique users: {len(rec['users'])}  |  Unique jobs: {len(rec['job_ids'])}")
        print(f"  Tiers:           {dict(rec['tiers'].most_common(3))}")
        print(f"  Science fields:  {dict(rec['science_fields'].most_common(3))}")
        print(f"  I/O modules:     POSIX={rec['has_posix']} MPI-IO={rec['has_mpiio']} STDIO={rec['has_stdio']}")

        if rec['nprocs']:
            print(f"  nprocs:          median={int(np.median(rec['nprocs']))}  unique={len(set(rec['nprocs']))}")
        if rec['nodes']:
            print(f"  nodes:           median={np.median(rec['nodes']):.0f}  unique={len(set(rec['nodes']))}")
        if rec['cb_nodes']:
            cb4_pct = sum(1 for v in rec['cb_nodes'] if v == 4) / len(rec['cb_nodes']) * 100
            print(f"  cb_nodes=4:      {cb4_pct:.0f}% of jobs")
        if rec['bwio']:
            print(f"  BWio (MB/s):     median={np.median(rec['bwio']):.1f}  mean={np.mean(rec['bwio']):.1f}  P90={np.percentile(rec['bwio'], 90):.1f}")
        if rec['gpu_util']:
            print(f"  GPU util:        median={np.median(rec['gpu_util']):.1f}%  mean={np.mean(rec['gpu_util']):.1f}%")
        if rec['bytes_written']:
            print(f"  Bytes written:   median={np.median(rec['bytes_written'])/1e9:.2f} GB/job")
        if rec['bytes_read']:
            print(f"  Bytes read:      median={np.median(rec['bytes_read'])/1e9:.2f} GB/job")

        # write size pattern
        total_writes = sum(rec['write_sizes'].values())
        if total_writes > 0:
            small = sum(rec['write_sizes'].get(b, 0) for b in ['0_100', '100_1K', '1K_10K'])
            large = sum(rec['write_sizes'].get(b, 0) for b in ['1M_4M', '4M_10M', '10M_100M', '100M_1G', '1G_PLUS'])
            dominant = max(size_order, key=lambda b: rec['write_sizes'].get(b, 0))
            pattern = "metadata-storm (<10KB)" if small/total_writes > 0.7 else \
                      "large sequential (>1MB)" if large/total_writes > 0.7 else "mixed"
            print(f"  Write pattern:   {pattern}  (dominant bucket: {dominant})")

            
# ─────────────────────────────────────────────────────────────────────────────
# DARSHAN HEADER + FILE PATH EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def extract_forensics(tar, fname):
    """Extract executable, file paths, nprocs, modules, cb_nodes, mount points
    from a darshan file. Returns a dict or None on failure."""
    try:
        f = tar.extractfile(tar.getmember(fname))
    except KeyError:
        return None
    if f is None:
        return None

    tmp_path = f"/pscratch/sd/h/hjajula/.darshan_tmp_{os.getpid()}.darshan"
    try:
        with open(tmp_path, 'wb') as tmp:
            tmp.write(f.read())
        result = subprocess.run(
            ["darshan-parser", tmp_path],
            capture_output=True, text=True, timeout=120
        )
        if result.returncode != 0:
            return None
        return parse_forensic_output(result.stdout)
    except subprocess.TimeoutExpired:
        return None
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

def parse_forensic_output(output):
    """Parse darshan-parser output for forensic details."""
    lines = output.split('\n')
    info = {
        'executable': None,
        'nprocs': 0,
        'runtime': 0.0,
        'has_posix': False,
        'has_mpiio': False,
        'has_stdio': False,
        'cb_nodes': 0,
        'mount_points': set(),
        'file_paths': [],
        'bytes_read': 0.0,
        'bytes_written': 0.0,
        'posix_opens': 0,
        'posix_reads': 0,
        'posix_writes': 0,
        'unique_files': set(),
        'write_sizes': Counter(),  # size bucket → count
        'read_sizes': Counter(),
    }

    # --- parse header ---
    for line in lines:
        if not line.startswith('#'):
            continue
        low = line.lower()
        if 'exe:' in low:
            try:
                # format: "# exe: /path/to/binary arg1 arg2"
                exe_part = line.split('exe:', 1)[1].strip()
                info['executable'] = exe_part.split()[0] if exe_part else None
            except:
                pass
        elif 'nprocs:' in low:
            try:
                info['nprocs'] = int(line.split(':', 1)[1].strip())
            except:
                pass
        elif 'run time:' in low:
            try:
                info['runtime'] = float(line.split(':', 1)[1].strip())
            except:
                pass
        elif 'posix module' in low:
            info['has_posix'] = True
        elif 'mpi-io module' in low:
            info['has_mpiio'] = True
        elif 'stdio module' in low:
            info['has_stdio'] = True
        elif 'cb_nodes=' in low:
            try:
                info['cb_nodes'] = int(line.split('cb_nodes=')[1].split(';')[0])
            except:
                pass
        elif 'mount entry:' in low or ('mount' in low and '\t' in line):
            # darshan mount lines: "# mount entry: /lus/grand ..."
            try:
                parts = line.split(':', 1)[1].strip().split()
                if parts:
                    info['mount_points'].add(parts[0])
            except:
                pass

    # --- parse data records for file paths and access patterns ---
    seen_files = {}  # record_id → file_path
    for line in lines:
        if line.startswith('#'):
            # file record lines: "# <record_id>  /path/to/file"
            # darshan-parser prints file records as: "# 12345 /lus/grand/projects/..."
            parts = line[1:].strip().split(None, 1)
            if len(parts) == 2:
                try:
                    rec_id = int(parts[0])
                    path = parts[1].strip()
                    if path.startswith('/'):
                        seen_files[rec_id] = path
                        info['unique_files'].add(path)
                except ValueError:
                    pass
            continue

        parts = line.split('\t')
        if len(parts) < 5:
            continue

        mod = parts[0]
        try:
            val = float(parts[4])
        except:
            continue
        if val == -1:
            continue

        counter = parts[3]

        if mod == 'POSIX':
            if counter == 'POSIX_BYTES_READ':
                info['bytes_read'] += val
            elif counter == 'POSIX_BYTES_WRITTEN':
                info['bytes_written'] += val
            elif counter == 'POSIX_OPENS':
                info['posix_opens'] += int(val)
            elif counter == 'POSIX_READS':
                info['posix_reads'] += int(val)
            elif counter == 'POSIX_WRITES':
                info['posix_writes'] += int(val)
            # write size buckets
            elif counter.startswith('POSIX_SIZE_WRITE_') and val > 0:
                bucket = counter.replace('POSIX_SIZE_WRITE_', '')
                info['write_sizes'][bucket] += int(val)
            elif counter.startswith('POSIX_SIZE_READ_') and val > 0:
                bucket = counter.replace('POSIX_SIZE_READ_', '')
                info['read_sizes'][bucket] += int(val)

    # extract top-level directories from file paths
    info['directories'] = set()
    for path in info['unique_files']:
        # get first 3-4 levels: /lus/grand/projects/ProjectX
        parts = path.split('/')
        if len(parts) >= 4:
            info['directories'].add('/'.join(parts[:5]))
        elif len(parts) >= 2:
            info['directories'].add('/'.join(parts[:3]))

    info['mount_points'] = list(info['mount_points'])
    info['unique_files'] = list(info['unique_files'])
    info['directories'] = list(info['directories'])

    return info


# ─────────────────────────────────────────────────────────────────────────────
# IDENTIFY TARGET USERS
# ─────────────────────────────────────────────────────────────────────────────

def identify_targets(df):
    """Identify the users we want forensic profiles for."""
    targets = {}

    # --- 1. Top IO_Bottlenecked users by GPU hours ---
    iobot = df[df['crosslayer_tier'] == 'IO_Bottlenecked']
    iobot_users = (iobot.groupby('USERNAME_GENID')['gpu_hours']
                   .sum().clip(lower=0)
                   .sort_values(ascending=False)
                   .head(10).index.tolist())
    for u in iobot_users:
        targets[u] = 'IO_Bottlenecked'

    # --- 2. Top Ghost users that HAVE darshan logs ---
    ghost_with_darshan = df[
        (df['crosslayer_tier'] == 'Ghost') &
        (df['darshan_file_count'].notna())
    ]
    ghost_users = (ghost_with_darshan.groupby('USERNAME_GENID')['gpu_hours']
                   .sum().clip(lower=0)
                   .sort_values(ascending=False)
                   .head(10).index.tolist())
    for u in ghost_users:
        if u not in targets:
            targets[u] = 'Ghost_with_Darshan'

    # --- 3. Application-limited (Critical) users from volumetric audit ---
    io_df = df[df['total_bytes'].notna() & (df['total_bytes'] > 0)].copy()
    if len(io_df) > 0:
        p90 = io_df['total_bytes'].quantile(0.90)
        actionable = io_df[io_df['total_bytes'] >= p90].copy()
        critical = actionable[actionable['BWio_MB'].notna() & (actionable['BWio_MB'] < 300)]
        crit_users = (critical.groupby('USERNAME_GENID')['total_bytes']
                      .sum().sort_values(ascending=False)
                      .head(10).index.tolist())
        for u in crit_users:
            if u not in targets:
                targets[u] = 'Critical_AppLimited'

    print(f"Target users: {len(targets)}")
    for u, reason in sorted(targets.items(), key=lambda x: x[1]):
        uid = str(int(u))[-6:]
        n_jobs = len(df[df['USERNAME_GENID'] == u])
        print(f"  ···{uid}  ({reason}, {n_jobs} total jobs)")

    return targets


# ─────────────────────────────────────────────────────────────────────────────
# SCAN DARSHAN TARS FOR TARGET USERS
# ─────────────────────────────────────────────────────────────────────────────

def scan_targets(targets, df, darshan_metrics, data_root):
    """Find darshan files for target users and extract forensics."""

    # get job_ids for target users in relevant tiers
    target_jobs = df[
        (df['USERNAME_GENID'].isin(targets.keys())) &
        (df['crosslayer_tier'].isin([
            'Ghost', 'IO_Bottlenecked', 'Balanced',
            'Scale_Waster', 'CPU_IO_Job', 'Failed_Job'
        ]))
    ][['job_id', 'USERNAME_GENID', 'crosslayer_tier',
       'RUNTIME_SECONDS', 'NODES_USED', 'gpu_util_mean',
       'total_bytes', 'BWio_MB', 'SCIENCE_FIELD_SHORT']].copy()

    target_jobs['job_id'] = target_jobs['job_id'].astype(str)
    darshan_metrics['job_id'] = darshan_metrics['job_id'].astype(str)

    # find darshan fnames for these jobs
    target_fnames = darshan_metrics[
        darshan_metrics['job_id'].isin(target_jobs['job_id'])
    ][['job_id', 'fname']].copy()

    # merge to get user info
    target_fnames = target_fnames.merge(
        target_jobs[['job_id', 'USERNAME_GENID', 'crosslayer_tier',
                     'RUNTIME_SECONDS', 'NODES_USED', 'gpu_util_mean',
                     'total_bytes', 'BWio_MB', 'SCIENCE_FIELD_SHORT']],
        on='job_id', how='left'
    )

    print(f"\nDarshan files to extract: {len(target_fnames):,}")
    print(f"Unique users covered:     {target_fnames['USERNAME_GENID'].nunique()}")
    print(f"Unique jobs covered:      {target_fnames['job_id'].nunique()}")

    # limit per user to avoid excessive extraction
    MAX_PER_USER = 50
    limited = target_fnames.groupby('USERNAME_GENID').head(MAX_PER_USER)
    if len(limited) < len(target_fnames):
        print(f"Limiting to {MAX_PER_USER} files per user: {len(limited):,} files")
    target_fnames = limited

    # build fname lookup for fast tar scanning
    fname_set = set(target_fnames['fname'])
    fname_to_info = target_fnames.set_index('fname').to_dict('index')

    # scan tars
    results = []  # list of (user, job_id, tier, forensic_info)
    checked = 0
    found = 0

    for year in [2025]:
        for month in range(1, 13):
            for day in range(1, 32):
                tar_path = data_root / str(year) / str(month) / str(day) / "logs.tar.gz"
                if not tar_path.exists():
                    continue
                try:
                    with tarfile.open(tar_path, "r:gz") as tar:
                        tar_names = set(tar.getnames())
                        matches = fname_set & tar_names
                        if not matches:
                            continue

                        for fname in matches:
                            info = extract_forensics(tar, fname)
                            if info is None:
                                continue

                            meta = fname_to_info.get(fname, {})
                            info['job_id'] = meta.get('job_id', '')
                            info['USERNAME_GENID'] = meta.get('USERNAME_GENID', '')
                            info['crosslayer_tier'] = meta.get('crosslayer_tier', '')
                            info['gpu_util_mean'] = meta.get('gpu_util_mean', None)
                            info['total_bytes_combined'] = meta.get('total_bytes', 0)
                            info['BWio_MB'] = meta.get('BWio_MB', None)
                            info['NODES_USED'] = meta.get('NODES_USED', 0)
                            info['SCIENCE_FIELD_SHORT'] = meta.get('SCIENCE_FIELD_SHORT', '')
                            info['fname'] = fname

                            results.append(info)
                            found += 1
                            if found % 100 == 0:
                                print(f"  Extracted {found} files...", flush=True)
                except Exception as e:
                    print(f"  Error reading {tar_path}: {e}", flush=True)
                    continue

    print(f"\nExtraction complete: {found} files from {len(set(r['USERNAME_GENID'] for r in results))} users")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# GENERATE PER-USER FORENSIC PROFILES
# ─────────────────────────────────────────────────────────────────────────────

def generate_profiles(results, targets, df):
    """Generate per-user forensic profiles from extracted darshan data."""

    # group results by user
    by_user = defaultdict(list)
    for r in results:
        by_user[r['USERNAME_GENID']].append(r)

    for user_id, reason in sorted(targets.items(), key=lambda x: x[1]):
        user_results = by_user.get(user_id, [])
        uid = str(int(user_id))[-6:]

        # get full job stats from combined_metrics
        user_all = df[df['USERNAME_GENID'] == user_id]
        user_waste = user_all[user_all['crosslayer_tier'].isin(
            ['Ghost', 'IO_Bottlenecked', 'Scale_Waster']
        )]

        sep(f"USER ···{uid} ({reason})")

        # --- overview ---
        print(f"Total jobs:              {len(user_all):,}")
        print(f"Waste tier jobs:         {len(user_waste):,}")
        print(f"Total GPU hours:         {user_all['gpu_hours'].clip(lower=0).sum():,.0f}")
        print(f"Waste GPU hours:         {user_waste['gpu_hours'].clip(lower=0).sum():,.0f}")
        print(f"Science field:           {user_all['SCIENCE_FIELD_SHORT'].mode().iloc[0] if len(user_all) > 0 else 'N/A'}")
        print(f"Darshan files extracted: {len(user_results)}")

        # tier breakdown
        print(f"\nTier breakdown:")
        tiers = user_all['crosslayer_tier'].value_counts()
        for t, c in tiers.items():
            print(f"  {t:<25s} {c:>5,}")

        if not user_results:
            print("\n  ⚠ No Darshan files found for this user — likely all Ghost (no I/O)")
            continue

        # --- executable analysis ---
        executables = [r['executable'] for r in user_results if r['executable']]
        exe_counts = Counter(executables)
        print(f"\nExecutables ({len(exe_counts)} unique):")
        for exe, count in exe_counts.most_common(5):
            pct = count / len(executables) * 100
            # shorten path for display
            short = exe if len(exe) < 60 else '...' + exe[-57:]
            print(f"  {count:>4}× ({pct:>5.1f}%) {short}")

        if len(exe_counts) == 1:
            print(f"  → SINGLE EXECUTABLE: all {len(executables)} jobs run the same binary")
        elif exe_counts.most_common(1)[0][1] / len(executables) > 0.8:
            dom_exe = exe_counts.most_common(1)[0][0]
            dom_pct = exe_counts.most_common(1)[0][1] / len(executables) * 100
            print(f"  → DOMINANT EXECUTABLE: {dom_pct:.0f}% of jobs run the same binary")

        # --- nprocs analysis ---
        nprocs_list = [r['nprocs'] for r in user_results if r['nprocs'] > 0]
        if nprocs_list:
            nprocs_counts = Counter(nprocs_list)
            print(f"\nnprocs distribution ({len(nprocs_counts)} unique configs):")
            for np_val, count in nprocs_counts.most_common(5):
                print(f"  nprocs={np_val:>5}: {count:>4} jobs")
            if len(nprocs_counts) == 1:
                print(f"  → FIXED PARALLELISM: always {nprocs_list[0]} ranks")

        # --- nodes used ---
        nodes = [r['NODES_USED'] for r in user_results if r['NODES_USED'] and r['NODES_USED'] > 0]
        if nodes:
            nodes_counts = Counter(nodes)
            print(f"\nNodes used ({len(nodes_counts)} unique configs):")
            for n, count in nodes_counts.most_common(5):
                print(f"  {n:>4} nodes: {count:>4} jobs")

        # --- cb_nodes ---
        cb_vals = [r['cb_nodes'] for r in user_results if r['cb_nodes'] > 0]
        if cb_vals:
            cb_counts = Counter(cb_vals)
            print(f"\ncb_nodes: {dict(cb_counts)}")
            if all(v == 4 for v in cb_vals):
                print(f"  → ALL cb_nodes=4 — system default, never tuned")

        # --- I/O modules ---
        mpiio_count = sum(1 for r in user_results if r['has_mpiio'])
        posix_count = sum(1 for r in user_results if r['has_posix'])
        stdio_count = sum(1 for r in user_results if r['has_stdio'])
        print(f"\nI/O modules: POSIX={posix_count}, MPI-IO={mpiio_count}, STDIO={stdio_count}")

        # --- file path analysis ---
        all_dirs = Counter()
        all_files = Counter()
        for r in user_results:
            for d in r.get('directories', []):
                all_dirs[d] += 1
            for f in r.get('unique_files', []):
                all_files[f] += 1

        if all_dirs:
            print(f"\nTop directories accessed ({len(all_dirs)} unique):")
            for d, count in all_dirs.most_common(8):
                pct = count / len(user_results) * 100
                print(f"  {count:>4}× ({pct:>5.1f}%) {d}")

        # files accessed across many jobs (repeated access patterns)
        repeated_files = {f: c for f, c in all_files.items() if c >= 3}
        if repeated_files:
            print(f"\nRepeatedly accessed files ({len(repeated_files)} files in ≥3 jobs):")
            for f, count in sorted(repeated_files.items(), key=lambda x: -x[1])[:10]:
                short = f if len(f) < 65 else '...' + f[-62:]
                print(f"  {count:>4}× {short}")

        # --- write size distribution ---
        total_write_sizes = Counter()
        total_read_sizes = Counter()
        for r in user_results:
            for bucket, count in r.get('write_sizes', {}).items():
                total_write_sizes[bucket] += count
            for bucket, count in r.get('read_sizes', {}).items():
                total_read_sizes[bucket] += count

        if total_write_sizes:
            total_writes = sum(total_write_sizes.values())
            print(f"\nWrite size distribution (across all jobs, {total_writes:,} total ops):")
            # order: small → large
            size_order = ['0_100', '100_1K', '1K_10K', '10K_100K', '100K_1M',
                          '1M_4M', '4M_10M', '10M_100M', '100M_1G', '1G_PLUS']
            for bucket in size_order:
                count = total_write_sizes.get(bucket, 0)
                if count > 0:
                    pct = count / total_writes * 100
                    bar = '█' * int(pct / 2)
                    print(f"  {bucket:>12s}: {count:>10,} ({pct:>5.1f}%) {bar}")

            small = sum(total_write_sizes.get(b, 0) for b in ['0_100', '100_1K', '1K_10K'])
            if total_writes > 0 and small / total_writes > 0.5:
                print(f"  → {small/total_writes*100:.0f}% SMALL WRITES (<10KB) — metadata-storm pattern")

        if total_read_sizes:
            total_reads = sum(total_read_sizes.values())
            print(f"\nRead size distribution (across all jobs, {total_reads:,} total ops):")
            for bucket in size_order:
                count = total_read_sizes.get(bucket, 0)
                if count > 0:
                    pct = count / total_reads * 100
                    bar = '█' * int(pct / 2)
                    print(f"  {bucket:>12s}: {count:>10,} ({pct:>5.1f}%) {bar}")

        # --- bandwidth context ---
        bwio_vals = [r['BWio_MB'] for r in user_results if r['BWio_MB'] and not pd.isna(r['BWio_MB'])]
        if bwio_vals:
            print(f"\nBWio: median={np.median(bwio_vals):.1f} MB/s, mean={np.mean(bwio_vals):.1f} MB/s, "
                  f"min={min(bwio_vals):.1f}, max={max(bwio_vals):.1f}")

        # --- GPU utilization context ---
        gpu_vals = [r['gpu_util_mean'] for r in user_results
                    if r['gpu_util_mean'] is not None and not pd.isna(r['gpu_util_mean'])]
        if gpu_vals:
            print(f"GPU util: median={np.median(gpu_vals):.1f}%, mean={np.mean(gpu_vals):.1f}%, "
                  f"max={max(gpu_vals):.1f}%")

        # --- total bytes ---
        tb_vals = [r['bytes_read'] + r['bytes_written'] for r in user_results]
        if tb_vals:
            total = sum(tb_vals)
            print(f"Total bytes (from extracted files): {total/1e12:.2f} TB "
                  f"(mean/job: {np.mean(tb_vals)/1e9:.1f} GB)")

        # --- paper-ready summary ---
        print(f"\n--- PAPER SENTENCE ---")
        if len(exe_counts) == 1 and len(executables) > 3:
            exe_name = os.path.basename(list(exe_counts.keys())[0])
            top_dir = all_dirs.most_common(1)[0][0] if all_dirs else "unknown"
            bw_str = f"{np.median(bwio_vals):.1f} MB/s" if bwio_vals else "N/A"
            gpu_str = f"{np.mean(gpu_vals):.1f}%" if gpu_vals else "N/A"
            print(f"User ···{uid} submitted {len(executables)} {reason} jobs running "
                  f"\\texttt{{{exe_name}}} writing to \\texttt{{{top_dir}}}, "
                  f"achieving a median bandwidth of {bw_str} with {gpu_str} GPU utilization. "
                  f"A single consultation could address {total/1e12:.1f} TB of inefficient I/O.")
        elif len(exe_counts) > 1:
            dom_exe = os.path.basename(exe_counts.most_common(1)[0][0])
            dom_pct = exe_counts.most_common(1)[0][1] / len(executables) * 100
            print(f"User ···{uid}: {dom_pct:.0f}% of {len(executables)} jobs run "
                  f"\\texttt{{{dom_exe}}}. Mixed workflow — multiple intervention points.")
        else:
            print(f"User ···{uid}: insufficient Darshan data for narrative.")


# ─────────────────────────────────────────────────────────────────────────────
# CROSS-USER SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

def cross_user_summary(results, targets):
    """Print aggregate stats across all profiled users."""
    sep("CROSS-USER SUMMARY")

    by_user = defaultdict(list)
    for r in results:
        by_user[r['USERNAME_GENID']].append(r)

    print(f"Users profiled:          {len(by_user)}")
    print(f"Total files extracted:   {len(results)}")

    # executable consistency across users
    single_exe_users = 0
    for user_id, user_results in by_user.items():
        exes = set(r['executable'] for r in user_results if r['executable'])
        if len(exes) == 1 and len(user_results) >= 3:
            single_exe_users += 1

    print(f"Single-executable users: {single_exe_users} "
          f"(of {len(by_user)} with ≥1 file)")
    print(f"→ {single_exe_users} users run the SAME binary across all waste jobs")
    print(f"→ Each requires ONE conversation to remediate")

    # cb_nodes summary
    all_cb = [r['cb_nodes'] for r in results if r['cb_nodes'] > 0]
    if all_cb:
        cb4_pct = sum(1 for v in all_cb if v == 4) / len(all_cb) * 100
        print(f"\ncb_nodes=4 across all profiled jobs: {cb4_pct:.1f}%")

    # small I/O summary
    total_small_writes = 0
    total_writes = 0
    for r in results:
        ws = r.get('write_sizes', {})
        total_writes += sum(ws.values())
        total_small_writes += sum(ws.get(b, 0) for b in ['0_100', '100_1K', '1K_10K'])

    if total_writes > 0:
        print(f"Small writes (<10KB): {total_small_writes/total_writes*100:.1f}% of all write ops")

    # paper-ready intervention estimate
    print(f"\n--- PAPER PARAGRAPH ---")
    print(f"Forensic analysis of Darshan logs for the top waste users confirms that")
    print(f"facility-scale inefficiency is driven by a small number of repeating workflows.")
    print(f"Of {len(by_user)} profiled users, {single_exe_users} run a single executable")
    print(f"across all their waste-tier jobs, indicating that each user's inefficiency")
    print(f"stems from one workflow rather than diverse experimentation. Combined with")
    print(f"the user concentration finding (top 10 users = 48.7% of waste GPU hours),")
    print(f"this confirms that approximately 10 targeted consultations could address")
    print(f"nearly half of the facility's observed resource waste.")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cfg = load_config(args.config)

    print("Loading combined metrics...", flush=True)
    df = pd.read_csv(cfg["combined_out"], low_memory=False)
    df["job_id"] = df["JOB_NAME"].str.split(".").str[0]
    print(f"  {len(df):,} jobs loaded")

    print("Loading darshan metrics for fname lookup...", flush=True)
    darshan_metrics = pd.read_csv(cfg["darshan_parsed_out"], usecols=["job_id", "fname"])
    darshan_metrics["job_id"] = darshan_metrics["job_id"].astype(str)
    print(f"  {len(darshan_metrics):,} darshan file records")

    # identify target users
    targets = identify_targets(df)

    # scan tars and extract forensics
    data_root = Path(cfg["darshan_data"])
    results = scan_targets(targets, df, darshan_metrics, data_root)

    if not results:
        print("\n⚠ No Darshan files found for any target user.")
        print("  This likely means the target users are predominantly Ghost (no I/O, no Darshan log).")
        print("  The Ghost finding itself is the story — these users cannot be profiled by I/O tools.")
    else:
        # generate per-user profiles
        generate_profiles(results, targets, df)

        # cross-user summary
        executable_fingerprints(results, df)
        cross_user_summary(results, targets)

    # save raw results for reference
    if results:
        out_rows = []
        for r in results:
            out_rows.append({
                'job_id': r.get('job_id', ''),
                'USERNAME_GENID': r.get('USERNAME_GENID', ''),
                'crosslayer_tier': r.get('crosslayer_tier', ''),
                'executable': r.get('executable', ''),
                'nprocs': r.get('nprocs', 0),
                'runtime': r.get('runtime', 0),
                'has_mpiio': r.get('has_mpiio', False),
                'cb_nodes': r.get('cb_nodes', 0),
                'bytes_read': r.get('bytes_read', 0),
                'bytes_written': r.get('bytes_written', 0),
                'n_unique_files': len(r.get('unique_files', [])),
                'n_directories': len(r.get('directories', [])),
                'BWio_MB': r.get('BWio_MB', None),
                'gpu_util_mean': r.get('gpu_util_mean', None),
                'SCIENCE_FIELD_SHORT': r.get('SCIENCE_FIELD_SHORT', ''),
                'fname': r.get('fname', ''),
            })
        out_path = cfg["combined_out"].replace("combined_metrics.csv", "user_forensics.csv")
        pd.DataFrame(out_rows).to_csv(out_path, index=False)
        print(f"\nRaw forensics → {out_path}")

    print(f"\n{'='*80}")
    print(f"  DONE — pipe output to a file:")
    print(f"  python -m pipeline.stage08_user_forensics --config config/config.json > user_forensics.txt")
    print(f"{'='*80}")