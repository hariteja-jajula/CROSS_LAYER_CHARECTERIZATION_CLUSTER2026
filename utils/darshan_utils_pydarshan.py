"""
darshan_utils_pydarshan.py
PyDarshan-based replacement for darshan_utils.py.
"""

import os
import subprocess
import tempfile
import darshan
import numpy as np


def parse_from_raw(raw, fname):
    """Parse from already-read raw bytes — avoids parallel tar contention."""
    import tempfile, subprocess, os, darshan
    t1 = t2 = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".darshan", delete=False) as tmp1:
            tmp1.write(raw)
            t1 = tmp1.name
        with tempfile.NamedTemporaryFile(suffix=".darshan", delete=False) as tmp2:
            t2 = tmp2.name
        result = subprocess.run(
            ["darshan-convert", t1, t2],
            capture_output=True, timeout=120
        )
        if result.returncode != 0:
            return None
        report = darshan.DarshanReport(t2, read_all=True)
        return parse_darshan_report(report)
    except subprocess.TimeoutExpired:
        print(f"Timeout: {fname}", flush=True)
        return None
    except Exception as e:
        print(f"Error parsing {fname}: {e}", flush=True)
        return None
    finally:
        for p in [t1, t2]:
            if p and os.path.exists(p):
                os.unlink(p)


def parse_darshan_report(report):
    row = _empty_row()
    row = _parse_metadata(report, row)
    if 'POSIX'   in report.modules: row = _parse_posix(report, row)
    if 'MPI-IO'  in report.modules: row = _parse_mpiio(report, row)
    if 'STDIO'   in report.modules: row = _parse_stdio(report, row)
    if 'HEATMAP' in report.modules: row = _parse_heatmap(report, row)
    row['unique_files'] = len([
        k for k, v in report.name_records.items()
        if not str(v).startswith('heatmap:')
        and str(v) not in ('<STDIN>', '<STDOUT>', '<STDERR>')
    ])
    row = _compute_derived(row)
    return row


def _parse_metadata(report, row):
    meta = report.metadata.get('job', {})
    row['runtime'] = meta.get('run_time', 0.0)
    row['nprocs']  = meta.get('nprocs', 0)
    hint_str = meta.get('metadata', {}).get('h', '')
    exe_raw  = meta.get('metadata', {}).get('e', '')
    row['executable'] = str(exe_raw) if exe_raw else ''
    if 'cb_nodes=' in hint_str:
        try: row['cb_nodes'] = int(hint_str.split('cb_nodes=')[1].split(';')[0])
        except: pass
    if 'cb_config_list=' in hint_str:
        try: row['cb_config_list'] = hint_str.split('cb_config_list=')[1].split(';')[0]
        except: pass
    mods = set(report.modules.keys())
    row['has_posix']   = 'POSIX'   in mods
    row['has_mpiio']   = 'MPI-IO'  in mods
    row['has_stdio']   = 'STDIO'   in mods
    row['has_lustre']  = 'LUSTRE'  in mods
    row['has_heatmap'] = 'HEATMAP' in mods
    mounts = []
    for _, path in report.name_records.items():
        p = str(path)
        if p.startswith('/'): mounts.append(p.split()[0])
        if '/lus/grand' in p: row['fs_lustre_grand'] = True
        if '/lus/eagle' in p: row['fs_lustre_eagle'] = True
    row['mount_points'] = ','.join(sorted(set(mounts))) if mounts else ''
    return row


def _parse_posix(report, row):
    dfs = report.records['POSIX'].to_df()
    c   = dfs['counters']
    fc  = dfs['fcounters']
    row['bytes_read']    = c['POSIX_BYTES_READ'].sum()
    row['bytes_written'] = c['POSIX_BYTES_WRITTEN'].sum()
    row['posix_reads']   = c['POSIX_READS'].sum()
    row['posix_writes']  = c['POSIX_WRITES'].sum()
    row['posix_opens']   = c['POSIX_OPENS'].sum()
    row['posix_stats']   = c['POSIX_STATS'].sum()
    row['posix_seeks']   = c['POSIX_SEEKS'].sum()
    row['posix_seq_reads']        = c['POSIX_SEQ_READS'].sum()
    row['posix_seq_writes']       = c['POSIX_SEQ_WRITES'].sum()
    row['posix_cons_reads']       = c['POSIX_CONSEC_READS'].sum()
    row['posix_cons_writes']      = c['POSIX_CONSEC_WRITES'].sum()
    row['posix_mem_not_aligned']  = c['POSIX_MEM_NOT_ALIGNED'].sum()
    row['posix_file_not_aligned'] = c['POSIX_FILE_NOT_ALIGNED'].sum()
    row['small_reads']  = c[['POSIX_SIZE_READ_0_100','POSIX_SIZE_READ_100_1K','POSIX_SIZE_READ_1K_10K']].sum().sum()
    row['medium_reads'] = c[['POSIX_SIZE_READ_10K_100K','POSIX_SIZE_READ_100K_1M']].sum().sum()
    row['large_reads']  = c[['POSIX_SIZE_READ_1M_4M','POSIX_SIZE_READ_4M_10M','POSIX_SIZE_READ_10M_100M','POSIX_SIZE_READ_100M_1G','POSIX_SIZE_READ_1G_PLUS']].sum().sum()
    row['small_writes'] = c[['POSIX_SIZE_WRITE_0_100','POSIX_SIZE_WRITE_100_1K','POSIX_SIZE_WRITE_1K_10K']].sum().sum()
    row['medium_writes']= c[['POSIX_SIZE_WRITE_10K_100K','POSIX_SIZE_WRITE_100K_1M']].sum().sum()
    row['large_writes'] = c[['POSIX_SIZE_WRITE_1M_4M','POSIX_SIZE_WRITE_4M_10M','POSIX_SIZE_WRITE_10M_100M','POSIX_SIZE_WRITE_100M_1G','POSIX_SIZE_WRITE_1G_PLUS']].sum().sum()
    row['stride_1'] = c['POSIX_STRIDE1_COUNT'].sum()
    row['stride_2'] = c['POSIX_STRIDE2_COUNT'].sum()
    row['stride_3'] = c['POSIX_STRIDE3_COUNT'].sum()
    row['stride_4'] = c['POSIX_STRIDE4_COUNT'].sum()
    row['fastest_rank_bytes'] = c['POSIX_FASTEST_RANK_BYTES'].sum()
    row['slowest_rank_bytes'] = c['POSIX_SLOWEST_RANK_BYTES'].sum()
    def _min_ts(col):
        v = fc[col]; v = v[v > 0]
        return float(v.min()) if len(v) > 0 else 0.0
    def _max_ts(col):
        v = fc[col]; return float(v.max()) if len(v) > 0 else 0.0
    row['io_open_start']  = _min_ts('POSIX_F_OPEN_START_TIMESTAMP')
    row['io_read_start']  = _min_ts('POSIX_F_READ_START_TIMESTAMP')
    row['io_write_start'] = _min_ts('POSIX_F_WRITE_START_TIMESTAMP')
    row['io_read_end']    = _max_ts('POSIX_F_READ_END_TIMESTAMP')
    row['io_write_end']   = _max_ts('POSIX_F_WRITE_END_TIMESTAMP')
    row['slowest_rank_time']   = float(fc['POSIX_F_SLOWEST_RANK_TIME'].max())
    row['fastest_rank_time']   = float(fc['POSIX_F_FASTEST_RANK_TIME'].max())
    row['variance_rank_time']  = float(fc['POSIX_F_VARIANCE_RANK_TIME'].max())
    row['variance_rank_bytes'] = float(fc['POSIX_F_VARIANCE_RANK_BYTES'].max())
    row['posix_meta_time']  = float(fc['POSIX_F_META_TIME'].max())
    row['posix_read_time']  = float(fc['POSIX_F_READ_TIME'].max())
    row['posix_write_time'] = float(fc['POSIX_F_WRITE_TIME'].max())
    return row


def _parse_mpiio(report, row):
    dfs = report.records['MPI-IO'].to_df()
    c   = dfs['counters']
    fc  = dfs['fcounters']
    row['mpiio_bytes_read']    = c['MPIIO_BYTES_READ'].sum()
    row['mpiio_bytes_written'] = c['MPIIO_BYTES_WRITTEN'].sum()
    row['mpiio_indep_reads']   = c['MPIIO_INDEP_READS'].sum()
    row['mpiio_indep_writes']  = c['MPIIO_INDEP_WRITES'].sum()
    row['mpiio_coll_reads']    = c['MPIIO_COLL_READS'].sum()
    row['mpiio_coll_writes']   = c['MPIIO_COLL_WRITES'].sum()
    def _min_ts(col):
        v = fc[col]; v = v[v > 0]
        return float(v.min()) if len(v) > 0 else 0.0
    def _max_ts(col):
        v = fc[col]; return float(v.max()) if len(v) > 0 else 0.0
    row['mpiio_read_start']  = _min_ts('MPIIO_F_READ_START_TIMESTAMP')
    row['mpiio_write_start'] = _min_ts('MPIIO_F_WRITE_START_TIMESTAMP')
    row['mpiio_read_end']    = _max_ts('MPIIO_F_READ_END_TIMESTAMP')
    row['mpiio_write_end']   = _max_ts('MPIIO_F_WRITE_END_TIMESTAMP')
    row['mpiio_slowest_rank_time']   = float(fc['MPIIO_F_SLOWEST_RANK_TIME'].max())
    row['mpiio_fastest_rank_time']   = float(fc['MPIIO_F_FASTEST_RANK_TIME'].max())
    row['mpiio_variance_rank_time']  = float(fc['MPIIO_F_VARIANCE_RANK_TIME'].max())
    row['mpiio_variance_rank_bytes'] = float(fc['MPIIO_F_VARIANCE_RANK_BYTES'].max())
    return row


def _parse_stdio(report, row):
    c = report.records['STDIO'].to_df()['counters']
    row['stdio_bytes_read']    = c['STDIO_BYTES_READ'].sum()
    row['stdio_bytes_written'] = c['STDIO_BYTES_WRITTEN'].sum()
    row['stdio_reads']         = c['STDIO_READS'].sum()
    row['stdio_writes']        = c['STDIO_WRITES'].sum()
    row['stdio_opens']         = c['STDIO_OPENS'].sum()
    row['stdio_seeks']         = c['STDIO_SEEKS'].sum()
    return row


def _parse_heatmap(report, row):
    hm = report.heatmaps.get('POSIX')
    if hm is None:
        return row

    try:
        hm_read  = hm.to_df(ops=['read'])
        hm_write = hm.to_df(ops=['write'])
    except Exception:
        return row

    bin_width = hm._bin_width_seconds
    row['heatmap_bin_width'] = bin_width

    read_per_bin  = hm_read.sum(axis=0).values.astype(float)
    write_per_bin = hm_write.sum(axis=0).values.astype(float)

    n = len(read_per_bin)
    row['heatmap_total_bins'] = n

    if n == 0:
        return row

    active_mask       = (read_per_bin + write_per_bin) > 0
    read_active_mask  = read_per_bin > 0
    write_active_mask = write_per_bin > 0

    active_idx       = np.where(active_mask)[0]
    read_active_idx  = np.where(read_active_mask)[0]
    write_active_idx = np.where(write_active_mask)[0]

    n_active       = int(active_mask.sum())
    n_read_active  = int(read_active_mask.sum())
    n_write_active = int(write_active_mask.sum())

    row['io_active_bins']       = n_active
    row['io_read_active_bins']  = n_read_active
    row['io_write_active_bins'] = n_write_active

    row['io_density']       = n_active / n
    row['io_read_density']  = n_read_active / n
    row['io_write_density'] = n_write_active / n
    # R/W overlap: bins where both read and write are active
    both_active_mask = read_active_mask & write_active_mask
    n_both_active = int(both_active_mask.sum())
    row['io_rw_overlap_bins'] = n_both_active
    row['io_rw_overlap_frac'] = n_both_active / n

    # Burstiness: gap structure between active bins
    if n_active >= 2:
        gaps = np.diff(active_idx)
        row['io_max_gap_bins']  = int(gaps.max())
        row['io_mean_gap_bins'] = float(gaps.mean())
        row['io_n_io_bursts']   = int((gaps > 1).sum() + 1)
    elif n_active == 1:
        row['io_max_gap_bins']  = 0
        row['io_mean_gap_bins'] = 0.0
        row['io_n_io_bursts']   = 1
    # else: defaults from _empty_row (0/0.0/0) apply

    if n_active > 0:
        row['io_first_active_bin'] = int(active_idx[0])
        row['io_last_active_bin']  = int(active_idx[-1])
        row['io_phase_start_frac'] = float(active_idx[0]) / n
        row['io_phase_end_frac']   = float(active_idx[-1]) / n

        third = n // 3
        if third > 0:
            row['io_read_front_heavy'] = bool(
                read_per_bin[:third].sum() > read_per_bin[third:].sum()
            )
            row['io_write_back_heavy'] = bool(
                write_per_bin[2 * third:].sum() > write_per_bin[:2 * third].sum()
            )

    if n_read_active > 0:
        row['io_read_first_active_bin'] = int(read_active_idx[0])
        row['io_read_last_active_bin']  = int(read_active_idx[-1])
        row['io_read_phase_start_frac'] = float(read_active_idx[0]) / n
        row['io_read_phase_end_frac']   = float(read_active_idx[-1]) / n

    if n_write_active > 0:
        row['io_write_first_active_bin'] = int(write_active_idx[0])
        row['io_write_last_active_bin']  = int(write_active_idx[-1])
        row['io_write_phase_start_frac'] = float(write_active_idx[0]) / n
        row['io_write_phase_end_frac']   = float(write_active_idx[-1]) / n

    if row['runtime'] > 0:
        row['io_time_frac'] = min((n_active * bin_width) / row['runtime'], 1.0)
        row['io_read_time_frac'] = min((n_read_active * bin_width) / row['runtime'], 1.0)
        row['io_write_time_frac'] = min((n_write_active * bin_width) / row['runtime'], 1.0)
    else:
        row['io_time_frac'] = 0.0
        row['io_read_time_frac'] = 0.0
        row['io_write_time_frac'] = 0.0

    return row


def _empty_row():
    return {
        'job_id': '', 'fname': '', 'runtime': 0.0, 'nprocs': 0, 'executable': '',
        'has_posix': False, 'has_mpiio': False, 'has_stdio': False,
        'has_lustre': False, 'has_heatmap': False,
        'cb_nodes': 0, 'cb_config_list': '',
        'fs_lustre_grand': False, 'fs_lustre_eagle': False, 'mount_points': '',

        'bytes_read': 0.0, 'bytes_written': 0.0,
        'posix_reads': 0.0, 'posix_writes': 0.0,
        'posix_opens': 0.0, 'posix_stats': 0.0, 'posix_seeks': 0.0,
        'posix_seq_reads': 0.0, 'posix_seq_writes': 0.0,
        'posix_cons_reads': 0.0, 'posix_cons_writes': 0.0,
        'posix_mem_not_aligned': 0.0, 'posix_file_not_aligned': 0.0,

        'small_reads': 0.0, 'medium_reads': 0.0, 'large_reads': 0.0,
        'small_writes': 0.0, 'medium_writes': 0.0, 'large_writes': 0.0,

        'stride_1': 0.0, 'stride_2': 0.0, 'stride_3': 0.0, 'stride_4': 0.0,
        'fastest_rank_bytes': 0.0, 'slowest_rank_bytes': 0.0,

        'io_open_start': 0.0, 'io_read_start': 0.0, 'io_write_start': 0.0,
        'io_read_end': 0.0, 'io_write_end': 0.0,

        'slowest_rank_time': 0.0, 'fastest_rank_time': 0.0,
        'variance_rank_time': 0.0, 'variance_rank_bytes': 0.0,

        'mpiio_bytes_read': 0.0, 'mpiio_bytes_written': 0.0,
        'mpiio_indep_reads': 0.0, 'mpiio_indep_writes': 0.0,
        'mpiio_coll_reads': 0.0, 'mpiio_coll_writes': 0.0,

        'mpiio_read_start': 0.0, 'mpiio_write_start': 0.0,
        'mpiio_read_end': 0.0, 'mpiio_write_end': 0.0,

        'mpiio_slowest_rank_time': 0.0, 'mpiio_fastest_rank_time': 0.0,
        'mpiio_variance_rank_time': 0.0, 'mpiio_variance_rank_bytes': 0.0,

        'stdio_bytes_read': 0.0, 'stdio_bytes_written': 0.0,
        'stdio_reads': 0.0, 'stdio_writes': 0.0,
        'stdio_opens': 0.0, 'stdio_seeks': 0.0,

        # Heatmap-derived combined I/O timing.
        'heatmap_bin_width': 0.0, 'heatmap_total_bins': 0,
        'io_active_bins': 0, 'io_density': 0.0,
        'io_first_active_bin': -1, 'io_last_active_bin': -1,
        'io_phase_start_frac': 0.0, 'io_phase_end_frac': 0.0,
        'io_time_frac': 0.0,

        # Heatmap-derived read/write split timing.
        'io_read_active_bins': 0, 'io_write_active_bins': 0,
        'io_read_density': 0.0, 'io_write_density': 0.0,

        'io_read_first_active_bin': -1, 'io_read_last_active_bin': -1,
        'io_write_first_active_bin': -1, 'io_write_last_active_bin': -1,

        'io_read_phase_start_frac': 0.0, 'io_read_phase_end_frac': 0.0,
        'io_write_phase_start_frac': 0.0, 'io_write_phase_end_frac': 0.0,

        'io_read_time_frac': 0.0,
        'io_write_time_frac': 0.0,
                # R/W overlap and burstiness
        'io_rw_overlap_bins': 0,
        'io_rw_overlap_frac': 0.0,
        'io_max_gap_bins': 0,
        'io_mean_gap_bins': 0.0,
        'io_n_io_bursts': 0,


        'io_read_front_heavy': False,
        'io_write_back_heavy': False,

        'unique_files': 0,

        'seq_read_ratio': 0.0, 'seq_write_ratio': 0.0,
        'cons_read_ratio': 0.0, 'cons_write_ratio': 0.0,

        'total_read_ops': 0.0, 'total_write_ops': 0.0,
        'small_read_ratio': 0.0, 'medium_read_ratio': 0.0, 'large_read_ratio': 0.0,
        'small_write_ratio': 0.0, 'medium_write_ratio': 0.0, 'large_write_ratio': 0.0,

        'stride_1_frac': 0.0, 'stride_2_frac': 0.0,
        'stride_3_frac': 0.0, 'stride_4_frac': 0.0,

        'rank_imbalance': 0.0,
        'rank_time_imbalance': 0.0,
        'rank_time_gap': 0.0,

        'mem_not_aligned_ratio': 0.0,
        'file_not_aligned_ratio': 0.0,

        'mpiio_coll_ratio': 0.0,

        'io_intensity': 0.0,
        'write_dominance': 0.0,
        'io_overlap_frac': 0.0,
        'io_time_frac_rank': 0.0,
        'metadata_ops_per_gb': 0.0,

        # POSIX time breakdown
        'posix_meta_time':  0.0,
        'posix_read_time':  0.0,
        'posix_write_time': 0.0,
        'meta_time_frac':   0.0,
        'read_time_frac':   0.0,
        'write_time_frac':  0.0,
    }

def _compute_derived(row):
    def _safe_div(a, b): return a / b if b > 0 else 0.0
    row['seq_read_ratio']   = _safe_div(row['posix_seq_reads'],  row['posix_reads'])
    row['seq_write_ratio']  = _safe_div(row['posix_seq_writes'], row['posix_writes'])
    row['cons_read_ratio']  = _safe_div(row['posix_cons_reads'], row['posix_reads'])
    row['cons_write_ratio'] = _safe_div(row['posix_cons_writes'],row['posix_writes'])
    total_read_ops  = row['small_reads']  + row['medium_reads']  + row['large_reads']
    total_write_ops = row['small_writes'] + row['medium_writes'] + row['large_writes']
    row['total_read_ops']     = total_read_ops
    row['total_write_ops']    = total_write_ops
    row['small_read_ratio']   = _safe_div(row['small_reads'],   total_read_ops)
    row['medium_read_ratio']  = _safe_div(row['medium_reads'],  total_read_ops)
    row['large_read_ratio']   = _safe_div(row['large_reads'],   total_read_ops)
    row['small_write_ratio']  = _safe_div(row['small_writes'],  total_write_ops)
    row['medium_write_ratio'] = _safe_div(row['medium_writes'], total_write_ops)
    row['large_write_ratio']  = _safe_div(row['large_writes'],  total_write_ops)
    total_strides = row['stride_1'] + row['stride_2'] + row['stride_3'] + row['stride_4']
    row['stride_1_frac'] = _safe_div(row['stride_1'], total_strides)
    row['stride_2_frac'] = _safe_div(row['stride_2'], total_strides)
    row['stride_3_frac'] = _safe_div(row['stride_3'], total_strides)
    row['stride_4_frac'] = _safe_div(row['stride_4'], total_strides)
    row['rank_imbalance'] = _safe_div(row['slowest_rank_bytes'], row['fastest_rank_bytes'])
    if row['fastest_rank_time'] > 0.001:
        row['rank_time_imbalance'] = _safe_div(row['slowest_rank_time'], row['fastest_rank_time'])
    else:
        row['rank_time_imbalance'] = 0.0
    row['rank_time_gap'] = row['slowest_rank_time'] - row['fastest_rank_time']
    total_ops = row['posix_reads'] + row['posix_writes']
    row['mem_not_aligned_ratio']  = _safe_div(row['posix_mem_not_aligned'],  total_ops)
    row['file_not_aligned_ratio'] = _safe_div(row['posix_file_not_aligned'], total_ops)
    total_coll  = row['mpiio_coll_reads']  + row['mpiio_coll_writes']
    total_mpiio = total_coll + row['mpiio_indep_reads'] + row['mpiio_indep_writes']
    row['mpiio_coll_ratio'] = _safe_div(total_coll, total_mpiio)
    total_bytes = row['bytes_read'] + row['bytes_written']
    row['io_intensity']    = _safe_div(total_bytes, row['slowest_rank_time'])
    row['write_dominance'] = _safe_div(row['bytes_written'], total_bytes)
    row['io_overlap_frac'] = row['io_time_frac']
    row['metadata_ops_per_gb'] = (
        _safe_div(row['posix_opens'] + row['posix_stats'], total_bytes / 1e9)
        if total_bytes > 1e6 else 0.0
    )
    row['io_time_frac_rank'] = _safe_div(row['slowest_rank_time'], row['runtime'])

    # Time-based attribution of POSIX I/O time across operation types
    total_posix_io_time = (row['posix_meta_time'] +
                            row['posix_read_time'] +
                            row['posix_write_time'])
    row['meta_time_frac']  = _safe_div(row['posix_meta_time'],  total_posix_io_time)
    row['read_time_frac']  = _safe_div(row['posix_read_time'],  total_posix_io_time)
    row['write_time_frac'] = _safe_div(row['posix_write_time'], total_posix_io_time)

    return row


