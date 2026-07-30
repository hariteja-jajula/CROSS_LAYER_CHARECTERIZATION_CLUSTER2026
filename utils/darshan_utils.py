"""
darshan_utils.py — UPDATED with additional extraction fields

New fields added (search for "# NEW" comments):
  - executable: hashed exe path from header
  - unique_files: count of unique file records
  - posix_opens, posix_stats, posix_seeks: metadata operation counts
  - posix_cons_reads, posix_cons_writes: consecutive I/O (vs sequential)
  - cons_read_ratio, cons_write_ratio: consecutive fractions
  - posix_mem_not_aligned, posix_file_not_aligned: alignment miss counts
  - fastest_rank_time: for dual-sided rank timing
  - fastest_rank_bytes: raw value (previously only used in ratio)
  - slowest_rank_bytes: raw value
  - mpiio_open_start, mpiio_read_end, mpiio_write_end: MPI-IO timestamps
  - stdio_opens, stdio_seeks: STDIO metadata ops
  - stride_1_reads, stride_2_reads, ...: stride pattern counters
  - per-bucket raw counts: small_reads_count, medium_reads_count, large_reads_count, etc.
  - io_density: active_bins / total_bins (how fragmented is I/O)
  - heatmap_total_bins: for cross-validation
  - mount_points: comma-separated list of mount points
"""

def parse_darshan_output(output):
    lines = output.split('\n')
    row = {
        'job_id': '', 'runtime': 0.0, 'nprocs': 0,
        'executable': '',                                     # NEW
        'has_posix': False, 'has_mpiio': False, 'has_stdio': False,
        'has_lustre': False,                                  # NEW
        'bytes_read': 0.0, 'bytes_written': 0.0,
        'posix_reads': 0.0, 'posix_writes': 0.0,
        'posix_opens': 0.0, 'posix_stats': 0.0,              # NEW
        'posix_seeks': 0.0,                                   # NEW
        'seq_read_ratio': 0.0, 'seq_write_ratio': 0.0,
        'cons_read_ratio': 0.0, 'cons_write_ratio': 0.0,     # NEW
        'small_read_ratio': 0.0, 'small_write_ratio': 0.0,
        'rank_imbalance': 0.0, 'slowest_rank_time': 0.0,
        'fastest_rank_time': 0.0,                             # NEW
        'cb_nodes': 0,
        'cb_config_list': 0,                                  # NEW
        'fs_lustre_grand': False, 'fs_lustre_eagle': False,
        'io_open_start': 0.0, 'io_read_start': 0.0, 'io_write_start': 0.0,
        'io_read_end': 0.0, 'io_write_end': 0.0,
        'mpiio_bytes_read': 0.0, 'mpiio_bytes_written': 0.0,
        'mpiio_indep_reads': 0.0, 'mpiio_indep_writes': 0.0,
        'mpiio_coll_reads': 0.0, 'mpiio_coll_writes': 0.0,
        'mpiio_coll_ratio': 0.0,
        'mpiio_read_start': 0.0, 'mpiio_write_start': 0.0,   # NEW
        'mpiio_read_end': 0.0, 'mpiio_write_end': 0.0,       # NEW
        'stdio_bytes_read': 0.0, 'stdio_bytes_written': 0.0,
        'stdio_reads': 0.0, 'stdio_writes': 0.0,
        'stdio_opens': 0.0, 'stdio_seeks': 0.0,              # NEW
        'heatmap_bin_width': 0.0,
        'io_active_bins': 0, 'io_first_active_bin': -1, 'io_last_active_bin': -1,
        'heatmap_total_bins': 0,                              # NEW
        'io_phase_start_frac': 0.0, 'io_phase_end_frac': 0.0,
        'io_density': 0.0,                                    # NEW
        'io_read_front_heavy': False, 'io_write_back_heavy': False,
        'io_intensity': 0.0, 'write_dominance': 0.0, 'io_overlap_frac': 0.0,
        'unique_files': 0,                                    # NEW
        'mount_points': '',                                   # NEW
    }

    acc = {
        'posix_reads': 0.0, 'posix_writes': 0.0,
        'posix_opens': 0.0, 'posix_stats': 0.0,              # NEW
        'posix_seeks': 0.0,                                   # NEW
        'posix_seq_reads': 0.0, 'posix_seq_writes': 0.0,
        'posix_cons_reads': 0.0, 'posix_cons_writes': 0.0,   # NEW
        'posix_mem_not_aligned': 0.0,                         # NEW
        'posix_file_not_aligned': 0.0,                        # NEW
        'small_reads': 0.0, 'medium_reads': 0.0, 'large_reads': 0.0,
        'small_writes': 0.0, 'medium_writes': 0.0, 'large_writes': 0.0,
        'fastest_rank_bytes': 0.0, 'slowest_rank_bytes': 0.0,
        'fastest_rank_time': 0.0,                             # NEW
        'file_records_seen': set(),                           # NEW
        'stride_1': 0.0, 'stride_2': 0.0, 'stride_3': 0.0, 'stride_4': 0.0,  # NEW
    }
    heatmap_read_records = {}
    heatmap_write_records = {}

    row = parse_header(lines, row)

    for line in lines:
        if line.startswith('#'):
            continue
        parts = line.split('\t')
        if len(parts) < 5:
            continue
        mod, rank, record_id = parts[0], parts[1], parts[2]
        try: parts[4] = float(parts[4])
        except: continue
        if parts[4] == -1: continue

        # NEW: track unique file records
        if record_id not in ('', '0'):
            acc['file_records_seen'].add(record_id)

        if mod == 'POSIX':
            parse_posix(parts, row, acc)
        elif mod == 'MPI-IO':
            parse_mpiio(parts, row)
        elif mod == 'STDIO':
            parse_stdio(parts, row, acc)
        elif mod == 'LUSTRE':                                 # NEW
            row['has_lustre'] = True
        elif mod == 'HEATMAP' and rank == '0':
            if parts[3] == 'HEATMAP_F_BIN_WIDTH_SECONDS':
                row['heatmap_bin_width'] = parts[4]
                heatmap_read_records.setdefault(record_id, [])
                heatmap_write_records.setdefault(record_id, [])
            elif parts[3].startswith('HEATMAP_READ_BIN_'):
                heatmap_read_records.setdefault(record_id, []).append(parts[4])
            elif parts[3].startswith('HEATMAP_WRITE_BIN_'):
                heatmap_write_records.setdefault(record_id, []).append(parts[4])

    # NEW: unique file count
    row['unique_files'] = len(acc['file_records_seen'])

    # derived POSIX ratios
    row['posix_reads'] = acc['posix_reads']
    row['posix_writes'] = acc['posix_writes']
    row['posix_opens'] = acc['posix_opens']                   # NEW
    row['posix_stats'] = acc['posix_stats']                   # NEW
    row['posix_seeks'] = acc['posix_seeks']                   # NEW
    row['seq_read_ratio'] = acc['posix_seq_reads'] / acc['posix_reads'] if acc['posix_reads'] > 0 else 0.0
    row['seq_write_ratio'] = acc['posix_seq_writes'] / acc['posix_writes'] if acc['posix_writes'] > 0 else 0.0
    # NEW: consecutive ratios (stricter than sequential — truly contiguous)
    row['cons_read_ratio'] = acc['posix_cons_reads'] / acc['posix_reads'] if acc['posix_reads'] > 0 else 0.0
    row['cons_write_ratio'] = acc['posix_cons_writes'] / acc['posix_writes'] if acc['posix_writes'] > 0 else 0.0

    total_read_ops = acc['small_reads'] + acc['medium_reads'] + acc['large_reads']
    total_write_ops = acc['small_writes'] + acc['medium_writes'] + acc['large_writes']
    row['small_read_ratio'] = acc['small_reads'] / total_read_ops if total_read_ops > 0 else 0.0
    row['small_write_ratio'] = acc['small_writes'] / total_write_ops if total_write_ops > 0 else 0.0

    # NEW: raw bucket counts for forensic analysis
    row['small_reads_count'] = acc['small_reads']
    row['medium_reads_count'] = acc['medium_reads']
    row['large_reads_count'] = acc['large_reads']
    row['small_writes_count'] = acc['small_writes']
    row['medium_writes_count'] = acc['medium_writes']
    row['large_writes_count'] = acc['large_writes']
    row['total_read_ops'] = total_read_ops
    row['total_write_ops'] = total_write_ops

    row['rank_imbalance'] = acc['slowest_rank_bytes'] / acc['fastest_rank_bytes'] if acc['fastest_rank_bytes'] > 0 else 0.0
    # NEW: raw rank bytes for consistency checks
    row['fastest_rank_bytes'] = acc['fastest_rank_bytes']
    row['slowest_rank_bytes'] = acc['slowest_rank_bytes']
    row['fastest_rank_time'] = acc['fastest_rank_time']

    # NEW: alignment miss ratios
    total_ops = acc['posix_reads'] + acc['posix_writes']
    row['mem_not_aligned_ratio'] = acc['posix_mem_not_aligned'] / total_ops if total_ops > 0 else 0.0
    row['file_not_aligned_ratio'] = acc['posix_file_not_aligned'] / total_ops if total_ops > 0 else 0.0

    # NEW: stride pattern
    total_strides = acc['stride_1'] + acc['stride_2'] + acc['stride_3'] + acc['stride_4']
    row['stride_1_frac'] = acc['stride_1'] / total_strides if total_strides > 0 else 0.0

    # derived MPI-IO
    total_coll = row['mpiio_coll_reads'] + row['mpiio_coll_writes']
    total_mpiio = total_coll + row['mpiio_indep_reads'] + row['mpiio_indep_writes']
    row['mpiio_coll_ratio'] = total_coll / total_mpiio if total_mpiio > 0 else 0.0

    # heatmap temporal
    if heatmap_read_records:
        n = len(next(iter(heatmap_read_records.values())))
        row['heatmap_total_bins'] = n
        read_union  = [sum(heatmap_read_records[rid][i]  for rid in heatmap_read_records  if i < len(heatmap_read_records[rid]))  for i in range(n)]
        write_union = [sum(heatmap_write_records[rid][i] for rid in heatmap_write_records if i < len(heatmap_write_records[rid])) for i in range(n)]
        active = [i for i, (r, w) in enumerate(zip(read_union, write_union)) if r > 0 or w > 0]
        row['io_active_bins'] = len(active)
        row['io_density'] = len(active) / n if n > 0 else 0.0
        if active:
            row['io_first_active_bin'] = active[0]
            row['io_last_active_bin']  = active[-1]
            row['io_phase_start_frac'] = active[0] / n
            row['io_phase_end_frac']   = active[-1] / n
            third = n // 3
            row['io_read_front_heavy']  = sum(read_union[:third])    > sum(read_union[third:])
            row['io_write_back_heavy']  = sum(write_union[2*third:]) > sum(write_union[:2*third])

    # derived I/O characterization
    total_bytes = row['bytes_read'] + row['bytes_written']
    row['io_intensity'] = total_bytes / row['slowest_rank_time'] if row['slowest_rank_time'] > 0 else 0.0
    row['write_dominance'] = row['bytes_written'] / total_bytes if total_bytes > 0 else 0.0
    raw_frac = (row['io_active_bins'] * row['heatmap_bin_width']) / row['runtime'] if row['runtime'] > 0 else 0.0
    row['io_overlap_frac'] = min(raw_frac, 1.0)
    row['io_time_frac'] = row['io_overlap_frac']

    # NEW: metadata intensity — opens+stats per byte (high = metadata-heavy)
    row['metadata_ops_per_gb'] = (acc['posix_opens'] + acc['posix_stats']) / (total_bytes / 1e9) if total_bytes > 1e6 else 0.0

    return row


def parse_header(lines, row):
    mounts = []                                               # NEW
    for line in lines:
        if not line.startswith('#'):
            continue
        if 'run time:' in line:
            try: row['runtime'] = float(line.split(':', 1)[1].strip())
            except: pass
        elif 'nprocs:' in line:
            try: row['nprocs'] = int(line.split(':', 1)[1].strip())
            except: pass
        elif 'exe:' in line.lower():                          # NEW
            try:
                exe_part = line.split(':', 1)[1].strip()
                row['executable'] = exe_part.split()[0] if exe_part else ''
            except: pass
        elif 'POSIX module' in line: row['has_posix'] = True
        elif 'MPI-IO module' in line: row['has_mpiio'] = True
        elif 'STDIO module' in line: row['has_stdio'] = True
        elif 'cb_nodes=' in line:
            try: row['cb_nodes'] = int(line.split('cb_nodes=')[1].split(';')[0])
            except: pass
            # NEW: also extract cb_config_list
            if 'cb_config_list=' in line:
                try: row['cb_config_list'] = int(line.split('cb_config_list=')[1].split(';')[0])
                except: pass
        elif '/lus/grand' in line and 'lustre' in line: row['fs_lustre_grand'] = True
        elif '/lus/eagle' in line and 'lustre' in line: row['fs_lustre_eagle'] = True
        # NEW: collect all mount points
        if 'mount entry:' in line.lower() or ('\t' in line and '/lus/' in line):
            try:
                parts = line.split('\t')
                for p in parts:
                    p = p.strip().lstrip('#').strip()
                    if p.startswith('/'):
                        mounts.append(p.split()[0])
            except: pass

    row['mount_points'] = ','.join(sorted(set(mounts))) if mounts else ''
    return row


def parse_posix(line, row, acc):
    c, v = line[3], line[4]
    if c == 'POSIX_BYTES_READ': row['bytes_read'] += v
    elif c == 'POSIX_BYTES_WRITTEN': row['bytes_written'] += v
    elif c == 'POSIX_READS': acc['posix_reads'] += v
    elif c == 'POSIX_WRITES': acc['posix_writes'] += v
    elif c == 'POSIX_OPENS': acc['posix_opens'] += v          # NEW
    elif c == 'POSIX_STATS': acc['posix_stats'] += v          # NEW
    elif c == 'POSIX_SEEKS': acc['posix_seeks'] += v          # NEW
    elif c == 'POSIX_SEQ_READS': acc['posix_seq_reads'] += v
    elif c == 'POSIX_SEQ_WRITES': acc['posix_seq_writes'] += v
    elif c == 'POSIX_CONSEC_READS': acc['posix_cons_reads'] += v    # NEW
    elif c == 'POSIX_CONSEC_WRITES': acc['posix_cons_writes'] += v  # NEW
    elif c == 'POSIX_MEM_NOT_ALIGNED': acc['posix_mem_not_aligned'] += v    # NEW
    elif c == 'POSIX_FILE_NOT_ALIGNED': acc['posix_file_not_aligned'] += v  # NEW
    elif c == 'POSIX_FASTEST_RANK_BYTES': acc['fastest_rank_bytes'] += v
    elif c == 'POSIX_SLOWEST_RANK_BYTES': acc['slowest_rank_bytes'] += v
    elif c == 'POSIX_F_SLOWEST_RANK_TIME': row['slowest_rank_time'] = max(row['slowest_rank_time'], v)
    elif c == 'POSIX_F_FASTEST_RANK_TIME': acc['fastest_rank_time'] = max(acc['fastest_rank_time'], v)  # NEW
    elif c in ['POSIX_SIZE_READ_0_100', 'POSIX_SIZE_READ_100_1K', 'POSIX_SIZE_READ_1K_10K']: acc['small_reads'] += v
    elif c in ['POSIX_SIZE_READ_10K_100K', 'POSIX_SIZE_READ_100K_1M']: acc['medium_reads'] += v
    elif c in ['POSIX_SIZE_READ_1M_4M', 'POSIX_SIZE_READ_4M_10M', 'POSIX_SIZE_READ_10M_100M', 'POSIX_SIZE_READ_100M_1G', 'POSIX_SIZE_READ_1G_PLUS']: acc['large_reads'] += v
    elif c in ['POSIX_SIZE_WRITE_0_100', 'POSIX_SIZE_WRITE_100_1K', 'POSIX_SIZE_WRITE_1K_10K']: acc['small_writes'] += v
    elif c in ['POSIX_SIZE_WRITE_10K_100K', 'POSIX_SIZE_WRITE_100K_1M']: acc['medium_writes'] += v
    elif c in ['POSIX_SIZE_WRITE_1M_4M', 'POSIX_SIZE_WRITE_4M_10M', 'POSIX_SIZE_WRITE_10M_100M', 'POSIX_SIZE_WRITE_100M_1G', 'POSIX_SIZE_WRITE_1G_PLUS']: acc['large_writes'] += v
    elif c == 'POSIX_F_OPEN_START_TIMESTAMP': row['io_open_start'] = v
    elif c == 'POSIX_F_READ_START_TIMESTAMP': row['io_read_start'] = v
    elif c == 'POSIX_F_WRITE_START_TIMESTAMP': row['io_write_start'] = v
    elif c == 'POSIX_F_READ_END_TIMESTAMP': row['io_read_end'] = v
    elif c == 'POSIX_F_WRITE_END_TIMESTAMP': row['io_write_end'] = v
    # NEW: stride pattern counters
    elif c == 'POSIX_STRIDE1_STRIDE': acc['stride_1'] += v
    elif c == 'POSIX_STRIDE2_STRIDE': acc['stride_2'] += v
    elif c == 'POSIX_STRIDE3_STRIDE': acc['stride_3'] += v
    elif c == 'POSIX_STRIDE4_STRIDE': acc['stride_4'] += v


def parse_mpiio(line, row):
    c, v = line[3], line[4]
    if c == 'MPIIO_BYTES_READ': row['mpiio_bytes_read'] += v
    elif c == 'MPIIO_BYTES_WRITTEN': row['mpiio_bytes_written'] += v
    elif c == 'MPIIO_INDEP_READS': row['mpiio_indep_reads'] += v
    elif c == 'MPIIO_INDEP_WRITES': row['mpiio_indep_writes'] += v
    elif c == 'MPIIO_COLL_READS': row['mpiio_coll_reads'] += v
    elif c == 'MPIIO_COLL_WRITES': row['mpiio_coll_writes'] += v
    # NEW: MPI-IO timestamps
    elif c == 'MPIIO_F_READ_START_TIMESTAMP':
        if v > 0: row['mpiio_read_start'] = min(row['mpiio_read_start'], v) if row['mpiio_read_start'] > 0 else v
    elif c == 'MPIIO_F_WRITE_START_TIMESTAMP':
        if v > 0: row['mpiio_write_start'] = min(row['mpiio_write_start'], v) if row['mpiio_write_start'] > 0 else v
    elif c == 'MPIIO_F_READ_END_TIMESTAMP':
        row['mpiio_read_end'] = max(row['mpiio_read_end'], v)
    elif c == 'MPIIO_F_WRITE_END_TIMESTAMP':
        row['mpiio_write_end'] = max(row['mpiio_write_end'], v)


def parse_stdio(line, row, acc=None):
    c, v = line[3], line[4]
    if c == 'STDIO_BYTES_READ': row['stdio_bytes_read'] += v
    elif c == 'STDIO_BYTES_WRITTEN': row['stdio_bytes_written'] += v
    elif c == 'STDIO_READS': row['stdio_reads'] += v
    elif c == 'STDIO_WRITES': row['stdio_writes'] += v
    # NEW
    elif c == 'STDIO_OPENS': row['stdio_opens'] += v
    elif c == 'STDIO_SEEKS': row['stdio_seeks'] += v
