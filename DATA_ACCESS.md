# Data Access

This framework consumes three public ALCF Polaris telemetry sources. They are
**not** included in this repository (they are tens of GB); download them from
ALCF and point `config/config.json` at your local copies.

## 1. GPU telemetry — `POLARIS_GPU_NODE`

Per-node GPU metrics (utilization, memory utilization, power, temperature),
sampled at 30-second cadence, as CSV (gzip).

- **Source:** <https://reports.alcf.anl.gov/data/polaris.html> → `POLARIS_GPU_NODE`
- **File used for the paper:** `ANL-ALCF-GPU-NODE-POLARIS_20250101_20251231.csv` (~79 GB)
- Data dictionary: `datadictionary_gpu_node.html` on the same site.

Convert once to date-partitioned Parquet (fast date-range reads) via
`pipeline/stage00_convert_gpu_to_parquet.py`.

## 2. Scheduler job records — `DIM_JOB_COMPOSITE`

Per-job scheduler records: anonymized user/project identifiers, queue, requested
and used nodes/cores, requested walltime, runtime, exit status, `GPUS_REQUESTED`,
and queued/start/end timestamps. `JOB_NAME` is the join key used throughout the
pipeline.

- **Source:** <https://reports.alcf.anl.gov/data/polaris.html> → `DIM_JOB_COMPOSITE`
- **File used for the paper:** `..._DIM_JOB_COMPOSITE_..._20250101_20251231.csv` (~140 MB)
- Data dictionary: `datadictionary_dim_job_composite.html`.

## 3. Darshan I/O logs (Polaris)

Application-level I/O summaries (POSIX, MPI-IO, stdio) plus the POSIX heatmap,
anonymized while preserving the original `JOB_NAME`.

- **Source:** ALCF Polaris Darshan log collection, described by **Snyder et al.,
  CUG'25** — *Proceedings of the Cray User Group 2025*, DOI
  [10.1145/3757348](https://doi.org/10.1145/3757348). Follow that paper's
  collection and anonymization workflow to obtain the logs.
- **Reference scripts:** the companion repo *Polaris Darshan Log Collection
  Scripts (CUG'25)* documents the collection/anonymization and analysis steps.
- Logs are consumed as daily tarballs laid out as
  `<darshan_data>/<year>/<month>/<day>/logs.tar.gz`; `pipeline/stage02_parse_darshan.py`
  walks that tree and parses each `.darshan` with PyDarshan.

## Configure

Edit `config/config.json` to point at your downloads:

```json
{
  "djc_csv":            "/path/to/DIM_JOB_COMPOSITE.csv",
  "gpu_telemetry_gz":   "/path/to/POLARIS_GPU_NODE.csv.gz",
  "parquet_out_dir":    "/path/to/gpu_telemetry_parquet",
  "darshan_data":       "/path/to/darshan_tarball_tree",
  "darshan_parsed_out": "/path/to/darshan_metrics_final.csv",
  "gpu_parsed_out":     "/path/to/gpu_metrics.csv",
  "combined_out":       "/path/to/combined_metrics.csv",
  "chunk_size":         500000
}
```

Then run the pipeline as described in the README. All layers are joined by
`JOB_NAME`; missing telemetry is preserved as *unavailable* (not zero) so that
absent signals are not confused with observed-zero activity.
