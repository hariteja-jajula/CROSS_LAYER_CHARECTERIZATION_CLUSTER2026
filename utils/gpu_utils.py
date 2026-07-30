import pandas as pd
import pyarrow.dataset as ds
from itertools import groupby


# ── data loading ──────────────────────────────────────────────────────────────

def process_job(args_tuple):
    job, cfg = args_tuple
    dataset = ds.dataset(cfg["parquet_out_dir"], partitioning="hive")
    gpu_df = extract_gpu_data(job, dataset)
    job_metrics = aggregate_gpu_metrics(gpu_df, job, cfg)
    per_device  = aggregate_gpu_per_device(gpu_df, job, cfg)
    return job_metrics, per_device


def extract_gpu_data(job, dataset):
    start = pd.to_datetime(job["START_TIMESTAMP"])
    end   = pd.to_datetime(job["END_TIMESTAMP"])
    try:
        gpu_df = dataset.to_table(
            filter=(ds.field("date") >= start.date().isoformat()) &
                   (ds.field("date") <= end.date().isoformat())
        ).to_pandas()
    except Exception as e:
        print(f"Skipping {job['JOB_NAME']} — parquet error: {e}", flush=True)
        return pd.DataFrame()

    gpu_df["TIMESTAMP"] = pd.to_datetime(gpu_df["TIMESTAMP"])
    mask = (
        gpu_df["HOST"].isin(set(str(job["LOCATION"]).split(","))) &
        gpu_df["TIMESTAMP"].between(start, end)
    )
    return gpu_df[mask].reset_index(drop=True)


# ── shared metric helper ──────────────────────────────────────────────────────
# Operates on plain Series (util, mem_util, pwr, temp) + sorted gpu_df slice.
# Called by both job-level and per-device aggregators.

def _compute_metrics_on_series(base, util, mem_util, pwr, temp, gpu_df, cfg=None):
    """
    base     : dict with at least JOB_NAME already set
    util     : GPU utilization Series
    mem_util : GPU memory utilization Series
    pwr      : GPU power Series
    temp     : GPU temperature Series
    gpu_df   : sorted DataFrame (needs TIMESTAMP, HOST)
    """
    res = dict(base)
    t_min = gpu_df["TIMESTAMP"].min()
    t_max = gpu_df["TIMESTAMP"].max()
    runtime_sec = (t_max - t_min).total_seconds()

    temp_max = temp  # for per-device temp IS the single series; job-level passes max-across-GPUs

    res.update({
        # utilization
        "util_mean":          util.mean(),
        "util_max":           util.max(),
        "util_std":           util.std(),
        "util_p25":           util.quantile(0.25),
        "util_p50":           util.quantile(0.50),
        "util_p75":           util.quantile(0.75),
        "util_p95":           util.quantile(0.95),
        "zero_util_frac":     (util == 0).mean(),
        "idle_frac":          (util < 5).mean(),
        "active_phase_frac":  (util > 10).mean(),
        # memory
        "mem_util_mean":      mem_util.mean(),
        "mem_util_max":       mem_util.max(),
        "mem_pressure_frac":  (mem_util > 80).mean(),
        "mem_bound_frac":     ((mem_util > 70) & (util < 40)).mean(),
        # power
        "power_mean":         pwr.mean(),
        "power_max":          pwr.max(),
        "power_std":          pwr.std(),
        "power_p95":          pwr.quantile(0.95),
        "power_efficiency":   util.mean() / (pwr.mean() + 1e-6),
        # temperature
        "temp_mean":          temp.mean(),
        "temp_max":           temp.max(),
        "temp_p95":           temp.quantile(0.95),
        "temp_headroom_mean": (83 - temp).mean(),
        "thermal_throttle_frac":    (temp > 83).mean(),
        "sustained_throttle_frac":  (temp.rolling(5).mean() > 80).mean(),
        # cross-signal
        "high_util_low_mem_frac":   ((util > 50) & (mem_util < 20)).mean(),
        "high_mem_low_util_frac":   ((mem_util > 50) & (util < 20)).mean(),
        "high_power_low_util_frac": ((pwr > pwr.quantile(0.75)) & (util < 20)).mean(),
    })

    # idle burst: longest consecutive idle stretch
    runs = [(k, sum(1 for _ in g)) for k, g in groupby((util < 5).values)]
    res["max_consecutive_idle_readings"] = max(
        (length for is_idle, length in runs if is_idle), default=0
    )

    # warmup / cooldown — first and last 10% of readings
    n     = len(gpu_df)
    tenth = max(1, n // 10)
    res["util_warmup_mean"]   = util.iloc[:tenth].mean()
    res["util_cooldown_mean"] = util.iloc[-tenth:].mean()

    # power cap proximity
    power_cap = (cfg or {}).get("gpu_power_cap_w", 400)
    res["power_cap_proximity_mean"] = (pwr / power_cap).mean()
    res["near_power_cap_frac"]      = (pwr > 0.9 * power_cap).mean()

    # telemetry coverage
    dcgm_interval = (cfg or {}).get("dcgm_interval_sec", 30)
    expected = runtime_sec / dcgm_interval
    res["telemetry_coverage_frac"] = len(gpu_df) / max(1, expected)
    res["telemetry_gap_detected"]  = int(res["telemetry_coverage_frac"] < 0.8)

    # temporal phases
    half = pd.Timedelta(seconds=runtime_sec / 2)
    b1   = pd.Timedelta(seconds=runtime_sec / 3)
    b2   = pd.Timedelta(seconds=2 * runtime_sec / 3)
    ts   = gpu_df["TIMESTAMP"]

    fh = util[ts <= t_min + half].mean()
    sh = util[ts >  t_min + half].mean()
    res.update({
        "util_first_half":  fh,
        "util_second_half": sh,
        "phase_drop":       fh - sh,
        "util_phase1": util[ts <= t_min + b1].mean(),
        "util_phase2": util[ts.between(t_min + b1, t_min + b2, inclusive="right")].mean(),
        "util_phase3": util[ts >  t_min + b2].mean(),
    })
    return res


# ── job-level aggregation ─────────────────────────────────────────────────────

def aggregate_gpu_metrics(gpu_df, job, cfg=None):
    res = {"JOB_NAME": job["JOB_NAME"], "gpu_telemetry_rows": len(gpu_df)}
    if gpu_df.empty:
        return res

    gpu_df = gpu_df.sort_values("TIMESTAMP").reset_index(drop=True)

    cols = {k: [f"GPU_{k.upper()}_{i}" for i in range(4)] for k in [
        "utilization", "memory_utilization", "memory_allocation_kb", "power_usage", "temperature"
    ]}

    # job-level uses AVG columns as the primary series
    util     = gpu_df["GPU_UTILIZATION_AVG"]
    mem_util = gpu_df["GPU_MEMORY_UTILIZATION_AVG"]
    pwr      = gpu_df["GPU_POWER_USAGE_AVG"]
    # for temperature at job level, use worst-case GPU per row
    temp_max_per_row = gpu_df[cols["temperature"]].max(axis=1)

    res = _compute_metrics_on_series(res, util, mem_util, pwr, temp_max_per_row, gpu_df, cfg)

    # job-level-only: imbalance across GPUs (meaningless per-device)
    res["gpu_imbalance_mean"]     = gpu_df[cols["utilization"]].std(axis=1).mean()
    res["gpu_imbalance_max"]      = gpu_df[cols["utilization"]].std(axis=1).max()
    res["gpu_mem_imbalance_mean"] = gpu_df[cols["memory_utilization"]].std(axis=1).mean()
    res["gpu_power_imbalance_mean"] = gpu_df[cols["power_usage"]].std(axis=1).mean()
    res["gpu_temp_imbalance_mean"]  = gpu_df[cols["temperature"]].std(axis=1).mean()

    # memory allocation (AVG column exists in schema)
    res["gpu_mem_alloc_mean_kb"] = gpu_df[cols["memory_allocation_kb"]].mean().mean()
    res["gpu_mem_alloc_max_kb"]  = gpu_df[cols["memory_allocation_kb"]].max().max()
    res["gpu_mem_alloc_std_kb"]  = gpu_df[cols["memory_allocation_kb"]].mean(axis=1).std()

    # node-level imbalance
    node_means = gpu_df.groupby("HOST")["GPU_UTILIZATION_AVG"].mean()
    res["node_util_imbalance_std"] = node_means.std()
    res["node_util_imbalance_max"] = node_means.max() - node_means.min()
    res["node_count_observed"]     = gpu_df["HOST"].nunique()

    return res


# ── per-device aggregation ────────────────────────────────────────────────────

def aggregate_gpu_per_device(gpu_df, job, cfg=None):
    """Returns a list of 4 dicts, one per GPU (GPU_ID 0-3).
    Each row has the same metrics as job-level minus imbalance/node fields,
    prefixed consistently for easy concat into gpu_per_device_metrics.csv.
    """
    if gpu_df.empty:
        return [{"JOB_NAME": job["JOB_NAME"], "GPU_ID": i, "gpu_telemetry_rows": 0}
                for i in range(4)]

    gpu_df = gpu_df.sort_values("TIMESTAMP").reset_index(drop=True)
    records = []

    for i in range(4):
        base = {
            "JOB_NAME":           job["JOB_NAME"],
            "GPU_ID":             i,
            "gpu_telemetry_rows": len(gpu_df),
        }
        util     = gpu_df[f"GPU_UTILIZATION_{i}"]
        mem_util = gpu_df[f"GPU_MEMORY_UTILIZATION_{i}"]
        pwr      = gpu_df[f"GPU_POWER_USAGE_{i}"]
        temp     = gpu_df[f"GPU_TEMPERATURE_{i}"]

        rec = _compute_metrics_on_series(base, util, mem_util, pwr, temp, gpu_df, cfg)

        # memory allocation per device
        rec["mem_alloc_mean_kb"] = gpu_df[f"GPU_MEMORY_ALLOCATION_KB_{i}"].mean()
        rec["mem_alloc_max_kb"]  = gpu_df[f"GPU_MEMORY_ALLOCATION_KB_{i}"].max()
        rec["mem_alloc_std_kb"]  = gpu_df[f"GPU_MEMORY_ALLOCATION_KB_{i}"].std()

        records.append(rec)

    return records