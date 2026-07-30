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

def aggregate_darshan(darshan_df):
    print("Aggregating darshan by job_id...", flush=True)
    
    agg = darshan_df.groupby("job_id").agg(
        bytes_read=("bytes_read", "sum"),
        bytes_written=("bytes_written", "sum"),
        posix_reads=("posix_reads", "sum"),
        posix_writes=("posix_writes", "sum"),
        mpiio_bytes_read=("mpiio_bytes_read", "sum"),
        mpiio_bytes_written=("mpiio_bytes_written", "sum"),
        mpiio_indep_reads=("mpiio_indep_reads", "sum"),
        mpiio_indep_writes=("mpiio_indep_writes", "sum"),
        mpiio_coll_reads=("mpiio_coll_reads", "sum"),
        mpiio_coll_writes=("mpiio_coll_writes", "sum"),
        stdio_bytes_read=("stdio_bytes_read", "sum"),
        stdio_bytes_written=("stdio_bytes_written", "sum"),
        slowest_rank_time=("slowest_rank_time", "max"),
        io_read_start=("io_read_start", "min"),
        io_write_start=("io_write_start", "min"),
        io_read_end=("io_read_end", "max"),
        io_write_end=("io_write_end", "max"),
        has_posix=("has_posix", "any"),
        has_mpiio=("has_mpiio", "any"),
        has_stdio=("has_stdio", "any"),
        fs_lustre_grand=("fs_lustre_grand", "any"),
        fs_lustre_eagle=("fs_lustre_eagle", "any"),
        cb_nodes=("cb_nodes", "max"),
        io_read_front_heavy=("io_read_front_heavy", "any"),
        io_write_back_heavy=("io_write_back_heavy", "any"),
        io_phase_start_frac=("io_phase_start_frac", "min"),
        io_phase_end_frac=("io_phase_end_frac", "max"),
        seq_read_ratio=("seq_read_ratio", "mean"),
        seq_write_ratio=("seq_write_ratio", "mean"),
        small_read_ratio=("small_read_ratio", "mean"),
        small_write_ratio=("small_write_ratio", "mean"),
        rank_imbalance=("rank_imbalance", "max"),
        darshan_file_count=("fname", "count"),
    ).reset_index()

    print(f"  Darshan aggregated: {len(agg):,} unique jobs", flush=True)
    return agg

def recompute_derived(df, t_wall_col="RUNTIME_SECONDS"):
    print("Recomputing derived metrics...", flush=True)

    # total bytes across all interfaces
    df["total_bytes"] = df["bytes_read"] + df["bytes_written"]

    # write dominance
    df["write_dominance"] = np.where(
        df["total_bytes"] > 0,
        df["bytes_written"] / df["total_bytes"],
        0.0
    )

    # T_IO from timestamps (PEARC paper equation 1 & 2)
    df["T_posix"] = (df["io_read_end"].combine(df["io_write_end"], max) -
                     df["io_read_start"].combine(df["io_write_start"], min)).clip(lower=0)
    df["T_IO"] = df["T_posix"]  # extend here if MPIIO/STDIO timestamps added later

    # BWio (PEARC paper equation 4)
    t_wall = df[t_wall_col]
    valid_tio = (df["T_IO"] > 1) & (df["T_IO"] <= 1.05 * t_wall)
    df["BWio"] = np.where(
        df["total_bytes"] > 0,
        np.where(valid_tio, df["total_bytes"] / df["T_IO"], df["total_bytes"] / t_wall),
        np.nan
    )
    df["BWio_MB"] = df["BWio"] / 1e6  # bytes/s → MB/s

    # io_intensity using slowest rank time
    df["io_intensity"] = np.where(
        df["slowest_rank_time"] > 0,
        df["total_bytes"] / df["slowest_rank_time"],
        0.0
    )

    # mpiio collective ratio
    total_mpiio = df["mpiio_coll_reads"] + df["mpiio_coll_writes"] + df["mpiio_indep_reads"] + df["mpiio_indep_writes"]
    df["mpiio_coll_ratio"] = np.where(
        total_mpiio > 0,
        (df["mpiio_coll_reads"] + df["mpiio_coll_writes"]) / total_mpiio,
        0.0
    )

    # rank imbalance flag
    df["high_rank_imbalance"] = df["rank_imbalance"] > 10

    print(f"  Derived metrics computed", flush=True)
    return df

def classify_crosslayer(df):
    
    # --- GPU efficiency axis ---
    # threshold: jobs using less than 10% GPU on average are underutilizing
    GPU_UTIL_LOW = 10.0      # p25 of gpu_util_mean across all jobs
    GPU_UTIL_HIGH = 70.0     # p75 of gpu_util_mean
    
    # --- I/O efficiency axis ---
    # BWio normalized against Polaris Lustre peak
    POLARIS_OSS_PEAK_MBS = 100000.0  # Polaris has much higher peak than Theta
    
    # --- Scale axis ---
    # jobs that requested many GPUs but used them poorly are highest priority
    # GPU_SCALE_HIGH = df["GPUS_REQUESTED"].quantile(0.75)
    GPU_SCALE_HIGH = (df["NODES_USED"] * 4).quantile(0.75)
    
    # compute percentile thresholds from actual data
    gpu_p25 = df["gpu_util_mean"].quantile(0.25)
    gpu_p75 = df["gpu_util_mean"].quantile(0.75)
    bwio_p50 = df["BWio_MB"][df["BWio_MB"].notna() & (df["BWio_MB"] > 0)].quantile(0.50)
    bwio_p90 = df["BWio_MB"][df["BWio_MB"].notna() & (df["BWio_MB"] > 0)].quantile(0.90)
    
    print(f"  GPU util P25: {gpu_p25:.1f}% | P75: {gpu_p75:.1f}%")
    print(f"  BWio P50: {bwio_p50:.1f} MB/s | P90: {bwio_p90:.1f} MB/s")
    print(f"  GPU scale P75: {GPU_SCALE_HIGH:.0f} GPUs")
    
    # --- Cross-layer waste score ---
    # Each dimension contributes 0-1 waste score
    
    # GPU waste: 1 if totally idle, 0 if fully utilized
    df["gpu_waste_score"] = np.where(
        df["gpu_util_mean"].notna(),
        1.0 - (df["gpu_util_mean"] / 100.0).clip(0, 1),
        np.nan
    )
    
    # I/O waste: 1 if zero I/O, scales down as BWio improves
    df["io_waste_score"] = np.where(
        df["BWio_MB"].notna() & (df["BWio_MB"] > 0),
        1.0 - (df["BWio_MB"] / bwio_p90).clip(0, 1),
        np.where(df["total_bytes"] > 0, 1.0, 0.0)
    )
    
    # Scale multiplier: larger jobs waste more resources when inefficient
    # df["scale_factor"] = (df["GPUS_REQUESTED"] / GPU_SCALE_HIGH).clip(0, 3)
    df["scale_factor"] = (df["NODES_USED"] * 4 / GPU_SCALE_HIGH).clip(0, 3)
    # composite waste score weighted by scale
    df["cross_layer_waste"] = (
        0.5 * df["gpu_waste_score"].fillna(0.5) +
        0.5 * df["io_waste_score"].fillna(0.5)
    ) * df["scale_factor"].fillna(1.0)
    
    # --- Cross-layer classification ---
    def crosslayer_tier(row):
        gpu = row["gpu_util_mean"]
        bwio = row["BWio_MB"]
        gpus = row["NODES_USED"] * 4
        total = row.get("total_bytes", 0)
        has_io = (not pd.isna(total)) and total > 0
        
        # Ghost job — allocated GPUs, did nothing
        if (not pd.isna(gpu)) and gpu < 5.0 and gpus >= 4 and not has_io:
            return "Ghost"
        
        # IO-bottlenecked — GPU idle but moving data (compute waiting on I/O)
        if (not pd.isna(gpu)) and gpu < GPU_UTIL_LOW and has_io:
            return "IO_Bottlenecked"
        
        # GPU-bottlenecked — high GPU util, no I/O (pure compute, acceptable)
        if (not pd.isna(gpu)) and gpu >= GPU_UTIL_HIGH and not has_io:
            return "Compute_Bound"
        
        # Balanced — both GPU and I/O active
        if (not pd.isna(gpu)) and gpu >= GPU_UTIL_LOW and has_io:
            return "Balanced"
        
        # Scale waster — big job, poor GPU, poor I/O
        if (not pd.isna(gpu)) and gpu < GPU_UTIL_LOW and gpus >= GPU_SCALE_HIGH and not has_io:
            return "Scale_Waster"
        
        return "Unclassified"
    
    df["crosslayer_tier"] = df.apply(crosslayer_tier, axis=1)
    
    # # --- GPU-aware volume tier ---
    # # Weight total bytes by GPU hours consumed to get resource-normalized I/O
    # # df["gpu_hours"] = df["GPUS_REQUESTED"].clip(lower=0) * df["RUNTIME_SECONDS"] / 3600
    # df["gpu_hours"] = df["NODES_USED"] * 4 * df["RUNTIME_SECONDS"] / 3600
    # df.loc[(df["GPUS_REQUESTED"] == -1) & (df["gpu_util_mean"].isna()), "gpu_hours"] = 0.0 
    # df["bytes_per_gpu_hour"] = np.where(
    #     df["gpu_hours"] > 0,
    #     df["total_bytes"] / df["gpu_hours"],
    #     0.0
    # )
    
    # print(f"\n  Cross-layer tiers:")
    # print(f"  {df['crosslayer_tier'].value_counts().to_dict()}")
    
    # return df
# --- GPU-aware volume tier ---
    df["gpu_hours"] = df["NODES_USED"] * 4 * df["RUNTIME_SECONDS"] / 3600
    df.loc[df["NODES_USED"] == 0, "gpu_hours"] = 0.0
    df["bytes_per_gpu_hour"] = np.where(
        df["gpu_hours"] > 0,
        df["total_bytes"] / df["gpu_hours"],
        0.0
    )

    print(f"\n  Cross-layer tiers:")
    print(f"  {df['crosslayer_tier'].value_counts().to_dict()}")

    return df


def classify_unclassified(df):
    print("Classifying remaining unclassified jobs...", flush=True)
    SHORT_JOB_THRESHOLD = 60.0
    GPU_UTIL_LOW = 10.0
    GPU_UTIL_HIGH = 70.0
    def refine_tier(row):
        if row["crosslayer_tier"] != "Unclassified":
            return row["crosslayer_tier"]

        runtime = row["RUNTIME_SECONDS"]
        gpus = row["NODES_USED"] * 4
        has_darshan = not pd.isna(row["darshan_file_count"])
        total = row.get("total_bytes", 0)
        has_io = (not pd.isna(total)) and total > 0
        exit_code = row.get("EXIT_STATUS", 0)
        gpu = row["gpu_util_mean"]

        if runtime < SHORT_JOB_THRESHOLD:
            return "Short_Job"
        if (not pd.isna(exit_code)) and exit_code != 0:
            return "Failed_Job"
        # if gpus == -1 or gpus == 0:
        #     return "CPU_IO_Job" if has_io else "CPU_No_IO"
        if row["GPUS_REQUESTED"] == -1 or gpus == 0:
            return "CPU_IO_Job" if has_io else "CPU_No_IO"
        # if has_darshan and has_io and gpus == 0:
        #     return "CPU_IO_Job"
        if not has_darshan and gpus <= 4 and runtime < 600:
            return "Interactive_Test"
        if has_darshan and not has_io:
            return "No_IO_No_GPU"
        if gpus > 4 and not has_darshan:
            return "Telemetry_Gap"
        if (not pd.isna(gpu)) and 5.0 <= gpu < GPU_UTIL_LOW and not has_io:
            return "Low_GPU_No_IO"
        if (not pd.isna(gpu)) and 5.0 <= gpu < GPU_UTIL_LOW and has_io:
            return "Low_GPU_With_IO"
        if (not pd.isna(gpu)) and GPU_UTIL_LOW <= gpu < GPU_UTIL_HIGH and not has_io:
            return "Moderate_Compute"
        if (not pd.isna(gpu)) and GPU_UTIL_LOW <= gpu < GPU_UTIL_HIGH and has_io:
            return "Balanced"
        if pd.isna(gpu) and not has_darshan:
            return "Telemetry_Gap"
        if pd.isna(gpu) and has_io:
            return "CPU_IO_Job"
        return "Unclassified"

    df["crosslayer_tier"] = df.apply(refine_tier, axis=1)
    print(f"\n  Refined cross-layer tiers:")
    print(f"  {df['crosslayer_tier'].value_counts().to_dict()}")
    return df


if __name__ == "__main__":
    cfg = load_config(args.config)

    # load all three sources
    print("Loading DJC...", flush=True)
    djc = pd.read_csv(cfg["djc_csv"], low_memory=False)
    djc["job_id"] = djc["JOB_NAME"].str.split(".").str[0]
    print(f"  DJC: {len(djc):,} jobs")

    print("Loading GPU metrics...", flush=True)
    gpu = pd.read_csv(cfg["gpu_parsed_out"])
    gpu["job_id"] = gpu["JOB_NAME"].str.split(".").str[0]
    print(f"  GPU: {len(gpu):,} jobs")

    print("Loading Darshan metrics...", flush=True)
    darshan_raw = pd.read_csv(cfg["darshan_parsed_out"])
    darshan_raw["job_id"] = darshan_raw["job_id"].astype(str)
    darshan = aggregate_darshan(darshan_raw)

    # left join DJC + GPU
    print("Joining DJC + GPU...", flush=True)
    combined = djc.merge(
        gpu.drop(columns=["JOB_NAME"], errors="ignore"),
        on="job_id", how="left"
    )
    print(f"  After DJC+GPU join: {len(combined):,} rows")

    # left join darshan
    print("Joining + Darshan...", flush=True)
    combined = combined.merge(darshan, on="job_id", how="left")
    print(f"  After +Darshan join: {len(combined):,} rows")
    print(f"  Jobs with darshan coverage: {combined['darshan_file_count'].notna().sum():,}")

    # recompute derived and classify
    combined = recompute_derived(combined, t_wall_col="RUNTIME_SECONDS")
    combined = classify_crosslayer(combined)
    combined = classify_unclassified(combined)

    out = cfg["combined_out"]
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out, index=False)
    print(f"Done — {len(combined):,} rows → {out}")