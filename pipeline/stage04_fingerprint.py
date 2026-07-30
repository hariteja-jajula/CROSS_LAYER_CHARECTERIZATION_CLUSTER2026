import pandas as pd
import numpy as np
import json, argparse
from pathlib import Path

GPU_UTIL_LOW = 10.0
GPU_UTIL_HIGH = 70.0

parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True)
args = parser.parse_args()

def load_config(path):
    with open(path) as f:
        return json.load(f)

def compute_user_fingerprint(df):
    print("Computing user-level cross-layer fingerprints...", flush=True)

    # only jobs with meaningful runtime
    active = df[df["RUNTIME_SECONDS"] >= 60].copy()

    agg = active.groupby("USERNAME_GENID").agg(
        total_jobs=("job_id", "count"),
        # GPU dimension
        gpu_util_mean=("gpu_util_mean", "mean"),
        gpu_util_std=("gpu_util_mean", "std"),
        gpu_waste_mean=("gpu_waste_score", "mean"),
        gpu_zero_frac=("gpu_zero_util_frac", "mean"),
        # I/O dimension
        bwio_mean=("BWio_MB", "mean"),
        bwio_std=("BWio_MB", "std"),
        io_waste_mean=("io_waste_score", "mean"),
        total_bytes_mean=("total_bytes", "mean"),
        total_bytes_std=("total_bytes", "std"),
        write_dominance_mean=("write_dominance", "mean"),
        # scale dimension
        gpu_hours_total=("gpu_hours", "sum"),
        gpus_mean=("GPUS_REQUESTED", "mean"),
        gpus_max=("GPUS_REQUESTED", "max"),
        # waste dimension
        cross_layer_waste_mean=("cross_layer_waste", "mean"),
        # tier counts
        n_ghost=("crosslayer_tier", lambda x: (x == "Ghost").sum()),
        n_io_bottlenecked=("crosslayer_tier", lambda x: (x == "IO_Bottlenecked").sum()),
        n_compute_bound=("crosslayer_tier", lambda x: (x == "Compute_Bound").sum()),
        n_balanced=("crosslayer_tier", lambda x: (x == "Balanced").sum()),
        n_scale_waster=("crosslayer_tier", lambda x: (x == "Scale_Waster").sum()),
        n_failed=("crosslayer_tier", lambda x: (x == "Failed_Job").sum()),
        n_moderate_compute=("crosslayer_tier", lambda x: (x == "Moderate_Compute").sum()),
    ).reset_index()

    # dominant tier per user
    tier_counts = active.groupby(["USERNAME_GENID", "crosslayer_tier"]).size().reset_index(name="cnt")
    dominant = tier_counts.sort_values("cnt", ascending=False).groupby("USERNAME_GENID").first().reset_index()
    dominant = dominant.rename(columns={"crosslayer_tier": "dominant_tier"})
    agg = agg.merge(dominant[["USERNAME_GENID", "dominant_tier"]], on="USERNAME_GENID", how="left")

    # repeatability score — CV of total_bytes per user (low CV = highly repeatable)
    agg["bytes_cv"] = agg["total_bytes_std"] / agg["total_bytes_mean"].replace(0, np.nan)
    agg["bytes_cv"] = agg["bytes_cv"].fillna(0)

    # ghost fraction — what fraction of their jobs are pure waste
    agg["ghost_frac"] = agg["n_ghost"] / agg["total_jobs"]
    agg["io_bottleneck_frac"] = agg["n_io_bottlenecked"] / agg["total_jobs"]

    # gpu hours wasted — Ghost + Scale_Waster jobs
    ghost_hours = active[active["crosslayer_tier"].isin(["Ghost", "Scale_Waster"])].groupby("USERNAME_GENID")["gpu_hours"].sum().reset_index()
    ghost_hours = ghost_hours.rename(columns={"gpu_hours": "wasted_gpu_hours"})
    agg = agg.merge(ghost_hours, on="USERNAME_GENID", how="left")
    agg["wasted_gpu_hours"] = agg["wasted_gpu_hours"].fillna(0)

    # equivalent A100 cost at $3/GPU-hour
    agg["wasted_cost_usd"] = agg["wasted_gpu_hours"] * 3.0

    print(f"  User fingerprints: {len(agg):,} unique users", flush=True)
    return agg

def compute_phase_correlation(df):
    print("Computing GPU-IO phase correlation for IO_Bottlenecked jobs...", flush=True)

    io_bot = df[
        (df["crosslayer_tier"] == "IO_Bottlenecked") &
        df["gpu_util_phase1"].notna() &
        df["io_phase_start_frac"].notna()
    ].copy()

    print(f"  IO_Bottlenecked jobs with phase data: {len(io_bot):,}")

    if len(io_bot) == 0:
        print("  No phase data available", flush=True)
        return None

    # early I/O jobs — I/O starts in first 20% of runtime
    early_io = io_bot[io_bot["io_phase_start_frac"] < 0.2]
    late_io = io_bot[io_bot["io_phase_start_frac"] >= 0.2]

    print(f"  Early I/O (start < 20% runtime): {len(early_io):,}")
    print(f"    GPU phase1 mean: {early_io['gpu_util_phase1'].mean():.1f}%")
    print(f"    GPU phase2 mean: {early_io['gpu_util_phase2'].mean():.1f}%")
    print(f"    GPU phase3 mean: {early_io['gpu_util_phase3'].mean():.1f}%")

    print(f"  Late I/O (start >= 20% runtime): {len(late_io):,}")
    print(f"    GPU phase1 mean: {late_io['gpu_util_phase1'].mean():.1f}%")
    print(f"    GPU phase2 mean: {late_io['gpu_util_phase2'].mean():.1f}%")
    print(f"    GPU phase3 mean: {late_io['gpu_util_phase3'].mean():.1f}%")

    # correlation between io_phase_start_frac and gpu_util_phase1
    corr = io_bot[["io_phase_start_frac", "gpu_util_phase1", "gpu_util_phase2", "gpu_util_phase3"]].corr()
    print(f"\n  Correlation matrix (io_phase_start_frac vs GPU phases):")
    print(corr["io_phase_start_frac"].to_string())

    return io_bot

def compute_temporal_concentration(df):
    print("Computing temporal concentration by month...", flush=True)

    # extract month from start timestamp
    df["month"] = pd.to_datetime(df["START_TIMESTAMP"], errors="coerce").dt.to_period("M")

    waste_tiers = ["Ghost", "Scale_Waster", "IO_Bottlenecked"]
    waste_jobs = df[df["crosslayer_tier"].isin(waste_tiers)].copy()

    # top 10 users by total wasted GPU hours
    top_wasters = (
        waste_jobs.groupby("USERNAME_GENID")["gpu_hours"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .index
    )

    print(f"\n  Top 10 GPU hour wasters across waste tiers:")
    monthly = waste_jobs[waste_jobs["USERNAME_GENID"].isin(top_wasters)].groupby(
        ["USERNAME_GENID", "month", "crosslayer_tier"]
    ).agg(
        jobs=("job_id", "count"),
        gpu_hours=("gpu_hours", "sum")
    ).reset_index()

    print(monthly.to_string())
    return monthly

def compute_procurement_impact(df):
    print("\nComputing procurement impact by tier...", flush=True)

    tier_summary = df.groupby("crosslayer_tier").agg(
        jobs=("job_id", "count"),
        gpu_hours_total=("gpu_hours", "sum"),
        total_bytes_TB=("total_bytes", lambda x: x.sum() / 1e12),
        mean_runtime=("RUNTIME_SECONDS", "mean"),
        mean_gpus=("GPUS_REQUESTED", "mean"),
    ).reset_index()

    total_gpu_hours = tier_summary["gpu_hours_total"].sum()
    tier_summary["gpu_hours_pct"] = tier_summary["gpu_hours_total"] / total_gpu_hours * 100
    tier_summary["wasted_cost_usd"] = tier_summary["gpu_hours_total"] * 3.0

    tier_summary = tier_summary.sort_values("gpu_hours_total", ascending=False)
    print(tier_summary.to_string(index=False))

    # total waste from Ghost + Scale_Waster
    waste = tier_summary[tier_summary["crosslayer_tier"].isin(["Ghost", "Scale_Waster", "IO_Bottlenecked"])]
    total_waste_hours = waste["gpu_hours_total"].sum()
    print(f"\n  Total wasted GPU hours (Ghost+Scale_Waster+IO_Bottlenecked): {total_waste_hours:,.0f}")
    print(f"  Equivalent cost at $3/GPU-hour: ${total_waste_hours * 3:,.0f}")
    print(f"  As % of total GPU hours: {total_waste_hours / total_gpu_hours * 100:.1f}%")

    return tier_summary

def compute_phase_patterns(df):
    print("Computing phase pattern fingerprints by tier...", flush=True)

    tiers = ['Ghost', 'IO_Bottlenecked', 'Compute_Bound', 
             'Balanced', 'Scale_Waster', 'Moderate_Compute']
    
    phase_cols = ['gpu_util_phase1', 'gpu_util_phase2', 'gpu_util_phase3',
                  'phase_drop', 'io_phase_start_frac', 'io_phase_end_frac',
                  'io_read_front_heavy', 'io_write_back_heavy',
                  'gpu_util_first_half', 'gpu_util_second_half']

    rows = []
    for tier in tiers:
        sub = df[df['crosslayer_tier'] == tier][phase_cols].dropna()
        if len(sub) == 0:
            continue
        row = {'tier': tier, 'n': len(sub)}
        for col in phase_cols:
            row[f'{col}_mean'] = round(sub[col].mean(), 3)
            row[f'{col}_median'] = round(sub[col].median(), 3)
        
        # classify the GPU trajectory
        p1 = row['gpu_util_phase1_mean']
        p3 = row['gpu_util_phase3_mean']
        if p1 < 1.0 and p3 < 1.0:
            row['gpu_trajectory'] = 'flat_idle'
        elif p1 > 70 and p3 > 70:
            row['gpu_trajectory'] = 'sustained_high'
        elif p3 > p1 * 1.5:
            row['gpu_trajectory'] = 'ramp_up'
        elif p1 > p3 * 1.5:
            row['gpu_trajectory'] = 'ramp_down'
        else:
            row['gpu_trajectory'] = 'moderate_stable'

        # classify I/O temporal pattern
        start = row['io_phase_start_frac_mean']
        end = row['io_phase_end_frac_mean']
        if start < 0.05 and end > 0.9:
            row['io_pattern'] = 'full_job_io'
        elif start < 0.1 and end < 0.5:
            row['io_pattern'] = 'front_loaded'
        elif start > 0.5:
            row['io_pattern'] = 'back_loaded'
        elif end < 0.1:
            row['io_pattern'] = 'no_io'
        else:
            row['io_pattern'] = 'mid_job'

        rows.append(row)

    phase_df = pd.DataFrame(rows)
    print(phase_df[['tier', 'n', 'gpu_util_phase1_mean', 'gpu_util_phase2_mean', 
                     'gpu_util_phase3_mean', 'io_phase_start_frac_mean',
                     'io_phase_end_frac_mean', 'gpu_trajectory', 'io_pattern']].to_string(index=False))
    return phase_df

if __name__ == "__main__":
    cfg = load_config(args.config)

    print("Loading combined metrics...", flush=True)
    df = pd.read_csv(cfg["combined_out"], low_memory=False)
    df["job_id"] = df["JOB_NAME"].str.split(".").str[0]
    print(f"  {len(df):,} jobs loaded")

    # 1. user fingerprints
    user_fp = compute_user_fingerprint(df)
    user_fp_out = cfg["combined_out"].replace("combined_metrics.csv", "user_fingerprints.csv")
    user_fp.to_csv(user_fp_out, index=False)
    print(f"  User fingerprints → {user_fp_out}")

    # 2. GPU-IO phase correlation
    phase_data = compute_phase_correlation(df)

    # 3. temporal concentration
    monthly = compute_temporal_concentration(df)
    monthly_out = cfg["combined_out"].replace("combined_metrics.csv", "monthly_concentration.csv")
    monthly.to_csv(monthly_out, index=False)
    print(f"  Monthly concentration → {monthly_out}")

    # 4. procurement impact
    procurement = compute_procurement_impact(df)
    procurement_out = cfg["combined_out"].replace("combined_metrics.csv", "procurement_impact.csv")
    procurement.to_csv(procurement_out, index=False)
    print(f"  Procurement impact → {procurement_out}")

    # 5. Phase patterns (Cross-layer discriminator)
    phase_patterns_df = compute_phase_patterns(df)
    phase_patterns_out = cfg["combined_out"].replace("combined_metrics.csv", "phase_patterns.csv")
    phase_patterns_df.to_csv(phase_patterns_out, index=False)
    print(f"  Phase patterns → {phase_patterns_out}")

    print("\nDone.")