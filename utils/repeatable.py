import pandas as pd
import numpy as np
import json, argparse, tarfile, subprocess, tempfile, os
from pathlib import Path
from utils.darshan_utils import parse_header

def compute_repeatability(df):
    print("Computing workflow repeatability scores...", flush=True)

    waste = df[df["crosslayer_tier"].isin([
        "Ghost", "Scale_Waster", "IO_Bottlenecked"
    ])].copy()

    waste["month"] = pd.to_datetime(
        waste["QUEUED_TIMESTAMP"], errors="coerce"
    ).dt.to_period("M")

    user_months = waste.groupby("USERNAME_GENID")["month"].nunique().reset_index()
    user_months.columns = ["USERNAME_GENID", "waste_months_active"]

    runtime_cv = waste.groupby("USERNAME_GENID")["RUNTIME_SECONDS"].agg(
        ["mean", "std"]
    ).reset_index()
    runtime_cv["runtime_cv"] = runtime_cv["std"] / runtime_cv["mean"].replace(0, np.nan)
    runtime_cv = runtime_cv[["USERNAME_GENID", "runtime_cv"]]

    gpus_cv = waste.groupby("USERNAME_GENID")["GPUS_REQUESTED"].agg(
        ["mean", "std"]
    ).reset_index()
    gpus_cv["gpus_cv"] = gpus_cv["std"] / gpus_cv["mean"].replace(0, np.nan)
    gpus_cv = gpus_cv[["USERNAME_GENID", "gpus_cv"]]

    bytes_cv = waste.groupby("USERNAME_GENID")["total_bytes"].agg(
        ["mean", "std"]
    ).reset_index()
    bytes_cv["bytes_cv"] = bytes_cv["std"] / bytes_cv["mean"].replace(0, np.nan)
    bytes_cv = bytes_cv[["USERNAME_GENID", "bytes_cv"]]

    rep = user_months.merge(runtime_cv, on="USERNAME_GENID", how="left")
    rep = rep.merge(gpus_cv, on="USERNAME_GENID", how="left")
    rep = rep.merge(bytes_cv, on="USERNAME_GENID", how="left")
    rep[["runtime_cv", "gpus_cv", "bytes_cv"]] = rep[
        ["runtime_cv", "gpus_cv", "bytes_cv"]
    ].fillna(0)

    rep["repeatability_score"] = (
        (1 / (1 + rep["runtime_cv"])) * 0.3 +
        (1 / (1 + rep["gpus_cv"])) * 0.2 +
        (1 / (1 + rep["bytes_cv"])) * 0.3 +
        (rep["waste_months_active"] / 12).clip(0, 1) * 0.2
    )

    rep["is_structural"] = (
        (rep["repeatability_score"] >= 0.7) &
        (rep["waste_months_active"] >= 3)
    )

    print(f"  Structural waste users: {rep['is_structural'].sum():,}")
    print(rep[rep["is_structural"]].sort_values(
        "repeatability_score", ascending=False
    )[["USERNAME_GENID", "waste_months_active", "repeatability_score",
       "runtime_cv", "bytes_cv"]].head(10).to_string(index=False))

    return rep

def extract_executable(tar, fname):
    """Extract just the header from a darshan file to get executable path."""
    f = tar.extractfile(tar.getmember(fname))
    if f is None:
        return None
    with tempfile.NamedTemporaryFile(suffix=".darshan", delete=False) as tmp:
        tmp.write(f.read())
        tmp_path = tmp.name
    try:
        result = subprocess.run(
            ["darshan-parser", tmp_path],
            capture_output=True, text=True, timeout=60
        )
        if result.returncode != 0:
            return None
        # extract exe from header lines
        for line in result.stdout.split('\n'):
            if line.startswith('#') and 'exe:' in line.lower():
                try:
                    return line.split(':', 1)[1].strip().split()[0]
                except:
                    pass
        return None
    except subprocess.TimeoutExpired:
        return None
    finally:
        os.unlink(tmp_path)

def confirm_executable_repeatability(structural_users, df, data_root, top_n=5):
    """For top structural offenders, pull darshan files and confirm same executable."""
    print("\nConfirming executable repeatability for top structural offenders...", flush=True)

    top_users = structural_users[structural_users["is_structural"]].sort_values(
        "repeatability_score", ascending=False
    ).head(top_n)["USERNAME_GENID"].tolist()

    waste_jobs = df[
        df["USERNAME_GENID"].isin(top_users) &
        df["crosslayer_tier"].isin(["Ghost", "Scale_Waster", "IO_Bottlenecked"])
    ].copy()

    # build fname → job_id lookup from darshan_metrics
    darshan_metrics = pd.read_csv(
        str(data_root).replace("Data", "cross_layer_hpc_tool/data/darshan_metrics.csv"),
        usecols=["job_id", "fname"]
    )
    darshan_metrics["job_id"] = darshan_metrics["job_id"].astype(str)
    waste_jobs["job_id"] = waste_jobs["job_id"].astype(str)

    target_fnames = darshan_metrics[
        darshan_metrics["job_id"].isin(waste_jobs["job_id"])
    ].copy()

    print(f"  Target darshan files to check: {len(target_fnames):,}")

    results = []
    checked = 0

    for year in [2025]:
        for month in range(1, 13):
            for day in range(1, 32):
                tar_path = data_root / str(year) / str(month) / str(day) / "logs.tar.gz"
                if not tar_path.exists():
                    continue
                with tarfile.open(tar_path, "r:gz") as tar:
                    tar_names = set(tar.getnames())
                    for _, row in target_fnames.iterrows():
                        if row["fname"] not in tar_names:
                            continue
                        exe = extract_executable(tar, row["fname"])
                        if exe:
                            results.append({
                                "job_id": row["job_id"],
                                "fname": row["fname"],
                                "executable": exe
                            })
                            checked += 1
                            if checked % 50 == 0:
                                print(f"  Checked {checked} files...", flush=True)

    if not results:
        print("  No executables found", flush=True)
        return None

    exe_df = pd.DataFrame(results)
    exe_df = exe_df.merge(
        waste_jobs[["job_id", "USERNAME_GENID", "crosslayer_tier",
                    "RUNTIME_SECONDS", "GPUS_REQUESTED", "gpu_util_mean"]],
        on="job_id", how="left"
    )

    print("\n  Executable consistency per user:")
    for user in top_users:
        user_exes = exe_df[exe_df["USERNAME_GENID"] == user]["executable"]
        if len(user_exes) == 0:
            continue
        unique_exes = user_exes.nunique()
        dominant_exe = user_exes.value_counts().index[0]
        dominant_frac = user_exes.value_counts().iloc[0] / len(user_exes)
        print(f"  User {user}:")
        print(f"    Jobs checked: {len(user_exes)}")
        print(f"    Unique executables: {unique_exes}")
        print(f"    Dominant exe: {dominant_exe}")
        print(f"    Dominant exe fraction: {dominant_frac:.1%}")

    return exe_df
def confirm_djc_repeatability(structural_users, df, top_n=10):
    print("\nConfirming repeatability via DJC metadata for Ghost/Scale_Waster users...", flush=True)

    top_users = structural_users[structural_users["is_structural"]].sort_values(
        "repeatability_score", ascending=False
    ).head(top_n)["USERNAME_GENID"].tolist()

    results = []
    for user in top_users:
        user_waste = df[
            (df["USERNAME_GENID"] == user) &
            (df["crosslayer_tier"].isin(["Ghost", "Scale_Waster", "IO_Bottlenecked"]))
        ]
        if len(user_waste) == 0:
            continue

        # compute interval between jobs
        times = pd.to_datetime(user_waste["QUEUED_TIMESTAMP"], errors="coerce").sort_values()
        intervals = times.diff().dt.total_seconds().dropna()

        results.append({
            "USERNAME_GENID": user,
            "total_waste_jobs": len(user_waste),
            "dominant_tier": user_waste["crosslayer_tier"].value_counts().index[0],
            "unique_walltimes": user_waste["WALLTIME_SECONDS"].nunique(),
            "unique_gpu_configs": user_waste["GPUS_REQUESTED"].nunique(),
            "unique_projects": user_waste["PROJECT_NAME_GENID"].nunique(),
            "median_interval_hours": intervals.median() / 3600 if len(intervals) > 0 else None,
            "interval_cv": intervals.std() / intervals.mean() if intervals.mean() > 0 else None,
            "months_active": pd.to_datetime(user_waste["QUEUED_TIMESTAMP"], errors="coerce").dt.to_period("M").nunique(),
            "walltime_cv": user_waste["WALLTIME_SECONDS"].std() / user_waste["WALLTIME_SECONDS"].mean() if user_waste["WALLTIME_SECONDS"].mean() > 0 else 0,
        })

    result_df = pd.DataFrame(results)

    # flag automated workflows — low interval CV + short median interval
    result_df["is_automated"] = (
        (result_df["interval_cv"] < 0.3) &
        (result_df["median_interval_hours"] < 2.0)
    )

    print(result_df[[
        "USERNAME_GENID", "total_waste_jobs", "dominant_tier",
        "median_interval_hours", "interval_cv", "months_active",
        "unique_walltimes", "is_automated"
    ]].to_string(index=False))

    return result_df
