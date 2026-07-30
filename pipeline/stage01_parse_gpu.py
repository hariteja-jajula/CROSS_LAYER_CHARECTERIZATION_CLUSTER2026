import os
import argparse
import pandas as pd
from multiprocessing import Pool
from utils.config_utils import load_config
from utils.gpu_utils import process_job

parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True)
args = parser.parse_args()
avail_cores = min(128, len(os.sched_getaffinity(0)))
print(f"Spinning up {avail_cores} workers...", flush=True)

if __name__ == "__main__":
    cfg        = load_config(args.config)
    job_df     = pd.read_csv(cfg["djc_csv"], low_memory=False)
    out_job    = cfg["djc_csv_filtered"]
    out_device = cfg["djc_csv_per_device"]

    # --- resume logic: skip already-processed jobs (checks job-level output) ---
    if os.path.exists(out_job):
        done   = set(pd.read_csv(out_job, usecols=["JOB_NAME"])["JOB_NAME"])
        job_df = job_df[~job_df["JOB_NAME"].isin(done)].reset_index(drop=True)
        print(f"Resuming — {len(done):,} already done, {len(job_df):,} remaining", flush=True)
    else:
        print(f"Starting fresh — {len(job_df):,} jobs", flush=True)

    with Pool(processes=32) as pool:
        job_rows, device_rows = [], []
        for i, (job_metric, per_device) in enumerate(
            pool.imap_unordered(process_job, [(row, cfg) for _, row in job_df.iterrows()]), 1
        ):
            job_rows.append(job_metric)
            device_rows.extend(per_device)  # 4 dicts per job

            if i % 1000 == 0 or i == len(job_df):
                pd.DataFrame(job_rows).to_csv(
                    out_job, mode='a', header=not os.path.exists(out_job), index=False
                )
                pd.DataFrame(device_rows).to_csv(
                    out_device, mode='a', header=not os.path.exists(out_device), index=False
                )
                job_rows, device_rows = [], []
                print(f"{i:,}/{len(job_df):,} done", flush=True)

    print(f"Done → {out_job}")
    print(f"Done → {out_device}")