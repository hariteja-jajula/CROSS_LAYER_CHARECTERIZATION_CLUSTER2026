"""
pipeline/stage02_parse_darshan.py
Reads raw bytes sequentially per tar, processes in parallel.
Eliminates parallel tar contention on Lustre.
"""

import pandas as pd
import os, json, argparse, tarfile, sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

parser = argparse.ArgumentParser()
parser.add_argument("--config",   required=True)
parser.add_argument("--workers",  type=int, default=16)
parser.add_argument("--dataroot", type=str, default=None)
args = parser.parse_args()

def load_config(path):
    with open(path) as f:
        return json.load(f)

def process_raw(task):
    fname, jobid, raw = task
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from utils.darshan_utils_pydarshan import parse_from_raw
    row = parse_from_raw(raw, fname)
    if row is not None:
        row['job_id'] = jobid
        row['fname']  = fname
    return row

if __name__ == "__main__":
    cfg       = load_config(args.config)
    n_workers = args.workers
    data_root = Path(args.dataroot) if args.dataroot else Path(cfg["darshan_data"])
    out       = cfg["darshan_parsed_out"]

    job_df = pd.read_csv(cfg["djc_csv"], low_memory=False)
    job_df["job_id"] = job_df["JOB_NAME"].str.split(".").str[0]
    job_lookup = set(job_df["job_id"].astype(str))
    print(f"Scheduler jobs loaded: {len(job_lookup):,}", flush=True)

    done_fnames = set()
    first_write = not os.path.exists(out)
    if os.path.exists(out):
        done_fnames = set(pd.read_csv(out, usecols=["fname"])["fname"].astype(str))
        print(f"Resuming — {len(done_fnames):,} already parsed", flush=True)

    results  = []
    count    = 0
    errors   = 0
    FLUSH_AT = 1000

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        for year in [2025]:
            for month in range(1, 13):
                for day in range(1, 32):
                    tar_path = data_root / str(year) / str(month) / str(day) / "logs.tar.gz"
                    if not tar_path.exists():
                        continue

                    day_tasks = []
                    try:
                        with tarfile.open(tar_path, "r:gz") as tar:
                            for fname in tar.getnames():
                                if not fname.endswith(".darshan"): continue
                                if fname in done_fnames: continue
                                jobid = fname.split("-")[0]
                                if jobid not in job_lookup: continue
                                f = tar.extractfile(tar.getmember(fname))
                                if f is None: continue
                                day_tasks.append((fname, jobid, f.read()))
                    except Exception as e:
                        print(f"Error reading {tar_path}: {e}", flush=True)
                        continue

                    if not day_tasks:
                        continue

                    print(f"  {year}/{month}/{day}: {len(day_tasks)} files", flush=True)

                    futures = {pool.submit(process_raw, t): t for t in day_tasks}
                    for future in as_completed(futures):
                        row = future.result()
                        if row is None:
                            errors += 1
                            continue
                        results.append(row)
                        count += 1
                        if count % FLUSH_AT == 0:
                            pd.DataFrame(results).to_csv(
                                out, mode='a', header=first_write, index=False
                            )
                            results     = []
                            first_write = False
                            print(f"{count:,} parsed  |  {errors:,} errors", flush=True)

    if results:
        pd.DataFrame(results).to_csv(
            out, mode='a', header=first_write, index=False
        )

    print(f"\nDone — {count:,} new rows written → {out}", flush=True)
    print(f"Errors / skipped: {errors:,}", flush=True)