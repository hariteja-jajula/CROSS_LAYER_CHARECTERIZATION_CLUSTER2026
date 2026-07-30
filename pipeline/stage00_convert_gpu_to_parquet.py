#!/usr/bin/env python3
"""
One-time (per dataset) conversion of GPU telemetry .csv.gz to
date-partitioned Parquet. Safe to run on new yearly files — existing
date partitions are detected and skipped, so old data is never touched.
"""
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import json, argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True)
args = parser.parse_args()

with open(args.config) as f:
    cfg = json.load(f)

out_dir = Path(cfg["parquet_out_dir"])
out_dir.mkdir(parents=True, exist_ok=True)

# Find date partitions that already have parquet files written
existing_dates = {
    p.name.split("=")[1]
    for p in out_dir.glob("date=*")
    if any(p.glob("*.parquet"))
}
if existing_dates:
    print(f"Skipping {len(existing_dates)} already-converted date partitions.")

chunk_size = cfg["chunk_size"]
total_rows_written = 0
skipped_rows = 0

for i, chunk in enumerate(pd.read_csv(
        cfg["gpu_telemetry_gz"],
        chunksize=chunk_size,
        compression="gzip"), start=1):

    # Slice date from TIMESTAMP string — avoids slow parse_dates
    chunk["date"] = chunk["TIMESTAMP"].str[:10]

    # Drop rows belonging to already-converted dates
    new_rows = chunk[~chunk["date"].isin(existing_dates)]
    skipped_rows += len(chunk) - len(new_rows)

    if new_rows.empty:
        if i % 50 == 0:
            print(f"Chunk {i} | all dates already exist, skipping.", flush=True)
        continue

    table = pa.Table.from_pandas(new_rows, preserve_index=False)
    pq.write_to_dataset(
        table,
        root_path=str(out_dir),
        partition_cols=["date"],
        compression="zstd",
        existing_data_behavior="overwrite_or_ignore"
    )

    total_rows_written += len(new_rows)
    if i % 50 == 0:
        used_gb = sum(f.stat().st_size for f in out_dir.rglob("*.parquet")) / 1e9
        print(f"Chunk {i} | written {total_rows_written:,} rows | "
              f"skipped {skipped_rows:,} rows | parquet size: {used_gb:.2f} GB", flush=True)

used_gb = sum(f.stat().st_size for f in out_dir.rglob("*.parquet")) / 1e9
print(f"\nDone. Written {total_rows_written:,} rows | "
      f"Skipped {skipped_rows:,} already-converted rows | "
      f"Total parquet size: {used_gb:.2f} GB")
