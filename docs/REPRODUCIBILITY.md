# Reproducibility notes

## Environment

- Python 3.13 (production runs recorded 3.13.11 in
  `data/derived/reproducibility_manifest.json`; 3.11+ is fine).
- `pip install pandas pyarrow numpy scikit-learn pydarshan`
- `darshan-parser` (from the Darshan tools) on `PATH` for stages 05/08, which
  shell out to it. Stages 02/07/10 use PyDarshan / the parsed CSV and do not.
- Production runs executed on a single CPU node (no GPU needed — GPU telemetry
  is consumed as CSV/Parquet, not read from live devices).

## Run order

`config/config.json` drives every stage. Recommended order (see README for the
one-liners):

```
stage00_validate → stage00_convert_gpu_to_parquet → stage01_parse_gpu → stage02_parse_darshan
→ stage03_build_combined → stage00_consistency → stage07_paper_stats → stage10_predictive
```

`stage02_parse_darshan` is by far the longest stage (walks the full daily-tarball
tree). Everything downstream reads the combined feature matrix, so once
`combined_metrics.csv` exists, `stage07_paper_stats` and `stage10_predictive` rerun in
minutes.

## Known hardcoded paths (override before running)

A few stage scripts still contain absolute paths from the original run
environment. `config/config.json` covers the data I/O, but these constants are
in-source and must be edited by hand if you rerun those specific stages:

| File | What is hardcoded |
|---|---|
| `pipeline/stage07_paper_stats.py` | `ROOT = Path("/pscratch/sd/h/hjajula/Darshan/cross_layer_hpc_tool")` near the top |
| `pipeline/stage00_consistency.py` | input/output paths under `/pscratch/...` |
| `pipeline/stage08_user_forensics.py` | output path under `/pscratch/...` |
| `pipeline/paper_results.py` | input/output paths under `/pscratch/...` |

These are intentionally left as-is so the shipped code matches exactly what
produced `run_artifacts/probe_52665894.out`; point them at your own paths to
rerun.

## Taxonomy note: stored labels vs. paper Table 1

The stored `crosslayer_tier` column uses two **legacy** labels that the paper
presents under refined names:

- **`Ghost`** (31,947 jobs) is split by `power_mean` into
  **Idle** (`power_mean < 50 W`, 9,836) and
  **Idle_Hidden_Activity** (`power_mean ≥ 50 W`, 22,111).
- **`Scale_Waster`** (995 jobs) is renamed **`Scale_Inefficient`**.

`is_wasteful` = 212,179 before the split, 190,068 after. Two CSV variants make
this explicit:

- `combined_metrics_final.csv` — pre-relabel (stored labels).
- `combined_metrics_final_paperstats.csv` — post-relabel (paper Table 1 labels
  baked in).

`stage07_paper_stats.py` performs the split/rename in-code, so it reproduces Table 1
from either variant. If you compute tier counts directly off the raw column,
remember to apply the split or you will see `Ghost`/`Scale_Waster` instead of
the paper's tiers.

## Determinism

The taxonomy is a deterministic rule cascade (Algorithm 1) — no randomness. The
predictive models (M0–M5) fix `random_state`; Random Forest uses
`n_estimators=200`, `max_depth=15`. The submission-time evaluation is strictly
causal: for each job, only jobs with `end_ts < queued_ts` (7-day lookback, ≤10
most-recent) are used as features, so there is no train/test leakage across the
temporal split.
