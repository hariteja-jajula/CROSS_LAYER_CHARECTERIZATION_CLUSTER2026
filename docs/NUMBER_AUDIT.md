# Camera-ready number audit

Every headline number in the paper was re-derived from the code and the feature
matrix for camera-ready. The authoritative run is
[`run_artifacts/probe_52665894.out`](../run_artifacts/probe_52665894.out)
(production run, 2025-05-07); `pipeline/stage07_paper_stats.py` and
`pipeline/stage10_predictive.py` regenerate these numbers from
`combined_metrics_final*.csv`.

This document exists so that a reader who knows Polaris well can see that the
"eyebrow-raising" numbers were checked, not hand-waved. **Verdict: every
headline number reproduces exactly.** A short list of framing footnotes is
recommended (below); no headline number needs to change.

## Corpus & coverage — reproduce exactly

| Quantity | Paper | Recomputed |
|---|---|---|
| Jobs | 262,634 | 262,634 |
| Allocated GPU-hours | 16,530,438 | 16,530,438 |
| Span | 13 months | queued-ts 2024-11-19 → 2025-12-31 (407 days) |
| Users / projects | 1,008 / 313 | 1,008 / 313 |
| DCGM / Darshan / joint coverage | 78.1% / 25.2% / 18.4% | matches |

**"13 months" is correct.** The corpus is dated by `QUEUED_TIMESTAMP`
(2024-11-19 → 2025-12-31), not by start time. A 407-day span is 13 months. No
change.

## Taxonomy (Table 1) — reproduces exactly after the documented split

All 17 tiers reproduce. The only subtlety is the stored-label → paper-label
split/rename (`Ghost` → Idle + Idle_Hidden_Activity; `Scale_Waster` →
Scale_Inefficient), which `stage07_paper_stats.py` performs in-code. See
[`REPRODUCIBILITY.md`](REPRODUCIBILITY.md).

- Under-utilized GPU-hours subtotal: **60.2%** (Table 1 row-sum 60.25%).
- **Cosmetic only:** Table 1 prints the subtotal as `60.25` while the abstract/
  body say `60.2`. Optionally round the table to `60.2` for consistency. No
  numeric impact.

## Eyebrow-raising numbers — checked, and why each holds

### 1. Failed_Job = 42.1% of allocated GPU-hours (6,963,090 h over 53,392 jobs)

Reproduces exactly. This looks alarming ("42% of the machine's GPU-hours
failed") but the code counts a job as `Failed_Job` by **scheduler exit status**,
not by crash. Inspection of the tier:

- ~85% of these GPU-hours are **walltime-exhausted** jobs (hit the requested
  wall limit), not early crashes.
- Exit code **−29** alone accounts for **78.5%** of the tier's GPU-hours —
  i.e. the dominant "failure" is the scheduler killing jobs at their wall limit,
  which is expected behavior for long-running / checkpoint-restart workloads.

**Recommended footnote:** clarify that `Failed_Job` is a non-zero-exit
scheduler bucket, dominated by walltime exhaustion (exit −29), not application
crashes.

### 2. 60.2% of GPU-hours under-utilized

Reproduces (sum of the wasteful tiers). This is a deliberately broad definition
(any tier flagged `is_wasteful`), stated as such in the paper. No change; the
per-tier breakdown in Table 1 lets readers apply a stricter bar if they wish.

### 3. ρ(GPU_util, io_time_frac) = −0.411, N = 14,390

Reproduces exactly on the BWio-valid subset (jobs with both a valid GPU-util
signal and a valid Darshan I/O-time fraction). **This is the one genuine framing
risk:** on the full joint set (N = 48,332) the same correlation is **−0.121**,
and the paper also reports **−0.272** elsewhere for a different conditioning.

**Recommended footnote:** state explicitly that −0.411 is measured on the
BWio-valid subset (N = 14,390 jobs where both signals are well-defined); the
correlation is weaker but same-signed on the broader joint set (−0.121,
N = 48,332). This pre-empts the "it's only −0.12 if you don't cherry-pick"
objection by owning the conditioning up front. No number changes — just say
which N each ρ belongs to.

### 4. 100% of MPI-IO jobs at cb_nodes=4 (12,605 jobs)

Reproduces: **12,605** jobs set `cb_nodes=4`, and that is **100%** of the jobs
that expose a collective-buffering node count. This is the ROMIO default on
Polaris, so 100% is believable and correct.

**Do NOT change 12,605 → 12,067.** (An earlier note floated 12,067; that is a
different filter and would make the "100%" claim false.) 12,605 @ 100% is the
literally-true pair. Keep as-is.

### 5. Top-10 users = 50.7% of GPU-hours; top-51 = 83.7%

Both reproduce exactly. High concentration is normal for a leadership-class
facility. No change.

### 6. Scale_Inefficient: 995 jobs, 985 wasteful, 99.0%

Reproduces (985 / 995 = 99.0%). No change.

### 7. REX-IO burst tier = 6,628 jobs; 86.6% / 89.4% burst fractions

Reproduce exactly. No change.

### 8. Prediction: M5 AUC 0.894, CI [0.8898, 0.8988]

Reproduces from `stage10_predictive.py`. Strictly-causal features
(`end_ts < queued_ts`, 7-day / ≤10-job lookback) — no leakage. The AUC is high
but the task (predict under-utilization from prior same-user/project behavior)
is genuinely predictable, and the cold-start and permutation-importance results
in the same run support it. No change.

- **Optional:** one macro-F1 prints as **0.763** in text vs **0.760**
  recomputed for M3 — a rounding discrepancy at the third decimal. Optionally
  align to 0.760.

## Facility-constant credibility fixes (already applied in this repo)

These are code-comment/label changes only — **zero numeric impact** on any
result — that pre-empt "that constant is wrong for Polaris" objections:

- `pipeline/stage00_validate.py`: the GPU-memory sanity ceiling was 85 GB (A100 80 GB
  assumption). Polaris A100s are the **40 GB SXM4** model; changed the check to
  45 GB with a comment. This only affects a pre-flight sanity assertion, not any
  reported statistic.
- `utils/combined.py` and `utils/combined_v2.py`: `BWio_PEAK_MB = 100_000` is a
  **per-job** I/O-rate sanity ceiling, not the facility aggregate bandwidth.
  Added a comment saying so (Polaris Grand/Eagle Lustre aggregate is ~650 GB/s);
  the constant is unchanged and is only used to cap per-job outliers.

## Summary for camera-ready

- **No headline number changes.**
- **Recommended footnotes (3):** Failed_Job = walltime-exhaustion bucket (§1);
  ρ conditioning / N (§3); (optional) keep both ρ values with their N labeled.
- **Optional cosmetic edits (2):** Table 1 subtotal `60.25` → `60.2`; M3
  macro-F1 `0.763` → `0.760`.
- **Explicitly do not:** change 12,605 → 12,067 (§4).
