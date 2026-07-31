# Legacy monolithic scripts

This directory preserves the original monolithic analysis scripts from the production workspace.

- `Framework.py` was the early single-file analysis driver used during exploratory development.
- `Framework_v2.py` was the follow-on monolithic driver for the three-phase taxonomy variant.

These files are retained for provenance only. The recommended reproducibility path is the staged pipeline in `pipeline/`, especially `pipeline/stage07_paper_stats.py`, which is byte-identical to the production `07_paper_stats.py` used to generate `run_artifacts/probe_52665894.out`.

The legacy scripts contain absolute paths from the original Perlmutter workspace and are not intended to be run without editing those paths.
