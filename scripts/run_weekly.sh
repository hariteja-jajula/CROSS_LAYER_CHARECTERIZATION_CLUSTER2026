#!/bin/bash
#SBATCH -A <your_allocation>
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -t 08:00:00
#SBATCH -N 1
#SBATCH -c 64
#SBATCH --job-name=hpc_tool_weekly
#SBATCH --output=logs/weekly_%j.log

source ~/darshan_env/bin/activate
cd /pscratch/sd/h/hjajula/cross_layer_hpc_tool/

python -m pipeline.stage00_convert_gpu_to_parquet --config config/config.json
python -m pipeline.stage01_parse_gpu              --config config/config.json
python -m pipeline.stage02_parse_darshan          --config config/config.json
python -m pipeline.stage03_build_combined         --config config/config.json
python -m pipeline.stage04_fingerprint            --config config/config.json
python -m pipeline.stage05_repeatability          --config config/config.json
python -m pipeline.stage06_results                --config config/config.json

echo "Report: $(pwd)/data/results_report.html"