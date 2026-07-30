#!/bin/bash
#SBATCH -N 1
#SBATCH -C cpu
#SBATCH -A <your_allocation>
#SBATCH -t 00:30:00
#SBATCH --qos debug
#SBATCH -J gpu_probe
#SBATCH -o /pscratch/sd/h/hjajula/Darshan/cross_layer_hpc_tool/probe_%j.out
#SBATCH -e /pscratch/sd/h/hjajula/Darshan/cross_layer_hpc_tool/probe_%j.err
#SBATCH --mail-user=<your_email>
#SBATCH --mail-type=END,FAIL
module load python
# module load conda
cd /pscratch/sd/h/hjajula/Darshan/cross_layer_hpc_tool
# conda activate darshan-py
python -m pipeline.stage07_paper_stats
# python -m pipeline.stage02_parse_darshan --config config/config.json
