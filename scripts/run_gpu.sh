#!/bin/bash
#SBATCH -N 1
#SBATCH -C cpu
#SBATCH -A <your_allocation>
#SBATCH -t 04:00:00
#SBATCH -J parse_gpu
#SBATCH --qos regular
#SBATCH -o /pscratch/sd/h/hjajula/Darshan/cross_layer_hpc_tool/logs/parse_gpu.out
#SBATCH -e /pscratch/sd/h/hjajula/Darshan/cross_layer_hpc_tool/logs/parse_gpu.err

module load python
cd /pscratch/sd/h/hjajula/Darshan/cross_layer_hpc_tool
python -m pipeline.stage01_parse_gpu --config config/config.json
