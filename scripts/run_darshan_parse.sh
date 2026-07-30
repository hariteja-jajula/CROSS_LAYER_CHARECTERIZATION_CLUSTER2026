#!/bin/bash
#SBATCH --job-name=darshan_parse
#SBATCH --account=<your_allocation>
#SBATCH --constraint=cpu
#SBATCH --nodes=1
#SBATCH --time=01:00:00
#SBATCH --qos=regular
#SBATCH --output=/pscratch/sd/h/hjajula/Darshan/cross_layer_hpc_tool/logs/darshan_parse_%j.out
#SBATCH --error=/pscratch/sd/h/hjajula/Darshan/cross_layer_hpc_tool/logs/darshan_parse_%j.err

echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "CPUs: $(nproc)"

cd /pscratch/sd/h/hjajula/Darshan/cross_layer_hpc_tool
mkdir -p logs

module load python
conda activate darshan-py

echo "Python: $(which python)"
echo "Darshan: $(python -c 'import darshan; print(darshan.__file__)')"

python pipeline/stage02_parse_darshan.py --config config/config.json --workers 16

echo "Job finished: $(date)"
