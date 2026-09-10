#!/bin/bash
#SBATCH --job-name=coverage
#SBATCH --partition=rome
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=logs/coverage_%j.out
#SBATCH --error=logs/coverage_%j.err
set -euo pipefail
cd $HOME/augustinian_babylm
mkdir -p logs
module purge
module load 2024 Python/3.12.3-GCCcore-13.3.0
source /scratch-shared/$USER/venv_eval2/bin/activate
export PYTHONNOUSERSITE=1
python analysis/coverage_deep.py
echo "COVERAGE DONE"
