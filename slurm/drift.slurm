#!/bin/bash
#SBATCH --job-name=drift
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=logs/drift_%j.out
#SBATCH --error=logs/drift_%j.err
set -euo pipefail
cd $HOME/augustinian_babylm
mkdir -p logs
module purge
module load 2024 Python/3.12.3-GCCcore-13.3.0 CUDA/12.6.0
source /scratch-shared/$USER/venv_eval2/bin/activate
export PYTHONNOUSERSITE=1
export HF_HOME=$HOME/hf_cache_artfix
mkdir -p "$HF_HOME"
python eval/drift_diagnostic.py
echo "DRIFT DONE"
