#!/bin/bash
#SBATCH --job-name=vpswap-artfix
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --output=logs/artfix_%j.out
#SBATCH --error=logs/artfix_%j.err
set -euo pipefail
cd $HOME/augustinian_babylm
module purge
module load 2024 Python/3.12.3-GCCcore-13.3.0 CUDA/12.6.0
source /scratch-shared/$USER/venv_eval2/bin/activate
export PYTHONNOUSERSITE=1
export HF_HOME=/scratch-shared/$USER/hf_cache_artfix
mkdir -p "$HF_HOME"
python eval/patch_vpswap_articles.py \
  --pairs_dir eval/vpswap_bb24 \
  --results_root results/vpswap \
  --cleanup
echo "ARTICLE FIX DONE"
