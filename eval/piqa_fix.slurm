#!/bin/bash
#SBATCH --job-name=piqa-fix
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --output=logs/piqa_%x_%j.out
#SBATCH --error=logs/piqa_%x_%j.err
set -euo pipefail
cd $HOME/babylm-eval/strict
mkdir -p logs
module purge
module load 2024 Python/3.12.3-GCCcore-13.3.0 CUDA/12.6.0
source /scratch-shared/$USER/babylm-eval/venv_eval/bin/activate
if [ -n "${HF_TOKEN:-}" ]; then
    hf auth login --token "$HF_TOKEN" 2>/dev/null || true
    export HF_TOKEN
fi
bash scripts/eval_zero_shot_global_piqa.sh "$MODEL" mlm strict-small
echo "PIQA DONE: $MODEL"
