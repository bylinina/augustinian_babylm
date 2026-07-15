#!/bin/bash
#SBATCH --job-name=bl26-glue
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=logs/glue_%x_%j.out
#SBATCH --error=logs/glue_%x_%j.err
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
echo "=== GLUE finetuning (post grad-accum fix): $MODEL ==="
bash scripts/eval_finetuning.sh --model_path "$MODEL" --revision best
echo "=== done: $MODEL ==="
