#!/bin/bash
#SBATCH --job-name=aoa-fix
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --output=logs/aoa_%x_%j.out
#SBATCH --error=logs/aoa_%x_%j.err
set -euo pipefail
cd $HOME/babylm-eval/strict
mkdir -p logs
module purge
module load 2024 Python/3.12.3-GCCcore-13.3.0 CUDA/12.6.0
source /scratch-shared/$USER/babylm-eval/venv_eval/bin/activate
[ -n "${HF_TOKEN:-}" ] && { hf auth login --token "$HF_TOKEN" 2>/dev/null || true; export HF_TOKEN; }
bash scripts/eval_aoa.sh "$MODEL" mlm strict-small
# loud guard: fail if the outputs did not materialize
M=$(basename "$MODEL")
test -s "results/$M/main/zero_shot/mlm/AoA_word/surprisal.json" || { echo "AOA FAILED: no surprisal.json"; exit 1; }
test -s "results/$M/main/zero_shot/mlm/AoA_word/aoa_score.json" || { echo "AOA FAILED: no aoa_score.json"; exit 1; }
echo "AOA DONE: $MODEL"
