#!/bin/bash
#SBATCH --job-name=deberta-eval
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=logs/eval_%x_%j.out
#SBATCH --error=logs/eval_%x_%j.err

# ---------------------------------------------------------------------------
# Evaluate ONE trained BabyLM DeBERTa, parametrized by vocab size.
# Submit one job per model:
#
#   sbatch --export=ALL,HF_TOKEN,VOCAB=50k  --job-name=eval-50k  slurm/eval.slurm
#   sbatch --export=ALL,HF_TOKEN,VOCAB=75k  --job-name=eval-75k  slurm/eval.slurm
#   sbatch --export=ALL,HF_TOKEN,VOCAB=100k --job-name=eval-100k slurm/eval.slurm
# ---------------------------------------------------------------------------
set -euo pipefail
mkdir -p logs

VOCAB="${VOCAB:-75k}"
echo "=== Evaluating vocab size: ${VOCAB} ==="

module purge
module load 2024
module load Python/3.12.3-GCCcore-13.3.0
module load CUDA/12.6.0

VENV="${TMPDIR:-/scratch-shared/$USER}/venv_babylm"
if [ ! -d "$VENV" ]; then
    python -m venv "$VENV"
    source "$VENV/bin/activate"
    pip install -q --upgrade pip
    pip install -q torch --index-url https://download.pytorch.org/whl/cu126
    pip install -q -r requirements.txt
else
    source "$VENV/bin/activate"
fi

export HF_HOME="${TMPDIR:-/scratch-shared/$USER}/hf_cache"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
[ -n "${HF_TOKEN:-}" ] && huggingface-cli login --token "$HF_TOKEN" || true

# Point --model at the local output dir from training (or an HF repo id).
srun python scripts/eval_deberta_babylm.py \
    --model "$HOME/models/deberta-babylm-${VOCAB}-base" \
    --max_eval_lines 2000

echo "Eval done for ${VOCAB}."
