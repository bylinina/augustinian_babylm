#!/bin/bash
#SBATCH --job-name=full-zeroshot
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:30:00
#SBATCH --output=logs/fullzs_%A_%a.out
#SBATCH --error=logs/fullzs_%A_%a.err
# Full zero-shot battery (BLiMP, Supp, EWoK, Entity, COMPS, Reading) over the
# 24 selected checkpoints. Reads eval/full_eval_targets.txt (repo<TAB>branch<TAB>sel).
# Resumable: skips a target whose COMPS report already exists (last task written).
set -euo pipefail

STRICT_DIR="${STRICT_DIR:-$HOME/babylm-eval/strict}"
TARGETS="${TARGETS:-$HOME/augustinian_babylm/eval/full_eval_targets.txt}"
cd "$STRICT_DIR"
mkdir -p logs

module purge
module load 2024 Python/3.12.3-GCCcore-13.3.0 CUDA/12.6.0
VENV="${EVAL_VENV:-/scratch-shared/$USER/babylm-eval/venv_eval}"
source "$VENV/bin/activate"
[ -n "${HF_TOKEN:-}" ] && { hf auth login --token "$HF_TOKEN" 2>/dev/null || huggingface-cli login --token "$HF_TOKEN" 2>/dev/null || true; }

LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$TARGETS")
REPO=$(echo "$LINE" | cut -f1)
BRANCH=$(echo "$LINE" | cut -f2)
MODEL_NAME=$(basename "$REPO")
echo "=== [$SLURM_ARRAY_TASK_ID] $REPO @ $BRANCH ==="

# resume marker: comps is the last zero-shot task the script runs
DONE="results/${MODEL_NAME}/${BRANCH}/zero_shot/mlm/comps"
if [ -d "$DONE" ]; then echo "already done, skipping"; exit 0; fi

bash scripts/eval_zero_shot.sh "$REPO" mlm evaluation_data/full_eval "$BRANCH"
echo "=== done: $REPO @ $BRANCH ==="
