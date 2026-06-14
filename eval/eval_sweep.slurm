#!/bin/bash
#SBATCH --job-name=babylm-eval-sweep
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:45:00
#SBATCH --output=logs/eval_%A_%a.out
#SBATCH --error=logs/eval_%A_%a.err

# BabyLM fast zero-shot eval over ALL (vocab x stepN) checkpoints.
# Usage:
#   cd $HOME/babylm-eval/strict && mkdir -p logs
#   python $HOME/augustinian_babylm/eval/list_eval_targets.py > eval_targets.txt
#   N=$(wc -l < eval_targets.txt)
#   sbatch --array=1-${N}%8 $HOME/augustinian_babylm/eval/eval_sweep.slurm
# Resumable: re-submitting skips checkpoints whose reading/report.txt exists.
set -euo pipefail

STRICT_DIR="${STRICT_DIR:-$HOME/babylm-eval/strict}"
VENV="${EVAL_VENV:-/scratch-shared/$USER/babylm-eval/venv_eval}"
cd "$STRICT_DIR"

module purge
module load 2024 Python/3.12.3-GCCcore-13.3.0 CUDA/12.6.0
source "$VENV/bin/activate"

# --- HF auth: eval downloads PRIVATE model repos; carry the token explicitly.
# Submit with:  sbatch --array=... --export=ALL,HF_TOKEN eval_sweep.slurm
if [ -n "${HF_TOKEN:-}" ]; then
    hf auth login --token "$HF_TOKEN" 2>/dev/null \
        || huggingface-cli login --token "$HF_TOKEN" 2>/dev/null || true
    export HF_TOKEN
fi

LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" eval_targets.txt)
REPO=$(echo "$LINE" | awk '{print $1}')
BRANCH=$(echo "$LINE" | awk '{print $2}')
MODEL_NAME=$(basename "$REPO")
echo "=== [$SLURM_ARRAY_TASK_ID] $REPO @ $BRANCH ==="

DONE_MARKER="results/${MODEL_NAME}/${BRANCH}/zero_shot/mlm/reading/report.txt"
if [ -f "$DONE_MARKER" ]; then
    echo "already complete, skipping"
    exit 0
fi

bash scripts/eval_zero_shot_fast.sh "$REPO" "$BRANCH" mlm
echo "=== done: $REPO @ $BRANCH ==="
