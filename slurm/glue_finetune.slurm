#!/bin/bash
#SBATCH --job-name=glue-ft
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --output=logs/glue_%A_%a.out
#SBATCH --error=logs/glue_%A_%a.err
# One array task = one (checkpoint x GLUE task). Reads eval/glue_targets.txt:
#   repo <TAB> branch <TAB> task <TAB> num_labels <TAB> bsz <TAB> epochs <TAB> metric
# No W&B (sidesteps missing ../.env). Resumable via results/.../finetune/<task> marker.
set -euo pipefail
export WANDB_DISABLED=true

STRICT_DIR="${STRICT_DIR:-$HOME/babylm-eval/strict}"
TARGETS="${TARGETS:-$HOME/augustinian_babylm/eval/glue_targets.txt}"
cd "$STRICT_DIR"
mkdir -p logs results

module purge
module load 2024 Python/3.12.3-GCCcore-13.3.0 CUDA/12.6.0
VENV="${EVAL_VENV:-/scratch-shared/$USER/babylm-eval/venv_eval}"
source "$VENV/bin/activate"
[ -n "${HF_TOKEN:-}" ] && { hf auth login --token "$HF_TOKEN" 2>/dev/null || huggingface-cli login --token "$HF_TOKEN" 2>/dev/null || true; }

LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$TARGETS")
REPO=$(echo "$LINE"   | cut -f1)
BRANCH=$(echo "$LINE" | cut -f2)
TASK=$(echo "$LINE"   | cut -f3)
NL=$(echo "$LINE"     | cut -f4)
BSZ=$(echo "$LINE"    | cut -f5)
EP=$(echo "$LINE"     | cut -f6)
METRIC=$(echo "$LINE" | cut -f7)
MODEL_NAME=$(basename "$REPO")
echo "=== [$SLURM_ARRAY_TASK_ID] $MODEL_NAME @ $BRANCH : $TASK (nl=$NL bsz=$BSZ ep=$EP metric=$METRIC) ==="

# resume marker
DONE="results/${MODEL_NAME}/${BRANCH}/finetune/${TASK}"
if [ -d "$DONE" ] && [ -f "$DONE/predictions.json" ]; then echo "already done, skipping"; exit 0; fi

D="evaluation_data/full_eval/glue_filtered"
python -m evaluation_pipeline.finetune.run \
    --model_name_or_path "$REPO" \
    --revision_name "$BRANCH" \
    --train_data   "$D/${TASK}.train.jsonl" \
    --valid_data   "$D/${TASK}.valid.jsonl" \
    --predict_data "$D/${TASK}.valid.jsonl" \
    --task "$TASK" \
    --num_labels "$NL" \
    --batch_size "$BSZ" \
    --learning_rate 3e-5 \
    --num_epochs "$EP" \
    --sequence_length 512 \
    --results_dir "results" \
    --metrics $([ "$NL" = "3" ] && echo "accuracy" || echo "accuracy f1 mcc") \
    --metric_for_valid "$METRIC" \
    --seed 42 \
    --padding_side left \
    --take_final
echo "=== done: $MODEL_NAME @ $BRANCH : $TASK ==="
