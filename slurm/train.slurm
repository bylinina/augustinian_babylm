#!/bin/bash
#SBATCH --job-name=deberta-train
#SBATCH --partition=gpu_a100          # or gpu_h100 if you have an allocation there
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=10:00:00
#SBATCH --output=logs/train_%x_%j.out
#SBATCH --error=logs/train_%x_%j.err

# ---------------------------------------------------------------------------
# Train ONE BabyLM DeBERTa, parametrized by vocab size.
# The vocab size is read from $VOCAB (default 75k). Submit one job per size:
#
#   sbatch --export=ALL,HF_TOKEN,VOCAB=50k  --job-name=deberta-50k  slurm/train.slurm
#   sbatch --export=ALL,HF_TOKEN,VOCAB=75k  --job-name=deberta-75k  slurm/train.slurm
#   sbatch --export=ALL,HF_TOKEN,VOCAB=100k --job-name=deberta-100k slurm/train.slurm
#
# (--job-name is optional but makes squeue + the log filenames readable.)
# ---------------------------------------------------------------------------
set -euo pipefail
mkdir -p logs

VOCAB="${VOCAB:-75k}"             # which tokenizer: 50k / 75k / 100k
echo "=== Training vocab size: ${VOCAB} ==="

module purge
module load 2024
module load Python/3.12.3-GCCcore-13.3.0
module load CUDA/12.6.0          # check `module avail CUDA` and adjust

# venv on scratch (node-local; rebuilt per job since $TMPDIR is wiped at job end)
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

srun python scripts/train_deberta_babylm.py \
    --tokenizer "augustinian-babylm/babylm-bpe-${VOCAB}" \
    --preset base \
    --grad_acc 4 \
    --output_path "$HOME/models/deberta-babylm-${VOCAB}-base" \
    --push_to_hub "augustinian-babylm/deberta-base-${VOCAB}" \
    --hub_private
    # epochs/lr/batch/warmup/decay/context-warmup all use the script's paper
    # defaults (Table 3): lr 2e-4, 50 epochs, wd 0.01, warmup 4000, ctx 64->128.

echo "Training done for ${VOCAB}."
