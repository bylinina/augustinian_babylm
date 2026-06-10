#!/bin/bash
#SBATCH --job-name=deberta-train
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --output=logs/train_%x_%j.out
#SBATCH --error=logs/train_%x_%j.err

# ---------------------------------------------------------------------------
# Train ONE BabyLM DeBERTa, parametrized by vocab size ($VOCAB, default 75k).
#   sbatch --export=ALL,VOCAB=50k  --job-name=deberta-50k  slurm/train.slurm
#   sbatch --export=ALL,VOCAB=75k  --job-name=deberta-75k  slurm/train.slurm
#   sbatch --export=ALL,VOCAB=100k --job-name=deberta-100k slurm/train.slurm
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
    # torch + torchvision MUST come from the SAME cu126 index. timm (requirements.txt)
    # pulls torchvision; mixing a cu126 torch with a PyPI torchvision breaks
    # `torchvision::nms` and kills the whole transformers import. Pinned matched pair.
    pip install -q torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu126
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
    --epochs 10 \
    --lr 2e-4 \
    --batch_size 256 \
    --grad_acc 4 \
    --dynamics_linear_every 1000 \
    --output_path "$HOME/models/deberta-babylm-${VOCAB}-base" \
    --push_to_hub "augustinian-babylm/deberta-base-${VOCAB}" \
    --hub_private

echo "Training done for ${VOCAB}."
