#!/bin/bash
#SBATCH --job-name=seedtrain
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --output=logs/seedtrain_%x_%j.out
#SBATCH --error=logs/seedtrain_%x_%j.err
# Train one 75k model (baseline or vision-init) with a given seed.
# env: SEED (required), ENCODER (empty = baseline), VOCAB (default 75k)
set -euo pipefail
cd $HOME/augustinian_babylm
mkdir -p logs
VOCAB="${VOCAB:-75k}"; SEED="${SEED:?SEED required}"; ENCODER="${ENCODER:-}"
module purge
module load 2024 Python/3.12.3-GCCcore-13.3.0 CUDA/12.6.0
VENV="${TMPDIR:-/scratch-shared/$USER}/venv_babylm"
if [ ! -d "$VENV" ]; then
    python -m venv "$VENV"; source "$VENV/bin/activate"
    pip install -q --upgrade pip
    pip install -q torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu126
    pip install -q -r requirements.txt
else
    source "$VENV/bin/activate"
fi
export HF_HOME="${TMPDIR:-/scratch-shared/$USER}/hf_cache"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
[ -n "${HF_TOKEN:-}" ] && huggingface-cli login --token "$HF_TOKEN" || true
if [ -n "$ENCODER" ]; then
    REPO="augustinian-babylm/deberta-base-${VOCAB}-${ENCODER}-s${SEED}"
    INIT_ARGS=(--init_embeddings augustinian-babylm/token-embeddings
               --init_encoder "$ENCODER" --init_tag "$VOCAB")
else
    REPO="augustinian-babylm/deberta-base-${VOCAB}-s${SEED}"
    INIT_ARGS=()
fi
echo "=== seed replication: $REPO (seed $SEED) ==="
srun python scripts/train_deberta_babylm.py \
    --tokenizer "augustinian-babylm/babylm-bpe-${VOCAB}" \
    --preset base --epochs 10 --lr 2e-4 --batch_size 256 --grad_acc 4 \
    --seed "$SEED" \
    --dynamics_linear_every 1000 \
    --output_path "/scratch-shared/$USER/models/deberta-babylm-${VOCAB}${ENCODER:+-$ENCODER}-s${SEED}" \
    "${INIT_ARGS[@]}" \
    --push_to_hub "$REPO" --hub_private
echo "TRAIN DONE: $REPO"
