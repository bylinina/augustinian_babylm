#!/bin/bash
#SBATCH --job-name=build-tables
#SBATCH --partition=genoa
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --output=logs/build_%x_%j.out
#SBATCH --error=logs/build_%x_%j.err

# Build the six MISSING token-embedding tables: {dinov3,sam,ibot} x {75k,100k}.
# Pure CPU (pandas/numpy aggregation over precomputed region embeddings).
# Built to MATCH the existing 50k tables: --no-seed_last_subword, --scale 0.55.
set -euo pipefail
mkdir -p logs

module purge
module load 2024
module load Python/3.12.3-GCCcore-13.3.0

# Isolated venv on node-local scratch; ignore poisoned ~/.local (py3.9).
export PYTHONNOUSERSITE=1
VENV="${TMPDIR:-/scratch-shared/$USER}/venv_build"
python -m venv "$VENV"
source "$VENV/bin/activate"
pip install -q --upgrade pip
pip install -q -U numpy pandas pyarrow transformers huggingface_hub safetensors tqdm

export HF_HOME="${TMPDIR:-/scratch-shared/$USER}/hf_cache"
[ -n "${HF_TOKEN:-}" ] && huggingface-cli login --token "$HF_TOKEN" || true

# Serial loop (parallel HF downloads have poisoned the cache before -> serial).
for ENC in dinov3 sam ibot; do
  for TAG in 75k 100k; do
    echo "=== build ${ENC} / ${TAG} ==="
    python scripts/build_token_embeddings.py \
        --region_repo augustinian-babylm/region-embeddings \
        --encoder_name "${ENC}" \
        --tokenizer "augustinian-babylm/babylm-bpe-${TAG}" \
        --tokenizer_tag "${TAG}" \
        --no-seed_last_subword \
        --scale 0.55 \
        --push_to_hub augustinian-babylm/token-embeddings
  done
done
echo "=== all six tables built and pushed ==="
