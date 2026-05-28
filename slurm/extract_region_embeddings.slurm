#!/bin/bash
#SBATCH --job-name=region-emb
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=06:00:00
#SBATCH --output=logs/regionemb_%x_%j.out
#SBATCH --error=logs/regionemb_%x_%j.err

# ---------------------------------------------------------------------------
# STAGE 1 (Ece): per-row visual embeddings (whole image / bbox region).
# NO tokenizer here — output is one 768-d vector per annotation row + its text
# as metadata. Token-level aggregation is a separate later stage.
#
# Parametrized by $ENC (dinov3|ibot|sam) and $ENC_ID. One job per encoder:
#
#   sbatch --export=ALL,HF_TOKEN,ENC=dinov3,ENC_ID=facebook/dinov3-vitb16-pretrain-lvd1689m \
#          --job-name=region-dinov3 slurm/extract_region_embeddings.slurm
#   sbatch --export=ALL,HF_TOKEN,ENC=sam,ENC_ID=facebook/sam-vit-base \
#          --job-name=region-sam slurm/extract_region_embeddings.slurm
#   # iBOT: ENC_ID = local path to the ByteDance ViT-B/16 .pth (see README step 2)
#   sbatch --export=ALL,HF_TOKEN,ENC=ibot,ENC_ID=$HOME/ibot_vitb16_pt22k.pth \
#          --job-name=region-ibot slurm/extract_region_embeddings.slurm
# ---------------------------------------------------------------------------
set -euo pipefail
mkdir -p logs

ENC="${ENC:-dinov3}"
ENC_ID="${ENC_ID:-facebook/dinov3-vitb16-pretrain-lvd1689m}"
echo "=== Stage 1: region embeddings | encoder=${ENC} (${ENC_ID}) ==="

module purge
module load 2024
module load Python/3.12.3-GCCcore-13.3.0
module load CUDA/12.6.0          # check `module avail CUDA` and adjust

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

srun python scripts/extract_region_embeddings.py \
    --encoder_name "${ENC}" \
    --encoder "${ENC_ID}" \
    --out_dir "$HOME/region_embeddings" \
    --push_to_hub augustinian-babylm/region-embeddings

echo "Done: Stage 1 region embeddings for ${ENC}."
