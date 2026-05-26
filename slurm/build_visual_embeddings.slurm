#!/bin/bash
#SBATCH --job-name=visual-emb
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=06:00:00
#SBATCH --output=logs/visualemb_%x_%j.out
#SBATCH --error=logs/visualemb_%x_%j.err

# ---------------------------------------------------------------------------
# Build visual embedding tables from the grounding dataset for ONE encoder,
# producing all three tokenizer tables (50k/75k/100k) in a single encode pass.
#
# Parametrized by $ENC (dinov3 | ibot | sam) and $ENC_ID (the HF model id).
# Submit one job per encoder:
#
#   sbatch --export=ALL,HF_TOKEN,ENC=dinov3,ENC_ID=facebook/dinov3-vitb16-pretrain-lvd1689m \
#          --job-name=vemb-dinov3 slurm/build_visual_embeddings.slurm
#
#   sbatch --export=ALL,HF_TOKEN,ENC=sam,ENC_ID=facebook/sam-vit-base \
#          --job-name=vemb-sam slurm/build_visual_embeddings.slurm
#
#   # iBOT: ENC_ID is the OFFICIAL ByteDance ViT-B/16 checkpoint (.pth) URL or
#   # local path (github.com/bytedance/ibot -> ViT-B/16 ImageNet-22k). Loaded via
#   # timm. Default ckpt key is 'teacher'; override with --ckpt_key in the srun line.
#   sbatch --export=ALL,HF_TOKEN,ENC=ibot,ENC_ID=<ibot_vitb16_checkpoint.pth_url> \
#          --job-name=vemb-ibot slurm/build_visual_embeddings.slurm
# ---------------------------------------------------------------------------
set -euo pipefail
mkdir -p logs

ENC="${ENC:-dinov3}"
ENC_ID="${ENC_ID:-facebook/dinov3-vitb16-pretrain-lvd1689m}"
echo "=== Building visual embeddings: encoder=${ENC} (${ENC_ID}) ==="

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

srun python scripts/build_visual_embeddings.py \
    --encoder_name "${ENC}" \
    --encoder "${ENC_ID}" \
    --tokenizers "50k=augustinian-babylm/babylm-bpe-50k,75k=augustinian-babylm/babylm-bpe-75k,100k=augustinian-babylm/babylm-bpe-100k" \
    --out_dir "$HOME/visual_embeddings" \
    --push_to_hub augustinian-babylm/visual-embeddings

echo "Done building visual embeddings for ${ENC}."
