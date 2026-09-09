#!/bin/bash
#SBATCH --job-name=vpswap-sanity
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=logs/sanity_%j.out
#SBATCH --error=logs/sanity_%j.err
set -euo pipefail
cd $HOME/augustinian_babylm
module purge
module load 2024 Python/3.12.3-GCCcore-13.3.0 CUDA/12.6.0
source /scratch-shared/$USER/venv_eval2/bin/activate
export PYTHONNOUSERSITE=1
export HF_HOME=/scratch-shared/$USER/hf_cache_artfix
mkdir -p "$HF_HOME"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
python eval/vpswap_score.py --pairs_dir eval/vpswap_bb24 \
  --model augustinian-babylm/deberta-base-75k-sam --revision step0 \
  --out /tmp/step0_check.jsonl --properties color
echo "SANITY DONE"
