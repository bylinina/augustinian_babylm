#!/bin/bash
#SBATCH --job-name=vpswap
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --output=logs/vpswap_%x_%j.out
#SBATCH --error=logs/vpswap_%x_%j.err
# Usage: sbatch --export=ALL,HF_TOKEN,MODEL=augustinian-babylm/deberta-base-75k-sam \
#   --job-name=vpswap-75ksam ~/augustinian_babylm/eval/eval_vpswap.slurm
set -euo pipefail
cd $HOME/augustinian_babylm
mkdir -p logs
module purge
module load 2024 Python/3.12.3-GCCcore-13.3.0 CUDA/12.6.0
source /scratch-shared/$USER/babylm-eval/venv_eval/bin/activate
[ -n "${HF_TOKEN:-}" ] && { hf auth login --token "$HF_TOKEN" 2>/dev/null || true; export HF_TOKEN; }
M=$(basename "$MODEL")
REVS="step0 chck_1M chck_2M chck_3M chck_4M chck_5M chck_6M chck_7M chck_8M chck_9M \
chck_10M chck_20M chck_30M chck_40M chck_50M chck_60M chck_70M chck_80M chck_90M chck_100M"
for rev in $REVS; do
  out="results/vpswap/$M/$rev.jsonl"
  if [ -s "$out" ]; then echo "skip $rev (done)"; continue; fi
  echo "=== $M @ $rev ==="
  python eval/vpswap_score.py --pairs_dir eval/vpswap_bb24 \
    --model "$MODEL" --revision "$rev" --out "$out"
done
echo "VPSWAP DONE: $MODEL"
