#!/bin/bash
set -euo pipefail
cd "$HOME/augustinian_babylm"; mkdir -p logs
[ -z "${HF_TOKEN:-}" ] && { echo "export HF_TOKEN first"; exit 1; }
echo "token: ${HF_TOKEN:0:5}..."
ALL_EVAL=""
# s3 pair: trained; evalpacks start immediately
for M in deberta-base-75k-s3 deberta-base-75k-sam-s3; do
  EID=$(sbatch --parsable --export=ALL,HF_TOKEN,MODEL=augustinian-babylm/$M \
        --job-name=ev-$M slurm/seed_evalpack.slurm)
  echo "evalpack $M: $EID"; ALL_EVAL="${ALL_EVAL}:${EID}"
done
# s2 pair: retrain (quota fixed, scratch output), then evalpack
for ENC in "" "sam"; do
  NAME="75k${ENC:+-$ENC}-s2"
  TID=$(sbatch --parsable --export=ALL,HF_TOKEN,SEED=2,ENCODER=$ENC,VOCAB=75k \
        --job-name=tr-$NAME slurm/train_seed.slurm)
  MODEL="augustinian-babylm/deberta-base-75k${ENC:+-$ENC}-s2"
  EID=$(sbatch --parsable --dependency=afterok:$TID \
        --export=ALL,HF_TOKEN,MODEL=$MODEL --job-name=ev-$NAME slurm/seed_evalpack.slurm)
  echo "retrain $NAME: $TID -> evalpack $EID"; ALL_EVAL="${ALL_EVAL}:${EID}"
done
ALL_EVAL="${ALL_EVAL#:}"
AID=$(sbatch --parsable --dependency=afterok:$ALL_EVAL --export=ALL,HF_TOKEN \
      --job-name=seed-analysis --partition=gpu_a100 --gpus=1 --time=01:00:00 \
      --output=logs/analysis_%j.out --error=logs/analysis_%j.err \
      --wrap "cd \$HOME/augustinian_babylm && module load 2024 Python/3.12.3-GCCcore-13.3.0 && export PATH=\$HOME/.local/bin:\$PATH && python3 eval/analyze_seeds.py")
echo "analysis: $AID"
squeue --me
