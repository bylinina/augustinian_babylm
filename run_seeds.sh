#!/bin/bash
# Overnight seed replication: 4 trainings -> 4 evalpacks -> 1 analysis.
# Run ONCE from a clean login node with HF_TOKEN exported.
set -euo pipefail
cd "$HOME/augustinian_babylm"
mkdir -p logs
[ -z "${HF_TOKEN:-}" ] && { echo "ERROR: export HF_TOKEN first"; exit 1; }
echo "HF_TOKEN present (${HF_TOKEN:0:5}...)"
# preflight: is --seed actually applied in the training script?
if ! grep -qE "set_seed|manual_seed" scripts/train_deberta_babylm.py; then
    echo "ABORT: --seed is parsed but no set_seed/manual_seed found."
    echo "Run the seed-wiring patch, then relaunch."; exit 1
fi
test -d eval/vpswap_bb24 || { echo "ABORT: vpswap pairs missing"; exit 1; }

ALL_EVAL=""
for SEED in 2 3; do
  for ENC in "" "sam"; do
    NAME="75k${ENC:+-$ENC}-s${SEED}"
    TID=$(sbatch --parsable --export=ALL,HF_TOKEN,SEED=$SEED,ENCODER=$ENC,VOCAB=75k \
          --job-name=tr-$NAME slurm/train_seed.slurm)
    MODEL="augustinian-babylm/deberta-base-75k${ENC:+-$ENC}-s${SEED}"
    EID=$(sbatch --parsable --dependency=afterok:$TID \
          --export=ALL,HF_TOKEN,MODEL=$MODEL \
          --job-name=ev-$NAME slurm/seed_evalpack.slurm)
    echo "train $NAME: $TID -> evalpack: $EID"
    ALL_EVAL="${ALL_EVAL}:${EID}"
  done
done
ALL_EVAL="${ALL_EVAL#:}"
AID=$(sbatch --parsable --dependency=afterok:$ALL_EVAL \
      --export=ALL,HF_TOKEN --job-name=seed-analysis \
      --partition=gpu_a100 --gpus=1 --time=01:00:00 \
      --output=logs/analysis_%j.out --error=logs/analysis_%j.err \
      --wrap "cd \$HOME/augustinian_babylm && module load 2024 Python/3.12.3-GCCcore-13.3.0 && export PATH=\$HOME/.local/bin:\$PATH && python3 eval/analyze_seeds.py")
echo "analysis job: $AID (runs after all evalpacks)"
squeue --me
