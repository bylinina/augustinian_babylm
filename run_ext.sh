#!/bin/bash
# Synthetic-coverage extension: images -> ext table -> 3 seeds train+eval.
set -euo pipefail
cd "$HOME/augustinian_babylm"; mkdir -p logs
[ -z "${HF_TOKEN:-}" ] && { echo "export HF_TOKEN first"; exit 1; }
echo "token: ${HF_TOKEN:0:5}..."
test -s analysis/out/synth/descriptions.jsonl || { echo "descriptions missing"; exit 1; }

GID=$(sbatch --parsable --export=ALL,HF_TOKEN slurm/gen_images.slurm)
echo "images: $GID"
BID=$(sbatch --parsable --dependency=afterok:$GID --export=ALL,HF_TOKEN slurm/build_ext_table.slurm)
echo "ext table: $BID"
for SEED in 1 2 3; do
  TID=$(sbatch --parsable --dependency=afterok:$BID \
        --export=ALL,HF_TOKEN,SEED=$SEED,ENCODER=sam_ext,VOCAB=75k \
        --job-name=tr-ext-s$SEED slurm/train_seed.slurm)
  EID=$(sbatch --parsable --dependency=afterok:$TID \
        --export=ALL,HF_TOKEN,MODEL=augustinian-babylm/deberta-base-75k-sam_ext-s$SEED \
        --job-name=ev-ext-s$SEED slurm/seed_evalpack.slurm)
  echo "train ext-s$SEED: $TID -> evalpack: $EID"
done
squeue --me
