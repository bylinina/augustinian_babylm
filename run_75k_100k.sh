#!/bin/bash
# Fire-and-forget: build 6 tables -> train 6 models -> eval sweep.
# Run ONCE from a clean login node:  bash run_75k_100k.sh
# HF_TOKEN must be live in your shell first.
set -euo pipefail
cd "$HOME/augustinian_babylm"
mkdir -p logs

if [ -z "${HF_TOKEN:-}" ]; then
  echo "ERROR: HF_TOKEN not set. Run: export HF_TOKEN=hf_...  then re-run." >&2
  exit 1
fi
echo "HF_TOKEN present (${HF_TOKEN:0:5}...)"

# --- 1. Build the six missing tables (CPU). Capture job id.
BUILD_ID=$(sbatch --parsable --export=ALL,HF_TOKEN slurm/build_tables.slurm)
echo "build job: $BUILD_ID"

# --- 2. Six training jobs, each waiting for the build to finish OK.
TRAIN_IDS=""
for VOCAB in 75k 100k; do
  for ENC in dinov3 sam ibot; do
    JID=$(sbatch --parsable \
        --dependency=afterok:$BUILD_ID \
        --export=ALL,HF_TOKEN,VOCAB=$VOCAB,ENCODER=$ENC \
        --job-name=vi-${VOCAB}-${ENC} \
        slurm/train_visioninit.slurm)
    echo "train ${VOCAB}/${ENC}: $JID"
    TRAIN_IDS="${TRAIN_IDS}:${JID}"
  done
done
TRAIN_IDS="${TRAIN_IDS#:}"   # strip leading colon
echo "all training jobs: $TRAIN_IDS"

# --- 3. Eval-launcher job: waits for ALL six trainings, regenerates the
#        target list (stepN branches now exist on HF), submits the eval array.
LAUNCH_ID=$(sbatch --parsable \
    --dependency=afterok:$TRAIN_IDS \
    --export=ALL,HF_TOKEN \
    --job-name=eval-launch \
    slurm/eval_launch.slurm)
echo "eval-launcher: $LAUNCH_ID (fires the eval array once training completes)"

echo
echo "Submitted. Chain: build($BUILD_ID) -> train($TRAIN_IDS) -> eval-launch($LAUNCH_ID)."
echo "Walk away. Check progress any time with:  squeue --me"
