#!/bin/bash
#SBATCH --job-name=eval-launch
#SBATCH --partition=genoa
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:15:00
#SBATCH --output=logs/evallaunch_%j.out
#SBATCH --error=logs/evallaunch_%j.err

# Runs AFTER all six trainings finish. Regenerates eval_targets.txt (the stepN
# branches exist on HF now) and submits the eval array sized to match.
set -euo pipefail

module purge
module load 2024
module load Python/3.12.3-GCCcore-13.3.0
export PYTHONNOUSERSITE=1
VENV="${TMPDIR:-/scratch-shared/$USER}/venv_evallaunch"
python -m venv "$VENV"
source "$VENV/bin/activate"
pip install -q --upgrade pip
pip install -q -U huggingface_hub

[ -n "${HF_TOKEN:-}" ] && huggingface-cli login --token "$HF_TOKEN" || true

STRICT_DIR="${STRICT_DIR:-$HOME/babylm-eval/strict}"
mkdir -p "$STRICT_DIR/logs"
cd "$STRICT_DIR"

python "$HOME/augustinian_babylm/eval/list_eval_targets.py" > eval_targets.txt
N=$(wc -l < eval_targets.txt)
echo "eval targets: $N (repo x stepN) pairs"
if [ "$N" -eq 0 ]; then echo "ERROR: empty target list"; exit 1; fi

sbatch --array=1-${N}%8 --export=ALL,HF_TOKEN \
    "$HOME/augustinian_babylm/eval/eval_sweep.slurm"
echo "eval array submitted (1-${N}%8)"
