#!/bin/bash
#SBATCH --job-name=bl26-full
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=20:00:00
#SBATCH --output=logs/full_%x_%j.out
#SBATCH --error=logs/full_%x_%j.err
# Full BabyLM-2026 leaderboard-format evaluation for one model (revision=main).
# Usage:
#   sbatch --export=ALL,HF_TOKEN,MODEL=augustinian-babylm/<model> \
#     --job-name=full-<model> eval_2026_full_rerun.slurm
# Prereqs: see eval/README.md "2026 evaluation runbook".
set -euo pipefail
cd $HOME/babylm-eval/strict
mkdir -p logs
module purge
module load 2024 Python/3.12.3-GCCcore-13.3.0 CUDA/12.6.0
source /scratch-shared/$USER/babylm-eval/venv_eval/bin/activate
if [ -n "${HF_TOKEN:-}" ]; then
    hf auth login --token "$HF_TOKEN" 2>/dev/null || true
    export HF_TOKEN
fi

# ---- preflight: fail fast on missing data/auth ----
test -s evaluation_data/full_eval/global_piqa_parallel/eng_latn.jsonl || {
  echo "PREFLIGHT FAIL: global_piqa data missing. Run: python -m evaluation_pipeline.global_piqa.dl"; exit 1; }
test -s evaluation_data/full_eval/aoa/cdi_childes.json || {
  echo "PREFLIGHT FAIL: AoA data missing. Run: python -m scripts.download_evals"; exit 1; }
test -s evaluation_data/full_eval/aoa/cdi_human.csv || {
  echo "PREFLIGHT FAIL: cdi_human.csv missing"; exit 1; }
python -c "from huggingface_hub import HfApi; HfApi().repo_info('$MODEL')" || {
  echo "PREFLIGHT FAIL: cannot read $MODEL (token missing?)"; exit 1; }
M=$(basename "$MODEL")

echo "=== [1/5] full zero-shot (main): $MODEL ==="
bash scripts/eval_zero_shot.sh "$MODEL" causal evaluation_data/full_eval main
test -s "results/$M/main/zero_shot/causal/blimp/blimp_filtered/predictions.json" || {
  echo "STAGE 1 FAIL: no blimp predictions"; exit 1; }

echo "=== [2/5] global_piqa (main + fast revisions): $MODEL ==="
bash scripts/eval_zero_shot_global_piqa.sh "$MODEL" causal strict-small

echo "=== [3/5] fast eval, all chck revisions: $MODEL ==="
bash scripts/eval_zero_shot_fast_all_revisions.sh "$MODEL" causal strict-small

echo "=== [4/5] AoA: $MODEL ==="
bash scripts/eval_aoa.sh "$MODEL" causal strict-small
test -s "results/$M/main/zero_shot/causal/AoA_word/surprisal.json" || {
  echo "STAGE 4 FAIL: no surprisal.json"; exit 1; }
test -s "results/$M/main/zero_shot/causal/AoA_word/aoa_score.json" || {
  echo "STAGE 4 FAIL: no aoa_score.json"; exit 1; }

echo "=== [5/5] GLUE finetuning (main): $MODEL ==="
bash scripts/eval_finetuning.sh --model_path "$MODEL"
for t in boolq mnli mrpc multirc qqp rte wsc; do
  test -s "results/$M/main/finetune/$t/results.txt" || {
    echo "STAGE 5 FAIL: no results for $t"; exit 1; }
done

echo "=== ALL DONE: $MODEL ==="
