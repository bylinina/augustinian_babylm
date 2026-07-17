#!/bin/bash
#SBATCH --job-name=evalpack
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=logs/evalpack_%x_%j.out
#SBATCH --error=logs/evalpack_%x_%j.err
# env: MODEL (full HF repo id)
set -euo pipefail
mkdir -p $HOME/augustinian_babylm/logs
module purge
module load 2024 Python/3.12.3-GCCcore-13.3.0 CUDA/12.6.0
VENV=/scratch-shared/$USER/babylm-eval/venv_eval
if ! "$VENV/bin/python" -c "import torch, transformers" 2>/dev/null; then
    echo "eval venv broken/missing -> rebuilding"
    rm -rf "$VENV"; python -m venv "$VENV"; source "$VENV/bin/activate"
    pip install -q --upgrade pip
    pip install -q torch==2.7.0 --index-url https://download.pytorch.org/whl/cu126
    pip install -q -r $HOME/babylm-eval/strict/requirements.txt 2>/dev/null || \
    pip install -q -r $HOME/babylm-eval/requirements.txt
else
    source "$VENV/bin/activate"
fi
[ -n "${HF_TOKEN:-}" ] && { hf auth login --token "$HF_TOKEN" 2>/dev/null || true; export HF_TOKEN; }
M=$(basename "$MODEL")

echo "=== [1/3] mint chck branches: $MODEL ==="
python $HOME/augustinian_babylm/scripts/mint_submission_branches.py \
    --repo "$MODEL" --corpus_words 9923835 --steps_per_epoch 2574 --epochs 10

echo "=== [2/3] VP-Swap trajectory: $MODEL ==="
cd $HOME/augustinian_babylm
REVS="step0 chck_1M chck_2M chck_3M chck_4M chck_5M chck_6M chck_7M chck_8M chck_9M \
chck_10M chck_20M chck_30M chck_40M chck_50M chck_60M chck_70M chck_80M chck_90M chck_100M"
for rev in $REVS; do
    out="results/vpswap/$M/$rev.jsonl"
    [ -s "$out" ] && { echo "skip $rev"; continue; }
    python eval/vpswap_score.py --pairs_dir eval/vpswap_bb24 \
        --model "$MODEL" --revision "$rev" --out "$out"
done
test -s "results/vpswap/$M/chck_100M.jsonl" || { echo "VPSWAP FAIL"; exit 1; }

echo "=== [3/3] full zero-shot (main): $MODEL ==="
cd $HOME/babylm-eval/strict
bash scripts/eval_zero_shot.sh "$MODEL" mlm evaluation_data/full_eval main
test -s "results/$M/main/zero_shot/mlm/blimp/blimp_filtered/predictions.json" || {
    echo "ZEROSHOT FAIL"; exit 1; }
echo "EVALPACK DONE: $MODEL"
