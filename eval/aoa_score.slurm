#!/bin/bash
#SBATCH --job-name=aoa-score
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --output=logs/aoascore_%j.out
#SBATCH --error=logs/aoascore_%j.err
set -euo pipefail
cd $HOME/babylm-eval/strict
module purge
module load 2024 Python/3.12.3-GCCcore-13.3.0 CUDA/12.6.0
source /scratch-shared/$USER/babylm-eval/venv_eval/bin/activate
[ -n "${HF_TOKEN:-}" ] && { hf auth login --token "$HF_TOKEN" 2>/dev/null || true; export HF_TOKEN; }
python3 - << 'PYEOF'
import json, pathlib
from evaluation_pipeline.AoA_word.run import AoAEvaluator, JsonProcessor
from transformers import PreTrainedTokenizerFast
cdi = pathlib.Path("evaluation_data/full_eval/aoa/cdi_human.csv")
for m in ["deberta-base-75k-sam", "deberta-base-75k", "deberta-base-100k-sam"]:
    d = pathlib.Path(f"results/{m}/main/zero_shot/mlm/AoA_word")
    data = json.load(open(d / "surprisal.json"))
    tok = PreTrainedTokenizerFast.from_pretrained(f"augustinian-babylm/{m}", trust_remote_code=True)
    score = AoAEvaluator(cdi).compute_curve_fitness(data, tok)["curve_fitness"]
    JsonProcessor.save_json({"aoa": score}, d / "aoa_score.json")
    print(f"{m}: aoa = {score:.4f}", flush=True)
PYEOF
echo "SCORING DONE"
