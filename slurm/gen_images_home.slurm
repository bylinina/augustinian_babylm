#!/bin/bash
#SBATCH --job-name=gen-images
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=logs/genimg_%j.out
#SBATCH --error=logs/genimg_%j.err
set -euo pipefail
cd $HOME/augustinian_babylm
mkdir -p logs
module purge
module load 2024 Python/3.12.3-GCCcore-13.3.0 CUDA/12.6.0
VENV="$HOME/venv_genimg"
python -m venv "$VENV"; source "$VENV/bin/activate"
pip install -q --upgrade pip
pip install -q torch==2.7.0 --index-url https://download.pytorch.org/whl/cu126
pip install -q diffusers transformers accelerate safetensors pillow
export HF_HOME="$HOME/hf_cache_artfix"
[ -n "${HF_TOKEN:-}" ] && { hf auth login --token "$HF_TOKEN" 2>/dev/null || true; export HF_TOKEN; }
python - << 'PY'
import json, torch
from pathlib import Path
from diffusers import AutoPipelineForText2Image
OUT = Path.home() / "synth_images"
OUT.mkdir(parents=True, exist_ok=True)
pipe = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/sdxl-turbo", torch_dtype=torch.float16).to("cuda")
index = (OUT / "index.jsonl").open("a")
done = {json.loads(l)["file"] for l in (OUT / "index.jsonl").open()} \
    if (OUT / "index.jsonl").exists() else set()
recs = [json.loads(l) for l in open("analysis/out/synth/descriptions.jsonl")]
print(f"{len(recs)} descriptions x 3 images")
for i, r in enumerate(recs):
    for s in range(3):
        fname = f"{r['id']}_v{s}.png"
        if fname in done: continue
        g = torch.Generator("cuda").manual_seed(1000 + s)
        img = pipe(prompt=r["description"][:1500], num_inference_steps=2,
                   guidance_scale=0.0, generator=g).images[0]
        img.save(OUT / fname)
        index.write(json.dumps({"file": fname, "desc_id": r["id"],
                                "matched": r["matched"]}) + "\n")
        index.flush()
    if i % 50 == 0:
        print(f"  {i}/{len(recs)}", flush=True)
print("IMAGES DONE ->", OUT)
PY
