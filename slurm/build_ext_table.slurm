#!/bin/bash
#SBATCH --job-name=ext-table
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=logs/exttable_%j.out
#SBATCH --error=logs/exttable_%j.err
# Stage 3+4+5: detect+pool synthetic regions, merge with original sam
# rows, build sam_ext/75k token table, push to HF.
set -euo pipefail
cd $HOME/augustinian_babylm
mkdir -p logs
module purge
module load 2024 Python/3.12.3-GCCcore-13.3.0 CUDA/12.6.0
VENV="${TMPDIR:-/scratch-shared/$USER}/venv_ext"
python -m venv "$VENV"; source "$VENV/bin/activate"
pip install -q --upgrade pip
pip install -q torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu126
pip install -q transformers datasets huggingface_hub numpy pandas pillow safetensors pyarrow tqdm
export HF_HOME="/scratch-shared/$USER/hf_cache"
[ -n "${HF_TOKEN:-}" ] && { hf auth login --token "$HF_TOKEN" 2>/dev/null || true; export HF_TOKEN; }
SCRATCH=/scratch-shared/$USER

echo "=== [1/3] synthetic region extraction (OWLv2 + SAM) ==="
if [ -s $SCRATCH/region_synth/region_embeddings.parquet ]; then
    echo "extraction output exists -> skipping"
else
    python scripts/extract_synthetic_regions.py \
        --images_dir $SCRATCH/synth_images \
        --out_dir $SCRATCH/region_synth
fi
test -s $SCRATCH/region_synth/region_embeddings.parquet || { echo "EXTRACT FAIL"; exit 1; }

echo "=== [2/3] merge with original sam rows ==="
python - << 'PY'
import json, os
from pathlib import Path
import numpy as np, pandas as pd
from huggingface_hub import snapshot_download
scratch = Path("/scratch-shared") / os.environ["USER"]
orig = Path(snapshot_download("augustinian-babylm/region-embeddings",
        repo_type="dataset", allow_patterns=["sam/*"])) / "sam"
po = pd.read_parquet(orig / "region_embeddings.parquet")
vo = np.load(orig / "region_embeddings.npy")
ps = pd.read_parquet(scratch / "region_synth/region_embeddings.parquet")
vs = np.load(scratch / "region_synth/region_embeddings.npy")
assert vo.shape[1] == vs.shape[1] == 768
ps = ps.copy()
ps["is_region"] = True
ps["text_norm"] = ps["text"]
drop = [c for c in ps.columns if c not in po.columns]
ps = ps.drop(columns=drop)
for c in po.columns:
    if c not in ps.columns: ps[c] = None
merged = pd.concat([po, ps[po.columns]], ignore_index=True)
out = scratch / "region_sam_ext" / "sam_ext"
out.mkdir(parents=True, exist_ok=True)
merged.to_parquet(out / "region_embeddings.parquet")
np.save(out / "region_embeddings.npy",
        np.concatenate([vo.astype(np.float16), vs.astype(np.float16)]))
cfg = json.loads((orig / "config.json").read_text())
cfg.update({"extended": True, "n_original": len(po), "n_synthetic": len(ps)})
(out / "config.json").write_text(json.dumps(cfg, indent=2))
# also mirror files one level up in case the builder reads region_dir directly
for f in out.iterdir():
    tgt = out.parent / f.name
    if not tgt.exists(): os.link(f, tgt)
print(f"merged: {len(po)} original + {len(ps)} synthetic rows")
PY

echo "=== [3/3] build sam_ext/75k table (settings cloned from sam/75k) ==="
SETTINGS=$(python - << 'PYS'
import json
from huggingface_hub import hf_hub_download
c = json.load(open(hf_hub_download("augustinian-babylm/token-embeddings",
    "sam/75k/config.json", repo_type="dataset")))
flags = []
flags.append("--no-seed_last_subword" if not c.get("seed_last_subword", True)
             else "--seed_last_subword")
flags.append(f"--region_kind {c.get('region_kind','both')}")
flags.append(f"--scale {c.get('scale',0.55)}")
if not c.get("postprocess", True): flags.append("--no-postprocess")
print(" ".join(flags))
print(f"ORIG_SEEDED={c['n_seeded']}", file=__import__('sys').stderr)
PYS
)
echo "cloned settings: $SETTINGS"
python scripts/build_token_embeddings.py \
    --region_dir $SCRATCH/region_sam_ext \
    --encoder_name sam_ext \
    --tokenizer augustinian-babylm/babylm-bpe-75k \
    --tokenizer_tag 75k \
    $SETTINGS \
    --push_to_hub augustinian-babylm/token-embeddings
python - << 'PYCHECK'
import json
from huggingface_hub import hf_hub_download
o = json.load(open(hf_hub_download("augustinian-babylm/token-embeddings",
    "sam/75k/config.json", repo_type="dataset")))
e = json.load(open(hf_hub_download("augustinian-babylm/token-embeddings",
    "sam_ext/75k/config.json", repo_type="dataset", force_download=True)))
print(f"original seeded: {o['n_seeded']} | ext seeded: {e['n_seeded']}")
assert e["n_seeded"] > o["n_seeded"], "EXT TABLE NOT A SUPERSET -- REFUSING"
print("SUPERSET CHECK PASSED")
PYCHECK
echo "EXT TABLE DONE"
