# augustinian-babylm-training

Training, evaluation, and **visual-embedding extraction** for the Augustinian
BabyLM DeBERTa project. Models follow the recipe in Lukas's
[babylm25](https://github.com/Leukas/babylm25) repo (Table 3 hyperparameters),
using our byte-level BPE tokenizers from the private `augustinian-babylm` HF org.

> **If you only need to run the visual embeddings:** jump to
> [the README's Stage 1 section](the README's Stage 1 section) — it is fully
> self-contained and covers different Snellius setups. The summary below mirrors it.

---

## Repo layout

```
.
├── scripts/
│   ├── extract_region_embeddings.py # STAGE 1: per-row visual embeddings (Ece)
│   ├── build_token_embeddings.py    # STAGE 2: region embeddings -> per-token table
│   ├── train_deberta_babylm.py     # pretrain a DeBERTa MLM
│   └── eval_deberta_babylm.py      # mask-fill samples + pseudo-perplexity
├── slurm/
│   ├── extract_region_embeddings.slurm
│   ├── train.slurm
│   └── eval.slurm
├── requirements.txt
├── the README's Stage 1 section    # step-by-step for the visual part (start here)
└── README.md
```

All Python scripts auto-install their Python deps on first run; **torch is
installed separately** with cluster-specific CUDA wheels (see setup below).

---

## What's already done vs. what's next

- **Done:** three BPE tokenizers (`augustinian-babylm/babylm-bpe-{50k,75k,100k}`)
  and three trained DeBERTa baselines (`augustinian-babylm/deberta-base-{50k,75k,100k}`),
  each with Pythia-style per-step checkpoints on HF branches (see Training & evaluation below).
- **Grounding dataset:** `augustinian-babylm/augustinian_babylm` (HF dataset, private)
  — `annotations.parquet` (image_uid, bbox, text, ...) + `images/` parquet shards.
- **Next (this is what the visual scripts do):** build per-token *visual*
  embedding tables from three vision encoders × three tokenizers, to initialize
  DeBERTa embeddings from vision instead of random Gaussian.

---

# ▶ Visual embeddings — TWO SEPARATE STAGES

We deliberately split this into two stages so they can evolve independently:

> **Stage 1 (vision, GPU) — Ece runs this.** Produce one pooled VISUAL embedding
> per annotation row: the whole-image embedding for sentence/caption rows, the
> bbox-region embedding for description rows. **No tokenizer is involved.** Text
> is saved alongside as metadata only. Output: one 768-d vector per row.
>
> **Stage 2 (tokenizer, CPU) — done later, by us.** Map those row-level
> region/sentence embeddings to a per-TOKEN embedding table for a given
> tokenizer. Averaging is the default, but this is exactly the part we want to
> experiment with, so it is kept separate and cheap (no GPU, no re-encoding).

Keeping them apart means we can try many token-construction strategies on the
SAME Stage 1 vectors without ever re-running the expensive encoders.

---

## STAGE 1 — per-row visual embeddings  (`extract_region_embeddings.py`)

For each unique image: encode once. For each annotation row on it: pool the
region (bbox → patches; whole image when no bbox) → one 768-d vector. Save raw,
**no normalization, no tokenizer**.

Three encoders (all ViT-B, 768-d):

| `ENC` | `ENC_ID` | path |
|---|---|---|
| `dinov3` | `facebook/dinov3-vitb16-pretrain-lvd1689m` | encode → slice patches in bbox → pool |
| `sam` | `facebook/sam-vit-base` | bbox prompt → predicted mask → pool 768-d feats over mask |
| `ibot` | local `.pth` (see Step 2) | timm loads ByteDance ViT-B/16 → slice patches → pool |

Output per encoder, pushed to `augustinian-babylm/region-embeddings/<encoder>/`:
`region_embeddings.parquet` = **every column of the original annotations** (uid,
source, image_uid, bbox, text, category, n_annotations, provenance, ...) for the
rows that were embedded, plus `is_region` and an `embedding` column — i.e. the
annotations table with embeddings appended. A row-aligned `region_embeddings.npy`
([N,768]) is also written for fast array loading, plus `config.json`.

### Step 0 — prerequisites (one time)
1. HF account that is a **member of `augustinian-babylm`** with read+write; make a
   token at huggingface.co → Settings → Access Tokens.
2. Snellius GPU access (partition `gpu_a100` assumed; adjust if different).

### Step 1 — clone (Snellius login node)
```bash
ssh <username>@snellius.surf.nl
cd $HOME
git clone https://github.com/bylinina/augustinian_babylm.git augustinian_babylm     
cd augustinian_babylm
```

### Step 2 — download the iBOT checkpoint (one time, login node)
DINOv3 and SAM download themselves. iBOT does not:
```bash
wget https://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/archive/2022/ibot/vitb_16_pt22k/checkpoint_student.pth -O $HOME/ibot_vitb16_pt22k.pth
ls -lh $HOME/ibot_vitb16_pt22k.pth      # ~327 MB
```
(One line. If unreachable, download on a laptop and `scp` to `$HOME`.)

### Step 3 — HF token
```bash
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxx
echo ${HF_TOKEN:0:5}        # hf_xx
```
Must be exported in the same shell you `sbatch` from. (Or run `hf auth login`
once to cache it permanently.)

### Step 4 — smoke-test each encoder (interactive GPU, ~15 min)
```bash
srun --partition=gpu_a100 --gpus=1 --cpus-per-task=8 --time=00:30:00 --pty bash
cd ~/augustinian_babylm
module load 2024 Python/3.12.3-GCCcore-13.3.0 CUDA/12.6.0
python -m venv $TMPDIR/venv_babylm && source $TMPDIR/venv_babylm/bin/activate
pip install -q --upgrade pip
pip install -q torch --index-url https://download.pytorch.org/whl/cu126
pip install -q -r requirements.txt
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxx
python -c "import torch; print('cuda:', torch.cuda.is_available())"   # expect True

# each on ONE line:
python scripts/extract_region_embeddings.py --encoder_name dinov3 --encoder facebook/dinov3-vitb16-pretrain-lvd1689m --limit_images 100
python scripts/extract_region_embeddings.py --encoder_name sam --encoder facebook/sam-vit-base --limit_images 50
python scripts/extract_region_embeddings.py --encoder_name ibot --encoder $HOME/ibot_vitb16_pt22k.pth --limit_images 100
exit
```
Healthy output: DINOv3 prints `registers=4 prefix=5 hidden=768`; SAM does not
raise the "768-d feature map" error; iBOT prints `loaded N tensors ... hidden=768`
with N in the hundreds; each ends with `Embedded X/N rows`. See Troubleshooting.

### Step 5 — full runs (login node) — one per encoder
```bash
sbatch --export=ALL,HF_TOKEN,ENC=dinov3,ENC_ID=facebook/dinov3-vitb16-pretrain-lvd1689m --job-name=region-dinov3 slurm/extract_region_embeddings.slurm
sbatch --export=ALL,HF_TOKEN,ENC=sam,ENC_ID=facebook/sam-vit-base --job-name=region-sam slurm/extract_region_embeddings.slurm
sbatch --export=ALL,HF_TOKEN,ENC=ibot,ENC_ID=$HOME/ibot_vitb16_pt22k.pth --job-name=region-ibot slurm/extract_region_embeddings.slurm
squeue --me
tail -f logs/regionemb_region-dinov3_*.out
```
Results land at `augustinian-babylm/region-embeddings/<encoder>/`. sbatch jobs are
independent of your SSH connection — you can log off while they run.

**⭐ For Ece: Stage 1 is the whole job. Stop here once all three encoders have
pushed their `region-embeddings/<encoder>/` outputs.**

---

## STAGE 2 — per-token table  (`build_token_embeddings.py`)  [later, no GPU]

Reads Stage 1's per-row embeddings + a tokenizer and aggregates them into a
`[V, 768]` table (default: average each token over the rows whose text contains
it; `--seed_last_subword` attributes a row to its last subword). Then optional
postprocessing (mean-center → L2 → ×0.55 → std≈0.02). Runs on CPU in minutes.

```bash
python scripts/build_token_embeddings.py     --region_repo augustinian-babylm/region-embeddings --encoder_name dinov3     --tokenizer augustinian-babylm/babylm-bpe-75k --tokenizer_tag 75k     --push_to_hub augustinian-babylm/token-embeddings
```
Swap `--aggregator`, `--region_kind {both,region,whole}`, `--seed_last_subword`,
or `--no-postprocess` to experiment. Output:
`augustinian-babylm/token-embeddings/<encoder>/<tag>/` with `E_init.safetensors`
(+`seeded_mask`), `E_raw.safetensors`, `coverage.parquet`.

---

## Setup variations (different Snellius situations)

- **Fresh interactive session each time.** The venv lives on `$TMPDIR` (node-local,
  wiped at session end), so each interactive session rebuilds it (~2–3 min). sbatch
  jobs rebuild it on their compute node automatically.
- **Persistent venv:** create it in `$HOME` (`python -m venv $HOME/venvs/babylm`),
  `source` that, and edit the `VENV=` line in the SLURM files. Costs home quota
  (~few GB); check `myquota`.
- **Different partition/GPU:** change `--partition` (and add it to the sbatch line).
  Check `sinfo` or ask your project admin. `gpu_h100` works if allocated.
- **`module load` name errors:** versions drift. Run `module avail Python` /
  `module avail CUDA` and update the names in the SLURM files.
- **Token via cache vs export:** after `hf auth login` once, you can drop `HF_TOKEN`
  from `--export`; jobs read the cached token.
- **iBOT host unreachable from compute nodes:** handled — Step 2 downloads to `$HOME`
  and we pass the local path, so compute nodes never touch that host.

---

## Troubleshooting

**SAM: "Could not find a 768-d feature map. Available shapes: [...]"** — send the
printed shapes; one-line fix in `SAMBackend._vit_patch_features`.

**iBOT: "Only N tensors matched" / very low embedded %** — checkpoint key/prefix
differs. Send:
```bash
python -c "import torch; sd=torch.load('$HOME/ibot_vitb16_pt22k.pth',map_location='cpu'); print(type(sd)); print(list(sd.keys())[:20])"
```
and try `--ckpt_key student`.

**404 on annotations.parquet** — `--dataset_repo` wrong; correct default is
`augustinian-babylm/augustinian_babylm`.

**401 Unauthorized** — token not set/exported, or not an org member.

**`bash: --encoder: command not found`** — a multi-line paste split the command;
keep each `python ...` on ONE line (or end lines with `\`).

---

# DeBERTa training & evaluation (reference)

Already run for the random-init baselines; documented here for reproducibility
and for the upcoming vision-init runs (same script + an embedding-init flag, TBD).

### Hyperparameters (paper Table 3)

| | |
|---|---|
| architecture | deberta-v3-base config; 768 hidden, 12 layers (`--preset base`) |
| optimizer | AdamW, betas (0.9, 0.95), eps 1e-8 |
| lr | 2e-4 |
| weight decay | 0.01 |
| schedule | cosine, fixed 4000 warmup steps |
| epochs | 10 (~10M words, strict-small) |
| batch | 256 effective, grad_acc 4 (= 64/device) |
| MLM | 15% (80/10/10) |
| context warmup | ctx 64 for epochs 0–4, ctx 128 from epoch 5 |
| tokenizer | required: `--tokenizer augustinian-babylm/babylm-bpe-{50k,75k,100k}` |

### Train (one job per vocab size)
```bash
sbatch --export=ALL,HF_TOKEN,VOCAB=50k  --job-name=deberta-50k  slurm/train.slurm
sbatch --export=ALL,HF_TOKEN,VOCAB=75k  --job-name=deberta-75k  slurm/train.slurm
sbatch --export=ALL,HF_TOKEN,VOCAB=100k --job-name=deberta-100k slurm/train.slurm
```
Smoke-test first: `python scripts/train_deberta_babylm.py --tokenizer augustinian-babylm/babylm-bpe-75k --preset base --debug`

### Evaluate
```bash
sbatch --export=ALL,HF_TOKEN,VOCAB=75k --job-name=eval-75k slurm/eval.slurm
```

### Training-dynamics checkpoints (Pythia-style)
Each run saves checkpoints at steps 0,1,2,4,…,512,1000 then every 1000
(`--dynamics_linear_every 1000`), plus best-by-eval. With `--push_to_hub`, the final
model is on `main` and each intermediate is its own branch:
```python
from transformers import AutoModelForMaskedLM
m = AutoModelForMaskedLM.from_pretrained("augustinian-babylm/deberta-base-50k")                      # final
m = AutoModelForMaskedLM.from_pretrained("augustinian-babylm/deberta-base-50k", revision="step256")  # mid-training
m = AutoModelForMaskedLM.from_pretrained("augustinian-babylm/deberta-base-50k", revision="best")     # best eval
```

### Notes
- **Memory:** base size at grad_acc 4 (64/device) fits A100-80GB. OOM → raise `--grad_acc` to 8.
- **Tokenizer loading:** we use `AutoTokenizer`, not `DebertaV2Tokenizer` (the latter
  can't read our byte-level BPE on transformers 5.x).
- **Train-loss display:** the logged training loss reads higher than eval loss due to
  grad-accumulation logging; judge progress by `eval_loss`. Not a bug.


### Checkpoint evaluation (BabyLM fast zero-shot dynamics)

All training checkpoints of the three baselines are evaluated on the BabyLM 2026
strict-small fast zero-shot tasks (blimp, blimp-supplement, ewok, entity_tracking,
reading), scored as masked LMs (`mlm` backend). The sweep, collector, and plotting
scripts live in [`eval/`](eval/) -- see [`eval/README.md`](eval/README.md) for the
full procedure. Output: `eval/results_dynamics.csv` (long-format: vocab, step,
task, section, item, value) and `eval/plots/` (accuracy vs. training step, per
task, per vocab). Evaluation uses our patched fork of `babylm-eval`
(github.com/bylinina/babylm-eval); the patches let the transformers-4.x eval
pipeline load these transformers-5.x-saved checkpoints.

The older `slurm/eval.slurm` (mask-fill + pseudo-perplexity) is superseded for
benchmark purposes by the `eval/` sweep above; it remains for quick sanity checks.
