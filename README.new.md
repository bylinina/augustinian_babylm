# augustinian-babylm

Does initializing word embeddings from **visual grounding** help a small language
model? This repo pretrains DeBERTa-v3-base models on the BabyLM strict-small
corpus (~10M words) and compares random-initialized baselines against
**vision-initialized** variants whose embeddings are seeded from image features,
across three BPE vocabulary sizes (50k / 75k / 100k) and three vision encoders
(DINOv3, iBOT, SAM). Models follow the [babylm25](https://github.com/Leukas/babylm25)
recipe (Table 3 hyperparameters) with our byte-level BPE tokenizers from the
private `augustinian-babylm` HF org.

The full evaluation, coverage analysis, and reproduce instructions are below;
detailed eval tables live in [`eval/README.md`](eval/README.md) and the
text-vs-image coverage study in [`analysis/README.md`](analysis/README.md).

---

## Results so far

Across three vocabulary sizes × three encoders, scored on the BabyLM fast-eval
suite over the full training trajectory:

- **BLiMP is a wash.** Final-checkpoint best-encoder deltas vs. baseline are small
  and inconsistent in sign (+0.9 at 50k, −0.8 at 75k, +1.7 at 100k). Vision
  initialization does not affect syntactic competence at any vocabulary size.
- **The entity-tracking spike is a 50k phenomenon.** At 50k the mean
  encoder-minus-baseline delta peaks at **+17.4 pts at step 1000** (all three
  encoders). At 75k and 100k the corresponding peaks are +1.0 and +1.7 pts —
  effectively absent. See `eval/plots/entity_overlay.png`.
- **Final gains are small and positive but within noise.** Best-encoder
  entity-tracking finals: +2.5 / +1.7 / +2.3 pts (single-seed; not separable from
  seed variance).
- **EWoK / supplement: small and inconsistent. No encoder dominates** — the
  leading encoder rotates across tasks and vocabularies.

**The effect tracks the seedable fraction.** The share of corpus word *types*
that receive a grounded embedding falls with vocabulary size — 37.7% (50k) →
29.1% (75k) → 23.8% (100k) — while token-level coverage stays ~87% (common words
stay grounded). The entity-tracking spike disappearing as vocabulary grows mirrors
this thinning seedable fraction. Full coverage analysis:
[`analysis/README.md`](analysis/README.md). Full delta tables:
[`eval/README.md`](eval/README.md).

> **In progress:** the full zero-shot battery (BLiMP, Supplement, EWoK, Entity
> Tracking, COMPS, Reading) and GLUE/SuperGLUE fine-tuning over 24 selected
> checkpoints are currently running; results and the official-baseline comparison
> will be added here when complete.

> Single-seed caveat: the 50k spike is robust (all three encoders, two adjacent
> checkpoints); sub-~3-pt differences elsewhere should not be over-interpreted.

---

## Repo layout

```
.
├── scripts/
│   ├── extract_region_embeddings.py  # Stage 1: per-row visual embeddings (GPU)
│   ├── build_token_embeddings.py     # Stage 2: region embeddings -> per-token table (CPU)
│   ├── train_deberta_babylm.py       # pretrain a DeBERTa MLM (+ Pythia checkpoints)
│   └── eval_deberta_babylm.py        # mask-fill + pseudo-perplexity (quick sanity)
├── slurm/
│   ├── extract_region_embeddings.slurm   # Stage 1 (one job per encoder)
│   ├── build_tables.slurm                # Stage 2 (all encoder x vocab tables, CPU)
│   ├── train.slurm                       # train ONE baseline (param: VOCAB)
│   ├── train_visioninit.slurm            # train ONE vision-init model (param: VOCAB, ENCODER)
│   ├── full_zeroshot.slurm               # full zero-shot battery over selected checkpoints
│   ├── glue_finetune.slurm               # GLUE/SuperGLUE fine-tuning array
│   └── eval_launch.slurm                 # regenerate targets + submit eval sweep
├── eval/                             # checkpoint-dynamics + full eval
│   ├── eval_sweep.slurm              # fast zero-shot sweep (one task per checkpoint)
│   ├── list_eval_targets.py          # enumerate (vocab, init, stepN) pairs from HF
│   ├── collect.py                    # results tree -> long-format CSV
│   ├── plot_dynamics.py              # per-vocab dynamics plots
│   ├── plot_entity_overlay.py        # single-panel cross-vocab entity-tracking overlay
│   ├── full_eval_targets.txt         # 24 selected checkpoints (2 per config)
│   ├── glue_params.tsv, glue_targets.txt  # GLUE task recipes + 168 fine-tune targets
│   ├── results_dynamics.csv          # flattened fast-eval results
│   ├── plots*/                       # trajectory figures + entity overlay
│   └── README.md                     # eval procedure + detailed result tables
├── analysis/                         # Stage 1.5: text-vs-image coverage study
│   └── README.md
├── run_75k_100k.sh                   # fire-and-forget: build tables -> train -> eval
├── requirements.txt
└── README.md
```

All Python scripts auto-install their deps on first run; **torch is installed
separately** with cluster-specific CUDA wheels (see Setup below).

---

## Reproduce the pipeline

The pipeline runs in six stages. Prerequisites and Stage 1 setup are detailed;
later stages assume the same Snellius + HF setup.

### Prerequisites (one time)

1. HF account that is a **member of `augustinian-babylm`** (read+write); make a
   token at huggingface.co → Settings → Access Tokens.
2. Snellius GPU access (partition `gpu_a100` assumed; adjust if different).
3. Clone:
   ```bash
   ssh <username>@snellius.surf.nl
   cd $HOME && git clone https://github.com/bylinina/augustinian_babylm.git
   cd augustinian_babylm
   ```
4. HF token — **must be exported in the same shell you `sbatch` from**:
   ```bash
   export HF_TOKEN=hf_xxxxxxxxxxxxxxxxx
   echo ${HF_TOKEN:0:5}        # expect hf_xx
   ```
   (Or `hf auth login` once to cache it. Note: we carry `HF_TOKEN` explicitly via
   `--export=ALL,HF_TOKEN` throughout, since cached tokens have failed silently.)

Inputs already on HF: the three tokenizers
(`augustinian-babylm/babylm-bpe-{50k,75k,100k}`) and the grounding dataset
(`augustinian-babylm/augustinian_babylm`: `annotations.parquet` + `images/`
shards).

### Stage 1 — per-row visual embeddings (GPU)

`extract_region_embeddings.py` encodes each unique image once, then pools the
region (bbox → patches; whole image when no bbox) into one 768-d vector per
annotation row. **No normalization, no tokenizer.** Three encoders (all ViT-B,
768-d), all using identical bbox-patch pooling — "SAM" means SAM's ViT-B image
encoder bbox-pooled (pre-neck 768-d features), **not** SAM segmentation.

One-time: download the iBOT checkpoint (DINOv3/SAM self-download):
```bash
wget https://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/archive/2022/ibot/vitb_16_pt22k/checkpoint_student.pth -O $HOME/ibot_vitb16_pt22k.pth
```
(~327 MB. If unreachable from the cluster, download on a laptop and `scp` to `$HOME`.)

Smoke-test each encoder (interactive GPU, ~15 min) before the full run:
```bash
srun --partition=gpu_a100 --gpus=1 --cpus-per-task=8 --time=00:30:00 --pty bash
cd ~/augustinian_babylm
module load 2024 Python/3.12.3-GCCcore-13.3.0 CUDA/12.6.0
python -m venv $TMPDIR/venv_babylm && source $TMPDIR/venv_babylm/bin/activate
pip install -q --upgrade pip
pip install -q torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu126
pip install -q -r requirements.txt
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxx
# each on ONE line:
python scripts/extract_region_embeddings.py --encoder_name dinov3 --encoder facebook/dinov3-vitb16-pretrain-lvd1689m --limit_images 100
python scripts/extract_region_embeddings.py --encoder_name sam --encoder facebook/sam-vit-base --limit_images 50
python scripts/extract_region_embeddings.py --encoder_name ibot --encoder $HOME/ibot_vitb16_pt22k.pth --limit_images 100
exit
```

Full runs (login node, one job per encoder):
```bash
sbatch --export=ALL,HF_TOKEN,ENC=dinov3,ENC_ID=facebook/dinov3-vitb16-pretrain-lvd1689m --job-name=region-dinov3 slurm/extract_region_embeddings.slurm
sbatch --export=ALL,HF_TOKEN,ENC=sam,ENC_ID=facebook/sam-vit-base --job-name=region-sam slurm/extract_region_embeddings.slurm
sbatch --export=ALL,HF_TOKEN,ENC=ibot,ENC_ID=$HOME/ibot_vitb16_pt22k.pth --job-name=region-ibot slurm/extract_region_embeddings.slurm
```
Output → `augustinian-babylm/region-embeddings/<encoder>/` (`region_embeddings.parquet`
+ `.npy` + `config.json`). Jobs are independent of your SSH session.

### Stage 2 — per-token embedding tables (CPU)

`build_token_embeddings.py` reads Stage 1's per-row embeddings + a tokenizer and
aggregates them into a `[V, 768]` table (default: average each token over the rows
whose text contains it), then postprocesses (mean-center → L2 → ×0.55 → std≈0.02).
CPU, minutes. `build_tables.slurm` runs all encoder × vocab combinations serially
(serial avoids the parallel-download cache corruption we hit):
```bash
sbatch --export=ALL,HF_TOKEN slurm/build_tables.slurm
```
Or one table manually (to experiment with `--aggregator`, `--region_kind`,
`--seed_last_subword`, `--no-postprocess`):
```bash
python scripts/build_token_embeddings.py \
    --region_repo augustinian-babylm/region-embeddings --encoder_name dinov3 \
    --tokenizer augustinian-babylm/babylm-bpe-75k --tokenizer_tag 75k \
    --no-seed_last_subword --scale 0.55 \
    --push_to_hub augustinian-babylm/token-embeddings
```
Output → `augustinian-babylm/token-embeddings/<encoder>/<tag>/`.
**Note:** the published tables use `--no-seed_last_subword` (all-subword
attribution); rebuild with the same flag for comparability.

### Stage 3 — train

**Baselines** (one job per vocab; random init):
```bash
sbatch --export=ALL,HF_TOKEN,VOCAB=50k  --job-name=deberta-50k  slurm/train.slurm
sbatch --export=ALL,HF_TOKEN,VOCAB=75k  --job-name=deberta-75k  slurm/train.slurm
sbatch --export=ALL,HF_TOKEN,VOCAB=100k --job-name=deberta-100k slurm/train.slurm
```

**Vision-init** (one job per vocab × encoder; pulls the Stage 2 table):
```bash
sbatch --export=ALL,HF_TOKEN,VOCAB=75k,ENCODER=dinov3 --job-name=vi-75k-dinov3 slurm/train_visioninit.slurm
# ... repeat for each (VOCAB in 50k/75k/100k) x (ENCODER in dinov3/sam/ibot)
```

Or run the **whole vision-init pipeline unattended** for 75k+100k — builds the six
token tables, trains the six models (each `afterok` the build), then launches the
eval sweep, as one dependency chain:
```bash
bash run_75k_100k.sh      # HF_TOKEN must be live in the shell first
```

Each run saves Pythia-style checkpoints (steps 0,1,2,4,…,512,1000 then every 1000)
plus a `best`-by-eval branch. With `--push_to_hub`, the final model is on `main`,
intermediates are branches:
```python
from transformers import AutoModelForMaskedLM
AutoModelForMaskedLM.from_pretrained("augustinian-babylm/deberta-base-50k")                       # final
AutoModelForMaskedLM.from_pretrained("augustinian-babylm/deberta-base-50k-dinov3", revision="step256")  # mid-training
AutoModelForMaskedLM.from_pretrained("augustinian-babylm/deberta-base-50k", revision="best")      # best eval
```
Vision-init repos are `deberta-base-{vocab}-{encoder}`; baselines are
`deberta-base-{vocab}`.

Key hyperparameters (paper Table 3): deberta-v3-base config (768 hidden, 12
layers), AdamW (0.9, 0.95), lr 2e-4, weight decay 0.01, cosine schedule with 4000
warmup steps, 10 epochs, batch 256 (grad_acc 4 = 64/device), 15% MLM, context
warmup (ctx 64 epochs 0–4, ctx 128 from epoch 5). OOM → raise `--grad_acc` to 8.

### Stage 4 — evaluate

Fast zero-shot dynamics over all checkpoints (the headline result above), and the
full battery + GLUE over selected checkpoints, run through our eval fork
[`bylinina/babylm-eval`](https://github.com/bylinina/babylm-eval) (compat patches
for loading transformers-5.x-saved checkpoints, incl. the tokenizer fallback,
are committed there). Full procedure, the 24-checkpoint selection, and detailed
result tables: [`eval/README.md`](eval/README.md).

### Stage 5 — coverage analysis

`analysis/` characterizes how much of the text corpus is groundable in the
image-text (the seedable-fraction story behind the results). All CPU.
See [`analysis/README.md`](analysis/README.md).

---

## Setup variations (Snellius)

- **Venv lifetime.** The training venv lives on `$TMPDIR` (node-local, wiped at job
  end), so jobs rebuild it (~2–3 min). For a persistent venv, create it in `$HOME`
  and edit the `VENV=` line in the SLURM files (costs home quota; check `myquota`).
- **Different partition/GPU.** Change `--partition` (check `sinfo`); `gpu_h100`
  works if allocated. CPU-only jobs (Stage 2, eval-launch) use `genoa`.
- **`module load` name errors.** Versions drift — run `module avail Python` /
  `module avail CUDA` and update names in the SLURM files.
- **Firewall / access from abroad.** Snellius blocks SSH from non-Dutch-academic
  IPs; use the CUA portal IP whitelist, the doornode jump host, or your
  university VPN (see SURF docs).

## Troubleshooting

- **SAM "Could not find a 768-d feature map"** — send the printed shapes; one-line
  fix in `SAMBackend._vit_patch_features`.
- **iBOT "Only N tensors matched" / low embedded %** — checkpoint key/prefix
  differs; default `--ckpt_key teacher` loads ~124 tensors correctly.
- **401 Unauthorized** — token not exported, or not an org member.
- **`bash: --encoder: command not found`** — a multi-line paste split the command;
  keep each `python ...` on ONE line (or end lines with `\`).
- **`torchvision::nms` / vision import crash** — torch and torchvision must come
  from the *same* cu126 index; pin them together.
- **Tokenizer `TokenizersBackend` not recognized (eval)** — fixed in the
  `bylinina/babylm-eval` fork via `PreTrainedTokenizerFast` fallback.
