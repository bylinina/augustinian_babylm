# augustinian-babylm-training

Training, evaluation, and **visual-embedding extraction** for the Augustinian
BabyLM DeBERTa project. Models follow the recipe in Lukas's
[babylm25](https://github.com/Leukas/babylm25) repo (Table 3 hyperparameters),
using our byte-level BPE tokenizers from the private `augustinian-babylm` HF org.

> **If you only need to run the visual embeddings:** jump to
> [RUNBOOK_visual_embeddings.md](RUNBOOK_visual_embeddings.md) — it is fully
> self-contained and covers different Snellius setups. The summary below mirrors it.

---

## Repo layout

```
.
├── scripts/
│   ├── build_visual_embeddings.py  # extract visual embedding tables (DINOv3/iBOT/SAM)
│   ├── train_deberta_babylm.py     # pretrain a DeBERTa MLM
│   └── eval_deberta_babylm.py      # mask-fill samples + pseudo-perplexity
├── slurm/
│   ├── build_visual_embeddings.slurm
│   ├── train.slurm
│   └── eval.slurm
├── requirements.txt
├── RUNBOOK_visual_embeddings.md    # step-by-step for the visual part (start here)
└── README.md
```

All Python scripts auto-install their Python deps on first run; **torch is
installed separately** with cluster-specific CUDA wheels (see setup below).

---

## What's already done vs. what's next

- **Done:** three BPE tokenizers (`augustinian-babylm/babylm-bpe-{50k,75k,100k}`)
  and three random-init DeBERTa baselines (`augustinian-babylm/deberta-base-{50k,75k,100k}`).
- **Grounding dataset:** `augustinian-babylm/augustinian_babylm` (HF dataset, private)
  — `annotations.parquet` (image_uid, bbox, text, ...) + `images/` parquet shards.
- **Next (this is what the visual scripts do):** build per-token *visual*
  embedding tables from three vision encoders × three tokenizers, to initialize
  DeBERTa embeddings from vision instead of random Gaussian.

---

# ▶ Visual embedding extraction (main task right now)

Builds, for each vision encoder, a per-token embedding table by: encoding each
image once, pooling the region (bbox → patches; whole image when there's no
bbox), averaging per token, then post-processing (mean-center → L2-normalize →
×0.55, giving std ≈ 0.02 to match transformer init). One run produces all three
tokenizer tables (50k/75k/100k) in a single encode pass.

**Three encoders** (all ViT-B, 768-dim, so no projection needed):

| `ENC` | `ENC_ID` to pass | path |
|---|---|---|
| `dinov3` | `facebook/dinov3-vitb16-pretrain-lvd1689m` | encode → slice patches in bbox → pool |
| `sam` | `facebook/sam-vit-base` | bbox prompt → predicted mask → pool 768-d features over mask |
| `ibot` | a local `.pth` path (see below) | timm loads official ByteDance ViT-B/16 → slice patches → pool |

**Output:** per `(encoder × tokenizer)`, pushed to
`augustinian-babylm/visual-embeddings/<encoder>/<tag>`:
`E_raw.safetensors`, `E_init.safetensors` (+`seeded_mask`), `coverage.parquet`,
`config.json`.

### Step 0 — prerequisites (one time)

You need:
1. An HF account that is a **member of the `augustinian-babylm` org** with at
   least **read** (to read the private dataset/tokenizers) and **write** (to push
   the results). Create a token at huggingface.co → Settings → Access Tokens.
2. Access to Snellius with a GPU allocation (partition `gpu_a100` assumed; adjust
   if your project uses another).

### Step 1 — clone the repo (on Snellius login node)

```bash
ssh <username>@snellius.surf.nl
cd $HOME
git clone <THIS_REPO_URL> augustinian_babylm     # <-- replace with the real URL
cd augustinian_babylm
```

### Step 2 — download the iBOT checkpoint (one time, login node)

DINOv3 and SAM download themselves from HF. iBOT does not — fetch its official
ByteDance ViT-B/16 (ImageNet-22K) checkpoint once to `$HOME`:

```bash
wget https://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/archive/2022/ibot/vitb_16_pt22k/checkpoint_student.pth -O $HOME/ibot_vitb16_pt22k.pth
ls -lh $HOME/ibot_vitb16_pt22k.pth      # ~327 MB
```
(Keep the whole command on ONE line. If that host is unreachable, download on a
laptop and `scp` it to `$HOME/ibot_vitb16_pt22k.pth`.)

### Step 3 — set your HF token

```bash
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxx
echo ${HF_TOKEN:0:5}        # should print hf_xx
```
You must `export` this in the **same shell** you submit jobs from (sbatch copies
it via `--export`). To make it permanent instead, run `hf auth login` once
(writes `~/.cache/huggingface/token`, picked up by all jobs automatically).

### Step 4 — smoke-test each encoder (interactive GPU, ~15 min total)

Strongly recommended before the full runs — validates each encoder on 100 images.

```bash
srun --partition=gpu_a100 --gpus=1 --cpus-per-task=8 --time=00:30:00 --pty bash
# once on the gcn… node:
cd ~/augustinian_babylm
module load 2024 Python/3.12.3-GCCcore-13.3.0 CUDA/12.6.0
python -m venv $TMPDIR/venv_babylm && source $TMPDIR/venv_babylm/bin/activate
pip install -q --upgrade pip
pip install -q torch --index-url https://download.pytorch.org/whl/cu126
pip install -q -r requirements.txt
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxx
python -c "import torch; print('cuda:', torch.cuda.is_available())"   # expect True

# each command on ONE line:
python scripts/build_visual_embeddings.py --encoder_name dinov3 --encoder facebook/dinov3-vitb16-pretrain-lvd1689m --limit_images 100
python scripts/build_visual_embeddings.py --encoder_name sam --encoder facebook/sam-vit-base --limit_images 50
python scripts/build_visual_embeddings.py --encoder_name ibot --encoder $HOME/ibot_vitb16_pt22k.pth --limit_images 100
exit
```
**What healthy output looks like:**
- DINOv3 prints `registers=4 prefix=5 hidden=768`.
- SAM runs without raising "Could not find a 768-d feature map".
- iBOT prints `loaded N tensors ... hidden=768` with N in the hundreds.
- All print per-tokenizer `seeded X/V (Y%)`. A modest % (concrete nouns) is normal.

If SAM or iBOT complain, see [Troubleshooting](#troubleshooting) — both have a
one-line diagnostic to send back.

### Step 5 — full runs (login node)

Each job encodes the whole dataset (~80k images) and produces all three
tokenizer tables. Run all three encoders (they're independent jobs):

```bash
sbatch --export=ALL,HF_TOKEN,ENC=dinov3,ENC_ID=facebook/dinov3-vitb16-pretrain-lvd1689m --job-name=vemb-dinov3 slurm/build_visual_embeddings.slurm
sbatch --export=ALL,HF_TOKEN,ENC=sam,ENC_ID=facebook/sam-vit-base --job-name=vemb-sam slurm/build_visual_embeddings.slurm
sbatch --export=ALL,HF_TOKEN,ENC=ibot,ENC_ID=$HOME/ibot_vitb16_pt22k.pth --job-name=vemb-ibot slurm/build_visual_embeddings.slurm
```
Monitor:
```bash
squeue --me
tail -f logs/visualemb_vemb-dinov3_*.out
```
Results land at `augustinian-babylm/visual-embeddings/<encoder>/<50k|75k|100k>`.
sbatch jobs are unaffected by your SSH connection — you can log off while they run.

---

## Setup variations (different Snellius situations)

- **Fresh interactive session each time.** The venv lives on `$TMPDIR` (node-local,
  wiped when the session ends), so every new interactive session rebuilds it
  (~2–3 min). The sbatch jobs rebuild it on their compute node automatically.
- **Want a persistent venv** (avoid rebuilding): create it in `$HOME` instead —
  `python -m venv $HOME/venvs/babylm` — and `source` that. Edit the SLURM files'
  `VENV=` line to match. Costs home-quota (~few GB for torch); check `myquota`.
- **Different partition/GPU.** If your project isn't on `gpu_a100`, change
  `--partition` (and add it to the `sbatch` line). Check with `sinfo` or ask your
  project admin. `gpu_h100` also works if allocated.
- **`module load` name errors.** Module versions drift between maintenance windows.
  If a load fails, run `module avail Python` / `module avail CUDA` and update the
  three names in `slurm/build_visual_embeddings.slurm` (and train/eval slurm).
- **Token via cache vs. export.** If you ran `hf auth login` once, you can drop
  `HF_TOKEN` from the `--export` list — jobs read the cached token. Either works.
- **iBOT host unreachable from compute nodes.** Already handled — Step 2 downloads
  to `$HOME` and we pass the local path, so compute nodes never hit that host.

---

## Troubleshooting

**SAM: "Could not find a 768-d feature map. Available shapes: [...]"**
The pre-neck feature path differs in this transformers version. Send the printed
shapes; it's a one-line fix in `SAMBackend._vit_patch_features`.

**iBOT: "Only N tensors matched" / very low seeded %**
The checkpoint key/prefix differs. Run and send the output:
```bash
python -c "import torch; sd=torch.load('$HOME/ibot_vitb16_pt22k.pth',map_location='cpu'); print(type(sd)); print(list(sd.keys())[:20])"
```
Also try `--ckpt_key student` (default is `teacher`, though the pt22k backbone
file has no wrapper key so it usually doesn't matter).

**404 on annotations.parquet**
`--dataset_repo` is wrong. The correct one is `augustinian-babylm/augustinian_babylm`
(already the default). Verify you can see it at
huggingface.co/datasets/augustinian-babylm/augustinian_babylm.

**401 Unauthorized**
Token not set/exported, or not a member of the org. Re-check Step 3 and that the
token has org access.

**`bash: --encoder: command not found`**
A multi-line paste split the command. Keep each `python ...` invocation on ONE
line (or end continued lines with `\`).

---

# DeBERTa training & evaluation (reference)

Already run for the random-init baselines; documented here for reproducibility
and for the upcoming vision-init runs (same script + an embedding-init flag, TBD).

### Hyperparameters (paper Table 3)

| | |
|---|---|
| architecture | deberta-v3-base config; 768 hidden, 12 layers (`--preset base`) |
| optimizer | AdamW, betas (0.9, 0.999), eps 1e-8 |
| lr | 2e-4 |
| weight decay | 0.01 |
| schedule | cosine, fixed 4000 warmup steps |
| epochs | 50 |
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
Each run saves checkpoints at steps 0,1,2,4,…,512,1000 then every 5000
(`--dynamics_linear_every`), plus best-by-eval. With `--push_to_hub`, the final
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
