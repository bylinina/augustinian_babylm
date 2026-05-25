# augustinian-babylm-training

Training & evaluation for the Augustinian BabyLM DeBERTa models. Follows the
recipe in Lukas's [babylm25](https://github.com/Leukas/babylm25) repo, using our
byte-level BPE tokenizers from the `augustinian-babylm` HF org.

## Layout

```
.
├── scripts/
│   ├── train_deberta_babylm.py    # pretrain a DeBERTa MLM
│   └── eval_deberta_babylm.py     # mask-fill samples + pseudo-perplexity
├── slurm/
│   ├── train.slurm                # Snellius batch job for training
│   └── eval.slurm                 # Snellius batch job for eval
├── requirements.txt
└── README.md
```

Both python scripts run **either** as a Colab cell (`main([...])`) **or** from
the CLI / Snellius. torch is installed separately (cluster-specific CUDA wheels);
everything else is in `requirements.txt`.

## Defaults (from Lukas's `train_mask.py`)

| | |
|---|---|
| architecture | deberta-v3-base config; 768 hidden, 12 layers (`--preset base`) |
| optimizer | AdamW, betas (0.9, 0.95), eps 1e-8 |
| lr | **1e-3** (his original was 0.007; see `--lr`) |
| schedule | cosine, 1% warmup |
| epochs | 10 |
| batch | 256, grad_acc 4 (= 64/device, effective batch 256) |
| MLM | 15% (80/10/10) |
| seq-len warmup | len 64 for epochs 0–4, len 256 from epoch 5 |
| tokenizer | **required** — pass `--tokenizer augustinian-babylm/babylm-bpe-{50k,75k,100k}` |

---

## Running on Snellius — step by step

### 1. Get an HF token
On huggingface.co: Settings → Access Tokens → create a token with **write**
access to the `augustinian-babylm` org (needed to read the private tokenizer and
push the trained model). Copy it (`hf_...`).

### 2. Log in to Snellius and clone the repo
```bash
ssh <username>@snellius.surf.nl
cd $HOME
git clone https://github.com/<you>/augustinian-babylm-training.git
cd augustinian-babylm-training
```

### 3. Set your token in the environment
```bash
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxx
```
(Don't hardcode it in the SLURM files — pass it through with `--export`.)

### 4. (Recommended) smoke-test first — 1 epoch, tiny subset
This catches setup/memory problems in ~minutes before you burn GPU hours.
Grab a short interactive GPU session:
```bash
srun --partition=gpu_a100 --gpus=1 --cpus-per-task=8 --time=00:20:00 --pty bash
# inside the allocation:
module load 2024 Python/3.12.3-GCCcore-13.3.0 CUDA/12.6.0
python -m venv $TMPDIR/venv_babylm && source $TMPDIR/venv_babylm/bin/activate
pip install -q --upgrade pip
pip install -q torch --index-url https://download.pytorch.org/whl/cu126
pip install -q -r requirements.txt
huggingface-cli login --token $HF_TOKEN
python scripts/train_deberta_babylm.py \
    --tokenizer augustinian-babylm/babylm-bpe-75k --preset base --debug
exit
```
If that finishes and prints a perplexity, you're good.

### 5. Submit the training jobs — one per vocab size
`train.slurm` is parametrized by `$VOCAB`, so each size runs as an independent
job from the same file. Submit all three:
```bash
sbatch --export=ALL,HF_TOKEN,VOCAB=50k  --job-name=deberta-50k  slurm/train.slurm
sbatch --export=ALL,HF_TOKEN,VOCAB=75k  --job-name=deberta-75k  slurm/train.slurm
sbatch --export=ALL,HF_TOKEN,VOCAB=100k --job-name=deberta-100k slurm/train.slurm
```
Watch them:
```bash
squeue --me                          # all jobs, by name
tail -f logs/train_deberta-75k_<jobid>.out
```
Each is fully independent (its own GPU, its own node-local venv — so each pays
the install time once). Outputs land in
`$HOME/models/deberta-babylm-{50k,75k,100k}-base` and are pushed to
`augustinian-babylm/deberta-base-{50k,75k,100k}` (private).

### 6. Evaluate — same pattern, one per model
```bash
sbatch --export=ALL,HF_TOKEN,VOCAB=50k  --job-name=eval-50k  slurm/eval.slurm
sbatch --export=ALL,HF_TOKEN,VOCAB=75k  --job-name=eval-75k  slurm/eval.slurm
sbatch --export=ALL,HF_TOKEN,VOCAB=100k --job-name=eval-100k slurm/eval.slurm
cat logs/eval_eval-75k_<jobid>.out
```
You'll get mask-fill predictions on sample sentences plus pseudo-perplexity and
masked-token accuracy on the dev set, for each model.

---

## Notes / gotchas
- **`module load` versions drift.** If a module isn't found, run `module avail
  Python` / `module avail CUDA` and update the names in the SLURM files.
- **Memory.** Base size at 256/grad_acc 4 (= 64/device) fits an A100-80GB. If you
  hit OOM, raise `--grad_acc` to 8.
- **lr 1e-3 is a touch hot.** If the loss spikes in the first few hundred steps
  instead of falling, drop to `--lr 5e-4`.
- **Tokenizer loading.** We load with `AutoTokenizer`, not `DebertaV2Tokenizer`,
  because the latter can't read our byte-level BPE on transformers 5.x.
