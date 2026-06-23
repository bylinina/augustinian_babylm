# Evaluation: BabyLM zero-shot dynamics, full battery, and GLUE

Three evaluation layers, all scored with the `mlm` backend (masked-LM
pseudo-log-likelihood; these are masked LMs, not causal):

1. **Fast zero-shot sweep** over *every* training checkpoint of all 12 models
   (baselines `deberta-base-{50k,75k,100k}` + vision-init
   `deberta-base-{50k,75k,100k}-{dinov3,sam,ibot}`), producing the
   training-dynamics CSV and plots. This is the source of the headline result in
   the [root README](../README.md).
2. **Full zero-shot battery** (BLiMP, Supplement, EWoK, Entity Tracking, COMPS,
   Reading) over 24 *selected* checkpoints (2 per model).
3. **GLUE/SuperGLUE fine-tuning** over those checkpoints.

All three run through our eval fork
[`bylinina/babylm-eval`](https://github.com/bylinina/babylm-eval) (compat patches
below).

## One-time setup

1. **Eval pipeline (forked).** A fork of `babylm-org/babylm-eval` with
   compatibility patches (see "Compat patches"). Clone and build the eval venv
   (SEPARATE from training: pins transformers 4.51.3 / torch 2.7.0):

       git clone https://github.com/bylinina/babylm-eval.git ~/babylm-eval
       module load 2024 Python/3.12.3-GCCcore-13.3.0 CUDA/12.6.0
       EVALENV=/scratch-shared/$USER/babylm-eval/venv_eval
       python -m venv $EVALENV && source $EVALENV/bin/activate
       pip install -q --upgrade pip
       cd ~/babylm-eval/strict
       pip install -q torch==2.7.0 --index-url https://download.pytorch.org/whl/cu126
       pip install -q -r requirements.txt

2. **Eval data.** From `~/babylm-eval/strict`, venv active, HF auth done
   (`hf auth login`; member of `augustinian-babylm`):

       python -m scripts.download_evals
       python -c "import nltk; nltk.download('punkt_tab')"
       # accept EWoK terms in a browser first:
       #   huggingface.co/datasets/ewok-core/ewok-core-1.0
       python -m evaluation_pipeline.ewok.dl_and_filter
       # unzip ewok_fast FROM THE strict/ ROOT (zip carries full internal paths):
       unzip -o -P BabyLM2025 evaluation_data/fast_eval/ewok_fast.zip

## 1. Fast zero-shot sweep (all checkpoints)

`list_eval_targets.py` enumerates `(repo, stepN)` pairs across all 12 models from
their HF branches; `eval_sweep.slurm` is a job array, one task per checkpoint
(`%8` caps concurrency). The array loads its own modules/venv — no setup in the
submitting shell, no GPU node to submit:

    cd ~/babylm-eval/strict && mkdir -p logs
    python ~/augustinian_babylm/eval/list_eval_targets.py > eval_targets.txt
    N=$(wc -l < eval_targets.txt)
    sbatch --array=1-${N}%8 --export=ALL,HF_TOKEN ~/augustinian_babylm/eval/eval_sweep.slurm

Resumable — re-submitting skips checkpoints whose results exist (marker:
`.../reading/report.txt`). Mop up failures:

    find results -name report.txt -path "*reading*" | wc -l   # vs wc -l eval_targets.txt
    sacct --starttime <DATE> --format=JobID%25,State | grep eval | grep -v COMPLETED

**Collect + plot** (login node, no GPU; needs `matplotlib`/`pandas` in the venv):

    cd ~/babylm-eval/strict
    python ~/augustinian_babylm/eval/collect.py --results_dir results --out results_dynamics.csv
    python ~/augustinian_babylm/eval/plot_dynamics.py --csv results_dynamics.csv --outdir plots --init_vocab 50k
    # cross-vocab entity-tracking overlay (single panel, all three vocabs):
    python ~/augustinian_babylm/eval/plot_entity_overlay.py --csv results_dynamics.csv --out plots/entity_overlay.png
    cp results_dynamics.csv ~/augustinian_babylm/eval/ && cp -r plots ~/augustinian_babylm/eval/

`results_dynamics.csv` columns: **vocab, init, step, task, section, item, value**
(`init` = baseline | dinov3 | sam | ibot; `section` = average | field | uid |
linguistics_term | score | correlation). `collect.py` finds reports by globbing on
the `zero_shot` path anchor, so it is robust to where `--results_dir` points. Raw
per-checkpoint outputs stay under `~/babylm-eval/strict/results/` (not committed).

## 2. Full zero-shot battery (selected checkpoints)

Two checkpoints are selected per model: the **`best`** branch (best-by-eval-loss,
committed during training) and the **best step by 4-task fast-eval composite**
(BLiMP/Entity/EWoK/Supplement averaged). The 24 targets are in
`full_eval_targets.txt` (`repo <TAB> branch <TAB> selection`). The full battery
adds COMPS and the reading correlation on top of the fast tasks, against
unsubsampled (`*_filtered`) data:

    cd ~/babylm-eval/strict
    N=$(wc -l < ~/augustinian_babylm/eval/full_eval_targets.txt)
    sbatch --array=1-${N}%2 --export=ALL,HF_TOKEN ~/augustinian_babylm/slurm/full_zeroshot.slurm

`%2` throttle avoids the parallel-download HF cache corruption we hit at higher
concurrency. Resumable (marker: `.../zero_shot/mlm/comps`). Output writes to a
`{task}/{dataset}` tree, e.g. `blimp/blimp_filtered`, `blimp/supplement_filtered`.

## 3. GLUE / SuperGLUE fine-tuning (selected checkpoints)

The official BabyLM finetuning subset — 7 tasks with per-task recipes in
`glue_params.tsv` (boolq, multirc, rte, wsc, mrpc, qqp, mnli; wsc runs 30 epochs,
mnli is 3-label, boolq/multirc use the larger batch). `glue_targets.txt` crosses
the 24 checkpoints × 7 tasks = 168 fine-tuning runs, one per array task:

    cd ~/babylm-eval/strict
    N=$(wc -l < ~/augustinian_babylm/eval/glue_targets.txt)   # 168
    sbatch --array=1-${N}%4 --export=ALL,HF_TOKEN ~/augustinian_babylm/slurm/glue_finetune.slurm

Each task fine-tunes one checkpoint on one GLUE task and writes scores (no model
weights saved). Resumable (marker: `.../finetune/<task>/predictions.json`).
Single seed (42); against the official baseline's 10-seed means, treat these as
point estimates. Heavy tasks (wsc/mnli/qqp) approach the 4h walltime — if any
TIMEOUT, re-submit the same array (resumes) or raise `--time`.

Baseline comparison: we do **not** re-fine-tune the official baseline (recipe
mismatch risk); its published GLUE numbers are cited as a reference row. The
controlled contrast is our own random-init DeBERTa vs. vision-init DeBERTa (same
architecture, init the only difference); the official GPT-BERT baseline is the
external yardstick.

## Compat patches (in the babylm-eval fork)

Checkpoints saved by transformers 5.x record `tokenizer_class "TokenizersBackend"`,
unresolvable by the eval pipeline's transformers 4.51.3. The fork carries, by
commit:

- **`1f6e578`** — zero-shot tokenizer loading falls back to
  `PreTrainedTokenizerFast` (loads `tokenizer.json` directly) at the sentence and
  reading call sites.
- **`df754b1`** — finetune trainer loads a tokenizer via a 3-level fallback
  (`AutoProcessor` → `AutoTokenizer` → `PreTrainedTokenizerFast`), needed because
  `AutoProcessor` assumes a multimodal processor that text-only DeBERTa repos lack;
  and `eval_zero_shot.sh` gains a `--revision_name` passthrough so non-`main`
  branches can be evaluated.

Anyone on transformers 4.x loading these (transformers-5.x-saved) repos needs the
same fallbacks.

## Notes / gotchas

- **Excluded from the fast sweep:** `main`, `best` (duplicates a stepN), and the
  step0/step1 branches of the 50k repo (corrupt safetensors on HF; scientifically
  empty as init/one-step checkpoints, so curves start at step2).
- **MLM scoring** is ~10–20× heavier per sentence than causal (masks each
  position), so per-checkpoint fast eval runs ~15 min, full-battery ~45 min. Array
  `--time` has headroom; raise it if tasks TIMEOUT.
- `plots/overview.png` and `plots/entity_overlay.png` are the headline figures.

---

## Results: fast-eval, vision-init vs baseline (50k detail)

The cross-vocabulary summary is in the [root README](../README.md). Below is the
detailed 50k delta table. Four 50k models, identical training except embedding
initialization: a random-init baseline and three vision-init runs seeding the same
~37% of token rows from DINOv3/SAM/iBOT region embeddings (same seeded mask; only
the seeded *values* differ).

![vision-init vs baseline](plots/visioninit_overview.png)

Deltas are vision-init minus baseline (accuracy points), averaged over checkpoints
in each training phase.

| task | encoder | early ≤1k | mid 1k–8k | late >8k | peak (step) | final spread* |
|------|---------|----------:|----------:|---------:|------------:|---------------:|
| blimp | dinov3 | +0.4 | -0.0 | +0.2 | +2.3 (s0) | 0.7 pts |
| | sam | -0.1 | -0.4 | +0.4 | +2.1 (s256) | |
| | ibot | +0.3 | -0.5 | +0.2 | +2.6 (s16) | |
| supplement | dinov3 | -0.8 | +0.5 | -3.3 | +6.0 (s64) | 4.8 pts |
| | sam | +1.9 | -0.3 | +0.8 | +6.8 (s16) | |
| | ibot | -2.8 | +0.5 | -0.1 | +5.6 (s128) | |
| ewok | dinov3 | -1.6 | +0.6 | +0.4 | +1.9 (s1000) | 3.2 pts |
| | sam | -3.1 | +0.2 | +1.9 | +1.3 (s2000) | |
| | ibot | -1.9 | -0.4 | +0.3 | +0.6 (s1000) | |
| entity_tracking | dinov3 | +1.7 | +0.2 | +4.8 | **+16.9 (s1000)** | 2.5 pts |
| | sam | +1.6 | +2.1 | +1.6 | **+16.7 (s1000)** | |
| | ibot | +1.8 | +1.6 | -0.5 | **+18.5 (s1000)** | |

*final spread = range of final accuracy across the three encoders (a noise proxy).

**BLiMP — clean wash.** Phase-mean deltas are within ±0.5 pts everywhere; the only
positive peaks (+2 to +3) occur at step 0–256, i.e. at initialization before
training acts. Finals: baseline 68.4 vs. 68.5–69.3 (spread 0.7 pts).

**Entity tracking — a sharp spike at step 1000, not a sustained phase.** All three
encoders jump to ~+17 pts over baseline at exactly step 1000 (baseline ~25%,
vision ~42–44%), but early *phase* means are only ~+1.7, so the effect is localized
to that checkpoint, not a broad early advantage. It does not translate into a
robust endpoint: late-phase means are +4.8 / +1.6 / -0.5 and the final
cross-encoder spread (2.5 pts) is comparable to any residual gain.

**Supplement / EWoK — small and inconsistent.** No consistent early effect
(supplement early delta is +1.9 for SAM but -2.8 for iBOT); final differences sit
within the cross-encoder spread (4.8 / 3.2 pts). On EWoK the baseline itself is at
chance (~50).

**No encoder dominates.** On every task the final cross-encoder spread is
comparable to or larger than any encoder's gain over baseline. Since all three
share the identical seeded mask, differences reflect feature quality, not coverage.

**Interpretation.** Vision-init clearly moves only entity tracking — the most
state/semantics-oriented task — and only transiently, while purely syntactic BLiMP
is untouched. Consistent with the Stage 1.5 coverage finding that vision grounds
concrete, content-bearing vocabulary, not syntactic structure. Single seed per run;
differences under ~2–3 pts should not be over-interpreted. The cross-vocabulary
extension (the spike is 50k-only and tracks the seedable fraction) is in the
[root README](../README.md); coverage detail in
[`analysis/README.md`](../analysis/README.md).

Per-task figures: `plots/visioninit_{blimp,supplement,ewok,entity_tracking}_fast.png`.
Data: `results_dynamics.csv`.

---

## Results: full battery + GLUE

> **In progress.** The full zero-shot battery (24 checkpoints × 6 tasks) and GLUE
> fine-tuning (24 × 7 tasks) are running. Result tables — full-battery accuracy by
> checkpoint, GLUE scores, the random-init vs. vision-init contrast, and the
> official-baseline reference row — will be added here when the arrays complete and
> the comparison is collected.
