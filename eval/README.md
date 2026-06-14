# Checkpoint evaluation: BabyLM fast zero-shot over training dynamics

Evaluates every training checkpoint of the three DeBERTa baselines
(augustinian-babylm/deberta-base-{50k,75k,100k}) on the BabyLM 2026
strict-small fast zero-shot tasks, producing a long-format CSV and dynamics
plots (accuracy vs. training step, per task, per vocab).

Tasks (fast set): blimp, blimp-supplement, ewok, entity_tracking, reading.
Backend flag is "mlm" everywhere (masked-LM pseudo-log-likelihood scoring) --
the models are masked LMs, not causal.

## One-time setup

1. Eval pipeline (forked). We use a fork of babylm-org/babylm-eval with
   compatibility patches (see "Compat patches" below). Clone it and build the
   eval venv (SEPARATE from training: pins transformers 4.51.3 / torch 2.7.0):

       git clone https://github.com/bylinina/babylm-eval.git ~/babylm-eval
       module load 2024 Python/3.12.3-GCCcore-13.3.0 CUDA/12.6.0
       EVALENV=/scratch-shared/$USER/babylm-eval/venv_eval
       python -m venv $EVALENV && source $EVALENV/bin/activate
       pip install -q --upgrade pip
       cd ~/babylm-eval/strict
       pip install -q torch==2.7.0 --index-url https://download.pytorch.org/whl/cu126
       pip install -q -r requirements.txt

2. Eval data. From ~/babylm-eval/strict, venv active, HF auth done
   (hf auth login; member of augustinian-babylm):

       python -m scripts.download_evals
       python -c "import nltk; nltk.download('punkt_tab')"
       # accept EWoK terms in a browser first:
       #   huggingface.co/datasets/ewok-core/ewok-core-1.0
       python -m evaluation_pipeline.ewok.dl_and_filter
       # unzip ewok_fast FROM THE strict/ ROOT (zip carries full internal paths):
       unzip -o -P BabyLM2025 evaluation_data/fast_eval/ewok_fast.zip
       ls evaluation_data/fast_eval/ewok_fast/

## Running the sweep (after training completes)

1. Verify training finished: sacct shows COMPLETED, and each repo has the full
   stepN branch set (~35 branches, up to ~step25000).
2. Generate the target list and submit the array (from ~/babylm-eval/strict).
   One array task = one (vocab, stepN) checkpoint; %8 caps concurrency. The
   array script loads its own modules/venv internally -- no setup needed in the
   submitting shell, no GPU node required to submit:

       mkdir -p logs
       python ~/augustinian_babylm/eval/list_eval_targets.py > eval_targets.txt
       N=$(wc -l < eval_targets.txt)
       sbatch --array=1-${N}%8 ~/augustinian_babylm/eval/eval_sweep.slurm

3. Resumable. Re-submitting the same command skips checkpoints whose results
   already exist (marker: .../reading/report.txt). Check and mop up failures:

       find results -name report.txt -path "*reading*" | wc -l   # vs wc -l eval_targets.txt
       sacct --starttime <YYYY-MM-DD> --format=JobID%25,State | grep eval | grep -v COMPLETED

## Collecting and plotting (login node, no GPU)

       pip install --user matplotlib pandas
       cd ~/babylm-eval/strict
       python ~/augustinian_babylm/eval/collect.py --results_dir results --out results_dynamics.csv
       python ~/augustinian_babylm/eval/plot_dynamics.py --csv results_dynamics.csv --outdir plots
       cp results_dynamics.csv ~/augustinian_babylm/eval/
       cp -r plots ~/augustinian_babylm/eval/

results_dynamics.csv columns: vocab, step, task, section, item, value
(section = average | field | uid | linguistics_term | score | correlation).
Raw per-checkpoint outputs stay under ~/babylm-eval/strict/results/ (not committed).

## Compat patches (in the babylm-eval fork)

Checkpoints saved by transformers 5.x record tokenizer_class "TokenizersBackend",
which the eval pipeline's transformers 4.51.3 cannot resolve. Three call sites
fall back to PreTrainedTokenizerFast (loads tokenizer.json directly):
evaluation_pipeline/__init__.py, sentence_zero_shot/dataset.py, reading/run.py.
Anyone on transformers 4.x loading these repos needs the same fallback.

## Notes / gotchas

- Excluded from the sweep: main, best (best duplicates a stepN), and the
  step0/step1 branches of the 50k repo (their safetensors were corrupt on HF;
  scientifically empty as init/one-step checkpoints, so curves start at step2).
- MLM scoring is ~10-20x heavier per sentence than causal (masks each token
  position), so per-checkpoint eval runs ~15 min; the array --time has headroom.
  If tasks TIMEOUT, raise --time in eval_sweep.slurm.
- plots/overview.png is the headline figure: accuracy vs step, per task, per vocab.



## Results: vision-init vs baseline (50k)

Four 50k models, identical training except embedding initialization: a random-init
baseline and three vision-init runs seeding the same 37% of token rows from DINOv3,
SAM, and iBOT region embeddings (same seeded mask; only the seeded *values* differ).

![vision-init vs baseline](plots/visioninit_overview.png)

**Summary: no robust effect on the language tasks; a localized entity-tracking
spike at step ~1000; no encoder dominates.** Deltas below are vision-init minus
baseline (accuracy points), averaged over checkpoints in each training phase.

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

**BLiMP -- clean wash.** Phase-mean deltas are within ±0.5 pts everywhere; the only
positive peaks (+2 to +3) occur at step 0–256, i.e. at initialization before
training acts. Finals: baseline 68.4 vs. 68.5–69.3 (spread 0.7 pts). Vision-init
does not change syntactic learning.

**Entity tracking -- a sharp spike at step 1000, not a sustained phase.** All three
encoders jump to ~+17 pts over baseline at exactly step 1000 (baseline ~25%,
vision ~42–44%), but the early *phase* means are only ~+1.7, so the effect is
localized to that checkpoint rather than a broad early advantage. It does not
translate into a robust endpoint: late-phase means are +4.8 / +1.6 / -0.5 and the
final cross-encoder spread (2.5 pts) is comparable to any residual gain. Entity
tracking was also the noisiest task in the baseline study, so part of the spike may
be measurement variance on a single seed.

**Supplement / EWoK -- small and inconsistent.** No consistent early effect (e.g.
supplement early delta is +1.9 for SAM but -2.8 for iBOT), and final differences
sit within the cross-encoder spread (4.8 / 3.2 pts). SAM ends marginally highest on
both, but on EWoK the baseline itself is at chance (~50).

**No encoder dominates.** On every task the final cross-encoder spread is comparable
to or larger than any encoder's gain over baseline, so DINOv3/SAM/iBOT differences
are within noise. Since all three share the identical seeded mask, any differences
reflect feature quality, not coverage.

**Interpretation.** The one place vision-init clearly moves the curve is entity
tracking -- the most state/semantics-oriented task -- and only transiently, while
purely syntactic BLiMP is untouched. This is consistent with the Stage 1.5 coverage
finding that vision grounds concrete, content-bearing vocabulary rather than
syntactic structure. Caveat: single seed per run; differences under ~2–3 pts (i.e.
all language-task effects) should not be over-interpreted. The entity-tracking
step-1000 spike is the largest signal but is localized and partly noise-prone --
multiple seeds would be needed to confirm it.

Per-task figures: `plots/visioninit_{blimp,supplement,ewok,entity_tracking}_fast.png`.
Data: `results_dynamics.csv` (vocab, init, step, task, section, item, value).
