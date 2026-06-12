# Eval sweep: BabyLM fast zero-shot over all training checkpoints

Prereqs (one-time): babylm-eval fork cloned at ~/babylm-eval with compat
patches; eval venv at /scratch-shared/$USER/babylm-eval/venv_eval;
eval data downloaded incl. ewok_fast unzipped (password BabyLM2025,
unzip FROM the strict/ root); nltk punkt_tab in ~/nltk_data.

After training completes:
  1. Verify: sacct shows COMPLETED; list_eval_targets.py shows ~37 steps/vocab.
  2. cd ~/babylm-eval/strict && mkdir -p logs
  3. python ~/augustinian_babylm/eval/list_eval_targets.py > eval_targets.txt
  4. N=$(wc -l < eval_targets.txt); sbatch --array=1-${N}%8 \
       ~/augustinian_babylm/eval/eval_sweep.slurm
  5. When done: python ~/augustinian_babylm/eval/collect.py \
       --results_dir results --out results_dynamics.csv
  6. python ~/augustinian_babylm/eval/plot_dynamics.py \
       --csv results_dynamics.csv --outdir plots
  7. Commit results_dynamics.csv + plots/ here; raw results stay on scratch.

Resumable: re-submitting the array skips finished checkpoints
(marker: reading/report.txt). Architecture flag is mlm everywhere.
