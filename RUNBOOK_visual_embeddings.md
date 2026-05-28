# Visual embedding extraction — runbook

The full, self-contained step-by-step now lives in the main **[README](README.md)**
under "▶ Visual embedding extraction (main task right now)", including:

- prerequisites (HF org membership + token)
- cloning on Snellius
- downloading the iBOT checkpoint
- per-encoder smoke tests
- the three full `sbatch` runs
- setup variations (persistent vs. node-local venv, partitions, module versions,
  token via cache vs. export)
- troubleshooting (SAM 768-d, iBOT key mismatch, 404, 401, multi-line paste)

Open the README and start at that section. (This file is kept only so old links
don't break.)
