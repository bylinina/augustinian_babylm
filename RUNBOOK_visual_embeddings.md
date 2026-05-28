# Visual embeddings — runbook

The visual-embedding work is split into **two stages**, documented in full in the
main **[README](README.md)**:

- **Stage 1 (Ece, GPU):** per-row visual embeddings (whole-image + bbox-region),
  no tokenizer. Script `scripts/extract_region_embeddings.py`,
  SLURM `slurm/extract_region_embeddings.slurm`.
- **Stage 2 (later, CPU):** region embeddings → per-token table.
  Script `scripts/build_token_embeddings.py`.

Open the README and start at "▶ Visual embeddings — TWO SEPARATE STAGES".
