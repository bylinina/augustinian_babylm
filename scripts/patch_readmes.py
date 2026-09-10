#!/usr/bin/env python3
"""Bring the READMEs in line with the current results. Run once from repo root."""
from pathlib import Path

def patch(path, pairs):
    p = Path(path)
    s = p.read_text(encoding="utf-8")
    orig = s
    for old, new in pairs:
        if old not in s:
            print(f"  [MISS] {path}: {old[:60]!r}")
            continue
        s = s.replace(old, new, 1)
    if s != orig:
        p.write_text(s, encoding="utf-8")
        print(f"  [ok]   {path}")
    else:
        print(f"  [warn] no change: {path}")


print("top-level README:")
patch("README.md", [
    ("on, peaking mid-training (+3.5 pts) and retaining +1.9 at 100M",
     "on, peaking around 10M words (+3.7 pts) and retaining +2.2 at 100M"),
    ("across 3 random seeds** (final delta +1.9 / +2.7 / +3.0; McNemar z = 3.75 /",
     "across 3 random seeds** (final delta +2.2 / +2.7 / +2.9; McNemar z = 4.37 /"),
    ("5.29 / 5.90)", "5.44 / 5.79)"),
    ("noun received a visual seed is strikingly stable across seeds (+0.029 /\n  +0.030 / +0.032)",
     "noun received a visual seed is strikingly stable across seeds (+0.034 /\n  +0.031 / +0.030)"),
    ("did **not** replicate (−0.048 / +0.008 / +0.036 across seeds",
     "did **not** replicate (−0.040 / +0.006 / +0.026 across seeds"),
    ("- The effect appears in 3 of 4 syntactic frames (the short copular frame\n"
     "  reverses; noted, unexplained) and concentrates in mid/high-frequency words —",
     "- The effect holds across all four syntactic frames (attributive +0.023,\n"
     "  existential +0.031, relative +0.027, copular ≈ 0) and concentrates in\n"
     "  mid/high-frequency words —"),
    ("yielding 1,151 newly", "yielding 1,155 newly"),
    ("sam in **3/3 seeds** (+1.6 / +2.0 / +0.7 pts; sam itself sits at +0.1 vs",
     "sam in **3/3 seeds** (+1.2 / +2.0 / +1.0 pts; sam itself sits at −0.2 vs"),
    ("per-word advantage change (r = −0.02)", "per-word advantage change (r = −0.017)"),
    ("visual anchor at 100M — three times the 0.10 floor",
     "visual anchor at 100M (RSA 0.31) — three times the 0.10 floor"),
    ("of typical properties; the copular-frame reversal is unexplained;\n",
     "of typical properties;\n"),
])

print("eval/README.md:")
patch("eval/README.md", [
    ("and our **VP-Swap** visual-property benchmark",
     "and my **VP-Swap** visual-property benchmark"),
    ("Our compat patches", "Compat patches"),
    ("(see docs/entity_tracking_artifact.md for why we always check)",
     "(see docs/entity_tracking_artifact.md for why this check always runs)"),
    ("   (~1h/model; raw JSONLs live on scratch, regenerable with this command).",
     "   (~1h/model). Per-item scores for all 9 VP-Swap models x 20 revisions\n"
     "   are published as a dataset; see the repository README."),
    ("## Dynamics evaluation (secondary)",
     """## C. Synthetic grounding (runbook)

Extends visual coverage to concrete words with no image support. Every
intermediate artifact is committed, so any stage can be run in isolation.

1. **Scene descriptions** (LLM; needs `ANTHROPIC_API_KEY`):
   `python scripts/make_synthetic_descriptions.py`
   -> `analysis/out/synth/descriptions.jsonl` + `coverage_report.json`
   Responses are cached in `cache_descriptions.jsonl`, which is committed:
   rerunning replays the cache and reproduces the descriptions exactly,
   with no API calls.
2. **Images** (SDXL-Turbo, fixed generator seeds, 3 per description):
   `sbatch slurm/gen_images_home.slurm` -> `$HOME/synth_images/` + `index.jsonl`
   (~25 min on one A100; resumable via `index.jsonl`.)
   Twelve sample images are committed under
   `analysis/out/synth/sample_images/`.
3. **Detection + pooling** (OWLv2 at threshold 0.25, then SAM features in
   the detected boxes through the Stage-1 backend):
   `sbatch slurm/synth_regions.slurm` -> `region_embeddings.parquet`
   A committed copy lives at `analysis/out/synth/region_embeddings.parquet`,
   so stage 4 can be run without stages 1-3.
4. **Extended seeding table**: `sbatch slurm/build_ext_table.slurm`
   -> `token-embeddings/sam_ext/75k` on the Hub.
5. **Analysis**: `python eval/analyze_ext.py --synth_parquet <parquet>`
   -> `ext_results.md` + `plots/vpswap_ext_groups.{png,pdf}`

Note: open-vocabulary detection near the score threshold is not perfectly
stable across runs, so the set of grounded words can vary by a few items
between regenerations.

## Dynamics evaluation (secondary)"""),
])

print("eval/vpswap_bb24/README.md:")
patch("eval/vpswap_bb24/README.md", [
    ("""6. Word-boundary validation of swap indices (upstream substring match
   can corrupt swaps, e.g. "tub" inside "bathtub").""",
     """6. Word-boundary validation of swap indices (upstream substring match
   can corrupt swaps, e.g. "tub" inside "bathtub").
7. Article agreement: when a noun is substituted, a preceding *a* / *an*
   is adjusted to agree with the incoming noun, so the distractor never
   differs from the original in grammaticality. Without this, swapping a
   vowel-initial noun under *a* yields an ungrammatical distractor that a
   model can reject on agreement alone, independently of any property
   knowledge. The correct article is taken from the generated sentences
   themselves where available (they are a better authority than a spelling
   rule: *a unicycle*, *a uterus*), falling back to the vowel rule."""),
])
