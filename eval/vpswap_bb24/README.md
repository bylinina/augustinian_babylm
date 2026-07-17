# VP-Swap for bb24.train

A Visual Property Swap benchmark constructed from **our training corpus**
(`bb24.train`, ~9.9M words; the custom corpus of our BabyLM 2024
submission, [Edman et al. 2024](https://aclanthology.org/2024.conll-babylm.14/)), following the VP-Swap protocol of
EgoBabyVLM ([facebookresearch/egobabyvlm](https://github.com/facebookresearch/egobabyvlm),
`apps/swapbench/visual_property_swap`) with documented deviations.

Minimal-pair probe of visual property knowledge across four properties:
**color, material, relative_size, shape**. Each pair-file line encodes two
sentences and word positions; at evaluation time each line yields two
minimal pairs (original vs. property-mismatched noun swap), scored by MLM
pseudo-log-likelihood: correct iff PLL(original) > PLL(swapped).

## Files

- `vp_swap_<property>_pairs.txt` — one pair per line:
  `freq_bin|VISUAL|w1|sentence1|idx1|w2|sentence2|idx2`
  (idx = character offset of the word in its sentence)
- `vp_swap_<property>_pairs.meta.jsonl` — line-aligned metadata:
  `{property, bin, frame, w1, freq_w1, seeded_w1, w2, freq_w2, seeded_w2}`
  where `seeded_*` = word received a vision-derived embedding in our
  grounding data (image_freq >= 1), `freq_*` = raw count in bb24.train.

Sizes: color 865, material 964, relative_size 944, shape 935 pairs
(x2 minimal-pair items each; 7,416 items total). Frequency bins follow
LT-Swap edges [1,2,4,...,512]; all 10 bins populated per property.

## Construction

Built by `scripts/build_vpswap.py` (seed 42, pairs_per_bin 200).
Sentence generation: `claude-sonnet-4-6` (temp 0.7); gating and A/B
judging: `claude-haiku-4-5` (temp 0). Funnel:

| stage | count |
|--|--|
| corpus nouns with Brysbaert norms | 11,921 |
| concreteness gate (Conc.M >= 4.0, POS=Noun) | 5,006 |
| inanimacy gate (LLM) | 1,669 (1,306 seeded / 363 unseeded) |
| sampled pairs per property | 2,000 |
| well-formed generations | 1,739–1,864 |
| attribution gate | 1,160–1,259 |
| 4x A/B judge (all-correct required) | 878–992 |
| word-boundary cleanup | **865 / 964 / 944 / 935** |

Sentences rotate over four syntactic **frames** (copular, attributive,
existential, relative clause), balanced at generation; the judge rejects
copular items at a higher rate (see meta for accepted-frame counts).

## Deviations from the upstream protocol

1. Physical-object gate replaced by Brysbaert concreteness >= 4.0 + POS
   (upstream: LLM yes/no). LLM-gate mode available via `--gate llm`.
2. Added inanimacy gate (excludes persons/professions/events, whose
   "typical color" otherwise attaches to clothing/props).
3. Added per-sentence attribution gate (property must be predicated of
   the target noun itself).
4. Added frame rotation (upstream: free generation).
5. Generator/judge LLMs differ from upstream (Llama-3.1-405B).
6. Word-boundary validation of swap indices (upstream substring match
   can corrupt swaps, e.g. "tub" inside "bathtub").

Caveat: as an LLM-generated benchmark, items inherit the generator's
notion of typical properties (see EgoBabyVLM paper, limitations).

## Scoring

`eval/vpswap_score.py` + `eval/eval_vpswap.slurm` (per-item JSONL output;
join meta by line number). Analysis: `eval/analyze_vpswap.py`.
Sanity check: untrained checkpoints (step0) score ~0.50 on this format.
