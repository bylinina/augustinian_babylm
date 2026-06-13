# Stage 1.5: text-vs-image coverage analysis

How much of the text-only BabyLM training corpus (`bb24.train`, ~9.9M word
tokens) is "grounded" in the image-text (the 563k annotation captions, ~2M
tokens, region descriptions + whole-image sentences pooled). This bounds how
much of a model's embedding table can be vision-initialized in Stage 2: a
word/token absent from the image-text cannot be seeded from vision.

## Scripts (all CPU, no GPU/torch)
- `word_coverage.py`   -- word level (lowercase, strip outer punctuation, whitespace split; no lemmatization).
- `token_coverage.py`  -- per tokenizer (50k/75k/100k), using the `tokenizers` library directly.
- `plot_coverage.py`   -- coverage bars + frequency scatters (needs matplotlib + pandas).

Outputs in `out/`: per-word and per-token CSVs (`*_count` columns),
`*_summary.json`, `divergence_report.txt`, and the plots.

## Key findings

**Type vs token coverage diverge sharply.** Few distinct words are grounded,
but the grounded ones are the common ones, so they cover most running text.
Word level: ~10% of corpus word *types* appear in image-text, but those account
for ~85% of corpus word *occurrences*.

**Subword tokenization raises coverage** (shared subwords are more groundable
than whole words). At image-frequency >=1:

| units      | type coverage | token coverage |
|------------|--------------:|---------------:|
| words      | 10.1%         | 84.9%          |
| 50k tokens | 37.7%         | 88.0%          |
| 75k tokens | 29.1%         | 87.2%          |
| 100k tokens| 23.8%         | 86.9%          |

**Larger vocab lowers type coverage, not token coverage.** As vocab grows
(50k->75k->100k) the seedable *fraction of the table* falls (37.7%->29.1%->
23.8%) -- the extra vocabulary is rarer whole-word tokens absent from the
noun-phrase image-text -- but token coverage stays ~87%, because high-frequency
tokens are grounded regardless of vocab size. Implication: larger-vocab models
leave more of `E_init` un-seeded (random), but the tokens that fire often in
training are seeded in all three.

**What is ungrounded.** The corpus words with zero image support are verbs
(known, asked, told, gave), discourse/abstract terms (however, often, history,
example), temporal/numeric (days, february, percent, million), and proper/domain
nouns. Image captions ground concrete objects and visual attributes, not
predicates, abstractions, or discourse structure (see `divergence_report.txt`).

## Caveats
- "Covered" = appears >=1 time; coverage is reported at >=1/>=5/>=20 thresholds
  because a single occurrence is weak grounding. The threshold choice matters.
- Image-text frequencies count annotation occurrences (captions are duplicated
  across regions), so they reflect annotation density, not concept diversity;
  the ~25k distinct image word-types is the honest breadth measure.
- Sources differ ~5x in size; the scatter uses per-million rates, not raw counts.
