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


## Deep dive: what lacks visual support

`coverage_deep.py` characterizes the ungrounded vocabulary along three axes
(POS via spaCy, concreteness via Brysbaert et al. 2014 norms, corpus frequency);
`extract_zero_support.py` dumps the complete frequency-sorted lists of
units with zero image support. POS tagging used a ~15M-char corpus subsample
(POS distribution is stable on a subsample).

**Concreteness is the clearest signal.** Per-word concreteness correlates with
being grounded at r=0.46, and type coverage rises monotonically across
concreteness bins: 11% (abstract, 1-2) -> 19% -> 39% -> 67% (concrete, 4-5).
A concrete word is ~6x more likely to have visual grounding than an abstract
one. This is the quantitative form of the vision-init premise: vision grounds
concrete vocabulary. Note the effect is a *type* phenomenon -- token coverage
stays ~85-96% across all concreteness bins, because abstract words include some
ultra-frequent items that appear in captions by chance.

**By part of speech**, content classes show the low type coverage that matters:
NOUN 32%, ADJ 26%, VERB 24%, ADV 17%, and proper nouns (PROPN) just 12% -- named
entities are the least grounded. Function-word classes (DET, ADP, PRON, AUX,
CCONJ) show ~100% *token* coverage but this is grounding-by-cooccurrence, not
semantic grounding: "the" appears in nearly every caption without vision
informing its meaning. Type coverage is therefore the honest measure of whether
vision meaningfully informs an embedding; the high aggregate token-coverage
figure (~87%) is inflated by these function words and should not be read as
"87% of the vocabulary is visually grounded."

**By frequency**, coverage falls off steeply for rare words: type coverage by
corpus-frequency band is 3% (1-4 occurrences) -> 14% -> 33% -> 62% -> 90%
(1000+). The frequent tokens that dominate training are seedable; the rare-token
tail is almost entirely un-seeded.

**Per tokenizer**, the fraction of corpus tokens with zero image support grows
with vocab size: 62.3% (50k) / 70.9% (75k) / 76.2% (100k) -- larger vocabularies
add rarer whole-word tokens absent from the (noun-phrase) image-text, so more of
the embedding table is left at random init by Stage 2. Some zero-support tokens
are structural (punctuation, whitespace, BPE artifacts like the space-prefix
marker) that vision should not ground regardless.

**Full lists:** `zero_support_words.csv` and `zero_support_tokens_{50k,75k,100k}.csv`
(frequency-sorted; rerun `extract_zero_support.py --max_image_count N` for
weakly-supported sets). Per-axis tables in `coverage_by_{pos,concreteness,freqband}.csv`
and `deep_summary.json`; plots `coverage_by_*.png`.

### Implication for vision-init
Vision-init can meaningfully initialize the frequent, concrete, content-bearing
slice of the embedding table and leaves the rare/abstract/proper-noun tail at
random init -- more so for larger vocabularies. Whether that un-seeded tail
matters is testable downstream via the checkpoint-dynamics eval.
