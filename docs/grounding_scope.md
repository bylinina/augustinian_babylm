# Does grounding abstract words help?

Our grounding data does not only cover concrete nouns. Because every
caption and referring expression contains function words and abstract
vocabulary, words like "not", "every", "three", "the", and many abstract
adjectives/adverbs also received visual initializations. This note asks
whether those seeds matter. Scripts: `eval/abstract_diagnostic.py`,
`eval/abstract_diagnostic2.py` (no training required; run against the
published checkpoints).

## What we found

**1. The abstract seeds are real signal, not noise.** One might expect a
word that appears in *every* image context ("the", "not") to get a
washed-out initialization — the average over thousands of unrelated
images. This is not what happens: the centered magnitude of abstract
words' seed vectors matches that of concrete words (means ~25 across all
concreteness bands; curated function words 10.3 vs concrete controls
10.3, with partial washout visible only in the median, 6.1 vs 9.5). The
visual contexts of abstract words are apparently systematic enough to
produce distinctive vectors.

**2. The model keeps them.** At the end of training, the relational
structure among function-word embeddings still reflects the visual
anchor: RSA 0.45 against a baseline floor of ≈0 (the text-only control
shows *no* such structure, so this is retained initialization, not
convergence). Absolute cosine to anchor is even *higher* for function
words (0.44) than for concrete controls (0.34). Whatever the model does
with these seeds, it does not discard them.

**3. But they don't help.** Benchmark phenomena that hinge on abstract
and function words show no benefit from vision-init and a slight cost:
across 3 seeds, BLiMP phenomena whose good/bad sentences differ in
*abstract* words average −0.8 pts (vision − baseline) while phenomena
differing in *concrete* words average +0.8 (correlation of phenomenon
delta with the concreteness of its varying words: r = +0.21; fast-eval
subset, 54 phenomena). EWoK `negation` is flat (−0.4).

**One suggestive exception.** EWoK `number` improves by +5.8 (3-seed
mean; +6.5 in the 9-model cut — the only abstract domain that gains
consistently). Numerals are seeded, and quantity is abstract content
that *is* visually manifest. Small samples; we flag this as speculation
consistent with the pattern, not as a finding.

**4. The objective itself does use them.** Per-token-class validation
MLM loss (`eval/mlm_loss_by_class.py`, identical masks across models):
vision-init predicts masked *function words* better than the baseline in
3/3 seeds (mean −0.080 nats), with unseeded tokens as a clean placebo
(≈0). So the retained structure is functionally load-bearing for the
training objective — the gap is in the *benchmarks*, which do not reward
what the model uses this information for.

## Conclusion (plain terms)

Visual grounding helps a word only if its meaning is the kind of thing
vision can inform. For concrete nouns, the seeds encode visual
properties and the model benefits (see the VP-Swap results). For
abstract and function words, the seeds are equally strong and equally
well preserved — the model faithfully carries visual information about
"every" and "not" through all of training — and the training objective itself measurably uses it (better mask prediction of function words in 3/3 seeds) — but no current benchmark rewards that use. The bottleneck is meaning
type, not seed quality and not retention.

## Caveats

Concreteness-band analyses cover only words with Brysbaert norms (~11.1k
of 21.1k seeded tokens); the curated function-word probe is small (64
words); the BLiMP gradient uses the fast subset; RSA cells use the
main seed pair at the final checkpoint.
