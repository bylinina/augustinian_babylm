# Abstract-word grounding: results

## Retention (RSA to visual anchor at 100M words)

| group | n | vision-init | baseline floor |
|--|--:|--:|--:|
| band 1-2 (abstract) | 731 | 0.188 | -0.011 |
| band 2-3 | 2186 | 0.189 | 0.001 |
| band 3-4 | 3123 | 0.131 | -0.004 |
| band 4-5 (concrete) | 5006 | 0.180 | 0.034 |
| function words | 64 | 0.446 | -0.055 |
| concrete controls | 30 | 0.741 | 0.250 |

## BLiMP concreteness gradient: r = +0.20 over 54 phenomena (fast subset)

## Held-out MLM loss by token class

| class | n | s1 | s2 | s3 | mean |
|--|--:|--:|--:|--:|--:|
| function | 1385 | -0.0958 | -0.0330 | -0.1113 | -0.0800 |
| abstract(<2.5) | 1445 | -0.0084 | +0.0481 | +0.0005 | +0.0134 |
| mid(2.5-4) | 1327 | +0.0112 | -0.0370 | -0.0714 | -0.0324 |
| concrete(>=4) | 589 | +0.0010 | -0.0936 | -0.0489 | -0.0472 |
| seeded-other | 2446 | -0.1297 | -0.0942 | -0.1899 | -0.1379 |
| unseeded | 1140 | +0.0051 | +0.0116 | -0.0534 | -0.0122 |

Word lists: `eval/abstract_words/function_words_seeded.csv`, `eval/abstract_words/abstract_band_seeded.csv`.

Conclusions: see `docs/grounding_scope.md`.
