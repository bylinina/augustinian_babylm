# VP-Swap: pair-level re-analysis (chck_100M)

Deltas: accuracy over both items of each pair; [95% CI] from resampling pairs.

## 1. By pair composition (vision − baseline)

Word-level seeded status, as in the paper. Last row: neither noun has any seeded in-sentence token (the only token-clean placebo; tiny).

| pairs | n | s1 | s2 | s3 | mean |
|--|--:|--:|--:|--:|--:|
| both seeded | 2689 | +0.031 [+0.021, +0.041] | +0.030 [+0.020, +0.040] | +0.035 [+0.025, +0.046] | +0.032 |
| one seeded | 864 | +0.001 [-0.017, +0.019] | +0.019 [+0.002, +0.035] | +0.012 [-0.005, +0.028] | +0.010 |
| neither seeded | 155 | -0.016 [-0.052, +0.016] | +0.023 [-0.013, +0.055] | +0.029 [-0.003, +0.061] | +0.012 |
| no seeded token in either noun | 31 | -0.032 [-0.145, +0.065] | +0.113 [-0.016, +0.226] | +0.129 [+0.032, +0.242] | +0.070 |

## 2. Matched frequency: both-seeded vs other pairs within bins

Both-seeded pairs reweighted to the bin distribution of the other (one/neither) pairs; bins with >= 10 pairs of each kind.

| seed | both seeded | one/neither seeded | difference |
|--|--:|--:|--:|
| s1 | +0.016 [+0.002, +0.030] | -0.002 [-0.017, +0.014] | +0.018 [-0.003, +0.038] |
| s2 | +0.010 [-0.006, +0.026] | +0.019 [+0.004, +0.034] | -0.009 [-0.032, +0.013] |
| s3 | +0.009 [-0.006, +0.023] | +0.014 [-0.001, +0.030] | -0.005 [-0.025, +0.016] |

bins used: [0, 1, 2, 3, 4, 5, 6, 7]

## 3. Mixed pairs, item by item (vision − baseline)

Seeded noun correct = top-right cell of Fig. 5; seeded noun swapped in = bottom-left. Knowledge of the seeded noun can only help or do nothing in either item, so a reliably negative bottom-left means the vision model picks the seeded noun even when it is the wrong one. PLL column: the seeded noun's advantage averaged over both contexts; it mixes that preference with knowledge that helps only when the seeded noun fits.

| seed | n pairs | seeded noun correct | seeded noun swapped in | Δ seeded-noun advantage, PLL |
|--|--:|--:|--:|--:|
| s1 | 864 | +0.051 [+0.023, +0.080] | -0.049 [-0.078, -0.022] | +0.330 [+0.220, +0.442] |
| s2 | 864 | +0.037 [+0.010, +0.064] | +0.000 [-0.027, +0.027] | +0.112 [+0.015, +0.211] |
| s3 | 864 | -0.001 [-0.029, +0.030] | +0.024 [-0.005, +0.053] | -0.077 [-0.201, +0.044] |

## 4. Synthetic extension at the pair level (ext − sam)

Treated noun: an in-sentence token whose seed is now >= 50% synthetic regions (from the sam vs sam_ext coverage tables). Clean control: no token of either noun changed.

| pairs | n | s1 | s2 | s3 | mean |
|--|--:|--:|--:|--:|--:|
| >= 1 treated noun | 651 | +0.018 [+0.000, +0.038] | -0.009 [-0.028, +0.008] | +0.009 [-0.007, +0.026] | +0.006 |
| touched, < 50% synthetic | 926 | +0.006 [-0.008, +0.020] | -0.001 [-0.015, +0.011] | -0.002 [-0.015, +0.011] | +0.001 |
| clean control | 2131 | +0.000 [-0.011, +0.012] | +0.003 [-0.008, +0.014] | -0.019 [-0.031, -0.009] | -0.005 |

