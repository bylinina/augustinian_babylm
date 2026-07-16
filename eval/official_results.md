# Vision-init vs baseline: official BabyLM 2026 evaluation

Deltas = encoder model minus same-vocab baseline, `best` revision.

## Per-task deltas (9 encoder x vocab combinations)

| task | mean delta | positive |
|--|--:|--:|
| comps | +1.30 | 9/9 |
| glue | +1.08 | 9/9 |
| ewok | +0.35 | 5/9 |
| blimp | +0.30 | 6/9 |
| entity_tracking | -0.21 | 5/9 |
| supplement | -0.69 | 3/9 |

## comps subtask deltas

| subtask | mean delta | positive |
|--|--:|--:|
| wugs | +3.65 | 9/9 |
| base | +1.37 | 9/9 |
| wugs_dist_before | +0.15 | 4/9 |
| wugs_dist_in_between | +0.04 | 4/9 |

## ewok subtask deltas

| subtask | mean delta | positive |
|--|--:|--:|
| number | +6.53 | 6/9 |
| active-passive | +4.81 | 7/9 |
| quantitative-properties | +3.08 | 8/9 |
| material-dynamics | +2.51 | 6/9 |
| material | +2.15 | 6/9 |
| direct | +1.02 | 8/9 |
| negation | +0.76 | 5/9 |
| concept swap | +0.59 | 6/9 |
| physical-dynamics | +0.47 | 4/9 |
| physical-relations | +0.45 | 4/9 |
| social-relations | +0.32 | 5/9 |
| antonym | +0.26 | 7/9 |
| agent-properties | +0.14 | 5/9 |
| physical-interactions | +0.10 | 5/9 |
| variable swap | +0.06 | 5/9 |
| indirect | +0.01 | 5/9 |
| material-properties | +0.00 | 4/9 |
| other | -0.16 | 3/9 |
| variable_swap | -0.74 | 2/9 |
| social-interactions | -0.76 | 3/9 |
| spatial-relations | -1.02 | 1/9 |
| social-properties | -1.42 | 2/9 |
| game | -4.44 | 0/9 |
