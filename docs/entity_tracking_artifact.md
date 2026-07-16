# Entity tracking under MLM pseudo-likelihood is artifact-prone

During checkpoint-dynamics evaluation we initially observed a striking
vision-init effect on the BabyLM entity-tracking task: a +17-point spike
at ~step 1000 (50k vocabulary, all three encoders). Closer analysis showed
it to be a scoring artifact, not entity-tracking competence:

- **Untrained MLMs score ~42%** on this task (far above the nominal
  chance level), because pseudo-log-likelihood interacts with the answer
  options' length/frequency priors.
- As training progresses this inflated baseline **collapses**; vision-init
  models merely delay the collapse by ~1000 steps.
- **Inverted subtask difficulty** (0-ops scoring *lowest*, 5-ops *highest*,
  at every checkpoint including the final one) confirms the score does not
  track task competence.
- The spike is 50k-only; at 75k/100k the corresponding peaks are +1.0/+1.7.

Consequences we draw:

1. We do not interpret entity-tracking scores of MLM-scored models as
   competence, including our own above-field leaderboard numbers.
2. Leaderboard entity-tracking columns partially separate *scoring-backend
   families* (MLM vs causal) rather than ability; untrained-model
   diagnostics at step 0 are a cheap, effective artifact check we now run
   for every minimal-pair-style evaluation (cf. VP-Swap: step0 = 0.49/0.50,
   clean).
