# GPT Pro review: MedTRACE generality and hard-scope ablation

This is a development ablation, not full TIME, full MedTRACE, a blind confirmation set, or V0.2 qualification.

## Direct answers

1. Did paraphrase supervision outperform equal-budget native continuation? **Yes under the preregistered development rule.** A2 minus A1 T2G macro was +0.1250, T1G macro was -0.0156, and A2 lost 0 T0-correct items relative to A1.
2. Did hard-negative scope outperform Original-Q and broad-only Q? **No on the registered hard-FPR criterion at comparable positive coverage.**
3. Primary evidence location: compare per-type activation, forced/gated damage and ON/OFF transitions in `HARD_SCOPE_ABLATION.csv`; this distinguishes representation separation from threshold scale and expert behavior.
4. Single next mechanism: **learn a hard-negative-sensitive routing representation that improves rejection without reducing positive coverage.**

## Evidence map

- `GENERALITY_PAIRED_REPORT.md` and `GENERALITY_PAIRED_DEV16.csv`: A0/A1/A2 T0/T1L/T1G/T2G with paired old-success retention and old-failure recovery.
- `HARD_SCOPE_REPORT.md` and `HARD_SCOPE_ABLATION.csv`: B0/B1/B2 positive, broad, same-question/different-image and same-image/other-fact results.
- `EXISTING_RESULT_ADDENDUM.md`: six exact/Judge disagreements and paired analysis of the earlier scope pilot.
- `DATA_ROLE_COUNTS.json`: actual hard/broad role counts and EqKey coverage.

All 6 exact/Judge disagreements remain unresolved and the old verdict table is unchanged. Raw QA, answers, images, tokens, activations, checkpoints, weights and Judge mapping remain private.
