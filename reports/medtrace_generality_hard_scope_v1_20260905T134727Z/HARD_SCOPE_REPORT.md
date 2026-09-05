# Hard-negative scope ablation

Status: `HARD_SCOPE_EVALUATION_COMPLETE`

| Condition | Subset | n | ON/FPR | Forced correct | Gated correct | Forced damage/Base-correct | Gated damage/Base-correct | Conditional ON damage |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| ORIGINAL_Q_MATCHED_OUTPUT_FIT | positive | 4 | 0/4 (0.0%) | 4/4 | 0/4 | 0/0 | 0/0 | NA |
| ORIGINAL_Q_MATCHED_OUTPUT_FIT | broad | 15 | 0/15 (0.0%) | 11/15 | 12/15 | 1/12 | 0/12 | NA |
| ORIGINAL_Q_MATCHED_OUTPUT_FIT | same_question_different_image | 5 | 0/5 (0.0%) | 3/5 | 2/5 | 0/2 | 0/2 | NA |
| ORIGINAL_Q_MATCHED_OUTPUT_FIT | same_image_other_fact | 12 | 0/12 (0.0%) | 4/12 | 3/12 | 1/3 | 0/3 | NA |
| BROAD_Q_MATCHED_OUTPUT_FIT | positive | 4 | 0/4 (0.0%) | 4/4 | 0/4 | 0/0 | 0/0 | NA |
| BROAD_Q_MATCHED_OUTPUT_FIT | broad | 15 | 0/15 (0.0%) | 11/15 | 12/15 | 1/12 | 0/12 | NA |
| BROAD_Q_MATCHED_OUTPUT_FIT | same_question_different_image | 5 | 0/5 (0.0%) | 3/5 | 2/5 | 0/2 | 0/2 | NA |
| BROAD_Q_MATCHED_OUTPUT_FIT | same_image_other_fact | 12 | 0/12 (0.0%) | 4/12 | 3/12 | 1/3 | 0/3 | NA |
| HARD_MIXED_Q_MATCHED_OUTPUT_FIT | positive | 4 | 3/4 (75.0%) | 4/4 | 3/4 | 0/0 | 0/0 | NA |
| HARD_MIXED_Q_MATCHED_OUTPUT_FIT | broad | 15 | 0/15 (0.0%) | 12/15 | 12/15 | 0/12 | 0/12 | NA |
| HARD_MIXED_Q_MATCHED_OUTPUT_FIT | same_question_different_image | 5 | 2/5 (40.0%) | 3/5 | 2/5 | 0/2 | 0/2 | 0.0% |
| HARD_MIXED_Q_MATCHED_OUTPUT_FIT | same_image_other_fact | 12 | 0/12 (0.0%) | 4/12 | 3/12 | 1/3 | 0/3 | NA |

All conditions use the same four positive fit examples, identical calibration/evaluation panels, the same initial rank-4 step-140 checkpoint, and matched 80-step native-plus-paraphrase output fitting. B1 trains Q on 20 broad negatives; B2 trains Q on four same-question/different-image hard negatives plus the same 16-row broad subset. Thresholds are independently calibrated at empirical calibration FPR=0 without reading evaluation labels or scores.

Registered hard-aware gain decision: `False`; comparable positive coverage: `True`. Pairwise ON/OFF changes are retained in the CSV/private aggregate. Same-image rows share the primary image and are a separate challenge, not image- or patient-disjoint evidence.
