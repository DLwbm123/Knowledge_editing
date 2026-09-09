# MedTRACE Stage6 fixed threshold transfer

Status: COMPUTE_COMPLETE

## Direct conclusion

The fixed old16 threshold was actually used and all eight deterministic single/bank natural ON/OFF replays passed. Both writers retain all 32 native, 128 continuity-positive and eight new existing-source alternative successes in bank32. Negative activation falls from 100% to 0% on both the old U and new H/U panels. This is a shared rejection-to-Base effect, not a W0-exclusive writer gain.

The cost is material: on the old 220-observation U panel, RC prevents no damage but discards 95 W0 corrections and 63 BE corrections. On the new 116-observation U panel, W0 avoids 10 damages but loses 15 corrections (net -5 correct); BE avoids 40 damages but loses 10 corrections (net +30). On the three H observations, each writer avoids two damages and loses no correction. These are raw paired observation counts, distinct from edit-macro percentages below.

Therefore the threshold transfers as a conservative intervention filter while preserving the tested positives, but it does not uniformly improve full-source accuracy. W0+fixed RC may be retained as a bounded practical candidate with BE+the same RC as the reference, not declared generally superior or clinically safe. No threshold retuning follows this evaluation. Hard rejection remains weakly supported: H covers only three edits/two source images, with prior source-image exposure disclosed. No independent visual-generality probe was available; broader hard-input confirmation remains unmeasured.

Compute closed in 747.7 seconds (12.46 minutes). W0, BE, Judge preparation, Judge and report exit codes were all zero. Judge required/scored=538/538, reused=467, new=71, missing=0. No writer/A2 training occurred. Publication status is tracked separately from this compute snapshot.

Unique old16 kappa = 0.7696741135364367. No new calibration and no writer training.
Stage5 New32 is a follow-up transfer evaluation, not an untouched blind test. R0 rankings are unchanged; rejection returns the frozen Base.
New sidecar reuses the cleared 340-row/55-image source pool. Prior source-image fit exposure is disclosed; new QA inputs do not alter old roles. Clinical review is not claimed. Official image generality is NA.
W1 is a single-edit appendix only. Missing H is not a passing hard-rejection result. All-OFF is Base return, not writer protection. Source-image clusters, not repeated edit-input counts, bound independent support.

## Final-bank transfer (edit macro)

| Cohort | Writer | Route | Panel | Inputs | Images | Correct | ON | Base-correct damage |
|---|---|---|---|---:|---:|---:|---:|---:|
| STAGE5_NEW32_FOLLOWUP_TRANSFER_NOT_BLIND | BE | R0 | U | 220 | 7 | 43.23% | 100.00% | 0.00% |
| STAGE5_NEW32_FOLLOWUP_TRANSFER_NOT_BLIND | BE | R0 | derived_imperative_style | 64 | 32 | 100.00% | 100.00% | NA |
| STAGE5_NEW32_FOLLOWUP_TRANSFER_NOT_BLIND | BE | R0 | derived_source_style | 64 | 32 | 100.00% | 100.00% | NA |
| STAGE5_NEW32_FOLLOWUP_TRANSFER_NOT_BLIND | BE | R0 | native_source_edit | 32 | 32 | 100.00% | 100.00% | NA |
| STAGE5_NEW32_FOLLOWUP_TRANSFER_NOT_BLIND | BE | RC_FIXED_OLD16 | U | 220 | 7 | 14.58% | 0.00% | 0.00% |
| STAGE5_NEW32_FOLLOWUP_TRANSFER_NOT_BLIND | BE | RC_FIXED_OLD16 | derived_imperative_style | 64 | 32 | 100.00% | 100.00% | NA |
| STAGE5_NEW32_FOLLOWUP_TRANSFER_NOT_BLIND | BE | RC_FIXED_OLD16 | derived_source_style | 64 | 32 | 100.00% | 100.00% | NA |
| STAGE5_NEW32_FOLLOWUP_TRANSFER_NOT_BLIND | BE | RC_FIXED_OLD16 | native_source_edit | 32 | 32 | 100.00% | 100.00% | NA |
| STAGE5_NEW32_FOLLOWUP_TRANSFER_NOT_BLIND | W0 | R0 | U | 220 | 7 | 57.81% | 100.00% | 0.00% |
| STAGE5_NEW32_FOLLOWUP_TRANSFER_NOT_BLIND | W0 | R0 | derived_imperative_style | 64 | 32 | 100.00% | 100.00% | NA |
| STAGE5_NEW32_FOLLOWUP_TRANSFER_NOT_BLIND | W0 | R0 | derived_source_style | 64 | 32 | 100.00% | 100.00% | NA |
| STAGE5_NEW32_FOLLOWUP_TRANSFER_NOT_BLIND | W0 | R0 | native_source_edit | 32 | 32 | 100.00% | 100.00% | NA |
| STAGE5_NEW32_FOLLOWUP_TRANSFER_NOT_BLIND | W0 | RC_FIXED_OLD16 | U | 220 | 7 | 14.58% | 0.00% | 0.00% |
| STAGE5_NEW32_FOLLOWUP_TRANSFER_NOT_BLIND | W0 | RC_FIXED_OLD16 | derived_imperative_style | 64 | 32 | 100.00% | 100.00% | NA |
| STAGE5_NEW32_FOLLOWUP_TRANSFER_NOT_BLIND | W0 | RC_FIXED_OLD16 | derived_source_style | 64 | 32 | 100.00% | 100.00% | NA |
| STAGE5_NEW32_FOLLOWUP_TRANSFER_NOT_BLIND | W0 | RC_FIXED_OLD16 | native_source_edit | 32 | 32 | 100.00% | 100.00% | NA |
| STAGE6_FROZEN_SOURCE_EVALUATION_SIDECAR | BE | R0 | H | 3 | 2 | 33.33% | 100.00% | 66.67% |
| STAGE6_FROZEN_SOURCE_EVALUATION_SIDECAR | BE | R0 | U | 116 | 23 | 13.79% | 100.00% | 91.99% |
| STAGE6_FROZEN_SOURCE_EVALUATION_SIDECAR | BE | R0 | existing_source_alternative | 8 | 8 | 100.00% | 100.00% | NA |
| STAGE6_FROZEN_SOURCE_EVALUATION_SIDECAR | BE | RC_FIXED_OLD16 | H | 3 | 2 | 100.00% | 0.00% | 0.00% |
| STAGE6_FROZEN_SOURCE_EVALUATION_SIDECAR | BE | RC_FIXED_OLD16 | U | 116 | 23 | 39.66% | 0.00% | 0.00% |
| STAGE6_FROZEN_SOURCE_EVALUATION_SIDECAR | BE | RC_FIXED_OLD16 | existing_source_alternative | 8 | 8 | 100.00% | 100.00% | NA |
| STAGE6_FROZEN_SOURCE_EVALUATION_SIDECAR | W0 | R0 | H | 3 | 2 | 33.33% | 100.00% | 66.67% |
| STAGE6_FROZEN_SOURCE_EVALUATION_SIDECAR | W0 | R0 | U | 116 | 23 | 43.97% | 100.00% | 24.04% |
| STAGE6_FROZEN_SOURCE_EVALUATION_SIDECAR | W0 | R0 | existing_source_alternative | 8 | 8 | 100.00% | 100.00% | NA |
| STAGE6_FROZEN_SOURCE_EVALUATION_SIDECAR | W0 | RC_FIXED_OLD16 | H | 3 | 2 | 100.00% | 0.00% | 0.00% |
| STAGE6_FROZEN_SOURCE_EVALUATION_SIDECAR | W0 | RC_FIXED_OLD16 | U | 116 | 23 | 39.66% | 0.00% | 0.00% |
| STAGE6_FROZEN_SOURCE_EVALUATION_SIDECAR | W0 | RC_FIXED_OLD16 | existing_source_alternative | 8 | 8 | 100.00% | 100.00% | NA |

RC_CAUSAL_ACCOUNTING.csv exactly decomposes accuracy change into avoided damage minus lost corrections. PAIRED_EFFECTS.csv includes edit and image-cluster uncertainty with fixed seed20260908, 10000 draws. No noninferiority or clinical-safety claim.

Judge required=538, missing=0. Publication remains separate until verified.
