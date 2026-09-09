# Stage7 APR development results

## Interpretation and execution provenance

All 23 edits and 46 closed-form solves completed; Judge covered 1754/1754 tuples (1417 reused, 337 new), with all pipeline exit codes zero. Wall time was 1647.93 seconds (27.47 minutes); recorded GPU-hour upper bound was 0.7502. No additional SGD was performed.

The percentages below are edit-macro summaries, not correct counts divided by the unique-input column. Exact denominators and paired/clustered sensitivity are in the CSV files. These are exposed development cohorts, not independent confirmation.

1. **Finite constraints held:** across 23 APR edits, maximum FP64 relative anchor error was 1.8764e-13 and maximum deployed FP32 error was 1.5388e-8, below the frozen tolerances. This establishes only the collected activation constraint, not unseen semantic preservation.
2. **Protection helped relative to C1 but did not eliminate T2G regression:** old formal T2G was C0 83.33%, C1 9.72%, APR 62.50%; historical KL W1 was 30.56% (different training budget). APR improves on C1 by 52.78 percentage points but loses 20.83 points against C0. Native and fit-positive scores were preserved; they are in-sample evidence.
3. **Real damage was reduced, not eliminated:** SLAKE evaluation H Base-correct damage fell from 100% to 53.85%; U from 45.45% to 17.63%. U correctness rose from 39.95% to 55.43%. Old formal T1L damage remained 100%. APR does not uniformly dominate historical W1 or unconstrained repair on locality. Source-style correctness fell from 100% to 96.88%, while cross-family remained 100%.
4. **Cost:** each expanded expert has 57860 scalar parameters / 231440 FP32 tensor bytes, versus 1476 original CP parameters (about 39.2 times the scalar count). Rank remains at most four, but the input Kronecker constraint is removed. Per-edit fit timings sum to 140.17 seconds; these are component timings, not an additional wall-time estimate. The geometry table includes container sizes and per-edit costs.
5. **Returning toward Base has costs:** relative to C0, APR on SLAKE evaluation H avoided 8 damaged answers and gained 1 correction; U avoided 38 damages but introduced 3, lost 9 prior corrections and gained 5. Old formal T2G lost 6 corrections and gained 1. These are paired edit-input counts, not independent patients. Fixed-RC H/U differences are zero because the gate masks the writer; they are not an APR benefit. See CORRECTION_TRADEOFF_COUNTS.csv for fit versus evaluation separation.

Conclusion: the constrained repair offers a development-set protection/locality tradeoff, but it has not solved the original generalization loss. No follow-on parameter search or Stage8 was launched. A dedicated target-propagation metric is not present in this release; correctness, damage, correction and preservation must not be relabeled as that metric.

Preparation source was aa85875ec82997f6404ba27bfc45ac1c1102e2b1; actual successful execution source was 129d7c5b09567cb8f630d0bb6279477789439b63. Initial first-edit input validation stopped before solving because inherited fit rows used a historical feature-cache EqKey schema. The fix verifies their original source packet and actual image/tokens/attention/positions using that schema; it does not rewrite data or relax the binding. Original failed logs and the original campaign clock were retained. Completed edits were not rerun. The focused synthetic checks, including historical-binding rejection on changed tokens, passed (2 tests).

Acknowledgment: TIME supplies the CP-expert context and editing objectives; M-ORE, AlphaEdit and ROME provide related fixed-coordinate, quadratic-protection and constrained/null-space editing ideas. Constraint algebra alone is not an originality claim. Clinical safety and full reproductions of those methods are not established.

Status: COMPUTE_COMPLETE
Finite activation constraints are not unseen-query semantic guarantees. All data are exposed development cohorts. No SGD or threshold retuning.
C0/C1/C2 share rank4 and the expanded input interface; input CP structure is no longer imposed. Storage is not the original1476 parameters.

| Cohort | Condition | Mode | Role | Panel | Inputs | Images | Correct | Base-correct damage |
|---|---|---|---|---|---:|---:|---:|---:|
| APR_OLD_STAGE3_COMMON7 | C0 | FORCED_ON | fit | H | 16 | 13 | 9.52% | NA |
| APR_OLD_STAGE3_COMMON7 | C0 | FORCED_ON | fit | U | 39 | 31 | 38.27% | 39.44% |
| APR_OLD_STAGE3_COMMON7 | C0 | FORCED_ON | fit | positive | 24 | 6 | 92.86% | NA |
| APR_OLD_STAGE3_COMMON7 | C0 | FORCED_ON | formal_development | T1G | 22 | 22 | 100.00% | NA |
| APR_OLD_STAGE3_COMMON7 | C0 | FORCED_ON | formal_development | T1L | 10 | 5 | 20.00% | 100.00% |
| APR_OLD_STAGE3_COMMON7 | C0 | FORCED_ON | formal_development | T2G | 23 | 6 | 83.33% | NA |
| APR_OLD_STAGE3_COMMON7 | C0 | FORCED_ON | formal_development | T2L | 2 | 1 | 100.00% | 0.00% |
| APR_OLD_STAGE3_COMMON7 | C0 | FORCED_ON | native | NATIVE_DIAGNOSTIC | 1 | 1 | 100.00% | NA |
| APR_OLD_STAGE3_COMMON7 | C0 | FORCED_ON | native | T0 | 6 | 6 | 100.00% | NA |
| APR_OLD_STAGE3_COMMON7 | C1 | FORCED_ON | fit | H | 16 | 13 | 0.00% | NA |
| APR_OLD_STAGE3_COMMON7 | C1 | FORCED_ON | fit | U | 39 | 31 | 39.46% | 25.00% |
| APR_OLD_STAGE3_COMMON7 | C1 | FORCED_ON | fit | positive | 24 | 6 | 0.00% | NA |
| APR_OLD_STAGE3_COMMON7 | C1 | FORCED_ON | formal_development | T1G | 22 | 22 | 0.00% | NA |
| APR_OLD_STAGE3_COMMON7 | C1 | FORCED_ON | formal_development | T1L | 10 | 5 | 40.00% | 33.33% |
| APR_OLD_STAGE3_COMMON7 | C1 | FORCED_ON | formal_development | T2G | 23 | 6 | 9.72% | NA |
| APR_OLD_STAGE3_COMMON7 | C1 | FORCED_ON | formal_development | T2L | 2 | 1 | 100.00% | 0.00% |
| APR_OLD_STAGE3_COMMON7 | C1 | FORCED_ON | native | NATIVE_DIAGNOSTIC | 1 | 1 | 0.00% | NA |
| APR_OLD_STAGE3_COMMON7 | C1 | FORCED_ON | native | T0 | 6 | 6 | 0.00% | NA |
| APR_OLD_STAGE3_COMMON7 | C2 | FORCED_ON | fit | H | 16 | 13 | 9.52% | NA |
| APR_OLD_STAGE3_COMMON7 | C2 | FORCED_ON | fit | U | 39 | 31 | 50.17% | 16.67% |
| APR_OLD_STAGE3_COMMON7 | C2 | FORCED_ON | fit | positive | 24 | 6 | 92.86% | NA |
| APR_OLD_STAGE3_COMMON7 | C2 | FORCED_ON | formal_development | T1G | 22 | 22 | 100.00% | NA |
| APR_OLD_STAGE3_COMMON7 | C2 | FORCED_ON | formal_development | T1L | 10 | 5 | 20.00% | 100.00% |
| APR_OLD_STAGE3_COMMON7 | C2 | FORCED_ON | formal_development | T2G | 23 | 6 | 62.50% | NA |
| APR_OLD_STAGE3_COMMON7 | C2 | FORCED_ON | formal_development | T2L | 2 | 1 | 100.00% | 0.00% |
| APR_OLD_STAGE3_COMMON7 | C2 | FORCED_ON | native | NATIVE_DIAGNOSTIC | 1 | 1 | 100.00% | NA |
| APR_OLD_STAGE3_COMMON7 | C2 | FORCED_ON | native | T0 | 6 | 6 | 100.00% | NA |
| APR_OLD_STAGE3_COMMON7 | HISTORICAL_W1 | FORCED_ON | fit | H | 16 | 13 | 0.00% | NA |
| APR_OLD_STAGE3_COMMON7 | HISTORICAL_W1 | FORCED_ON | fit | U | 39 | 31 | 38.27% | 8.33% |
| APR_OLD_STAGE3_COMMON7 | HISTORICAL_W1 | FORCED_ON | fit | positive | 24 | 6 | 92.86% | NA |
| APR_OLD_STAGE3_COMMON7 | HISTORICAL_W1 | FORCED_ON | formal_development | T1G | 22 | 22 | 100.00% | NA |
| APR_OLD_STAGE3_COMMON7 | HISTORICAL_W1 | FORCED_ON | formal_development | T1L | 10 | 5 | 20.00% | 100.00% |
| APR_OLD_STAGE3_COMMON7 | HISTORICAL_W1 | FORCED_ON | formal_development | T2G | 23 | 6 | 30.56% | NA |
| APR_OLD_STAGE3_COMMON7 | HISTORICAL_W1 | FORCED_ON | formal_development | T2L | 2 | 1 | 100.00% | 0.00% |
| APR_OLD_STAGE3_COMMON7 | HISTORICAL_W1 | FORCED_ON | native | NATIVE_DIAGNOSTIC | 1 | 1 | 100.00% | NA |
| APR_OLD_STAGE3_COMMON7 | HISTORICAL_W1 | FORCED_ON | native | T0 | 6 | 6 | 100.00% | NA |
| APR_SLAKE_STAGE2_BANK16 | C0 | FORCED_ON | challenge | same_answer_context_not_conflict | 32 | 6 | 100.00% | 0.00% |
| APR_SLAKE_STAGE2_BANK16 | C0 | FORCED_ON | evaluation | H | 20 | 7 | 0.00% | 100.00% |
| APR_SLAKE_STAGE2_BANK16 | C0 | FORCED_ON | evaluation | U | 202 | 41 | 39.95% | 45.45% |
| APR_SLAKE_STAGE2_BANK16 | C0 | FORCED_ON | evaluation | cross_family_confirmation | 32 | 16 | 100.00% | NA |
| APR_SLAKE_STAGE2_BANK16 | C0 | FORCED_ON | evaluation | source_style_confirmation | 32 | 16 | 100.00% | NA |
| APR_SLAKE_STAGE2_BANK16 | C0 | FORCED_ON | fit | H | 28 | 9 | 0.00% | 100.00% |
| APR_SLAKE_STAGE2_BANK16 | C0 | FORCED_ON | fit | U | 217 | 43 | 35.37% | 48.08% |
| APR_SLAKE_STAGE2_BANK16 | C0 | FORCED_ON | fit | cross_family_confirmation | 64 | 16 | 100.00% | NA |
| APR_SLAKE_STAGE2_BANK16 | C0 | FORCED_ON | native | T0 | 16 | 16 | 100.00% | NA |
| APR_SLAKE_STAGE2_BANK16 | C1 | FORCED_ON | challenge | same_answer_context_not_conflict | 32 | 6 | 70.00% | 0.00% |
| APR_SLAKE_STAGE2_BANK16 | C1 | FORCED_ON | evaluation | H | 20 | 7 | 80.00% | 7.69% |
| APR_SLAKE_STAGE2_BANK16 | C1 | FORCED_ON | evaluation | U | 202 | 41 | 65.28% | 0.62% |
| APR_SLAKE_STAGE2_BANK16 | C1 | FORCED_ON | evaluation | cross_family_confirmation | 32 | 16 | 56.25% | NA |
| APR_SLAKE_STAGE2_BANK16 | C1 | FORCED_ON | evaluation | source_style_confirmation | 32 | 16 | 84.38% | NA |
| APR_SLAKE_STAGE2_BANK16 | C1 | FORCED_ON | fit | H | 28 | 9 | 91.67% | 6.25% |
| APR_SLAKE_STAGE2_BANK16 | C1 | FORCED_ON | fit | U | 217 | 43 | 56.77% | 5.33% |
| APR_SLAKE_STAGE2_BANK16 | C1 | FORCED_ON | fit | cross_family_confirmation | 64 | 16 | 50.00% | NA |
| APR_SLAKE_STAGE2_BANK16 | C1 | FORCED_ON | native | T0 | 16 | 16 | 31.25% | NA |
| APR_SLAKE_STAGE2_BANK16 | C2 | FORCED_ON | challenge | same_answer_context_not_conflict | 32 | 6 | 100.00% | 0.00% |
| APR_SLAKE_STAGE2_BANK16 | C2 | FORCED_ON | evaluation | H | 20 | 7 | 33.33% | 53.85% |
| APR_SLAKE_STAGE2_BANK16 | C2 | FORCED_ON | evaluation | U | 202 | 41 | 55.43% | 17.63% |
| APR_SLAKE_STAGE2_BANK16 | C2 | FORCED_ON | evaluation | cross_family_confirmation | 32 | 16 | 100.00% | NA |
| APR_SLAKE_STAGE2_BANK16 | C2 | FORCED_ON | evaluation | source_style_confirmation | 32 | 16 | 96.88% | NA |
| APR_SLAKE_STAGE2_BANK16 | C2 | FORCED_ON | fit | H | 28 | 9 | 56.25% | 43.75% |
| APR_SLAKE_STAGE2_BANK16 | C2 | FORCED_ON | fit | U | 217 | 43 | 53.19% | 18.45% |
| APR_SLAKE_STAGE2_BANK16 | C2 | FORCED_ON | fit | cross_family_confirmation | 64 | 16 | 100.00% | NA |
| APR_SLAKE_STAGE2_BANK16 | C2 | FORCED_ON | native | T0 | 16 | 16 | 100.00% | NA |
| APR_SLAKE_STAGE2_BANK16 | HISTORICAL_W1 | FORCED_ON | challenge | same_answer_context_not_conflict | 32 | 6 | 100.00% | 0.00% |
| APR_SLAKE_STAGE2_BANK16 | HISTORICAL_W1 | FORCED_ON | evaluation | H | 20 | 7 | 30.00% | 53.85% |
| APR_SLAKE_STAGE2_BANK16 | HISTORICAL_W1 | FORCED_ON | evaluation | U | 202 | 41 | 59.87% | 12.81% |
| APR_SLAKE_STAGE2_BANK16 | HISTORICAL_W1 | FORCED_ON | evaluation | cross_family_confirmation | 32 | 16 | 100.00% | NA |
| APR_SLAKE_STAGE2_BANK16 | HISTORICAL_W1 | FORCED_ON | evaluation | source_style_confirmation | 32 | 16 | 100.00% | NA |
| APR_SLAKE_STAGE2_BANK16 | HISTORICAL_W1 | FORCED_ON | fit | H | 28 | 9 | 38.54% | 59.38% |
| APR_SLAKE_STAGE2_BANK16 | HISTORICAL_W1 | FORCED_ON | fit | U | 217 | 43 | 51.82% | 15.54% |
| APR_SLAKE_STAGE2_BANK16 | HISTORICAL_W1 | FORCED_ON | fit | cross_family_confirmation | 64 | 16 | 100.00% | NA |
| APR_SLAKE_STAGE2_BANK16 | HISTORICAL_W1 | FORCED_ON | native | T0 | 16 | 16 | 100.00% | NA |

Full numerical geometry, FP32/FP64 anchor errors and paired sensitivity are separate. Fit/native are in-sample, not generality evidence. Historical W1 had gradient training, not the same fitting budget. System all-OFF masks writer differences. No clinical safety, originality, full TIME/AlphaEdit/M-ORE or M3Bench reproduction claim.

Judge missing: 0. Publication is tracked separately.
