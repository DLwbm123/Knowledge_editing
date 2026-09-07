# Generality and safety trade-off

Pre-registered comparisons against M0 on the same matched panel and PRIMARY operating point:

- `M1_C1_CP_CONJUNCTION_CONTROL`: hard_fpr_drop_at_least_0_10=FAIL; evaluation_positive_drop_at_most_0_02=FAIL; t1g_drop_at_most_0_02=FAIL; t2g_drop_at_most_0_02=FAIL; base_correct_damage_not_increased=FAIL; paired hard delta 95% CI=(-0.1, -0.021428571428571432).
- `M2_C1_CP_RESPONSE_VERIFIER`: hard_fpr_drop_at_least_0_10=FAIL; evaluation_positive_drop_at_most_0_02=FAIL; t1g_drop_at_most_0_02=PASS; t2g_drop_at_most_0_02=PASS; base_correct_damage_not_increased=PASS; paired hard delta 95% CI=(-0.10952380952380954, -0.00714285714285714).
- `M3_C1_PRE_CP_FEATURE_VERIFIER`: hard_fpr_drop_at_least_0_10=FAIL; evaluation_positive_drop_at_most_0_02=FAIL; t1g_drop_at_most_0_02=PASS; t2g_drop_at_most_0_02=PASS; base_correct_damage_not_increased=FAIL; paired hard delta 95% CI=(-0.18333333333333335, 0.016666666666666666).

Decision: `ROUTER_DEVELOPMENT_SIGNAL_NOT_MET`; simplest passing condition: `NONE`.
