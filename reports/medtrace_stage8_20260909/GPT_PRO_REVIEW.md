# Stage8 bounded APR closeout

## Closeout interpretation

**Decision: PARTIAL_TRADEOFF.** The fixed-radius repair does not resolve generalization versus locality. No further APR grid, new facts or automatic Stage9 is authorized or launched. Do not promote APR/TR as the default research mechanism on this evidence; retain W0 plus the fixed RC as an engineering reference, not an originality claim.

All 23 edits completed (46 new B2/B3 endpoints), with all pipeline exit codes zero. Judge scored 1879/1879 exact tuples:1788 reused,91 new,0 missing. Wall time1270.76 seconds (21.18 minutes), GPU0 only; accounted GPU time0.3404 hours. No SGD. Successful execution source:85f8e9f616a780ab6a611b7a544884e41df0a0ee; publication identity is the containing Git commit, distinct from execution. The PENDING field below is the original pre-publication computation snapshot.

### Main FORCED_ON comparison

These percentages are edit-macro averages on shared support, not pooled sample accuracy. The old T2G panel contains23 probes across6 edit tasks/6 source images. SLAKE evaluation H contains20 inputs across15 edit tasks/7 images; U202 inputs across16 tasks/41 images. Repeated inputs/images are not independent patients; patient identity remains UNKNOWN. Exact support and sensitivity are in the attached CSV files.

| Metric | B0 W0 | B1 full APR | B2 damped APR | B3 trust-region APR |
|---|---:|---:|---:|---:|
| Old formal T2G correctness |83.33%|62.50%|83.33%|75.00%|
| SLAKE H correctness |0.00%|33.33%|0.00%|0.00%|
| SLAKE H Base-correct damage |100.00%|53.85%|100.00%|100.00%|
| SLAKE U correctness |39.95%|55.43%|49.41%|51.37%|
| SLAKE U Base-correct damage |45.45%|17.63%|31.16%|26.12%|
| SLAKE source-style correctness |100.00%|96.88%|100.00%|100.00%|
| SLAKE cross-family correctness |100.00%|100.00%|100.00%|100.00%|

1. **B2 versus B1:** weakening the repair restored old T2G by20.83 percentage points, and source-style by3.125 points, but eliminated the observed H protection and weakened U protection. It is a tradeoff, not a uniformly better APR.
2. **B3 versus B2:** reoptimizing the direction increased U correctness by1.96 points and reduced U damage by5.04 points, but lowered old T2G by8.33 points and did not help H. The edit-bootstrap95% interval for U correctness difference is[-0.56,+4.45] points; image-cluster interval[-1.14,+5.06]. For T2G it is[-20.83,+8.33] points. No clear overall behavioral superiority of TR over simple damping is established.
3. **B3 versus B0:** U correctness improved by11.42 points (edit-bootstrap95% interval[5.49,18.72]; image-cluster[5.63,18.08]), and damage fell by19.33 points. But old T2G dropped8.33 points and H remained unprotected. This is PARTIAL_TRADEOFF, not solved generalization or clinical safety.

All native scores and fit-positive scores were retained; these are in-sample preservation evidence. Old formal T1G remained100%, T1L correctness20%/damage100%, and T2L correctness100% for B0/B1/B2/B3. No gain on those formal locality panels should be inferred from U improvement.

### Corrections lost, not merely residual norms

Relative to B0, on SLAKE evaluation U, B2 avoided18 damaged answers, introduced0 damage, lost0 prior corrections and gained1. B3 avoided26 damages, introduced1, lost3 corrections and gained1. B1 avoided38 damages but introduced3, lost9 corrections and gained5. On evaluation H, B2/B3 avoided no damage and gained no correction; B1 avoided8 damages and gained1 correction. On old T2G, B2 lost0/gained0 corrections; B3 lost3/gained1; B1 lost6/gained1. These are paired edit-input counts, not independent patient counts.

### Mechanism, cost and limits

All46 deployed B2/B3 endpoints satisfied the locked norm and anchor tolerances. Maximum actual relative update norm: B2 0.100000000419, B3 0.100000000312 (permitted relative slack1e-5). Maximum anchor error: B2 FP644.5575e-9 / FP327.7604e-9; B3 FP641.5194e-14 / FP327.8318e-9. B2 inherits rounding from the saved Stage7 FP32 APR, so its FP64 arithmetic diagnostic is not an exact-zero algebraic claim. Summed paired fitting time was108.05 seconds; the same pair timing appears in both condition rows and must not be double-counted. Each expert retains57860 scalars/231440 FP32 tensor bytes; rank remains at most4, not the original1476-parameter CP representation.

The read-only diagnostic covers all23 old T2G probes and4 conditions (92 rows), with correctness and first token divergence. Evaluation activation projection, perturbation and logit-margin fields are NA because they were not cached; no extra diagnostic GPU replay was run. These missing diagnostics limit causal explanation. Target propagation has no dedicated score and remains NA. H/U fixed-RC results conceal writer differences through OFF routing and are not an APR benefit. This is viewed development, not new independent evidence; the standard trust-region/constraint solve alone is not novel.

{'status': 'COMPUTE_COMPLETE', 'coverage': [{'cohort': 'OLD_STAGE3_COMMON7', 'edit': 27, 'status': 'COMPLETE'}, {'cohort': 'OLD_STAGE3_COMMON7', 'edit': 35, 'status': 'COMPLETE'}, {'cohort': 'OLD_STAGE3_COMMON7', 'edit': 66, 'status': 'COMPLETE'}, {'cohort': 'OLD_STAGE3_COMMON7', 'edit': 143, 'status': 'COMPLETE'}, {'cohort': 'OLD_STAGE3_COMMON7', 'edit': 168, 'status': 'COMPLETE'}, {'cohort': 'OLD_STAGE3_COMMON7', 'edit': 172, 'status': 'COMPLETE'}, {'cohort': 'OLD_STAGE3_COMMON7', 'edit': 270, 'status': 'COMPLETE'}, {'cohort': 'SLAKE_STAGE2_BANK16', 'edit': 101, 'status': 'COMPLETE'}, {'cohort': 'SLAKE_STAGE2_BANK16', 'edit': 102, 'status': 'COMPLETE'}, {'cohort': 'SLAKE_STAGE2_BANK16', 'edit': 103, 'status': 'COMPLETE'}, {'cohort': 'SLAKE_STAGE2_BANK16', 'edit': 104, 'status': 'COMPLETE'}, {'cohort': 'SLAKE_STAGE2_BANK16', 'edit': 105, 'status': 'COMPLETE'}, {'cohort': 'SLAKE_STAGE2_BANK16', 'edit': 106, 'status': 'COMPLETE'}, {'cohort': 'SLAKE_STAGE2_BANK16', 'edit': 107, 'status': 'COMPLETE'}, {'cohort': 'SLAKE_STAGE2_BANK16', 'edit': 108, 'status': 'COMPLETE'}, {'cohort': 'SLAKE_STAGE2_BANK16', 'edit': 109, 'status': 'COMPLETE'}, {'cohort': 'SLAKE_STAGE2_BANK16', 'edit': 110, 'status': 'COMPLETE'}, {'cohort': 'SLAKE_STAGE2_BANK16', 'edit': 111, 'status': 'COMPLETE'}, {'cohort': 'SLAKE_STAGE2_BANK16', 'edit': 112, 'status': 'COMPLETE'}, {'cohort': 'SLAKE_STAGE2_BANK16', 'edit': 113, 'status': 'COMPLETE'}, {'cohort': 'SLAKE_STAGE2_BANK16', 'edit': 114, 'status': 'COMPLETE'}, {'cohort': 'SLAKE_STAGE2_BANK16', 'edit': 115, 'status': 'COMPLETE'}, {'cohort': 'SLAKE_STAGE2_BANK16', 'edit': 116, 'status': 'COMPLETE'}], 'judge_required': 1879, 'judge_scored': 1879, 'judge_missing': 0, 'judge_new': 91, 'judge_reused': 1788, 'new_sgd_training': 0, 'publication': 'PENDING'}

Viewed development only. Fixed eta=.10; no new SGD or router. Full source/role-separated results and paired sensitivity are attached.
Target propagation NA. T2G token divergence is observed; missing activation/logit diagnostics are NA, not inferred causes.
Historical C1/W1 are reused, not equal-budget new controls. Fixed-RC all-OFF results cannot establish writer benefit.
Rank4 expanded input:57860 scalars/231440 FP32 bytes per expert, not original1476 CP parameters.

|Cohort|Method|Role|Panel|Correct macro|Damage macro|
|---|---|---|---|---:|---:|
|APR_OLD_STAGE3_COMMON7|B0|fit|H|0.09523809523809523|None|
|APR_OLD_STAGE3_COMMON7|B0|fit|U|0.3826530612244898|0.39444444444444443|
|APR_OLD_STAGE3_COMMON7|B0|fit|positive|0.9285714285714286|None|
|APR_OLD_STAGE3_COMMON7|B0|formal_development|T1G|1.0|None|
|APR_OLD_STAGE3_COMMON7|B0|formal_development|T1L|0.2|1.0|
|APR_OLD_STAGE3_COMMON7|B0|formal_development|T2G|0.8333333333333334|None|
|APR_OLD_STAGE3_COMMON7|B0|formal_development|T2L|1.0|0.0|
|APR_OLD_STAGE3_COMMON7|B0|native|NATIVE_DIAGNOSTIC|1.0|None|
|APR_OLD_STAGE3_COMMON7|B0|native|T0|1.0|None|
|APR_OLD_STAGE3_COMMON7|B1|fit|H|0.09523809523809523|None|
|APR_OLD_STAGE3_COMMON7|B1|fit|U|0.5017006802721088|0.16666666666666666|
|APR_OLD_STAGE3_COMMON7|B1|fit|positive|0.9285714285714286|None|
|APR_OLD_STAGE3_COMMON7|B1|formal_development|T1G|1.0|None|
|APR_OLD_STAGE3_COMMON7|B1|formal_development|T1L|0.2|1.0|
|APR_OLD_STAGE3_COMMON7|B1|formal_development|T2G|0.625|None|
|APR_OLD_STAGE3_COMMON7|B1|formal_development|T2L|1.0|0.0|
|APR_OLD_STAGE3_COMMON7|B1|native|NATIVE_DIAGNOSTIC|1.0|None|
|APR_OLD_STAGE3_COMMON7|B1|native|T0|1.0|None|
|APR_OLD_STAGE3_COMMON7|B2|fit|H|0.0|None|
|APR_OLD_STAGE3_COMMON7|B2|fit|U|0.4659863945578231|0.25|
|APR_OLD_STAGE3_COMMON7|B2|fit|positive|0.9285714285714286|None|
|APR_OLD_STAGE3_COMMON7|B2|formal_development|T1G|1.0|None|
|APR_OLD_STAGE3_COMMON7|B2|formal_development|T1L|0.2|1.0|
|APR_OLD_STAGE3_COMMON7|B2|formal_development|T2G|0.8333333333333334|None|
|APR_OLD_STAGE3_COMMON7|B2|formal_development|T2L|1.0|0.0|
|APR_OLD_STAGE3_COMMON7|B2|native|NATIVE_DIAGNOSTIC|1.0|None|
|APR_OLD_STAGE3_COMMON7|B2|native|T0|1.0|None|
|APR_OLD_STAGE3_COMMON7|B3|fit|H|0.0|None|
|APR_OLD_STAGE3_COMMON7|B3|fit|U|0.4064625850340136|0.25|
|APR_OLD_STAGE3_COMMON7|B3|fit|positive|0.9285714285714286|None|
|APR_OLD_STAGE3_COMMON7|B3|formal_development|T1G|1.0|None|
|APR_OLD_STAGE3_COMMON7|B3|formal_development|T1L|0.2|1.0|
|APR_OLD_STAGE3_COMMON7|B3|formal_development|T2G|0.75|None|
|APR_OLD_STAGE3_COMMON7|B3|formal_development|T2L|1.0|0.0|
|APR_OLD_STAGE3_COMMON7|B3|native|NATIVE_DIAGNOSTIC|1.0|None|
|APR_OLD_STAGE3_COMMON7|B3|native|T0|1.0|None|
|APR_OLD_STAGE3_COMMON7|HISTORICAL_C1|fit|H|0.0|None|
|APR_OLD_STAGE3_COMMON7|HISTORICAL_C1|fit|U|0.3945578231292517|0.25|
|APR_OLD_STAGE3_COMMON7|HISTORICAL_C1|fit|positive|0.0|None|
|APR_OLD_STAGE3_COMMON7|HISTORICAL_C1|formal_development|T1G|0.0|None|
|APR_OLD_STAGE3_COMMON7|HISTORICAL_C1|formal_development|T1L|0.4|0.3333333333333333|
|APR_OLD_STAGE3_COMMON7|HISTORICAL_C1|formal_development|T2G|0.09722222222222222|None|
|APR_OLD_STAGE3_COMMON7|HISTORICAL_C1|formal_development|T2L|1.0|0.0|
|APR_OLD_STAGE3_COMMON7|HISTORICAL_C1|native|NATIVE_DIAGNOSTIC|0.0|None|
|APR_OLD_STAGE3_COMMON7|HISTORICAL_C1|native|T0|0.0|None|
|APR_OLD_STAGE3_COMMON7|HISTORICAL_W1|fit|H|0.0|None|
|APR_OLD_STAGE3_COMMON7|HISTORICAL_W1|fit|U|0.3826530612244898|0.08333333333333333|
|APR_OLD_STAGE3_COMMON7|HISTORICAL_W1|fit|positive|0.9285714285714286|None|
|APR_OLD_STAGE3_COMMON7|HISTORICAL_W1|formal_development|T1G|1.0|None|
|APR_OLD_STAGE3_COMMON7|HISTORICAL_W1|formal_development|T1L|0.2|1.0|
|APR_OLD_STAGE3_COMMON7|HISTORICAL_W1|formal_development|T2G|0.3055555555555556|None|
|APR_OLD_STAGE3_COMMON7|HISTORICAL_W1|formal_development|T2L|1.0|0.0|
|APR_OLD_STAGE3_COMMON7|HISTORICAL_W1|native|NATIVE_DIAGNOSTIC|1.0|None|
|APR_OLD_STAGE3_COMMON7|HISTORICAL_W1|native|T0|1.0|None|
|APR_SLAKE_STAGE2_BANK16|B0|challenge|same_answer_context_not_conflict|1.0|0.0|
|APR_SLAKE_STAGE2_BANK16|B0|evaluation|H|0.0|1.0|
|APR_SLAKE_STAGE2_BANK16|B0|evaluation|U|0.399496336996337|0.45450036075036077|
|APR_SLAKE_STAGE2_BANK16|B0|evaluation|cross_family_confirmation|1.0|None|
|APR_SLAKE_STAGE2_BANK16|B0|evaluation|source_style_confirmation|1.0|None|
|APR_SLAKE_STAGE2_BANK16|B0|fit|H|0.0|1.0|
|APR_SLAKE_STAGE2_BANK16|B0|fit|U|0.35370879120879123|0.4807539682539682|
|APR_SLAKE_STAGE2_BANK16|B0|fit|cross_family_confirmation|1.0|None|
|APR_SLAKE_STAGE2_BANK16|B0|native|T0|1.0|None|
|APR_SLAKE_STAGE2_BANK16|B1|challenge|same_answer_context_not_conflict|1.0|0.0|
|APR_SLAKE_STAGE2_BANK16|B1|evaluation|H|0.3333333333333333|0.5384615384615384|
|APR_SLAKE_STAGE2_BANK16|B1|evaluation|U|0.5542582417582418|0.1762941919191919|
|APR_SLAKE_STAGE2_BANK16|B1|evaluation|cross_family_confirmation|1.0|None|
|APR_SLAKE_STAGE2_BANK16|B1|evaluation|source_style_confirmation|0.96875|None|
|APR_SLAKE_STAGE2_BANK16|B1|fit|H|0.5625|0.4375|
|APR_SLAKE_STAGE2_BANK16|B1|fit|U|0.5319368131868132|0.18447420634920633|
|APR_SLAKE_STAGE2_BANK16|B1|fit|cross_family_confirmation|1.0|None|
|APR_SLAKE_STAGE2_BANK16|B1|native|T0|1.0|None|
|APR_SLAKE_STAGE2_BANK16|B2|challenge|same_answer_context_not_conflict|1.0|0.0|
|APR_SLAKE_STAGE2_BANK16|B2|evaluation|H|0.0|1.0|
|APR_SLAKE_STAGE2_BANK16|B2|evaluation|U|0.4941048534798535|0.3115936147186147|
|APR_SLAKE_STAGE2_BANK16|B2|evaluation|cross_family_confirmation|1.0|None|
|APR_SLAKE_STAGE2_BANK16|B2|evaluation|source_style_confirmation|1.0|None|
|APR_SLAKE_STAGE2_BANK16|B2|fit|H|0.0|1.0|
|APR_SLAKE_STAGE2_BANK16|B2|fit|U|0.4375|0.3273313492063492|
|APR_SLAKE_STAGE2_BANK16|B2|fit|cross_family_confirmation|1.0|None|
|APR_SLAKE_STAGE2_BANK16|B2|native|T0|1.0|None|
|APR_SLAKE_STAGE2_BANK16|B3|challenge|same_answer_context_not_conflict|1.0|0.0|
|APR_SLAKE_STAGE2_BANK16|B3|evaluation|H|0.0|1.0|
|APR_SLAKE_STAGE2_BANK16|B3|evaluation|U|0.5137362637362638|0.26116747835497833|
|APR_SLAKE_STAGE2_BANK16|B3|evaluation|cross_family_confirmation|1.0|None|
|APR_SLAKE_STAGE2_BANK16|B3|evaluation|source_style_confirmation|1.0|None|
|APR_SLAKE_STAGE2_BANK16|B3|fit|H|0.03125|0.96875|
|APR_SLAKE_STAGE2_BANK16|B3|fit|U|0.489010989010989|0.24002976190476188|
|APR_SLAKE_STAGE2_BANK16|B3|fit|cross_family_confirmation|1.0|None|
|APR_SLAKE_STAGE2_BANK16|B3|native|T0|1.0|None|
|APR_SLAKE_STAGE2_BANK16|HISTORICAL_C1|challenge|same_answer_context_not_conflict|0.7|0.0|
|APR_SLAKE_STAGE2_BANK16|HISTORICAL_C1|evaluation|H|0.8|0.07692307692307693|
|APR_SLAKE_STAGE2_BANK16|HISTORICAL_C1|evaluation|U|0.6527586996336996|0.00625|
|APR_SLAKE_STAGE2_BANK16|HISTORICAL_C1|evaluation|cross_family_confirmation|0.5625|None|
|APR_SLAKE_STAGE2_BANK16|HISTORICAL_C1|evaluation|source_style_confirmation|0.84375|None|
|APR_SLAKE_STAGE2_BANK16|HISTORICAL_C1|fit|H|0.9166666666666666|0.0625|
|APR_SLAKE_STAGE2_BANK16|HISTORICAL_C1|fit|U|0.5676510989010989|0.053348214285714284|
|APR_SLAKE_STAGE2_BANK16|HISTORICAL_C1|fit|cross_family_confirmation|0.5|None|
|APR_SLAKE_STAGE2_BANK16|HISTORICAL_C1|native|T0|0.3125|None|
|APR_SLAKE_STAGE2_BANK16|HISTORICAL_W1|challenge|same_answer_context_not_conflict|1.0|0.0|
|APR_SLAKE_STAGE2_BANK16|HISTORICAL_W1|evaluation|H|0.3|0.5384615384615384|
|APR_SLAKE_STAGE2_BANK16|HISTORICAL_W1|evaluation|U|0.5987293956043956|0.1280573593073593|
|APR_SLAKE_STAGE2_BANK16|HISTORICAL_W1|evaluation|cross_family_confirmation|1.0|None|
|APR_SLAKE_STAGE2_BANK16|HISTORICAL_W1|evaluation|source_style_confirmation|1.0|None|
|APR_SLAKE_STAGE2_BANK16|HISTORICAL_W1|fit|H|0.3854166666666667|0.59375|
|APR_SLAKE_STAGE2_BANK16|HISTORICAL_W1|fit|U|0.5182005494505495|0.15535714285714286|
|APR_SLAKE_STAGE2_BANK16|HISTORICAL_W1|fit|cross_family_confirmation|1.0|None|
|APR_SLAKE_STAGE2_BANK16|HISTORICAL_W1|native|T0|1.0|None|

No automatic next experiment. Comparative research decision requires reading the three locked paired comparisons; no SOTA, clinical, independent-test or originality claim. Standard trust-region/constraint algebra is not itself novel; acknowledge TIME, M-ORE, AlphaEdit and ROME.
