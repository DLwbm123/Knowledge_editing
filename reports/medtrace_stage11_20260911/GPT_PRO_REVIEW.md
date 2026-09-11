# Stage11 joint fact writer comparison

## Closeout and writer recommendation

All45 planned final trajectories and90 step endpoints completed; Judge1171/1171, missing0 (1017 reused,154 new). Recorded wall time78.60 minutes; process-interval GPU time upper bound2.563 hours. Execution source2569831. All15 original Stage9 edits passed the bounded audit, with no quarantined records. No new experiment was launched.

At the fixed primary step320, free rank4 is the preferred development backbone, not a validated clinical method. It jointly fits all23 selected native/H-fit pairs, preserves all15 native targets and reaches100% cross-family correctness, versus CP's20/23,80% and86.67%. Held-out H pair correctness improves only from9/19 (47.37%) to11/19 (57.89%): transfer remains incomplete. Native is trained in these pairs, so this is not unseen-edit/patient confirmation.

The improvement is not uniform protection: free rank4 U base-correct damage is25.77%, above CP18.80%, and H-evaluation macro correctness is53.57%, below CP64.29%. Pair correctness improves despite that lower marginal H score because native correctness is preserved; do not rank methods by H alone. FIT_PairCorrect only covers selected training H, whereas the full H-fit table includes other legal fit rows, explaining100% selected pairs alongside95.56% all-H-fit macro correctness.

Free rank16 matches rank4's23/23 fit and11/19 evaluation pairs, but has lower cross-family correctness (93.33%) and more U damage (31.43%). Thus the larger writer is not justified by this experiment. Actual parameter counts are1476 /73728 /294912; FP32 parameter bytes5904 /294912 /1179648. Free rank4 costs about50 times CP's parameter storage; rank16 costs4 times free rank4. Same rank does not isolate capacity from parameter geometry/normalization and optimization. Detailed optimizer/file sizes and measured times are in COSTS.csv.

Both free writers already reach these same pair scores at step160; step320 remains the prespecified primary endpoint. This is a budget comparison, not post-hoc checkpoint selection. Overall: joint fitting improved, held-out pair transfer improved modestly, damage remains a tradeoff, and anatomical-direction-specific benefits from Stage10 remain unproven. Keep free rank4 as the provisional simple baseline for any separately authorized future work; no automatic next stage and no claim of independent confirmation or solved official T2G regression.

Coverage limits: patients UNKNOWN; pair counts are not independent patient counts. First-answer-token rank was not implemented. Unique support columns in PAIR_OUTCOMES are per edit and must not be summed as globally unique images. Inherited BE/W0/W01/W1 all-NA rows below are generic reporting placeholders, not additional Stage11 experiments or official T2G support. Historical RUN_STATUS publication=PENDING is the pre-release compute snapshot; the Git commit containing this report records public delivery. Raw QA, images, masks, predictions/tokens, weights and Judge mappings remain private.

{'status': 'COMPUTE_COMPLETE_SUPPORTED_SUBSET', 'planned_final_endpoints': 45, 'supported_final_endpoints': 45, 'completed_final_endpoints': 45, 'completed_step_endpoints': 90, 'judge_required': 1171, 'judge_scored': 1171, 'judge_missing': 0, 'judge_reused': 1017, 'judge_new': 154, 'publication': 'PENDING'}

Viewed development only. Compare step320 as primary and step160 as fixed interim. J0 reran from W0 because historical RNG state was absent; all objectives and schedules match. No automatic next experiment.
FIT pairs use actually selected H-fit; EVAL pairs use held-out H within trained edits, not unseen patients. Missing exact-question pairs only restrict pair denominators. Patient UNKNOWN. First-answer-token rank NA. See stratified tables and costs.

|Writer|Step|Pair panel|Both correct / pairs|Pair micro|Edit macro|
|---|---:|---|---:|---:|---:|
|J0_CP_R4|160|FIT_PairCorrect|19/23|0.8260869565217391|0.7333333333333333|
|J0_CP_R4|160|EVAL_PairCorrect|7/19|0.3684210526315789|0.2857142857142857|
|J0_CP_R4|320|FIT_PairCorrect|20/23|0.8695652173913043|0.8|
|J0_CP_R4|320|EVAL_PairCorrect|9/19|0.47368421052631576|0.42857142857142855|
|J1_FREE_R4|160|FIT_PairCorrect|23/23|1.0|1.0|
|J1_FREE_R4|160|EVAL_PairCorrect|11/19|0.5789473684210527|0.5357142857142857|
|J1_FREE_R4|320|FIT_PairCorrect|23/23|1.0|1.0|
|J1_FREE_R4|320|EVAL_PairCorrect|11/19|0.5789473684210527|0.5357142857142857|
|J2_FREE_R16|160|FIT_PairCorrect|23/23|1.0|1.0|
|J2_FREE_R16|160|EVAL_PairCorrect|11/19|0.5789473684210527|0.5357142857142857|
|J2_FREE_R16|320|FIT_PairCorrect|23/23|1.0|1.0|
|J2_FREE_R16|320|EVAL_PairCorrect|11/19|0.5789473684210527|0.5357142857142857|

|Writer|Step|Role|Panel|Correct macro|Base-correct damage|
|---|---:|---|---|---:|---:|
|J0_CP_R4|160|challenge|same_answer_context_not_conflict|0.7857142857142857|0.0|
|J0_CP_R4|320|challenge|same_answer_context_not_conflict|0.6785714285714286|0.2|
|J0_CP_R4|160|evaluation|H|0.42857142857142855|0.46153846153846156|
|J0_CP_R4|320|evaluation|H|0.6428571428571429|0.23076923076923078|
|J0_CP_R4|160|evaluation|U|0.5184981684981685|0.2041053391053391|
|J0_CP_R4|320|evaluation|U|0.5351037851037851|0.188020683020683|
|J0_CP_R4|160|evaluation|cross_family_confirmation|0.7666666666666667|None|
|J0_CP_R4|320|evaluation|cross_family_confirmation|0.8666666666666667|None|
|J0_CP_R4|160|evaluation|source_style_confirmation|1.0|None|
|J0_CP_R4|320|evaluation|source_style_confirmation|1.0|None|
|J0_CP_R4|160|fit|H|0.7333333333333333|0.25555555555555554|
|J0_CP_R4|320|fit|H|0.9333333333333333|0.05555555555555555|
|J0_CP_R4|160|fit|U|0.4249084249084249|0.2735978835978836|
|J0_CP_R4|320|fit|U|0.44395604395604393|0.2577777777777778|
|J0_CP_R4|160|fit|cross_family_confirmation|0.8666666666666667|None|
|J0_CP_R4|320|fit|cross_family_confirmation|0.9666666666666667|None|
|J0_CP_R4|160|native|T0|0.7333333333333333|None|
|J0_CP_R4|320|native|T0|0.8|None|
|J1_FREE_R4|160|challenge|same_answer_context_not_conflict|0.75|0.3|
|J1_FREE_R4|320|challenge|same_answer_context_not_conflict|0.75|0.3|
|J1_FREE_R4|160|evaluation|H|0.5357142857142857|0.3076923076923077|
|J1_FREE_R4|320|evaluation|H|0.5357142857142857|0.3076923076923077|
|J1_FREE_R4|160|evaluation|U|0.5170940170940171|0.26511303511303513|
|J1_FREE_R4|320|evaluation|U|0.5222222222222223|0.2577056277056277|
|J1_FREE_R4|160|evaluation|cross_family_confirmation|1.0|None|
|J1_FREE_R4|320|evaluation|cross_family_confirmation|1.0|None|
|J1_FREE_R4|160|evaluation|source_style_confirmation|1.0|None|
|J1_FREE_R4|320|evaluation|source_style_confirmation|1.0|None|
|J1_FREE_R4|160|fit|H|0.9777777777777777|0.0|
|J1_FREE_R4|320|fit|H|0.9555555555555556|0.02222222222222222|
|J1_FREE_R4|160|fit|U|0.49230769230769234|0.2262169312169312|
|J1_FREE_R4|320|fit|U|0.49743589743589745|0.19883597883597884|
|J1_FREE_R4|160|fit|cross_family_confirmation|1.0|None|
|J1_FREE_R4|320|fit|cross_family_confirmation|1.0|None|
|J1_FREE_R4|160|native|T0|1.0|None|
|J1_FREE_R4|320|native|T0|1.0|None|
|J2_FREE_R16|160|challenge|same_answer_context_not_conflict|0.7619047619047619|0.3|
|J2_FREE_R16|320|challenge|same_answer_context_not_conflict|0.7619047619047619|0.3|
|J2_FREE_R16|160|evaluation|H|0.5357142857142857|0.3076923076923077|
|J2_FREE_R16|320|evaluation|H|0.5357142857142857|0.3076923076923077|
|J2_FREE_R16|160|evaluation|U|0.47991452991452993|0.32927128427128427|
|J2_FREE_R16|320|evaluation|U|0.4851037851037851|0.31427128427128426|
|J2_FREE_R16|160|evaluation|cross_family_confirmation|0.9666666666666667|None|
|J2_FREE_R16|320|evaluation|cross_family_confirmation|0.9333333333333333|None|
|J2_FREE_R16|160|evaluation|source_style_confirmation|1.0|None|
|J2_FREE_R16|320|evaluation|source_style_confirmation|1.0|None|
|J2_FREE_R16|160|fit|H|0.9777777777777777|0.0|
|J2_FREE_R16|320|fit|H|0.9444444444444444|0.03333333333333333|
|J2_FREE_R16|160|fit|U|0.508058608058608|0.20875661375661375|
|J2_FREE_R16|320|fit|U|0.4970695970695971|0.22854497354497355|
|J2_FREE_R16|160|fit|cross_family_confirmation|1.0|None|
|J2_FREE_R16|320|fit|cross_family_confirmation|1.0|None|
|J2_FREE_R16|160|native|T0|1.0|None|
|J2_FREE_R16|320|native|T0|1.0|None|
|BE|320|formal_development|T1G|None|None|
|BE|320|formal_development|T1L|None|None|
|BE|320|formal_development|T2G|None|None|
|BE|320|formal_development|T2L|None|None|
|BE|320|formal_development|T3G|None|None|
|BE|320|formal_development|T3L|None|None|
|BE|320|formal_development|T4G|None|None|
|BE|320|formal_development|T4L|None|None|
|BE|320|formal_development|T5|None|None|
|BE|320|native|T0|None|None|
|W0|320|formal_development|T1G|None|None|
|W0|320|formal_development|T1L|None|None|
|W0|320|formal_development|T2G|None|None|
|W0|320|formal_development|T2L|None|None|
|W0|320|formal_development|T3G|None|None|
|W0|320|formal_development|T3L|None|None|
|W0|320|formal_development|T4G|None|None|
|W0|320|formal_development|T4L|None|None|
|W0|320|formal_development|T5|None|None|
|W0|320|native|T0|None|None|
|W01|320|formal_development|T1G|None|None|
|W01|320|formal_development|T1L|None|None|
|W01|320|formal_development|T2G|None|None|
|W01|320|formal_development|T2L|None|None|
|W01|320|formal_development|T3G|None|None|
|W01|320|formal_development|T3L|None|None|
|W01|320|formal_development|T4G|None|None|
|W01|320|formal_development|T4L|None|None|
|W01|320|formal_development|T5|None|None|
|W01|320|native|T0|None|None|
|W1|320|formal_development|T1G|None|None|
|W1|320|formal_development|T1L|None|None|
|W1|320|formal_development|T2G|None|None|
|W1|320|formal_development|T2L|None|None|
|W1|320|formal_development|T3G|None|None|
|W1|320|formal_development|T3L|None|None|
|W1|320|formal_development|T4G|None|None|
|W1|320|formal_development|T4L|None|None|
|W1|320|formal_development|T5|None|None|
|W1|320|native|T0|None|None|
|BE|320|formal_development|T1G|None|None|
|BE|320|formal_development|T1L|None|None|
|BE|320|formal_development|T2G|None|None|
|BE|320|formal_development|T2L|None|None|
|BE|320|formal_development|T3G|None|None|
|BE|320|formal_development|T3L|None|None|
|BE|320|formal_development|T4G|None|None|
|BE|320|formal_development|T4L|None|None|
|BE|320|formal_development|T5|None|None|
|BE|320|native|T0|None|None|
|W0|320|formal_development|T1G|None|None|
|W0|320|formal_development|T1L|None|None|
|W0|320|formal_development|T2G|None|None|
|W0|320|formal_development|T2L|None|None|
|W0|320|formal_development|T3G|None|None|
|W0|320|formal_development|T3L|None|None|
|W0|320|formal_development|T4G|None|None|
|W0|320|formal_development|T4L|None|None|
|W0|320|formal_development|T5|None|None|
|W0|320|native|T0|None|None|
|W01|320|formal_development|T1G|None|None|
|W01|320|formal_development|T1L|None|None|
|W01|320|formal_development|T2G|None|None|
|W01|320|formal_development|T2L|None|None|
|W01|320|formal_development|T3G|None|None|
|W01|320|formal_development|T3L|None|None|
|W01|320|formal_development|T4G|None|None|
|W01|320|formal_development|T4L|None|None|
|W01|320|formal_development|T5|None|None|
|W01|320|native|T0|None|None|
|W1|320|formal_development|T1G|None|None|
|W1|320|formal_development|T1L|None|None|
|W1|320|formal_development|T2G|None|None|
|W1|320|formal_development|T2L|None|None|
|W1|320|formal_development|T3G|None|None|
|W1|320|formal_development|T3L|None|None|
|W1|320|formal_development|T4G|None|None|
|W1|320|formal_development|T4L|None|None|
|W1|320|formal_development|T5|None|None|
|W1|320|native|T0|None|None|

See the closeout interpretation above. Completion alone is not effectiveness. No clinical or original-mechanism claim.
