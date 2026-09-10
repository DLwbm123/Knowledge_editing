# Stage10 anatomical evidence development results

## Closeout interpretation

Compute is complete on the frozen supported subset: 5 of the 15 Stage9 edits, 10 E1/E2 trajectories at step160, Judge387/387 (386 reused, 1 new). The other10 edits have no eligible evidence and were not trained. E0 is exact cached Stage9 F1 on these same5 edits, not the full15-edit Stage9 average. Training evidence comprises6 instances from3 images, not6 independent patients.

|Same-support FORCED_ON metric (%)|E0 supervision baseline|E1 anatomical direction|E2 randomized direction|
|---|---:|---:|---:|
|Native correct|60.00|60.00|60.00|
|H evaluation correct|20.00|60.00|60.00|
|H base-correct damage (lower better)|80.00|40.00|40.00|
|U evaluation correct|64.62|64.62|64.62|
|U base-correct damage (lower better)|10.30|10.30|10.30|
|Cross-family evaluation correct|60.00|50.00|50.00|
|Source-style evaluation correct|100.00|100.00|100.00|
|Exact-question PairCorrect|20.00 (1/5)|20.00 (1/5)|20.00 (1/5)|

The observed H improvement is shared by the randomized-direction control; it does not establish an anatomical-direction-specific benefit. Native errors persist, PairCorrect does not improve, and cross-family performance falls10 percentage points. E1 versus E0 H correctness paired edit bootstrap interval is [0,80] percentage points; cross-family is [-30,0]. These small development-set intervals do not establish broad generalization. E1 and E2 have equal reported correctness/damage outcomes, not identical weights or scores.

Only2 eligible frozen evaluation probes per new method are available. E1 DeltaR-DeltaC is -0.01552 and +0.03193; E2 is -0.01153 and +0.03533. The mixed directions and similar sham results do not support a selective anatomical reliance claim. E0 occlusion diagnostics were not collected. Equal area/shape/fill does not imply equal pixel perturbation energy, which was not recorded. Forward-token counts concern training, not comprehensive generation/Judge/diagnostic cost. These are coverage limits, not reasons to rerun or search variants.

Stage9 cached supplement: PairCorrect F0=0/19, F1=F2=7/19 (36.84% pair-micro; 28.57% edit-macro over14 pair-supported edits). Local anatomical subset F1/F2 is1/7, global acquisition/region subset6/12. This different denominator must not be mixed with Stage10's1/5. Each F1/F2 has4 native errors exactly matching an H source answer; that output relation alone does not prove causal copying.

The original Judge launch was blocked by insufficient GPU0 free memory after training finished. Judge-only recovery reused all outputs and the existing tuple cache; no training or ordinary generation was replayed. Recorded elapsed wall time is3173.04 seconds (52.88 minutes), including the interruption/recovery, not pure GPU compute time. RUN_STATUS and STARTUP retain historical publication/startup states; the public Git commit containing this report is the release artifact. No next-stage experiment was launched. Private QA, images, masks, raw predictions, weights and Judge mappings are excluded.

{'status': 'COMPUTE_COMPLETE_SUPPORTED_SUBSET', 'completed_trajectories': 10, 'evidence_unsupported_trajectories': 20, 'judge_required': 387, 'judge_scored': 387, 'judge_missing': 0, 'judge_new': 1, 'judge_reused': 386, 'publication': 'PENDING'}

Same E support; E0 reuses original F1. Evidence does not enter ordinary inference. Official organ masks are not lesion labels; occluded R is not given a no/other-organ gold.
Stage9 PairCorrect and observed native-failure classification are separate cached-output supplements; not new algorithm evidence.
Source exposure is viewed development. Old7 original T2G unsupported. Missing organ ROI does not mean the organ is absent. No further method automatically launched.

|Method|Role|Panel|Correct macro|Damage macro|
|---|---|---|---:|---:|
|B0|challenge|same_answer_context_not_conflict|1.0|0.0|
|B0|evaluation|H|0.0|1.0|
|B0|evaluation|U|0.4615384615384615|0.4953535353535353|
|B0|evaluation|cross_family_confirmation|1.0|None|
|B0|evaluation|source_style_confirmation|1.0|None|
|B0|fit|H|0.0|1.0|
|B0|fit|U|0.41428571428571426|0.494047619047619|
|B0|fit|cross_family_confirmation|1.0|None|
|B0|native|T0|1.0|None|
|B2|challenge|same_answer_context_not_conflict|1.0|0.0|
|B2|evaluation|H|0.0|1.0|
|B2|evaluation|U|0.4923076923076923|0.4509090909090909|
|B2|evaluation|cross_family_confirmation|1.0|None|
|B2|evaluation|source_style_confirmation|1.0|None|
|B2|fit|H|0.0|1.0|
|B2|fit|U|0.42857142857142855|0.444047619047619|
|B2|fit|cross_family_confirmation|1.0|None|
|B2|native|T0|1.0|None|
|E0|challenge|same_answer_context_not_conflict|0.5|0.0|
|E0|evaluation|H|0.2|0.8|
|E0|evaluation|U|0.6461538461538462|0.10303030303030303|
|E0|evaluation|cross_family_confirmation|0.6|None|
|E0|evaluation|source_style_confirmation|1.0|None|
|E0|fit|H|0.6|0.4|
|E0|fit|U|0.45714285714285713|0.18523809523809523|
|E0|fit|cross_family_confirmation|0.7|None|
|E0|native|T0|0.6|None|
|E1|challenge|same_answer_context_not_conflict|0.5|0.0|
|E1|evaluation|H|0.6|0.4|
|E1|evaluation|U|0.6461538461538462|0.10303030303030303|
|E1|evaluation|cross_family_confirmation|0.5|None|
|E1|evaluation|source_style_confirmation|1.0|None|
|E1|fit|H|0.6|0.4|
|E1|fit|U|0.45714285714285713|0.18523809523809523|
|E1|fit|cross_family_confirmation|0.6|None|
|E1|native|T0|0.6|None|
|E2|challenge|same_answer_context_not_conflict|0.5|0.0|
|E2|evaluation|H|0.6|0.4|
|E2|evaluation|U|0.6461538461538462|0.10303030303030303|
|E2|evaluation|cross_family_confirmation|0.5|None|
|E2|evaluation|source_style_confirmation|1.0|None|
|E2|fit|H|0.6|0.4|
|E2|fit|U|0.45714285714285713|0.18523809523809523|
|E2|fit|cross_family_confirmation|0.6|None|
|E2|native|T0|0.6|None|
