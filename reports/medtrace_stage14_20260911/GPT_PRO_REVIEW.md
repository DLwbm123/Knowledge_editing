# Stage14 fixed supervision-matched closeout

## Verified interpretation

Stage14 supports a targeted benefit of the selected H facts beyond this particular matched-count extra-QA control, not universal superiority or a better fixed-RC deployed system. Extra correct QA improves U behavior but does not reproduce the H benefit. The original Stage13R FACT/NO_H confirmation is unchanged; the new EXTRA comparison is a supplement on a previously published cohort.

Seven new C_EXTRA_QA trajectories completed on GPU1, each at320 steps and73,728 FP32 parameters. Their own CP-W0 initialization and native/fit/U schedules were preserved. All126 new edit-input associations match the corresponding C_FACT/C_NO_H/BE evaluation input sets. Four naturally encountered deployment replays passed. Judge coverage is1,416/1,416 full context-bound tuples:1,399 reused and17 new, with none missing.

### New7, forced writer comparison

|Metric|C_FACT|C_EXTRA_QA|C_NO_H|BalancEdit adaptation|
|---|---:|---:|---:|---:|
|Native correct|6/7|6/7|6/7|6/7|
|H-evaluation PairCorrect, edit macro|85.71%|0%|0%|0%|
|H-evaluation PairCorrect, pair micro|7/9|0/9|0/9|0/9|
|U correctness, inherited hierarchical macro|83.33%|72.62%|50.00%|3.57%|
|U correct inputs, micro|22/28|17/28|13/28|1/28|
|Damaged Base-correct H inputs|0/5|5/5|5/5|5/5|
|Damaged Base-correct U inputs|1/22|5/22|10/22|21/22|
|Corrected Base-wrong U inputs|1/6|0/6|1/6|0/6|
|Source-style correctness|14/14|14/14|14/14|14/14|
|Cross-family correctness|12/14|12/14|12/14|12/14|
|Same-answer challenge correctness|9/12|12/12|12/12|12/12|

The main C_FACT minus C_EXTRA_QA H-pair edit-macro difference is+85.71 percentage points, with the inherited edit-paired bootstrap interval[57.14,100.00]pp (7 edits,10,000 draws, seed20260908). H four-cells (both correct/native only/H only/both wrong) are7/0/0/2 for FACT and0/7/0/2 for EXTRA, NO_H and BE. The failed native is retained. These are not patient-independent estimates: the9 H pairs share only3 source images. Leaving each H source image out yields differences85.71pp (7 remaining edits/7 pairs),85.71pp (7/7), and50.00pp (2/4). This is sensitivity analysis, not a reliable three-cluster confidence interval.

Generic extra QA raises U macro correctness by22.62pp over NO_H, while FACT remains10.71pp above EXTRA and damages fewer initially correct U inputs. However, FACT loses3 same-answer challenge successes relative to all controls. Thus the result supports H-specific benefit on this narrow test, with a real retention/generalization tradeoff; it does not establish that every alternative correct QA selection would fail.

### Frozen deployment remains the bottleneck

|H PairCorrect, edit macro|C_FACT|C_EXTRA_QA|C_NO_H|BalancEdit adaptation|
|---|---:|---:|---:|---:|
|Original R0|78.57%|7.14%|7.14%|7.14%|
|Fixed old RC|64.29%|64.29%|64.29%|64.29%|

RC returns Base on all9 H,28 U and12 challenge inputs. All four therefore tie on reported deployment correctness, including U macro84.52% and challenge4/12. Its zero Base-correct H/U damage is a consequence of bypassing the writer, not four independent demonstrations of writer preservation. Relative to FACT forced outputs, RC loses2 H corrections, avoids1 U error while losing1 U correction, and loses5 challenge corrections. R0 remains the original secondary analysis; it was not selected to replace the predeclared RC primary system.

### Support, compute and recovery boundaries

- The7 controls all use SAME_H_IMAGE support with matched answer type (6 open,1 yes/no). They use4 unique original G QA across2 existing training images; the different-image subset is empty. Source matching is not statistical independence or equal information. Total answer/prefix training tokens differ: EXTRA66,880 versus FACT69,440, although both use8,960 student forwards and2,240 auxiliary CE calls. There is no exact equal-FLOPs claim.
- No old15 control was legal under the current cross-edit evaluation-image exclusion. Their historical full panels remain intact: native15/15 for all original writers; FACT H-pair macro53.57%, micro11/19 across14 pair-supported edits, versus0 for NO_H/BE. The old15 provide no new EXTRA comparison and cannot be pooled with new7 as22 independent new cases.
- The original GPU phases completed in22.12 minutes wall time, with0.3612 GPU-process hours. EXTRA continuation and training-input endpoint generation took1,113.39 seconds; reported FACT continuation time is historical and not a controlled speed comparison on shared GPUs.
- Training, generation, packet preparation and Judge initially exited0. The original report exited1 only because NFS extended-attribute copying failed after file-content copying. A one-line change from metadata-preserving copy2 to content-only copyfile completed the CPU report in13.71 seconds. No training, generation or Judge was rerun. The original failure log and exit codes are preserved; RUN_STATUS.json distinguishes the recovery.

Execution:a3b85bc89406cc2c7ce2af0ed1e851244e4d945b; CPU-only report recovery:fb94ccd190ebc46bce351032b0dc59b53dae9be5. Source preparation:1e56d47. Publication lineage is recorded separately. Public artifacts exclude original QA/images/answers/tokens, weights, activations, Judge mappings and private runtime paths. Source consistency review is not human clinical signoff, and patient identity/pretraining exposure remain UNKNOWN.

## Original automatic report

{'status': 'COMPUTE_COMPLETE', 'planned_new_trajectories': 7, 'completed_new_trajectories': 7, 'unsupported_old': 15, 'judge_required': 1416, 'judge_scored': 1416, 'judge_missing': 0, 'judge_reused': 1399, 'judge_new': 17, 'publication': 'PENDING'}

Only C_EXTRA_QA is new. OLD15 and NEW7 remain separate; the original Stage13R C_FACT/C_NO_H confirmation is unchanged. This extra-control comparison is a post-publication paired supplement, not an unseen prospective confirmation.
All historical panels are retained verbatim under original_full_panels. The old15 have no new G support because their training images also occur in the current combined evaluation image set. No role was relaxed, no historical sample was deleted or retrained.
The frozen source overlay uses original authorized correct QA and source-text-only different-proposition review. SAME_H_IMAGE does not imply equal information, statistical independence, equal answer tokens or equal FLOPs. CE slot count is matched; actual token/forward costs are reported.

|Cohort|Subset|Writer|Native|H pair edit-macro|H pair micro|Pairs|Edit support|
|---|---|---|---:|---:|---:|---:|---:|
|OLD15|FULL_COHORT|C_FACT|1.0|0.5357142857142857|0.5789473684210527|19|14|
|OLD15|FULL_COHORT|C_NO_H|1.0|0.0|0.0|19|14|
|OLD15|FULL_COHORT|C_EXTRA_QA|None|None|None|0|0|
|OLD15|FULL_COHORT|BE|1.0|0.0|0.0|19|14|
|OLD15|COMMON_SUPPORTED|C_FACT|None|None|None|0|0|
|OLD15|COMMON_SUPPORTED|C_NO_H|None|None|None|0|0|
|OLD15|COMMON_SUPPORTED|C_EXTRA_QA|None|None|None|0|0|
|OLD15|COMMON_SUPPORTED|BE|None|None|None|0|0|
|NEW7|FULL_COHORT|C_FACT|0.8571428571428571|0.8571428571428571|0.7777777777777778|9|7|
|NEW7|FULL_COHORT|C_NO_H|0.8571428571428571|0.0|0.0|9|7|
|NEW7|FULL_COHORT|C_EXTRA_QA|0.8571428571428571|0.0|0.0|9|7|
|NEW7|FULL_COHORT|BE|0.8571428571428571|0.0|0.0|9|7|
|NEW7|COMMON_SUPPORTED|C_FACT|0.8571428571428571|0.8571428571428571|0.7777777777777778|9|7|
|NEW7|COMMON_SUPPORTED|C_NO_H|0.8571428571428571|0.0|0.0|9|7|
|NEW7|COMMON_SUPPORTED|C_EXTRA_QA|0.8571428571428571|0.0|0.0|9|7|
|NEW7|COMMON_SUPPORTED|BE|0.8571428571428571|0.0|0.0|9|7|

## Paired effects

- {'cohort': 'OLD15', 'subset': 'COMMON_SUPPORTED', 'mode': 'FORCED_ON', 'role': 'evaluation', 'metric': 'PairCorrect_edit_macro', 'candidate': 'C_FACT', 'control': 'C_EXTRA_QA', 'paired_edits': 0, 'delta': None, 'ci_low': None, 'ci_high': None, 'comparison_identity': 'post-publication supplemental', 'patients': 'UNKNOWN'}
- {'cohort': 'OLD15', 'subset': 'COMMON_SUPPORTED', 'mode': 'FORCED_ON', 'role': 'evaluation', 'metric': 'PairCorrect_edit_macro', 'candidate': 'C_FACT', 'control': 'C_NO_H', 'paired_edits': 0, 'delta': None, 'ci_low': None, 'ci_high': None, 'comparison_identity': 'original fixed comparison reused', 'patients': 'UNKNOWN'}
- {'cohort': 'OLD15', 'subset': 'COMMON_SUPPORTED', 'mode': 'BE_ROUTE_R0', 'role': 'evaluation', 'metric': 'PairCorrect_edit_macro', 'candidate': 'C_FACT', 'control': 'C_EXTRA_QA', 'paired_edits': 0, 'delta': None, 'ci_low': None, 'ci_high': None, 'comparison_identity': 'post-publication supplemental', 'patients': 'UNKNOWN'}
- {'cohort': 'OLD15', 'subset': 'COMMON_SUPPORTED', 'mode': 'BE_ROUTE_R0', 'role': 'evaluation', 'metric': 'PairCorrect_edit_macro', 'candidate': 'C_FACT', 'control': 'C_NO_H', 'paired_edits': 0, 'delta': None, 'ci_low': None, 'ci_high': None, 'comparison_identity': 'original fixed comparison reused', 'patients': 'UNKNOWN'}
- {'cohort': 'OLD15', 'subset': 'COMMON_SUPPORTED', 'mode': 'RC_FIXED_OLD16', 'role': 'evaluation', 'metric': 'PairCorrect_edit_macro', 'candidate': 'C_FACT', 'control': 'C_EXTRA_QA', 'paired_edits': 0, 'delta': None, 'ci_low': None, 'ci_high': None, 'comparison_identity': 'post-publication supplemental', 'patients': 'UNKNOWN'}
- {'cohort': 'OLD15', 'subset': 'COMMON_SUPPORTED', 'mode': 'RC_FIXED_OLD16', 'role': 'evaluation', 'metric': 'PairCorrect_edit_macro', 'candidate': 'C_FACT', 'control': 'C_NO_H', 'paired_edits': 0, 'delta': None, 'ci_low': None, 'ci_high': None, 'comparison_identity': 'original fixed comparison reused', 'patients': 'UNKNOWN'}
- {'cohort': 'NEW7', 'subset': 'COMMON_SUPPORTED', 'mode': 'FORCED_ON', 'role': 'evaluation', 'metric': 'PairCorrect_edit_macro', 'candidate': 'C_FACT', 'control': 'C_EXTRA_QA', 'paired_edits': 7, 'delta': 0.8571428571428571, 'ci_low': 0.5714285714285714, 'ci_high': 1.0, 'comparison_identity': 'post-publication supplemental', 'patients': 'UNKNOWN'}
- {'cohort': 'NEW7', 'subset': 'COMMON_SUPPORTED', 'mode': 'FORCED_ON', 'role': 'evaluation', 'metric': 'PairCorrect_edit_macro', 'candidate': 'C_FACT', 'control': 'C_NO_H', 'paired_edits': 7, 'delta': 0.8571428571428571, 'ci_low': 0.5714285714285714, 'ci_high': 1.0, 'comparison_identity': 'original fixed comparison reused', 'patients': 'UNKNOWN'}
- {'cohort': 'NEW7', 'subset': 'COMMON_SUPPORTED', 'mode': 'BE_ROUTE_R0', 'role': 'evaluation', 'metric': 'PairCorrect_edit_macro', 'candidate': 'C_FACT', 'control': 'C_EXTRA_QA', 'paired_edits': 7, 'delta': 0.7142857142857143, 'ci_low': 0.42857142857142855, 'ci_high': 1.0, 'comparison_identity': 'post-publication supplemental', 'patients': 'UNKNOWN'}
- {'cohort': 'NEW7', 'subset': 'COMMON_SUPPORTED', 'mode': 'BE_ROUTE_R0', 'role': 'evaluation', 'metric': 'PairCorrect_edit_macro', 'candidate': 'C_FACT', 'control': 'C_NO_H', 'paired_edits': 7, 'delta': 0.7142857142857143, 'ci_low': 0.42857142857142855, 'ci_high': 1.0, 'comparison_identity': 'original fixed comparison reused', 'patients': 'UNKNOWN'}
- {'cohort': 'NEW7', 'subset': 'COMMON_SUPPORTED', 'mode': 'RC_FIXED_OLD16', 'role': 'evaluation', 'metric': 'PairCorrect_edit_macro', 'candidate': 'C_FACT', 'control': 'C_EXTRA_QA', 'paired_edits': 7, 'delta': 0.0, 'ci_low': 0.0, 'ci_high': 0.0, 'comparison_identity': 'post-publication supplemental', 'patients': 'UNKNOWN'}
- {'cohort': 'NEW7', 'subset': 'COMMON_SUPPORTED', 'mode': 'RC_FIXED_OLD16', 'role': 'evaluation', 'metric': 'PairCorrect_edit_macro', 'candidate': 'C_FACT', 'control': 'C_NO_H', 'paired_edits': 7, 'delta': 0.0, 'ci_low': 0.0, 'ci_high': 0.0, 'comparison_identity': 'original fixed comparison reused', 'patients': 'UNKNOWN'}

Interpret H selection using C_FACT minus C_EXTRA_QA jointly with native/H/U damage, source-style, cross-family and same-answer challenge tables. Confidence intervals crossing zero, native failures and unfavorable control comparisons do not trigger rescue or filtering.
Edit bootstrap uses the inherited 10,000 draws and seed20260908. NEW7 shares only three H images; LEAVE_ONE_H_IMAGE_OUT.csv reports reduced-support sensitivity, not a reliable patient-level confidence interval.
R0 and fixed old RC are parallel frozen deployment tradeoffs. Derived route outputs reuse exact bound Base/routes. RC returning Base on all H/U cannot establish individual writer protection; inspect ON counts and corrections lost. Original RC remains Stage13R predeclared primary system, R0 secondary.
BalancEdit is an external adaptation with different capacity and original supervision budget, not a matched-H or paper-exact baseline. Constructed questions are not official T2G, and same-answer different images are not official T1G.
Source consistency review is not human clinical signoff. Patients and pretraining exposure are UNKNOWN. No clinical, broad task or general deployment superiority claim; no additional method/seed/native/threshold or automatic next experiment.
