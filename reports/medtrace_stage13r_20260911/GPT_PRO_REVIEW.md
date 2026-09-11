# Stage13R frozen new-source comparison

## Verified closeout interpretation

All 7 new edits and 21 writer branches completed on GPU0. Training, generation, Judge preparation, Judge and reporting exited successfully. Wall time was 59.10 minutes; GPU process time was 0.975 GPU-hours. Judge coverage is 214/214 complete context-bound tuples (100 reused, 114 newly scored). No computation was rerun for this publication.

The main result is **a continuation of the H-supervision benefit on this small new-source cohort, not a deployment superiority claim**. One native edit remained incorrect under all three writers and was retained in every denominator.

| FORCED_ON metric | C_FACT | C_NO_H | BalancEdit adaptation |
|---|---:|---:|---:|
| Native correct | 6/7 (85.71%) | 6/7 (85.71%) | 6/7 (85.71%) |
| H-evaluation PairCorrect, edit macro | 85.71% | 0% | 0% |
| H-evaluation PairCorrect, pair micro | 7/9 (77.78%) | 0/9 | 0/9 |
| U-evaluation correctness, inherited hierarchical macro | 83.33% | 50.00% | 3.57% |
| U-evaluation correct inputs, micro | 22/28 | 13/28 | 1/28 |
| Damaged Base-correct H inputs | 0/5 | 5/5 | 5/5 |
| Damaged Base-correct U inputs | 1/22 | 10/22 | 21/22 |
| Source-style question correctness | 14/14 | 14/14 | 14/14 |
| Cross-family question correctness | 12/14 | 12/14 | 12/14 |
| Same-answer challenge correctness | 9/12 (75%) | 12/12 (100%) | 12/12 (100%) |

The primary C_FACT minus C_NO_H effect is +85.71 percentage points, with the inherited edit-paired bootstrap 95% interval [57.14,100.00] pp (10,000 draws, seed20260908). H pair four-cells, in the order both-correct/native-only/H-only/both-wrong, are 7/0/0/2 for C_FACT and 0/7/0/2 for each control. Shared H images make these seven edit outcomes correlated; the interval is not an independent-patient uncertainty estimate.

### Fixed deployment, not post-hoc system selection

| Deployment H PairCorrect, edit macro | C_FACT | C_NO_H | BalancEdit adaptation |
|---|---:|---:|---:|
| Original BE_ROUTE_R0, secondary | 78.57% | 7.14% | 7.14% |
| Predeclared RC_FIXED_OLD16, primary system | 64.29% | 64.29% | 64.29% |

Under RC, all 9 H, 28 U and 12 same-answer challenge inputs are OFF and return Base. All three systems have the same reported metrics: native85.71%, H64.29%, U84.52% (inherited hierarchical macro), source-style100%, cross-family85.71%, challenge33.33%, and zero damage among Base-correct H/U. Equal observed results and a zero-width paired interval do not prove general equivalence.

Relative to C_FACT FORCED_ON, RC loses 2 H corrections. On U it avoids 1 error and loses 1 correction. On same-answer challenge it loses 5 corrections. Thus RC protects against the controls' substantial errors but suppresses some of the stronger writer's useful changes. Do not combine forced writer benefits with routed zero damage as if one system achieved both. R0 remains a secondary, previously fixed analysis, not a replacement primary system.

### Source support and limits

- Full local original SLAKE train was inventoried:9,835 bilingual QA;16 fresh eligible images/151 English QA remained after actual-development, reserved-role, identity and quality exclusions. VQA-RAD raw train supplementation yielded no fresh image. Local SLAKE train was bound to author revision a9083ce6c34ac3ffb17671a605962924d8a8f9e9; no replacement download or changed gold was used.
- Two historical draft queues were PENDING with zero attempts; they were not treated as actual student exposure. Old Stage13's N0 conclusion about the875-row pool is unchanged. This is not a relaxation of sealed/evaluation-only boundaries.
- Prospective roles assigned10 images to adaptation and6 to evaluation. The actual run uses9 training images and4 evaluation/challenge images, globally disjoint. There are126 edit-input associations but only78 distinct realized inputs,13 images and29 question strings. The9 H pairs use only3 H images. Seven edits comprise5 body-region,1 organ-visibility and1 largest-visible-organ fact; this is not representative full-M3Bench or disease-diagnosis coverage.
- The importer initially capped challenge at1. Before evaluation generation, all qualifying challenge records in the already-frozen evaluation partition were restored (12 associations total), with original files preserved. Training selection, training files, image roles and source pool did not change; no model outputs were used for this correction. The original preparation-summary count of3 main evaluation images predates this recorded import correction; final evaluation including challenge uses4.
- Fit and evaluation answer files were separate. The training process ended before the evaluation process started. This is code-path separation, not an OS-level sandbox claim. Source review is agent source consistency, not human clinical certification. Patient identity and foundation-model pretraining exposure are UNKNOWN.

### Cost and evidence boundary

Shared native CP/A2/CP-W0 phases recorded 163.52/137.07/686.15 seconds, respectively. Paired continuation phases recorded C_FACT1048.28 seconds versus C_NO_H778.01 seconds, including their training-input endpoint generation; C_FACT used8,960 versus6,720 student forwards and69,440 versus59,840 labeled/generated-prefix tokens. The extra H supervision and compute are part of the intervention. BalancEdit recorded226.64 seconds under its own recipe and information budget; it is not a same-supervision or paper-exact comparison. C_FACT still loses3 same-answer challenge successes relative to either control.

Checks confirmed identical per-edit native/fit/U schedules, all14 freeR4 checkpoints at320 steps with73,728 parameters, matching evaluation input sets across methods, unchanged Base guards, and6 naturally observed adapter branch replays passing. Only these relevant artifacts were checked; no historical training, Base sweep or Judge rerun was performed. Timing excludes uninstrumented interactive source/engineering preparation.

Execution versions: source preparation7502240; training12da8f7; generation/reporting and challenge-import completion677342c. See CLOSEOUT_VALIDATION.json for complete SHAs. Preparation status fields below and nested in RUN_SUPPORT_COST.json are historical snapshots; RUN_STATUS.json is the completed-run status and PUBLICATION.json records release lineage.

## Original automatic report

{'status': 'COMPUTE_COMPLETE', 'actual_N': 7, 'completed_writers': 21, 'judge_required': 214, 'judge_scored': 214, 'judge_reused': 100, 'judge_new': 114, 'judge_missing': 0, 'publication': 'PENDING'}

Full original train expansion, not a repeat scan of the old875-row pool. See SOURCE_EXPANSION_SUMMARY.json for source revision and prospective roles.
Unexecuted draft queues are not actual exposure. All existing reservations, evaluation-only identities, real development inputs and quality exclusions remain protected.
Native and principal H/U evaluation images are outside previous MedTRACE development; training/evaluation image sets are globally disjoint. Patients and pretraining exposure remain UNKNOWN. Same-image paraphrases are text generalization only.
Original native CP -> A2 -> CP-W0 initialization is counted. C_FACT/C_NO_H use the same own-edit initial function and320-step schedules, differing only H supervision. BalancEdit is the frozen adaptation, not paper-exact or supervision-budget-matched.
FORCED_ON is writer behavior; R0/old fixed RC is deployment behavior. No threshold recalibration or algorithm selection.

## Primary paired effects

- {'mode': 'FORCED_ON', 'role': 'evaluation', 'metric': 'PairCorrect_edit_macro', 'paired_edits': 7, 'delta': 0.8571428571428571, 'ci_low': 0.5714285714285714, 'ci_high': 1.0, 'C_FACT_micro': 0.7777777777777778, 'C_NO_H_micro': 0.0, 'patients': 'UNKNOWN'}
- {'mode': 'BE_ROUTE_R0', 'role': 'evaluation', 'metric': 'PairCorrect_edit_macro', 'paired_edits': 7, 'delta': 0.7142857142857143, 'ci_low': 0.42857142857142855, 'ci_high': 1.0, 'C_FACT_micro': 0.6666666666666666, 'C_NO_H_micro': 0.1111111111111111, 'patients': 'UNKNOWN'}
- {'mode': 'RC_FIXED_OLD16', 'role': 'evaluation', 'metric': 'PairCorrect_edit_macro', 'paired_edits': 7, 'delta': 0.0, 'ci_low': 0.0, 'ci_high': 0.0, 'C_FACT_micro': 0.5555555555555556, 'C_NO_H_micro': 0.5555555555555556, 'patients': 'UNKNOWN'}

See writer/system tables for native, all-source H/U correctness, Base-correct damage and positive/challenge panels; pair four-cells/micro/edit-macro have explicit support. Small shared-image support is not independent patient evidence.
Source curation is agent source consistency, not human clinical certification. No sealed answers entered training; training finished in a separate process before evaluation references were read. No automatic next stage.
