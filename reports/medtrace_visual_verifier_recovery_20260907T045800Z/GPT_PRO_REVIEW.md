# GPT Pro review: frozen-C1 visual-verifier recovery

Computation and evaluation are complete: 21 edit/seed tasks, 42 verifier fits closed. Seven source facts are repeated across three seeds. This is viewed development diagnosis, not full TIME/MedTRACE or clinical validation.

|Condition|Matched hard macro FPR|Matched positive joint ON+correct|T1G joint|T2G joint|Base-correct negative damage|
|---|---:|---:|---:|---:|---:|
|M0|18.10%|80.95%|34.52%|39.29%|1/177|
|M1|12.14%|67.86%|30.95%|34.92%|2/177|
|M2|12.26%|66.67%|34.52%|41.67%|1/177|
|M3|9.76%|77.38%|50.00%|53.17%|3/177|

The same matched panel and PRIMARY_SAFETY_FIRST operating point are compared. Native, original evaluation positives, T1G/T1L/T2G, broad and same-image-other-fact results remain separate in RESULTS_BY_EDIT_SEED.csv and RESULTS_MACRO_MICRO.json. Gated correctness and joint ON+correct are separate quantities.

## Evidence for the three routing explanations

- Compensating mean fusion versus conjunction: M1 reduces hard FPR by 5.95 percentage points but loses 13.10 points of positive joint success relative to M0. This is consistent with conjunction suppressing some compensating-score errors, but it does not establish mean fusion as the sole cause or provide a retained solution.
- Usable information in the existing four-dimensional CP response: M2 versus M1 changes hard FPR by +0.12 points and positive joint success by -1.19 points. This supervised CP4 readout does not improve the primary matched-panel trade-off over conjunction. It improves T1G/T2G relative to M1, so the evidence is mixed; it does not show that all CP4 information is unusable.
- Pre-CP versus compressed feature availability: M3 versus M2 lowers hard FPR by 2.50 points and raises positive joint success by 10.71 points. Its image-head evaluation accuracy is also higher (89.48% versus 85.32%). This supports a benefit from the pre-CP representation under this fixed supervision, but input width and parameter count differ (28,674 versus 10), so irreversible information loss is not established.

M1 changes only the decision rule. M2 adds supervised linear readouts on CP4; improvement would demonstrate usable information under this supervision, not prove a need for larger CP. M3 changes readout input width and parameter count while matching pooling and data; any improvement is diagnostic and cannot by itself establish information-theoretic loss. Fit/calibration/evaluation gaps are reported in ROUTER_TRAINING_AND_PARAMETER_REPORT.md.

Predeclared development retention signal passed: False. Even M3 lowers hard FPR by only 8.33 points versus the required 10, loses 3.57 points of positive joint success versus the allowed 2, and increases base-correct negative damage from 1/177 to 3/177. No condition is retained. See GENERALITY_SAFETY_TRADEOFF.md for every criterion and paired edit-cluster intervals. All OFF/ON outcomes and all failures/retries remain disclosed; no performance-based resampling or retraining was used.

Fit provenance: {'NEW_800_STEP_FIT': 6, 'VALIDATED_REUSE': 36}. Earlier compatible heads/outputs were reused only after binding checks. M0–M3 scores and calibration were recalculated; the first original task was fully computed anew. 7 NFS metadata failures were repaired as an engineering retry, not a method failure.

C1 Q/P/rho and V4 inputs/generation remained frozen. Historical LoRA QUAL_VALIDATION_FAIL, C1/C2/C3, A2 and Judge/reference disagreements were not changed. Public files exclude private QA, images, tokens, features, checkpoints, Judge mappings and full logs.
