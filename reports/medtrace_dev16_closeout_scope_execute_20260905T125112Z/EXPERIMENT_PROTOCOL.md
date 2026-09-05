# MedTRACE DEV16 closeout and scope execution protocol

Status: `FROZEN_BEFORE_NEW_MODEL_OUTPUT`

- Continuation starts from public prelaunch commit `d7ec198966cc2f95d7205e357346bd21cd17dc13`; the completed DEV16 generation used `d9a5af30be7ea12b38c3576f6f81c5d111ce97ea`.
- The 16 existing experts and 163 existing outputs are read-only. Track A prepares one method-blind packet for all 163 CP outputs plus the 59 corresponding saved T2G base outputs; no backbone generation is repeated.
- Judge preflight tokenizes the complete packet on CPU, includes the fixed 24-token output budget, and chooses one shared 1,024/2,048/4,096-token lane before vLLM loads. Truncation is forbidden.
- The scope census searches only the authorized SLAKE training split for supervision. Other loaded splits remain audit-only. Full eligible relation counts are recorded before a hard-first stratified 120-row retention cap.
- Text augmentation contains 16 source-question-only candidates: 12 independently checklist-approved paraphrases, split by non-overlapping rewrite family into fit/calibration/evaluation groups of four, and four rejected non-equivalents.
- Negative roles contain 20 source-annotated rows each, grouped by full source-image identity with no image crossing roles. Patient-disjointness is not claimed.
- All 72 scope fit/calibration/evaluation inputs receive clean-base, target-free layer-21 activations and model-visible EqKeys before scoring. Any EqKey collision, role conflict, or label conflict stops scope execution.
- Input fitting uses the registered InfoNCE + margin + input-orthogonality loss for exactly 200 steps. Calibration uses strict `score > threshold` and empirical FPR 0. Output recovery freezes Q and threshold and updates only output factors and rho on the native edit.
- Evaluation compares Base, original-Q threshold control, final forced-on, and the identical final expert with intrinsic gating. OFF requests execute the base path and must match base tokens exactly.
- This is `TIME_INSPIRED_CP_R4_DEV16_FORCED_ON` plus `MEDTRACE_NATIVE_TEXT_SCOPE_AUGMENTATION_PILOT`, not full TIME, full MedTRACE, V0.2 qualification, patient-level generalization, or a formal benchmark.
