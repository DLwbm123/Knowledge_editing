# GPT Pro review: fixed MedTRACE prototype calibration and E2E closeout

This is a correction on an already viewed development panel, not a blind test, V0.2 qualification, full TIME, or clinical validation. Historical LoRA `QUAL_VALIDATION_FAIL` remains unchanged.

1. **Root cause and repair.** Legacy R1/R2 calibration rebuilt prototypes from calibration positives while deployment used fit prototypes. The scorer is now frozen-prototype-only outside fit; 216 checkpoint/Q/cache bindings passed and R0 was numerically unchanged.
2. **Calibration change.** The selected configurations remain R0@200/SAFETY_FIRST and R2@800/SAFETY_FIRST. R2 calibration positive TPR changed from 98.6% to 93.8%; fixed hard and broad FPR are both 0.0%/0.0%. Full score differences are in `PROTOTYPE_SOURCE_AND_SCORE_PARITY.json`.
3. **Corrected E2E.** On the 180 native plus evaluation-positive executions, R0 is ON 144/180 and correct when gated 144/180. R2 is ON 169/180, forced-correct 126/180, and gated-correct 125/180. R2 hard/broad activation is 20/84 and 13/636; hard support comes from seven edits, not 21 independent facts.
4. **Reuse.** All 216 route checkpoints, 72 final experts, frozen base/forced outputs and 1756 exact Judge tuples were reused. No new answer required Judge; 7 full deterministic GPU replays verified the derivation contract.
5. **Mechanism judgment.** `R2_ROUTER_CALIBRATION_FIXED__END_TO_END_NOT_SUPPORTED_OVER_R0_ON_CURRENT_DEV`. The calibration defect is fixed, but R2 forced-on correctness remains the binding failure. The matched A2 → new-Q/old-P → recovered-output measurements support `TARGETED_Q_P_COUPLING_OR_OUTPUT_RECOVERY_CHANGE_NEEDED_BEFORE_EXPANSION` rather than further threshold tuning.

R0@200 versus R2@800 is the preregistered selected-configuration comparison, not a same-budget pure representation causal claim. Formal T1L/T1G/T2G, native, paraphrase, hard, broad and same-image challenge rows remain separate in `FIXED_E2E_METRICS.csv`.
