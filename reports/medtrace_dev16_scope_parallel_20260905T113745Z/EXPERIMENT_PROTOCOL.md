# MedTRACE DEV16 and scope-pilot protocol

Status: `FROZEN_BEFORE_MODEL_OUTPUT`

## Boundaries

- Start commit: `b56b0633c030d4908bb67a21ae035aea2b61342c`.
- Execution-code commit: `3363035c2cb7b60397d0fc72ec4f6b40dc2dc5ed`.
- V0.1 SHA-256: `f6c07541345c7033d68c20f4dcd4ab21f928c08ea09c9cdde815986ae3bfecbd`.
- V0.2 SHA-256: `0ba02155eb658201e9c0d140c6bcd811587b3a44af4923e2b72eebf3583d179c`; conflicting V0.1 clauses are superseded.
- LoRA remains `QUAL_VALIDATION_FAIL`; LoRA QUAL inputs and probes are excluded.
- The historical rank-4 step-140 checkpoint remains read-only and is not substituted for any new DEV16 expert.
- This run is `TIME_INSPIRED_CP_R4_DEV16_FORCED_ON` plus a bounded MedTRACE scope-development pilot, not full TIME, full MedTRACE, or formal benchmark performance.

## Track A: frozen DEV16

All 16 frozen DEV events run in original order. Each receives an independent rank-4 CP expert at `model.layers.21.mlp.down_proj`, a new AdamW optimizer, and a stable seed derived as `SHA256(20260905 || NUL || record_id)`, first 32 bits masked to a non-negative 31-bit integer.

The observed module must be 14,336 to 4,096, factorized as 112x128 input and 64x64 output. CP beta is 1, epsilon is 1e-6, token RMS normalization is shared by route/execution contractions, and factor normalization preserves the materialized map.

AdamW is fixed at learning rate 1e-3, weight decay 0, betas 0.9/0.999, epsilon 1e-8, and gradient clipping 1. Training runs at most 200 optimizer steps per edit. Every 20 steps, and only on the native training input, the runner performs a greedy 128-token early-stop check. The first normalized full-target match with first target token rank 1 and no cap hit is selected; otherwise step 200 is retained without dropping the edit.

After selection, native plus every frozen associated T1L/T1G/T2G probe is generated once under the V4 native prompt and frozen greedy generation contract with up to 1,024 new tokens. Probes never enter loss, checkpoint selection, rank selection, layer selection, or early stopping. Each checkpoint is reloaded for one native replay. The hook is detached, request state is checked empty, the native base path is regenerated, and the sampled base guard is verified before the next edit.

Exact match, target-copy diagnostics, and semantic Judge verdicts remain separate. The current locked vLLM Judge is applied only after model generation finishes.

## Track B: source census and staged scope pilot

The primary edit remains the historical first DEV16 record. Source census scans the complete local SLAKE train/validation/test and VQA-RAD source files once, while excluding LoRA QUAL, frozen formal records, formal probes, and the seven previously exposed diagnostics from fit/calibration reuse.

Source annotations, not model behavior or string inequality alone, define candidates. UNKNOWN records remain unassigned. Model-visible EqKey uses processed image tensor bytes/dtype/shape/view order, image sizes, target-free routing token IDs, attention mask, and assistant predictor index. EqKey is an offline leakage lock only.

Execution mode is frozen before any scope scoring:

- `V02_ELIGIBLE` only if every applicable V0.2 gate is satisfied; the 4-positive/20-negative calibration minimum is necessary but not sufficient.
- `EXPLORATORY_EVALUABLE` requires nonempty, legally labelled and role-isolated fit/calibration/evaluation positives and negatives.
- `PARTIAL_COMPONENT_RUN` permits fit and native output recovery when fit labels exist but later roles are incomplete; missing calibration produces no threshold or FPR.
- `SOURCE_LABELS_INSUFFICIENT` stops only Track B training, never Track A.

The primary source QA is in SLAKE train. Existing source material provides source-supported negative candidates but no additional same-triple positive. Any added textual positives must use the separately labelled `MEDTRACE_NATIVE_TEXT_SCOPE_AUGMENTATION_PILOT`, preserve every medical qualifier, reveal no target answer, and be frozen before activation scoring.

If fit begins, only input factors train for exactly 200 steps with the single pre-registered pilot configuration: mean absolute contraction score, positive-vs-negative InfoNCE, max-negative/min-positive hinge margin 0.1, input orthogonality weight 0.01, temperature 0.1, AdamW 1e-3/0, and clip 1. Calibration is one-shot, strict `score > threshold`, target empirical FPR 0, then maximum TPR with higher-threshold tie-breaking. Evaluation cannot alter the threshold.

Output recovery freezes input factors, normalization, pooling, and any threshold; only output factors and rho may change. No all-factor normalization is allowed during that stage. Base, original-Q threshold control, final forced-on, and identical-final-expert intrinsic-gated paths are created only when their prerequisites exist.

## Stop conditions

Target leakage, contradictory EqKey labels, role reuse, non-finite values, base modification, cross-edit state contamination, GPU UUID mismatch, or infrastructure save/reload failure stops the affected track. Low accuracy, no threshold, all-OFF routing, or insufficient scope material remains a reported result and does not cancel Track A.
