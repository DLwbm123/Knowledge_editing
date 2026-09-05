# MedTRACE core implementation status

Status: `MEDTRACE_ZERO_EFFECT_PASS__MEDTRACE_CP_EXPRESSIVITY_PASS__REQUEST_LIFECYCLE_PASS`

Implemented: asymmetric CP residuals, shared token-wise RMS-normalized route/execution contraction, factor normalization, assistant-only layer hook, explicit per-generation-request lifecycle, zero-effect disable path, and FPR-constrained threshold calibration metadata.

Real-model evidence: zero-effect parity passed again on eight unique frozen V4 DEV queries after request isolation. The unchanged one-layer rank-4 step-140 forced-on CP expert passed native/short and reverse-order replay, including first-token rank 1, normalized literal target matches, manual-no-cache/cached/HF parity, disable, reload, dense/factor parity, and a 686-tensor sampled frozen-base guard.

Not yet claimed: intrinsic scope, trajectory MMD, HSIC, curvature projection, sequential editing, full V0.2 completion, or formal benchmark performance. Source-image and formal-relation provenance has been backfilled for the development event, but scope remains `NOT_READY/NOT_RUN`: there are no authorized scope-fit negatives or independent 4-positive/20-negative calibration roles, and V0.2 model-visible EqKeys are unavailable.
