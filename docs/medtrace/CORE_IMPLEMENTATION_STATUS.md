# MedTRACE core implementation status

Status: `MEDTRACE_ZERO_EFFECT_PASS__MEDTRACE_CP_EXPRESSIVITY_PASS`

Implemented: asymmetric CP residuals, shared token-wise RMS-normalized route/execution contraction, factor normalization, assistant-only layer hook, zero-effect disable path, and FPR-constrained threshold calibration metadata.

Real-model evidence: zero-effect parity passed on eight unique frozen V4 DEV queries. A one-layer rank-4 forced-on CP expert passed at step 140, including frozen semantic Judge correctness, first-token rank 1, unrestricted/short manual-no-cache/cached/HF parity, disable, reload, dense/factor parity, and frozen-base checks.

Not yet claimed: intrinsic scope, trajectory MMD, HSIC, curvature projection, sequential editing, full V0.2 completion, or formal benchmark performance. Intrinsic scope is paused because the available development event lacks the source-image/fact/equivalence-group calibration attribution required by the binding specification.
