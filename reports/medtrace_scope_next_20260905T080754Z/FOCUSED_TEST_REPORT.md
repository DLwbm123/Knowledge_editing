# MedTRACE focused test report

Status: `PASS`

## CPU checks

Command: `python scripts/medtrace/verify_core.py`

Result: 5 tests passed.

Coverage includes:

- nonzero-residual request isolation for length 4 then fresh length 7 and reverse order;
- same-request no-cache growth and cached one-token routing;
- exception, disable, detach, and reset cleanup;
- teacher-forced assistant predictor shift;
- factorized/dense forward and gradient parity;
- factor normalization and save/reload invariance;
- zero residual exact bypass;
- CP-only optimizer membership and nonzero gain gradient;
- threshold calibration metadata and deterministic tie-breaking.

`python -m py_compile scripts/medtrace/run_fixed_judge_vllm.py` and `git diff --check` also passed.

## Real-model checks

- Rank-4 step-140 CP checkpoint: `MEDTRACE_CP_ENGINEERING_PASS`.
- Native and short requests: normalized literal target match and three-path token parity.
- Reverse request order after reload: pass.
- Eight-query zero-effect rerun: `MEDTRACE_ZERO_EFFECT_PASS`.
- One-row vLLM Judge development run: output parsed, preserved `is_correct=true`, and produced a separate execution lock; no cross-backend bridge was run.
- No retraining, rank search, full G1R, or 11,088-query base rerun was performed.

Private prompts, answers, images, token sequences, checkpoint tensors, and raw Judge payloads remain on the experiment server.
