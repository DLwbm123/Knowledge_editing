# MedTRACE generation request lifecycle review

Status: `PASS`

## Root cause

`MedTraceLayerHook` retained `generation_boundary` across calls. Sequence growth is sufficient to recognize continuation inside one manual no-cache request, but it cannot distinguish that continuation from a separate, longer prompt. The old `all_paths()` called native and short prompts under one mutable generation state, so all three generation paths could agree while sharing the same stale mask.

The nonzero-residual CPU reproduction demonstrated the failure shape: a length-4 request correctly edited predictor 3; a separate length-7 prompt using the stale boundary edited predictors 3-6 rather than only predictor 6.

## Minimal shared fix

- The hook now exposes an explicit `generation_request()` context with begin/end operations.
- Cleanup is guaranteed by `try/finally`, including exception exits.
- Manual no-cache, manual cached, and Hugging Face generation each receive a fresh request lifecycle.
- A full-sequence no-cache continuation keeps its first assistant predictor boundary within that request.
- One-token cached forwards remain active.
- Detach clears all live routing state.

The change is in the shared hook and its shared generation caller, not duplicated in prompt-specific call sites.

## CPU evidence

Five MedTRACE core tests passed. The lifecycle test uses nonzero `rho` and covers length 4 then a fresh length 7, reverse order, exception cleanup, same-request no-cache growth, cached one-token routing, teacher-forced shift, disable, detach, and reset. The existing dense/factor, gradient, normalization, zero-effect, optimizer-boundary, and threshold-metadata checks also passed.

## Real-model evidence

The unchanged rank-4 step-140 checkpoint was loaded on the frozen V4 LLaVA-Med backbone. No training occurred.

| Request/order | Expanded prompt | First active predictor | Final no-cache active span | Cached decode span | Three-path parity |
|---|---:|---:|---|---|---|
| Native, first | 603 | 602 | 602-612 | 0 | pass |
| Short, second | 620 | 619 | 619-629 | 0 | pass |
| Short, first after reload | 620 | 619 | 619-629 | 0 | pass |
| Native, second after reload | 603 | 602 | 602-612 | 0 | pass |

Native and short each produced 11 identical tokens across manual no-cache, manual cached, and Hugging Face cached generation. Both normalized literal target checks passed. Save/reload, disable/detach, and dense/factor parity passed. The base guard found no change among its 686 sampled tensors; this is a sampled guard, not a claim of exhaustive byte comparison.

The eight-query zero-effect canary was rerun with a fresh lifecycle for each query and passed frozen-base matching plus base/disabled/active-zero/detached exact identity.

## Boundary

This closes the cross-request mask-state defect. It preserves the prior forced-on CP expressivity result; it does not establish intrinsic scope, multi-edit behavior, or formal benchmark performance.
