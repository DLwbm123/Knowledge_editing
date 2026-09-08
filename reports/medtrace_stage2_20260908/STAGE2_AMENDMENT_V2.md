# Stage2 prospective amendment V2

Frozen before new-episode student evaluation, 2026-09-08. This is a development comparison, not a retrospective registration of P4 superiority.

- Preserve Stage1's 70 trajectories, seven facts, checkpoints, original primary analyses and qualified flags. No retraining/rejudging that grid. Its 26 original T2G inputs are already visible and cannot become blind, fit, calibration, lambda or checkpoint selectors.
- Replace the V1 L16-primary continuation by the fixed new-fact panel: P4 W0, P4 W1 normalized lambda=.1, P4 W2, L16 W2, and frozen BalancEdit V4 adaptation. L16 W1 lambda=1 remains an unqualified technical fallback, not a qualified strong baseline.
- New CP seed base 20260910. Each new fact gets native CP-R4 layer21 down_proj at most 200 steps with original native-only stopping, followed by 80 native+approved-fit steps to common A2. Four writers use step320 endpoint; 80/160 diagnostics cannot select the endpoint. BE independently starts from frozen Base for 50 steps using its existing per-record seed function.
- Exact V1 losses, fit-only full-vocabulary FP32 Base||ON KL and optimizer are reused from `methods/medtrace/selective_write.py` and `run_selective_write.py`. No new lambda/rank/layer/optimizer sweep. W0 omits negative backward; compute is not FLOP-matched. L16 is not capacity-matched to P4.
- Freeze legal source/role/exposure and paraphrase-family eligibility before students. Include initialization failures in the eligible denominator. Actual N may be below 16; source gaps never cancel old DEV16 BE.
- Forced-on is the mechanism panel. Independently actual BE_NATIVE_ROUTED is the system panel. New CP uses the same pre-edit Base-feature BE decisions, not test gold/task/expected-expert lookup. Shared-gate FPR is identical by definition. No new router, sequential, full M3Bench reproduction or clinical claim.
- Native and both cross-family text panels must be interpreted jointly with source-correct damage, H_all/H_keep/U and original locality. A 5pp point-estimate reference is not a confidence-bound proof of noninferiority. Zero support is NA.
- GPU2/3 sharing is authorized at sufficient free VRAM; no unrelated process management. 24h wall/48 GPU-h upper cap; after 20h prioritize Judge/closure. One bounded detached pipeline, no cron or recurring monitoring.

Track A is old DEV16, not new confirmation: frozen BalancEdit full last-layer up_projection, per-edit FP32 full copy, Adam lr=.01, 50 steps, alpha=.2, Euclidean radius `(1-alpha)*negative_distance + alpha*positive_distance`, native/official-rephrase/black-image Base anchors. Training reads no evaluation probes. Both routes are actual generation. The first normal task also performs target-free no-edit, module binding, save/unload/reload and request-lifecycle integration checks; low semantic scores do not gate the remaining queue.

Publication contains only source, frozen non-private config and aggregate reports. QA/images/raw answers/tokens/weights/teacher distributions/Judge maps/full logs remain private. Research execution, subsequent repair and public release commits are separate provenance fields.
