# GPU Execution Plan — Approval Required

Status: `GPU_APPROVAL_REQUIRED`

## Hard gates before model load

1. Check out or independently verify the frozen official LLaVA-Med source commit recorded by the canonical runtime lock.
2. Confirm one visible device only, physical GPU 2 or 3, and bind its exact UUID. GPU 1 is forbidden.
3. Re-read the immutable CPU-gate package and refuse any existing output destination.
4. Confirm no editor, Judge or evaluator process is active for this run.

## Qualification

For each of the four editors, run two isolated non-formal events: the first frozen T0 event and the first task-specific event whose router-positive source is `identity_fallback_no_frozen_rephrase`.

Require for every qualification event:

- nonempty generation;
- finite loss and gradients;
- unchanged base weights;
- save/reload generation, route and NLL parity;
- correct route contract;
- exact frozen runtime topology, target list, generation contract and editor configuration.

Any failure is a hard stop. Qualification outputs must be separate from formal outputs.

## Formal raw phase after explicit approval and qualification PASS

1. Run all task-specific single events in frozen file order, using non-overwriting chunks of at most 25 events.
2. Run one T0 sequential trajectory per editor at prefixes 1, 50, 100 and 179.
3. Report T2L/T3L/T3G/T4L/T4G sequential metrics as `NA_NO_FROZEN_CROSS_TASK_ORDER`; do not invent an ordering.
4. Close raw counts and structural manifests before any Judge or evaluator is started.
5. Create a method-blind Judge payload only after raw closure and use the independent packet in this directory.

This plan does not authorize GPU execution.
