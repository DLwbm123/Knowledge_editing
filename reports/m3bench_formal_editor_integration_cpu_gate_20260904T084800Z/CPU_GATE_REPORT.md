# M3Bench Formal Editor Integration CPU Gate

Status: `M3BENCH_FORMAL_EDITOR_INTEGRATION_CPU_GATE_PASS__GPU_APPROVAL_REQUIRED`

This gate is for the public-release-aligned T0-T4 dataset. It is not a paper-exact reproduction claim.

## Baseline and isolation

- Required parent: `066f64a5fad8efb39316e7f51f958e4c7e243b60`
- Child branch: `m3bench-formal-editor-integration-cpu-gate-20260904T082117Z`
- Baseline worktree status: clean
- Remote: `https://github.com/DLwbm123/Knowledge_editing.git`
- GPU used: no
- Model loaded: no
- Editing methods started: no
- Judge/evaluator started: no
- Frozen inputs or historical artifacts modified: no
- CPU/no-model tests: 65/65 passed

## Reproducible CPU gate result

All 22 frozen-data and integration checks passed, including the existing handoff manifest, 11,088-row unique inventory, task counts, event-key uniqueness, query resolution, referenced image readability and the one required image-binding hash pass.

| Scope | Result |
|---|---:|
| T0 ordered edits | 179 |
| Frozen prefixes | 1, 50, 100, 179 |
| Single events per method | 1,108 |
| Single raw outputs per method | 2,500 |
| Single raw outputs across four methods | 10,000 |
| Sequential-final raw outputs per method | 1,616 |
| Sequential-final raw outputs across four methods | 6,464 |

Single-event counts per method are T0 179, T1L 14, T1G 179, T2L 358, T2G 179, T3L 1, T3G 9, T4L 76 and T4G 113.

The only frozen ordered trajectory is T0. Sequential evaluation therefore uses T0 edits with T0/T1L/T1G/T2G probes. T2L/T3L/T3G/T4L/T4G do not have a frozen cross-task order and remain `NA_NO_FROZEN_CROSS_TASK_ORDER` for sequential metrics.

## Confirmed integration repairs

1. Added a task-specific `single-events` path so all 1,108 edit events can be executed without forcing every edit into T0.
2. Bound the editor runtime to the selected official-native loader and frozen generation contract.
3. Enforced the frozen effect-repaired configuration for each editor before a run can start.
4. Replaced the old physical-device allowance with an explicit GPU 2/3 allow-list, GPU 1 exclusion and mandatory UUID binding.
5. Changed runtime comparison from byte equality of host/provenance fields to exact model-topology and target-list equality.
6. Preserved the router-positive provenance for each edit. Of 1,108 events, 594 use a frozen legacy rephrase and 514 use a deterministic identity fallback because no frozen rephrase exists.

The 514 identity-fallback events are not silently treated as equivalent to legacy rephrases. The GPU plan requires a representative identity-fallback qualification before any formal run.

## Remaining manual GPU gate

Before GPU execution, the operator must approve the plan and an exact source checkout for the frozen official LLaVA-Med code commit must be established or independently verified. The current shared source directory has no Git metadata, so this check was intentionally deferred rather than inferred.

No GPU action is authorized by this report.
