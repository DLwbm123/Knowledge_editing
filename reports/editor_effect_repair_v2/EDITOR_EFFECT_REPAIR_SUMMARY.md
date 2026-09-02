# M3Bench Editor Effect Repair V2

Status: `M3BENCH_EDITOR_EFFECT_REPAIR_PASS__FORMAL_RERUN_BLOCKED_BY_GOVERNANCE`.

| Method | T0/8 | Raw changed | Median NLL decrease | Route hit | Empty | 4-edit run | Fresh replay |
|---|---:|---:|---:|---:|---:|---|---|
| LoRA | 1/8 | 6/8 | 2.0269 | 8/8 | 0 | PASS | PASS |
| GRACE | 0/8 | 8/8 | 1.4353 | 8/8 | 0 | PASS | PASS |
| BalanceEdit | 8/8 | 8/8 | 3.2573 | 8/8 | 0 | PASS | PASS |
| BELoRA | 6/8 | 8/8 | 3.1998 | 8/8 | 0 | PASS | PASS |

BELoRA is a paper-spec independent reimplementation; 50 steps/edit is an explicit deviation from the 5-step paper-spec setting because 5–20 steps were a generation no-op on the approved smoke cohort. The first tested checkpoint that changed generation was 50.

All one-edit and 4-edit effect checks passed, all outputs were nonempty, and base weights remained unchanged. Formal-200 and scoring were not started because the current sequence authority is C.
