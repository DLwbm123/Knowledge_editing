# Implementation diagnostic

Status: `M3BENCH_MINIMAL_DIAGNOSTIC_METRICS_READY`

Result label: `PROVISIONAL_AS_RUN_ORIGINAL_SEQUENCE_DIAGNOSTIC`

## Findings

- T0 near zero (<=0.05): GRACE, BELoRA.
- Locality collapse on available T1L (<=0.10): LoRA, GRACE, BELoRA.
- Single-to-sequential-final degradation >=0.15: none.
- Exactly identical method metric vectors: none.
- Zero-denominator metric families: sequential-final:T2L, sequential-final:T3G, sequential-final:T3L, sequential-final:T4G, sequential-final:T4L, sequential-final:T5, single-edit:T2L, single-edit:T3G, single-edit:T3L, single-edit:T4G, single-edit:T4L, single-edit:T5.
- Same raw answer in all four anonymous groups: 0/9665 symmetric event keys.
- LoRA: empty=5/9665; truncation-suspect=0; invalid-generation-marker=0.
- GRACE: empty=0/9665; truncation-suspect=0; invalid-generation-marker=0.
- BalanceEdit: empty=60/9665; truncation-suspect=0; invalid-generation-marker=0.
- BELoRA: empty=0/9665; truncation-suspect=0; invalid-generation-marker=0.
- Method-output join error: not indicated by key parity/raw-equality checks.

The truncation count is a conservative suffix heuristic because the event manifest does not expose a finish reason.

## Integrity

- parent commit: `80f9c3f1036040b2b531b651b7756f2bb306c717`
- existing parent aggregate SHA-256: `7d555ad3e9340797fa801e8c999c08151efdb03a15d875c113562b8fe8de8e64`
- scoring code commit: `91c38927ad89a4f3fdd78906c3f1241d8a8b6a62`
- final CSV SHA-256: `61787f0e5c67982c479390d2b4727dfddb4f0e8eeef772d3823b78c3ed636297`

Parent raw was read-only and unchanged; GPU editing was not rerun.

These results are implementation diagnostics on the original as-run target sequence, not final audited/amended M3Bench results.
