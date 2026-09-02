# Provisional M3Bench metrics

Status: `M3BENCH_MINIMAL_DIAGNOSTIC_METRICS_READY`

Result label: `PROVISIONAL_AS_RUN_ORIGINAL_SEQUENCE_DIAGNOSTIC`

These implementation-diagnostic results use the original as-run target sequence. They are not the final audited/amended M3Bench result.

## Single-edit

| Method | T0 | T1L | T1G | T2L | T2G | T3L | T3G | T4L | T4G |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| LoRA | 25/200 (0.125000) | 68/1664 (0.040865) | 98/800 (0.122500) | NA | 88/800 (0.110000) | NA | NA | NA | NA |
| GRACE | 0/200 (0.000000) | 55/1664 (0.033053) | 5/800 (0.006250) | NA | 28/800 (0.035000) | NA | NA | NA | NA |
| BalanceEdit | 200/200 (1.000000) | 931/1664 (0.559495) | 800/800 (1.000000) | NA | 762/800 (0.952500) | NA | NA | NA | NA |
| BELoRA | 2/200 (0.010000) | 59/1664 (0.035457) | 7/800 (0.008750) | NA | 30/800 (0.037500) | NA | NA | NA | NA |

T5: `NA` (no legal T5 events in the manifest).

## Sequential-final

| Method | T0 | T1L | T1G | T2L | T2G | T3L | T3G | T4L | T4G |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| LoRA | 108/200 (0.540000) | 821/1664 (0.493389) | 433/800 (0.541250) | NA | 385/800 (0.481250) | NA | NA | NA | NA |
| GRACE | 0/200 (0.000000) | 55/1664 (0.033053) | 5/800 (0.006250) | NA | 28/800 (0.035000) | NA | NA | NA | NA |
| BalanceEdit | 200/200 (1.000000) | 1081/1664 (0.649639) | 694/800 (0.867500) | NA | 756/800 (0.945000) | NA | NA | NA | NA |
| BELoRA | 2/200 (0.010000) | 62/1664 (0.037260) | 7/800 (0.008750) | NA | 30/800 (0.037500) | NA | NA | NA | NA |

T5: `NA` (no legal T5 events in the manifest).

## Prefix trajectory

See `PROVISIONAL_M3BENCH_PREFIX_TRAJECTORY.csv` for prefix-1, 50, 100, and final numerator/denominator/value rows.

## Integrity

- parent commit: `80f9c3f1036040b2b531b651b7756f2bb306c717`
- existing parent aggregate SHA-256: `7d555ad3e9340797fa801e8c999c08151efdb03a15d875c113562b8fe8de8e64`
- scoring code commit: `91c38927ad89a4f3fdd78906c3f1241d8a8b6a62`
- final CSV SHA-256: `61787f0e5c67982c479390d2b4727dfddb4f0e8eeef772d3823b78c3ed636297`

BELoRA is a paper-spec independent reimplementation.
