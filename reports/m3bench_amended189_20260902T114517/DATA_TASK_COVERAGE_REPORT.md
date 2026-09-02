# Data and Task Coverage Report

Status: `BLOCKED__DATA_OR_TASK_COVERAGE`.

The approved protocol requires a nonzero, explainable denominator for every task from T0 through T5 before any smoke, qualification, or formal GPU run. Six required task types currently have a zero denominator.

| Task | Candidate events | Eligible denominator | Source status |
|---|---:|---:|---|
| T0 | 189 | 189 | available from frozen parent |
| T1L | 1,610 | 70 | available from frozen parent |
| T1G | 756 | 746 | available from frozen parent |
| T2L | 0 | 0 | metadata exists, but frozen parent scope resolves none |
| T2G | 756 | 715 | available from frozen parent |
| T3L | 0 | 0 | metadata exists, but frozen parent scope resolves none |
| T3G | 0 | 0 | metadata exists, but frozen parent scope resolves none |
| T4L | 0 | 0 | metadata exists, but frozen parent scope resolves none |
| T4G | 0 | 0 | metadata exists, but frozen parent scope resolves none |
| T5 | 0 | 0 | no frozen task catalog or source-image set is ready |

The currently available frozen catalog contains 3,311 candidate events and 1,720 eligible events. It is method-agnostic, so the four planned method groups would receive the same event keys. That symmetry does not cure the missing task types.

T2L, T3L/G, and T4L/G require an authoritative cohort reconstruction and frozen base-eligibility adjudication over the complete source inventory. T5 additionally requires the missing source-image/task assets. Neither condition is currently satisfied.

Gate decision: do not start smoke8, held-out8 qualification, or the formal rerun. A reduced-scope run would require a separate protocol amendment and could not be reported as full M3Bench.
