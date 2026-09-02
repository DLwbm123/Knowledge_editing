# Core-9 data and task coverage

Status: `M3BENCH_AMENDED189_CORE9_BLOCKED__T4L__NO_LEGAL_QUESTION_ROLE_BINDING`.

| Task | Candidate edits | Candidate probes | Eligible edits | Eligible probes | State |
|---|---:|---:|---:|---:|---|
| T0 | 189 | 189 | 189 | 189 | inherited frozen coverage; not regenerated |
| T1L | not recomputed | 1,610 | not recomputed | 70 | inherited frozen event coverage; not regenerated |
| T1G | not recomputed | 756 | not recomputed | 746 | inherited frozen event coverage; not regenerated |
| T2L | 56 | 1,496 | not frozen | not frozen | candidate relations recovered; base eligibility not run |
| T2G | not recomputed | 756 | not recomputed | 715 | inherited frozen event coverage; not regenerated |
| T3L | 33 | 189 | not frozen | not frozen | candidate relations recovered; base eligibility not run |
| T3G | 33 | 189 | not frozen | not frozen | candidate relations recovered; base eligibility not run |
| T4L | 0 | 0 | 0 | 0 | hard-stop: exact question-role binding absent |
| T4G | 3 | 15 | not frozen | not frozen | candidate relations recovered; base eligibility not run |

The inherited event counts are context from the frozen parent catalog, not a newly rebuilt Core-9 catalog. No Core-9 catalog was frozen because every task must have a nonzero eligible edit count.

Metric contract, when a legal catalog exists:

- primary: `PRIMARY_MACRO_PER_EDIT`, averaging each task over eligible edit requests;
- secondary diagnostic: `SECONDARY_MICRO_POOLED`;
- pooled event numerator/denominator must not be reported as the primary score;
- a T5-excluding harmonic mean, if later computed, must remain explicitly distinct from the paper's ten-task result.
