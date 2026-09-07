# GPT Pro review: MedTRACE execution-preservation long run

Primary comparison: additional step 800, SAFETY_FIRST, viewed development panel.

- `C1_R2_FIXED_Q_LONG_RECOVERY`: positive forced 180/180 (micro 100.0%, edit macro 100.0%); gated 169/180 (micro 93.9%, edit macro 93.9%); joint ON+correct micro 93.9%, edit macro 93.9%; hard FPR 20/84 (micro 23.8%, edit macro 25.7%); broad FPR 13/636 (micro 2.0%, edit macro 1.9%).
- `C2_R2_SHARED_Q_JOINT`: positive forced 180/180 (micro 100.0%, edit macro 100.0%); gated 171/180 (micro 95.0%, edit macro 95.0%); joint ON+correct micro 95.0%, edit macro 95.0%; hard FPR 23/84 (micro 27.4%, edit macro 30.0%); broad FPR 14/636 (micro 2.2%, edit macro 2.1%).
- `C3_R2_SHARED_Q_EXECUTION_ANCHOR`: positive forced 180/180 (micro 100.0%, edit macro 100.0%); gated 166/180 (micro 92.2%, edit macro 92.2%); joint ON+correct micro 92.2%, edit macro 92.2%; hard FPR 23/84 (micro 27.4%, edit macro 30.0%); broad FPR 10/636 (micro 1.6%, edit macro 1.4%).

Status: `EXECUTION_RETENTION_NOT_IMPROVED`; `ROUTING_EXECUTION_PARETO_NOT_IMPROVED`.

Seeds repeat the same 12 facts; hard support is seven edits. These are DEV results, not blind or clinical validation.

Review whether C1 isolates recovery budget, C2 changes the routing/execution tradeoff, and C3 adds positive execution retention without hiding ALL_OFF behavior. Check edit-level denominators and the separate formal T1L/T1G/T2G rows in the CSV. Do not interpret this viewed panel as blind qualification.
