# Audit Hypothesis Disposition

Prior audit statements were treated as hypotheses and checked against the frozen handoff and current code.

| Hypothesis | Disposition | Evidence / action |
|---|---|---|
| Final cohorts are missing or zero | Rebutted | All T0-T4 cohort files are present, nonzero and match frozen counts. |
| The amended ordered sequence has 189 edits | Rebutted | The finalized T0 sequence is dense and exact at 179 edits. |
| The old T0-only runner can execute every finalized task | Confirmed blocker | T2L/T3/T4 contain task-specific edit instances outside T0. Added a task-specific single-event catalog and runner. |
| Every task can be assigned a sequential position | Rebutted | Only T0 has a frozen order. Unsupported task-specific sequential metrics remain NA. |
| The selected official-native runtime was enforced by the formal runner | Confirmed blocker | The old runner instantiated the project/manual loader. It now requires the canonical runtime lock and official-native loader. |
| Frozen editor configurations were enforced at runtime | Confirmed blocker | The old runner recorded a bundle hash but did not compare instantiated configuration. Exact per-method config checks were added. |
| The previous GPU guard was sufficient | Confirmed blocker | It allowed GPU 1 and did not bind the expected UUID. The guard now permits only approved GPU 2/3 and requires UUID equality. |
| Full inventory JSON equality is the correct runtime test | Rebutted in part | Device, path and classification fields can change without topology drift. The gate now ignores only those host/provenance fields and still rejects module-shape or target-list drift. |
| Existing legacy rephrases cover every finalized edit | Rebutted | 594/1,108 events bind to a frozen legacy rephrase; 514 require a disclosed identity fallback and GPU qualification. |

No change was made merely to match a prior audit conclusion.
