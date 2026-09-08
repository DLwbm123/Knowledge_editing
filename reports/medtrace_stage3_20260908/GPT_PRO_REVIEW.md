# MedTRACE Stage3 factual review

Status: PARTIAL; compute=COMPLETE_EXECUTABLE; Judge=COMPLETE_CURRENT_TUPLES; publication=PENDING.
S1 = BE_ROUTE + P4-W1_KL_0.1; S0 = same route + P4-W0_TASK_ONLY; BE = native BalancEdit V4 adaptation.

1. How does S1 perform on original V4 generalization and locality?

| Method | Prefix | Role / panel | Strict role | Inputs | V4 primary macro (eligible) | All-source macro | All-source micro | Damage macro |
|---|---:|---|---|---:|---:|---:|---:|---:|
| S1 | 0 | formal_development / T1G | EDIT_TARGET | 22 | 1.0000 (22) | 1.0000 | 1.0000 | NA |
| S1 | 0 | formal_development / T1L | STRICT_BASE | 10 | 0.0000 (6) | 0.2000 | 0.2000 | 1.0000 |
| S1 | 0 | formal_development / T2G | EDIT_TARGET | 23 | 0.3056 (23) | 0.3056 | 0.3043 | NA |
| S1 | 0 | formal_development / T2L | STRICT_BASE | 2 | 1.0000 (2) | 1.0000 | 1.0000 | 0.0000 |
| S1 | 0 | formal_development / T3G | UNKNOWN | 0 | NA (0) | NA | NA | NA |
| S1 | 0 | formal_development / T3L | UNKNOWN | 0 | NA (0) | NA | NA | NA |
| S1 | 0 | formal_development / T4G | UNKNOWN | 0 | NA (0) | NA | NA | NA |
| S1 | 0 | formal_development / T4L | UNKNOWN | 0 | NA (0) | NA | NA | NA |
| S1 | 0 | formal_development / T5 | UNKNOWN | 0 | NA (0) | NA | NA | NA |
| S1 | 0 | native / NATIVE_DIAGNOSTIC | EDIT_TARGET | 1 | NA (0) | 1.0000 | 1.0000 | NA |
| S1 | 0 | native / T0 | EDIT_TARGET | 6 | 1.0000 (6) | 1.0000 | 1.0000 | NA |

Official task labels remain separate from fit and source-style/cross-family confirmation. T5 is NA; no official Overall is computed.
Primary metric mapping reuses m3bench_repro/evaluation/metrics.py: T0 reliability=postcorrect; T*G=postcorrect among Base-wrong; T*L=postcorrect among Base-correct. Primary macro averages eligible probes per edit then edits; semantic is the separate all-source correctness metric. Token parity is diagnostic. Task-specific native rows remain NATIVE_DIAGNOSTIC, not T0 anchors.

2. What protection does S1 retain against W0, and at what cost?

- U (evaluation), S1−S0 semantic: NA; paired edits=0.
- U (evaluation), S1−S0 v4_primary: NA; paired edits=0.
- U (evaluation), S1−S0 base_correct_damage: NA; paired edits=0.
- U (evaluation), S1−BE semantic: NA; paired edits=0.
- U (evaluation), S1−BE v4_primary: NA; paired edits=0.
- U (evaluation), S1−BE base_correct_damage: NA; paired edits=0.
- cross_family_confirmation (evaluation), S1−S0 semantic: NA; paired edits=0.
- cross_family_confirmation (evaluation), S1−S0 v4_primary: NA; paired edits=0.
- cross_family_confirmation (evaluation), S1−S0 base_correct_damage: NA; paired edits=0.
- cross_family_confirmation (evaluation), S1−BE semantic: NA; paired edits=0.
- cross_family_confirmation (evaluation), S1−BE v4_primary: NA; paired edits=0.
- cross_family_confirmation (evaluation), S1−BE base_correct_damage: NA; paired edits=0.
- source_style_confirmation (evaluation), S1−S0 semantic: NA; paired edits=0.
- source_style_confirmation (evaluation), S1−S0 v4_primary: NA; paired edits=0.
- source_style_confirmation (evaluation), S1−S0 base_correct_damage: NA; paired edits=0.
- source_style_confirmation (evaluation), S1−BE semantic: NA; paired edits=0.
- source_style_confirmation (evaluation), S1−BE v4_primary: NA; paired edits=0.
- source_style_confirmation (evaluation), S1−BE base_correct_damage: NA; paired edits=0.
- T1G (formal_development), S1−S0 semantic: 0.0; paired edits=6.
- T1G (formal_development), S1−S0 v4_primary: 0.0; paired edits=6.
- T1G (formal_development), S1−S0 base_correct_damage: NA; paired edits=0.
- T1G (formal_development), S1−BE semantic: 0.0; paired edits=6.
- T1G (formal_development), S1−BE v4_primary: 0.0; paired edits=6.
- T1G (formal_development), S1−BE base_correct_damage: NA; paired edits=0.
- T1L (formal_development), S1−S0 semantic: 0.0; paired edits=2.
- T1L (formal_development), S1−S0 v4_primary: 0.0; paired edits=2.
- T1L (formal_development), S1−S0 base_correct_damage: 0.0; paired edits=2.
- T1L (formal_development), S1−BE semantic: 0.0; paired edits=2.
- T1L (formal_development), S1−BE v4_primary: 0.0; paired edits=2.
- T1L (formal_development), S1−BE base_correct_damage: 0.0; paired edits=2.
- T2G (formal_development), S1−S0 semantic: -0.5277777777777778; paired edits=6.
- T2G (formal_development), S1−S0 v4_primary: -0.5277777777777778; paired edits=6.
- T2G (formal_development), S1−S0 base_correct_damage: NA; paired edits=0.
- T2G (formal_development), S1−BE semantic: -0.5694444444444444; paired edits=6.
- T2G (formal_development), S1−BE v4_primary: -0.5694444444444444; paired edits=6.
- T2G (formal_development), S1−BE base_correct_damage: NA; paired edits=0.
- T2L (formal_development), S1−S0 semantic: 0.0; paired edits=1.
- T2L (formal_development), S1−S0 v4_primary: 0.0; paired edits=1.
- T2L (formal_development), S1−S0 base_correct_damage: 0.0; paired edits=1.
- T2L (formal_development), S1−BE semantic: 0.5; paired edits=1.
- T2L (formal_development), S1−BE v4_primary: 0.5; paired edits=1.
- T2L (formal_development), S1−BE base_correct_damage: -0.5; paired edits=1.
- NATIVE_DIAGNOSTIC (native), S1−S0 semantic: 0.0; paired edits=1.
- NATIVE_DIAGNOSTIC (native), S1−S0 v4_primary: NA; paired edits=0.
- NATIVE_DIAGNOSTIC (native), S1−S0 base_correct_damage: NA; paired edits=0.
- NATIVE_DIAGNOSTIC (native), S1−BE semantic: 0.0; paired edits=1.
- NATIVE_DIAGNOSTIC (native), S1−BE v4_primary: NA; paired edits=0.
- NATIVE_DIAGNOSTIC (native), S1−BE base_correct_damage: NA; paired edits=0.
- T0 (native), S1−S0 semantic: 0.0; paired edits=6.
- T0 (native), S1−S0 v4_primary: 0.0; paired edits=6.
- T0 (native), S1−S0 base_correct_damage: NA; paired edits=0.
- T0 (native), S1−BE semantic: 0.0; paired edits=6.
- T0 (native), S1−BE v4_primary: 0.0; paired edits=6.
- T0 (native), S1−BE base_correct_damage: NA; paired edits=0.

Paired edit intervals are in SINGLE_PAIRED_EFFECTS.csv. They resample edits; shared source images remain correlated. Missing judged pairs are NA, not zero effects.

3. How does the native BalancEdit system compare in behavior and cost?

| Method | Prefix | Role / panel | Strict role | Inputs | V4 primary macro (eligible) | All-source macro | All-source micro | Damage macro |
|---|---:|---|---|---:|---:|---:|---:|---:|
| BE | 0 | evaluation / U | STRICT_BASE | 100 | NA (0) | 0.5800 | 0.5800 | 0.4200 |
| BE | 0 | evaluation / cross_family_confirmation | EDIT_TARGET | 40 | NA (0) | 1.0000 | 1.0000 | NA |
| BE | 0 | evaluation / source_style_confirmation | EDIT_TARGET | 40 | NA (0) | 1.0000 | 1.0000 | NA |
| BE | 0 | formal_development / T1G | EDIT_TARGET | 83 | 0.9545 (83) | 0.9545 | 0.9518 | NA |
| BE | 0 | formal_development / T1L | STRICT_BASE | 26 | 0.2857 (19) | 0.4143 | 0.3077 | 0.7143 |
| BE | 0 | formal_development / T2G | EDIT_TARGET | 72 | 0.8939 (72) | 0.8939 | 0.8750 | NA |
| BE | 0 | formal_development / T2L | STRICT_BASE | 39 | 0.0217 (39) | 0.0217 | 0.0256 | 0.9783 |
| BE | 0 | formal_development / T3G | UNKNOWN | 0 | NA (0) | NA | NA | NA |
| BE | 0 | formal_development / T3L | UNKNOWN | 0 | NA (0) | NA | NA | NA |
| BE | 0 | formal_development / T4G | UNKNOWN | 0 | NA (0) | NA | NA | NA |
| BE | 0 | formal_development / T4L | UNKNOWN | 0 | NA (0) | NA | NA | NA |
| BE | 0 | formal_development / T5 | UNKNOWN | 0 | NA (0) | NA | NA | NA |
| BE | 0 | native / NATIVE_DIAGNOSTIC | EDIT_TARGET | 23 | NA (0) | 1.0000 | 1.0000 | NA |
| BE | 0 | native / T0 | EDIT_TARGET | 22 | 0.9545 (22) | 0.9545 | 0.9545 | NA |

Only ROUTED versus ROUTED and separately FORCED_ON versus FORCED_ON are compared. METHOD_COSTS.csv reports available training, teacher, generation, loading, routing, memory and storage measurements; missing measurements are NA. Layers, capacity and supervision differ; no matched-capacity claim.

4. Do benefits survive sixteen coexisting experts; where do failures come from?

| Method | Prefix | Role / panel | Strict role | Inputs | V4 primary macro (eligible) | All-source macro | All-source micro | Damage macro |
|---|---:|---|---|---:|---:|---:|---:|---:|
| S1 | 1 | evaluation / H | STRICT_BASE | 2 | NA (0) | 1.0000 | 1.0000 | 0.0000 |
| S1 | 1 | evaluation / U | STRICT_BASE | 12 | NA (0) | 0.5833 | 0.5833 | 0.0000 |
| S1 | 1 | evaluation / cross_family_confirmation | EDIT_TARGET | 2 | NA (0) | 1.0000 | 1.0000 | NA |
| S1 | 1 | evaluation / source_style_confirmation | EDIT_TARGET | 2 | NA (0) | 1.0000 | 1.0000 | NA |
| S1 | 1 | native / T0 | EDIT_TARGET | 1 | NA (0) | 1.0000 | 1.0000 | NA |
| S1 | 4 | evaluation / H | STRICT_BASE | 6 | NA (0) | 0.3750 | 0.5000 | 0.5000 |
| S1 | 4 | evaluation / U | STRICT_BASE | 50 | NA (0) | 0.4776 | 0.4800 | 0.3175 |
| S1 | 4 | evaluation / cross_family_confirmation | EDIT_TARGET | 8 | NA (0) | 1.0000 | 1.0000 | NA |
| S1 | 4 | evaluation / source_style_confirmation | EDIT_TARGET | 8 | NA (0) | 1.0000 | 1.0000 | NA |
| S1 | 4 | native / T0 | EDIT_TARGET | 4 | NA (0) | 1.0000 | 1.0000 | NA |
| S1 | 8 | evaluation / H | STRICT_BASE | 13 | NA (0) | 0.4375 | 0.5385 | 0.3750 |
| S1 | 8 | evaluation / U | STRICT_BASE | 99 | NA (0) | 0.5649 | 0.5657 | 0.1737 |
| S1 | 8 | evaluation / cross_family_confirmation | EDIT_TARGET | 16 | NA (0) | 1.0000 | 1.0000 | NA |
| S1 | 8 | evaluation / source_style_confirmation | EDIT_TARGET | 16 | NA (0) | 1.0000 | 1.0000 | NA |
| S1 | 8 | native / T0 | EDIT_TARGET | 8 | NA (0) | 1.0000 | 1.0000 | NA |
| S1 | 16 | evaluation / H | STRICT_BASE | 20 | NA (0) | 0.5000 | 0.5500 | 0.3077 |
| S1 | 16 | evaluation / U | STRICT_BASE | 202 | NA (0) | 0.5053 | 0.5050 | 0.2399 |
| S1 | 16 | evaluation / cross_family_confirmation | EDIT_TARGET | 32 | NA (0) | 1.0000 | 1.0000 | NA |
| S1 | 16 | evaluation / source_style_confirmation | EDIT_TARGET | 32 | NA (0) | 1.0000 | 1.0000 | NA |
| S1 | 16 | native / T0 | EDIT_TARGET | 16 | NA (0) | 1.0000 | 1.0000 | NA |

EXPERT_BANK16_RESULTS.csv reports rejection with a correct own writer, wrong-writer failures, selected writer errors and equivalent-expert correct outputs, each with explicit denominators. EXPERT_BANK16_HISTORY.csv pairs each input's first evaluable prefix with prefix16; absent pairs are not counted. Conflicting/unknown roles retain original-reference descriptive scores but have no strict semantic/locality score. NOW_EDITED_CONTEXT is scored against its frozen effective target and excluded from strict Base damage. This is viewed Stage2 insertion replay, not causal online training or M3Bench 200-edit sequential.

5. What is covered, what is missing, and should the route be expanded?

Planned method endpoints=2220; executable=109; unsupported=2111; complete=109; pending=0. Current Judge tuples=1492; missing=0. EXECUTION_STATUS.json retains every queued method and missing prefix. Publication is unverified.
No new experiment or performance permission gate is introduced. Finish the existing pending computation/Judge/publication only; no scale-up conclusion is supported by this partial snapshot.

Historical continuity: Stage2 GPT_PRO_REVIEW.md records a real original-T2G loss for P4-W1 and sparse old H support; a perfect constructed-text panel does not erase that finding. Historical scores are not relabeled as current execution. See the Stage2 publication at 74d2a337f7d2b830d58819f76c87058cef0c5f3b.
Micro averages count observed inputs; macro averages aggregate equivalent inputs within source groups, then edits. Image/group supports are not patient counts (patient identity UNKNOWN). Empty denominators are NA. Judge uses full answers; EOS/cap-hit flags are reported only when supplied by the generator.
