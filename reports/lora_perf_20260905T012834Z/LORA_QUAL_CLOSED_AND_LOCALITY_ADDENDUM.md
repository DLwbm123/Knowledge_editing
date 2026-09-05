# LoRA QUAL closure and locality addendum

Status: `QUAL_VALIDATION_FAIL` (unchanged).

This CPU-only addendum recomputes descriptive statistics from the frozen QUAL mapping and verdict files. It does not rerun inference or Judge, change the cohort, alter the original readiness threshold, or reopen QUAL for tuning.

## T1L primary metric

The primary T1L metric is the unweighted mean of per-edit correctness rates over the seven eligible edits:

| Anonymous edit | Correct | Probes | Per-edit rate |
|---|---:|---:|---:|
| edit_90c90ed983 | 0 | 4 | 0.0000 |
| edit_f2a6178c52 | 2 | 5 | 0.4000 |
| edit_92ab05cffd | 3 | 3 | 1.0000 |
| edit_6a735ea692 | 0 | 5 | 0.0000 |
| edit_6be5f86b91 | 0 | 1 | 0.0000 |
| edit_1ae5b57d03 | 0 | 4 | 0.0000 |
| edit_b590d30278 | 0 | 5 | 0.0000 |

- Macro per edit: `(0 + 0.4 + 1 + 0 + 0 + 0 + 0) / 7 = 0.2000`.
- Pooled micro: `5 / 27 = 0.1852`.
- The reported 0.20 is not a claim that exactly 20% of the 27 probes were correct.

## Same-question, other-file diagnostic split

Normalization for literal equality is case-folding followed by comparison of alphanumeric token sequences.

| Diagnostic | Checks | Literal normalized equality with edit target | Semantic agreement with edit target |
|---|---:|---:|---:|
| `T1L_COPY_TO_EDIT_TARGET` | 27 | 27/27 | 27/27 |
| `T1G_TARGET_CONSISTENCY` | 62 | 62/62 | 62/62 |
| `ALL_SAME_QUESTION_OTHER_FILE_DIAGNOSTIC` | 89 | 89/89 | 89/89 |

The historical all-in-one field remains valid as a descriptive same-question/different-file diagnostic, but it combines two different interpretations. T1G is intended to preserve the edited fact under a semantically consistent image perturbation; its 62/62 target consistency is therefore not counted as harmful interference. T1L measures locality and remains the failed primary gate.

All 89 outputs are literal normalized matches to the edit target, so “semantic agreement” and “literal target equality” happen to have equal counts in this run. The terms are still kept separate because the Judge criterion is semantic, not lexical.

## T1L reference compatibility

- In 5/27 T1L checks, the probe reference and edit target are literal normalized matches; these are explicitly compatible.
- In the remaining 22/27, the strings differ. Existing artifacts do not directly adjudicate semantic compatibility between those two references, so their compatibility is `unknown`, not automatically “conflicting.”
- All 22 unknown-compatibility rows match the edit target and fail the ordinary probe-reference Judge. This is a `T1L_POTENTIAL_CONFLICT_PATTERN` count of 22, not a confirmed `T1L_CONFLICTING_REFERENCE_COPY` count.
- `T1L_CONFLICTING_REFERENCE_COPY`: `UNKNOWN_WITH_EXISTING_EVIDENCE`.

## Preserved conclusion

The selected LoRA configuration remains a high-reliability, low-locality research reference. QUAL T0/T1G/T2G do not override the frozen T1L minimum of 0.25. The result remains `QUAL_VALIDATION_FAIL`; no formal LoRA run is authorized by this addendum.
