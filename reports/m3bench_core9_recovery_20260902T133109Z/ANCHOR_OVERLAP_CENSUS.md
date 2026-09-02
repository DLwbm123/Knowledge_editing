# Amended-189 anchor overlap census

The frozen amended sequence was verified as 189 dense positions with the approved content hash. The audit used only frozen edit metadata and the locked public source snapshot; it did not read edited outputs.

| Task | Retained edits with metadata anchor | Candidate edits | Candidate probe relations | Dominant drop reason |
|---|---:|---:|---:|---|
| T2L | 56 | 56 | 1,496 | 133 edits had no anchor; 1,240 duplicate relations were removed |
| T3L | 33 | 33 | 189 | 156 edits had no directional image-A anchor |
| T3G | 33 | 33 | 189 | 156 edits had no directional image-A anchor |
| T4L | 65 | 0 | 0 | 153 metadata rows failed exact edit-question to `question_a` role matching |
| T4G | 35 | 3 | 15 | 46 concept mismatches and 2 missing target-finding questions |

Family-level metadata anchors are nonzero: T2L 56, T3 33, T4 union 100. The stricter task relation audit nevertheless proves that T4L has no legal candidate relation for this amended sequence. Base inference can classify an existing relation, but cannot create a missing edit-question/`question_a` relation.

Decision: `M3BENCH_AMENDED189_CORE9_BLOCKED__T4L__NO_LEGAL_QUESTION_ROLE_BINDING`.

