# Route B Post-Raw Gate Report

## Decision

`M3BENCH_POSTRAW_BLOCKED__INVALID_EDIT_TARGET_CONTAMINATION`

Route B stopped before scoring. The reported dataset issues occur in formal edit targets, so the frozen raw evidence is not eligible for Judge or evaluator execution under the current protocol.

## Verified gates

- Parent evidence: 7,758 unique files re-read; 213,008,308,552 bytes checked; zero missing, size mismatch, hash mismatch, or manifest conflict.
- Parent raw closure: PASS; 6,523 files; 212,979,427,180 bytes; aggregate SHA-256 `7d555ad3e9340797fa801e8c999c08151efdb03a15d875c113562b8fe8de8e64`.
- Audit return: 200/200 records validated; 64 true and 136 false; confidence census 193 high, 4 medium, 3 low.
- Private mapping: 200/200 records joined; 192 distinct underlying events and 8 repeated occurrences.
- Generated evidence checksums: 10/10 PASS.
- Pipeline tests: 5/5 PASS.

## Contamination finding

- Classification: `B_FORMAL_EDIT_TARGET`.
- Invalid formal positions: 19, 57, and 67.
- The affected question/reference pair occurs in 688 scoring events.
- Direct single-event impact: 240 events.
- Sequential impact from the earliest affected state: 24,752 events.

## Actions intentionally not performed

- Full Judge: not started.
- Evaluator: not started.
- Semantic metrics: not computed.
- Method identities: not unblinded.
- Automatic raw rerun: not started.
- Protected data access: zero violations.
- Human signoff: not completed; the imported package remains an independent model audit.

## Required next decision

A separately approved amended-sequence rerun is required. If no validated state snapshot immediately before position 19 exists, each anonymous sequential trajectory must be rerun under the amended sequence. The frozen parent run remains unchanged.
