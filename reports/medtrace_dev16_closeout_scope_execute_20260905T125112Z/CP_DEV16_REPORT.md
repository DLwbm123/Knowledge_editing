# CP-DEV16 evaluation

Status: `CP_DEV16_EVALUATION_COMPLETE`

This is a 16-edit development run of `TIME_INSPIRED_CP_R4_DEV16_FORCED_ON`, not full TIME, full MedTRACE, or an independent benchmark. T0 was used for fitting and early stopping.

| Task | Valid edits | n | Exact micro | Exact macro | Semantic micro | Semantic macro | Truncated without EOS |
|---|---:|---:|---:|---:|---:|---:|---:|
| T0 | 16 | 16 | 16/16 (100.0%) | 100.0% | 15/16 (93.8%) | 93.8% | 0 |
| T1L | 7 | 26 | 8/26 (30.8%) | 41.4% | 8/26 (30.8%) | 41.4% | 0 |
| T1G | 16 | 62 | 62/62 (100.0%) | 100.0% | 57/62 (91.9%) | 92.2% | 0 |
| T2G | 16 | 59 | 22/59 (37.3%) | 38.0% | 31/59 (52.5%) | 52.1% | 3 |

All 16 experts reached the native training stop within 200 steps; mean selected step was 148.75. Mean target NLL changed from 4.317495 to 0.168988; all 16 decreased. Event-time sum was 783.596 seconds and peak allocated VRAM was 15.428 GiB.

The legacy `native_target_copy` field was a normalized substring test. It is retained only as `legacy_target_substring_present`; exact-reference, exact-edit-target, whole-word and semantic verdicts are separate. Length-limit classification is derived from saved token IDs because the original run did not save a stop reason.

T1L source/Judge diagnostic: {'source_and_judge_supported_conflict': 18, 'compatible': 8}. `unknown` is neither removed from the ordinary locality denominator nor called a clinical conflict.
