# M3Bench current-stack V4: GPT Pro review brief

## Decision requested

Review whether the observed LoRA semantic-qualification failure is most likely
an implementation/training issue, a qualification-protocol issue, or an
expected consequence of the preserved paper-spec LoRA reference profile. Do
not infer formal benchmark performance from QUAL8.

## Frozen V4 baseline and runtime evidence

- Current canonical base prediction coverage: 11,088/11,088; missing,
  duplicate, empty, and error counts were all zero.
- Existing verdicts were reused only for 10,777 byte-exact raw matches; the
  remaining 311 changed raw records were re-evaluated by the fixed,
  method-blind semantic Judge.
- `BASE_VERDICTS_V4` is frozen with 4,460 correct and 6,628 incorrect
  predictions.
- The final amended T0 cohort has 179 ordered edits. Legacy selection artifacts
  are superseded; V2 G1R, QUAL8, DEV16, QUAL16, and SEQ16 manifests were
  frozen before method output.
- G1R passed for all 412 deduplicated no-edit queries. Frozen V4, a fresh
  official-native runtime, and the formal runtime matched exactly for prompt
  token IDs, raw generated token IDs, decoded text, normalized text, and image
  SHA; empty/error count was zero.

This eliminates current official/formal no-edit drift as the explanation for
the QUAL8 result, within the tested 412-query G1R canary.

## QUAL8 result

Every method completed 8/8 events with structural integration PASS, no empty
outputs, active NLL decreases, and changed target generations. A single fixed
method-blind semantic Judge then completed 32/32 opaque records with no parse
failures.

| Method | Integration | Effect active | Semantic-qualified | Semantic success |
| --- | --- | --- | --- | --- |
| LoRA paper-spec reference | PASS | PASS | FAIL | 3/8 |
| GRACE | PASS | PASS | FAIL | 2/8 |
| BalanceEdit | PASS | PASS | PASS | 8/8 |
| BELoRA | PASS | PASS | PASS | 7/8 |

The LoRA-first gate requires semantic qualification. Therefore LoRA
calibration, Sequential16, formal single, formal sequential, raw closure,
and final T0-T4 scoring were not started.

## What can and cannot be concluded

- The canonical current runtime and formal no-edit runtime agree on the G1R
  contract.
- The four implementations can execute the QUAL8 edit path and produce active
  training-side effects.
- The preserved LoRA paper-spec reference did not demonstrate adequate
  image-aware semantic edit success in QUAL8.
- This is **not** a successful M3Bench reproduction, not a formal method
  comparison, and not evidence that any method matches the paper.
- BalanceEdit and BELoRA passed only the small QUAL8 gate; no full formal run
  was authorized or executed for them.

## Suggested code-review focus

1. Trace the LoRA target-label construction, optimizer/update scope, training
   steps, and decoding path to determine why NLL decreases and token changes
   do not translate into semantic target success.
2. Verify that QUAL8 semantic scoring is correctly method-blind and that the
   LoRA target-generation field is sourced from the post-edit model state.
3. Compare the LoRA paper-spec profile with the intended calibrated profile;
   do not relax the semantic gate or start formal runs solely to obtain a
   result table.

## Public artifacts

- `G1R_V2_REPORT.md` and `.json`: runtime parity aggregate.
- `QUAL8_ALL_METHODS_REPORT.md` and `.json`: qualification aggregate.

These files intentionally exclude raw answers, questions, targets, images,
opaque mappings, private paths, model weights, and private manifests.
