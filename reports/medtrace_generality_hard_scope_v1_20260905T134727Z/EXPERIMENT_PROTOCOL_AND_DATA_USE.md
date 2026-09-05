# MedTRACE generality and hard-scope ablation protocol

Status: `FROZEN_BEFORE_NEW_MODEL_SCORING`

This development ablation starts from public commit `86c2036d4537793d803dfe982f6485cdbab9ce2c` on branch `medtrace-generality-hard-scope-v1-20260905T134727Z`. It is not full TIME, full MedTRACE, a blind benchmark, or V0.2 qualification. The historical LoRA result remains `QUAL_VALIDATION_FAIL`.

## Track A: equal-budget behavioral generality

All 16 DEV edits are retained. Four source-question-only paraphrases per edit were drafted without target answers, model outputs, probe scores or routing scores. The same Codex assistant performed linguistic equivalence self-review; this is not independent clinical review. Exact collisions with native questions, official rephrases, frozen DEV probes, LoRA QUAL questions/probes and prior scope calibration/evaluation positives are rejected before training.

- A0 `CP_NATIVE_ORIGINAL`: historical checkpoint and saved output only.
- A1 `CP_NATIVE_CONTINUE_80`: each A0 expert receives 80 fresh optimizer steps; every step accumulates two 0.5-weighted native micro-forwards.
- A2 `CP_NATIVE_PLUS_PARAPHRASE_80`: the same start, optimizer and 80-step budget; every step accumulates one 0.5 native and one 0.5 rotating-paraphrase micro-forward.

All CP factors update; the backbone remains frozen. Step 80 is fixed and cannot early-stop. Original native and all associated T1L/T1G/T2G probes are generated once with the frozen 1,024-token greedy contract. Probe results are already viewed and are used only as a paired development panel.

## Track B: hard-aware routing with matched output fitting

The primary and historical rank-4 step-140 checkpoint remain fixed. The previous four-positive fit/calibration/evaluation split is reused. Source review found 13 valid same-question/different-image hard-negative groups, assigned 4/4/5 to fit/calibration/evaluation. Twelve same-image/other-fact rows form a separate challenge; one same-image cross-language row was excluded because it expresses the same primary fact and is not a negative.

- B0 `ORIGINAL_Q_MATCHED_OUTPUT_FIT`: original Q, no input training.
- B1 `BROAD_Q_MATCHED_OUTPUT_FIT`: Q trains on four fit positives and 20 broad fit negatives.
- B2 `HARD_MIXED_Q_MATCHED_OUTPUT_FIT`: Q trains on the same positives, four hard negatives and the same 16-row broad subset.

B1/B2 use the previously frozen 200-step routing objective. Each condition calibrates its threshold on the identical four-positive/20-negative calibration panel at empirical FPR=0. All three then receive the same fixed 80-step native-plus-paraphrase output-only fit; Q and threshold must remain unchanged. Base, forced-on and actually gated outputs are evaluated on the same four-positive/20-negative main panel plus the 12-row same-image challenge.

## Isolation and reporting

GPU2 is reserved for Track A and GPU3 for Track B, subject to UUID and free-memory checks; GPU1 is forbidden. Old checkpoints, raw outputs, verdicts, manifests and reports are read-only. Raw QA, images, tokens, activations, weights and Judge mappings remain private. Public delivery contains aggregate counts, configurations, code, artifact hashes and final reports only.

Low scores, all-OFF routing, unchanged FPR or limited hard-negative counts remain reportable results. Engineering stops are limited to leakage, base mutation, non-finite optimization, state contamination, save/reload failure or GPU identity mismatch.
