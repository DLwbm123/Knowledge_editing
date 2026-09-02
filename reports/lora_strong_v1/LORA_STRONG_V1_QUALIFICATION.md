# LoRA-Strong-v1 qualification

Status: `LORA_STRONG_HELDOUT_HARD_STOP__LOCALITY_COLLAPSE__PROTOCOL_AMENDMENT_REQUIRED`

`LoRA-paper-spec-5` remains unchanged. The strong candidate passed the 4-edit overfit gate and learned all 16 held-out targets, but failed the frozen locality threshold. It is therefore not qualified for sequential smoke, amended-189, or use as the new method's LoRA component.

| Stage | Result |
|---|---:|
| Overfit semantic T0 | 4/4 |
| Overfit NLL < 0.5 | 4/4 |
| Overfit first-token rank 1 | 4/4 |
| Overfit empty | 0 |
| Overfit base unchanged | 4/4 |
| Overfit save/reload parity | 4/4 |
| DEV8 bounded grid | 16 configurations, 128 runs |
| DEV16 selected fixed config | lr 5e-4, 50 steps; semantic 16/16 |
| Held-out adaptive semantic T0 | 16/16 |
| Held-out NLL decrease | 16/16 |
| Held-out empty | 0 |
| Held-out base unchanged | 16/16 |
| Held-out save/reload parity | 16/16 |
| Held-out locality retention | 2/16 = 0.125 (required >= 0.70) |

All held-out edits stopped at step 15. A read-only aggregate diagnostic found that all 16 locality generations matched the current edit target, confirming global answer collapse rather than a Judge transport failure.

No rescue was started because the frozen rescue trigger was semantic T0 below 12/16, while observed semantic T0 was 16/16. Reusing this held-out set to choose a gentler configuration would invalidate it as held-out evidence, so a protocol amendment and fresh locality-qualified holdout are required before further tuning.

Sequential smoke and formal-189 were not started. Private records, questions, references, images, model answers, mappings, and adapter states are excluded from this report.
