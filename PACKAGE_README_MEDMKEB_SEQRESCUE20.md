# MedMKEB Sequential Rescue 20 Package

This package captures the ENGRAM-projected tiny LoRA MedMKEB sequential rescue run for the fixed model-known 20-record subset.

## Main Entry Points

- Runner: `scripts/engram/run_medmkeb_sequential_rescue.py`
- Hparams: `hparams/ENGRAM/llava_med_5edit_cure_tiny_lora.yaml`
- Previous model-known report: `outputs/medmkeb_engram_projected_lora/modelknown_20/FINAL_MEDMKEB_MODELKNOWN_20_REPORT.md`
- Rescue report: `outputs/medmkeb_engram_projected_lora/sequential_rescue_20/FINAL_MEDMKEB_SEQUENTIAL_RESCUE_20_REPORT.md`

## Best Rescue Variant

- Config: `C_beta0.3_steps10_qkgate_ref0`
- Status: `basic_pass`
- Final mean new-answer NLL decrease: `0.302485`
- Final reference delta abs: `0.0387687`
- Positive new-answer edits: `20/20`
- Locality damage records: `6/20`
- Rollback: `pass`
- Record-id match rate: `1.0`
- NaN/Inf count: `0`

## Validation

- ENGRAM pytest suite: pass
- CURE smoke tests: pass
- Remote GPU run completed on the fixed 20 selected records.
- Generation diagnostics are included for 5 records, but NLL/logprob metrics are the primary evidence.
- No medical or clinical efficacy claim is made.

## Package Hygiene

Runtime projector banks, model weights, tensor files, caches, Python bytecode, `.DS_Store`, and AppleDouble `._*` files are excluded.
