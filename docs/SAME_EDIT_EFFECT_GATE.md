# SAME-Edit Effect Gate

## Purpose

Validate SAME-Edit as an independent medical MLLM editing method on bounded MedMKEB / LLaVA-Med records.

## Environment

- Output directory: `/remote-home/wangbomin/Knowledge_editing/outputs/same_edit_effect_gate/20260622_030421_one_edit`
- Device: `cuda`
- Model: `hparams default`
- Data: `datasets/MedMKEB/eval.json`
- Max edits: `5`
- One-edit steps: `50`
- Five-edit steps per edit: `50`
- LoRA rank / experts / top-k: `8` / `4` / `1`

## Code Changes

- Added `scripts/same_edit/run_same_edit_effect_gate.py`.
- Updated this document with the latest gate results.
- No ENGRAM / CURE / DSCA core logic was changed.

## Commands

- Full command is recorded in `/remote-home/wangbomin/Knowledge_editing/outputs/same_edit_effect_gate/20260622_030421_one_edit/run_commands.sh`.

## Results

| mode | method | mean_delta_new_nll | positive_new | ref_abs_delta_mean | locality_damage | max_retention_drop | rollback |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| one_edit | plain_moe_lora | -12.18831711821258 | 1 | 4.486883163452148 | 1 | None | True |
| one_edit | same_full | -12.188299711793661 | 1 | 4.486662864685059 | 1 | None | True |

## SAME Diagnostics

- Diagnostics JSONL: `/remote-home/wangbomin/Knowledge_editing/outputs/same_edit_effect_gate/20260622_030421_one_edit/same_diagnostics.jsonl`
- Rollback report: `/remote-home/wangbomin/Knowledge_editing/outputs/same_edit_effect_gate/20260622_030421_one_edit/rollback_report.json`

## Conclusion

- Verdict: `fail`
- Stop reason: `gate1_one_edit_failed: same_full one-edit reference delta exceeds threshold; same_full one-edit has locality damage`

This is a bounded engineering gate, not a full reproduction of the original CoIN 8-task MCIT setting.
