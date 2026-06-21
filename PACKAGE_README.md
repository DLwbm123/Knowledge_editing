# Knowledge Editing ENGRAM/CURE Patch

This package contains the current ENGRAM research prototype patch for the medical multimodal knowledge editing workspace, including the CURE-MedEdit delta-space Crisp projection prototype and 5-edit validation artifacts.

Paper context:
- Kwon et al., 2026, "AI Engram: In Search of Memory Traces in Artificial Intelligence"

Prototype name:
- `Knowledge_editing_engram_patch`

Primary new-method entry points:
- `scripts/engram/run_medmkeb_modelknown_editing.py`
- `scripts/engram/run_localized_replacement_5edit.py`
- `scripts/engram/run_cure_mededit_5edit.py`
- `scripts/engram/run_cure_sequential_pareto.py`
- `hparams/ENGRAM/llava_med_5edit_localized_replacement_tiny_lora.yaml`
- `hparams/ENGRAM/llava_med_5edit_cure_tiny_lora.yaml`
- `easyeditor/models/engram/crisp_projection.py`
- `easyeditor/models/engram/crisp_kfac_collector.py`
- `easyeditor/models/engram/engram_hparams.py`

Method summary:
- Builds a 5-record synthetic replacement smoke set under `outputs/engram_localized_replacement_5edit/`.
- Keeps the existing direct ENGRAM erase result as the failure baseline.
- Extracts record-id-aware ENGRAM projector banks from the successful 8-module q/k/gate scope.
- Trains tiny low-rank replacement deltas and applies ENGRAM projection as `Delta_safe = Delta_candidate @ P`, implemented in low-rank form as `B @ (A @ P)`.
- Adds CURE-MedEdit as a delta-space MVP: `Delta_cure = Pi_crisp(Delta_candidate @ P_engram)`, where `Pi_crisp` is a CrispEdit-style K-FAC low-curvature projection.
- Compares:
  - A: no-edit baseline
  - B: direct ENGRAM erase baseline
  - C: unprojected tiny-LoRA replacement
  - D: ENGRAM-projected tiny-LoRA replacement
  - E: CURE dual-projected tiny-LoRA replacement

MedMKEB validation update from 2026-06-20:
- Primary method: `C_engram_projected_tiny_lora`.
- Main comparison: `A_no_edit`, `B_tiny_lora_replacement`, `C_engram_projected_tiny_lora`.
- Data: bounded public MedMKEB/VLKEB-style model-known 20 subset, record-id matched, no private or patient data.
- Main gate used NLL/logprob metrics with generation skipped; generation diagnostics were run only after the NLL gate.
- Tests passed on the remote validation environment:
  - `tests/test_cure_crisp_projection.py`
  - `tests/test_cure_kfac_collector_tiny_mllm.py`
  - `tests/test_engram_*.py`
- Record-id preflight passed:
  - selected records: `20`
  - unique records: `20`
  - image-resolved records: `20`
  - record-id match rate: `1.0`
  - positional matching used: `False`
- Non-sequential result:
  - C acceptance: `pass`
  - C mean new-answer NLL decrease: `0.7758035659790039`
  - C mean reference delta abs: `0.012247514724731446`
  - C positive new-answer edits: `20 / 20`
  - C locality damage edits: `0`
  - C rollback pass rate: `1.0`
  - C record-id match rate: `1.0`
  - C NaN/Inf count: `0`
  - C had lower reference/locality drift than B (`0.0122475` vs `0.0274149`) and no locality damage (`0` vs `2`).
- Sequential result:
  - status: `complete`
  - C acceptance: `partial`
  - C final mean new-answer NLL decrease: `1.6826518207788468`
  - C final reference delta abs: `0.2771370053291321`
  - C positive new-answer edits: `20 / 20`
  - C locality damage records: `17`
  - B final reference delta abs: `0.4803401231765747`
  - B locality damage records: `19`
  - C retained `0.9282298949458015` of B's final new-answer signal and reduced reference drift to `0.5769599330915264` of B.
  - Sequential acceptance is `partial` because the stricter diagnostic `C_reference_ratio_vs_B_at_most_0_50_if_possible` did not pass.
- Decision: C is strong enough on non-sequential MedMKEB model-known 20, but sequential scaling still needs analysis before expanding.

Validation summary from the 2026-06-18 remote smoke:
- Local syntax and ENGRAM tests passed: `35 passed, 3 warnings`.
- Remote syntax and ENGRAM tests passed with `/root/anaconda3/bin/python`: `35 passed, 7 warnings`.
- Full remote 5-edit localized replacement run completed with exit code `0`.
- 20-edit was not run.
- Generation was skipped in the validation run, so this package does not claim generation quality.

Validation summary from the 2026-06-19 CURE sequential Pareto run:
- Remote syntax and test gates passed:
  - `tests/test_cure_crisp_projection.py`: `7 passed`
  - `tests/test_cure_kfac_collector_tiny_mllm.py`: `3 passed`
  - `tests/test_engram_*.py`: `35 passed`
- Run completed with `--skip-generation`.
- 20-edit was not run.
- Direct ENGRAM erase was not rerun.
- The compact grid ran one C baseline plus eight prioritized CURE configs.

Key non-sequential 5-edit result:
- `D_engram_projected_tiny_lora_replacement`
- Mean new-answer NLL decrease: `2.0294344425201416`
- Mean reference delta: `0.0007637977600097656`
- Positive new-answer edits: `5 / 5`
- Locality damage edits: `0 / 5`
- Rollback pass rate: `1.0`
- Record-id match rate: `1.0`
- NaN/Inf count: `0`

Key sequential smoke result:
- Status: `complete`
- Beta: `0.5`
- Mean new-answer NLL decrease: `4.560218060016632`
- Mean reference delta: `0.0020395755767822266`
- Positive new-answer edits: `5 / 5`
- Locality damage edits: `0 / 5`
- Rollback pass rate: `1.0`
- Record-id match rate: `1.0`
- NaN/Inf count: `0`

Key CURE sequential Pareto result:
- Pareto-promising config: `E_beta0.5_gamma0.5_streaming`
- Method: `E_cure_dual_projected_tiny_lora`
- Beta: `0.5`
- Crisp energy threshold: `0.5`
- Crisp cache update policy: `streaming_average`
- Mean new-answer NLL decrease: `4.536747014522552`
- Mean reference delta abs: `0.002671670913696289`
- Previous-edit retention: `4.550813093781471`
- Positive new-answer edits: `5 / 5`
- Locality damage records: `0`
- Rollback pass: `true`
- Record-id match rate: `1.0`
- NaN/Inf count: `0`
- Relative to C baseline:
  - New-answer ratio: `0.985815307204495`
  - Reference ratio: `0.7479808295619902`
  - Retention ratio: `0.9857176436936593`

Important interpretation boundary:
- This is a replacement-new-answer prototype, not a demonstrated medical efficacy result.
- The old-answer erasure metric for the tiny-LoRA replacement variants is still negative on this smoke set, so do not describe this as a successful old-answer erase.
- The included records are synthetic engineering smoke fixtures, not private or patient data.
- Metrics are record-id-aware; do not use positional matching unless explicitly requested.
- Delta-space Crisp projection is not the original CrispEdit gradient-projected training loop.

Useful validation artifacts:
- `outputs/medmkeb_engram_projected_lora/modelknown_20/FINAL_MEDMKEB_MODELKNOWN_20_REPORT.md`
- `outputs/medmkeb_engram_projected_lora/modelknown_20/nonseq/nonseq_aggregates.csv`
- `outputs/medmkeb_engram_projected_lora/modelknown_20/sequential/sequential_summary.csv`
- `outputs/medmkeb_engram_projected_lora/modelknown_20/record_id_preflight.json`
- `outputs/medmkeb_engram_projected_lora/modelknown_20/projector_extraction_summary.json`
- `outputs/medmkeb_engram_projected_lora/generation_diagnostics/`
- `outputs/medmkeb_engram_projected_lora/test_logs/`
- `outputs/medmkeb_engram_projected_lora/PACKAGE_HYGIENE_REPORT.md`
- `outputs/engram_localized_replacement_5edit/FINAL_LOCALIZED_REPLACEMENT_5EDIT_REPORT.md`
- `outputs/engram_localized_replacement_5edit/DIRECT_ERASE_FAILURE_SUMMARY.md`
- `outputs/engram_localized_replacement_5edit/replacement_data_summary.json`
- `outputs/engram_localized_replacement_5edit/effective_replacement_config.json`
- `outputs/engram_localized_replacement_5edit/nonseq/nonseq_replacement_results.json`
- `outputs/engram_localized_replacement_5edit/nonseq/nonseq_replacement_aggregates.csv`
- `outputs/engram_localized_replacement_5edit/sequential/sequential_replacement_results.json`
- `outputs/engram_localized_replacement_5edit/projector_bank/index.json`
- `outputs/engram_localized_replacement_5edit/projector_bank/edits/*/metadata.json`
- `outputs/cure_mededit_5edit/nonseq_real/FINAL_CURE_MEDEDIT_5EDIT_REPORT.md`
- `outputs/cure_mededit_5edit/sequential_real/FINAL_CURE_SEQUENTIAL_5EDIT_REPORT.md`
- `outputs/cure_mededit_5edit/sequential_pareto/FINAL_CURE_SEQUENTIAL_PARETO_REPORT.md`
- `outputs/cure_mededit_5edit/sequential_pareto/pareto_summary.json`
- `outputs/cure_mededit_5edit/sequential_pareto/pareto_step_matrix.csv`
- `outputs/cure_mededit_5edit/sequential_pareto/projection_diagnostics.json`
- `outputs/cure_mededit_5edit/sequential_pareto/test_logs/`

Package exclusions:
- Model weights
- Downloaded datasets
- Hugging Face or CUDA caches
- Large projector tensor files such as `tensors.pt`
- External/private image data and AppleDouble files
- The small synthetic 5-edit smoke fixture images may be included when needed by the runner
- Private or patient data
