# LoRA-Perf-v1 status

Status: `ENGINEERING_READY__CALIBRATION_NOT_STARTED`

- Frozen source baseline: `7f827a7f025d18a15da2602a9a0dd49eef3f153e`
- Selection: frozen `LORA_DEV16_V2`; validation: frozen `LORA_QUAL16_V2`, exactly once after one DEV configuration is selected.
- Initial bounded profiles: rank 16, alpha 16, all LM MLP gate/up/down projections, learning rates 1e-4, 2e-4, and 5e-4, continuous checkpoints 5/10/20/40/80.
- The 2026-09-02 LoRA-strong traces are retained as historical evidence but are not reused as V4 validation: they predate the current V4 frozen cohorts and did not measure the required T1L/T1G/T2G panel.
- The old paper-spec semantic qualification result is not a calibration prerequisite.

No QUAL16 method output has been inspected or generated in this stage.
