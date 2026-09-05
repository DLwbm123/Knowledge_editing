# LoRA-Perf-v1 status

Status: `LORA_PERF_DEV_SELECTED__QUAL_JUDGE_PENDING`

- Frozen source baseline: `7f827a7f025d18a15da2602a9a0dd49eef3f153e`
- Selection: frozen `LORA_DEV16_V2`; validation: frozen `LORA_QUAL16_V2`, exactly once after one DEV configuration is selected.
- Initial bounded profiles: rank 16, alpha 16, all LM MLP gate/up/down projections, learning rates 1e-4, 2e-4, and 5e-4, continuous checkpoints 5/10/20/40/80.
- The 2026-09-02 LoRA-strong traces are retained as historical evidence but are not reused as V4 validation: they predate the current V4 frozen cohorts and did not measure the required T1L/T1G/T2G panel.
- The old paper-spec semantic qualification result is not a calibration prerequisite.

DEV Judge coverage and strict schema passed at 3,765/3,765. The unique selected configuration is rank 16, alpha 16, all LM MLP gate/up/down projections, learning rate 5e-4, checkpoint 80:

- T0 15/16; T1L macro 0.4143; T1G 0.8906; T2G 0.8906.
- Target NLL decreased 16/16; empty/error 0; base unchanged 16/16.
- Same-question, other-image edit-target copy rate 0.9205; this is reported as a limitation, not counted as visual-edit success.

The single allowed QUAL16 generation has completed 16/16 mechanical checks on authorized GPU2. Its fixed semantic Judge is pending; no formal LoRA or MedTRACE GPU claim is made yet.
