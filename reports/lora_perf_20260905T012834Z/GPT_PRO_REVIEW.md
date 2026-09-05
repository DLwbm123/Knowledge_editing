# GPT Pro review packet: LoRA-Perf-v1 to MedTRACE

## Review decision requested

Audit whether the bounded LoRA calibration, fixed semantic evaluation, DEV-only selection, one-shot QUAL validation, and resulting stop decision faithfully implement `CODEX_PROMPT_LORA_PERFORMANCE_FIRST_TO_MEDTRACE.md`. Please distinguish implementation correctness from method performance. Do not reinterpret the failed QUAL result as a passed baseline.

## Frozen scope and provenance

- Repository: `https://github.com/DLwbm123/Knowledge_editing` (public).
- LoRA branch: `m3bench-lora-perf-20260905T012834Z`.
- MedTRACE branch: `medtrace-core-20260905T012834Z`.
- Frozen V4 source baseline: `7f827a7f025d18a15da2602a9a0dd49eef3f153e`.
- LoRA runtime code used for DEV: `e9ce6d690201f4d8e6e6e42cb865353c61081473`; later evaluation/QUAL support is on the same public branch.
- MedTRACE CPU-core commit: `2d6e5d5ebc9a9439ad80bde449dce9746c6b586d`.
- Method specification SHA256: `f6c07541345c7033d68c20f4dcd4ab21f928c08ea09c9cdde815986ae3bfecbd`.
- MedTRACE V0.2 revision SHA256: `0ba02155eb658201e9c0d140c6bcd811587b3a44af4923e2b72eebf3583d179c`.
- Fixed Judge: Qwen/Qwen3-32B-AWQ snapshot `0499c3ac83fdef8810b907a23894ba91e95eddd8`, deterministic constrained boolean output.
- Private questions, images, raw answers, model weights, optimizer states, and Judge mappings are excluded from the public branch.

## Implemented LoRA path

- Explicit rank, alpha, layer scope, target-module, learning-rate, and step configuration is enforced at injection time.
- Target answer-token NLL is recorded separately from EOS/template-tail NLL.
- Each `(edit, profile)` is one continuous AdamW trajectory with checkpoints at steps 5/10/20/40/80; no segmented-resume substitution was used.
- Each event records finite gradients, actual trainable parameters and parameter deltas, base-integrity state, reset state, optimizer/RNG state, target generation, and all frozen probes.
- Candidate selection is readiness first, then T0, T1L, T1G/T2G, then lower cost; no paper-distance objective is used.

## DEV16 result and selection

Three initial profiles used rank 16, alpha 16, all language-model MLP `gate_proj/up_proj/down_proj`, with learning rates 1e-4, 2e-4, and 5e-4. All three completed 16/16 events with finite gradients and unchanged base parameters. Fixed Judge coverage was 3,765/3,765 with 3,765 schema-valid verdicts.

The unique selection was learning rate 5e-4 at step 80:

| Metric | DEV result | Readiness minimum |
|---|---:|---:|
| T0 | 15/16 | 14/16 |
| T1L macro | 0.4143 | 0.25 |
| T1G | 0.8906 | 0.85 |
| T2G | 0.8906 | 0.60 |
| Target NLL decreased | 16/16 | 14/16 |
| Empty/error | 0 | 0 |
| Same-question other-image target-copy rate | 0.9205 | reported, not a pass gate |

The high copy rate was retained as a locality warning. It was not relabeled as visual editing success.

## One-shot QUAL16 result

The DEV-selected configuration was run exactly once on frozen QUAL16. Configuration and selection-lock equality was checked before execution. All 16 events passed the mechanical gate: finite gradients, unchanged base parameters, and no empty/error outputs. Total event runtime was 511.2 seconds; peak allocated/reserved GPU memory was 19.57/20.18 GiB. Fixed Judge coverage and schema validity were both 255/255.

| Metric | QUAL result | Readiness minimum | Outcome |
|---|---:|---:|---|
| T0 | 16/16 | 14/16 | pass |
| T1L macro | 0.2000 | 0.25 | **fail** |
| T1G | 1.0000 | 0.85 | pass |
| T2G | 1.0000 | 0.60 | pass |
| Target NLL decreased | 16/16 | 14/16 | pass |
| Empty/error | 0 | 0 | pass |
| Same-question other-image target-copy rate | 1.0000 | reported, not a pass gate | locality warning |

T1L uses 27 probes from 7 eligible edits; T1G has 62 probes, T2G has 61 probes, and the copy diagnostic has 89 checks. The primary aggregation is macro per edit where defined.

**Frozen conclusion:** `QUAL_VALIDATION_FAIL`. The 0.25 T1L threshold was not lowered, and QUAL was not recycled into DEV tuning. Formal LoRA evaluation was therefore not launched. `LORA_DEVELOPMENT_READY__START_MEDTRACE` was not set.

## MedTRACE implementation status

Status is `MEDTRACE_SPEC_AND_CPU_CORE_READY`, not full MedTRACE completion. The public branch contains:

- asymmetric CP residual experts;
- the same token-wise RMS-normalized down-projection input for route and execution;
- assistant-only injection and exact disabled/zero-residual paths;
- factor normalization and factorized-versus-dense parity;
- FPR-constrained threshold calibration stored as metadata, not an optimizer parameter;
- the prescribed capacity ladder metadata: one layer at R=4, R=8, R=16, then two layers at R=8 only if needed.

Three focused CPU tests pass. Real LLaVA-Med zero-effect parity, forced-on CP expressivity, intrinsic routing/scope, sequential editing, and full V0.2 are not claimed. Because LoRA QUAL failed, the prompt's GPU transition gate was not satisfied.

## Requested audit points

1. Verify that the one-shot QUAL failure is correctly enforced and that no hidden retuning path reads QUAL outputs.
2. Review whether macro-per-edit T1L aggregation and the same-question other-image copy diagnostic are implemented as intended.
3. Review the LoRA injection scope, trainable-parameter isolation, target-token NLL masking, continuous checkpoint trajectory, reset, and save/reload contracts.
4. Assess whether the DEV-to-QUAL drop in T1L (0.4143 to 0.2000) plus copy-rate increase (0.9205 to 1.0000) supports the stated locality-failure interpretation.
5. Review the MedTRACE V0.1/V0.2 binding and CPU core without treating synthetic tests as real-model evidence.

## Public evidence map

- `reports/lora_perf_20260905T012834Z/PERF_PROTOCOL.json`
- `reports/lora_perf_20260905T012834Z/DEV_PARETO.csv`
- `reports/lora_perf_20260905T012834Z/QUAL_METRICS.csv`
- `reports/lora_perf_20260905T012834Z/STATUS.md`
- `reports/lora_perf_20260905T012834Z/VALIDATION_REPORT.md`
- `scripts/m3bench_lora_perf.py`
- `scripts/m3bench_lora_perf_finalize.py`
- `m3bench_repro/editors/methods.py`
- `m3bench_repro/editors/llava_runtime.py`
- MedTRACE branch: `docs/medtrace/MEDTRACE_SPEC_BINDING.md`, `docs/medtrace/CORE_IMPLEMENTATION_STATUS.md`, `methods/medtrace/core.py`, and `tests/medtrace/test_core.py`.

This packet reports verified artifacts and explicit limitations. It does not claim paper reproduction, formal 179-event performance, or a completed MedTRACE method.
