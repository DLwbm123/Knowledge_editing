# MedTRACE bounded route-representation campaign protocol

Status: `FROZEN_BEFORE_GPU_SCORING`

This run is `MEDTRACE_ROUTE_REPRESENTATION_DEVELOPMENT_CAMPAIGN_V1`. It starts from public commit `34dc083706b690f1a1f5ae19132571dc02be78d4`, executes code commit `7d4fc0558e0ea1a02bd4a3e17a5201eed47806c7`, and is not full TIME, V0.2 qualification, clinical validation, or a blind benchmark. The historical LoRA result remains `QUAL_VALIDATION_FAIL`.

## Track A

All 16 DEV edits are paired at seed bases `20260906`, `20260907`, and `20260908`. Each edit/seed trains a fresh rank-4 layer-21 A0 with the frozen native stop rule, then copies A0 into equal-budget A1 native/native and A2 native/rotating-paraphrase continuations of 80 steps and 160 micro-forwards. Evaluation is the already viewed DEV panel and cannot select a checkpoint.

## Track B

The authorized source contract yielded 7 `HARD_EVALUABLE`, 5 `BROAD_ONLY`, and 4 `FIT_ONLY_INSUFFICIENT` edits. The primary edit preserves its frozen 4/4/5 hard groups. Other SLAKE edits use stable source-group selection; VQA-RAD remains audit-only. Eight additional positive prompts per non-primary edit are source-question-only wrappers that preserve every word of the source question; they were produced without targets, model outputs, probe scores, or routing scores and are not independent clinical review.

For each executable edit/seed, R0 magnitude, R1 signed CP response, and R2 visual-conditioned CP response train the same Q for 800 steps and save steps 200/800. R2's image-token span is derived from the realized LLaVA multimodal expansion, never from text-token counts. SAFETY_FIRST and COVERAGE_CONSTRAINED thresholds use calibration only. One R0 and one R1/R2 profile are frozen before evaluation scores are opened.

Selected profiles receive the same 80-step A2 output-only fit. Q, prototypes, and threshold remain frozen. Base, forced-on, and actually gated paths are compared; at least one OFF input per profile executes a real base replay before bulk base reuse is allowed.

## Budget and isolation

Only physical GPU2 and GPU3 with the registered UUIDs may run. The campaign stops adding GPU work at 20 wall-clock hours or 48 accumulated GPU-hours and reserves the remaining wall-clock budget for Judge and reporting. A STOP file or SIGTERM ends work at a stage boundary. Old checkpoints, raw outputs, verdicts, and manifests remain read-only; private QA, images, tokens, activations, checkpoints, weights, and Judge mappings remain server-only.

Low scores, all-OFF routing, incomplete source support, and failed tasks are retained. Job completion is not evidence that a mechanism works.
