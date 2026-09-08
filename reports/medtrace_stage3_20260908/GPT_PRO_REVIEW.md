# Stage3 startup and frozen coverage — results pending

## Later resource amendment: four GPUs

The user subsequently authorized GPU0 and GPU1 in addition to GPU2/3. Scheduling code `9ffae2c` supports this explicit Stage3-only amendment. The replacement coordinator adopts the original GPU2/3 workers without interrupting their training; two added resident workers consume the same mutually exclusive queue. The original manifest/config and experiment start clock are retained, with a separate resource amendment. No method, input, support or scientific boundary changes.

The 24-hour wall and 48 GPU-hour bounds remain unchanged. Accounting conservatively bounds the initial two-GPU interval, the expanded four-GPU generation interval and the later single-GPU Judge interval; generation stops by 40 GPU-hours to preserve closure capacity. Three focused scheduling/Stage3 checks passed. This amendment supersedes the initial two-worker ETA below; four-worker throughput and Judge duration remain provisional. Final results are still pending.

Observed 2026-09-08 08:37:19 UTC. Run: `medtrace_stage3_20260908_r01`. This report records an active experiment, not scientific results. GPU2/3 are running independently of the interactive session; no unrelated GPU processes were stopped. Visible process roles are neutral `main` / `run`.

The four bank prefixes 1/4/8/16 are RAW_READY (57/228/456/906 method-input records). Their measured task times were 9.44/37.50/89.90/112.85 seconds. The first three-method single-edit group is RAW_READY in 495.64 seconds. The queue snapshot contains five RAW_READY, two RUNNING, 42 PENDING and 691 UNSUPPORTED_TRAINING_INPUTS groups. RAW_READY means generation/replay completed, not Judge completion.

The first full single group took about 8.3 minutes. A simple two-worker extrapolation for the remaining 44 groups is about three hours of generation; this is provisional because most remaining groups have two methods and input lengths vary. Judge time is not yet measured. The hard bound remains 24 hours wall / 48 GPU-hours, with four hours reserved for closure after the generation window.

## Frozen methods and coverage

S1 is BE_ROUTE + P4-W1_KL_0.1; S0 uses the same route + P4-W0_TASK_ONLY; B is native BalancEdit V4 adaptation. Original formulas, runtime and evaluation rules remain locked. BalancEdit and CP differ in edited layers, capacity and supervision, so costs must accompany behavior comparisons. FORCED_ON is a paired diagnostic.

The authoritative catalog contains 179 T0 events and 1,108 total events / 2,496 probes. Across 736 planned training groups, executable support is BE 45, S0 45 and S1 seven; the common comparison subset was frozen at seven before outputs. BE/S0 cover 22 T0 anchors and 23 T2L groups, not 45 T0 events. No T3/T4 group has the required approved training support. Unsupported groups remain planned rather than being dropped from denominators. This augmented-supervision evaluation is not full V4 coverage or paper-exact reproduction.

Track B uses the existing 16 Stage2 experts without retraining. Prefix-local target-free routing executes the actually selected writer, with strict-role and conflict handling fixed before output inspection. These viewed insertion replays are not blind confirmation, causal online learning or sequential-179 training.

## Evidence and open questions

Twenty-five focused Stage3 checks passed in the existing environment. A read-only historical Judge binding check resolved 7,780 exact-bound tuples; tokenizer preflight also passed. The first real GPU tasks progressed without a recorded queue failure at this observation. Scientific comparisons still require Judge closure and the final tables.

All five requested decisions remain pending: original V4 generalization/locality; S1 protection gains and costs versus S0; behavior/cost versus BalancEdit; bank error attribution to selection, rejection or writer; and whether scale-up is justified. The Stage2 T2G decline remains a historical measured tradeoff, not a resolved defect. See the [Stage2 behavior audit](../medtrace_stage2_20260908/STAGE1_FULL_BEHAVIOR_ADDENDUM.md); no new Stage3 result is inferred from it.

## Provenance and delivery boundary

Research branch `medtrace-stage3-20260908`: frozen runner/source `22c397a`, scoring `751608a`. The manifest source lock predates the scoring commit; this does not indicate retraining with a different method. Prior Stage2 public anchor: `74d2a337f7d2b830d58819f76c87058cef0c5f3b`.

This release includes source, tests, sanitized manifest and timestamped startup status only. Final result CSVs are not yet available. Images, QA, raw outputs, token IDs, teacher distributions, checkpoints and private Judge maps remain private. The detached coordinator will run generation, Judge and aggregation within its bound. Per the current long-experiment instructions, no separate polling/publication waiter is launched; final GitHub delivery will be verified on the next requested check after completion.
