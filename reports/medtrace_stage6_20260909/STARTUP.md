# Stage6 startup, not final results

Run ID: `medtrace_stage6_20260909_r01`. Detached coordinator 1104561; W0 worker 1104583 on GPU0 and BE worker 1104586 on GPU1. GPU2/3 are not used by this run. Long-process names and argv use neutral main/run aliases. No experiment training is authorized or implemented.

Local source branch `medtrace-stage6-20260909`; implementation through `6467aec`. Actual remote execution source is `55cbfb80c8bd53769202291bd876f89304677fc9` (separate remote commit history). Stage5 outputs/checkpoints/source packets are read-only. The one old16 threshold is 0.7696741135364367, with the exact source file hash retained in RUN_AND_TRANSFER_LOCK.json. No new calibration is called.

Existing transfer: 3064 R0/fixed-route entries derived; 467 needed full Judge tuples available without new Judge calls. These are preliminary until at most eight natural single/bank ON/OFF replays close. Missing natural branches are documented, not manufactured.

Evaluation sidecar frozen before new model outputs: 127 observations, comprising 116 U, 3 H and 8 existing-source alternative positive questions. Negative source-image support is 23; H spans 3 edits and 2 images, not the suggested broad hard-negative target. Prior fit-image exposure is disclosed; unchanged old inputs are excluded from the new sidecar. New source alternatives are not official M3Bench metrics or independent clinical reviews. Unknown/contradictory pairs are kept privately for review.

One local fixed-transfer/source-lock test passed in the existing remote environment. Compilation passed. Process startup, logs, GPU UUID assignment and live PIDs checked. This startup is not completed inference or a method-performance claim.

The pipeline is bounded at six wall hours from the original CAMPAIGN_START, at most twelve GPU-hours on the two authorized GPUs, with generation stopped by hour four to reserve Judge/report time. Source review is finished; no more candidate searching. The coordinator launches only this inference, unchanged Judge and aggregation chain. No recurring monitor was created or resumed. When the user next requests progress/completion, inspect this exact run and complete public delivery; do not rerun completed generation or Judge.

Server run root and private logs are retained in the task context, not embedded in this public-safe report. Public delivery remains pending while computation is running.
