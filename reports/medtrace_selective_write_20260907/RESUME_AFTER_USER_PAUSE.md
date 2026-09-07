# User-requested pause and explicit GPU2/3 resume

Current status: resumed at the user's explicit request on GPU2/3, using implementation c4397a7. Both live workers loaded their saved step320 experts and logged ENDPOINT_ONLY_RESUME with zero added optimizer steps. Snapshot: 12 RAW_READY, 2 RUNNING, 56 PENDING; no Judge result yet. The paragraphs below retain the historical pause state, not the current device allocation.

The existing runner now has an endpoint-only branch, reusing its original endpoint/Base-parity/fixed-router checks. Seven CPU tests passed, followed by successful loading of both actual checkpoints. The original campaign config and start are unchanged; ACTIVE_RESUME.json separately accounts for active time before pause plus time since explicit resume. Original teacher/cache bindings remain tied to their scientific source, while RESUME_CODE_PROVENANCE.json records the newer execution wrapper. Unknown pre-pause final training duration is left null rather than fabricated; resumed task elapsed_seconds covers its current endpoint attempt only.

Fresh logs and process bookkeeping are under run-root `resume01/`; the former STOP marker is retained there as STOP_USER_PAUSE_ARCHIVED.json. Old coordinator/worker logs, results and checkpoints are not overwritten. GPU0/1 are excluded from the resumed device allowlist. This is one explicit resume, not a recurring restart or monitor. A further interruption requires inspecting state before another recovery; the one-shot resume refuses to overwrite an existing resume01.

## Historical pause record

Run `20260907_selective_write_v1_r01` is paused. Its coordinator and both GPU0/1 workers exited; no unrelated process was stopped. A STOP marker prevents continued task claims. No automatic restart is authorized.

Retained state: 12 RAW_READY results, 2 PAUSED_BY_USER tasks, 56 PENDING tasks. Both paused tasks (`SW_e02_P4_W0_TASK_ONLY`, `SW_e02_L16_W0_TASK_ONLY`) finished 320 optimizer steps and retain `attempt_chunk16/step0320.pt`, including optimizer/protection state. They were interrupted during endpoint generation, not training. Their incomplete in-memory endpoint outputs may need regeneration. Teacher caches, A2 references, source locks, prior checkpoints and completed results remain intact.

The private resume ledger is `/remote-home/wangbomin/medtrace_runs/20260907_selective_write_v1_r01/private/USER_PAUSE_STATE.json`. It records pause time, the preserved original start, checkpoint paths and sizes, and interrupted phases. Queue entries are explicitly PAUSED_BY_USER rather than stale RUNNING or scientific failures.

On a later explicit resume request: recheck GPU authorization/availability; preserve and reuse the 12 completed results; add/use endpoint-only continuation from the two step320 experts rather than blindly retraining them. The current worker does not yet implement endpoint-only resume. Account separately for paused wall time and use fresh coordinator log/attempt bookkeeping; do not overwrite the original start or run the old coordinator command unchanged. Remove the STOP marker only as part of that authorized resume. This note does not claim resume support has already been implemented or tested.

At shutdown GPU0 showed 1 MiB/0%; GPU1 still showed 15,363 MiB/100% from other work. The project's GPU allocations were released; GPU1 was not claimed globally idle. GitHub's earlier running snapshot is historical; this pause record is retained locally and on the execution host.
