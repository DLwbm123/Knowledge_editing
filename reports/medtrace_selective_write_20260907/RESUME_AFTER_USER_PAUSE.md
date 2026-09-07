# User-requested pause and explicit GPU2/3 resume

Current status: **PAUSED_RESOURCE_BUSY; neutral-name relaunch NOT STARTED**. Following the second explicit stop for process renaming, all project processes are stopped. GPU2/3 acquired other allocations before the relaunch and the idle-only guard refused to start resume02. Current queue is 12 RAW_READY, 2 PAUSED_BY_USER, 56 PENDING. Eight CPU tests and the existing Judge-runtime alias probe pass; code and the neutral launcher are ready. The original STOP marker remains; no automatic restart is scheduled.

Historical first resume: GPU2/3 ran implementation c4397a7. Both workers loaded their saved step320 experts and logged ENDPOINT_ONLY_RESUME with zero added optimizer steps. Its snapshot was 12 RAW_READY, 2 RUNNING, 56 PENDING; no Judge result existed.

The existing runner now has an endpoint-only branch, reusing its original endpoint/Base-parity/fixed-router checks. Seven CPU tests passed, followed by successful loading of both actual checkpoints. The original campaign config and start are unchanged; ACTIVE_RESUME.json separately accounts for active time before pause plus time since explicit resume. Original teacher/cache bindings remain tied to their scientific source, while RESUME_CODE_PROVENANCE.json records the newer execution wrapper. Unknown pre-pause final training duration is left null rather than fabricated; resumed task elapsed_seconds covers its current endpoint attempt only.

The first resume logs and process bookkeeping are retained under run-root `resume01/`; the former STOP marker is retained there as STOP_USER_PAUSE_ARCHIVED.json. GPU0/1 are excluded from the resumed device allowlist. No recurring restart or monitor is installed; existing attempt directories are never overwritten.

The user then requested another stop/restart specifically to remove their username and method names from process listings. Coordinator 914110 and workers 914120/914122 were verified and stopped; no unrelated process was signalled. The 12 RAW_READY results and two step320 checkpoints were retained. USER_RENAME_PAUSE_STATE.json records 6038.835871 active seconds before this second pause. The next attempt is `resume02/`, with previous resume state archived separately and both pauses excluded from the active budget.

Neutral launch implementation 5ef3a16 (regression correction 8d3a344) uses an existing-environment directory alias and a tiny Python entrypoint named main.py. Linux process names are `main`, `run` or `job`; original Python script paths and arguments are supplied internally and retained in private run bookkeeping. The common coordinator launch path also wraps finalization and the later Judge. No algorithm, model, data, seed or hyperparameter changes. This is display hygiene, not anonymity: working directories, environment inspection and open-file inspection still expose actual paths to authorized observers. Eight CPU regression tests pass.

## Historical pause record

Run `20260907_selective_write_v1_r01` is paused. Its coordinator and both GPU0/1 workers exited; no unrelated process was stopped. A STOP marker prevents continued task claims. No automatic restart is authorized.

Retained state: 12 RAW_READY results, 2 PAUSED_BY_USER tasks, 56 PENDING tasks. Both paused tasks (`SW_e02_P4_W0_TASK_ONLY`, `SW_e02_L16_W0_TASK_ONLY`) finished 320 optimizer steps and retain `attempt_chunk16/step0320.pt`, including optimizer/protection state. They were interrupted during endpoint generation, not training. Their incomplete in-memory endpoint outputs may need regeneration. Teacher caches, A2 references, source locks, prior checkpoints and completed results remain intact.

The private resume ledger is `/remote-home/wangbomin/medtrace_runs/20260907_selective_write_v1_r01/private/USER_PAUSE_STATE.json`. It records pause time, the preserved original start, checkpoint paths and sizes, and interrupted phases. Queue entries are explicitly PAUSED_BY_USER rather than stale RUNNING or scientific failures.

On a later explicit resume request: recheck GPU authorization/availability; preserve and reuse the 12 completed results; add/use endpoint-only continuation from the two step320 experts rather than blindly retraining them. The current worker does not yet implement endpoint-only resume. Account separately for paused wall time and use fresh coordinator log/attempt bookkeeping; do not overwrite the original start or run the old coordinator command unchanged. Remove the STOP marker only as part of that authorized resume. This note does not claim resume support has already been implemented or tested.

At shutdown GPU0 showed 1 MiB/0%; GPU1 still showed 15,363 MiB/100% from other work. The project's GPU allocations were released; GPU1 was not claimed globally idle. GitHub's earlier running snapshot is historical; this pause record is retained locally and on the execution host.
