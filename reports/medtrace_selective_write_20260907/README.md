# MedTRACE selective-write V1

Stage 1 is complete: 70/70 trajectories judged, all process exits zero, completed at 2026-09-08 01:23:58 Asia/Shanghai. This is a seven-edit development experiment, not a formal benchmark or new-edit replication. Start with RESULTS_SUMMARY_ZH.md and GPT_PRO_REVIEW.md. RUN_COMPLETION.json and the final result tables supersede older timestamped progress/process snapshots.

Source entry points:

- `methods/medtrace/selective_write.py`: P4/L16 transfer, full-vocabulary KL, group sampling, dual updates, global calibration selector.
- `scripts/medtrace/run_selective_write.py`: prepare, fit-only teacher cache, original A2 reference, 320-step worker and endpoint replays.
- `scripts/medtrace/coordinate_selective_write.py`: bounded, pause-aware attempt; final continuation on user-authorized GPU2/3 with sufficient free VRAM, shared occupancy allowed; unload project 7B workers before Judge.
- `scripts/medtrace/finalize_selective_write.py`: one-shot full-answer Judge packet, source-image/edit summaries, fixed-router identity check, calibration-only W1 lock and paired edit intervals.
- `tests/medtrace/test_selective_write.py`: runnable CPU checks.

Scientific training/cache binding remains 0c72cfe. Later source changes added endpoint-only continuation (c4397a7), neutral process names (5ef3a16/8d3a344), and authorized free-memory sharing (5981c54); they did not change scientific losses/data. Coordinator was initially added at 7f2397d and fixed-Judge closure at 502a912. Runtime preparation and checkpoints remain private; public source snapshots are not substitutes for a complete research runtime tree or permission to use data.

The run is `20260907_selective_write_v1_r01` on the existing `my-gpu` host. Earlier preparation, pause and restart records are retained. The coordinator recorded its own PIDs and exit codes; all have exited. GPU sharing was explicitly authorized, no unrelated process was stopped, and no recurring automation was created.

Scheduling amendment 5106c14: after `prepare` and before `coordinate_selective_write.py`, run `finalize_selective_write.py pair-pending --run-root RUN --public-dir REPORT`. For this already-started attempt it was applied only to PENDING priorities; active jobs were preserved. QUEUE_SCHEDULING_AMENDMENT.json is the explicit execution-order overlay on the unchanged original task/input manifest. The training/cache implementation remains 0c72cfe; the amended finalizer source belongs to 5106c14.

New-edit confirmation N=0 and full-LiveEdit compatibility remain explicit limitations. This release includes complete Stage-1 aggregate results and the previously published source; private inputs, raw answers, tensors and weights are withheld.
