# MedTRACE selective-write V1

This is an in-progress development run, not a completed or formal benchmark experiment. Start with GPT_PRO_REVIEW.md.

Source entry points:

- `methods/medtrace/selective_write.py`: P4/L16 transfer, full-vocabulary KL, group sampling, dual updates, global calibration selector.
- `scripts/medtrace/run_selective_write.py`: prepare, fit-only teacher cache, original A2 reference, 320-step worker and endpoint replays.
- `scripts/medtrace/coordinate_selective_write.py`: one bounded attempt on user-authorized, idle GPU0/1; first full task integration before continuation workers; unload all 7B workers before Judge.
- `scripts/medtrace/finalize_selective_write.py`: one-shot full-answer Judge packet, source-image/edit summaries, fixed-router identity check, calibration-only W1 lock and paired edit intervals.
- `tests/medtrace/test_selective_write.py`: runnable CPU checks.

Training/cache implementation is locked to 0c72cfe (the actual training file is unchanged by later closure/report commits). Coordinator was added at 7f2397d and fixed-Judge closure at 502a912. Runtime preparation and checkpoints remain private; public source snapshots are not substitutes for a complete research runtime tree or permission to use data.

The run is `20260907_selective_write_v1_r01` on the existing `my-gpu` host. An earlier CPU-only preparation rejection is retained separately. The live coordinator records only its own worker PIDs and exit codes. No recurring automation was created, no GPU2/3 job was shared or terminated, and no old results were overwritten.

Scheduling amendment 5106c14: after `prepare` and before `coordinate_selective_write.py`, run `finalize_selective_write.py pair-pending --run-root RUN --public-dir REPORT`. For this already-started attempt it was applied only to PENDING priorities; active jobs were preserved. QUEUE_SCHEDULING_AMENDMENT.json is the explicit execution-order overlay on the unchanged original task/input manifest. The training/cache implementation remains 0c72cfe; the amended finalizer source belongs to 5106c14.

New-edit confirmation N=0 and current full-LiveEdit compatibility remain explicit limitations. Complete stage-1 results, when available, must still be synchronized and publicly published; this snapshot does not claim that future publication has occurred.
