# GPT Pro review: selective-write V1 in progress

Current answer to all four scientific questions:

- **Does ordinary KL help?** Not yet established. The first task's fit drift is decreasing; the W0 comparison, endpoint semantic preservation and calibration-selected W1 are pending.
- **Does group-constrained W2 add value over tuned W1?** Not yet evaluated. W1 has three preregistered weights; W2 is fixed and must not be tuned on evaluation.
- **Is CP capacity/parameterization limiting?** Not yet evaluated. P4 and L16 start from the same A2 function, with real-activation transfer checks passed; their parameter counts differ by about 200x, so any difference is joint evidence, not capacity-matched proof.
- **Does the effect reproduce on new edits?** No such evidence yet. N=0 authorized-and-exposure-verified new episodes in the inspected assets; the old seven facts are viewed development data.

Execution: seed 20260906, 70 trajectories, 320 appended steps, P4/L16 each with W0, W1(0.1/1/10) and W2. Original A2 is a read-only reference. FORCED_ON is primary; original M0 decisions are frozen; DISABLED must restore Base. The latest user explicitly authorized GPU0/1, replacing the earlier plan's GPU1 prohibition. GPU2/3 are occupied and untouched.

Latest verified progress: first e01 P4/W1(0.1) is RAW_READY after 320 optimizer steps. Training plus final fit diagnostics took 188.890 seconds; the complete task including shared initialization/reference and natural endpoint work took 1169.806 seconds. Its 140 endpoint inputs passed DISABLED/Base parity, fixed-router branch replay and unchanged-backbone checks. The coordinator's FIRST_TASK_PASS is present and both GPU0/1 continuation workers are running: one raw-ready task, two running, 67 pending at this snapshot. No new semantic verdict exists yet. LIVE_PROGRESS.json and SELECTIVE_WRITE_BY_EDIT.csv distinguish trained checkpoints from completed endpoints. Both research and public snapshots have been pushed and their review pages anonymously accessed successfully; final results publication remains pending completion. No whole-queue ETA is inferred from this single completed task.

Ponytail's reuse principle guided the implementation: retain existing hook lifecycle, runtime, atomic queue/storage, telemetry and fixed Judge; add the selective-write loss/cache/worker and explicit partial-result closure. No new router or external teacher was added. First-task integration must complete before the coordinator expands to both idle authorized GPUs. Low semantic scores do not stop otherwise-valid tasks. OOM has one recorded chunk-size retry; failures remain failures.

Read SELECTIVE_WRITE_PROTOCOL.json, FIT_SOURCE_INVENTORY.json, TRANSFER_CPU_RESULTS.json, CPU_REGRESSION_REPORT.md, NOVEL_EDIT_CONFIRMATION.md and BASELINE_AND_NOVELTY_LEDGER.md. SELECTIVE_WRITE_BY_EDIT.csv includes all 70 task slots without fabricated scores. This is an **in-progress implementation and run**, not a completed experiment or a full LiveEdit/TIME/M-ORE/M3Bench reproduction.

Scheduling-only correction: QUEUE_SCHEDULING_AMENDMENT.json records before/after priorities for pending tasks. Execution now advances edit blocks with P4/L16 paired within W0/W1/W2 conditions, so all W2 tasks are not left until the final condition-wide sweep. Running/completed tasks, the original input manifest, all task IDs, data, seeds, losses and hyperparameters are unchanged. No semantic or selection result was available or consulted. Six CPU regression tests pass after this amendment.
