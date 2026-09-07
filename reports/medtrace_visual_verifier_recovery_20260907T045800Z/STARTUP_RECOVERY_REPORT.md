# Visual-verifier R1 startup recovery

The failed attempt `20260907T032521Z` executed research commit `3aa09dc7196e25e10e90c3965a3f0e3fba678da2`, confirmed from its campaign configuration, launcher CODE field, actual worktree HEAD and traceback. The publication commit `787217abdda75e2951a27285119c8fe5485df3ec` was not the model execution identity.

The actual worktree was `worktrees/medtrace-frozen-expert-visual-verifier-r1-20260907T032521Z` beneath the established research storage root. Its tracked source was clean; one untracked report directory was present and preserved. The runner SHA256 was `c8341e838c3841f0827f2d6f8d389e95fddae4d3522c5a12475eb04be21c968c`. Its origin was a local research Git bundle, not the public release repository. The new repair worktree and branch descend from that verified commit; no remotes, old records or user changes were replaced.

## Confirmed failure

The launcher in the failed run called `recover` and then two `worker` commands with the same run root, old source run and frozen execution run. Neither prepare nor the launcher created `private/CAMPAIGN_START.json`. Each worker loaded the 7B model before `budget_exhausted()` attempted to read that missing file. Both exited 1; all 21 queue entries had status PENDING and attempts=0. No result files or live workers remained. Downstream rc=99 means NOT_RUN. The original launcher, queue, configuration and worker-log hashes are retained in the new private parent-attempt record. No reliable epoch exists for the failed attempt; it remains UNKNOWN.

## Repair and startup tests

The existing runner now exposes a coordinator-owned `start` command. Preparation freezes inputs and task identities; start validates them, locks the start file and atomically records the true UTC/epoch, attempt/parent identity, source/config/manifest hashes, UUID allow-list and budget. A repeated or concurrent start validates and returns the same record. Worker preflight reads and validates it before model loading; it also validates source and required frozen inputs.

Natural ALL_OFF/ALL_ON outcomes now record the absent branch explicitly. Each available branch is replayed using a deterministic first candidate; unavailable branches are not synthesized. Actual answer/token mismatch and frozen-parameter contamination remain errors. Only real replay rows count toward real branch coverage; CPU fixtures never enter performance statistics.

The CPU test command in the existing server Python/Torch environment was:

```sh
CUDA_VISIBLE_DEVICES= /root/anaconda3/bin/python3.12 -m unittest discover -s tests/medtrace -p 'test_*verifier*.py' -v
```

All 15 tests passed (9 original focused tests and 6 recovery integration tests). The added tests exercise the actual prepare/start/worker-preflight chain with a forbidden loader mock, missing/corrupt/misbound starts, concurrent start/resume, 21 exclusive claims, both constant-decision outcomes plus replay mismatch, and the independent exit-code/queue/Judge completion requirements. The initial module-form unittest invocation could not collect this non-package test directory; discovery resolved collection without an environment change.

## New attempt and reuse

Attempt `20260907T045800Z_visual_verifier_retry01` actually began at `2026-09-07T04:57:14.092199+00:00`. The directory label is not used as the start time. The new 12-hour wall / 24 GPU-hour budget begins at that recorded epoch; two times full elapsed wall time is the conservative resource bound including model loading, generation, Judge and waiting. The worker reserves 30 minutes plus an observed task estimate for closure. This is an upper bound, not a claim of full utilization.

An additional, earlier attempt `20260907T021904Z` contained 19 COMPLETE and 2 FAILED tasks. Its execution commit `6ad90540f8196daee006f56fa4fa33897bb077a5` used the superseded M0/M1 score implementation, so its scores/calibration/final task results cannot establish R1 completion. The verified code diff to R1 changes only that score path and its test. Recovery checks the AST identity of feature construction, training-input assembly and base generation, plus unchanged verifier/core/generation dependencies. Compatible feature/base caches and fully bound fit/forced-output artifacts may be reused while preserving the old code/EqKey identity and recording the current execution identity separately. Every M0–M3 score and threshold is recomputed. The first original queue task is fully recomputed on GPU2 as the integration task; it is retained in the 21-task results.

The bounded coordinator keeps GPU2's model resident after the first task and then enables GPU3 for the remaining queue. It waits for every child, validates result digests against unique task IDs, and executes prepare-Judge, Judge (or validated reuse-only closure), then evaluation. It is a finite process, not a service or scheduled monitor. No C1/C2/C3 long training, LoRA retraining, new encoder or new sample was authorized or executed by these changes.

## Reporting correction

The original finalizer called gated correctness “joint” in the development signal. The approved plan explicitly requires ON AND correct. Recovery reports both and applies the existing numeric thresholds to actual joint success; this does not alter any trained parameter, score, selected threshold, sample or raw answer. Historical results and the LoRA QUAL_VALIDATION_FAIL remain unchanged.

## Observed NFS error and one recovery per task

Seven task attempts encountered `OSError: [Errno 5] Input/output error` while reusing caches. A targeted traceback isolated `shutil.copy2 -> copystat -> _copyxattr -> os.setxattr` on the NFS mount. The e02 feature payload had the expected size and loaded successfully as 69 cached inputs with its original locks. This was a metadata-copy failure after payload transfer, not evidence of a corrupt source feature.

The remaining absent base caches were materialized with `shutil.copyfile`; existing files were not overwritten. Each affected FAILED task was returned once to PENDING under the existing queue lock, after its complete prior attempt entry and reason were saved in TASK_RETRY_HISTORY.jsonl. RUNNING tasks were not reset or stolen. All seven retries completed; no task was retried for a low score, ALL_OFF or lack of benefit. The running source and epoch remained unchanged. The distributed follow-up source replaces the two `copy2` calls with `copyfile` to prevent recurrence; this later hardening commit is distinct from the execution commit.

The first original task passed in 128.992 seconds, including feature extraction 33.366 seconds, two fresh fits 45.993 seconds, scores 2.544 seconds, generation 35.752 seconds and real replay 4.105 seconds. GPU3 then joined while GPU2 retained its loaded model. All 21 logical tasks subsequently completed, and both workers exited 0. Judge preparation validated 759 reusable tuples and identified 275 new tuples.

The self-contained public source snapshot (including the NFS hardening and portable CPU fixtures) was tested in a separate temporary Git checkout on the same existing server environment: all 15 tests passed. No dependency was installed or upgraded.

See RUN_COMPLETION.json, JUDGE_AND_RAW_CLOSURE.json and GPT_PRO_REVIEW.md for the independent judging, publication and scientific outcomes.
