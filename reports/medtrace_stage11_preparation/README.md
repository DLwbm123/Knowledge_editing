# Stage11 prepared; user explicitly deferred execution

Status: NOT_LAUNCHED. No Stage11 run root, GPU worker, Judge, scheduler or automatic monitor has been started. User reserved GPU0/1 for other work and requested preparation only. Obtain a new user instruction before running preparation/launch commands; do not auto-resume when memory becomes free.

Prepared three fixed conditions: J0 CP rank4, J1 free rank4, J2 free rank16. Same original Stage9 15-edit source-bound cohort, W0 step320 initialization, F1 objective and deterministic 320-step schedule; outputs at step160/320. Actual supported count is subject to the one bounded source/contradiction audit at preparation time. Stage10 five-edit ROI results are a reused subgroup, not cohort selection.

Inspected historical F1 step160 checkpoint: expert, optimizer, step, condition, order, curve, forwards, seed; no RNG state. J0 must rerun320 from W0, not claim strict continuation. Maximum14400 new optimizer steps, 45 final trajectories. New checkpoints retain PyTorch CPU/CUDA and Python RNG state. First-answer-token rank is explicitly NA. Base/teacher/Judge use existing binding-aware reuse. No new loss, rank search, clinical facts or automatic next experiment.

Entrypoints: scripts/medtrace/stage11.py (prepare/worker/prepare-judge/report), scripts/medtrace/coordinate_stage11.py (explicit detached launch), shared LowRankExpert unchanged. Default authorized devices only0/1; live checks at eventual launch allow one worker if only one device fits. 8h wall/16GPUh cap, last1.5h reserved for judging/reporting. Process arguments use existing neutral run/job/main wrapper.

CPU test: tests/test_joint_fact_writer.py covers residual transfer/scaling, A/B optimizer binding, live extra-rank gradients, normalization, save/reload and hook disable/reset. Real native/H generation parity checks are deferred until the first actual task, not claimed as passed now. Hardware-dependent training, checkpoint outputs, Judge coverage and effectiveness remain unverified until execution.

Future launch sequence, only after renewed user instruction: verify authorized free memory and output mount, run stage11.py prepare with the original Stage11 protocol and execution commit, then coordinate_stage11.py launch. Do not run prepare now: it starts the campaign budget clock. Complete source audit before any worker. Review quantitative FIT/EVAL PairCorrect four-cell outcomes, native/cross-expression and U damage before recommending a writer; do not use task completion as evidence of success.

Prior evidence: Stage10 public b84a0f35c6d8b6b4f8f8c475e48ef1e3ab3d2ef3, Stage9 public4cfe33ecc81d91127b96aeeeef4c0b6d6ed15edd. Preserve all prior conclusions and private assets. Publication of completed results is deferred because this experiment has not run.
