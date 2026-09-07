# Router training and parameter report

C1 Q/P/rho and the backbone were frozen for every task.

- M2: 10 parameters per edit/seed, 40 parameter bytes; mean current training-or-reused-head-load time 3.355s per verifier.
- M3: 28674 parameters per edit/seed, 114696 parameter bytes; mean current training-or-reused-head-load time 3.355s per verifier.

M2/M3 separability: `{"M2": {"calibration": {"image_accuracy": 0.9761904761904762, "question_accuracy": 0.8288359788359788}, "evaluation": {"image_accuracy": 0.8531746031746031, "question_accuracy": 0.8089675756342423}, "fit": {"image_accuracy": 0.8965079365079365, "question_accuracy": 0.8408826945412311}, "image_fit_to_eval_gap": 0.043333333333333335}, "M3": {"calibration": {"image_accuracy": 0.9976190476190476, "question_accuracy": 0.9230158730158731}, "evaluation": {"image_accuracy": 0.8948412698412699, "question_accuracy": 0.9132682132682133}, "fit": {"image_accuracy": 0.9580952380952381, "question_accuracy": 0.910674691162496}, "image_fit_to_eval_gap": 0.06325396825396823}}`

M2 and M3 are supervised diagnostic controls with extra parameters; neither is an intrinsic parameter-free router.

Fit provenance: {'NEW_800_STEP_FIT': 6, 'VALIDATED_REUSE': 36}. The reused-head load duration is not the original 800-step training time. See GPU_AND_TIMING.json for separated current-fit/load times and cache/checkpoint storage.
