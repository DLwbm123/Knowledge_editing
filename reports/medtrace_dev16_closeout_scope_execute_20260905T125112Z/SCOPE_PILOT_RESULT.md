# MedTRACE scope pilot

Status: `SCOPE_PILOT_EVALUATION_COMPLETE`

Execution mode is `EXPLORATORY_EVALUABLE`; this is a single-image, single-fact text-augmentation pilot and is not V0.2 qualified.

- Roles: fit 4 positive/20 negative; calibration 4/20; evaluation 4/20.
- Original-Q control: threshold `5.1938782`, calibration TPR/FPR `1.000/0.000`, evaluation positive activation `1.000`, negative FPR `0.100`.
- Final Q: threshold `4.388545`, calibration TPR/FPR `1.000/0.000`, evaluation positive activation `1.000`, negative FPR `0.100`.
- New positives semantic correctness: Base `0/4`, forced-on `4/4`, gated `4/4`.
- Base-correct negative preservation: `14/14`.
- All-negative exact behavior preservation: `20/20`.
- OFF token parity: `18/18`.
- Activated-negative semantic damage: `0/20`.
- Native gate ON/correct: `True/True`.

Full raw QA, answers, images, activations, EqKeys, checkpoint and Judge mapping remain private.
