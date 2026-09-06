# Calibration correction report

Status: `ROUTER_CALIBRATION_FIXED`

- Reused candidates: 216 checkpoints from 36 completed B tasks; no Q training or model generation was run.
- Old alternative: `R2_VISUAL_CONDITIONED_CP_RESPONSE@800/SAFETY_FIRST` with calibration positive TPR 98.6%, hard FPR 0.0%, broad FPR 0.0%.
- Fixed alternative: `R2_VISUAL_CONDITIONED_CP_RESPONSE@800/SAFETY_FIRST` with calibration positive TPR 93.8%, hard FPR 0.0%, broad FPR 0.0%.
- Fixed R0 remains `R0_MAGNITUDE_LAST_PROMPT@200/SAFETY_FIRST`.

This correction uses the already viewed development panel and is not a blind or unseen confirmation. Legacy R1/R2 calibration and selection remain preserved as execution evidence but are not deployment-consistent calibration evidence. Track A values were not read or modified by the recalculation.
