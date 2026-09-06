# Router representation development report

Status: `ROUTER_CALIBRATION_AND_SELECTION_COMPLETE`

The preregistered candidates were magnitude-only R0, signed-response R1, and visual-conditioned signed-response R2 at steps 200 and 800, each with SAFETY_FIRST and COVERAGE_CONSTRAINED thresholds. Selection used calibration aggregates only; evaluation scores were opened afterward.

- Selected R0: `R0_MAGNITUDE_LAST_PROMPT@200/SAFETY_FIRST` (calibration positive TPR 85.4%, hard FPR 0.0%, broad FPR 0.0%).
- Selected alternative: `R2_VISUAL_CONDITIONED_CP_RESPONSE@800/SAFETY_FIRST` (calibration positive TPR 98.6%, hard FPR 0.0%, broad FPR 0.0%).

R1/R2 store fit-response prototypes and therefore add routing metadata; they are development extensions, not exact V0.2/TIME reproduction.
