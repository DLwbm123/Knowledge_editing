# Historical M3 rejection decomposition

Read-only diagnostic from the completed recovery run. No new model or Judge call. Neither operating point replaces the historical PRIMARY result.

|Point|Positive N|ON|Question-only reject|Image-only reject|Both reject|Image-always-ON diagnostic upper bound|
|---|---:|---:|---:|---:|---:|---:|
|PRIMARY_SAFETY_FIRST|84|65|16|1|2|66|
|SECONDARY_COVERAGE90|84|65|16|1|2|66|

Per-edit/seed counts and both thresholds: REJECTION_BY_EDIT_SEED.csv. Hard false-positive head scores and margins: HISTORICAL_HARD_FP_MARGINS.csv. Paired ON/OFF gains and losses (positive activation / negative rejection, not semantic accuracy): HISTORICAL_M0_M3_PAIRED_TRANSITIONS.csv.
The upper bound is a question-gate diagnostic at historical frozen thresholds, not a method score. New two-head calibration may change both thresholds even though question scores are identical. Historical scientific_gain=false remains unchanged.
