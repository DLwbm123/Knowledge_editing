# Prototype calibration root cause

Status: `CONFIRMED_AND_FIXED`

The original execution source and final source were identical for the affected scorer. R1/R2 checkpoints stored prototypes built from scope-fit positives, but the legacy calibration function rebuilt prototypes from calibration positives. Request inference instead read the checkpoint prototypes, so calibration and deployment evaluated different functions. R0 never used a prototype and is unchanged.

The repair separates fit-only prototype construction from scoring with frozen prototypes. Calibration, evaluation and request inference now share score definition `117533708068ae04c46478433e5d752f5bfb254c86304401fbf5e6f7b20e8ea8`; scoring receives no role labels. All 216 checkpoint files and their Q states matched the original run ledger, and every saved prototype was exactly rebuilt from the original fit-positive cache.
