# MedTRACE Stage4 factual review

Status: COMPLETE_EXECUTABLE_CONFIRMATION_UNAVAILABLE. Judge missing=0. Publication state here is the generator-time snapshot; verify the Git commit for later public delivery.

1. Did weaker KL recover original T2G, and what protection was lost?

- OLD_STAGE3_COMMON7 R0 original T2G W01-W0: -30.56 pp; CI [-43.06, -16.67] pp; 6 edits / 23 inputs / 6 probe images
- OLD_STAGE3_COMMON7 R0 original T2G W01-W1: +22.22 pp; CI [8.33, 36.11] pp; 6 edits / 23 inputs / 6 probe images
- OLD_STAGE3_COMMON7 FORCED_ON original T2G W01-W0: -30.56 pp; CI [-43.06, -16.67] pp; 6 edits / 23 inputs / 6 probe images
- OLD_STAGE3_COMMON7 FORCED_ON original T2G W01-W1: +22.22 pp; CI [8.33, 36.11] pp; 6 edits / 23 inputs / 6 probe images
- SLAKE16 FORCED_ON H damage W01-W0: -38.46 pp; CI [-69.23, -15.38] pp; 13 edits / 15 inputs / 6 probe images
- SLAKE16 FORCED_ON H damage W01-W1: +7.69 pp; CI [0.00, 23.08] pp; 13 edits / 15 inputs / 6 probe images
- SLAKE16 FORCED_ON U damage W01-W0: -27.96 pp; CI [-40.79, -17.90] pp; 16 edits / 130 inputs / 31 probe images
- SLAKE16 FORCED_ON U damage W01-W1: +4.69 pp; CI [1.04, 9.20] pp; 16 edits / 130 inputs / 31 probe images

2. Is the loss still present under FORCED_ON?

The separate FORCED_ON effects above isolate the writer from rejection; negative values remain a writer-side observed tradeoff, not a closed route. Intermediate checkpoints are diagnostic only.

3. Does RC reject non-target inputs with fixed support?

- Fixed W1 RC-R0, prefix16 H fpr: -96.67 pp; CI [-100.00, -90.00] pp; 15 edits / 20 inputs / 7 probe images
- Fixed W1 RC-R0, prefix16 H base_correct_damage: -30.77 pp; CI [-53.85, -7.69] pp; 13 edits / 15 inputs / 6 probe images
- Fixed W1 RC-R0, prefix16 H semantic: +26.67 pp; CI [6.67, 53.33] pp; 15 edits / 20 inputs / 7 probe images
- Fixed W1 RC-R0, prefix16 H base_wrong_became_correct: +0.00 pp; CI [0.00, 0.00] pp; 5 edits / 5 inputs / 1 probe images
- Fixed W1 RC-R0, prefix16 U fpr: -92.10 pp; CI [-95.14, -89.04] pp; 16 edits / 202 inputs / 41 probe images
- Fixed W1 RC-R0, prefix16 U base_correct_damage: -23.99 pp; CI [-31.59, -17.05] pp; 16 edits / 130 inputs / 31 probe images
- Fixed W1 RC-R0, prefix16 U semantic: +13.67 pp; CI [8.46, 19.37] pp; 16 edits / 202 inputs / 41 probe images
- Fixed W1 RC-R0, prefix16 U base_wrong_became_correct: -5.42 pp; CI [-10.52, -1.25] pp; 16 edits / 72 inputs / 23 probe images
- Fixed W1 RC-R0, prefix16 T0 semantic: +0.00 pp; CI [0.00, 0.00] pp; 16 edits / 16 inputs / 16 probe images
- Fixed W1 RC-R0, prefix16 source_style_confirmation semantic: +0.00 pp; CI [0.00, 0.00] pp; 16 edits / 32 inputs / 16 probe images
- Fixed W1 RC-R0, prefix16 cross_family_confirmation semantic: +0.00 pp; CI [0.00, 0.00] pp; 16 edits / 32 inputs / 16 probe images

4. Writer, rejection, and other-expert contributions

WRITER_PAIRED_EFFECTS isolates W01 at fixed R0/FORCED_ON. BANK_PAIRED_EFFECTS isolates rejection at fixed W1 and supplies W0/W01 interactions. ROUTE_FAILURE_DECOMPOSITION reports P(ON|strict Base), P(wrong|ON,Base-correct,strict Base), final damage, all-source accuracy and Base-wrong correction/change with separate denominators. Other-expert correctness is a descriptive attribution, not an independent causal contribution. Owner mismatch alone is not scored as wrong.

5. Independent confirmation

CONFIRMATION_UNAVAILABLE: 32 capped candidates, zero complete authorized new edits. Existing-source/fact-specific positive support and patient independence are not invented; see UNSEEN_CONFIRMATION_REPORT.md. The 7 and 16 development cohorts remain separate.

6. Preregistered descriptive decisions

NO_DESCRIPTIVE_JOINT_WRITER_CANDIDATE: one or more fixed criteria failed or lacked support.
DESCRIPTIVE_SCOPE_CANDIDATE

No independent confirmation supports a resolved-generalization, noninferiority, clinical-safety, intrinsic-routing or sequential-200 claim. No automatic Stage5. Threshold constraints are engineering constraints, not 95% guarantees.

Statistics: fixed 10,000 paired-edit bootstrap, seed 20260908; image-cluster and leave-one-image-out sensitivities accompany shared-source dependence. Cases/patients UNKNOWN. Zero denominators NA. Native diagnostics and constructed confirmation are not official T0/T2G substitutes. All-source correctness, Base-wrong correction and Base-correct damage remain side by side.

Stage3 continuity: 7 common groups include six formal T0 anchors; original W1 T2G macro 30.56%, paired W1-W0 -52.78 pp also under FORCED_ON, and T1L damage 100% on six Base-correct inputs. Stage3 was complete before this run; its static pending-publication wording did not cause a rerun.
