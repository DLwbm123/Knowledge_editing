# MedTRACE scope source census

Status: `SOURCE_CENSUS_COMPLETE__POSITIVE_AUGMENTATION_REQUIRED`

The private census scanned all locally available source QA records once: SLAKE train 9,835, validation 2,099, test 2,094, and VQA-RAD 2,248. It also loaded the frozen DEV16, LoRA QUAL16, 179 formal edit records, and 1,612 formal probes to build a 1,378-key exclusion index before selecting candidates.

The fixed primary record resolves uniquely to SLAKE train. Its native question and one official rephrase are available as same-source development positives. No additional source QA shares the same structured triple and answer, so role-isolated positive sets require the separately labelled textual augmentation pilot before any activation scoring.

After exclusions, 120 deterministic source-annotated negative candidates were retained. All are broad unrelated source QA under the current bounded selector; the same-image and same-question candidates were already part of excluded formal material. Candidate rows remain unassigned until model-visible EqKeys and positive augmentation are frozen. UNKNOWN candidates do not enter supervision.

No scope activation, fit, threshold, calibration metric, or evaluation output has been computed. The private record-level census SHA-256 is `670227c34476dddc78a71e1d165db81ef81bddfd10b0a2255d8d087e9d03bf9b`; raw QA, answers, paths, and candidate mappings remain server-only.
