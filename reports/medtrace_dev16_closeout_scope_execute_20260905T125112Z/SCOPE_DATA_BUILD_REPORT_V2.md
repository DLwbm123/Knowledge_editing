# MedTRACE scope data build V2

Status before activation extraction: `EXPLORATORY_EVALUABLE__EQKEY_PENDING`.

## Census correction

- Loaded inventory: 16,276 rows (SLAKE train/validation/test and VQA-RAD).
- Actually searched allowed supervision source: 9,835 SLAKE/train rows. Validation, test and VQA-RAD were inventory/audit-only and did not enter role selection.
- Excluded model-visible source keys: 1,378.
- Eligible pool before the 120-row cap: 9,734. It contains 9,708 broad unrelated source QAs, 13 same-image/other-fact rows and 13 same-question/different-image conflicting-answer rows.
- Retained hard-first pool after the cap: 120, containing all 13 + 13 harder rows and 94 broad rows.
- The actual normalized source scan found zero additional English source rows that could be automatically promoted as same-source equivalent positives. This is an observed scan result, not a constant. It does not establish cross-language semantic absence.

Source annotation and expert-relative relation verification are stored as separate private fields. Rows with unknown expert-relative relation are excluded from supervision. Full source-image identity is used; no patient-disjoint claim is made because patient identifiers are unavailable.

## Frozen exploratory roles

Sixteen source-question-only paraphrase candidates were reviewed without model outputs or scores: 12 were approved as equivalent and four were rejected. Approved rewrites are separated by rewrite family into fit, calibration and evaluation, four positives each.

Each role also has 20 source-image-disjoint negatives, for 72 frozen items in total. The selected negatives are all broad unrelated source QAs; therefore this pilot can test broad rejection only and cannot support a hard-negative or clinical-conflict routing claim. EqKeys, processed-image identity and target-free routing metadata will be computed before GPU scoring; any collision stops the scope track.

The primary, raw questions/answers, images, source rows and review text remain in the private server artifact. Role counts alone do not confer V0.2 qualification.
