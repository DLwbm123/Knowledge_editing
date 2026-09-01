# GPT Pro review entrypoint: C0R expert-review handoff

Status: `M3BENCH_C0R_EXPERT_REVIEW_VALIDATED__PRIVATE_IMPORT_PENDING__GPU_NOT_STARTED`

This branch inherits the complete loopback-only review implementation from parent commit
`79889611c754d4854f8ea4addc2fa50c42238928` and adds the post-review evidence needed for
an external protocol/code audit.

## Latest validated evidence

- The private professional-review workbook contains 200/200 explicit verdict codes.
- Frozen sequence, opaque review IDs, image filenames, questions, and target/reference fields match the issued form exactly.
- Verdict counts: direct valid 127; visual-deixis valid 40; context valid 22; invalid 10; uncertain 1; custom 0.
- All 11 invalid/uncertain rows contain a human note at spreadsheet-validation level.
- No source image, image derivative, private workbook, question/target table, reviewer note, local absolute path, or session token is included in this public branch.

Machine-readable aggregate: [`reports/C0R_EXPERT_REVIEW_SUMMARY.json`](../reports/C0R_EXPERT_REVIEW_SUMMARY.json)

## Code GPT Pro should inspect

- Fast-review UI and preset semantics: [`scripts/c0r_review/launch_local_review_fast.py`](../scripts/c0r_review/launch_local_review_fast.py)
- Append-only verdict chain, freeze gate, and Reviewer B queue: [`scripts/c0r_review/reviewctl.py`](../scripts/c0r_review/reviewctl.py)
- Image containment and verification: [`scripts/c0r_review/resolve_authorized_images.py`](../scripts/c0r_review/resolve_authorized_images.py)
- Mechanical spreadsheet-code normalization: [`scripts/c0r_review/professional_review.py`](../scripts/c0r_review/professional_review.py)
- Normalization tests: [`tests/test_c0r_professional_review.py`](../tests/test_c0r_professional_review.py)
- Output contract: [`protocols/c0r_local_review/OUTPUT_SCHEMA.json`](../protocols/c0r_local_review/OUTPUT_SCHEMA.json)

The normalization code deliberately does not interpret images. Codes 1-3 reuse the existing UI presets;
codes 4-5 require the human note and use `issue_type=other` rather than inferring a clinical issue;
code 6 hard-stops until explicit structured human fields are supplied.

## Protocol correction and preservation boundary

The earlier aggregate documents are historical deployment snapshots. A later explicit user authorization
allowed an offline expert package containing 200 exact source-image copies. Therefore, a global claim of
zero image export is no longer valid. This public branch itself still contains zero image bytes or derivatives.

The private UI output currently contains five preserved partial verdicts and has not been frozen. The completed
professional workbook must be imported into a new non-overwriting artifact, checked against those five entries,
then frozen before constructing the blinded Reviewer B queue. This branch does not perform that private import.

GPU editing, Judge, evaluator, semantic metrics, method ranking, and unblinding remain unstarted.
