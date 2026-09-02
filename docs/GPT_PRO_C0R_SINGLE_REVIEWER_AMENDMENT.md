# C0R single-reviewer amendment package

Start with:

1. [`SINGLE_REVIEWER_PROTOCOL_AMENDMENT.md`](../reports/SINGLE_REVIEWER_PROTOCOL_AMENDMENT.md)
2. [`SINGLE_REVIEWER_FINAL_TARGET_CENSUS.json`](../reports/SINGLE_REVIEWER_FINAL_TARGET_CENSUS.json)
3. [`AMENDMENT_OPTIONS.md`](../reports/AMENDMENT_OPTIONS.md)
4. [`CONTAMINATION_RECALCULATION.json`](../reports/CONTAMINATION_RECALCULATION.json)
5. [`RAW_RERUN_SCOPE_PROPOSAL.md`](../reports/RAW_RERUN_SCOPE_PROPOSAL.md)
6. [`single_reviewer_amendment.py`](../scripts/c0r_review/single_reviewer_amendment.py) and its [unit test](../tests/test_c0r_single_reviewer_amendment.py)

The public package intentionally excludes item-level positions, source records, questions, targets, image hashes, source images, reviewer identifiers, raw model answers, and Judge outputs. Those private mappings remain in the non-public run artifact. No final amendment has been selected and no GPU work has started.
