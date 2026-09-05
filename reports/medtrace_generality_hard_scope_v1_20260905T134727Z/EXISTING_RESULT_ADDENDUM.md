# Existing-result addendum

## Exact versus semantic Judge

Six rows (T0: 1; T1G: 5) had normalized exact=true but semantic=false. All six raw answers are byte-for-byte equal to their bound references before normalization; five belong to one edit and one to another. Therefore normalization did not alter their semantics and the evidence points to a Judge/reference-relative objective disagreement rather than model output mismatch. The published verdicts remain unchanged because no independent second clinical review was performed; these six are `UNRESOLVED_JUDGE_REFERENCE_DISAGREEMENT`.

## Existing scope paired behavior

- Original-Q and Final-Q each activated 2/20 negatives; the activated sets were identical (overlap 2, Original-only 0, Final-only 0).
- Negative semantic correctness: Base 14/20, Original-Q control 14/20, Final forced-on 13/20, Final gated 14/20.
- Token changes versus Base: Original-Q control 0/20, Final forced-on 11/20, Final gated 0/20.
- Among 14 Base-correct negatives, forced-on damaged 1 and gated damaged 0.
- Final-Q ON occurred on 2 Base-correct negatives; conditional gated damage is 0.0%.
- Original-Q control answered 4/4 evaluation positives correctly.

Because forced-on caused no observed Base-correct negative damage on this broad panel, the old panel did not demonstrate a behavioral-damage reduction attributable to gating.
