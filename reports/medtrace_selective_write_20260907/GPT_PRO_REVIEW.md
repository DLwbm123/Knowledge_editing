# GPT Pro review

See [the numerical summary](RESULTS_SUMMARY_ZH.md) for all eleven A2/P4/L16 conditions and limitations. Key findings: calibrated P4 W1 (lambda 0.1) retains 100% evaluation positive semantics and reduces U Base-correct damage from 31.46% to 10.63%; P4 W2 has the same positive/damage result but larger H/U KL. L16 W2 retains 100% positives and reduces damage from 77.72% to 10.63%, whereas all L16 W1 candidates fail calibration; its lambda 1 comparison is explicitly an unqualified fallback. W2 fit H constraints pass only 3/7 P4 and 6/7 L16 edits. No new-edit replication or universal W2 superiority follows.

Judged trajectories: 70/70. New-edit confirmation N=0; no new-edit replication claim.

Calibration-only W1 selection: `{"L16": {"eligible": [], "lambda": 1.0, "qualified": false, "role": "calibration"}, "P4": {"eligible": [0.1], "lambda": 0.1, "qualified": true, "role": "calibration"}}`.

FORCED_ON is primary. FIXED_ROUTER decisions are invariant across all write conditions; its FPR cannot improve here. DISABLED was replayed against Base. Full raw answers, not excerpts, were sent to the locked Judge.

See SELECTIVE_WRITE_BEHAVIOR_MACRO.csv and SELECTIVE_WRITE_BEHAVIOR_BY_EDIT.csv for separate native/fit/calibration/evaluation, T1G/T2G/T1L, H/U and challenge strata. Source-image macro and row micro denominators are explicit. See PAIRED_EDIT_EFFECTS.csv for W1/W2 minus W0 and W2 minus selected W1, with edit-cluster bootstrap intervals; missing paired edits are reported, not imputed.

W1 ordinary KL benefit and W2 incremental benefit must be read separately. A low fit KL alone is insufficient; check held-out positive preservation and negative behavior together. Cross-P4/L16 differences are joint capacity/parameterization/optimization evidence, not capacity-matched proof. Multipliers at cap or positive constraint residuals mean the prescribed fit constraints were not reached.

Full-answer target consistency is reported separately from source-reference correctness. Base-correct damage and Base-wrong changes have distinct denominators. Maintaining an already-wrong Base answer is not clinical correction. No novelty, full TIME/LiveEdit/M-ORE reproduction, SOTA or clinical safety claim.

Review questions: (1) Does calibrated ordinary KL improve forced-on behavior without losing positive semantics? (2) Does W2 add a paired benefit over that W1? (3) Is any effect confined to L16, with the capacity confound disclosed? (4) New edits have NOT been tested; do not treat seven viewed facts as replication.
