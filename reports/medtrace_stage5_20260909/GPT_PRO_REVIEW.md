# MedTRACE Stage5 factual review

Status: COMPUTE_COMPLETE; publication tracked separately in PUBLICATION_STATUS.json when available.

## Direct answers and measured limitations

1. **Does RC preserve positives while reducing misfires?** On old SLAKE16, all three writers retain 100% native and both constructed positive-style accuracy (16 native, 32 probes per style), while U activation drops from 92.10% to 0% and H activation from 100% to 3.33% (edit macro). New32 cannot independently confirm this rejection benefit: H calibration is absent, so every new RC is the explicitly unsupported R0 fallback. New positive accuracy is 100% on the available writers, with zero additional RC rejection, not evidence of successful new calibration. Official new image/text generality is NA.
2. **W0_RC versus W1_RC?** On old common7 official T2G (6 edits, 23 probes), edit-macro accuracy is W0 83.33%, W1 30.56%, BE 87.50%. The locked bank16 threshold-transfer diagnostic changes none of these values or ON coverage (100%). This is a development diagnostic, separate from SLAKE16 and new confirmation. On SLAKE16 RC, both W0 and W1 have observed zero H/U Base-correct damage. The new W1 common-support cohort has only one edit, so it cannot establish that KL is unnecessary.
3. **Does BE benefit from the same rejection?** Yes on old SLAKE16: BE H damage falls from 100% to 3.85%, U from 90.43% to 0%; native and constructed positive accuracy stay 100%. U is entirely Base return, therefore this gain is a generic rejection effect, not a CP-specific writer gain. H retains residual BE damage while W0/W1 show zero observed damage on the sparse accepted support.
4. **Does KL add protection beyond OFF-to-Base?** With old R0 fixed, W1 U damage is 23.99% versus W0 59.26%, but formal T2G is much worse. Under old RC, negative acceptance becomes too sparse to identify a broad incremental KL benefit. New data provides only one supported W1 and no evaluation H, so that question remains under-supported, not resolved negatively.

## Final-bank evidence (R0 = RC on new data)

| Writer | Actual bank N | Native correct | Each derived style correct | U full-source accuracy, edit macro | U Base-correct damage, edit macro |
|---|---:|---:|---:|---:|---:|
| BE | 32 | 32/32 | 64/64 | 43.23% | 0% |
| W0 | 32 | 32/32 | 64/64 | 57.81% | 0% |
| W1 | 1 | 1/1 | 2/2 | 42.86% | 0% |

U contains 220 edit-input observations for N32 but only 7 distinct evaluation source images; W1 contains 7 observations/images. Reused image observations are not independent samples. All new-bank U requests activate an expert (100% FPR), so zero observed damage is not evidence of selective rejection or universal safety. Exact metric numerators/denominators, common-support comparisons and image-cluster sensitivity are retained in the CSV files. Different N prohibits a same-bank W1 comparison.

The single-edit and bank results must not be conflated: single U damage is BE 21.88%, W0 3.12%, W1 0% (last value N=1); bank selection can use a different expert. No expert was dropped for poor performance. Wall time was 3.56 hours on GPU2; this is also a conservative GPU-hour upper bound, not measured utilization. There were 32 shared A2 initializations and 65 writer endpoints, no training failures, plus 31 explicitly unsupported W1 endpoints. Final Judge reused 311 Base-before scores and generated 783 additional scores; those 311 Base-before scores were themselves newly evaluated earlier in Stage5, not inherited Stage4 Judge savings.

Track A reused frozen outputs/Judge with zero old writer training and zero additional Judge calls. Its six necessary natural ON/OFF generation replays all passed. Source review covers 32 distinct source images plus 23 negative-source images, patient identity UNKNOWN; no clinical-human signoff or independent semantic-family claim. Private source text input and original frozen source-packet provenance are not published. The released source accepts explicit private conflict annotations; the original packet generation predates that input-schema parameterization and must not be silently regenerated as if byte-identical.

## What the new confirmation can establish

The original 32 source-image-distinct candidates were retained. Source-only derived texts have two surface styles but one native semantic lineage per fact; they are not official M3Bench rephrases or independently clinically reviewed facts.
New calibration has no H support. RC therefore equals R0 under CALIBRATION_UNSUPPORTED, rather than providing an independent confirmation of calibrated rejection. No evaluation threshold was fitted.

## Writer and routing contribution

EXISTING_RC_WRITER_FACTORIAL.csv compares old W0/W1/BE at identical frozen routing; NEW_EDIT_SINGLE_RESULTS.csv gives per-episode RC and FORCED_ON. NEW_PAIRED_EFFECTS.csv uses paired common support only.
W0/W1 comparisons on new edits are supported by at most one edit; do not claim KL unnecessary or generally effective from this sample. Final-bank sizes differ when writer support is missing; cross-size effects are not same-router writer ablations.
BE_RC_ADAPTATION remains separate from native BalancEdit. An all-OFF result is Base return, not a writer improvement. Damage, full-source accuracy and Base-wrong improvement retain separate denominators.

## Coverage and closure

Completed writers: {'BE': 32, 'W0': 32, 'W1': 1}. Coverage states: {'RAW_READY': 65, 'UNSUPPORTED_MISSING_FIT_H_OR_U': 31}. Bank sizes: {'BE': 32, 'W0': 32, 'W1': 1}.
Judge: {'required': 1094, 'scored': 1094, 'missing': 0, 'reused': 311, 'new': 783}. Original image/text generality: NA where unavailable.
No old writer training or historical Judge rerun was required by Track A. Old transfer is a development diagnostic, not new confirmation. Natural-branch replay evidence is separate from derived output provenance.
Publication is a separate pending state until a later verified Git commit/public URL. Never rerun computation for a network failure.

## Old fixed-routing factorial (edit macro)

| Writer | Route | Panel | Full-source accuracy | Base-correct damage | Activation |
|---|---|---|---:|---:|---:|
| BE | R0 | H | 0.00% | 100.00% | 100.00% |
| BE | R0 | U | 8.39% | 90.43% | 92.10% |
| BE | RC | H | 73.33% | 3.85% | 3.33% |
| BE | RC | U | 64.20% | 0.00% | 0.00% |
| W0 | R0 | H | 63.33% | 50.00% | 100.00% |
| W0 | R0 | U | 28.68% | 59.26% | 92.10% |
| W0 | RC | H | 76.67% | 0.00% | 3.33% |
| W0 | RC | U | 64.20% | 0.00% | 0.00% |
| W1 | R0 | H | 50.00% | 30.77% | 100.00% |
| W1 | R0 | U | 50.53% | 23.99% | 92.10% |
| W1 | RC | H | 76.67% | 0.00% | 3.33% |
| W1 | RC | U | 64.20% | 0.00% | 0.00% |

## New single-edit confirmation (edit macro)

| Writer | Route | Panel | Inputs | Available edits | Full-source accuracy | Base-correct damage |
|---|---|---|---:|---:|---:|---:|
| BE | R0 | U | 220 | 32 | 20.39% | 21.88% |
| BE | R0 | derived_imperative_style | 64 | 32 | 100.00% | NA |
| BE | R0 | derived_source_style | 64 | 32 | 100.00% | NA |
| BE | R0 | native_source_edit | 32 | 32 | 100.00% | NA |
| BE | RC | U | 220 | 32 | 20.39% | 21.88% |
| BE | RC | derived_imperative_style | 64 | 32 | 100.00% | NA |
| BE | RC | derived_source_style | 64 | 32 | 100.00% | NA |
| BE | RC | native_source_edit | 32 | 32 | 100.00% | NA |
| W0 | R0 | U | 220 | 32 | 25.07% | 3.12% |
| W0 | R0 | derived_imperative_style | 64 | 32 | 100.00% | NA |
| W0 | R0 | derived_source_style | 64 | 32 | 100.00% | NA |
| W0 | R0 | native_source_edit | 32 | 32 | 100.00% | NA |
| W0 | RC | U | 220 | 32 | 25.07% | 3.12% |
| W0 | RC | derived_imperative_style | 64 | 32 | 100.00% | NA |
| W0 | RC | derived_source_style | 64 | 32 | 100.00% | NA |
| W0 | RC | native_source_edit | 32 | 32 | 100.00% | NA |
| W1 | R0 | U | 7 | 1 | 42.86% | 0.00% |
| W1 | R0 | derived_imperative_style | 2 | 1 | 100.00% | NA |
| W1 | R0 | derived_source_style | 2 | 1 | 100.00% | NA |
| W1 | R0 | native_source_edit | 1 | 1 | 100.00% | NA |
| W1 | RC | U | 7 | 1 | 42.86% | 0.00% |
| W1 | RC | derived_imperative_style | 2 | 1 | 100.00% | NA |
| W1 | RC | derived_source_style | 2 | 1 | 100.00% | NA |
| W1 | RC | native_source_edit | 1 | 1 | 100.00% | NA |

Tables use available denominators, never count unavailable W1 endpoints as successes. The separate 96-row writer coverage manifest retains every planned edit/writer pair. For common-support inference consult paired_inputs/paired_edits; one edit has no bootstrap confidence interval.
