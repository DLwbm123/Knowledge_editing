# GPT Pro review packet: MedTRACE DEV16 closeout and scope pilot

## Review boundary

This packet reports a 16-edit development experiment of `TIME_INSPIRED_CP_R4_DEV16_FORCED_ON` and a single-image, single-fact `MEDTRACE_NATIVE_TEXT_SCOPE_AUGMENTATION_PILOT`. It is not full TIME, full MedTRACE, an independent benchmark result, or V0.2 qualification. T0 was used for fitting and native early stopping.

The existing DEV16 checkpoints and 163 saved generations were evaluated without retraining or backbone regeneration. All 222 Track A Judge requests (163 CP plus 59 matched saved Base T2G outputs) and all 96 scope Judge requests parsed successfully under the fixed Qwen3-32B-AWQ protocol. Track A required a shared 2,048-token Judge lane: maximum prompt length 1,232 plus the 24-token output budget, with no truncation. The scope packet fit the original 1,024-token lane.

## Track A: CP-DEV16

| Task | Valid edits | n | Exact micro | Exact macro | Semantic micro | Semantic macro |
|---|---:|---:|---:|---:|---:|---:|
| T0 | 16 | 16 | 16/16 (100.0%) | 100.0% | 15/16 (93.8%) | 93.8% |
| T1L | 7 | 26 | 8/26 (30.8%) | 41.4% | 8/26 (30.8%) | 41.4% |
| T1G | 16 | 62 | 62/62 (100.0%) | 100.0% | 57/62 (91.9%) | 92.2% |
| T2G | 16 | 59 | 22/59 (37.3%) | 38.0% | 31/59 (52.5%) | 52.1% |

All 16 experts met the native training stop by step 200; the mean selected step was 148.75. Mean target NLL fell from 4.317495 to 0.168988 and decreased for all 16 edits. This supports native fit, not independent generalization.

The 163 saved outputs contain 160 EOS-terminated generations and three length-limit generations without EOS, all in T2G. Classification is derived from saved token IDs because the historical runner did not retain stop reasons. The full saved answers, including the three long outputs, were judged without summarization or first-sentence replacement.

For T1L, the source/Judge diagnostic contains 18 `source_and_judge_supported_conflict` and eight `compatible` rows. This is a source-supported lexical/semantic diagnostic, not independent image-based clinical adjudication.

For T2G, the same-protocol Base versus forced-on CP table is:

- Base wrong / CP correct: 31
- Base wrong / CP wrong: 28
- Base correct / CP wrong: 0
- Base correct / CP correct: 0

Thus the analysis-only Base-or-CP oracle upper bound is also 31/59 (52.5%); threshold routing cannot repair the 28 cases where both saved outputs are wrong. CP semantic failures were categorized as 14 substantive-error-or-unknown, eight laterality-conflict, four negation-or-polarity-conflict, and two repetition-or-truncation. The laterality and negation categories are lexical diagnostics rather than clinical ground-truth adjudications.

## Track B: exploratory scope pilot

The corrected census loaded 16,276 inventory rows but searched only the 9,835 permitted SLAKE/train rows. Before the 120-row cap there were 9,734 eligible candidates: 9,708 broad unrelated rows, 13 same-image/other-fact rows and 13 same-question/different-image conflicting-answer rows. Hard-first retention kept all 26 harder rows plus 94 broad rows.

Twelve source-question-only paraphrases passed text-equivalence review and were frozen as four fit, four calibration and four evaluation positives using disjoint rewrite families. Each role also received 20 distinct-source-image negatives. The 72 computed EqKeys were unique and role-isolated. The actual selected negatives were all broad unrelated QAs, so the evaluated claim is broad rejection only; it does not establish hard-negative or clinical-conflict routing.

| Scope metric | Original-Q control | Final-Q expert |
|---|---:|---:|
| Calibration TPR / FPR | 4/4 / 0/20 | 4/4 / 0/20 |
| Evaluation positive activation | 4/4 | 4/4 |
| Evaluation negative activation (FPR) | 2/20 (10%) | 2/20 (10%) |

On the four new evaluation paraphrases, Base was semantically correct on 0/4, while forced-on and gated CP were both correct on 4/4. The native request gated ON and remained correct. Among 20 evaluation negatives, 14 were Base-correct and all 14 were preserved. Gated output was token-identical to Base for all 20 negatives, including both false-positive activations; activated-negative semantic damage was 0/20. All 18 OFF requests had exact token parity with Base. The Base guard found no changed backbone parameters.

Input-factor fitting ran the preregistered 200 steps. Native output recovery selected step 0, meaning the historical output factors already passed the recovery criterion; no output-factor optimizer update was selected. Final-Q did not improve evaluation FPR over the original-Q threshold control in this small panel.

## Suggested reviewer questions

1. Is 31/59 T2G semantic accuracy sufficient to justify further router work, given that the saved Base is wrong on all 59 and both paths fail on 28?
2. Does the 4/4 paraphrase result warrant a larger positive panel, or is it too coupled to one image/fact and reviewed text augmentation?
3. Should the next experiment prioritize genuinely hard expert-relative negatives, since this evaluation selected only broad negatives and final-Q did not reduce the 2/20 false-positive rate?
4. Are the T1L source-supported conflict labels adequate for development diagnostics, or should a separate image-aware clinical review be required before interpreting locality harm?

Private raw questions/answers, images, token IDs, activations, EqKeys, checkpoints and Judge mappings are intentionally withheld. Public aggregate sources are `CP_DEV16_RESULTS.csv`, `CP_DEV16_REPORT.md`, `T2G_DIAGNOSTIC.md`, `SCOPE_METRICS.json`, `SCOPE_PILOT_RESULT.md`, `SCOPE_DATA_BUILD_REPORT_V2.md`, `SCOPE_MANIFEST_V2.json`, `RAW_CLOSURE.json` and `RUN_COMPLETION.json` in this directory.
