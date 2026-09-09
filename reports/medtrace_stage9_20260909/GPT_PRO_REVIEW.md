# Stage9 fact-contrast development results

## Verified closeout interpretation

Primary training/generation/Judge completed for the supported subset:15 SLAKE edit tasks x3 conditions =45 trajectories, each160 additional optimizer steps (7200 total). Eight planned edits/24 trajectories remain uniformly unsupported, not failed runs:old7 lack sufficient direct H-source binding and SLAKE edit115 lacks a pair covered by the explicit relation check. No old T2G training/evaluation comparison is available for the new methods. This is not completion of all23 planned edits.

Wall time2838.91 seconds (47.32 minutes); accounted GPU time1.3031 hours. All pipeline exit codes zero. Judge1620/1620,1343 reused,277 new,0 missing. Preparation/training implementation ecf3921; actual launch checkout1a955e4fefe168dcc349e8f91e6b5ca6f3485fc5. Publication identity is the containing Git commit. The PENDING fields below preserve the original pre-publication snapshot.

### Shared-support FORCED_ON results (edit-macro percentages)

| Metric | F0 positive continuation | F1 source CE + weak U-KL | F2 plus bidirectional contrast |
|---|---:|---:|---:|
| Native correctness (in-sample) |100.00%|73.33%|73.33%|
| Source-style correctness |100.00%|100.00%|100.00%|
| Cross-family correctness |100.00%|76.67%|76.67%|
| H-fit correctness (all original fit H, not only selected pairs) |0.00%|73.33%|73.33%|
| H-evaluation correctness |0.00%|42.86%|42.86%|
| H-evaluation Base-correct damage |100.00%|46.15%|46.15%|
| U-evaluation correctness |32.45%|51.85%|52.36%|
| U-evaluation Base-correct damage |58.74%|20.41%|19.67%|

H evaluation support:19 inputs,14 edit tasks,7 source images; U190 inputs,15 tasks,41 images. Source-style and cross-family each30 inputs/15 edits/15 images. Repeated source images are not independent patients; patient remains UNKNOWN. Historical W0/APR/B2/W1 in this report are recomputed aggregates on these15 tasks, not directly comparable to the full16-task Stage8 percentages.

**Decision: unresolved protection/editing tradeoff; no demonstrated added H benefit from contrast.** F1 versus F0 improves held-out-role H correctness, so the source-supervision combination did more than just improve H-fit. But H-fit73.33% versus H-evaluation42.86% shows incomplete transfer, and native/cross-family degradation remains. F1 also adds U-KL, so F1-F0 does not isolate H CE alone. F2 versus F1 leaves H, native, source-style and cross-family correctness unchanged. U correctness improves only0.51 percentage points (edit-bootstrap95% interval[0,1.54] points; image-cluster[0,0.91]); one additional avoided U damage does not establish a strong contrast mechanism advantage. Accept F1 as the simpler explanatory baseline rather than attributing its supervision benefit to F2. Neither is a solved default editor. No F3, new rank, new data or next-stage experiment was started.

The matched fit audit selected23 H task-pair exposures across15 edits, identical for F1/F2; these are not23 independent facts/images. F0 uses the same eligible cohort/positive schedule but does not consume H answers in its objective. Selection never uses the new outputs. Historical sources and all raw QA remain private; source verification is not human clinical signoff.

### Costs of preservation, relative to historical W0 on the same inputs

Both F1/F2 avoid9 H damages with no gained Base-wrong corrections. On U, F1 avoids31 damages but introduces2 and loses9 existing corrections; F2 avoids32, introduces2 and loses9. Neither gains a new Base-wrong U correction relative to W0. Relative to F0, both lose11 U corrections and gain1 while avoiding50/51 damages and introducing3. These are paired edit-input counts, not independent patient outcomes. See CORRECTION_TRADEOFF_COUNTS.csv.

### Remaining reporting limits

Primary natural-generation scoring is complete, but the optional/auxiliary teacher-forced H-evaluation ranking and same-image/question-change score diagnostics were not implemented in this run. No causal or target-propagation claim is made from their absence; Stage8 NA fields are not backfilled. Training curves count forward calls but do not yet provide an explicit processed-token total. These are reporting/protocol coverage gaps, not evidence of training failure or authorization to rerun successful endpoints. Full independent fact-group deduplication beyond the published edit/input/image support is not established. Do not describe this release as exhaustive completion of every auxiliary protocol item.

{'status': 'COMPUTE_COMPLETE_SUPPORTED_SUBSET', 'planned_trajectories': 69, 'complete_trajectories': 45, 'unsupported_trajectories': 24, 'judge_required': 1620, 'judge_scored': 1620, 'judge_missing': 0, 'judge_new': 277, 'judge_reused': 1343, 'publication': 'PENDING'}

New H source-answer supervision is shared by F1/F2; only F2-F1 isolates contrast. Historical W0/APR/B2/W1 are not equal-training controls. Unsupported old tasks are not replaced. Viewed development only; patient UNKNOWN.
No automatic F3 or new facts. See fit-source manifest for permission/provenance limits. No clinical safety, independent confirmation or originality claim.

|Cohort|Method|Role|Panel|Correct macro|Damage macro|
|---|---|---|---|---:|---:|
|FACT_SLAKE_STAGE2_BANK16|B0|challenge|same_answer_context_not_conflict|1.0|0.0|
|FACT_SLAKE_STAGE2_BANK16|B0|evaluation|H|0.0|1.0|
|FACT_SLAKE_STAGE2_BANK16|B0|evaluation|U|0.41501831501831504|0.43718133718133717|
|FACT_SLAKE_STAGE2_BANK16|B0|evaluation|cross_family_confirmation|1.0|None|
|FACT_SLAKE_STAGE2_BANK16|B0|evaluation|source_style_confirmation|1.0|None|
|FACT_SLAKE_STAGE2_BANK16|B0|fit|H|0.0|1.0|
|FACT_SLAKE_STAGE2_BANK16|B0|fit|U|0.3567765567765568|0.4728042328042328|
|FACT_SLAKE_STAGE2_BANK16|B0|fit|cross_family_confirmation|1.0|None|
|FACT_SLAKE_STAGE2_BANK16|B0|native|T0|1.0|None|
|FACT_SLAKE_STAGE2_BANK16|B1|challenge|same_answer_context_not_conflict|1.0|0.0|
|FACT_SLAKE_STAGE2_BANK16|B1|evaluation|H|0.35714285714285715|0.5384615384615384|
|FACT_SLAKE_STAGE2_BANK16|B1|evaluation|U|0.5523199023199024|0.18804713804713805|
|FACT_SLAKE_STAGE2_BANK16|B1|evaluation|cross_family_confirmation|1.0|None|
|FACT_SLAKE_STAGE2_BANK16|B1|evaluation|source_style_confirmation|0.9666666666666667|None|
|FACT_SLAKE_STAGE2_BANK16|B1|fit|H|0.5333333333333333|0.4666666666666667|
|FACT_SLAKE_STAGE2_BANK16|B1|fit|U|0.5212454212454213|0.1901058201058201|
|FACT_SLAKE_STAGE2_BANK16|B1|fit|cross_family_confirmation|1.0|None|
|FACT_SLAKE_STAGE2_BANK16|B1|native|T0|1.0|None|
|FACT_SLAKE_STAGE2_BANK16|B2|challenge|same_answer_context_not_conflict|1.0|0.0|
|FACT_SLAKE_STAGE2_BANK16|B2|evaluation|H|0.0|1.0|
|FACT_SLAKE_STAGE2_BANK16|B2|evaluation|U|0.49371184371184373|0.3228427128427128|
|FACT_SLAKE_STAGE2_BANK16|B2|evaluation|cross_family_confirmation|1.0|None|
|FACT_SLAKE_STAGE2_BANK16|B2|evaluation|source_style_confirmation|1.0|None|
|FACT_SLAKE_STAGE2_BANK16|B2|fit|H|0.0|1.0|
|FACT_SLAKE_STAGE2_BANK16|B2|fit|U|0.441025641025641|0.31582010582010583|
|FACT_SLAKE_STAGE2_BANK16|B2|fit|cross_family_confirmation|1.0|None|
|FACT_SLAKE_STAGE2_BANK16|B2|native|T0|1.0|None|
|FACT_SLAKE_STAGE2_BANK16|F0|challenge|same_answer_context_not_conflict|1.0|0.0|
|FACT_SLAKE_STAGE2_BANK16|F0|evaluation|H|0.0|1.0|
|FACT_SLAKE_STAGE2_BANK16|F0|evaluation|U|0.32454212454212455|0.5874458874458874|
|FACT_SLAKE_STAGE2_BANK16|F0|evaluation|cross_family_confirmation|1.0|None|
|FACT_SLAKE_STAGE2_BANK16|F0|evaluation|source_style_confirmation|1.0|None|
|FACT_SLAKE_STAGE2_BANK16|F0|fit|H|0.0|1.0|
|FACT_SLAKE_STAGE2_BANK16|F0|fit|U|0.2816849816849817|0.6177513227513227|
|FACT_SLAKE_STAGE2_BANK16|F0|fit|cross_family_confirmation|1.0|None|
|FACT_SLAKE_STAGE2_BANK16|F0|native|T0|1.0|None|
|FACT_SLAKE_STAGE2_BANK16|F1|challenge|same_answer_context_not_conflict|0.7857142857142857|0.0|
|FACT_SLAKE_STAGE2_BANK16|F1|evaluation|H|0.42857142857142855|0.46153846153846156|
|FACT_SLAKE_STAGE2_BANK16|F1|evaluation|U|0.5184981684981685|0.2041053391053391|
|FACT_SLAKE_STAGE2_BANK16|F1|evaluation|cross_family_confirmation|0.7666666666666667|None|
|FACT_SLAKE_STAGE2_BANK16|F1|evaluation|source_style_confirmation|1.0|None|
|FACT_SLAKE_STAGE2_BANK16|F1|fit|H|0.7333333333333333|0.25555555555555554|
|FACT_SLAKE_STAGE2_BANK16|F1|fit|U|0.4249084249084249|0.2735978835978836|
|FACT_SLAKE_STAGE2_BANK16|F1|fit|cross_family_confirmation|0.8666666666666667|None|
|FACT_SLAKE_STAGE2_BANK16|F1|native|T0|0.7333333333333333|None|
|FACT_SLAKE_STAGE2_BANK16|F2|challenge|same_answer_context_not_conflict|0.7857142857142857|0.0|
|FACT_SLAKE_STAGE2_BANK16|F2|evaluation|H|0.42857142857142855|0.46153846153846156|
|FACT_SLAKE_STAGE2_BANK16|F2|evaluation|U|0.5236263736263737|0.1966979316979317|
|FACT_SLAKE_STAGE2_BANK16|F2|evaluation|cross_family_confirmation|0.7666666666666667|None|
|FACT_SLAKE_STAGE2_BANK16|F2|evaluation|source_style_confirmation|1.0|None|
|FACT_SLAKE_STAGE2_BANK16|F2|fit|H|0.7333333333333333|0.25555555555555554|
|FACT_SLAKE_STAGE2_BANK16|F2|fit|U|0.4249084249084249|0.28026455026455027|
|FACT_SLAKE_STAGE2_BANK16|F2|fit|cross_family_confirmation|0.8666666666666667|None|
|FACT_SLAKE_STAGE2_BANK16|F2|native|T0|0.7333333333333333|None|
|FACT_SLAKE_STAGE2_BANK16|HISTORICAL_W1|challenge|same_answer_context_not_conflict|1.0|0.0|
|FACT_SLAKE_STAGE2_BANK16|HISTORICAL_W1|evaluation|H|0.32142857142857145|0.5384615384615384|
|FACT_SLAKE_STAGE2_BANK16|HISTORICAL_W1|evaluation|U|0.5997557997557997|0.13659451659451657|
|FACT_SLAKE_STAGE2_BANK16|HISTORICAL_W1|evaluation|cross_family_confirmation|1.0|None|
|FACT_SLAKE_STAGE2_BANK16|HISTORICAL_W1|evaluation|source_style_confirmation|1.0|None|
|FACT_SLAKE_STAGE2_BANK16|HISTORICAL_W1|fit|H|0.4111111111111111|0.5666666666666667|
|FACT_SLAKE_STAGE2_BANK16|HISTORICAL_W1|fit|U|0.5065934065934066|0.15904761904761905|
|FACT_SLAKE_STAGE2_BANK16|HISTORICAL_W1|fit|cross_family_confirmation|1.0|None|
|FACT_SLAKE_STAGE2_BANK16|HISTORICAL_W1|native|T0|1.0|None|
