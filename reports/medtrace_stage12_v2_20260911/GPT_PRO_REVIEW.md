# Stage12 V2 focused closeout

## Conclusion and evidence boundary

H-source supervision has a positive incremental result on the frozen writer-level core objective: native correctness stays15/15, selected FIT pairs increase0/23 to23/23, and EVAL pairs increase0/19 to11/19. Both methods retain100% source-style and cross-family correctness. This supports using C_FACT as the fixed development writer, not a new algorithm, SOTA, unseen-edit confirmation or clinical validation.

The prespecified fixed-RC system does not distinguish the writers on these panels: C_FACT, C_NO_H and matched BalancEdit have the same aggregate results after returning all evaluation H/U inputs to Base. The system's zero H/U damage is rejection protection, not evidence that H supervision independently improved deployed system performance under this gate. Preserve C_FACT+BE_ROUTE+RC_FIXED_OLD16 as the preregistered candidate for separately authorized broader evaluation, without claiming it outperforms matched RC baselines here.

### Main FORCED_ON comparison, step320

|Metric (%)|C_NO_H|C_FACT|
|---|---:|---:|
|Native correctness|100.00|100.00|
|Selected FIT PairCorrect (micro)|0.00 (0/23)|100.00 (23/23)|
|EVAL PairCorrect (micro)|0.00 (0/19)|57.89 (11/19)|
|EVAL PairCorrect (edit macro)|0.00|53.57|
|H evaluation correctness (macro)|0.00|53.57|
|H base-correct damage (macro, lower better)|100.00|30.77|
|U evaluation correctness (macro)|51.59|52.22|
|U base-correct damage (macro, lower better)|27.36|25.77|
|Source-style / cross-family correctness|100.00 /100.00|100.00 /100.00|
|Same-answer challenge correctness (macro)|100.00|75.00|

EVAL PairCorrect improves57.89 percentage points pair-micro and53.57 points edit-macro. The existing edit-bootstrap95% interval for the macro difference is[32.14,75.00] points across14 pair-supported edits; patient independence is unknown. FIT pairs cover15 edits and only the23 selected training-H inputs. All-H-fit macro correctness is95.56%, not100%; it includes other legal fit rows and must not be conflated with selected FIT pairs. No exact-question pair for one edit does not remove its other evaluation rows.

Remaining costs are real:8/19 evaluation pairs still fail, U damage remains25.77%, and same-answer challenge performance falls25 points (challenge base-correct damage rises0% to30%). The positive core result is not uniform dominance or a general factual safety claim.

### Fixed single-edit deployment and rejection accounting

Common support is602 edit-input bindings across15 edits, representing239 distinct image/question pairs,61 distinct images and126 question strings. These are not independent patient counts. BalancEdit outputs matched all602 bindings; no baseline retraining was added. R0 uses the frozen native BE route, and RC only rejects its selection with kappa0.7696741135364367. Eight actual adapter ON/OFF branch replays passed; unchanged historical Base-route extraction was reused, not newly calibrated.

At fixed RC, all three writers retain100% native, source-style and cross-family correctness. H/U evaluation correctness is82.14% /64.59% (macro), with zero base-correct damage. All19 H and190 U evaluation bindings are OFF and return Base. The30 source-style and30 cross-family positives remain ON; there is no positive wrong rejection in those panels.

Relative to FORCED_ON C_FACT, fixed RC avoids4 H errors and33 U errors but loses9 U corrections (net U improvement24/190). Thus returning Base sacrifices useful edits as well as avoiding damage. All three fixed-RC systems also share the58.33% challenge result. At R0, C_FACT H correctness53.57% exceeds both C_NO_H and BE3.57%; U correctness56.21% is slightly below C_NO_H57.16% but above BE24.57%. Supervision and capacity differ from BE, so this is not an equal-training ablation.

### Execution, cost and recovery

Only15 C_NO_H trajectories were newly trained (4800 optimizer steps); C_FACT was reused. C_NO_H recorded145680 training tokens and14400 forwards versus the historical C_FACT164240 tokens and19200 forwards: H supervision adds18560 answer/predictor tokens and4800 forwards. Parameterization is identical at73728 FP32 parameters. Summed writer wall times are1959.45s versus2830.85s, but the latter includes Stage11's extra step160 generation and different parallel execution; do not attribute the full wall difference solely to H supervision. C_NO_H generation time totals349.93s versus703.44s for the two-endpoint Stage11 candidate. Counts exclude Base-teacher cache construction and Judge calls where not included by the original counter.

Training/generation and Judge processes used GPU0 for approximately33.00 and1.33 minutes respectively (process intervals, not kernel utilization). Judge1244/1244 completed, with1189 reused and55 new verdicts. Initial export failed because mixed metric/effect rows had different CSV fields; the recovery aligns the union of fields and reruns only CPU reporting. Two focused report/deployment checks passed. No training, generation or Judge was repeated. Historical failure logs remain private, and pre-release publication=PENDING in the generated status is a snapshot, not a new computation request.

Training execution source: bfd7bf7; report-only repair is recorded in subsequent research history. Public delivery includes code, aggregated tables and sanitized costs only; no QA, images, masks, raw answers/tokens, weights, activations, Judge mappings or credentials. Source-independent confirmation and scaled sequential/bank evaluation remain missing and require a separately authorized unified evaluation, not an automatic new qualification round.

{'status': 'COMPUTE_COMPLETE', 'training_completed': 15, 'planned_training': 15, 'judge_required': 1244, 'judge_missing': 0, 'judge_new': 55, 'judge_reused': 1189, 'system_replay': 'PASSED', 'publication': 'PENDING'}

Fixed candidate C_FACT + BE_ROUTE + RC_FIXED_OLD16. Viewed development only; patients UNKNOWN. No new sidecar/bank/algorithm or automatic next experiment.
C_FACT reused Stage11 step320. Only C_NO_H trained320; no step160 evaluation. Same parameterization/optimizer/native-fit-U schedule; the ablation removes H supervision and its compute, not a comparison to all equally supervised generic methods.
System rows are frozen-output derivations, with a few actual adapter branch replays; unchanged historical Base-derived routing is reused. Unsupported official M3Bench metrics remain NA, not constructed T2G.

|Writer|Pair panel|Both correct / pairs|Pair micro|Edit macro|
|---|---|---:|---:|---:|
|C_FACT|FIT_PairCorrect|23/23|1.0|1.0|
|C_FACT|EVAL_PairCorrect|11/19|0.5789473684210527|0.5357142857142857|
|C_NO_H|FIT_PairCorrect|0/23|0.0|0.0|
|C_NO_H|EVAL_PairCorrect|0/19|0.0|0.0|

|Writer|Mode|Role|Panel|Correct macro|Base-correct damage|
|---|---|---|---|---:|---:|
|C_FACT|FORCED_ON|challenge|same_answer_context_not_conflict|0.75|0.3|
|C_FACT|FORCED_ON|evaluation|H|0.5357142857142857|0.3076923076923077|
|C_FACT|FORCED_ON|evaluation|U|0.5222222222222223|0.2577056277056277|
|C_FACT|FORCED_ON|evaluation|cross_family_confirmation|1.0|None|
|C_FACT|FORCED_ON|evaluation|source_style_confirmation|1.0|None|
|C_FACT|FORCED_ON|fit|H|0.9555555555555556|0.02222222222222222|
|C_FACT|FORCED_ON|fit|U|0.49743589743589745|0.19883597883597884|
|C_FACT|FORCED_ON|fit|cross_family_confirmation|1.0|None|
|C_FACT|FORCED_ON|native|T0|1.0|None|
|C_NO_H|FORCED_ON|challenge|same_answer_context_not_conflict|1.0|0.0|
|C_NO_H|FORCED_ON|evaluation|H|0.0|1.0|
|C_NO_H|FORCED_ON|evaluation|U|0.5158730158730158|0.2735762385762386|
|C_NO_H|FORCED_ON|evaluation|cross_family_confirmation|1.0|None|
|C_NO_H|FORCED_ON|evaluation|source_style_confirmation|1.0|None|
|C_NO_H|FORCED_ON|fit|H|0.0|1.0|
|C_NO_H|FORCED_ON|fit|U|0.4868131868131868|0.2023015873015873|
|C_NO_H|FORCED_ON|fit|cross_family_confirmation|1.0|None|
|C_NO_H|FORCED_ON|native|T0|1.0|None|
|BASE|BASE|challenge|same_answer_context_not_conflict|0.5833333333333334|0.0|
|BASE|BASE|evaluation|H|0.8214285714285714|0.0|
|BASE|BASE|evaluation|U|0.645909645909646|0.0|
|BASE|BASE|evaluation|cross_family_confirmation|0.5333333333333333|None|
|BASE|BASE|evaluation|source_style_confirmation|0.6333333333333333|None|
|BASE|BASE|fit|H|0.9777777777777777|0.0|
|BASE|BASE|fit|U|0.5835164835164836|0.0|
|BASE|BASE|fit|cross_family_confirmation|0.3|None|
|BASE|BASE|native|T0|0.0|None|
|BE|BE_ROUTE_R0|challenge|same_answer_context_not_conflict|1.0|0.0|
|BE|BE_ROUTE_R0|evaluation|H|0.03571428571428571|0.9230769230769231|
|BE|BE_ROUTE_R0|evaluation|U|0.24572649572649574|0.6451827801827802|
|BE|BE_ROUTE_R0|evaluation|cross_family_confirmation|1.0|None|
|BE|BE_ROUTE_R0|evaluation|source_style_confirmation|1.0|None|
|BE|BE_ROUTE_R0|fit|H|0.04444444444444444|0.9444444444444444|
|BE|BE_ROUTE_R0|fit|U|0.17252747252747253|0.7128306878306878|
|BE|BE_ROUTE_R0|fit|cross_family_confirmation|1.0|None|
|BE|BE_ROUTE_R0|native|T0|1.0|None|
|BE|RC_FIXED_OLD16|challenge|same_answer_context_not_conflict|0.5833333333333334|0.0|
|BE|RC_FIXED_OLD16|evaluation|H|0.8214285714285714|0.0|
|BE|RC_FIXED_OLD16|evaluation|U|0.645909645909646|0.0|
|BE|RC_FIXED_OLD16|evaluation|cross_family_confirmation|1.0|None|
|BE|RC_FIXED_OLD16|evaluation|source_style_confirmation|1.0|None|
|BE|RC_FIXED_OLD16|fit|H|0.9777777777777777|0.0|
|BE|RC_FIXED_OLD16|fit|U|0.5835164835164836|0.0|
|BE|RC_FIXED_OLD16|fit|cross_family_confirmation|1.0|None|
|BE|RC_FIXED_OLD16|native|T0|1.0|None|
|C_FACT|BE_ROUTE_R0|challenge|same_answer_context_not_conflict|0.7857142857142857|0.2|
|C_FACT|BE_ROUTE_R0|evaluation|H|0.5357142857142857|0.3076923076923077|
|C_FACT|BE_ROUTE_R0|evaluation|U|0.5620879120879121|0.16807599807599807|
|C_FACT|BE_ROUTE_R0|evaluation|cross_family_confirmation|1.0|None|
|C_FACT|BE_ROUTE_R0|evaluation|source_style_confirmation|1.0|None|
|C_FACT|BE_ROUTE_R0|fit|H|0.9777777777777777|0.0|
|C_FACT|BE_ROUTE_R0|fit|U|0.5212454212454213|0.13820105820105819|
|C_FACT|BE_ROUTE_R0|fit|cross_family_confirmation|1.0|None|
|C_FACT|BE_ROUTE_R0|native|T0|1.0|None|
|C_FACT|RC_FIXED_OLD16|challenge|same_answer_context_not_conflict|0.5833333333333334|0.0|
|C_FACT|RC_FIXED_OLD16|evaluation|H|0.8214285714285714|0.0|
|C_FACT|RC_FIXED_OLD16|evaluation|U|0.645909645909646|0.0|
|C_FACT|RC_FIXED_OLD16|evaluation|cross_family_confirmation|1.0|None|
|C_FACT|RC_FIXED_OLD16|evaluation|source_style_confirmation|1.0|None|
|C_FACT|RC_FIXED_OLD16|fit|H|0.9777777777777777|0.0|
|C_FACT|RC_FIXED_OLD16|fit|U|0.5835164835164836|0.0|
|C_FACT|RC_FIXED_OLD16|fit|cross_family_confirmation|1.0|None|
|C_FACT|RC_FIXED_OLD16|native|T0|1.0|None|
|C_NO_H|BE_ROUTE_R0|challenge|same_answer_context_not_conflict|1.0|0.0|
|C_NO_H|BE_ROUTE_R0|evaluation|H|0.03571428571428571|0.9230769230769231|
|C_NO_H|BE_ROUTE_R0|evaluation|U|0.5715506715506715|0.158020683020683|
|C_NO_H|BE_ROUTE_R0|evaluation|cross_family_confirmation|1.0|None|
|C_NO_H|BE_ROUTE_R0|evaluation|source_style_confirmation|1.0|None|
|C_NO_H|BE_ROUTE_R0|fit|H|0.04444444444444444|0.9444444444444444|
|C_NO_H|BE_ROUTE_R0|fit|U|0.5106227106227106|0.1371164021164021|
|C_NO_H|BE_ROUTE_R0|fit|cross_family_confirmation|1.0|None|
|C_NO_H|BE_ROUTE_R0|native|T0|1.0|None|
|C_NO_H|RC_FIXED_OLD16|challenge|same_answer_context_not_conflict|0.5833333333333334|0.0|
|C_NO_H|RC_FIXED_OLD16|evaluation|H|0.8214285714285714|0.0|
|C_NO_H|RC_FIXED_OLD16|evaluation|U|0.645909645909646|0.0|
|C_NO_H|RC_FIXED_OLD16|evaluation|cross_family_confirmation|1.0|None|
|C_NO_H|RC_FIXED_OLD16|evaluation|source_style_confirmation|1.0|None|
|C_NO_H|RC_FIXED_OLD16|fit|H|0.9777777777777777|0.0|
|C_NO_H|RC_FIXED_OLD16|fit|U|0.5835164835164836|0.0|
|C_NO_H|RC_FIXED_OLD16|fit|cross_family_confirmation|1.0|None|
|C_NO_H|RC_FIXED_OLD16|native|T0|1.0|None|

Interpretation is stated in the closeout above. Zero routed damage comes from returning Base on H/U, not demonstrated writer protection. No new-source or scale validation claimed.
