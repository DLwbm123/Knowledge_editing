# BalancEdit matched DEV16

Actual judged edits: 16/16. Frozen V4 adaptation, independent single-edit 50-step full up-projection transforms; not author sequential reproduction. Both native routed and forced-on outputs actually generated. Low scores never gate evaluation.

| Path | Role | Panel | Inputs | Edits | Source correct macro | Base-correct inputs/edits | Damage macro |
|---|---|---|---:|---:|---:|---:|---:|
| BASE | calibration | H | 96 | 7 | 10.71% | 12/3 | 0.00% |
| BASE | calibration | U | 116 | 7 | 37.90% | 44/7 | 0.00% |
| BASE | calibration | positive | 28 | 7 | 7.14% | 2/1 | 0.00% |
| BASE | challenge | same_image_other_fact_challenge | 26 | 3 | 25.00% | 9/2 | 0.00% |
| BASE | evaluation | H | 112 | 7 | 2.86% | 4/1 | 0.00% |
| BASE | evaluation | U | 112 | 7 | 39.83% | 44/7 | 0.00% |
| BASE | evaluation | positive | 28 | 7 | 0.00% | 0/0 | NA |
| BASE | fit | H | 120 | 7 | 12.14% | 13/4 | 0.00% |
| BASE | fit | U | 116 | 7 | 36.41% | 42/7 | 0.00% |
| BASE | fit | positive | 35 | 7 | 0.00% | 0/0 | NA |
| BASE | fit | scope_fit_positive_not_used_for_positive_CE | 4 | 1 | 0.00% | 0/0 | NA |
| BASE | formal_development | T1G | 62 | 16 | 0.00% | 0/0 | NA |
| BASE | formal_development | T1L | 26 | 7 | 75.71% | 19/7 | 0.00% |
| BASE | formal_development | T2G | 59 | 16 | 0.00% | 0/0 | NA |
| BASE | native | positive | 16 | 16 | 0.00% | 0/0 | NA |
| BE_FORCED_ON | calibration | H | 96 | 7 | 4.46% | 12/3 | 100.00% |
| BE_FORCED_ON | calibration | U | 116 | 7 | 0.00% | 44/7 | 100.00% |
| BE_FORCED_ON | calibration | positive | 28 | 7 | 100.00% | 2/1 | 0.00% |
| BE_FORCED_ON | challenge | same_image_other_fact_challenge | 26 | 3 | 5.56% | 9/2 | 100.00% |
| BE_FORCED_ON | evaluation | H | 112 | 7 | 10.71% | 4/1 | 0.00% |
| BE_FORCED_ON | evaluation | U | 112 | 7 | 0.95% | 44/7 | 97.62% |
| BE_FORCED_ON | evaluation | positive | 28 | 7 | 100.00% | 0/0 | NA |
| BE_FORCED_ON | fit | H | 120 | 7 | 12.14% | 13/4 | 62.50% |
| BE_FORCED_ON | fit | U | 116 | 7 | 0.00% | 42/7 | 100.00% |
| BE_FORCED_ON | fit | positive | 35 | 7 | 94.29% | 0/0 | NA |
| BE_FORCED_ON | fit | scope_fit_positive_not_used_for_positive_CE | 4 | 1 | 100.00% | 0/0 | NA |
| BE_FORCED_ON | formal_development | T1G | 62 | 16 | 93.75% | 0/0 | NA |
| BE_FORCED_ON | formal_development | T1L | 26 | 7 | 41.43% | 19/7 | 71.43% |
| BE_FORCED_ON | formal_development | T2G | 59 | 16 | 87.50% | 0/0 | NA |
| BE_FORCED_ON | native | positive | 16 | 16 | 93.75% | 0/0 | NA |
| BE_NATIVE_ROUTED | calibration | H | 96 | 7 | 11.61% | 12/3 | 16.67% |
| BE_NATIVE_ROUTED | calibration | U | 116 | 7 | 12.10% | 44/7 | 71.97% |
| BE_NATIVE_ROUTED | calibration | positive | 28 | 7 | 100.00% | 2/1 | 0.00% |
| BE_NATIVE_ROUTED | challenge | same_image_other_fact_challenge | 26 | 3 | 5.56% | 9/2 | 100.00% |
| BE_NATIVE_ROUTED | evaluation | H | 112 | 7 | 10.71% | 4/1 | 0.00% |
| BE_NATIVE_ROUTED | evaluation | U | 112 | 7 | 9.05% | 44/7 | 79.85% |
| BE_NATIVE_ROUTED | evaluation | positive | 28 | 7 | 100.00% | 0/0 | NA |
| BE_NATIVE_ROUTED | fit | H | 120 | 7 | 11.43% | 13/4 | 62.50% |
| BE_NATIVE_ROUTED | fit | U | 116 | 7 | 13.89% | 42/7 | 73.06% |
| BE_NATIVE_ROUTED | fit | positive | 35 | 7 | 94.29% | 0/0 | NA |
| BE_NATIVE_ROUTED | fit | scope_fit_positive_not_used_for_positive_CE | 4 | 1 | 100.00% | 0/0 | NA |
| BE_NATIVE_ROUTED | formal_development | T1G | 62 | 16 | 93.75% | 0/0 | NA |
| BE_NATIVE_ROUTED | formal_development | T1L | 26 | 7 | 41.43% | 19/7 | 71.43% |
| BE_NATIVE_ROUTED | formal_development | T2G | 59 | 16 | 87.50% | 0/0 | NA |
| BE_NATIVE_ROUTED | native | positive | 16 | 16 | 93.75% | 0/0 | NA |

Seven old Stage1 facts use identical native/fit/cal/evaluation/H/U/challenge rows. Other DEV16 edits use their available original T0/T1G/T2G/T1L probes. Compare Stage1 FORCED_ON only with BE_FORCED_ON; routing is a separate system axis. Original Stage1 outputs were not retrained. See Stage1 full behavior addendum and METHOD_COSTS.csv. NA is zero support, not zero damage.
