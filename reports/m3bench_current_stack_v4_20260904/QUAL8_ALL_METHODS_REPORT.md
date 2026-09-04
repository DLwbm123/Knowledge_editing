# Current-stack V4 QUAL8

The fixed method-blind semantic Judge completed 32/32 opaque records with no
parse failures. This report contains aggregate counts only.

| Method | Integration | Effect active | Semantic-qualified | Semantic target success |
| --- | --- | --- | --- | --- |
| LoRA paper-spec reference | PASS | PASS | FAIL | 3/8 |
| GRACE | PASS | PASS | FAIL | 2/8 |
| BalanceEdit | PASS | PASS | PASS | 8/8 |
| BELoRA | PASS | PASS | PASS | 7/8 |

All methods had zero empty outputs. The LoRA result does not meet the
semantic-qualification threshold and therefore does not authorize calibration
or formal LoRA execution under the frozen LoRA-first gate.

Raw answers, questions, targets, images, opaque mappings, and private paths
are excluded from this public report.
