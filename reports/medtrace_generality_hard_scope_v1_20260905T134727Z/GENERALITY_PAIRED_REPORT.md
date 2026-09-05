# DEV16 paired generality ablation

Status: `GENERALITY_PAIRED_EVALUATION_COMPLETE`

| Condition | Task | Valid edits | n | Semantic micro | Semantic macro | Exact micro |
|---|---|---:|---:|---:|---:|---:|
| CP_NATIVE_ORIGINAL | T0 | 16 | 16 | 15/16 (93.8%) | 93.8% | 100.0% |
| CP_NATIVE_ORIGINAL | T1L | 7 | 26 | 8/26 (30.8%) | 41.4% | 30.8% |
| CP_NATIVE_ORIGINAL | T1G | 16 | 62 | 58/62 (93.5%) | 93.8% | 100.0% |
| CP_NATIVE_ORIGINAL | T2G | 16 | 59 | 32/59 (54.2%) | 53.6% | 37.3% |
| CP_NATIVE_CONTINUE_80 | T0 | 16 | 16 | 15/16 (93.8%) | 93.8% | 100.0% |
| CP_NATIVE_CONTINUE_80 | T1L | 7 | 26 | 8/26 (30.8%) | 41.4% | 30.8% |
| CP_NATIVE_CONTINUE_80 | T1G | 16 | 62 | 58/62 (93.5%) | 93.8% | 100.0% |
| CP_NATIVE_CONTINUE_80 | T2G | 16 | 59 | 37/59 (62.7%) | 63.0% | 61.0% |
| CP_NATIVE_PLUS_PARAPHRASE_80 | T0 | 16 | 16 | 15/16 (93.8%) | 93.8% | 100.0% |
| CP_NATIVE_PLUS_PARAPHRASE_80 | T1L | 7 | 26 | 8/26 (30.8%) | 41.4% | 30.8% |
| CP_NATIVE_PLUS_PARAPHRASE_80 | T1G | 16 | 62 | 57/62 (91.9%) | 92.2% | 100.0% |
| CP_NATIVE_PLUS_PARAPHRASE_80 | T2G | 16 | 59 | 44/59 (74.6%) | 75.5% | 79.7% |

A1 and A2 both start from the same per-edit A0 checkpoint and execute 80 optimizer steps and 160 batch-size-1 micro-forwards per edit. A1 uses native/native with 0.5 weights; A2 uses native/rotating-paraphrase with 0.5 weights. Token/FLOPs proxies are reported because question lengths differ.

A2 minus A1 T2G macro: +0.1250; T1G macro: -0.0156; A2 T0 correct-count loss versus A1: 0. Registered retain decision: `True`.

A1 retained 25/32 old T2G successes and recovered 12/27 old failures. A2 retained 28/32 and recovered 16/27.

The original probes have already been viewed and are reused only as a development ablation panel, not an unseen confirmation set.
