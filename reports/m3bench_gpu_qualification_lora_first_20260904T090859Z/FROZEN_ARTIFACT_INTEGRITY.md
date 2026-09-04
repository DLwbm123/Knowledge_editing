# Frozen artifact integrity

- The CPU gate package and its frozen files were read-only inputs and were not modified.
- The exact official source checkout remained clean.
- The no-edit, QUAL8, DEV16, QUAL16, and sequential16 selections were frozen before any corresponding model output.
- The first failed model-load attempts produced no canary rows and are preserved in execution history.
- The successful 414-row outputs are private, append-only run evidence; public reports contain only aggregate counts and checksums.
- No historical raw prediction, verdict, report, sequence, image, or model checkpoint was modified.
- No QUAL8 run, LoRA calibration, sequential16 run, formal run, Judge, or evaluator was started after the parity hard stop.

Later-stage manifests were frozen before output but remain unexecuted:

- QUAL8: `d4ef0feb247852b641c3c4cf7f83a36bfa34d7f5c23f60ad72e39f0f9f8c74dd`
- DEV16: `3527e2f6157df1597eecf23ab08cd155389bd46920352c9f1f72e4709fc053a7`
- QUAL16: `0c43ee74e68e84b3b1d6771ee1ffb21d28b2189ea07f1ef28cf7653fbacb2586`
- Sequential16: `1c5a1401df5ab7ab079dee9806bb1894a11c0cf19621293ec9526da60d580fb5`

The 514 identity-fallback event inventory is T2L 318, T3L 1, T3G 9, T4L 76, and T4G 110. It was not evaluated because G1 failed first.

Testing:

- M3Bench-focused tests: 98 passed.
- Full repository collection was attempted but stopped on seven pre-existing environment/import issues outside this gate (legacy Lightning API, missing TensorBoard, and unavailable Engram-v2 modules). No M3Bench-focused regression was observed.
