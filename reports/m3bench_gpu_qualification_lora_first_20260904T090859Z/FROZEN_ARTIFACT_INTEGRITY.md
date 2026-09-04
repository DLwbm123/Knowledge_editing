# Frozen artifact integrity

- The CPU gate package and its frozen files were read-only inputs and were not modified.
- The exact official source checkout remained clean.
- The no-edit, QUAL8, DEV16, QUAL16, and sequential16 selections were frozen before any corresponding model output.
- The first failed model-load attempts produced no canary rows and are preserved in execution history.
- The successful 414-row outputs are private, append-only run evidence; public reports contain only aggregate counts and checksums.
- No historical raw prediction, verdict, report, sequence, image, or model checkpoint was modified.
- No QUAL8 run, LoRA calibration, sequential16 run, formal run, Judge, or evaluator was started after the parity hard stop.

Testing:

- M3Bench-focused tests: 98 passed.
- Full repository collection was attempted but stopped on seven pre-existing environment/import issues outside this gate (legacy Lightning API, missing TensorBoard, and unavailable Engram-v2 modules). No M3Bench-focused regression was observed.
