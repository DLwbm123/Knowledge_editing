# Actual editor source binding

Status: `SOURCE_BINDING_BEFORE_EDITOR_EFFECT_REPAIR`

The formal runner was traced to source commit
`9c38d467268fb455a5ede7601aa87408cb26b264`. The executable path is:

`scripts/editor_paperspec_formal.py` -> `m3bench_repro.editors.llava_runtime`
and `m3bench_repro.editors.methods` -> `routed_layers` / `routing`.

This branch contains a public source snapshot of that runtime. Private absolute
paths were replaced by environment-variable inputs before publication; no
editor algorithm was changed in this binding commit. Original source blob
digests and the complete import-chain inventory are recorded in the adjacent
JSON report.

The snapshot is the baseline for pre-patch traces. Old raw outputs remain
historical inputs and are not modified or relabeled.
