# BELoRA one-edit effect gate

Status: **PASS** for this paper-spec independent reimplementation (the author implementation is unavailable). Target NLL decreased from 6.629343 to 6.375094; adapter gradient and state delta were positive. Self-routing, prefill/decode adapter lifetime, nonempty generation, save/reload parity, miss/base parity, and frozen-base integrity passed. The greedy output did not change on this one record, so broader smoke testing remains required.

The preserved first gate report failed only because the checker counted the expected no-adapter base route-key forward as a generation forward. A new non-overwriting attempt passed after correcting that checker condition.
