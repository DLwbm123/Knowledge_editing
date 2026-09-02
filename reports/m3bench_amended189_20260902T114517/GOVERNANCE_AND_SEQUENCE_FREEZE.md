# Exclusion-Only-189 Governance and Sequence Freeze

Status: `PASS`.

The approved amendment retains the 189 targets marked valid by the frozen authoritative image-aware review and excludes 10 confirmed-invalid targets plus 1 unresolved target. No replacement was selected. Reviewer B remains optional and unexecuted; this is not dual-reader validation or clinical consensus.

| Item | Result |
|---|---:|
| Original targets | 200 |
| Retained targets | 189 |
| Confirmed-invalid exclusions | 10 |
| Unresolved exclusions | 1 |
| Replacement targets | 0 |
| Amended positions | dense 1–189 |
| Relative retained order | preserved |
| Method outputs used for selection | no |
| Image bindings checked | 189/189 |

The frozen amended input SHA-256 is `ad5972dc600e7a8539d15e4573278ea7de2551a0aedc9702d4503280eacbf8ee`. The original 200-target sequence SHA-256 is `9feb8da5b7a7b8c36f6bdd506c6ba4a1490ba27e0d6af4c4f6b949a1cd89aeec`.

The formal runner is locked to 189 records with checkpoints at 1, 50, 100, and 189. Its no-model preflight passed all eight checks, including dense positions, unique record IDs, complete T0 binding, and unchanged method sources. The editor-paper-spec test suite passed 27/27 tests.
