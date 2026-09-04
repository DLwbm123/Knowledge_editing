# Canonical no-edit parity report

Status: `M3BENCH_GPU_GATE_BLOCKED__NO_EDIT_PARITY_MISMATCH__RUNTIME`

The canary was frozen before GPU output. It contains 414 unique queries after applying the required T0/T1L/T3L/T3G/T4L full selections, 64 occurrence selections for T1G/T2L/T2G/T4G, and final query-ID deduplication.

| Check | Passed | Total |
|---|---:|---:|
| Fresh official vs formal raw token IDs | 414 | 414 |
| Fresh official vs formal prompt token IDs | 414 | 414 |
| Fresh official vs formal decoded text | 414 | 414 |
| Fresh official vs formal normalized text | 414 | 414 |
| Image SHA binding | 414 | 414 |
| Empty/error-free | 414 | 414 |
| Frozen canonical vs fresh official decoded text | 402 | 414 |
| Frozen canonical vs formal decoded text | 402 | 414 |
| Frozen canonical vs fresh normalized text | 402 | 414 |
| Frozen semantic verdict preserved through exact raw equality | 402 | 414 |

All 12 mismatches are confined to historical frozen output versus both fresh paths. The two fresh paths agree exactly on every checked field. Nine mismatches are from SLAKE and three from VQA-RAD; none is empty, errored, or at the 1,024-token ceiling. Only one of the 12 has the same generated-token count as its frozen row.

The frozen canonical base rows do not contain raw or prompt token IDs. Consequently, historical token-ID equality cannot be proven directly. The run used the exact fresh official path as the token-ID oracle and the frozen base as the decoded/normalized anchor; this limitation does not weaken the hard stop because 12 decoded outputs already differ.

The frozen base provenance records PyTorch 2.1.2+cu121 and Transformers 4.36.2, while this qualification used PyTorch 2.6.0+cu124 and Transformers 4.51.3. This stack drift is the leading explanation, not yet a proven root cause. Per protocol, no editor qualification or training was started.

Artifacts retained privately:

- Manifest SHA-256: `8c39832036259f265f5d5bd672cec55d68283674ec957ecf9eb2b312c2e1e66a`
- Private input SHA-256: `f62a00be757d65ca12d160f3a9f18316eb9f8a0169b20229c926b54b3403aaff`
- Official output SHA-256: `323e2a92e208f89e329e4a5992c7a61b307a38344c573c04c06288ad64927105`
- Formal output SHA-256: `624f99f78ae4ba77864bda06da57d2f2ab63ad68d7889e2d1ca434a6b17dcd1d`
- Private machine report SHA-256: `b4d03512e886205d887cf6b1480e70b9e9c96afbad537b00c5c75a519fe2b656`

No model answers, questions, references, images, local paths, or private input rows are included in this public report.
