# Positive execution retention and forced-on diagnosis

Status: `R2_ROUTER_CALIBRATION_FIXED__END_TO_END_NOT_SUPPORTED_OVER_R0_ON_CURRENT_DEV`

The private mapping independently reproduced the review counts. Native replay and four evaluation paraphrases remain separate in `FIXED_E2E_METRICS.csv`; repeated seeds are not new facts or patients.

| profile | old ON | fixed ON | forced correct | old gated correct | fixed gated correct | decision changes |
|---|---:|---:|---:|---:|---:|---:|
| R0 | 144/180 | 144/180 | 180/180 | 144/180 | 144/180 | 0 |
| ALT R2 | 157/180 | 169/180 | 126/180 | 122/180 | 125/180 | 14 |

For ALT across the fixed native plus fit-paraphrase diagnostic inputs, mean target-content NLL was 0.009068 before scope Q, 1.866056 after Q with the old P/rho, and 0.491436 after the fixed 80-step output recovery. Rank-one rates were 100.0%, 6.1%, and 57.2%. Thus threshold repair cannot recover forced-on failures; the matched-state diagnosis records the Q-direction change separately from incomplete output recovery.

Actual deterministic replay covered 7 fixed requests spanning ON, OFF, native, new paraphrase and hard-negative paths; every replay matched the frozen full token sequence and answer. All other corrected gates are explicitly derived from the frozen base/forced outputs.
