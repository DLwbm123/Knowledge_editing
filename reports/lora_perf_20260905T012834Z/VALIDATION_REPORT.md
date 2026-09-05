# LoRA-Perf-v1 validation

Status: `QUAL_VALIDATION_FAIL`

DEV selected exactly one configuration: rank 16, alpha 16, all LM MLP gate/up/down projections, learning rate 5e-4, checkpoint 80. The one permitted QUAL16 run completed 16/16 events with finite gradients and unchanged base parameters; peak allocated/reserved GPU memory was 19.57/20.18 GiB and total event runtime was 511.2 seconds.

The fixed Judge completed 255/255 outputs with 255/255 strict-schema verdicts. Results were T0 16/16, T1L macro 0.2000 over 27 probes from 7 eligible edits, T1G 1.0000 over 62 probes, T2G 1.0000 over 61 probes, target NLL decrease 16/16, empty/error 0, and same-question other-image edit-target copy rate 1.0000 over 89 checks. Because T1L is below the frozen 0.25 minimum, validation failed. No threshold was changed and QUAL will not be used for another selection round.
