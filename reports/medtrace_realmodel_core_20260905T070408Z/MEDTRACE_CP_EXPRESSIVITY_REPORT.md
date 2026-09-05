# MedTRACE forced-on CP expressivity report

Status: `MEDTRACE_CP_EXPRESSIVITY_PASS`

This is a single-edit forced-on capacity result, not intrinsic-scope, sequential, or full-benchmark evidence.

## Frozen setup

- Backbone: original frozen V4 LLaVA-Med v1.5 Mistral 7B; no LoRA weights or optimizer state loaded.
- Development record: opaque ID `core9q-603881e73eba9ac4e23261b7`, the first frozen DEV16 row, selected independently of LoRA results.
- Expert: one asymmetric CP residual at `model.layers.21.mlp.down_proj`, 14336 to 4096, rank 4, 1,476 trainable parameters.
- Optimizer: AdamW, learning rate 0.001, weight decay 0, gradient clipping 1.0.
- Budget: at most 200 steps; greedy generation every 20 steps; at most 128 new tokens.
- Training signal: answer-only target loss. Formal locality probes were not used for training.

## Capacity result

Rank 4 first met the pre-registered candidate conditions at step 140, so rank 8, rank 16, and two-layer rank 8 were not run.

| Step | Answer NLL | Content NLL | EOS/tail NLL | First-token rank | Literal normalized target |
|---:|---:|---:|---:|---:|:---:|
| 0 | 2.8179 | 3.0997 | 0.000507 | 104 | no |
| 20 | 2.8161 | 3.0977 | 0.000505 | 104 | no |
| 40 | 2.7898 | 3.0687 | 0.000502 | 99 | no |
| 60 | 2.6666 | 2.9333 | 0.000500 | 87 | no |
| 80 | 2.1998 | 2.4197 | 0.000466 | 49 | no |
| 100 | 1.0306 | 1.1336 | 0.000359 | 5 | no |
| 120 | 0.2107 | 0.2317 | 0.000822 | 1 | no |
| 140 | 0.0605 | 0.0665 | 0.000547 | 1 | yes |

The selected output was also accepted by the frozen method-blind Qwen3-32B-AWQ Judge (`is_correct=true`, strict JSON parsed). The Judge snapshot was `0499c3ac83fdef8810b907a23894ba91e95eddd8`; prompt and configuration hashes matched the V4 Judge lock.

## Engineering gates

- Unrestricted native generation: semantic-correct and literal normalized target match.
- Short-answer generation: literal normalized target match.
- Manual no-cache, manual cached, and Hugging Face cached generation: token-identical for unrestricted and short-answer prompts; 11 tokens in every path.
- First answer token rank: 1.
- Save/reload: exact generation replay for both prompt forms.
- Disable/rollback: exact frozen S0 restored; short prompt disable and detach were identical.
- Factorized versus dense real-activation maximum absolute error: `1.7881393432617188e-07`.
- Base guard: all 686 sampled frozen parameter tensors unchanged; no base parameter was trainable.
- Peak allocated GPU memory during training: 16,566,680,064 bytes (15.43 GiB).
- Measured training-loop runtime: 23.62 seconds; checkpoint loading and final external Judge loading are excluded.

An initial short-prompt diagnostic found a genuine no-cache/cached mismatch: full-sequence no-cache recomputation applied the residual only at the newest token while cached generation retained prior edited assistant states. Commit `25804bde310a44c5a893b78736ad59ca086d30d3` fixed the shared hook to preserve the complete assistant predictor span. The final repeated zero-effect and engineering gates passed after this fix; the failed diagnostic remains private as provenance.

## Artifact provenance

- Training code commit: `737fd466888c2efcdf99a79f45561a6b12fe5ee5`.
- Final verification code commit: `25804bde310a44c5a893b78736ad59ca086d30d3`.
- Private training result SHA-256: `a215a92a2df38a7f4fbca1afe84a99a444f17d3e908b217dfa9315839847667d`.
- Private trajectory SHA-256: `1c12c86a285057ac6308ac2feb873774a100af54da9dfdb123d509756f94e33f`.
- Private Judge output SHA-256: `1a2db3bf37f9ac77c5c51b2d56fcd2cc65fe2d67baa29c2e3f9aab7fecb6e8ee`.
- Final private engineering verification SHA-256: `11bd4da0b2f8bcb16121da0758bac4634b22629bb7a74391218b385f8a6ffbbc`.

Raw prompts, answers, token sequences, images, checkpoint tensors, and optimizer/runtime artifacts remain private on the experiment server.

## Next-stage status

`MEDTRACE_INTRINSIC_SCOPE_NOT_RUN__CALIBRATION_CONTRACT_INCOMPLETE`

The available event exposes only seven probes and does not provide the source-image, fact, and equivalence-group attribution required to prove calibration/evaluation separation under V0.2. Intrinsic scope was therefore not trained or evaluated. This does not invalidate the zero-effect or forced-on CP result.
