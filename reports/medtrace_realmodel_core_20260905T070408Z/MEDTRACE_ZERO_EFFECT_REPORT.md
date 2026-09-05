# MedTRACE real-model zero-effect report

Status: `MEDTRACE_ZERO_EFFECT_PASS`

- Runtime: frozen V4 LLaVA-Med v1.5 Mistral 7B, original backbone only.
- Development migration: opaque record `core9q-603881e73eba9ac4e23261b7`, selected as the first row of the frozen DEV16 manifest without consulting LoRA outcomes.
- Scope: one primary plus seven existing probes, eight unique queries total.
- Hook: `model.layers.21.mlp.down_proj`, real dimensions 14336 to 4096; CP factor shapes 112x128 and 64x64.
- Frozen baseline: prompt token IDs, generated token IDs, and decoded text matched the existing V4 base predictions for all 8/8 queries.
- Four-state identity: base, attached-disabled, attached-active with zero `rho`, and detached/reset were exactly identical for all 8/8 queries.
- Generation paths: manual no-cache, manual cached, and Hugging Face cached generation were token-identical on the primary and paired canary. Both ended normally with EOS at token index 19 (20 generated tokens), below the 128-token cap.
- Predictor semantics: the final prompt position is active for the first answer token; each one-token cached decode forward uses a one-position active mask.
- Base guard: all 686 sampled frozen parameter tensors were unchanged; no base parameter required gradients.

Private raw prompts, images, token sequences, decoded answers, and per-query records remain on the experiment server. This report contains only aggregate and opaque evidence.
