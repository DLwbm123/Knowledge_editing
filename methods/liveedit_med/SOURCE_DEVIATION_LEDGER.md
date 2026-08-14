# Source Deviation Ledger

Permitted intentional deviations from pinned LiveEdit:

1. The model adapter targets the project's LLaVA-Med Mistral implementation and
   the full `model.layers.21` decoder-block output.
2. Natural generation is target-free and validates manual no-cache, cached, and
   Hugging Face greedy token parity.
3. An empty visual candidate set returns exact clean S0 through
   `EMPTY_CANDIDATE_BASE_BYPASS`; no empty softmax is evaluated.
4. Checkpoints and bank items use safetensors plus JSON rather than pickle.
5. Medical records use deterministic edit-level splitting and model-visible
   equivalence-key isolation.
6. Port-owned LayerNorm and Linear modules are explicitly initialized to
   PyTorch's exact defaults because the project LLaVA utility globally makes
   both `reset_parameters` methods no-ops; this restores, rather than changes,
   the official source initialization semantics.

No source attention, extractor, generator, residual, routing, loss, optimizer,
or scheduler equation is altered in the source-objective run.
