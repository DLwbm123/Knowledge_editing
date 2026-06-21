# SAME-Edit Deviations from PRISM/SAME

## Local wrapper instead of PRISM PEFT extension

- PRISM/SAME behavior: registers a custom PEFT type (`MOE_LORA_SAME`) and subclasses PEFT LoRA tuner modules.
- SAME-Edit behavior: replaces selected `torch.nn.Linear` modules with a local `SAMEEditLinear` wrapper inside EasyEdit.
- reason for deviation: this checkout does not vendor PRISM's custom PEFT registry or PEFT fork, and the medical MLLM code already uses local EasyEdit trainer/executor boundaries.
- expected effect on results: module-level forward, routing, covariance, and hook behavior are preserved for linear layers, but PEFT adapter export/merge semantics are not identical.

## Quantized linear wrappers are not implemented in the first version

- PRISM/SAME behavior: includes an 8-bit SAME linear path when bitsandbytes quantized layers are present.
- SAME-Edit behavior: only wraps standard `torch.nn.Linear` modules.
- reason for deviation: the first medical smoke targets local LLaVA-Med full-precision/float16 layers, and quantized wrappers need separate runtime validation.
- expected effect on results: no effect on the intended first smoke; quantized checkpoints must be run unwrapped or extended later.

## Rank remainder handling

- PRISM/SAME behavior: each expert uses `r // expert_num` bottleneck width, so non-divisible ranks silently floor the per-expert width.
- SAME-Edit behavior: uses `per_expert_r = max(1, r // expert_num)` and logs the resulting effective rank.
- reason for deviation: a strict floor can create zero-width experts when `r < expert_num`, which is invalid for `torch.nn.Linear`.
- expected effect on results: when `r` is not divisible by `expert_num`, effective rank can be lower than requested; when `r < expert_num`, effective rank becomes `expert_num` instead of zero.

## State format

- PRISM/SAME behavior: stores SAME carry-over buffers through adapter checkpoint files plus `same_state.bin`.
- SAME-Edit behavior: stores a self-contained `same_edit_state.pt` plus `same_edit_summary.json`.
- reason for deviation: this checkout's EasyEdit runners use PyTorch checkpoints and JSON summaries rather than PRISM's adapter-safetensors merge flow.
- expected effect on results: sequential edit state is preserved, but checkpoints are not directly loadable by PRISM without conversion.

## Prototype routing

- PRISM/SAME behavior: the reference integration can combine routing towers/prototypes with SAME modules.
- SAME-Edit behavior: prototype routing is left disabled in the first implementation; learned hidden routing and oracle edit routing are implemented.
- reason for deviation: prototype extraction needs medical image/text prototype policy and should be validated after one-edit and 5-edit smoke tests.
- expected effect on results: first smoke isolates MoE-LoRA, masks, covariance, and hooks; prototype-based routing accuracy is not measured yet.
