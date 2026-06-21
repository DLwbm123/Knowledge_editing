# New Paper Methods

## DSCA

DSCA implements Stage 1 of "Dynamic Subspace Concept Alignment for Lifelong VLM Editing" as a concept-cluster activation-space editor.

### Supported Backbones

- BLIP2-OPT through `opt_model.model.decoder.layers.{dsca_layer}`
- MiniGPT-4 through `llama_model.model.layers.{dsca_layer}`
- LLaVA and PaliGemma are deferred until reliable full-sequence masks are verified.

### Stage 1 Behavior

DSCA freezes the base VLM by default. It extracts visual, prompt, and fused features from the same full-sequence masks used by LiveEdit, dynamically assigns edit samples to concept clusters, initializes non-gradient subspace bases with buffered residualized PCA, and routes active DSAM modules with visual filtering followed by fused soft routing.

If the repository is empty, no visual candidate passes threshold, or the routed cluster has no active DSAM basis, the hook is an exact no-op.

### Verification

Lightweight checks:

```bash
python3 -m pytest -q tests/test_dsca.py
```

## LiveEdit

LiveEdit implements "Lifelong Knowledge Editing for Vision Language Models with Low-Rank Mixture-of-Experts" as an EasyEdit-style multimodal editor.

Reference implementation: https://github.com/qizhou000/LiveEdit

### Supported Backbones

- BLIP2-OPT through `opt_model.model.decoder.layers.{liveedit_layer}`
- MiniGPT-4 through `llama_model.model.layers.{liveedit_layer}`
- LLaVA is intentionally deferred for LiveEdit and raises `NotImplementedError` until its expanded image-token masks are verified in this repository.

### Training

Use the existing multimodal trainer with:

```python
from easyeditor import LiveEditMultimodalTrainingHparams, MultimodalTrainer, CaptionDataset

hparams = LiveEditMultimodalTrainingHparams.from_hparams("hparams/TRAINING/LiveEdit/blip2.yaml")
train_ds = CaptionDataset(train_json_path, config=hparams)
val_ds = CaptionDataset(eval_json_path, config=hparams)
trainer = MultimodalTrainer(config=hparams, train_set=train_ds, val_set=val_ds)
trainer.run()
```

The base VLLM is frozen by default. LiveEdit trains only the expert generator, edit-side feature extractor, input-side feature extractor, sentinel/prototype parameters, and residual normalization.

### Editing And Inference

`alg_name: LiveEdit` appends generated low-rank experts to an `ExpertRepository`; it does not modify base VLLM weights. During inference the configured transformer layer hook:

1. extracts visual and prompt features from full-sequence masks,
2. hard-routes experts by visual similarity against the dynamic sentinel threshold,
3. soft-routes selected experts by prompt similarity using `sigmoid(sim) * softmax(sim)`,
4. applies `ReLU(h U^T) V` as a residual to the full hidden state, matching the official implementation.

If the repository is empty, or hard routing selects no experts, the hook is an exact no-op.

### Masks

BLIP2 and MiniGPT-4 outputs now include:

- `vision_mask`
- `prompt_mask`
- `answer_mask`
- `attention_mask`

LiveEdit feature extraction fails fast when these masks are absent or misaligned. Padding and answer tokens are never used for routing features.

### Repository

The expert repository supports append, clear, save/load, device/dtype moves through `nn.Module.to`, and `state_dict` round trips. Set `liveedit_repository_path` to save/load accumulated experts.

### Official Defaults

The configs use official LiveEdit defaults:

- `liveedit_layer: 21`
- `liveedit_module_dim: 1024`
- `liveedit_feature_k: 4`
- `liveedit_rank: 4`
- `liveedit_lora_scale: 5`
- `liveedit_cross_att_heads: 8`
- all LiveEdit loss weights set to `1.0`

### Verification

Lightweight checks:

```bash
python3 -m py_compile easyeditor/trainer/algs/liveedit_utils.py easyeditor/trainer/algs/liveedit.py easyeditor/trainer/blip2_models/blip2_opt.py easyeditor/trainer/blip2_models/mini_gpt4.py
python3 -m pytest -q tests/test_liveedit.py
python3 -m pytest -q tests/test_asam.py
```
