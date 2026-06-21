# LiveEdit Correctness Audit

## Summary

This audit checks the EasyEdit LiveEdit integration against the paper "Lifelong Knowledge Editing for Vision Language Models with Low-Rank Mixture-of-Experts" and the official implementation at <https://github.com/qizhou000/LiveEdit>.

LiveEdit is implemented for BLIP2-OPT and MiniGPT-4 only. LLaVA remains intentionally unsupported because this EasyEdit checkout does not yet expose verified full-sequence region masks for expanded image tokens.

## Files Inspected

- `easyeditor/trainer/algs/liveedit_utils.py`
- `easyeditor/trainer/algs/liveedit.py`
- `easyeditor/trainer/blip2_models/blip2_opt.py`
- `easyeditor/trainer/blip2_models/mini_gpt4.py`
- `easyeditor/trainer/MultimodalTrainer.py`
- `easyeditor/util/alg_train_dict.py`
- `easyeditor/util/alg_dict.py`
- `easyeditor/models/liveedit/liveedit_main.py`
- `tests/test_liveedit.py`
- Official `editor/vllm_editors/liveedit/modules.py`
- Official `editor/vllm_editors/liveedit/liveedit.py`
- Official `editor/vllms_for_edit/blip2/blip2.py`
- Official `editor/vllms_for_edit/minigpt4/minigpt4.py`
- Official `configs/liveedit/{blip2-opt-2.7b,minigpt-4-vicuna-7b}.yaml`

## Files Modified

- `easyeditor/trainer/algs/liveedit_utils.py`
- `easyeditor/trainer/algs/liveedit.py`
- `tests/test_liveedit.py`
- `docs/LIVEEDIT_AUDIT.md`

## Official Code Components Referenced

- `Attention`: scaled cross-attention readout.
- `QVExtractor`: edit-side and input-side visual/query feature extraction.
- `LowRankGenerator`: cross-attention generation of low-rank column/row expert matrices.
- `retrieve_moes`: hard visual routing against prompt-conditioned visual prototype, followed by soft prompt routing.
- `get_edit_residual`: `ReLU(h U^T) V` residual with prompt soft-routing coefficients.
- `train_a_batch`: reliability, generality, locality, soft-routing, and hard-routing losses.
- Official configs: `module_dim=1024`, `cross_att_head_n=8`, `lora_rank=4`, `eqe_n=4`, `lora_scale=5`, `lr=1e-4`, edit layer 21.

## Equation Mapping

- Eq. 1 expert generation: `ExpertGenerator` combines two `LowRankGenerator` modules to produce per-sample `U_e` and `V_e` from the masked edit signal `(h_v, h_p, h_o)`.
- Eq. 2 cross attention: `CrossAttentionReadout` computes scaled query/key attention before value aggregation. EasyEdit adds explicit token masks so padding and non-region tokens cannot affect attention.
- Eq. 3 edit-side feature extraction: `edit_feature_extractor.extract` computes `phi_hat_v` through prompt-to-vision readout and `psi_hat_p` from prompt-only readout.
- Eq. 4 input-side feature extraction: `input_feature_extractor.extract` mirrors Eq. 3 with separate parameters. The visual sentinel/prototype is prompt-conditioned through `extract_sentinel`.
- Eq. 5 hard routing: `hard_route` selects expert `e` iff `sim(phi_bar_v, phi_hat_v_e) > sim(phi_bar_v, phi_bar_Theta)`. Top-k is optional and applied after thresholding; no top-k is forced by default.
- Eq. 6-7 soft routing and residual: `soft_routing_weights` computes `sigmoid(sim) * softmax(sim over selected experts)`. `low_rank_residual` applies `ReLU(h U_e^T) V_e` and zeroes unselected experts.
- Eq. 8-11 edit loss: `LiveEdit.edit_step` generates batch temporary experts, computes reliability and generality CE losses, and computes locality KL from base logits to edited logits with target-token masks only.
- Eq. 12-16 routing loss: `liveedit_routing_losses` implements hard-routing InfoNCE with row-local sentinel candidates, SR1 absolute prompt routing with positive-excluding negatives, and SR2 relative prompt InfoNCE over generality plus locality prompt features.

## Mask Construction

BLIP2-OPT builds full-sequence masks after concatenating Q-Former query embeddings and OPT token embeddings:

- `vision_mask`: projected image query tokens.
- `prompt_mask`: non-padding text positions with target label `-100`, excluding vision tokens.
- `answer_mask`: non-padding target positions with labels not equal to `-100`.
- `attention_mask`: full non-padding sequence mask.

MiniGPT-4 builds the same mask contract after prompt wrapping:

- `vision_mask`: exact wrapped `<ImageHere>` embedding span.
- `prompt_mask`: prompt/system/text positions with target label `-100`, excluding vision tokens.
- `answer_mask`: target answer positions only.
- `attention_mask`: full non-padding sequence mask.

Edit-time expert generation requires all four masks. Inference routing requires only `vision_mask`, `prompt_mask`, and `attention_mask`, because answer tokens may be absent during ordinary inference.

## Empty And Frozen Behavior

- Empty repository calls the base VLLM path with no residual and identical logits.
- If hard routing selects zero experts, soft-routing weights are exactly zero and the low-rank residual is zero.
- `liveedit_freeze_vllm` defaults to true. LiveEdit freezes the base VLLM, keeps it in eval mode, and trains only the expert generator, edit/input feature extractors, visual sentinel, and residual normalization.
- `state_dict()` excludes base VLLM parameters and keeps LiveEdit modules plus repository buffers and metadata.

## Deviations And EasyEdit Adaptation

- Official code slices region spans from wrapper-returned token ranges. EasyEdit instead uses full-sequence boolean masks because its multimodal wrappers already construct teacher-forced `inputs_embeds`, labels, and attention masks.
- Official training code can simulate multiple requests per knowledge unit with explicit MoE masks. The EasyEdit dataset batch exposes one `edit_inner` sample per row, so temporary experts are generated per row and mixed across the temporary expert pool.
- Official BLIP2 hook path is for Hugging Face `language_model`; this EasyEdit wrapper exposes the OPT decoder as `opt_model.model.decoder.layers.{liveedit_layer}`.
- LLaVA remains deferred until its image-token expansion and full-sequence masks are verified in this repository.

## Verification

Commands run during this audit:

```bash
python3 -m py_compile \
  easyeditor/trainer/algs/liveedit_utils.py \
  easyeditor/trainer/algs/liveedit.py \
  easyeditor/trainer/blip2_models/blip2_opt.py \
  easyeditor/trainer/blip2_models/mini_gpt4.py \
  easyeditor/trainer/MultimodalTrainer.py

python3 -m pytest -q tests/test_liveedit.py tests/test_asam.py
```

Result:

```text
py_compile passed
45 passed, 3 warnings
liveedit registration smoke ok
```

The warnings are existing `timm` deprecation warnings from package imports.
