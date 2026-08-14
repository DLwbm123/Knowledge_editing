# LiveEdit Source-to-Port Map

Pinned source: `qizhou000/LiveEdit@3615a37b05294509f411df045621940f276a5e6b`.

| Official source | Pinned lines/symbol | Project counterpart | Parity rule |
|---|---:|---|---|
| `modules.py` | 7-39, `Attention` | `upstream_modules.py::Attention` | Exact initialization and tensor equations |
| `modules.py` | 41-88, `QVExtractor` | `upstream_modules.py::QVExtractor` | Separate query systems and visual sentinel |
| `modules.py` | 90-110, `LowRankGenerator` | `upstream_modules.py::LowRankGenerator` | Exact scale `1/(5*sqrt(rank))` |
| `liveedit.py` | 169-178, `get_new_edit` | `source_ops.py::generate_expert_and_keys` | Exclude pre-visual reps |
| `liveedit.py` | 180-190, `get_edit_residual` | `source_ops.py::apply_low_rank_expert_residual` | LayerNorm, ReLU, reconstruction |
| `liveedit.py` | 192-198, `get_moe_fuse_coe` | `source_ops.py::compute_text_soft_weights` | Sigmoid times softmax, no renormalization |
| `liveedit.py` | 132-153, `retrieve_moes` | `source_ops.py::route_repository` | Exact sentinel inequality; explicit empty bypass |
| `liveedit.py` | 251-352, `organize_batch_data` | `data.py`, `source_ops.py::deterministic_source_masks` | Seeded sampling and prefix distractor masks |
| `liveedit.py` | 354-476, `train_a_batch` | `trainer.py`, `source_ops.py` | Source loss signs and Adam schedule |
| `dataset/vllm.py` | 10-145 | `data.py::adapt_record` | Medical field mapping without source mutation |
| `llava.py` | 8-80 | `llavamed_adapter.py` | Adapt audited Mistral multimodal expansion |
| `evaluation/vllm_editor_eval.py` | compatibility evaluator | project natural-generation protocol | Teacher-forced accuracy is secondary only |
