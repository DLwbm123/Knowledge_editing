"""Hash-based comparison helpers for end-to-end upstream trace parity."""
from __future__ import annotations

import hashlib
import json
import random
from types import MethodType
from typing import Any, Mapping, Sequence

import torch


REQUIRED_TRACE_BOUNDARIES = (
    "processor_outputs",
    "merged_multimodal_input_embeddings",
    "attention_mask_and_position_ids",
    "layer21_full_block_clean_hidden_state",
    "visual_span_indices_and_tensor",
    "question_span_indices_and_tensor",
    "target_answer_span_indices_and_tensor",
    "edit_end_visual_routing_key",
    "edit_end_text_routing_key",
    "input_end_visual_routing_key",
    "input_end_text_routing_key",
    "question_conditioned_visual_sentinel_key",
    "generated_moe_c",
    "generated_moe_r",
    "visual_similarity_scores",
    "sentinel_score",
    "hard_candidate_mask",
    "text_raw_score",
    "sigmoid_absolute_weight",
    "softmax_relative_weight",
    "final_weight",
    "expert_residual",
    "post_layer_hidden_state",
    "final_logits",
    "source_teacher_forced_token_prediction",
    "source_loss_components",
    "optimizer_step_parameter_hashes",
)


def seed_everything(seed: int = 42) -> None:
    """Set the deterministic boundary required by the Stage-A protocol."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().contiguous().cpu()
    payload = str(tensor.dtype).encode() + str(tuple(tensor.shape)).encode() + tensor.view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(payload).hexdigest()


def compare_tensor(name: str, upstream: torch.Tensor, port: torch.Tensor, *, atol: float, rtol: float) -> dict[str, Any]:
    if upstream.shape != port.shape:
        return {"name": name, "passed": False, "reason": "shape_mismatch", "upstream_shape": list(upstream.shape), "port_shape": list(port.shape)}
    delta = (upstream.detach().float().cpu() - port.detach().float().cpu()).abs()
    passed = bool(torch.allclose(upstream.detach().float().cpu(), port.detach().float().cpu(), atol=atol, rtol=rtol))
    return {
        "name": name, "passed": passed, "max_abs_error": float(delta.max().item()) if delta.numel() else 0.0,
        "mean_abs_error": float(delta.mean().item()) if delta.numel() else 0.0,
        "upstream_sha256": tensor_sha256(upstream), "port_sha256": tensor_sha256(port),
        "shape": list(upstream.shape), "atol": atol, "rtol": rtol,
    }


def compare_tensor_mapping(
    name: str,
    upstream: Mapping[str, torch.Tensor],
    port: Mapping[str, torch.Tensor],
    *,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    """Compare a named tensor bundle as one required trace boundary."""
    upstream_keys, port_keys = sorted(upstream), sorted(port)
    if upstream_keys != port_keys:
        return {
            "name": name,
            "passed": False,
            "reason": "key_mismatch",
            "upstream_keys": upstream_keys,
            "port_keys": port_keys,
        }
    items = [compare_tensor(key, upstream[key], port[key], atol=atol, rtol=rtol) for key in upstream_keys]
    return {
        "name": name,
        "passed": all(item["passed"] for item in items),
        "items": items,
        "comparison": "tensor_mapping",
    }


def summarize_trace(rows: Sequence[Mapping[str, Any]], *, required_names: Sequence[str]) -> dict[str, Any]:
    by_name = {str(row["name"]): dict(row) for row in rows}
    missing = [name for name in required_names if name not in by_name]
    passed = sum(bool(by_name[name].get("passed")) for name in required_names if name in by_name)
    return {"required": len(required_names), "passed": passed, "missing": missing,
            "all_passed": not missing and passed == len(required_names), "comparisons": [by_name[name] for name in required_names if name in by_name]}


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def compare_discrete(name: str, upstream: Any, port: Any) -> dict[str, Any]:
    """Compare token ids, masks, spans, hashes, and other exact boundaries."""
    if isinstance(upstream, torch.Tensor):
        upstream = upstream.detach().cpu().tolist()
    if isinstance(port, torch.Tensor):
        port = port.detach().cpu().tolist()
    return {"name": name, "passed": upstream == port, "upstream": upstream, "port": port,
            "comparison": "exact"}


def state_dict_sha256(state: Mapping[str, torch.Tensor]) -> str:
    """Stable hash independent of dictionary insertion order and device."""
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().contiguous().cpu()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def legacy_llava_merge_input_ids_with_image_features(
    model: Any,
    image_features: torch.Tensor,
    inputs_embeds: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor | None,
):
    """Compatibility copy of the source-era HF LLaVA merge boundary.

    LiveEdit's pinned LLaVA wrapper calls this private method directly.  It was
    still present in transformers 4.48.3 and was removed before the project's
    4.51.3 runtime.  Keeping the exact source-era operation here avoids changing
    the shared Python environment while allowing the immutable upstream wrapper
    to execute.
    """
    num_images, num_image_patches, embed_dim = image_features.shape
    batch_size, sequence_length = input_ids.shape
    pad_token_id = getattr(model, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = model.config.pad_token_id
    left_padding = not torch.sum(input_ids[:, -1] == torch.tensor(pad_token_id, device=input_ids.device))
    special_image_token_mask = input_ids == model.config.image_token_index
    num_special_image_tokens = torch.sum(special_image_token_mask, dim=-1)
    max_embed_dim = num_special_image_tokens.max() * (num_image_patches - 1) + sequence_length
    batch_indices, non_image_indices = torch.where(input_ids != model.config.image_token_index)
    new_token_positions = torch.cumsum(special_image_token_mask * (num_image_patches - 1) + 1, -1) - 1
    nb_image_pad = max_embed_dim - 1 - new_token_positions[:, -1]
    if left_padding:
        new_token_positions += nb_image_pad[:, None]
    text_to_overwrite = new_token_positions[batch_indices, non_image_indices]
    final_embedding = torch.zeros(batch_size, max_embed_dim, embed_dim,
                                  dtype=inputs_embeds.dtype, device=inputs_embeds.device)
    final_attention_mask = torch.zeros(batch_size, max_embed_dim,
                                       dtype=attention_mask.dtype, device=inputs_embeds.device)
    final_labels = None
    if labels is not None:
        final_labels = torch.full((batch_size, max_embed_dim), model.config.ignore_index,
                                  dtype=input_ids.dtype, device=input_ids.device)
    target_device = inputs_embeds.device
    batch_indices = batch_indices.to(target_device)
    non_image_indices = non_image_indices.to(target_device)
    text_to_overwrite = text_to_overwrite.to(target_device)
    attention_mask = attention_mask.to(target_device)
    final_embedding[batch_indices, text_to_overwrite] = inputs_embeds[batch_indices, non_image_indices]
    final_attention_mask[batch_indices, text_to_overwrite] = attention_mask[batch_indices, non_image_indices]
    if final_labels is not None:
        final_labels[batch_indices, text_to_overwrite] = labels[batch_indices, non_image_indices]
    image_to_overwrite = torch.full((batch_size, max_embed_dim), True,
                                    dtype=torch.bool, device=inputs_embeds.device)
    image_to_overwrite[batch_indices, text_to_overwrite] = False
    image_to_overwrite &= image_to_overwrite.cumsum(-1) - 1 >= nb_image_pad[:, None].to(target_device)
    if image_to_overwrite.sum() != image_features.shape[:-1].numel():
        raise ValueError("LIVEEDIT_STAGE_A_IMAGE_TOKEN_COUNT_MISMATCH")
    final_embedding[image_to_overwrite] = image_features.contiguous().reshape(-1, embed_dim).to(target_device)
    final_attention_mask |= image_to_overwrite
    position_ids = (final_attention_mask.cumsum(-1) - 1).masked_fill_(final_attention_mask == 0, 1)
    pad_batch, pad_indices = torch.where(input_ids == pad_token_id)
    indices_to_mask = new_token_positions[pad_batch, pad_indices]
    final_embedding[pad_batch, indices_to_mask] = 0
    return final_embedding, final_attention_mask, final_labels, position_ids


def install_source_era_llava_merge(model: Any) -> bool:
    """Install the compatibility method only when the runtime removed it."""
    if hasattr(model, "_merge_input_ids_with_image_features"):
        return False
    model._merge_input_ids_with_image_features = MethodType(
        lambda self, image_features, inputs_embeds, input_ids, attention_mask, labels:
        legacy_llava_merge_input_ids_with_image_features(
            self, image_features, inputs_embeds, input_ids, attention_mask, labels
        ),
        model,
    )
    return True


def source_era_llava_processor_outputs(processor: Any, text: str, image: Any) -> dict[str, torch.Tensor]:
    """Run the processor boundary as it behaved for the pinned LiveEdit source.

    Newer ``transformers`` expands one ``<image>`` placeholder into hundreds of
    placeholder IDs inside ``LlavaProcessor``.  The pinned upstream wrapper
    instead expects one placeholder and performs expansion in
    ``_merge_input_ids_with_image_features``.  Calling the two owned processor
    components directly restores that source-era public behavior without
    mutating the model or the installed package.
    """
    tokenized = processor.tokenizer([text], return_tensors="pt", padding=True)
    pixels = processor.image_processor(images=[image], return_tensors="pt")
    return {
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"],
        "pixel_values": pixels["pixel_values"],
    }
