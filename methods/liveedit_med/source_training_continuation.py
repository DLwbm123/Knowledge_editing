"""Shared LiveEdit source-training continuation with explicit semantics.

This module is intentionally training-only.  Inference continues to use the
full-block ``Layer21ResidualHook`` and is not switched by this mode.
"""
from __future__ import annotations

from enum import Enum

import torch
from torch.utils.checkpoint import checkpoint


class SourceTrainingContinuationMode(str, Enum):
    """The two audited continuations from a captured layer-21 output."""

    STRICT_SOURCE_REAPPLY_LAYER21 = "strict_source_reapply_layer21"
    CORRECTED_SEMANTICS_CONTINUE_LAYER22 = "corrected_semantics_continue_layer22"


def coerce_source_training_mode(
    value: SourceTrainingContinuationMode | str,
) -> SourceTrainingContinuationMode:
    if isinstance(value, SourceTrainingContinuationMode):
        return value
    try:
        return SourceTrainingContinuationMode(value)
    except ValueError as error:
        raise RuntimeError(f"LIVEEDIT_MED_UNKNOWN_SOURCE_TRAINING_CONTINUATION:{value}") from error


def _decoder_context(
    llava_model,
    hidden: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor | None,
):
    core = llava_model.model
    batch, length, _ = hidden.shape
    cache_position = torch.arange(length, device=hidden.device)
    if position_ids is None:
        position_ids = cache_position.unsqueeze(0).expand(batch, -1)
    else:
        position_ids = position_ids.to(hidden.device)
    causal_mask = core._update_causal_mask(attention_mask, hidden, cache_position, None, False)
    position_embeddings = core.rotary_emb(hidden, position_ids)
    return cache_position, position_ids, causal_mask, position_embeddings


def _run_layer(
    layer,
    hidden: torch.Tensor,
    *,
    causal_mask: torch.Tensor,
    position_ids: torch.Tensor,
    cache_position: torch.Tensor,
    position_embeddings,
    gradient_checkpointing: bool,
) -> torch.Tensor:
    def layer_forward(value):
        output = layer(
            value,
            attention_mask=causal_mask,
            position_ids=position_ids,
            past_key_value=None,
            output_attentions=False,
            use_cache=False,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )
        return output[0] if isinstance(output, (tuple, list)) else output

    if gradient_checkpointing:
        return checkpoint(layer_forward, hidden, use_reentrant=False)
    return layer_forward(hidden)


def forward_source_training_hidden(
    llava_model,
    captured_layer21_output: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    residual: torch.Tensor | None = None,
    mode: SourceTrainingContinuationMode | str,
    position_ids: torch.Tensor | None = None,
    gradient_checkpointing: bool = False,
) -> torch.Tensor:
    """Continue the frozen decoder under one explicit source-training mode.

    The pinned upstream function feeds the captured clean layer-21 output back
    into layer 21.  Its training hook then adds the precomputed residual to the
    re-run layer-21 output.  The corrected port instead adds the residual to
    the captured output and starts from layer 22.

    ``residual=None`` is supported for the legacy corrected suffix wrapper,
    where the caller has already produced the edited layer-21 tensor.
    """
    selected_mode = coerce_source_training_mode(mode)
    core = llava_model.model
    hidden = captured_layer21_output
    cache_position, position_ids, causal_mask, position_embeddings = _decoder_context(
        llava_model, hidden, attention_mask, position_ids
    )

    if selected_mode is SourceTrainingContinuationMode.STRICT_SOURCE_REAPPLY_LAYER21:
        hidden = _run_layer(
            core.layers[21], hidden,
            causal_mask=causal_mask, position_ids=position_ids,
            cache_position=cache_position, position_embeddings=position_embeddings,
            gradient_checkpointing=gradient_checkpointing,
        )
        if residual is not None:
            hidden = hidden + residual.to(device=hidden.device, dtype=hidden.dtype)
        start = 22
    else:
        if residual is not None:
            hidden = hidden + residual.to(device=hidden.device, dtype=hidden.dtype)
        start = 22

    for layer in core.layers[start : core.config.num_hidden_layers]:
        hidden = _run_layer(
            layer, hidden,
            causal_mask=causal_mask, position_ids=position_ids,
            cache_position=cache_position, position_embeddings=position_embeddings,
            gradient_checkpointing=gradient_checkpointing,
        )
    return core.norm(hidden)


def forward_source_training_logits(
    llava_model,
    captured_layer21_output: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    residual: torch.Tensor | None = None,
    mode: SourceTrainingContinuationMode | str,
    position_ids: torch.Tensor | None = None,
    gradient_checkpointing: bool = False,
    logits_to_keep=0,
) -> torch.Tensor:
    hidden = forward_source_training_hidden(
        llava_model,
        captured_layer21_output,
        attention_mask,
        residual=residual,
        mode=mode,
        position_ids=position_ids,
        gradient_checkpointing=gradient_checkpointing,
    )
    selected = hidden if isinstance(logits_to_keep, int) and logits_to_keep == 0 else hidden[:, logits_to_keep, :]
    return llava_model.lm_head(selected)
