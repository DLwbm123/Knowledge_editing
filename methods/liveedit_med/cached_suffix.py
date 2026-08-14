"""Exact differentiable LLaVA-Med Mistral suffix from cached layer-21 outputs."""
from __future__ import annotations

import torch
from torch.utils.checkpoint import checkpoint


def forward_suffix_hidden(llava_model, layer21_hidden: torch.Tensor, attention_mask: torch.Tensor, *, gradient_checkpointing: bool = False) -> torch.Tensor:
    """Run decoder layers 22..31 and final norm exactly as HF Mistral."""
    core = llava_model.model
    hidden = layer21_hidden
    batch, length, _ = hidden.shape
    cache_position = torch.arange(length, device=hidden.device)
    position_ids = cache_position.unsqueeze(0).expand(batch, -1)
    causal_mask = core._update_causal_mask(attention_mask, hidden, cache_position, None, False)
    position_embeddings = core.rotary_emb(hidden, position_ids)
    for layer in core.layers[22: core.config.num_hidden_layers]:
        def layer_forward(value, current_layer=layer):
            return current_layer(
                value, attention_mask=causal_mask, position_ids=position_ids, past_key_value=None,
                output_attentions=False, use_cache=False, cache_position=cache_position,
                position_embeddings=position_embeddings,
            )[0]
        hidden = checkpoint(layer_forward, hidden, use_reentrant=False) if gradient_checkpointing else layer_forward(hidden)
    return core.norm(hidden)


def forward_suffix(llava_model, layer21_hidden: torch.Tensor, attention_mask: torch.Tensor, *, gradient_checkpointing: bool = False, logits_to_keep=0) -> torch.Tensor:
    """Run the exact suffix and optionally project only selected positions."""
    hidden = forward_suffix_hidden(llava_model, layer21_hidden, attention_mask, gradient_checkpointing=gradient_checkpointing)
    selected = hidden if isinstance(logits_to_keep, int) and logits_to_keep == 0 else hidden[:, logits_to_keep, :]
    return llava_model.lm_head(selected)


def answer_nll(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """HF causal-LM token loss with -100 ignored."""
    shift_logits = logits[:, :-1].float()
    shift_labels = labels[:, 1:].long()
    return torch.nn.functional.cross_entropy(
        shift_logits.reshape(-1, shift_logits.shape[-1]), shift_labels.reshape(-1), ignore_index=-100,
    )


def answer_kl(candidate_logits: torch.Tensor, base_logits: torch.Tensor) -> torch.Tensor:
    """Official KL direction: KL(base || candidate), averaged over answer positions."""
    base = base_logits.float()
    candidate = candidate_logits.float()
    return (base.softmax(-1) * (base.log_softmax(-1) - candidate.log_softmax(-1))).sum(-1).mean()
