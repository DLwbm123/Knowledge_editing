"""Exact differentiable LLaVA-Med Mistral suffix from cached layer-21 outputs."""
from __future__ import annotations

import torch

from .source_training_continuation import (
    SourceTrainingContinuationMode,
    forward_source_training_hidden,
)


def forward_suffix_hidden(llava_model, layer21_hidden: torch.Tensor, attention_mask: torch.Tensor, *, gradient_checkpointing: bool = False) -> torch.Tensor:
    """Run decoder layers 22..31 and final norm exactly as HF Mistral."""
    return forward_source_training_hidden(
        llava_model,
        layer21_hidden,
        attention_mask,
        mode=SourceTrainingContinuationMode.CORRECTED_SEMANTICS_CONTINUE_LAYER22,
        gradient_checkpointing=gradient_checkpointing,
    )


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
