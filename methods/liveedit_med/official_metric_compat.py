"""Official-style teacher-forced metrics for the LiveEdit-Med port.

The upstream evaluator compares argmax token IDs under a target/locality mask.
The local LLaVA-Med wrapper returns full causal-LM labels, so predictions and
labels are shifted once before the same masked comparison is applied.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch


def causal_teacher_forced_tokens(logits: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return next-token argmax, references, and the official comparison mask."""
    if logits.ndim != 3 or labels.ndim != 2 or logits.shape[:2] != labels.shape:
        raise ValueError("LIVEEDIT_MED_OFFICIAL_METRIC_SHAPE_MISMATCH")
    predicted = logits[:, :-1].argmax(-1)
    references = labels[:, 1:]
    mask = references.ne(-100)
    return predicted, references, mask


def masked_token_accuracy(predicted: torch.Tensor, references: torch.Tensor, mask: torch.Tensor) -> dict[str, Any]:
    if predicted.shape != references.shape or mask.shape != references.shape:
        raise ValueError("LIVEEDIT_MED_OFFICIAL_METRIC_MASK_MISMATCH")
    mask = mask.bool()
    count = int(mask.sum().item())
    correct = int(((predicted == references) & mask).sum().item())
    return {
        "correct_tokens": correct,
        "total_tokens": count,
        "accuracy": float(correct / count) if count else None,
        "exact": bool(count and correct == count),
    }


def teacher_forced_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, Any]:
    predicted, references, mask = causal_teacher_forced_tokens(logits, labels)
    result = masked_token_accuracy(predicted, references, mask)
    result.update({
        "predicted_token_ids": predicted[mask].detach().cpu().tolist(),
        "reference_token_ids": references[mask].detach().cpu().tolist(),
        "mask_count": int(mask.sum().item()),
    })
    return result


def locality_preservation(pre_logits: torch.Tensor, post_logits: torch.Tensor, labels: torch.Tensor) -> dict[str, Any]:
    pre, _references, mask = causal_teacher_forced_tokens(pre_logits, labels)
    post, _references2, post_mask = causal_teacher_forced_tokens(post_logits, labels)
    if not torch.equal(mask, post_mask):
        raise ValueError("LIVEEDIT_MED_LOCALITY_MASK_DRIFT")
    result = masked_token_accuracy(post, pre, mask)
    result.update({
        "pre_token_ids": pre[mask].detach().cpu().tolist(),
        "post_token_ids": post[mask].detach().cpu().tolist(),
        "mask_count": int(mask.sum().item()),
    })
    return result


def aggregate_metric(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    correct = sum(int(row["correct_tokens"]) for row in rows)
    total = sum(int(row["total_tokens"]) for row in rows)
    exact = sum(bool(row["exact"]) for row in rows)
    return {
        "example_count": len(rows),
        "exact_count": exact,
        "exact_rate": float(exact / len(rows)) if rows else None,
        "correct_tokens": correct,
        "total_tokens": total,
        "token_accuracy": float(correct / total) if total else None,
    }
