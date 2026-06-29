from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


def _finite_float(value: torch.Tensor) -> float:
    scalar = float(value.detach().cpu())
    if scalar != scalar or scalar in {float("inf"), float("-inf")}:
        raise RuntimeError(f"ENGRAM erasure metric is not finite: {scalar}")
    return scalar


def _token_strings(token_ids: List[int], tokenizer: Optional[Any]) -> List[str]:
    if tokenizer is None:
        return [str(token_id) for token_id in token_ids]
    if hasattr(tokenizer, "convert_ids_to_tokens"):
        return [str(token) for token in tokenizer.convert_ids_to_tokens(token_ids)]
    if hasattr(tokenizer, "decode"):
        return [str(tokenizer.decode([token_id], skip_special_tokens=False)) for token_id in token_ids]
    return [str(token_id) for token_id in token_ids]


def sequence_nll_and_logprob(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
    *,
    input_ids: Optional[torch.Tensor] = None,
    tokenizer: Optional[Any] = None,
) -> Dict[str, Any]:
    """Compute causal-LM mean token NLL and summed logprob for valid answer tokens."""
    if logits.dim() != 3:
        raise ValueError(f"Expected logits [batch, seq, vocab], got {tuple(logits.shape)}")
    if labels.dim() != 2:
        raise ValueError(f"Expected labels [batch, seq], got {tuple(labels.shape)}")
    if logits.shape[:2] != labels.shape:
        min_len = min(logits.shape[1], labels.shape[1])
        logits = logits[:, :min_len]
        labels = labels[:, :min_len]
        if isinstance(input_ids, torch.Tensor):
            input_ids = input_ids[:, :min_len]
    if logits.shape[1] < 2:
        return {
            "nll": float("nan"),
            "logprob": float("nan"),
            "num_tokens": 0,
            "answer_token_count": 0,
            "answer_tokens": [],
            "answer_token_ids": [],
            "ignored_token_count": int(labels.numel()),
            "shift_applied": True,
        }

    shift_logits = logits[:, :-1]
    shift_labels = labels[:, 1:]
    valid = shift_labels.ne(ignore_index)
    ignored_token_count = int((~valid).sum().detach().cpu())
    if not valid.any():
        return {
            "nll": float("nan"),
            "logprob": float("nan"),
            "num_tokens": 0,
            "answer_token_count": 0,
            "answer_tokens": [],
            "answer_token_ids": [],
            "ignored_token_count": ignored_token_count,
            "shift_applied": True,
        }

    log_probs = F.log_softmax(shift_logits.float(), dim=-1)
    safe_labels = shift_labels.masked_fill(~valid, 0)
    token_log_probs = log_probs.gather(dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(-1)
    selected = token_log_probs[valid]
    logprob = selected.sum()
    nll = -selected.mean()
    answer_token_ids = [int(token_id) for token_id in shift_labels[valid].detach().cpu().tolist()]
    return {
        "nll": _finite_float(nll),
        "logprob": _finite_float(logprob),
        "num_tokens": int(selected.numel()),
        "answer_token_count": int(selected.numel()),
        "answer_tokens": _token_strings(answer_token_ids, tokenizer),
        "answer_token_ids": answer_token_ids,
        "ignored_token_count": ignored_token_count,
        "shift_applied": True,
    }


def _extract_logits_labels(output: Any, sample: Optional[Dict[str, Any]] = None) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    logits = getattr(output, "logits", None)
    labels = getattr(output, "labels", None)
    input_ids = getattr(output, "input_ids", None)
    if isinstance(output, dict):
        logits = output.get("logits", logits)
        labels = output.get("labels", labels)
        input_ids = output.get("input_ids", input_ids)
    if labels is None and sample is not None:
        labels = sample.get("labels")
    if input_ids is None and sample is not None:
        input_ids = sample.get("input_ids")
    if logits is None:
        raise RuntimeError("model output did not include logits")
    if labels is None:
        raise RuntimeError("model output and sample did not include labels")
    if not isinstance(logits, torch.Tensor):
        raise RuntimeError(f"logits is not a tensor: {type(logits).__name__}")
    if not isinstance(labels, torch.Tensor):
        raise RuntimeError(f"labels is not a tensor: {type(labels).__name__}")
    if input_ids is not None and not isinstance(input_ids, torch.Tensor):
        input_ids = None
    return logits, labels, input_ids


def model_answer_nll_and_logprob(
    model: torch.nn.Module,
    sample: Dict[str, Any],
    ignore_index: int = -100,
) -> Dict[str, float]:
    """Evaluate answer-token NLL/logprob from a model forward pass."""
    model.eval()
    with torch.no_grad():
        output = model(sample)
    logits, labels, input_ids = _extract_logits_labels(output, sample)
    tokenizer = getattr(model, "tokenizer", None)
    return sequence_nll_and_logprob(logits, labels, ignore_index=ignore_index, input_ids=input_ids, tokenizer=tokenizer)


def safe_model_answer_nll_and_logprob(
    model: torch.nn.Module,
    sample: Dict[str, Any],
    ignore_index: int = -100,
) -> Dict[str, Any]:
    try:
        metrics = model_answer_nll_and_logprob(model, sample, ignore_index=ignore_index)
    except Exception as exc:
        return {
            "available": False,
            "unavailable_reason": f"{type(exc).__name__}: {exc}",
        }
    if metrics.get("num_tokens", 0) <= 0:
        return {
            "available": False,
            "unavailable_reason": "no valid answer tokens after causal shift",
            **metrics,
        }
    return {"available": True, **metrics}


def erasure_delta_metrics(
    *,
    target_before: Optional[Dict[str, float]] = None,
    target_after: Optional[Dict[str, float]] = None,
    reference_before: Optional[Dict[str, float]] = None,
    reference_after: Optional[Dict[str, float]] = None,
    target_generation_before: Optional[str] = None,
    target_generation_after: Optional[str] = None,
    locality_generation_before: Optional[str] = None,
    locality_generation_after: Optional[str] = None,
    unavailable_reason: Optional[str] = None,
) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {
        "target_generation_before": target_generation_before,
        "target_generation_after": target_generation_after,
        "locality_generation_before": locality_generation_before,
        "locality_generation_after": locality_generation_after,
    }
    if unavailable_reason:
        metrics.update(
            {
                "erase_logprob_metrics_available": False,
                "erase_logprob_unavailable_reason": unavailable_reason,
            }
        )
        return metrics

    if target_before and target_after:
        metrics.update(
            {
                "erase_target_logprob_before": target_before["logprob"],
                "erase_target_logprob_after": target_after["logprob"],
                "erase_target_nll_before": target_before["nll"],
                "erase_target_nll_after": target_after["nll"],
                "erase_success_logprob_drop": target_before["logprob"] - target_after["logprob"],
                "erase_success_nll_increase": target_after["nll"] - target_before["nll"],
            }
        )
    if reference_before and reference_after:
        metrics.update(
            {
                "reference_logprob_before": reference_before["logprob"],
                "reference_logprob_after": reference_after["logprob"],
                "reference_nll_before": reference_before["nll"],
                "reference_nll_after": reference_after["nll"],
                "reference_delta_abs": abs(reference_after["nll"] - reference_before["nll"]),
            }
        )
    metrics["erase_logprob_metrics_available"] = bool(target_before and target_after)
    return metrics
