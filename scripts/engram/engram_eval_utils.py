"""Pure utilities for deterministic ENGRAM evaluation."""
from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Iterable, Tuple

import torch
import torch.nn.functional as F


def first_tensor(value: Any) -> torch.Tensor | None:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, dict):
        for item in value.values():
            found = first_tensor(item)
            if found is not None:
                return found
    if isinstance(value, (list, tuple)):
        for item in value:
            found = first_tensor(item)
            if found is not None:
                return found
    for attr in ("last_hidden_state", "logits", "hidden_states"):
        if hasattr(value, attr):
            found = first_tensor(getattr(value, attr))
            if found is not None:
                return found
    return None


def tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().contiguous().cpu()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(str(tuple(value.shape)).encode())
    digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def tensor_metadata(tensor: torch.Tensor) -> Dict[str, Any]:
    return {
        "sha256": tensor_sha256(tensor),
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "stride": list(tensor.stride()),
        "numel": int(tensor.numel()),
    }


def nested_input_hash(value: Any) -> str:
    digest = hashlib.sha256()

    def visit(item: Any, key: str) -> None:
        digest.update(key.encode())
        if isinstance(item, torch.Tensor):
            digest.update(tensor_sha256(item).encode())
        elif isinstance(item, dict):
            for child_key in sorted(item):
                visit(item[child_key], f"{key}.{child_key}")
        elif isinstance(item, (list, tuple)):
            for index, child in enumerate(item):
                visit(child, f"{key}[{index}]")
        elif item is None or isinstance(item, (str, int, float, bool)):
            digest.update(json.dumps(item, sort_keys=True).encode())
        else:
            digest.update(repr(item).encode())

    visit(value, "root")
    return digest.hexdigest()


def shifted_teacher_forced_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
) -> Dict[str, Any]:
    if logits.ndim != 3 or labels.ndim != 2:
        raise ValueError(f"Expected logits [B,T,V] and labels [B,T], got {tuple(logits.shape)} and {tuple(labels.shape)}")
    if logits.shape[:2] != labels.shape:
        raise ValueError(f"Logits/labels sequence mismatch: {tuple(logits.shape[:2])} vs {tuple(labels.shape)}")
    shift_logits = logits[:, :-1, :].float()
    shift_labels = labels[:, 1:]
    mask = shift_labels.ne(ignore_index)
    token_count = int(mask.sum().item())
    if token_count == 0:
        raise RuntimeError("No non-ignored target labels after causal shift")
    safe = shift_labels.masked_fill(~mask, 0)
    log_probs = torch.log_softmax(shift_logits, dim=-1)
    token_log_probs = log_probs.gather(-1, safe.unsqueeze(-1)).squeeze(-1)
    selected = token_log_probs.masked_select(mask)
    positions = mask.nonzero(as_tuple=False).detach().cpu().tolist()
    return {
        "target_nll": float((-selected.mean()).detach().cpu()),
        "avg_target_logprob": float(selected.mean().detach().cpu()),
        "target_token_count": token_count,
        "target_positions_after_shift": positions,
        "target_token_logprobs": selected.detach().cpu().tolist(),
        "labels_sha256": tensor_sha256(labels),
    }


def legacy_tail_metrics(logits: torch.Tensor, raw_target_labels: torch.Tensor, ignore_index: int = -100) -> Dict[str, Any]:
    if logits.ndim != 3 or raw_target_labels.ndim != 2:
        raise ValueError("Legacy metric expects logits [B,T,V] and raw labels [B,N]")
    shifted = logits[:, :-1, :]
    shifted = shifted[:, -raw_target_labels.shape[1] :, :]
    labels = raw_target_labels
    if shifted.shape[:2] != labels.shape:
        raise ValueError(f"Legacy shape mismatch: {tuple(shifted.shape[:2])} vs {tuple(labels.shape)}")
    mask = labels.ne(ignore_index)
    safe = labels.masked_fill(~mask, 0)
    log_probs = torch.log_softmax(shifted.float(), dim=-1)
    values = log_probs.gather(-1, safe.unsqueeze(-1)).squeeze(-1).masked_select(mask)
    return {
        "target_nll": float((-values.mean()).detach().cpu()),
        "target_token_count": int(values.numel()),
        "target_token_logprobs": values.detach().cpu().tolist(),
    }


def compare_tensors(left: torch.Tensor, right: torch.Tensor) -> Dict[str, Any]:
    if tuple(left.shape) != tuple(right.shape):
        return {"shape_equal": False, "left_shape": list(left.shape), "right_shape": list(right.shape)}
    delta = left.detach().float() - right.detach().float()
    return {
        "shape_equal": True,
        "exact_equal": bool(torch.equal(left.detach().cpu(), right.detach().cpu())),
        "max_abs_diff": float(delta.abs().max().cpu()) if delta.numel() else 0.0,
        "mean_abs_diff": float(delta.abs().mean().cpu()) if delta.numel() else 0.0,
        "left_sha256": tensor_sha256(left),
        "right_sha256": tensor_sha256(right),
    }


def full_state_sha256(model: torch.nn.Module) -> Tuple[str, Dict[str, Any]]:
    digest = hashlib.sha256()
    count = 0
    total_bytes = 0
    for kind, items in (("parameter", model.named_parameters()), ("buffer", model.named_buffers())):
        for name, tensor in items:
            value = tensor.detach().contiguous().cpu()
            raw = value.view(torch.uint8).numpy().tobytes()
            digest.update(kind.encode())
            digest.update(name.encode())
            digest.update(str(value.dtype).encode())
            digest.update(str(tuple(value.shape)).encode())
            digest.update(raw)
            count += 1
            total_bytes += len(raw)
    return digest.hexdigest(), {"tensor_count": count, "total_bytes": total_bytes}


def module_state_inventory(model: torch.nn.Module) -> Dict[str, Any]:
    training = [name for name, module in model.named_modules() if module.training]
    dropout = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Dropout) or "dropout" in module.__class__.__name__.lower():
            dropout.append({"name": name, "class": module.__class__.__name__, "p": getattr(module, "p", None), "training": module.training})
    parameters = [
        {"name": name, "shape": list(value.shape), "dtype": str(value.dtype), "device": str(value.device), "stride": list(value.stride())}
        for name, value in model.named_parameters()
    ]
    buffers = [
        {"name": name, "shape": list(value.shape), "dtype": str(value.dtype), "device": str(value.device), "stride": list(value.stride())}
        for name, value in model.named_buffers()
    ]
    return {"training_modules": training, "dropout_modules": dropout, "parameters": parameters, "buffers": buffers}
