from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

LOG = logging.getLogger(__name__)


@dataclass
class CovarianceStat:
    cov: torch.Tensor
    count: int = 0
    input_dim: int = 0
    absorb_bias: bool = False
    warnings: List[str] = field(default_factory=list)

    @property
    def cov_dim(self) -> int:
        return int(self.cov.shape[0])


@dataclass
class SelectedLayer:
    name: str
    module: nn.Linear
    input_dim: int
    output_dim: int
    absorb_bias: bool

    @property
    def cov_dim(self) -> int:
        return self.input_dim + (1 if self.absorb_bias else 0)


def dtype_from_name(name: str) -> torch.dtype:
    normalized = str(name).lower()
    if normalized in {"float64", "fp64", "double"}:
        return torch.float64
    return torch.float32


def as_device(device: Any) -> torch.device:
    if isinstance(device, torch.device):
        return device
    if isinstance(device, str):
        if device.startswith(("cuda", "cpu", "mps")):
            return torch.device(device)
        if device.isdigit():
            return torch.device(f"cuda:{device}")
    if isinstance(device, int):
        return torch.device(f"cuda:{device}")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def first_tensor(value: Any) -> Optional[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, dict):
        for item in value.values():
            found = first_tensor(item)
            if found is not None:
                return found
        return None
    if isinstance(value, (list, tuple)):
        for item in value:
            found = first_tensor(item)
            if found is not None:
                return found
    return None


def _positions(mask: torch.Tensor) -> List[List[int]]:
    if mask.dim() == 1:
        return [torch.where(mask.bool())[0].detach().cpu().tolist()]
    return [torch.where(row.bool())[0].detach().cpu().tolist() for row in mask]


def loss_predictor_mask_from_labels(
    labels: torch.Tensor,
    *,
    ignore_index: int = -100,
    attention_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Select hidden positions whose logits predict non-ignored labels."""
    if labels.dim() != 2:
        raise ValueError(f"Expected labels [batch, seq], got {tuple(labels.shape)}")
    mask = torch.zeros_like(labels, dtype=torch.bool)
    if labels.shape[1] > 1:
        shift_labels = labels[:, 1:]
        loss_positions = shift_labels.ne(ignore_index)
        if isinstance(attention_mask, torch.Tensor) and tuple(attention_mask.shape) == tuple(labels.shape):
            loss_positions = loss_positions & attention_mask[:, :-1].bool() & attention_mask[:, 1:].bool()
        mask[:, :-1] = loss_positions
    raw_answer = labels.ne(ignore_index)
    return mask, {
        "raw_label_answer_positions": _positions(raw_answer),
        "shifted_loss_positions": _positions(mask),
        "num_selected_tokens": int(mask.sum().detach().cpu()),
        "fallback_used": False,
    }


def _shift_answer_mask_left(answer_mask: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    shifted = torch.zeros_like(answer_mask, dtype=torch.bool)
    if answer_mask.shape[-1] > 1:
        shifted[:, :-1] = answer_mask[:, 1:].bool()
    if isinstance(attention_mask, torch.Tensor) and tuple(attention_mask.shape) == tuple(answer_mask.shape):
        shifted = shifted & attention_mask.bool()
    return shifted


def prompt_last_mask_from_batch(batch: Dict[str, Any]) -> Optional[torch.Tensor]:
    prompt = batch.get("prompt_mask")
    if not isinstance(prompt, torch.Tensor):
        return None
    attn = batch.get("attention_mask")
    answer = batch.get("answer_mask")
    prompt_bool = prompt.bool()
    if isinstance(attn, torch.Tensor) and tuple(attn.shape) == tuple(prompt.shape):
        prompt_bool = prompt_bool & attn.bool()
    result = torch.zeros_like(prompt_bool)
    for row_idx, row in enumerate(prompt_bool):
        candidates = torch.where(row)[0]
        if isinstance(answer, torch.Tensor) and tuple(answer.shape) == tuple(prompt.shape):
            answer_pos = torch.where(answer[row_idx].bool())[0]
            if answer_pos.numel() > 0:
                candidates = candidates[candidates < answer_pos.min()]
        if candidates.numel() > 0:
            result[row_idx, int(candidates[-1].item())] = True
    return result


def token_scope_mask_from_batch(
    batch: Dict[str, Any],
    token_scope: str,
    *,
    mask_fallback: str = "all",
    ignore_index: int = -100,
) -> Tuple[Optional[torch.Tensor], Dict[str, Any]]:
    scope = str(token_scope or "all").lower()
    fallback = str(mask_fallback or "all").lower()
    diag: Dict[str, Any] = {
        "token_scope": scope,
        "raw_label_answer_positions": [],
        "shifted_loss_positions": [],
        "num_selected_tokens": None,
        "fallback_used": False,
    }
    if scope == "all":
        diag["num_selected_tokens"] = None
        return None, diag
    if scope in {"attention", "non_padding", "nonpad"}:
        mask = batch.get("attention_mask")
    elif scope == "answer":
        mask = batch.get("answer_mask")
    elif scope == "prompt":
        mask = batch.get("prompt_mask")
    elif scope == "vision":
        mask = batch.get("vision_mask")
    elif scope == "non_answer":
        attn = batch.get("attention_mask")
        ans = batch.get("answer_mask")
        if isinstance(attn, torch.Tensor) and isinstance(ans, torch.Tensor):
            mask = attn.bool() & ~ans.bool()
        else:
            mask = None
    elif scope == "prompt_last":
        mask = prompt_last_mask_from_batch(batch)
    elif scope == "loss_predictor":
        labels = batch.get("labels")
        attn = batch.get("attention_mask")
        if isinstance(labels, torch.Tensor) and labels.dim() == 2 and (
            not isinstance(attn, torch.Tensor) or tuple(labels.shape) == tuple(attn.shape)
        ):
            mask, loss_diag = loss_predictor_mask_from_labels(labels, ignore_index=ignore_index, attention_mask=attn)
            diag.update(loss_diag)
            diag["token_scope"] = scope
            return mask, diag
        diag["fallback_used"] = True
        diag["fallback_reason"] = "labels unavailable or not aligned with attention_mask"
        ans = batch.get("answer_mask")
        if isinstance(ans, torch.Tensor) and fallback not in {"none", "empty"}:
            mask = _shift_answer_mask_left(ans, attn if isinstance(attn, torch.Tensor) else None)
            diag["raw_label_answer_positions"] = _positions(ans.bool())
            diag["shifted_loss_positions"] = _positions(mask.bool())
            diag["num_selected_tokens"] = int(mask.sum().detach().cpu())
            return mask, diag
        if fallback in {"none", "empty"} and isinstance(attn, torch.Tensor):
            mask = torch.zeros_like(attn, dtype=torch.bool)
            diag["num_selected_tokens"] = 0
            return mask, diag
        return None, diag
    else:
        raise ValueError(f"Unknown ENGRAM token scope: {scope}")

    if isinstance(mask, torch.Tensor):
        diag["num_selected_tokens"] = int(mask.bool().sum().detach().cpu())
    return mask, diag


def flatten_activation_rows(
    activations: Any,
    input_dim: int,
    mask: Optional[torch.Tensor] = None,
    *,
    mask_fallback: str = "all",
) -> Tuple[torch.Tensor, Optional[str]]:
    x = first_tensor(activations)
    if x is None:
        return torch.empty(0, input_dim), "no tensor input found"
    if x.numel() == 0:
        return torch.empty(0, input_dim, device=x.device), None
    if x.shape[-1] != input_dim:
        return torch.empty(0, input_dim, device=x.device), f"last dim {x.shape[-1]} != expected {input_dim}"

    flat = x.detach().reshape(-1, input_dim)
    if mask is None:
        return flat, None

    if not isinstance(mask, torch.Tensor):
        if str(mask_fallback).lower() == "none":
            return torch.empty(0, input_dim, device=x.device), "mask missing and fallback=none"
        return flat, "mask missing; falling back to all rows"

    if tuple(mask.shape) == tuple(x.shape[:-1]):
        flat_mask = mask.reshape(-1).to(device=x.device, dtype=torch.bool)
    elif x.dim() == 2 and mask.dim() == 1 and mask.shape[0] == x.shape[0]:
        flat_mask = mask.to(device=x.device, dtype=torch.bool)
    else:
        if str(mask_fallback).lower() == "none":
            return torch.empty(0, input_dim, device=x.device), "mask shape does not align and fallback=none"
        return flat, f"mask shape {tuple(mask.shape)} does not align with activation {tuple(x.shape)}; falling back to all rows"

    if flat_mask.numel() != flat.shape[0]:
        if str(mask_fallback).lower() == "none":
            return torch.empty(0, input_dim, device=x.device), "flattened mask does not align and fallback=none"
        return flat, "flattened mask does not align; falling back to all rows"
    return flat[flat_mask], None


def covariance_from_rows(
    rows: torch.Tensor,
    *,
    input_dim: int,
    absorb_bias: bool,
    device: Any = "cpu",
    dtype: torch.dtype = torch.float32,
) -> CovarianceStat:
    cov_dim = input_dim + (1 if absorb_bias else 0)
    cov_device = as_device(device)
    cov = torch.zeros(cov_dim, cov_dim, device=cov_device, dtype=dtype)
    if rows.numel() == 0:
        return CovarianceStat(cov=cov, count=0, input_dim=input_dim, absorb_bias=absorb_bias)

    with torch.no_grad():
        rows = rows.to(device=cov_device, dtype=dtype)
        if absorb_bias:
            ones = torch.ones(rows.shape[0], 1, device=cov_device, dtype=dtype)
            rows = torch.cat([rows, ones], dim=-1)
        cov.add_(rows.transpose(0, 1).matmul(rows))
    return CovarianceStat(cov=cov, count=int(rows.shape[0]), input_dim=input_dim, absorb_bias=absorb_bias)


def covariance_from_activations(
    activations: Any,
    *,
    input_dim: int,
    mask: Optional[torch.Tensor] = None,
    absorb_bias: bool = False,
    device: Any = "cpu",
    dtype: torch.dtype = torch.float32,
    mask_fallback: str = "all",
) -> CovarianceStat:
    rows, warning = flatten_activation_rows(activations, input_dim, mask, mask_fallback=mask_fallback)
    stat = covariance_from_rows(rows, input_dim=input_dim, absorb_bias=absorb_bias, device=device, dtype=dtype)
    if warning:
        stat.warnings.append(warning)
    return stat


class LayerCovarianceCollector:
    """Forward-pre-hook covariance collector for selected Linear modules."""

    def __init__(
        self,
        layers: Sequence[SelectedLayer],
        *,
        covariance_device: Any = "cpu",
        covariance_dtype: torch.dtype = torch.float32,
        token_scope: str = "all",
        mask_fallback: str = "all",
    ) -> None:
        self.layers = list(layers)
        self.covariance_device = covariance_device
        self.covariance_dtype = covariance_dtype
        self.token_scope = str(token_scope or "all").lower()
        self.mask_fallback = str(mask_fallback or "all").lower()
        self.stats: Dict[str, CovarianceStat] = {
            layer.name: CovarianceStat(
                cov=torch.zeros(layer.cov_dim, layer.cov_dim, device=as_device(covariance_device), dtype=covariance_dtype),
                count=0,
                input_dim=layer.input_dim,
                absorb_bias=layer.absorb_bias,
            )
            for layer in self.layers
        }
        self._handles: List[Any] = []
        self._current_batch: Dict[str, Any] = {}
        self._warned: set[str] = set()
        self._logged_scope_keys: set[str] = set()
        self.selection_logs: List[Dict[str, Any]] = []

    def __enter__(self) -> "LayerCovarianceCollector":
        for layer in self.layers:
            self._handles.append(layer.module.register_forward_pre_hook(self._make_hook(layer)))
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._current_batch = {}

    def set_batch(self, batch: Dict[str, Any]) -> None:
        self._current_batch = batch or {}

    def clear_batch(self) -> None:
        self._current_batch = {}

    def _make_hook(self, layer: SelectedLayer):
        def hook(module: nn.Module, inputs: Tuple[Any, ...]) -> None:
            mask = self._mask_for(inputs)
            stat = covariance_from_activations(
                inputs,
                input_dim=layer.input_dim,
                mask=mask,
                absorb_bias=layer.absorb_bias,
                device=self.covariance_device,
                dtype=self.covariance_dtype,
                mask_fallback=self.mask_fallback,
            )
            if stat.count == 0 and stat.warnings:
                self._log_once(layer.name, stat.warnings[-1])
            if stat.count == 0:
                return
            target = self.stats[layer.name]
            target.cov.add_(stat.cov.to(device=target.cov.device, dtype=target.cov.dtype))
            target.count += int(stat.count)
            target.warnings.extend(stat.warnings)

        return hook

    def _mask_for(self, inputs: Tuple[Any, ...]) -> Optional[torch.Tensor]:
        batch = self._current_batch or {}
        mask, diag = token_scope_mask_from_batch(batch, self.token_scope, mask_fallback=self.mask_fallback)
        self._log_scope_selection(diag)
        if diag.get("fallback_used"):
            LOG.warning("[ENGRAM] token_scope=%s fallback_used=true reason=%s", self.token_scope, diag.get("fallback_reason"))
        return mask

    def _log_scope_selection(self, diag: Dict[str, Any]) -> None:
        key = repr(
            (
                diag.get("token_scope"),
                diag.get("raw_label_answer_positions"),
                diag.get("shifted_loss_positions"),
                diag.get("num_selected_tokens"),
                diag.get("fallback_used"),
            )
        )
        if key in self._logged_scope_keys:
            return
        self._logged_scope_keys.add(key)
        self.selection_logs.append(dict(diag))
        LOG.info(
            "[ENGRAM] token_scope=%s raw_label_answer_positions=%s shifted_loss_positions=%s "
            "num_selected_tokens=%s fallback_used=%s",
            diag.get("token_scope"),
            diag.get("raw_label_answer_positions"),
            diag.get("shifted_loss_positions"),
            diag.get("num_selected_tokens"),
            diag.get("fallback_used"),
        )

    def _log_once(self, layer_name: str, message: str) -> None:
        key = f"{layer_name}:{message}"
        if key in self._warned:
            return
        self._warned.add(key)
        LOG.warning("[ENGRAM] %s: %s", layer_name, message)
