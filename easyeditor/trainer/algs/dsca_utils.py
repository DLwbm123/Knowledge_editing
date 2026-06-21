import copy
import json
import math
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, is_dataclass, replace
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class DSCAPhaseTimer:
    """Low-overhead JSONL phase timer for DSCA profiling."""

    def __init__(
        self,
        enabled: bool = False,
        log_path: Optional[str] = None,
        device: Optional[torch.device] = None,
        step: Optional[int] = None,
    ) -> None:
        self.enabled = bool(enabled and log_path)
        self.log_path = log_path
        self.device = torch.device(device) if device is not None else None
        self.step = step
        self._starts: Dict[str, float] = {}
        if self.enabled and self.log_path:
            directory = os.path.dirname(self.log_path)
            if directory:
                os.makedirs(directory, exist_ok=True)

    def _sync(self) -> None:
        if not self.enabled or self.device is None or self.device.type != "cuda" or not torch.cuda.is_available():
            return
        target = self.device if self.device.index is not None else torch.cuda.current_device()
        torch.cuda.synchronize(target)

    def _memory(self) -> Dict[str, float]:
        if self.device is None or self.device.type != "cuda" or not torch.cuda.is_available():
            return {}
        target = self.device if self.device.index is not None else torch.cuda.current_device()
        return {
            "cuda_memory_allocated_mb": float(torch.cuda.memory_allocated(target) / (1024.0 * 1024.0)),
            "cuda_memory_reserved_mb": float(torch.cuda.memory_reserved(target) / (1024.0 * 1024.0)),
        }

    def record(self, phase: str, event: str = "record", elapsed_sec: Optional[float] = None, extra: Optional[Dict[str, Any]] = None) -> None:
        if not self.enabled or not self.log_path:
            return
        payload: Dict[str, Any] = {
            "step": self.step,
            "phase": phase,
            "event": event,
        }
        if elapsed_sec is not None:
            payload["elapsed_sec"] = float(elapsed_sec)
        payload.update(self._memory())
        if extra:
            payload["extra"] = _json_safe(extra)
        with open(self.log_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")

    def start(self, phase: str, extra: Optional[Dict[str, Any]] = None) -> None:
        if not self.enabled:
            return
        self._sync()
        self._starts[phase] = time.perf_counter()
        self.record(phase, event="start", extra=extra)

    def stop(self, phase: str, extra: Optional[Dict[str, Any]] = None) -> None:
        if not self.enabled:
            return
        self._sync()
        start = self._starts.pop(phase, None)
        elapsed = None if start is None else time.perf_counter() - start
        self.record(phase, event="done", elapsed_sec=elapsed, extra=extra)


def _json_safe(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return _json_safe(value.detach().cpu().item())
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "device": str(value.device),
        }
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)


def _require_mask(mask: Optional[torch.Tensor], name: str, hidden: torch.Tensor) -> torch.Tensor:
    if mask is None:
        raise RuntimeError(f"DSCA requires `{name}` but it was not provided.")
    mask = mask.to(hidden.device).bool()
    if mask.shape != hidden.shape[:2]:
        raise RuntimeError(f"DSCA `{name}` shape {tuple(mask.shape)} does not match hidden {tuple(hidden.shape[:2])}.")
    return mask


def validate_dsca_masks(
    hidden: torch.Tensor,
    masks: Dict[str, Optional[torch.Tensor]],
    require_answer: bool = False,
) -> Dict[str, torch.Tensor]:
    if hidden.dim() != 3:
        raise RuntimeError(f"DSCA hidden states must be [batch, seq, dim], got {tuple(hidden.shape)}.")
    vision_mask = _require_mask(masks.get("vision_mask"), "vision_mask", hidden)
    prompt_mask = _require_mask(masks.get("prompt_mask"), "prompt_mask", hidden)
    attention_mask = _require_mask(masks.get("attention_mask"), "attention_mask", hidden)
    answer_mask = masks.get("answer_mask")
    if answer_mask is not None:
        answer_mask = _require_mask(answer_mask, "answer_mask", hidden)
    elif require_answer:
        raise RuntimeError("DSCA requires `answer_mask` for this operation.")
    else:
        answer_mask = torch.zeros_like(attention_mask)

    if (vision_mask & prompt_mask).any() or (vision_mask & answer_mask).any() or (prompt_mask & answer_mask).any():
        raise RuntimeError("DSCA region masks must be disjoint.")
    if ((vision_mask | prompt_mask | answer_mask) & ~attention_mask).any():
        raise RuntimeError("DSCA region masks include padding positions.")
    for name, mask in (("vision_mask", vision_mask), ("prompt_mask", prompt_mask)):
        empty = mask.sum(dim=1) == 0
        if empty.any():
            rows = empty.nonzero().flatten().tolist()
            raise RuntimeError(f"DSCA `{name}` has no selected tokens for batch rows {rows}.")
    return {
        "vision_mask": vision_mask,
        "prompt_mask": prompt_mask,
        "answer_mask": answer_mask,
        "attention_mask": attention_mask,
    }


def get_dsca_masks_from_output_or_batch(output: Any, batch: Dict[str, Any], hidden: torch.Tensor, require_answer: bool = False):
    masks = {}
    for name in ("vision_mask", "prompt_mask", "answer_mask", "attention_mask"):
        value = None if output is None else getattr(output, name, None)
        if value is None and isinstance(output, dict):
            value = output.get(name)
        if value is None:
            value = batch.get(name)
        masks[name] = value
    return validate_dsca_masks(hidden, masks, require_answer=require_answer)


def masked_mean(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if hidden.dim() != 3 or mask.dim() != 2:
        raise RuntimeError("DSCA masked_mean expects hidden [B,T,D] and mask [B,T].")
    mask = mask.to(hidden.device).bool()
    denom = mask.sum(dim=1, keepdim=True).clamp_min(1).to(hidden.dtype)
    return (hidden * mask.unsqueeze(-1).to(hidden.dtype)).sum(dim=1) / denom


def extract_dsca_region_representations(hidden: torch.Tensor, masks: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    valid = validate_dsca_masks(hidden, masks, require_answer=False)
    vision_mask = valid["vision_mask"]
    prompt_mask = valid["prompt_mask"]
    attention_mask = valid["attention_mask"]
    fused_mask = (vision_mask | prompt_mask) & attention_mask
    return {
        "h_v": masked_mean(hidden, vision_mask),
        "h_t": masked_mean(hidden, prompt_mask),
        "h_f": masked_mean(hidden, fused_mask),
    }


def extract_tensor_from_module_output(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)):
        return output[0]
    if isinstance(output, dict):
        key = "last_hidden_state" if "last_hidden_state" in output else next(iter(output))
        return output[key]
    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state
    raise RuntimeError(f"Unsupported DSCA hook output type: {type(output)}")


def replace_tensor_in_module_output(output: Any, new_tensor: torch.Tensor) -> Any:
    if isinstance(output, torch.Tensor):
        return new_tensor
    if isinstance(output, tuple):
        values = list(output)
        values[0] = new_tensor
        return tuple(values)
    if isinstance(output, list):
        values = list(output)
        values[0] = new_tensor
        return values
    if isinstance(output, dict):
        values = output.copy()
        key = "last_hidden_state" if "last_hidden_state" in values else next(iter(values))
        values[key] = new_tensor
        return values
    if hasattr(output, "last_hidden_state"):
        if is_dataclass(output):
            try:
                return replace(output, last_hidden_state=new_tensor)
            except TypeError:
                pass
        output.last_hidden_state = new_tensor
        return output
    raise RuntimeError(f"Unsupported DSCA hook output type: {type(output)}")


def find_module_by_path(model: nn.Module, module_path: str) -> nn.Module:
    current: Any = model
    for part in module_path.split("."):
        try:
            if part.isdigit():
                current = current[int(part)]
            else:
                current = getattr(current, part)
        except (AttributeError, IndexError, KeyError, TypeError) as exc:
            raise ValueError(f"DSCA path `{module_path}` could not resolve component `{part}`.") from exc
    if not isinstance(current, nn.Module):
        raise ValueError(f"DSCA path `{module_path}` did not resolve to a module.")
    return current


def cosine_matrix(lhs: torch.Tensor, rhs: torch.Tensor, eps: float = 1.0e-8) -> torch.Tensor:
    lhs_n = F.normalize(lhs.float(), dim=-1, eps=eps)
    rhs_n = F.normalize(rhs.float(), dim=-1, eps=eps)
    return lhs_n @ rhs_n.t()


def orthonormalize_rows(rows: torch.Tensor, rank: int, hidden_size: int) -> torch.Tensor:
    output_device = rows.device
    output_dtype = rows.dtype if rows.is_floating_point() else torch.float32
    rows = rows.detach().to(device="cpu", dtype=torch.float32)
    if rows.numel() == 0:
        rows = torch.empty(0, hidden_size, device="cpu", dtype=torch.float32)
    basis: List[torch.Tensor] = []
    for row in rows:
        vec = row.clone()
        for prev in basis:
            vec = vec - (vec @ prev) * prev
        norm = vec.norm()
        if norm > 1.0e-6:
            basis.append(vec / norm)
        if len(basis) == rank:
            break
    if len(basis) == rank:
        return torch.stack(basis, dim=0)[:rank].to(device=output_device, dtype=output_dtype)
    eye = torch.eye(hidden_size, device=rows.device, dtype=rows.dtype)
    for row in eye:
        vec = row.clone()
        for prev in basis:
            vec = vec - (vec @ prev) * prev
        norm = vec.norm()
        if norm > 1.0e-6:
            basis.append(vec / norm)
        if len(basis) == rank:
            break
    if not basis:
        result = torch.zeros(rank, hidden_size, device=rows.device, dtype=rows.dtype)
    else:
        result = torch.stack(basis, dim=0)[:rank]
    return result.to(device=output_device, dtype=output_dtype)


def pca_basis(features: torch.Tensor, rank: int) -> torch.Tensor:
    if features.dim() != 2:
        raise RuntimeError("DSCA PCA features must be [N,D].")
    output_device = features.device
    output_dtype = features.dtype if features.is_floating_point() else torch.float32
    work = features.detach().to(device="cpu", dtype=torch.float32)
    centered = work - work.mean(dim=0, keepdim=True)
    if centered.shape[0] <= 1 or centered.norm().item() == 0.0:
        return orthonormalize_rows(centered[:1], rank, features.shape[1]).to(device=output_device, dtype=output_dtype)
    _, _, vh = torch.linalg.svd(centered, full_matrices=False)
    return orthonormalize_rows(vh[:rank], rank, features.shape[1]).to(device=output_device, dtype=output_dtype)


def residualize_features(features: torch.Tensor, prior_bases: Sequence[torch.Tensor]) -> torch.Tensor:
    residual = features.float()
    for basis in prior_bases:
        if basis is None or basis.numel() == 0:
            continue
        basis = basis.to(residual.device, residual.dtype)
        residual = residual - (residual @ basis.t()) @ basis
    return residual.to(features.dtype)


def residualized_pca_basis(features: torch.Tensor, rank: int, prior_bases: Sequence[torch.Tensor]) -> torch.Tensor:
    output_device = features.device
    output_dtype = features.dtype if features.is_floating_point() else torch.float32
    work_features = features.detach().to(device="cpu", dtype=torch.float32)
    work_bases = [
        basis.detach().to(device="cpu", dtype=torch.float32)
        for basis in prior_bases
        if basis is not None and basis.numel() > 0
    ]
    return pca_basis(residualize_features(work_features, work_bases), rank).to(device=output_device, dtype=output_dtype)


class DSAModule(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        rank: int,
        gate_bottleneck: int,
        init_std: float = 0.02,
        residual_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.rank = int(rank)
        self.residual_scale = float(residual_scale)
        self.W = nn.Parameter(torch.empty(rank, hidden_size))
        self.b = nn.Parameter(torch.zeros(rank))
        self.gate_down = nn.Linear(hidden_size, gate_bottleneck)
        self.gate_up = nn.Linear(gate_bottleneck, hidden_size)
        self.register_buffer("R", torch.zeros(rank, hidden_size))
        self.register_buffer("basis_initialized", torch.tensor(False, dtype=torch.bool))
        self.last_gate: Optional[torch.Tensor] = None
        self.reset_parameters(init_std)

    @property
    def active(self) -> bool:
        return bool(self.basis_initialized.item())

    def reset_parameters(self, init_std: float = 0.02) -> None:
        nn.init.normal_(self.W, mean=0.0, std=init_std)
        nn.init.zeros_(self.b)
        self.gate_down.reset_parameters()
        self.gate_up.reset_parameters()

    def set_basis(self, basis: torch.Tensor) -> None:
        if basis.shape != self.R.shape:
            raise RuntimeError(f"DSCA basis shape {tuple(basis.shape)} does not match {tuple(self.R.shape)}.")
        with torch.no_grad():
            self.R.copy_(orthonormalize_rows(basis, self.rank, self.hidden_size).to(self.R.device, self.R.dtype))
            self.basis_initialized.fill_(True)

    def forward(self, hidden: torch.Tensor, apply_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if not self.active:
            self.last_gate = torch.zeros_like(hidden)
            return torch.zeros_like(hidden)
        if self.residual_scale == 0.0:
            self.last_gate = torch.zeros_like(hidden)
            return torch.zeros_like(hidden)

        work_hidden = torch.nan_to_num(hidden.float(), nan=0.0, posinf=0.0, neginf=0.0)
        work_r = self.R.to(device=hidden.device, dtype=torch.float32)
        work_w = self.W.to(device=hidden.device, dtype=torch.float32)
        work_b = self.b.to(device=hidden.device, dtype=torch.float32)
        current_coords = torch.einsum("btd,rd->btr", work_hidden, work_r)
        target_coords = torch.einsum("btd,rd->btr", work_hidden, work_w) + work_b
        subspace_delta = torch.einsum("btr,rd->btd", target_coords - current_coords, work_r)
        gate_input = work_hidden.to(self.gate_down.weight.dtype)
        gate = torch.sigmoid(self.gate_up(F.gelu(self.gate_down(gate_input)))).float()
        residual = gate * subspace_delta * self.residual_scale
        if apply_mask is not None:
            residual = residual * apply_mask.to(hidden.device).bool().unsqueeze(-1).to(residual.dtype)
        residual = torch.nan_to_num(residual, nan=0.0, posinf=0.0, neginf=0.0)
        if hidden.is_floating_point():
            finfo = torch.finfo(hidden.dtype)
            residual = residual.clamp(min=-finfo.max, max=finfo.max)
        self.last_gate = gate.detach().to(hidden.dtype)
        return residual.to(hidden.dtype)


class DSCAConceptRepository(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        rank: int,
        gate_bottleneck: int,
        cluster_alpha: float = 2.0,
        proto_ema: float = 0.95,
        min_samples: int = 32,
        refine_interval: int = 500,
        max_buffer_size: Optional[int] = None,
        dsam_init_std: float = 0.02,
        residual_scale: float = 1.0,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.rank = int(rank)
        self.gate_bottleneck = int(gate_bottleneck)
        self.cluster_alpha = float(cluster_alpha)
        self.proto_ema = float(proto_ema)
        self.min_samples = int(min_samples)
        self.refine_interval = int(refine_interval)
        self.max_buffer_size = max_buffer_size
        self.dsam_init_std = float(dsam_init_std)
        self.residual_scale = float(residual_scale)
        self.register_buffer("p_f", torch.empty(0, hidden_size, device=device, dtype=dtype))
        self.register_buffer("p_v", torch.empty(0, hidden_size, device=device, dtype=dtype))
        self.register_buffer("distance_mu", torch.empty(0, device=device, dtype=dtype))
        self.register_buffer("distance_m2", torch.empty(0, device=device, dtype=dtype))
        self.register_buffer("distance_count", torch.empty(0, device=device, dtype=torch.long))
        self.register_buffer("assignment_count", torch.empty(0, device=device, dtype=torch.long))
        self.register_buffer("active", torch.empty(0, device=device, dtype=torch.bool))
        self.dsams = nn.ModuleList()
        self.pca_buffers: List[torch.Tensor] = []
        self.metadata: List[Dict[str, Any]] = []

    def __len__(self) -> int:
        return int(self.p_f.shape[0])

    def clear(self) -> None:
        self.p_f = self.p_f[:0]
        self.p_v = self.p_v[:0]
        self.distance_mu = self.distance_mu[:0]
        self.distance_m2 = self.distance_m2[:0]
        self.distance_count = self.distance_count[:0]
        self.assignment_count = self.assignment_count[:0]
        self.active = self.active[:0]
        self.dsams = nn.ModuleList()
        self.pca_buffers = []
        self.metadata = []

    def _new_dsam(self) -> DSAModule:
        return DSAModule(
            self.hidden_size,
            self.rank,
            self.gate_bottleneck,
            init_std=self.dsam_init_std,
            residual_scale=self.residual_scale,
        ).to(device=self.p_f.device, dtype=self.p_f.dtype)

    def create_cluster(self, h_f: torch.Tensor, h_v: torch.Tensor, metadata: Optional[Dict[str, Any]] = None) -> int:
        h_f = h_f.detach().to(self.p_f.device, self.p_f.dtype).flatten()
        h_v = h_v.detach().to(self.p_v.device, self.p_v.dtype).flatten()
        self.p_f = torch.cat([self.p_f, h_f.unsqueeze(0)], dim=0)
        self.p_v = torch.cat([self.p_v, h_v.unsqueeze(0)], dim=0)
        zero = torch.zeros(1, device=self.p_f.device, dtype=self.p_f.dtype)
        self.distance_mu = torch.cat([self.distance_mu, zero], dim=0)
        self.distance_m2 = torch.cat([self.distance_m2, zero], dim=0)
        self.distance_count = torch.cat([self.distance_count, torch.zeros(1, device=self.p_f.device, dtype=torch.long)])
        self.assignment_count = torch.cat([self.assignment_count, torch.ones(1, device=self.p_f.device, dtype=torch.long)])
        self.active = torch.cat([self.active, torch.zeros(1, device=self.p_f.device, dtype=torch.bool)])
        self.dsams.append(self._new_dsam())
        self.pca_buffers.append(h_f.detach().unsqueeze(0))
        self.metadata.append(copy.deepcopy(metadata or {}))
        return len(self) - 1

    def _threshold(self, cluster_id: int) -> torch.Tensor:
        count = int(self.distance_count[cluster_id].item())
        if count == 0:
            return torch.tensor(float("inf"), device=self.p_f.device, dtype=self.p_f.dtype)
        if count == 1:
            floor = torch.tensor(1.0e-6, device=self.p_f.device, dtype=self.p_f.dtype)
            return self.distance_mu[cluster_id] + self.cluster_alpha * torch.maximum(self.distance_mu[cluster_id], floor)
        var = self.distance_m2[cluster_id] / (self.distance_count[cluster_id].to(self.p_f.dtype) - 1.0)
        sigma = torch.sqrt(var.clamp_min(0.0))
        return self.distance_mu[cluster_id] + self.cluster_alpha * sigma

    def update_distance_stats(self, cluster_id: int, distance: torch.Tensor) -> None:
        distance = distance.detach().to(self.distance_mu.device, self.distance_mu.dtype)
        count = self.distance_count[cluster_id].to(self.distance_mu.dtype) + 1.0
        delta = distance - self.distance_mu[cluster_id]
        self.distance_mu[cluster_id] += delta / count
        delta2 = distance - self.distance_mu[cluster_id]
        self.distance_m2[cluster_id] += delta * delta2
        self.distance_count[cluster_id] += 1

    def update_prototype_ema(self, cluster_id: int, h_f: torch.Tensor, h_v: torch.Tensor) -> None:
        h_f = h_f.detach().to(self.p_f.device, self.p_f.dtype)
        h_v = h_v.detach().to(self.p_v.device, self.p_v.dtype)
        self.p_f[cluster_id] = self.proto_ema * self.p_f[cluster_id] + (1.0 - self.proto_ema) * h_f
        self.p_v[cluster_id] = self.proto_ema * self.p_v[cluster_id] + (1.0 - self.proto_ema) * h_v
        self.assignment_count[cluster_id] += 1

    def append_to_buffer(self, cluster_id: int, h_f: torch.Tensor) -> None:
        item = h_f.detach().to(self.p_f.device, self.p_f.dtype).flatten().unsqueeze(0)
        buf = torch.cat([self.pca_buffers[cluster_id].to(item.device, item.dtype), item], dim=0)
        if self.max_buffer_size is not None and buf.shape[0] > self.max_buffer_size:
            buf = buf[-self.max_buffer_size :]
        self.pca_buffers[cluster_id] = buf

    def assign_or_create(
        self,
        h_f: torch.Tensor,
        h_v: torch.Tensor,
        metadata: Optional[Dict[str, Any]] = None,
        timer: Optional[DSCAPhaseTimer] = None,
    ) -> Tuple[int, bool]:
        h_f = h_f.detach().to(self.p_f.device, self.p_f.dtype).flatten()
        h_v = h_v.detach().to(self.p_v.device, self.p_v.dtype).flatten()
        if len(self) == 0:
            if timer:
                timer.start("create_cluster", {"reason": "empty_repository", "cluster_count": len(self)})
            new_id = self.create_cluster(h_f, h_v, metadata)
            if timer:
                timer.stop("create_cluster", {"cluster_id": new_id, "cluster_count": len(self)})
            return new_id, True
        if timer:
            timer.start("assign_or_create_cluster", {"cluster_count": len(self), "hidden_size": self.hidden_size})
        distances = torch.norm(self.p_f - h_f.unsqueeze(0), dim=1)
        cluster_id = int(torch.argmin(distances).item())
        distance = distances[cluster_id]
        if distance > self._threshold(cluster_id):
            if timer:
                timer.stop("assign_or_create_cluster", {"cluster_id": cluster_id, "created": True})
                timer.start("create_cluster", {"reason": "distance_threshold", "cluster_count": len(self)})
            new_id = self.create_cluster(h_f, h_v, metadata)
            if timer:
                timer.stop("create_cluster", {"cluster_id": new_id, "cluster_count": len(self)})
            return new_id, True
        if timer:
            timer.start("prototype_ema_update", {"cluster_id": cluster_id})
        self.update_prototype_ema(cluster_id, h_f, h_v)
        if timer:
            timer.stop("prototype_ema_update", {"cluster_id": cluster_id})
        self.update_distance_stats(cluster_id, distance)
        if timer:
            timer.start("append_pca_buffer", {"cluster_id": cluster_id, "old_buffer_size": int(self.pca_buffers[cluster_id].shape[0])})
        self.append_to_buffer(cluster_id, h_f)
        if timer:
            timer.stop("append_pca_buffer", {"cluster_id": cluster_id, "new_buffer_size": int(self.pca_buffers[cluster_id].shape[0])})
            timer.stop("assign_or_create_cluster", {"cluster_id": cluster_id, "created": False})
        return cluster_id, False

    def assign_batch(
        self,
        h_f: torch.Tensor,
        h_v: torch.Tensor,
        metadata: Optional[Iterable[Dict[str, Any]]] = None,
        timer: Optional[DSCAPhaseTimer] = None,
        initialize_basis: bool = True,
    ) -> Tuple[List[int], int]:
        ids: List[int] = []
        created = 0
        metadata_list = list(metadata) if metadata is not None else [{} for _ in range(h_f.shape[0])]
        for row, (hf, hv) in enumerate(zip(h_f, h_v)):
            cid, is_new = self.assign_or_create(hf, hv, metadata_list[row] if row < len(metadata_list) else None, timer=timer)
            ids.append(cid)
            created += int(is_new)
            if initialize_basis:
                self.initialize_basis_if_ready(cid, timer=timer, reason="assign_batch")
        return ids, created

    def _prior_active_bases(self, cluster_id: int) -> List[torch.Tensor]:
        return [self.dsams[i].R.detach() for i in range(cluster_id) if bool(self.active[i].item())]

    def initialize_basis_if_ready(
        self,
        cluster_id: int,
        timer: Optional[DSCAPhaseTimer] = None,
        force: bool = False,
        reason: str = "initialize",
    ) -> bool:
        if cluster_id >= len(self) or self.pca_buffers[cluster_id].shape[0] < self.min_samples:
            return False
        already_active = bool(self.active[cluster_id].item())
        if already_active and not force:
            if timer:
                timer.start(
                    "initialize_basis_if_ready",
                    {
                        "cluster_id": cluster_id,
                        "force": force,
                        "reason": reason,
                        "already_active": already_active,
                        "buffer_size": int(self.pca_buffers[cluster_id].shape[0]),
                        "rank": self.rank,
                        "hidden_size": self.hidden_size,
                    },
                )
                timer.stop(
                    "initialize_basis_if_ready",
                    {"cluster_id": cluster_id, "active": True, "skipped": "already_active"},
                )
            return False
        if timer:
            timer.start(
                "initialize_basis_if_ready",
                {
                    "cluster_id": cluster_id,
                    "force": force,
                    "reason": reason,
                    "already_active": already_active,
                    "buffer_size": int(self.pca_buffers[cluster_id].shape[0]),
                    "rank": self.rank,
                    "hidden_size": self.hidden_size,
                },
            )
            timer.start(
                "residualized_pca",
                {
                    "cluster_id": cluster_id,
                    "buffer_shape": list(self.pca_buffers[cluster_id].shape),
                    "rank": self.rank,
                    "device": str(self.pca_buffers[cluster_id].device),
                    "dtype": str(self.pca_buffers[cluster_id].dtype),
                    "prior_active_bases": len(self._prior_active_bases(cluster_id)),
                },
            )
        basis = residualized_pca_basis(self.pca_buffers[cluster_id], self.rank, self._prior_active_bases(cluster_id))
        if timer:
            timer.stop("residualized_pca", {"cluster_id": cluster_id, "basis_shape": list(basis.shape)})
            timer.start("set_basis", {"cluster_id": cluster_id, "basis_shape": list(basis.shape)})
        self.dsams[cluster_id].set_basis(basis.to(self.p_f.device, self.p_f.dtype))
        if timer:
            timer.stop("set_basis", {"cluster_id": cluster_id})
        self.active[cluster_id] = True
        if timer:
            timer.stop("initialize_basis_if_ready", {"cluster_id": cluster_id, "active": True})
        return True

    def refine_subspaces_if_due(self, global_step: int, timer: Optional[DSCAPhaseTimer] = None) -> int:
        if self.refine_interval <= 0 or global_step % self.refine_interval != 0:
            return 0
        if timer:
            timer.start(
                "refine_subspaces",
                {
                    "global_step": global_step,
                    "refine_interval": self.refine_interval,
                    "cluster_count": len(self),
                    "pca_buffer_sizes": [int(buf.shape[0]) for buf in self.pca_buffers],
                },
            )
        activated = 0
        for idx in range(len(self)):
            activated += int(self.initialize_basis_if_ready(idx, timer=timer, force=True, reason="refine_subspaces"))
        if timer:
            timer.stop("refine_subspaces", {"activated": activated, "active_count": self.num_active()})
        return activated

    def num_active(self) -> int:
        return int(self.active.sum().item())

    def mean_subspace_overlap(self) -> torch.Tensor:
        bases = [self.dsams[i].R for i in range(len(self)) if bool(self.active[i].item())]
        if len(bases) < 2:
            return torch.tensor(0.0, device=self.p_f.device, dtype=self.p_f.dtype)
        vals = []
        for i in range(len(bases)):
            for j in range(len(bases)):
                if i != j:
                    vals.append((bases[i] @ bases[j].t()).pow(2).sum())
        return torch.stack(vals).mean()

    def _apply(self, fn):
        super()._apply(fn)
        self.pca_buffers = [fn(buf) for buf in self.pca_buffers]
        return self

    def get_extra_state(self) -> Dict[str, Any]:
        return {
            "hidden_size": self.hidden_size,
            "rank": self.rank,
            "gate_bottleneck": self.gate_bottleneck,
            "cluster_alpha": self.cluster_alpha,
            "proto_ema": self.proto_ema,
            "min_samples": self.min_samples,
            "refine_interval": self.refine_interval,
            "max_buffer_size": self.max_buffer_size,
            "dsam_init_std": self.dsam_init_std,
            "residual_scale": self.residual_scale,
            "pca_buffers": [buf.detach().cpu() for buf in self.pca_buffers],
            "metadata": copy.deepcopy(self.metadata),
            "num_clusters": len(self),
        }

    def set_extra_state(self, state: Dict[str, Any]) -> None:
        if not state:
            return
        self.pca_buffers = [buf.detach().to(self.p_f.device, self.p_f.dtype) for buf in state.get("pca_buffers", [])]
        self.metadata = copy.deepcopy(state.get("metadata", [{} for _ in range(len(self))]))

    def _ensure_cluster_count(self, count: int) -> None:
        while len(self.dsams) < count:
            self.dsams.append(self._new_dsam())
        while len(self.pca_buffers) < count:
            self.pca_buffers.append(torch.empty(0, self.hidden_size, device=self.p_f.device, dtype=self.p_f.dtype))
        while len(self.metadata) < count:
            self.metadata.append({})

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs) -> None:
        for name in ("p_f", "p_v", "distance_mu", "distance_m2", "distance_count", "assignment_count", "active"):
            key = prefix + name
            if key in state_dict:
                setattr(self, name, torch.empty_like(state_dict[key]))
        count = 0
        for key in state_dict:
            if key.startswith(prefix + "dsams."):
                rest = key[len(prefix + "dsams.") :]
                idx = rest.split(".", 1)[0]
                if idx.isdigit():
                    count = max(count, int(idx) + 1)
        if count == 0 and prefix + "p_f" in state_dict:
            count = int(state_dict[prefix + "p_f"].shape[0])
        self._ensure_cluster_count(count)
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

    def save(self, path: str) -> None:
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        torch.save({"state_dict": self.state_dict()}, path)

    @classmethod
    def load(cls, path: str, map_location: Optional[str] = None) -> "DSCAConceptRepository":
        if not os.path.isfile(path):
            raise FileNotFoundError(f"DSCA repository not found: {path}")
        payload = torch.load(path, map_location=map_location or "cpu")
        state = payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload
        if not isinstance(state, dict) or "p_f" not in state:
            raise RuntimeError(f"DSCA repository at {path} is invalid.")
        extra = state.get("_extra_state", {})
        repo = cls(
            hidden_size=extra.get("hidden_size", state["p_f"].shape[1]),
            rank=extra.get("rank", state.get("dsams.0.R", torch.empty(1, state["p_f"].shape[1])).shape[0]),
            gate_bottleneck=extra.get("gate_bottleneck", 64),
            cluster_alpha=extra.get("cluster_alpha", 2.0),
            proto_ema=extra.get("proto_ema", 0.95),
            min_samples=extra.get("min_samples", 32),
            refine_interval=extra.get("refine_interval", 500),
            max_buffer_size=extra.get("max_buffer_size", None),
            dsam_init_std=extra.get("dsam_init_std", 0.02),
            residual_scale=extra.get("residual_scale", 1.0),
            dtype=state["p_f"].dtype,
        )
        repo.load_state_dict(state)
        return repo


def dsca_route(
    h_v: torch.Tensor,
    h_f: torch.Tensor,
    repository: DSCAConceptRepository,
    tau_visual: float,
    route_temperature: float,
    candidate_topk: Optional[int] = None,
    timer: Optional[DSCAPhaseTimer] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    batch_size = h_v.shape[0]
    cluster_count = len(repository)
    if cluster_count == 0:
        weights = torch.zeros(batch_size, 0, dtype=h_v.dtype, device=h_v.device)
        selected = torch.zeros(batch_size, 0, dtype=torch.bool, device=h_v.device)
        return weights, selected, {"visual_sim": weights, "fused_sim": weights}
    active = repository.active.to(h_v.device).bool()
    if timer:
        timer.start(
            "find_visual_candidates",
            {
                "cluster_count": cluster_count,
                "active_count": int(active.sum().item()),
                "candidate_topk": candidate_topk,
                "tau_visual": tau_visual,
            },
        )
    visual_sim = cosine_matrix(h_v, repository.p_v.to(h_v.device, h_v.dtype)).to(h_v.dtype)
    selected = (visual_sim > float(tau_visual)) & active.unsqueeze(0)
    if candidate_topk is not None and candidate_topk > 0 and cluster_count > candidate_topk:
        topk = torch.topk(visual_sim, k=candidate_topk, dim=1).indices
        topk_mask = torch.zeros_like(selected)
        topk_mask.scatter_(1, topk, True)
        selected = selected & topk_mask
    if timer:
        timer.stop(
            "find_visual_candidates",
            {
                "selected_counts": selected.sum(dim=1),
                "visual_sim_shape": list(visual_sim.shape),
            },
        )
        timer.start("fused_soft_routing", {"route_temperature": route_temperature})
    fused_sim = cosine_matrix(h_f, repository.p_f.to(h_f.device, h_f.dtype)).to(h_f.dtype)
    temp = max(float(route_temperature), 1.0e-8)
    masked = (fused_sim / temp).masked_fill(~selected, torch.finfo(fused_sim.dtype).min)
    rel = torch.softmax(masked.float(), dim=1).to(h_f.dtype)
    weights = torch.where(selected, rel, torch.zeros_like(rel))
    if timer:
        timer.stop(
            "fused_soft_routing",
            {
                "fused_sim_shape": list(fused_sim.shape),
                "nonzero_weights": int(torch.count_nonzero(weights).item()),
            },
        )
    return weights, selected, {"visual_sim": visual_sim, "fused_sim": fused_sim}


def dsca_sparse_loss(weights: torch.Tensor) -> torch.Tensor:
    if weights.numel() == 0:
        return torch.tensor(0.0, device=weights.device, dtype=weights.dtype)
    return weights.abs().mean()


def dsca_contrastive_distill_loss(edited: torch.Tensor, teacher: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    if edited.shape[0] == 0:
        return edited.sum() * 0.0
    logits = cosine_matrix(edited, teacher.detach()) / max(float(temperature), 1.0e-8)
    target = torch.arange(edited.shape[0], device=edited.device)
    return F.cross_entropy(logits, target)


@dataclass
class DSCAContext:
    batch: Optional[Dict[str, Any]] = None
    enabled: bool = True
    capture_only: bool = False
    update_clusters: bool = False
    timer: Optional[DSCAPhaseTimer] = None
    is_generation: bool = False
    debug_events: Optional[List[Dict[str, Any]]] = None
    sample_id: Optional[int] = None
    call_label: Optional[str] = None
    generation_mode: Optional[str] = None
    generation_reuse_prefill_route: Optional[bool] = None
    generation_use_cache: Optional[bool] = None
    force_route_ids: Optional[List[int]] = None
    disable_normal_routing: bool = False
    residual_apply_mask_mode: Optional[str] = None
    extend_generation_masks: bool = False


@contextmanager
def dsca_intervention_context(obj: Any, context: DSCAContext):
    old_context = getattr(obj, "_dsca_context", None)
    obj._dsca_context = context
    try:
        yield
    finally:
        obj._dsca_context = old_context
