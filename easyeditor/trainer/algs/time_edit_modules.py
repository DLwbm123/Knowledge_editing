from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def choose_factor_shape(hidden_size: int) -> Tuple[int, int]:
    hidden_size = int(hidden_size)
    root = int(math.isqrt(hidden_size))
    for s1 in range(root, 0, -1):
        if hidden_size % s1 == 0:
            return s1, hidden_size // s1
    return 1, hidden_size


def scale_from_mode(mode: str, alpha: float, rank: int) -> float:
    mode = str(mode)
    alpha = float(alpha)
    rank = max(1, int(rank))
    if mode == "lora_like":
        return alpha / math.sqrt(rank)
    if mode == "paper_inverse":
        if alpha == 0:
            raise ValueError("paper_inverse scale_mode requires alpha != 0.")
        return 1.0 / (alpha * math.sqrt(rank))
    if mode == "none":
        return 1.0
    raise ValueError(f"Unsupported TIME scale_mode: {mode}")


def apply_activation(value: torch.Tensor, activation: str) -> torch.Tensor:
    activation = str(activation).lower()
    if activation == "gelu":
        return F.gelu(value)
    if activation == "relu":
        return F.relu(value)
    if activation in {"silu", "swish"}:
        return F.silu(value)
    if activation in {"identity", "linear", "none"}:
        return value
    raise ValueError(f"Unsupported TIME activation: {activation}")


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
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


class TIMEExpert(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        rank: int,
        s1: Optional[int] = None,
        s2: Optional[int] = None,
        init_std: float = 1.0e-3,
        factors: Optional[Dict[str, torch.Tensor]] = None,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.rank = int(rank)
        self.s1, self.s2 = choose_factor_shape(self.hidden_size) if s1 is None or s2 is None else (int(s1), int(s2))
        if self.s1 * self.s2 != self.hidden_size:
            raise ValueError(f"TIME factor shape mismatch: {self.s1} * {self.s2} != {self.hidden_size}")
        if factors is None:
            self.U_in = nn.Parameter(torch.randn(self.rank, self.s1, device=device, dtype=dtype) * float(init_std))
            self.V_in = nn.Parameter(torch.randn(self.rank, self.s2, device=device, dtype=dtype) * float(init_std))
            self.U_out = nn.Parameter(torch.randn(self.rank, self.s1, device=device, dtype=dtype) * float(init_std))
            self.V_out = nn.Parameter(torch.randn(self.rank, self.s2, device=device, dtype=dtype) * float(init_std))
        else:
            self.U_in = nn.Parameter(self._factor_tensor(factors["U_in"], self.rank, self.s1, device, dtype))
            self.V_in = nn.Parameter(self._factor_tensor(factors["V_in"], self.rank, self.s2, device, dtype))
            self.U_out = nn.Parameter(self._factor_tensor(factors["U_out"], self.rank, self.s1, device, dtype))
            self.V_out = nn.Parameter(self._factor_tensor(factors["V_out"], self.rank, self.s2, device, dtype))

    @staticmethod
    def _factor_tensor(
        value: torch.Tensor,
        rank: int,
        width: int,
        device: Optional[torch.device],
        dtype: torch.dtype,
    ) -> torch.Tensor:
        tensor = value.detach().clone().to(device=device, dtype=dtype)
        if tuple(tensor.shape) != (rank, width):
            raise ValueError(f"Expected TIME factor shape {(rank, width)}, got {tuple(tensor.shape)}")
        return tensor

    def factors_dict(self, detach: bool = False, cpu: bool = False) -> Dict[str, torch.Tensor]:
        result = {
            "U_in": self.U_in,
            "V_in": self.V_in,
            "U_out": self.U_out,
            "V_out": self.V_out,
        }
        if detach:
            result = {name: value.detach() for name, value in result.items()}
        if cpu:
            result = {name: value.cpu() for name, value in result.items()}
        return result


class TIMEExpertRepository(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        rank: int = 4,
        s1: Optional[int] = None,
        s2: Optional[int] = None,
        init_std: float = 1.0e-3,
        target_layer: int = 21,
        alpha: float = 0.1,
        gamma: float = 0.5,
        tau: float = 1.0,
        scale_mode: str = "lora_like",
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.s1, self.s2 = choose_factor_shape(self.hidden_size) if s1 is None or s2 is None else (int(s1), int(s2))
        if self.s1 * self.s2 != self.hidden_size:
            raise ValueError(f"TIME factor shape mismatch: {self.s1} * {self.s2} != {self.hidden_size}")
        self.rank = int(rank)
        self.init_std = float(init_std)
        self.target_layer = int(target_layer)
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.tau = float(tau)
        self.scale_mode = str(scale_mode)
        self.activation = str(activation)
        self.experts = nn.ModuleList()
        self.metadata: List[Dict[str, Any]] = []

    @property
    def num_experts(self) -> int:
        return len(self.experts)

    def __len__(self) -> int:
        return len(self.experts)

    def add_expert(
        self,
        record_id: Any,
        factors: Optional[Dict[str, torch.Tensor]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        device: Optional[torch.device] = None,
    ) -> int:
        if device is None:
            try:
                device = next(self.parameters()).device
            except StopIteration:
                device = torch.device("cpu")
        expert = TIMEExpert(
            hidden_size=self.hidden_size,
            rank=self.rank,
            s1=self.s1,
            s2=self.s2,
            init_std=self.init_std,
            factors=factors,
            device=device,
            dtype=torch.float32,
        )
        self.experts.append(expert)
        edit_index = len(self.experts) - 1
        entry = {
            "record_id": str(record_id),
            "edit_index": edit_index,
            "target_layer": self.target_layer,
            "H": self.hidden_size,
            "s1": self.s1,
            "s2": self.s2,
            "rank": self.rank,
            "alpha": self.alpha,
            "gamma": self.gamma,
            "tau": self.tau,
            "scale_mode": self.scale_mode,
            "activation": self.activation,
            "timestamp": utc_timestamp(),
        }
        if metadata:
            entry.update(_json_safe(metadata))
        self.metadata.append(entry)
        return edit_index

    def delete_last_expert(self) -> Optional[Dict[str, Any]]:
        if not self.experts:
            return None
        del self.experts[-1]
        return self.metadata.pop()

    def freeze_all(self) -> None:
        for expert in self.experts:
            for param in expert.parameters():
                param.requires_grad_(False)

    def unfreeze_expert(self, index: int) -> None:
        if index < 0:
            index += len(self.experts)
        for expert_id, expert in enumerate(self.experts):
            enabled = expert_id == index
            for param in expert.parameters():
                param.requires_grad_(enabled)

    def trainable_parameters(self) -> List[nn.Parameter]:
        return [param for expert in self.experts for param in expert.parameters() if param.requires_grad]

    def current_parameters(self) -> List[nn.Parameter]:
        if not self.experts:
            return []
        return list(self.experts[-1].parameters())

    def previous_parameters(self) -> List[nn.Parameter]:
        params: List[nn.Parameter] = []
        for expert in list(self.experts)[:-1]:
            params.extend(list(expert.parameters()))
        return params

    def get_factors(self, detach: bool = False) -> Dict[str, torch.Tensor]:
        if not self.experts:
            empty_rank_s1 = torch.empty(0, self.rank, self.s1)
            empty_rank_s2 = torch.empty(0, self.rank, self.s2)
            return {
                "U_in": empty_rank_s1,
                "V_in": empty_rank_s2,
                "U_out": empty_rank_s1.clone(),
                "V_out": empty_rank_s2.clone(),
            }
        factors = {name: [] for name in ("U_in", "V_in", "U_out", "V_out")}
        for expert in self.experts:
            expert_factors = expert.factors_dict(detach=detach)
            for name in factors:
                factors[name].append(expert_factors[name])
        return {name: torch.stack(values, dim=0) for name, values in factors.items()}

    def state_bundle(self) -> Dict[str, Any]:
        return {
            "schema_version": 1,
            "hidden_size": self.hidden_size,
            "s1": self.s1,
            "s2": self.s2,
            "rank": self.rank,
            "init_std": self.init_std,
            "target_layer": self.target_layer,
            "alpha": self.alpha,
            "gamma": self.gamma,
            "tau": self.tau,
            "scale_mode": self.scale_mode,
            "activation": self.activation,
            "metadata": list(self.metadata),
            "experts": [expert.factors_dict(detach=True, cpu=True) for expert in self.experts],
        }

    def load_state_bundle(self, bundle: Dict[str, Any], device: Optional[torch.device] = None) -> None:
        self.hidden_size = int(bundle["hidden_size"])
        self.s1 = int(bundle["s1"])
        self.s2 = int(bundle["s2"])
        self.rank = int(bundle["rank"])
        self.init_std = float(bundle.get("init_std", self.init_std))
        self.target_layer = int(bundle.get("target_layer", self.target_layer))
        self.alpha = float(bundle.get("alpha", self.alpha))
        self.gamma = float(bundle.get("gamma", self.gamma))
        self.tau = float(bundle.get("tau", self.tau))
        self.scale_mode = str(bundle.get("scale_mode", self.scale_mode))
        self.activation = str(bundle.get("activation", self.activation))
        self.experts = nn.ModuleList()
        self.metadata = [dict(item) for item in bundle.get("metadata", [])]
        if device is None:
            device = torch.device("cpu")
        for factors in bundle.get("experts", []):
            self.experts.append(
                TIMEExpert(
                    self.hidden_size,
                    self.rank,
                    self.s1,
                    self.s2,
                    init_std=self.init_std,
                    factors=factors,
                    device=device,
                    dtype=torch.float32,
                )
            )

    def save(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_bundle(), output)

    @classmethod
    def load(cls, path: str | Path, device: Optional[torch.device] = None) -> "TIMEExpertRepository":
        bundle = torch.load(path, map_location=device or "cpu")
        repo = cls(
            hidden_size=int(bundle["hidden_size"]),
            rank=int(bundle["rank"]),
            s1=int(bundle["s1"]),
            s2=int(bundle["s2"]),
            init_std=float(bundle.get("init_std", 1.0e-3)),
            target_layer=int(bundle.get("target_layer", 21)),
            alpha=float(bundle.get("alpha", 0.1)),
            gamma=float(bundle.get("gamma", 0.5)),
            tau=float(bundle.get("tau", 1.0)),
            scale_mode=str(bundle.get("scale_mode", "lora_like")),
            activation=str(bundle.get("activation", "gelu")),
        )
        repo.load_state_bundle(bundle, device=device)
        return repo


@dataclass
class TIMEForwardDebug:
    scores: torch.Tensor
    selected: torch.Tensor
    weights: torch.Tensor
    residual: torch.Tensor
    top_expert_ids: torch.Tensor
    top_scores: torch.Tensor
    selected_counts: torch.Tensor


class TIMECPResidual(nn.Module):
    def __init__(
        self,
        repository: TIMEExpertRepository,
        disable_selection: bool = False,
        disable_score_mixing: bool = False,
        topk: int = 0,
        routing_mode: str = "threshold",
        layer_norm: bool = True,
    ) -> None:
        super().__init__()
        self.repository = repository
        self.disable_selection = bool(disable_selection)
        self.disable_score_mixing = bool(disable_score_mixing)
        self.topk = int(topk or 0)
        self.routing_mode = str(routing_mode or "threshold").lower()
        self.layer_norm = nn.LayerNorm(repository.hidden_size, elementwise_affine=False) if layer_norm else nn.Identity()

    def _topk_mask(self, scores: torch.Tensor, candidate: Optional[torch.Tensor] = None) -> torch.Tensor:
        k = min(max(1, int(self.topk or 1)), scores.shape[-1])
        masked = scores if candidate is None else scores.masked_fill(~candidate, float("-inf"))
        top_values, top_indices = torch.topk(masked, k=k, dim=-1)
        top_valid = torch.isfinite(top_values)
        selected = torch.zeros_like(scores, dtype=torch.bool)
        selected.scatter_(-1, top_indices, top_valid)
        return selected

    def _selection(self, scores: torch.Tensor, force_expert_ids: Optional[Iterable[int]]) -> torch.Tensor:
        if self.repository.num_experts == 0:
            return torch.zeros_like(scores, dtype=torch.bool)
        if self.disable_selection:
            selected = torch.ones_like(scores, dtype=torch.bool)
        else:
            mode = str(self.routing_mode or "threshold").lower()
            if mode == "threshold":
                selected = scores > float(self.repository.gamma)
            elif mode == "topk":
                selected = self._topk_mask(scores)
            elif mode == "threshold_topk":
                thresholded = scores > float(self.repository.gamma)
                selected = self._topk_mask(scores, thresholded) if self.topk > 0 else thresholded
            elif mode == "force_current":
                selected = torch.zeros_like(scores, dtype=torch.bool)
                selected[..., self.repository.num_experts - 1] = True
            else:
                raise ValueError(f"Unsupported TIME routing_mode: {self.routing_mode}")
        if force_expert_ids:
            for idx in force_expert_ids:
                idx = int(idx)
                if 0 <= idx < selected.shape[-1]:
                    selected[..., idx] = True
        return selected

    def forward(
        self,
        hidden: torch.Tensor,
        token_mask: Optional[torch.Tensor] = None,
        disable_time: bool = False,
        force_expert_ids: Optional[Iterable[int]] = None,
        return_debug: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, TIMEForwardDebug]:
        if hidden.dim() != 3:
            raise RuntimeError(f"TIME expects hidden states [B,L,H], got {tuple(hidden.shape)}.")
        if hidden.shape[-1] != self.repository.hidden_size:
            raise RuntimeError(f"TIME hidden size mismatch: {hidden.shape[-1]} vs {self.repository.hidden_size}.")
        zero = torch.zeros_like(hidden)
        if disable_time or self.repository.num_experts == 0:
            debug = self._empty_debug(hidden, zero)
            return (zero, debug) if return_debug else zero

        original_dtype = hidden.dtype
        z_tilde = self.layer_norm(hidden.float())
        B, L, H = z_tilde.shape
        Z = z_tilde.reshape(B, L, self.repository.s1, self.repository.s2)
        factors = self.repository.get_factors(detach=False)
        U_in = factors["U_in"].to(device=hidden.device, dtype=torch.float32)
        V_in = factors["V_in"].to(device=hidden.device, dtype=torch.float32)
        U_out = factors["U_out"].to(device=hidden.device, dtype=torch.float32)
        V_out = factors["V_out"].to(device=hidden.device, dtype=torch.float32)

        proj = torch.einsum("blxy,mrx,mry->blmr", Z, U_in, V_in)
        act_proj = apply_activation(proj, self.repository.activation)
        scores = act_proj.abs().sum(dim=-1)
        selected = self._selection(scores, force_expert_ids)

        if token_mask is not None:
            token_mask = token_mask.to(device=hidden.device).bool()
            if token_mask.shape != hidden.shape[:2]:
                raise RuntimeError(f"TIME token_mask shape {tuple(token_mask.shape)} does not match {tuple(hidden.shape[:2])}.")
        selected_float = selected.to(dtype=torch.float32)
        if self.disable_score_mixing:
            denom = selected_float.sum(dim=-1, keepdim=True).clamp_min(1.0)
            weights = selected_float / denom
        else:
            safe_tau = max(float(self.repository.tau), 1.0e-8)
            logits = scores / safe_tau
            logits = logits.masked_fill(~selected, -1.0e30)
            weights = torch.softmax(logits, dim=-1) * selected_float
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1.0e-12)
        weights = torch.where(selected, weights, torch.zeros_like(weights))

        output_basis = torch.einsum("mrx,mry->mrxy", U_out, V_out).reshape(self.repository.num_experts, self.repository.rank, H)
        scale = scale_from_mode(self.repository.scale_mode, self.repository.alpha, self.repository.rank)
        expert_residual = float(scale) * torch.einsum("blmr,mrh->blmh", act_proj, output_basis)
        residual = torch.einsum("blm,blmh->blh", weights.to(dtype=torch.float32), expert_residual)
        if token_mask is not None:
            residual = residual * token_mask.unsqueeze(-1).to(dtype=residual.dtype)
        residual = residual.to(dtype=original_dtype)

        top_scores, top_expert_ids = scores.max(dim=-1)
        selected_counts = selected.sum(dim=-1)
        debug = TIMEForwardDebug(
            scores=scores,
            selected=selected,
            weights=weights,
            residual=residual,
            top_expert_ids=top_expert_ids,
            top_scores=top_scores,
            selected_counts=selected_counts,
        )
        return (residual, debug) if return_debug else residual

    def _empty_debug(self, hidden: torch.Tensor, residual: torch.Tensor) -> TIMEForwardDebug:
        scores = torch.zeros(*hidden.shape[:2], self.repository.num_experts, device=hidden.device, dtype=torch.float32)
        selected = torch.zeros_like(scores, dtype=torch.bool)
        weights = torch.zeros_like(scores)
        top_ids = torch.full(hidden.shape[:2], -1, device=hidden.device, dtype=torch.long)
        top_scores = torch.zeros(hidden.shape[:2], device=hidden.device, dtype=torch.float32)
        selected_counts = torch.zeros(hidden.shape[:2], device=hidden.device, dtype=torch.long)
        return TIMEForwardDebug(scores, selected, weights, residual, top_ids, top_scores, selected_counts)


def extract_first_tensor(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)):
        return output[0]
    if isinstance(output, dict):
        key = "last_hidden_state" if "last_hidden_state" in output else next(iter(output))
        return output[key]
    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state
    raise RuntimeError(f"Unsupported TIME hook output type: {type(output)}")


def replace_first_tensor(output: Any, new_tensor: torch.Tensor) -> Any:
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
        output.last_hidden_state = new_tensor
        return output
    raise RuntimeError(f"Unsupported TIME hook output type: {type(output)}")


def find_module_by_path(model: nn.Module, module_path: str) -> nn.Module:
    current: Any = model
    for part in module_path.split("."):
        try:
            current = current[int(part)] if part.isdigit() else getattr(current, part)
        except (AttributeError, IndexError, KeyError, TypeError) as exc:
            raise ValueError(f"TIME path `{module_path}` could not resolve component `{part}`.") from exc
    if not isinstance(current, nn.Module):
        raise ValueError(f"TIME path `{module_path}` did not resolve to a module.")
    return current


def time_memory_estimate(hidden_size: int, rank: int, s1: int, s2: int, dtype_bytes: int = 4) -> Dict[str, float]:
    time_params = int(rank) * (2 * int(s1) + 2 * int(s2))
    lora_params = 2 * int(hidden_size) * int(rank)
    return {
        "time_params_per_expert": float(time_params),
        "time_bytes_per_expert": float(time_params * dtype_bytes),
        "lora_params_per_expert": float(lora_params),
        "lora_bytes_per_expert": float(lora_params * dtype_bytes),
        "time_vs_lora_param_ratio": float(time_params / max(1, lora_params)),
    }
