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
    raw_scores: torch.Tensor
    score_variants: Dict[str, torch.Tensor]
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
        residual_sign: str = "plus",
        expert_gain: float = 1.0,
        score_norm: str = "none",
        relative_threshold: Optional[float] = None,
        mixing_mode: Optional[str] = None,
        calibration_mode: str = "none",
        calibration_beta: float = 0.0,
        max_selected_experts: Optional[int] = None,
        score_pool: str = "token",
        score_eps: float = 1.0e-8,
        layer_norm: bool = True,
    ) -> None:
        super().__init__()
        self.repository = repository
        self.disable_selection = bool(disable_selection)
        self.disable_score_mixing = bool(disable_score_mixing)
        self.topk = int(topk or 0)
        self.routing_mode = str(routing_mode or "threshold").lower()
        self.residual_sign = str(residual_sign or "plus").lower()
        self.expert_gain = float(expert_gain)
        self.score_norm = str(score_norm or "none").lower()
        self.relative_threshold = None if relative_threshold is None else float(relative_threshold)
        self.calibration_mode = str(calibration_mode or "none").lower()
        self.calibration_beta = float(calibration_beta or 0.0)
        self.max_selected_experts = None if max_selected_experts is None else int(max_selected_experts)
        self.score_pool = str(score_pool or "token").lower()
        self.calibration_stats: Dict[str, List[float]] = {}
        self.score_eps = float(score_eps)
        self.self_score_cache: Dict[str, List[float]] = {}
        requested_mixing = str(mixing_mode or "softmax").lower()
        if self.disable_score_mixing:
            requested_mixing = "average"
        self.mixing_mode = requested_mixing
        if self.residual_sign not in {"plus", "minus"}:
            raise ValueError(f"Unsupported TIME residual_sign: {self.residual_sign}")
        if self.score_norm not in {"none", "factor", "factor_z", "self_score", "factor_self_score"}:
            raise ValueError(f"Unsupported TIME score_norm: {self.score_norm}")
        if self.mixing_mode not in {"softmax", "average", "own_oracle"}:
            raise ValueError(f"Unsupported TIME mixing_mode: {self.mixing_mode}")
        if self.calibration_mode not in {"none", "self_ratio", "zscore_neg", "neg_margin", "self_minus_neg_mean"}:
            raise ValueError(f"Unsupported TIME calibration_mode: {self.calibration_mode}")
        if self.score_pool not in {"token", "mean", "max", "last", "answer_mean"}:
            raise ValueError(f"Unsupported TIME score_pool: {self.score_pool}")
        self.layer_norm = nn.LayerNorm(repository.hidden_size, elementwise_affine=False) if layer_norm else nn.Identity()

    def _topk_mask(self, scores: torch.Tensor, candidate: Optional[torch.Tensor] = None) -> torch.Tensor:
        k = min(max(1, int(self.topk or 1)), scores.shape[-1])
        masked = scores if candidate is None else scores.masked_fill(~candidate, float("-inf"))
        top_values, top_indices = torch.topk(masked, k=k, dim=-1)
        top_valid = torch.isfinite(top_values)
        selected = torch.zeros_like(scores, dtype=torch.bool)
        selected.scatter_(-1, top_indices, top_valid)
        return selected

    def _relative_threshold_mask(self, scores: torch.Tensor) -> torch.Tensor:
        threshold = 1.0 if self.relative_threshold is None else float(self.relative_threshold)
        max_scores = scores.max(dim=-1, keepdim=True).values
        return scores >= (threshold * max_scores)

    def _self_score_values(self, base_norm: str, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        keys = {
            "none": ("time_self_score_none", "time_self_score_raw", "self_score_none", "self_score_raw", "time_self_score"),
            "factor": ("time_self_score_factor", "self_score_factor"),
            "factor_z": ("time_self_score_factor_z", "self_score_factor_z"),
            "self_score": ("time_self_score_unit", "self_score_unit"),
            "factor_self_score": ("time_self_score_unit", "self_score_unit"),
        }.get(base_norm, ("time_self_score",))
        values: List[float] = []
        cached = self.self_score_cache.get(base_norm)
        for idx in range(self.repository.num_experts):
            value = None
            if base_norm in {"self_score", "factor_self_score"}:
                value = 1.0
            if cached is not None and idx < len(cached):
                value = cached[idx]
            if value is None and idx < len(self.repository.metadata):
                metadata = self.repository.metadata[idx]
                for key in keys:
                    if key in metadata:
                        value = metadata[key]
                        break
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                numeric = 1.0
            values.append(max(abs(numeric), self.score_eps))
        if not values:
            values = [1.0]
        return torch.tensor(values, device=device, dtype=dtype)

    def _score_variants(
        self,
        proj: torch.Tensor,
        Z: torch.Tensor,
        U_in: torch.Tensor,
        V_in: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        raw_scores = apply_activation(proj, self.repository.activation).abs().sum(dim=-1)
        factor_den = (
            U_in.norm(dim=-1).clamp_min(self.score_eps)
            * V_in.norm(dim=-1).clamp_min(self.score_eps)
        )
        factor_proj = proj / factor_den.view(1, 1, self.repository.num_experts, self.repository.rank)
        factor_scores = apply_activation(factor_proj, self.repository.activation).abs().sum(dim=-1)
        z_norm = Z.reshape(*Z.shape[:2], -1).norm(dim=-1).clamp_min(self.score_eps)
        factor_z_proj = factor_proj / z_norm.view(*z_norm.shape, 1, 1)
        factor_z_scores = apply_activation(factor_z_proj, self.repository.activation).abs().sum(dim=-1)
        raw_self = self._self_score_values("none", raw_scores.device, raw_scores.dtype).view(1, 1, -1)
        factor_self = self._self_score_values("factor", factor_scores.device, factor_scores.dtype).view(1, 1, -1)
        variants = {
            "none": raw_scores,
            "factor": factor_scores,
            "factor_z": factor_z_scores,
            "self_score": raw_scores / raw_self,
            "factor_self_score": factor_scores / factor_self,
        }
        return raw_scores, variants

    def _calibration_values(
        self,
        name: str,
        device: torch.device,
        dtype: torch.dtype,
        default: float,
    ) -> torch.Tensor:
        values = self.calibration_stats.get(name) or []
        result: List[float] = []
        for idx in range(self.repository.num_experts):
            value = values[idx] if idx < len(values) else default
            try:
                result.append(float(value))
            except (TypeError, ValueError):
                result.append(float(default))
        if not result:
            result = [float(default)]
        return torch.tensor(result, device=device, dtype=dtype).view(1, 1, -1)

    def _calibrated_scores(self, base_scores: torch.Tensor) -> torch.Tensor:
        mode = str(self.calibration_mode or "none")
        if mode == "none":
            return base_scores
        if mode == "self_ratio":
            self_scores = self._calibration_values(
                "self_score",
                base_scores.device,
                base_scores.dtype,
                1.0,
            )
            if "self_score" not in self.calibration_stats:
                self_scores = self._self_score_values(str(self.score_norm), base_scores.device, base_scores.dtype).view(1, 1, -1)
            return base_scores / self_scores.abs().clamp_min(self.score_eps)

        mu = self._calibration_values("mu_neg", base_scores.device, base_scores.dtype, 0.0)
        std = self._calibration_values("std_neg", base_scores.device, base_scores.dtype, 1.0).abs().clamp_min(self.score_eps)
        if mode in {"zscore_neg", "neg_margin"}:
            return (base_scores - mu) / std
        if mode == "self_minus_neg_mean":
            self_scores = self._calibration_values(
                "self_score",
                base_scores.device,
                base_scores.dtype,
                1.0,
            )
            if "self_score" not in self.calibration_stats:
                self_scores = self._self_score_values(str(self.score_norm), base_scores.device, base_scores.dtype).view(1, 1, -1)
            denom = self_scores - mu
            denom = torch.where(
                denom.abs() < self.score_eps,
                torch.full_like(denom, self.score_eps),
                denom,
            )
            return (base_scores - mu) / denom
        return base_scores

    def pool_scores_for_routing(self, scores: torch.Tensor, token_mask: Optional[torch.Tensor]) -> torch.Tensor:
        if scores.numel() == 0:
            return scores.reshape(scores.shape[0], 0)
        mode = str(self.score_pool or "token")
        if mode == "token":
            return scores.mean(dim=1)
        if token_mask is None:
            mask = torch.ones(scores.shape[:2], device=scores.device, dtype=torch.bool)
        else:
            mask = token_mask.to(scores.device).bool()
        if mode in {"mean", "answer_mean"}:
            denom = mask.sum(dim=1, keepdim=True).clamp_min(1).to(scores.dtype)
            return (scores * mask.unsqueeze(-1).to(scores.dtype)).sum(dim=1) / denom
        if mode == "max":
            masked = scores.masked_fill(~mask.unsqueeze(-1), float("-inf"))
            pooled = masked.max(dim=1).values
            return torch.where(torch.isfinite(pooled), pooled, torch.zeros_like(pooled))
        if mode == "last":
            positions = mask.long().sum(dim=1).clamp_min(1) - 1
            batch_index = torch.arange(scores.shape[0], device=scores.device)
            return scores[batch_index, positions]
        raise ValueError(f"Unsupported TIME score_pool: {self.score_pool}")

    def _cap_selection(self, selected: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        cap = self.max_selected_experts
        if cap is None or int(cap) <= 0 or scores.shape[-1] <= int(cap):
            return selected
        k = min(max(1, int(cap)), scores.shape[-1])
        masked = scores.masked_fill(~selected, float("-inf"))
        top_values, top_indices = torch.topk(masked, k=k, dim=-1)
        top_valid = torch.isfinite(top_values)
        capped = torch.zeros_like(selected, dtype=torch.bool)
        capped.scatter_(-1, top_indices, top_valid)
        return selected & capped

    def _selection(self, scores: torch.Tensor, force_expert_ids: Optional[Iterable[int]]) -> torch.Tensor:
        if self.repository.num_experts == 0:
            return torch.zeros_like(scores, dtype=torch.bool)
        if self.disable_selection:
            selected = torch.ones_like(scores, dtype=torch.bool)
        else:
            mode = str(self.routing_mode or "threshold").lower()
            if mode == "threshold":
                threshold = float(self.calibration_beta) if self.calibration_mode == "neg_margin" else float(self.repository.gamma)
                selected = scores > threshold
            elif mode == "topk":
                selected = self._topk_mask(scores)
            elif mode == "threshold_topk":
                thresholded = scores > float(self.repository.gamma)
                selected = self._topk_mask(scores, thresholded) if self.topk > 0 else thresholded
            elif mode == "relative_threshold":
                thresholded = self._relative_threshold_mask(scores)
                selected = self._topk_mask(scores, thresholded) if self.topk > 0 else thresholded
            elif mode == "relative_topk":
                thresholded = self._relative_threshold_mask(scores)
                selected = self._topk_mask(scores, thresholded)
            elif mode == "force_current":
                selected = torch.zeros_like(scores, dtype=torch.bool)
                selected[..., self.repository.num_experts - 1] = True
            else:
                raise ValueError(f"Unsupported TIME routing_mode: {self.routing_mode}")
        selected = self._cap_selection(selected, scores)
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
        raw_scores, score_variants = self._score_variants(proj, Z, U_in, V_in)
        if token_mask is not None:
            token_mask = token_mask.to(device=hidden.device).bool()
            if token_mask.shape != hidden.shape[:2]:
                raise RuntimeError(f"TIME token_mask shape {tuple(token_mask.shape)} does not match {tuple(hidden.shape[:2])}.")
        base_scores = score_variants[str(self.score_norm)]
        calibrated_scores = self._calibrated_scores(base_scores)
        if str(self.score_pool) == "token":
            scores = calibrated_scores
        else:
            scores = self.pool_scores_for_routing(calibrated_scores, token_mask).unsqueeze(1).expand(B, L, self.repository.num_experts)
        selected = self._selection(scores, force_expert_ids)
        selected_float = selected.to(dtype=torch.float32)
        if self.mixing_mode == "average":
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
        residual = residual * float(self.expert_gain)
        if self.residual_sign == "minus":
            residual = -residual
        residual = residual.to(dtype=original_dtype)

        top_scores, top_expert_ids = scores.max(dim=-1)
        selected_counts = selected.sum(dim=-1)
        debug = TIMEForwardDebug(
            scores=scores,
            raw_scores=raw_scores,
            score_variants=score_variants,
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
        return TIMEForwardDebug(scores, scores, {"none": scores}, selected, weights, residual, top_ids, top_scores, selected_counts)


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
