from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from .covariance import as_device


@dataclass
class SolverConfig:
    method: str = "pinv"
    rcond: float = 1.0e-6
    svd_rank: Optional[int] = None
    energy_threshold: Optional[float] = None
    solve_device: Any = "cpu"
    normalize_covariance: bool = False
    jitter: float = 0.0


@dataclass
class EngramLayerUpdate:
    module_name: str
    weight: torch.Tensor
    bias: Optional[torch.Tensor] = None
    projector: Optional[torch.Tensor] = None
    delta_safe_weight: Optional[torch.Tensor] = None
    delta_safe_bias: Optional[torch.Tensor] = None
    alpha: float = 1.0
    beta: float = 0.0
    engram_update_direction: str = "subtract"
    direction_sign: int = -1
    behavior_objective: Optional[str] = None
    paper_direction_equivalent: str = "paper_subtract"
    stats: Dict[str, Any] = field(default_factory=dict)


def normalize_update_direction(value: Optional[str]) -> str:
    direction = str(value or "subtract").strip().lower()
    if direction not in {"subtract", "add", "auto_nll"}:
        raise ValueError(
            "engram_update_direction must be one of {'subtract', 'add', 'auto_nll'}, "
            f"got {value!r}"
        )
    return direction


def direction_sign_for_update(value: Optional[str]) -> int:
    direction = normalize_update_direction(value)
    if direction == "subtract":
        return -1
    if direction == "add":
        return 1
    raise NotImplementedError(
        "engram_update_direction='auto_nll' is reserved for forward-metric sign calibration "
        "and is not implemented in this runner yet."
    )


def paper_direction_equivalent(direction: Optional[str]) -> str:
    direction = normalize_update_direction(direction)
    if direction == "subtract":
        return "paper_style_W_minus_alpha_E"
    if direction == "add":
        return "equivalent_to_paper_subtract_with_signed_alpha_negative"
    return "auto_nll_probe_selected"


def _matrix_rank_from_s(s: torch.Tensor, rcond: float) -> int:
    if s.numel() == 0:
        return 0
    cutoff = float(rcond) * float(s.max().detach().cpu())
    return int((s > cutoff).sum().detach().cpu())


def _condition_proxy(s: torch.Tensor, rcond: float) -> float:
    if s.numel() == 0:
        return float("inf")
    max_s = float(s.max().detach().cpu())
    usable = s[s > float(rcond) * max_s]
    if usable.numel() == 0:
        return float("inf")
    return max_s / max(float(usable.min().detach().cpu()), 1.0e-30)


def _truncated_pinv(total: torch.Tensor, config: SolverConfig) -> Tuple[torch.Tensor, Dict[str, Any]]:
    u, s, vh = torch.linalg.svd(total, full_matrices=False)
    keep = torch.ones_like(s, dtype=torch.bool)
    if config.rcond is not None and s.numel() > 0:
        keep &= s > float(config.rcond) * s.max()
    if config.svd_rank is not None:
        rank_mask = torch.arange(s.numel(), device=s.device) < int(config.svd_rank)
        keep &= rank_mask
    if config.energy_threshold is not None and s.numel() > 0:
        threshold = float(config.energy_threshold)
        if not 0.0 < threshold <= 1.0:
            raise ValueError(f"energy_threshold must be in (0, 1], got {threshold}")
        energy = torch.cumsum(s.square(), dim=0) / torch.clamp(s.square().sum(), min=1.0e-30)
        energy_rank = int((energy < threshold).sum().detach().cpu()) + 1
        keep &= torch.arange(s.numel(), device=s.device) < energy_rank

    if keep.sum() == 0:
        pinv = torch.zeros_like(total.transpose(0, 1))
    else:
        u_k = u[:, keep]
        s_k = s[keep]
        vh_k = vh[keep, :]
        pinv = vh_k.transpose(0, 1).matmul(torch.diag(1.0 / s_k)).matmul(u_k.transpose(0, 1))

    stats = {
        "solver": "svd",
        "rank_total": int(keep.sum().detach().cpu()),
        "rank_numerical": _matrix_rank_from_s(s, float(config.rcond)),
        "condition_proxy": _condition_proxy(s, float(config.rcond)),
        "singular_max": float(s.max().detach().cpu()) if s.numel() else 0.0,
        "singular_min_kept": float(s[keep].min().detach().cpu()) if keep.any() else 0.0,
    }
    return pinv, stats


def compute_projector(
    sigma_plus: torch.Tensor,
    sigma_minus: torch.Tensor,
    *,
    config: SolverConfig,
    num_target_vectors: int = 0,
    num_reference_vectors: int = 0,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    with torch.no_grad():
        solve_device = as_device(config.solve_device)
        plus = sigma_plus.to(device=solve_device, dtype=torch.float32)
        minus = sigma_minus.to(device=solve_device, dtype=torch.float32)
        if plus.shape != minus.shape:
            raise ValueError(f"Covariance shape mismatch: plus={tuple(plus.shape)} minus={tuple(minus.shape)}")
        if not torch.isfinite(plus).all():
            raise RuntimeError("ENGRAM Sigma_plus contains NaN or inf values")
        if not torch.isfinite(minus).all():
            raise RuntimeError("ENGRAM Sigma_minus contains NaN or inf values")
        total = plus + minus
        if config.normalize_covariance:
            plus = plus / max(float(num_target_vectors), 1.0)
            total = total / max(float(num_target_vectors + num_reference_vectors), 1.0)
        if float(config.jitter or 0.0) > 0:
            total = total + float(config.jitter) * torch.eye(total.shape[0], device=solve_device, dtype=total.dtype)

        s_plus = torch.linalg.svdvals(plus)
        s_total = torch.linalg.svdvals(total)
        if str(config.method).lower() == "svd":
            pinv_total, stats = _truncated_pinv(total, config)
        elif str(config.method).lower() == "pinv":
            pinv_total = torch.linalg.pinv(total, rcond=float(config.rcond))
            stats = {
                "solver": "pinv",
                "rank_total": _matrix_rank_from_s(s_total, float(config.rcond)),
                "condition_proxy": _condition_proxy(s_total, float(config.rcond)),
            }
        else:
            raise ValueError(f"Unknown ENGRAM solver method: {config.method}")

        projector = plus.matmul(pinv_total)
        if not torch.isfinite(projector).all():
            raise RuntimeError("ENGRAM projector contains NaN or inf values")
        stats.update(
            {
                "rank_plus": _matrix_rank_from_s(s_plus, float(config.rcond)),
                "num_target_vectors": int(num_target_vectors),
                "num_reference_vectors": int(num_reference_vectors),
                "projector_norm": float(projector.norm().detach().cpu()),
            }
        )
        return projector, stats


def compute_linear_engram_update(
    module_name: str,
    module: nn.Linear,
    projector: torch.Tensor,
    *,
    input_dim: int,
    absorb_bias: bool,
    alpha: float,
    beta: float = 0.0,
    engram_update_direction: str = "subtract",
    direction_sign: Optional[int] = None,
    behavior_objective: Optional[str] = None,
    candidate_weight_delta: Optional[torch.Tensor] = None,
    candidate_bias_delta: Optional[torch.Tensor] = None,
    store_projector: bool = True,
    stats: Optional[Dict[str, Any]] = None,
) -> EngramLayerUpdate:
    with torch.no_grad():
        if not isinstance(module.weight, torch.nn.Parameter):
            raise RuntimeError(f"ENGRAM cannot edit {module_name}: weight is not a Parameter")
        if not module.weight.dtype.is_floating_point:
            raise RuntimeError(f"ENGRAM cannot edit {module_name}: non-floating/quantized dtype {module.weight.dtype}")
        if getattr(module.weight, "is_sparse", False):
            raise RuntimeError(f"ENGRAM cannot edit {module_name}: sparse/quantized parameter is not directly editable")

        solve_device = projector.device
        weight = module.weight.detach().to(device=solve_device, dtype=torch.float32)
        if absorb_bias and module.bias is not None:
            bias = module.bias.detach().to(device=solve_device, dtype=torch.float32)
            weight_aug = torch.cat([weight, bias[:, None]], dim=1)
            engram_aug = weight_aug.matmul(projector)
            engram_weight = engram_aug[:, :input_dim].detach().cpu()
            engram_bias = engram_aug[:, input_dim].detach().cpu()
        else:
            engram_weight = weight.matmul(projector).detach().cpu()
            engram_bias = None
        if not torch.isfinite(engram_weight).all():
            raise RuntimeError(f"ENGRAM update for {module_name} contains NaN or inf values")
        if engram_bias is not None and not torch.isfinite(engram_bias).all():
            raise RuntimeError(f"ENGRAM bias update for {module_name} contains NaN or inf values")

        delta_safe_weight = None
        delta_safe_bias = None
        if candidate_weight_delta is not None:
            cand_w = candidate_weight_delta.to(device=solve_device, dtype=torch.float32)
            if cand_w.shape != weight.shape:
                raise ValueError(f"Candidate weight delta for {module_name} has shape {tuple(cand_w.shape)}, expected {tuple(weight.shape)}")
            if absorb_bias and module.bias is not None:
                if candidate_bias_delta is None:
                    cand_b = torch.zeros(module.bias.shape, device=solve_device, dtype=torch.float32)
                else:
                    cand_b = candidate_bias_delta.to(device=solve_device, dtype=torch.float32)
                cand_aug = torch.cat([cand_w, cand_b[:, None]], dim=1)
                delta_aug = cand_aug.matmul(projector)
                delta_safe_weight = delta_aug[:, :input_dim].detach().cpu()
                delta_safe_bias = delta_aug[:, input_dim].detach().cpu()
            else:
                delta_safe_weight = cand_w.matmul(projector).detach().cpu()
            if delta_safe_weight is not None and not torch.isfinite(delta_safe_weight).all():
                raise RuntimeError(f"ENGRAM replacement delta for {module_name} contains NaN or inf values")
            if delta_safe_bias is not None and not torch.isfinite(delta_safe_bias).all():
                raise RuntimeError(f"ENGRAM replacement bias delta for {module_name} contains NaN or inf values")

        norm_w = float(weight.norm().detach().cpu())
        norm_e = float(engram_weight.norm().detach().cpu())
        update_direction = normalize_update_direction(engram_update_direction)
        sign = int(direction_sign if direction_sign is not None else direction_sign_for_update(update_direction))
        effective_update_norm_ratio = abs(float(alpha)) * (norm_e / (norm_w + 1.0e-12))
        layer_stats = dict(stats or {})
        layer_stats.update(
            {
                "module_name": module_name,
                "in_dim": int(module.in_features),
                "out_dim": int(module.out_features),
                "norm_W": norm_w,
                "norm_E": norm_e,
                "norm_ratio": norm_e / (norm_w + 1.0e-12),
                "effective_norm_ratio": effective_update_norm_ratio,
                "effective_update_norm_ratio": effective_update_norm_ratio,
                "engram_update_direction": update_direction,
                "direction_sign": sign,
                "behavior_objective": behavior_objective,
                "paper_direction_equivalent": paper_direction_equivalent(update_direction),
            }
        )
        return EngramLayerUpdate(
            module_name=module_name,
            weight=engram_weight,
            bias=engram_bias,
            projector=projector.detach().cpu() if store_projector else None,
            delta_safe_weight=delta_safe_weight,
            delta_safe_bias=delta_safe_bias,
            alpha=float(alpha),
            beta=float(beta),
            engram_update_direction=update_direction,
            direction_sign=sign,
            behavior_objective=behavior_objective,
            paper_direction_equivalent=paper_direction_equivalent(update_direction),
            stats=layer_stats,
        )


def apply_update_to_module(module: nn.Linear, update: EngramLayerUpdate, *, direction: int = -1) -> None:
    """Apply or undo an ENGRAM update.

    direction=-1 applies W <- W + direction_sign * alpha * E + beta * Delta.
    direction=+1 rolls that operation back.
    """
    if direction not in {-1, 1}:
        raise ValueError(f"direction must be -1 (apply) or +1 (rollback), got {direction}")
    if not module.weight.dtype.is_floating_point:
        raise RuntimeError(f"ENGRAM cannot edit non-floating/quantized parameter dtype {module.weight.dtype}")
    with torch.no_grad():
        operation_sign = -float(direction)
        update_sign = int(getattr(update, "direction_sign", -1))
        weight_delta = (
            operation_sign
            * float(update_sign)
            * float(update.alpha)
            * update.weight.to(module.weight.device, dtype=torch.float32)
        )
        if update.delta_safe_weight is not None:
            weight_delta += operation_sign * float(update.beta) * update.delta_safe_weight.to(
                module.weight.device, dtype=torch.float32
            )
        module.weight.add_(weight_delta.to(dtype=module.weight.dtype))
        if not torch.isfinite(module.weight).all():
            raise RuntimeError("ENGRAM produced NaN or inf values in updated weight")

        if module.bias is not None and update.bias is not None:
            bias_delta = (
                operation_sign
                * float(update_sign)
                * float(update.alpha)
                * update.bias.to(module.bias.device, dtype=torch.float32)
            )
            if update.delta_safe_bias is not None:
                bias_delta += operation_sign * float(update.beta) * update.delta_safe_bias.to(
                    module.bias.device, dtype=torch.float32
                )
            module.bias.add_(bias_delta.to(dtype=module.bias.dtype))
            if not torch.isfinite(module.bias).all():
                raise RuntimeError("ENGRAM produced NaN or inf values in updated bias")
