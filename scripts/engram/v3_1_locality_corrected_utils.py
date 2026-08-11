"""Pure utilities for the ENGRAM V3.1 locality-corrected gate."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


def kl_candidate_s0(candidate: torch.Tensor, baseline: torch.Tensor) -> torch.Tensor:
    logp = candidate.log_softmax(dim=-1)
    return (logp.exp() * (logp - baseline.log_softmax(dim=-1))).sum(dim=-1).mean()


def equality_kl_gradient_is_unusable(logits: torch.Tensor, tolerance: float = 1e-4) -> bool:
    candidate = logits.detach().clone().requires_grad_(True)
    gradient = torch.autograd.grad(kl_candidate_s0(candidate, logits.detach()), candidate)[0]
    return bool(torch.isfinite(gradient).all() and float(gradient.detach().abs().max()) <= tolerance)


def preservation_nll(logits: torch.Tensor, baseline_ids: torch.Tensor) -> torch.Tensor:
    if logits.ndim != 2 or baseline_ids.ndim != 1 or logits.shape[0] != baseline_ids.numel():
        raise ValueError("Preservation logits and IDs are not aligned")
    return -logits.log_softmax(dim=-1).gather(1, baseline_ids.long().unsqueeze(1)).mean()


def token_margins(logits: torch.Tensor, baseline_ids: torch.Tensor) -> torch.Tensor:
    target = logits.gather(1, baseline_ids.long().unsqueeze(1)).squeeze(1)
    masked = logits.clone()
    masked.scatter_(1, baseline_ids.long().unsqueeze(1), -torch.inf)
    return target - masked.max(dim=1).values


def fragile_positive_position(logits: torch.Tensor, baseline_ids: torch.Tensor) -> int:
    margins = token_margins(logits.detach().float(), baseline_ids)
    positive = torch.where(margins > 0)[0]
    if positive.numel() == 0:
        raise ValueError("S0 preservation trajectory has no positive top-1 margin")
    values = margins[positive]
    return int(positive[torch.argmin(values)].item())


def preservation_margin_loss(logits: torch.Tensor, baseline_ids: torch.Tensor, position: int) -> torch.Tensor:
    margins = token_margins(logits, baseline_ids)
    if not 0 <= int(position) < margins.numel():
        raise IndexError(position)
    return -margins[int(position)]


def select_modules(rows: Sequence[Mapping[str, Any]], top_k: int = 3) -> list[dict[str, Any]]:
    valid = []
    for original in rows:
        row = dict(original)
        local = float(row["locality_size_normalized_norm"])
        if not math.isfinite(local) or local <= 0.0:
            continue
        target = float(row["target_size_normalized_norm"])
        row["score"] = target / (local + 1e-12)
        valid.append(row)
    if len(valid) < top_k:
        raise ValueError("Insufficient modules with nonzero finite locality sensitivity")
    return sorted(valid, key=lambda row: (-float(row["score"]), int(row["layer"]), str(row["module_name"])))[:top_k]


def fixed_right_basis(negative_gradient: torch.Tensor, rank: int = 4) -> dict[str, torch.Tensor | float]:
    matrix = negative_gradient.detach().float().cpu()
    if matrix.ndim != 2 or rank <= 0 or rank > min(matrix.shape):
        raise ValueError("Invalid fixed-right SVD request")
    torch.manual_seed(42)
    q = min(rank + 4, min(matrix.shape))
    _u, singular, v = torch.svd_lowrank(matrix, q=q, niter=4)
    order = torch.argsort(singular, descending=True)[:rank]
    singular = singular[order]
    a_fixed = v[:, order].T.contiguous()
    a_fixed = torch.nn.functional.normalize(a_fixed, dim=1)
    energy = float(singular.float().square().sum() / matrix.float().square().sum().clamp_min(1e-30))
    return {"A_fixed": a_fixed, "singular_values": singular.contiguous(), "captured_energy_fraction": energy}


class FixedRightWeight(torch.nn.Module):
    """Linear FP32 update DeltaW = B @ A_fixed with exact-zero initialization."""

    def __init__(self, base: torch.Tensor, a_fixed: torch.Tensor):
        super().__init__()
        if base.ndim != 2 or a_fixed.ndim != 2 or base.shape[1] != a_fixed.shape[1]:
            raise ValueError("Base and A_fixed shapes are incompatible")
        # The immutable S0 base does not need an FP32 duplicate.  Keeping its
        # original compute dtype avoids hundreds of MiB of persistent GPU
        # storage; A_fixed, B, and the effective addition remain FP32.
        self.register_buffer("base", base.detach().clone())
        self.register_buffer("A_fixed", a_fixed.detach().float().clone())
        self.B = torch.nn.Parameter(torch.zeros(base.shape[0], a_fixed.shape[0], device=base.device, dtype=torch.float32))

    def delta(self) -> torch.Tensor:
        return self.B @ self.A_fixed

    def forward(self, original: torch.Tensor) -> torch.Tensor:
        return (self.base.float() + self.delta()).to(original.dtype)


def flatten_tensors(values: Sequence[torch.Tensor]) -> torch.Tensor:
    if not values:
        raise ValueError("At least one tensor is required")
    return torch.cat([value.reshape(-1) for value in values])


def copy_flat(parameters: Sequence[torch.Tensor], flat: torch.Tensor) -> None:
    cursor = 0
    with torch.no_grad():
        for parameter in parameters:
            count = parameter.numel()
            parameter.copy_(flat[cursor : cursor + count].reshape_as(parameter))
            cursor += count
    if cursor != flat.numel():
        raise ValueError("Flat tensor size mismatch")


def induced_effective_norm(flat_delta_b: torch.Tensor, parameters: Sequence[torch.Tensor], bases: Sequence[torch.Tensor]) -> float:
    cursor = 0
    squared = 0.0
    for parameter, a_fixed in zip(parameters, bases):
        count = parameter.numel()
        delta_b = flat_delta_b[cursor : cursor + count].reshape_as(parameter)
        squared += float((delta_b.double() @ a_fixed.detach().double()).square().sum())
        cursor += count
    if cursor != flat_delta_b.numel():
        raise ValueError("Flat factor update size mismatch")
    return math.sqrt(squared)


def normalize_effective_step(direction: torch.Tensor, parameters: Sequence[torch.Tensor], bases: Sequence[torch.Tensor], step_length: float) -> torch.Tensor:
    if step_length <= 0:
        raise ValueError("step_length must be positive")
    norm = induced_effective_norm(direction, parameters, bases)
    if not math.isfinite(norm) or norm <= 0:
        raise ValueError("Zero or non-finite effective direction")
    return direction * (float(step_length) / norm)


def locality_basis(gradients: Sequence[torch.Tensor], relative_tolerance: float = 1e-7) -> dict[str, Any]:
    nonzero = [value.detach().float().reshape(-1).cpu() for value in gradients if torch.isfinite(value).all() and float(value.double().norm()) > 1e-12]
    if not nonzero:
        raise ValueError("No nonzero locality directions")
    matrix = torch.stack(nonzero, dim=1)
    u, singular, _v = torch.linalg.svd(matrix, full_matrices=False)
    threshold = max(float(singular[0]) * relative_tolerance, 1e-10)
    rank = int((singular > threshold).sum().item())
    if rank == 0:
        raise ValueError("Numerical locality rank is zero")
    basis = u[:, :rank].contiguous()
    eye = torch.eye(rank, dtype=basis.dtype)
    orthogonality_error = float((basis.T @ basis - eye).double().norm())
    return {"basis": basis, "singular_values": singular, "rank": rank, "nonzero_directions": len(nonzero), "orthogonality_error": orthogonality_error}


def project_gradient(gradient: torch.Tensor, basis: torch.Tensor) -> tuple[torch.Tensor, float]:
    flat = gradient.reshape(-1)
    projected = flat - basis.to(flat.device, flat.dtype) @ (basis.to(flat.device, flat.dtype).T @ flat)
    residual = float((basis.to(projected.device, projected.dtype).T @ projected).double().norm() / projected.double().norm().clamp_min(1e-12))
    return projected.reshape_as(gradient), residual


def choose_directional_sign(plus: Mapping[str, Any], minus: Mapping[str, Any]) -> int:
    def valid(row: Mapping[str, Any]) -> bool:
        return bool(
            float(row["effect_loss"]) < float(row["baseline_effect_loss"])
            and (float(row["primary_margin"]) > float(row["baseline_primary_margin"]) or float(row["primary_sequence_score"]) > float(row["baseline_primary_sequence_score"]))
            and float(row["maximum_locality_nll_drift"]) < 0.01
            and bool(row["paired_first_top1_equal"])
            and bool(row["rollback_exact"])
        )
    candidates = [(1, plus), (-1, minus)]
    candidates = [(sign, row) for sign, row in candidates if valid(row)]
    if not candidates:
        raise ValueError("Neither finite-difference sign is valid")
    return min(candidates, key=lambda item: (float(item[1]["effect_loss"]), -float(item[1]["primary_margin"]), -item[0]))[0]


UNSUPPORTED_SPECIFICITY = {"japanese", "chinese", "african", "european", "american", "male", "female"}


def unsupported_specificity_terms(baseline: str, candidate: str, canonical: str) -> list[str]:
    normalize = lambda text: set(str(text).casefold().replace("-", " ").replace("/", " ").split())
    # The preservation contract is relative to the exact S0 generated answer.
    # A canonical teacher-forced label must not license new specificity in the
    # edited free generation (the preregistered record-1333 decision depends on
    # this distinction).
    supported = normalize(baseline)
    return sorted((normalize(candidate) & UNSUPPORTED_SPECIFICITY) - supported)


def ensure_new_output_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=False)
    return path
