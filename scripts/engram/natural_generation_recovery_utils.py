"""Pure helpers for ENGRAM V2.1/V3 natural-generation recovery."""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


def canonical_natural_response(target: str) -> str:
    value = str(target).strip()
    value = re.sub(r"[\s.?!]+$", "", value).strip()
    if not value:
        raise ValueError("Target must not be empty")
    return f"The answer is {value}."


def target_only_response(target: str) -> str:
    value = str(target).strip()
    if not value:
        raise ValueError("Target must not be empty")
    return value


def assert_no_target_leakage(prompt_texts: Sequence[str], target: str) -> None:
    normalized_target = " ".join(re.findall(r"\w+", str(target).casefold()))
    for index, prompt in enumerate(prompt_texts):
        normalized_prompt = " ".join(re.findall(r"\w+", str(prompt).casefold()))
        if normalized_target and re.search(r"(?<!\w)" + re.escape(normalized_target) + r"(?!\w)", normalized_prompt):
            raise AssertionError(f"Target leaked into generation prompt {index}")


def expanded_predictor_positions(prefix_length: int, answer_length: int, multimodal_expansion: int) -> list[int]:
    if prefix_length <= 0 or answer_length <= 0 or multimodal_expansion < 0:
        raise ValueError("Invalid shifted-label geometry")
    return [prefix_length - 1 + multimodal_expansion + index for index in range(answer_length)]


def materialize_fp32_shadow(base: torch.Tensor, delta_fp32: torch.Tensor, compute_dtype: torch.dtype) -> torch.Tensor:
    if delta_fp32.dtype != torch.float32:
        raise TypeError("Master shadow delta must remain FP32")
    return (base.detach().float() + delta_fp32).to(dtype=compute_dtype)


def global_l2(values: Sequence[torch.Tensor]) -> float:
    return math.sqrt(sum(float(value.detach().double().square().sum().item()) for value in values))


def project_delta_to_norm(delta: torch.Tensor, maximum_norm: float) -> torch.Tensor:
    if maximum_norm <= 0:
        raise ValueError("maximum_norm must be positive")
    norm = float(delta.detach().double().norm().item())
    if norm <= maximum_norm:
        return delta.detach().clone()
    return (delta.detach().double() * (float(maximum_norm) / norm)).to(delta.dtype)


def project_deltas_to_relative_budget(
    deltas: Sequence[torch.Tensor],
    bases: Sequence[torch.Tensor],
    relative_cap: float,
) -> list[torch.Tensor]:
    if len(deltas) != len(bases) or not deltas:
        raise ValueError("Delta/base collections must be non-empty and aligned")
    denominator = global_l2(bases)
    numerator = global_l2(deltas)
    maximum = float(relative_cap) * denominator
    if numerator <= maximum:
        return [value.detach().clone() for value in deltas]
    ratio = maximum / max(numerator, 1e-30)
    return [(value.detach().double() * ratio).to(value.dtype) for value in deltas]


def relative_displacement(deltas: Sequence[torch.Tensor], bases: Sequence[torch.Tensor]) -> float:
    return global_l2(deltas) / max(global_l2(bases), 1e-30)


def select_candidate_modules(rows: Sequence[Mapping[str, Any]], top_k: int = 3) -> list[Mapping[str, Any]]:
    if top_k <= 0 or len(rows) < top_k:
        raise ValueError("Insufficient candidate-module rows")
    scored = []
    for row in rows:
        score = float(row["target_size_normalized_norm"]) / (float(row["locality_size_normalized_norm"]) + 1e-12)
        scored.append({**dict(row), "score": score})
    return sorted(scored, key=lambda row: (-float(row["score"]), int(row["layer"]), str(row["module_name"])))[:top_k]


def rank4_svd_initialization(gradient: torch.Tensor, rank: int = 4) -> dict[str, torch.Tensor]:
    if gradient.ndim != 2 or rank <= 0 or rank > min(gradient.shape):
        raise ValueError("Invalid matrix/rank for SVD initialization")
    matrix = -gradient.detach().float().cpu()
    torch.manual_seed(42)
    q = min(max(rank + 4, rank), min(matrix.shape))
    u, singular, v = torch.svd_lowrank(matrix, q=q, niter=4)
    order = torch.argsort(singular, descending=True)[:rank]
    u, singular, v = u[:, order], singular[order], v[:, order]
    root = singular.clamp_min(0).sqrt()
    b = u * root.unsqueeze(0)
    a = root.unsqueeze(1) * v.T
    return {"A": a.contiguous(), "B": b.contiguous(), "singular_values": singular.contiguous()}


def project_effect_gradient(effect: torch.Tensor, locality_basis: torch.Tensor) -> torch.Tensor:
    flat = effect.reshape(-1)
    if locality_basis.ndim != 2 or locality_basis.shape[0] != flat.numel():
        raise ValueError("Locality basis does not match factor-gradient dimension")
    if locality_basis.shape[1] == 0:
        return effect.detach().clone()
    projected = flat - locality_basis @ (locality_basis.T @ flat)
    return projected.reshape_as(effect)


def orthonormal_locality_basis(probe_gradients: Sequence[torch.Tensor], tolerance: float = 1e-10) -> torch.Tensor:
    if not probe_gradients:
        raise ValueError("At least one locality gradient is required")
    matrix = torch.stack([value.detach().float().reshape(-1) for value in probe_gradients], dim=1)
    u, singular, _ = torch.linalg.svd(matrix, full_matrices=False)
    keep = singular > float(tolerance)
    return u[:, keep].contiguous()


def tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().contiguous().cpu()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode())
    digest.update(str(tuple(tensor.shape)).encode())
    digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def save_v3_bank(root: Path, factors: Mapping[str, Mapping[str, torch.Tensor]], metadata: Mapping[str, Any]) -> None:
    root.mkdir(parents=True, exist_ok=False)
    payload = {
        name: {key: value.detach().cpu().float().clone() for key, value in row.items() if isinstance(value, torch.Tensor)}
        for name, row in factors.items()
    }
    torch.save(payload, root / "factors.pt")
    factor_hashes = {
        name: {key: tensor_sha256(value) for key, value in row.items()}
        for name, row in payload.items()
    }
    manifest = {**dict(metadata), "factor_hashes": factor_hashes}
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def load_v3_bank(root: Path) -> dict[str, Any]:
    manifest = json.loads((root / "manifest.json").read_text())
    factors = torch.load(root / "factors.pt", map_location="cpu", weights_only=False)
    for name, row in factors.items():
        for key, value in row.items():
            if tensor_sha256(value) != manifest["factor_hashes"][name][key]:
                raise RuntimeError(f"V3 factor checksum mismatch: {name}:{key}")
    return {"manifest": manifest, "factors": factors}


def exact_tensor_collection_equal(left: Sequence[torch.Tensor], right: Sequence[torch.Tensor]) -> bool:
    return len(left) == len(right) and all(torch.equal(a.detach().cpu(), b.detach().cpu()) for a, b in zip(left, right))
