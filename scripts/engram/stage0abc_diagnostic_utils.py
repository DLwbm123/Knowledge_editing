"""Safety helpers for the read-only ENGRAM V2 Stage-0A/B/C diagnostics."""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, Iterator

import torch

from scripts.engram.stage0_generation_audit_utils import tensor_sha256


def create_new_output_dir(path: Path) -> Path:
    path = path.resolve()
    path.mkdir(parents=True, exist_ok=False)
    return path


def first_prefix_parity(reference: list[int], candidate: list[int], length: int = 64) -> bool:
    return len(reference) >= length and len(candidate) >= length and reference[:length] == candidate[:length]


def projected_adjoint_via_autograd(
    output_gradient: torch.Tensor,
    shape: tuple[int, ...],
    projection: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    raw = torch.zeros(shape, dtype=output_gradient.dtype, device=output_gradient.device, requires_grad=True)
    projected = projection(raw)
    if projected.shape != output_gradient.shape:
        raise ValueError("Projection output and output-gradient shapes differ")
    return torch.autograd.grad((projected * output_gradient).sum(), raw)[0]


@contextmanager
def temporary_parameter_delta(parameter: torch.nn.Parameter, delta: torch.Tensor) -> Iterator[Dict[str, Any]]:
    snapshot = parameter.detach().clone()
    before_hash = tensor_sha256(snapshot)
    with torch.no_grad():
        parameter.add_(delta.to(device=parameter.device, dtype=parameter.dtype))
    ledger: Dict[str, Any] = {
        "before_hash": before_hash,
        "temporary_hash": tensor_sha256(parameter),
        "delta_norm": float(delta.float().norm().item()),
    }
    try:
        yield ledger
    finally:
        with torch.no_grad():
            parameter.copy_(snapshot)
        ledger["after_hash"] = tensor_sha256(parameter)
        ledger["rollback_exact"] = bool(torch.equal(parameter.detach(), snapshot))
        if not ledger["rollback_exact"] or ledger["after_hash"] != before_hash:
            raise RuntimeError("Temporary parameter delta rollback was not exact")
